"""FSDP2-sharded Qwen3.8-Flash-Next (BF16) with LoRA, for computing and training token logprobs.

- Every rank builds the model on the meta device, shards it with FSDP2 (one
  group per decoder layer, plus backbone and vision tower; lm_head is replicated), and reads
  only its own dim-0 slice of each checkpoint tensor, so the 336 GB checkpoint
  is read once in total.
- The frozen 102 GB n-gram embedding is not an FSDP parameter: all ranks map one
  copy in /dev/shm (see ngram_shm.py) and look rows up on the CPU.
- `offload()` / `onload()` move the shards between GPU and CPU so the vLLM
  servers can use the GPUs during rollouts.
- `token_logprobs()` returns logprobs of the target positions only, computing
  the 248K-way lm_head in checkpointed chunks.
"""
from contextlib import nullcontext
import json
import math
from pathlib import Path
import zlib

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from safetensors import safe_open
from safetensors.torch import save_file

from . import lora as lora_cfg
from . import ngram_shm, qsa_patch

LOGIT_CHUNK = 2048
LIBRARY_HEADROOM = 2 << 30  # bytes of GPU memory kept outside PyTorch's cache during training
CPU_ACTIVATIONS_ABOVE = 12_288  # tokens; longer sequences keep checkpointed activations in CPU memory
# Unpinned: the pinned host allocator rounds blocks up to powers of two and never frees them, which
# (8 ranks x ~48 GB, next to vLLM's sleeping weights) hit the pod's 1.5 TiB memory limit at step 12.


class SharedCPUEmbedding(torch.nn.Module):
    """Frozen embedding whose table lives in shared CPU memory, invisible to FSDP and .to()."""

    def __init__(self, table):
        super().__init__()
        self.__dict__['table'] = table

    @property
    def weight(self):
        return self.table

    def forward(self, ids):
        return F.embedding(ids.cpu(), self.table)


class Policy:
    def __init__(self, model_dir, rank, world):
        import transformers
        from transformers import AutoConfig
        from peft import get_peft_model
        qsa_patch.patch()
        self.rank, self.world, self.model_dir = rank, world, Path(model_dir)
        config = AutoConfig.from_pretrained(model_dir)
        config.text_config.use_cache = False
        with torch.device('meta'):
            model = transformers.Qwen4ExpForConditionalGeneration._from_config(config, dtype=torch.bfloat16)
        info, table = ngram_shm.load()
        ngram_module = model.get_submodule(info['prefix'].removesuffix('.ngram_embedding')).ngram_embedding
        if ngram_module.weight.shape != table.shape:
            raise ValueError(f'n-gram table {tuple(table.shape)} != model {tuple(ngram_module.weight.shape)}')
        parent = model.get_submodule(info['prefix'].removesuffix('.ngram_embedding'))
        parent.ngram_embedding = SharedCPUEmbedding(table)

        model.config.text_config.use_cache = False
        self.peft = get_peft_model(model, lora_cfg.peft_config())
        self.model = self.peft.base_model.model  # Qwen4ExpForConditionalGeneration
        self.text = self.model.model.language_model
        self.text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in self.text.layers:
            fully_shard(layer, mp_policy=mp)
        fully_shard(self.model.model.visual, mp_policy=mp)
        fully_shard(self.text, mp_policy=mp)
        # lm_head (1.27 GB, frozen) is replicated, not sharded: it is called once per logit chunk, and chunk
        # counts differ across ranks, so a sharded lm_head would issue mismatched all-gathers and deadlock.
        rotary = {name: type(m)(m.config) for name, m in self.model.named_modules() if hasattr(m, 'inv_freq')}
        self.model.to_empty(device='cuda')
        self._load_weights()
        with torch.no_grad():  # Rotary buffers are computed, not stored; rebuild them after to_empty().
            for name, fresh in rotary.items():
                module = self.model.get_submodule(name)
                for bname, buf in module.named_buffers(recurse=False):
                    buf.copy_(getattr(fresh, bname).to(buf.device, buf.dtype))
        self.lora_params = [p for n, p in self.model.named_parameters() if 'lora_' in n]
        for n, p in self.model.named_parameters():
            p.requires_grad_('lora_' in n)
        torch.cuda.synchronize()

    # ----------------------------------------------------------------- loading
    def _checkpoint_index(self):
        return json.loads((self.model_dir / 'model.safetensors.index.json').read_text())['weight_map']

    def _load_weights(self):
        index = self._checkpoint_index()
        handles = {}

        def tensor_slice(key, start, stop):
            file = index[key]
            if file not in handles:
                handles[file] = safe_open(self.model_dir / file, 'pt')
            sl = handles[file].get_slice(key)
            shape = sl.get_shape()
            if not shape:
                return sl[:]
            return sl[start:stop]

        loaded = 0
        named = list(self.model.named_parameters()) + list(self.model.named_buffers())
        for name, param in named:
            if 'lora_' in name:
                continue
            key = name.replace('.base_layer.', '.')
            if key not in index:
                if 'rotary' in name:
                    continue
                raise KeyError(f'{key} not in checkpoint')
            local = param.to_local() if hasattr(param, 'to_local') else param
            if hasattr(param, 'to_local') and local.dim() > 0:
                rows = param.shape[0]
                chunk = math.ceil(rows / self.world)
                start = min(self.rank * chunk, rows)
                stop = start + local.shape[0]
            else:
                start, stop = 0, (local.shape[0] if local.dim() else None)
            with torch.no_grad():
                if local.numel():
                    local.copy_(tensor_slice(key, start, stop).to(local.dtype))
            loaded += 1
        for name, param in self.model.named_parameters():
            if 'lora_' in name:
                local = param.to_local()
                with torch.no_grad():
                    if 'lora_A' in name:
                        # PEFT default init: kaiming-uniform A, zero B, so the adapter starts as identity.
                        full_in = param.shape[1]
                        bound = 1 / math.sqrt(full_in)
                        g = torch.Generator(device='cpu').manual_seed(zlib.crc32(name.encode()))
                        full = (torch.rand(param.shape, generator=g) * 2 - 1) * bound
                        rows = param.shape[0]
                        chunk = math.ceil(rows / self.world)
                        start = min(self.rank * chunk, rows)
                        local.copy_(full[start:start + local.shape[0]].to(local.dtype))
                    else:
                        local.zero_()
        self.loaded = loaded

    # ------------------------------------------------------------- placement
    def offload(self):
        self.model.to('cpu')
        torch.cuda.empty_cache()
        torch._C._host_emptyCache()

    def onload(self):
        # Cap PyTorch's cache so cuDNN/cuBLAS, which allocate outside it, always have room: with the
        # cache filling the GPU next to sleeping vLLM (~6.6 GB), they failed with ~10 MB free.
        free, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction((torch.cuda.memory_reserved() + free - LIBRARY_HEADROOM) / total)
        self.model.to('cuda')

    # ---------------------------------------------------------------- compute
    def token_logprobs(self, ids, targets):
        """Logprobs of ids[t] for every t in targets (positions >= 1), as a float32 tensor."""
        x = torch.tensor([ids], device='cuda')
        targets = torch.tensor(targets, device='cuda')
        offload = len(ids) > CPU_ACTIVATIONS_ABOVE
        ctx = torch.autograd.graph.save_on_cpu(pin_memory=False) if offload and torch.is_grad_enabled() else nullcontext()
        with ctx:
            # Call the text backbone (an FSDP module) directly; text-only 3D positions equal arange.
            hidden = self.text(input_ids=x, use_cache=False).last_hidden_state[0]
        h = hidden.index_select(0, targets - 1)
        labels = x[0].index_select(0, targets)
        out = []
        for s in range(0, h.shape[0], LOGIT_CHUNK):
            out.append(checkpoint(self._chunk_logprobs, h[s:s + LOGIT_CHUNK], labels[s:s + LOGIT_CHUNK],
                                  use_reentrant=False))
        return torch.cat(out) if out else h.new_zeros(0, dtype=torch.float32)

    def _chunk_logprobs(self, h, labels):
        logits = self.model.lm_head(h).float()
        return logits.gather(-1, labels[:, None])[:, 0] - torch.logsumexp(logits, -1)

    # ------------------------------------------------------------------ saving
    def save_adapter(self, out):
        """Gather LoRA weights and write a PEFT adapter vLLM can load (rank 0 writes)."""
        tensors = {}
        for name, param in self.model.named_parameters():
            if 'lora_' not in name:
                continue
            full = param.full_tensor() if hasattr(param, 'full_tensor') else param
            if self.rank == 0:
                key = 'base_model.model.' + name.replace('.default.', '.')
                tensors[key] = full.detach().to(torch.bfloat16).cpu().contiguous()
        if self.rank == 0:
            out = Path(out)
            out.mkdir(parents=True, exist_ok=True)
            save_file(tensors, out / 'adapter_model.safetensors')
            config = lora_cfg.peft_config().to_dict()
            config.update(target_modules=lora_cfg.TARGET_REGEX, peft_type='LORA', task_type='CAUSAL_LM')
            (out / 'adapter_config.json').write_text(json.dumps(config, indent=2, default=str))
        dist.barrier()

    def lora_state(self):
        return {n: p.to_local().detach().cpu().clone() for n, p in self.model.named_parameters() if 'lora_' in n}

    def load_lora_state(self, state):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in state:
                    p.to_local().copy_(state[n])
