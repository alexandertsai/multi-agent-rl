"""LoRA targets shared by the trainer (PEFT) and the vLLM server.

Rank-16 LoRA on every attention projection (Gated DeltaNet and QSA layers) and
on each layer's shared expert. The 512 routed experts, router, indexer, n-gram
embedding and vision tower stay frozen.
"""
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

RANK, ALPHA = 16, 32
TARGET_REGEX = (r'.*language_model\.layers\.\d+\.('
                r'self_attn\.(q_proj|k_proj|v_proj|o_proj)'
                r'|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)'
                r'|mlp\.shared_expert\.(gate_proj|up_proj|down_proj))')


def peft_config():
    from peft import LoraConfig
    return LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.0, target_modules=TARGET_REGEX, bias='none')


def random_adapter(out, model_config, std=0.0):
    """Write a PEFT-format adapter with the right shapes; std=0 gives an identity adapter."""
    import re
    from accelerate import init_empty_weights
    import transformers
    with init_empty_weights():
        model = transformers.Qwen4ExpForConditionalGeneration(model_config)
    tensors = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and re.fullmatch(TARGET_REGEX, name):
            o, i = module.weight.shape
            tensors[f'base_model.model.{name}.lora_A.weight'] = torch.randn(RANK, i) * (1 / i ** 0.5)
            tensors[f'base_model.model.{name}.lora_B.weight'] = torch.randn(o, RANK) * std
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in tensors.items()}, out / 'adapter_model.safetensors')
    config = peft_config().to_dict()
    config['target_modules'] = TARGET_REGEX
    config['peft_type'] = 'LORA'
    config['task_type'] = 'CAUSAL_LM'
    (out / 'adapter_config.json').write_text(json.dumps(config, indent=2, default=str))
    return len(tensors) // 2
