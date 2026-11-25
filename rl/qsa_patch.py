"""Vectorised Qwen Sparse Attention indexer for training.

The transformers reference indexer loops over every query position in Python
(one topk per token), which makes an 8K-token forward pass CPU-bound. For a
single unpadded causal sequence, the pooled key of each complete 4-token block
does not depend on the query, so all block keys are computed once and each
query takes a top-k over the blocks it can see. Selection matches the
reference: the top `indexer_budget / compress_ratio` complete blocks by
sum-over-heads ReLU score, plus the query's trailing incomplete block.

`patch()` swaps the indexer's forward; batches with padding or a KV cache fall
back to the reference implementation.
"""
import math

import torch

from transformers.models.qwen4_exp import modeling_qwen4_exp as qm

_reference_forward = qm.Qwen4ExpTextQSAIndexer.forward
QUERY_CHUNK = 2048


def _is_plain_causal(attention_mask, seq_length):
    visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    if visible.shape[0] != 1 or visible.shape[-1] != seq_length or visible.shape[-2] != seq_length:
        return False
    causal = torch.ones(seq_length, seq_length, dtype=torch.bool, device=visible.device).tril_()
    return bool(torch.equal(visible[0, 0], causal))


def selected_blocks(self, hidden_states, position_embeddings):
    """Return (block_scores_topk_indices [L, k], valid [L, k], n_complete [L])."""
    _, seq_length, _ = hidden_states.shape
    cos, sin = position_embeddings
    qk = self.index_qk_proj(hidden_states)
    q, token_k = torch.split(
        qk, [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim], dim=-1)
    q = self.q_layernorm(q.reshape(1, seq_length, -1, self.index_head_dim))
    q = qm.apply_rotary_pos_emb(q, cos=cos[:, -seq_length:], sin=sin[:, -seq_length:], unsqueeze_dim=2)[0]
    raw_keys = token_k.reshape(seq_length, self.index_head_dim)
    r = self.compress_ratio
    n_blocks = seq_length // r
    if n_blocks == 0:
        return None
    pooled = raw_keys[: n_blocks * r].view(n_blocks, r, -1).float().mean(1).to(raw_keys.dtype)
    pooled = self.k_layernorm(pooled)
    starts = torch.arange(n_blocks, device=hidden_states.device) * r
    block_keys = qm.apply_rotary_pos_emb(
        pooled.unsqueeze(1), cos=cos[0].index_select(0, starts), sin=sin[0].index_select(0, starts)
    ).squeeze(1).float()  # [nb, d]
    k = min(self.block_topk, n_blocks)
    positions = torch.arange(seq_length, device=hidden_states.device)
    n_complete = (positions + 1) // r  # complete blocks visible to each query
    top = torch.empty(seq_length, k, dtype=torch.long, device=hidden_states.device)
    scale = 1 / math.sqrt(self.index_head_dim)
    for s in range(0, seq_length, QUERY_CHUNK):
        e = min(s + QUERY_CHUNK, seq_length)
        nb = min(n_blocks, int(n_complete[e - 1]))
        if nb == 0:
            top[s:e] = 0
            continue
        scores = torch.relu(torch.einsum('qhd,bd->qbh', q[s:e].float(), block_keys[:nb])).sum(-1) * scale
        visible = torch.arange(nb, device=scores.device)[None] < n_complete[s:e, None]
        scores = scores.masked_fill(~visible, float('-inf'))
        kk = min(k, nb)
        top[s:e, :kk] = scores.topk(kk, dim=-1).indices
        top[s:e, kk:] = 0
    valid = torch.arange(k, device=top.device)[None] < n_complete[:, None].clamp(max=k)
    return top, valid, n_complete


def fast_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
    seq_length = hidden_states.shape[1]
    if past_key_values is not None or not _is_plain_causal(attention_mask, seq_length):
        return _reference_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values)
    with torch.no_grad():
        picked = selected_blocks(self, hidden_states, position_embeddings)
        r = self.compress_ratio
        positions = torch.arange(seq_length, device=hidden_states.device)
        # Trailing incomplete block of each query: tokens from r * n_complete up to the query.
        tail_start = ((positions + 1) // r) * r
        mask = (positions[None, :] >= tail_start[:, None]) & (positions[None, :] <= positions[:, None])
        if picked is not None:
            top, valid, _ = picked
            n_blocks = seq_length // r
            block_mask = torch.zeros(seq_length, n_blocks + 1, dtype=torch.bool, device=mask.device)
            block_mask.scatter_(1, torch.where(valid, top, n_blocks), True)
            token_mask = block_mask[:, :n_blocks].repeat_interleave(r, dim=1)
            mask[:, : n_blocks * r] |= token_mask
    mask = mask[None, None]
    if attention_mask.is_floating_point():
        mask = torch.where(mask, attention_mask.new_zeros(()), torch.finfo(attention_mask.dtype).min)
    return mask


def patch():
    qm.Qwen4ExpTextQSAIndexer.forward = fast_forward
