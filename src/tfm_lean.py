"""Memory-lean KV-cache build for TabPFN-3.5 (tabpfn==9.0.0), so a full ~560k-row fold
context fits a 16 GB T4.

Ported from hemingweb's public notebook "S6E10 EXP01 TabPFN35 Full Context" (also used by
samanyu1808's "TabPFN route og"); an idea taken from a public notebook, not an artifact.
The stock fit_with_cache path runs every ICL layer over all context rows at once (full
q/k/v, fp32 norm copies, unchunked MLP) and OOMs. This computes K/V for the context in row
chunks, then the attention output and the MLP in row chunks, updating the residual stream
in place. Queries are independent given K/V, and the softmax scaling depends only on the K
length, so the predictions are the same as the stock path's.

Imported only by learners._fit_predict_tabpfn; nothing else needs tabpfn installed.
"""
import torch
from tabpfn.architectures import tabpfn_v3_5 as A

_orig_forward = A.ICLTransformerBlock.forward
CHUNK = 16384


def _lean_forward(self, x_BRE, single_eval_pos, save_peak_memory_factor=None, *,
                  cached_kv=None, return_kv=False):
    if not return_kv or cached_kv is not None:
        return _orig_forward(self, x_BRE, single_eval_pos, save_peak_memory_factor,
                             cached_kv=cached_kv, return_kv=return_kv)
    att = self.icl_attention
    x = x_BRE.contiguous()
    B, R, _ = x.shape
    N = R if single_eval_pos is None else single_eval_pos
    nh = att.num_kv_heads_test
    CH = CHUNK
    with torch.no_grad():
        K = V = None
        for s in range(0, N, CH):
            e = min(s + CH, N)
            h = self.layernorm(x[:, s:e])
            k = att.k_projection(h).view(B, e - s, att.num_kv_heads, att.head_dim)
            v = att.v_projection(h).view(B, e - s, att.num_kv_heads, att.head_dim)
            k = att.k_norm(k)
            if K is None:
                K = torch.empty((B, N, att.num_kv_heads, att.head_dim), dtype=v.dtype,
                                device=x.device)
                V = torch.empty_like(K)
            K[:, s:e] = k.to(v.dtype)
            V[:, s:e] = v
            del h, k, v
        for s in list(range(0, N, CH)) + list(range(N, R, CH)):
            e = min(s + CH, N if s < N else R)
            xc = x[:, s:e]
            q = att.q_norm(att.q_projection(self.layernorm(xc))
                           .view(B, e - s, att.num_heads, att.head_dim))
            if s < N or nh is None or N == R:
                out = A._batched_scaled_dot_product_attention(
                    q, K, V, softmax_scaling_layer=att.softmax_scaling_layer)
            else:
                out = A._batched_scaled_dot_product_attention(
                    q, K[:, :, :nh], V[:, :, :nh], softmax_scaling_layer=att.softmax_scaling_layer)
            xc.add_(att.out_projection(out.reshape(B, e - s, att.head_dim * att.num_heads)))
            del q, out
        if nh is not None:
            k_cache, v_cache = K[:, :, :nh].contiguous(), V[:, :, :nh].contiguous()
        else:
            k_cache, v_cache = K, V
        del K, V
        kv_entry = A.KVCacheEntry(key=k_cache.detach(), value=v_cache.detach())
        for s in range(0, R, CH):
            xc = x[:, s:min(s + CH, R)]
            xc.add_(self.mlp(self.layernorm_mlp(xc)))
    return x, kv_entry


def apply(chunk=16384):
    global CHUNK
    CHUNK = int(chunk)
    A.ICLTransformerBlock.forward = _lean_forward
