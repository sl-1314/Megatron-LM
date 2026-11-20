# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Batch-invariant (vLLM BI=1 matched) autograd operators for Qwen3.5 lite."""

from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from vllm import _custom_ops as vllm_ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.kernels.triton.fused_sigmoid_gate_mul import fused_sigmoid_gate_mul
from vllm.model_executor.determinism.batch_invariant import (
    linear_batch_invariant,
    mean_dim as bi_mean_dim_kernel,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_batch_invariant_moe_config,
    invoke_fused_moe_triton_kernel,
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    vllm_topk_softmax,
)
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.models.qwen3_5.common.fused_gdn_gated_norm import fused_gdn_gated_rms_norm
from vllm.third_party.flash_linear_attention.ops import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn

try:
    from flashinfer.norm import gemma_fused_add_rmsnorm as _fi_gemma_fused_add_rmsnorm
    from flashinfer.norm import gemma_rmsnorm as _fi_gemma_rmsnorm
except ImportError:
    _fi_gemma_rmsnorm = None
    _fi_gemma_fused_add_rmsnorm = None
from vllm.triton_utils import tl
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs
from vllm.vllm_flash_attn import flash_attn_varlen_func

try:
    from fla.modules.l2norm import l2norm as _fla_l2norm
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,
    )
except ImportError:
    _fla_l2norm = None
    _fla_chunk_gated_delta_rule = None


def _compute_dtype_for_triton(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.float32:
        return tl.float32
    raise ValueError(f"Unsupported Triton MoE dtype: {dtype}")


@lru_cache(maxsize=8)
def get_vllm_mrope_cache(
    head_dim: int,
    max_position_embeddings: int,
    rope_theta: float,
    partial_rotary_factor: float,
    mrope_section: tuple[int, ...],
    device_str: str,
) -> tuple[torch.Tensor, int, list[int]]:
    with set_current_vllm_config(VllmConfig()):
        rope = get_rope(
            head_size=head_dim,
            max_position=max_position_embeddings,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": rope_theta,
                "partial_rotary_factor": partial_rotary_factor,
                "mrope_interleaved": True,
                "mrope_section": list(mrope_section),
            },
            dtype=torch.bfloat16,
            mrope_positions_factor=4,
        ).to(device=torch.device(device_str))
    return rope.cos_sin_cache, int(rope.rotary_dim), list(rope.mrope_section)


# ---------------------------------------------------------------------------
# Linear & Mean & GemmaRMSNorm
# ---------------------------------------------------------------------------


class _BILinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
        ctx.orig_w_dtype = weight.dtype
        if weight.dtype != x.dtype:
            weight = weight.to(x.dtype)
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(x.dtype)
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        return linear_batch_invariant(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_x = linear_batch_invariant(grad_out, weight.t().contiguous().to(grad_out.dtype))
        else:
            grad_x = None
        g2 = grad_out.reshape(-1, grad_out.shape[-1])
        if ctx.needs_input_grad[1]:
            x2 = x.reshape(-1, x.shape[-1])
            gw = linear_batch_invariant(g2.t().contiguous(), x2.t().contiguous())
            grad_weight = gw.to(ctx.orig_w_dtype) if gw.dtype != ctx.orig_w_dtype else gw
        else:
            grad_weight = None
        grad_bias = g2.sum(dim=0) if (ctx.has_bias and ctx.needs_input_grad[2]) else None
        return grad_x, grad_weight, grad_bias


def bi_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return _BILinearFn.apply(x, weight, bias)


class _BIMeanLastDimFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        ctx.dim_size = x.shape[-1]
        ctx.x_shape = x.shape
        return bi_mean_dim_kernel(x, dim=-1, keepdim=True, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return grad_out.expand(ctx.x_shape) / float(ctx.dim_size)


class _BIGemmaRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float):
        orig_shape = x.shape
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        ctx.save_for_backward(x2, weight)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        if _fi_gemma_rmsnorm is not None and x2.is_cuda:
            w_cast = weight.to(x2.dtype) if weight.dtype != x2.dtype else weight
            out = _fi_gemma_rmsnorm(x2, w_cast, eps)
            return out.view(orig_shape)
        xf = x2.to(torch.float32)
        variance = bi_mean_dim_kernel(xf.pow(2), dim=-1, keepdim=True, dtype=torch.float32)
        w = weight.to(torch.float32) + 1.0
        return (xf * torch.rsqrt(variance + eps) * w).to(x.dtype).view(orig_shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x2, weight = ctx.saved_tensors
        with torch.enable_grad():
            x_in = x2.detach().requires_grad_(True)
            w_in = weight.detach().requires_grad_(True)
            xf = x_in.float()
            var = xf.pow(2).mean(dim=-1, keepdim=True)
            ref = (xf * torch.rsqrt(var + ctx.eps) * (w_in.float() + 1.0)).to(x_in.dtype)
            gx, gw = torch.autograd.grad(
                ref, (x_in, w_in), grad_out.reshape(-1, x2.shape[-1])
            )
        return gx.view(ctx.orig_shape), gw, None


class _BIFusedAddGemmaRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ):
        orig_shape = x.shape
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        r2 = residual.reshape(-1, residual.shape[-1]).contiguous()
        ctx.save_for_backward(x2, r2, weight)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        if _fi_gemma_fused_add_rmsnorm is not None and x2.is_cuda:
            x_buf = x2.clone()
            r_buf = r2.clone()
            w_cast = weight.to(x2.dtype) if weight.dtype != x2.dtype else weight
            _fi_gemma_fused_add_rmsnorm(x_buf, r_buf, w_cast, eps)
            return x_buf.view(orig_shape), r_buf.view(orig_shape)
        xf = x2.to(torch.float32) + r2.to(torch.float32)
        res_out = xf.to(x.dtype)
        variance = bi_mean_dim_kernel(xf.pow(2), dim=-1, keepdim=True, dtype=torch.float32)
        w = weight.to(torch.float32) + 1.0
        normed = (xf * torch.rsqrt(variance + eps) * w).to(x.dtype)
        return normed.view(orig_shape), res_out.view(orig_shape)

    @staticmethod
    def backward(ctx, grad_norm: torch.Tensor, grad_res: torch.Tensor):
        x2, r2, weight = ctx.saved_tensors
        with torch.enable_grad():
            x_in = x2.detach().requires_grad_(True)
            r_in = r2.detach().requires_grad_(True)
            w_in = weight.detach().requires_grad_(True)
            xf = x_in.float() + r_in.float()
            res_ref = xf.to(x_in.dtype)
            var = xf.pow(2).mean(dim=-1, keepdim=True)
            norm_ref = (xf * torch.rsqrt(var + ctx.eps) * (w_in.float() + 1.0)).to(
                x_in.dtype
            )
            gx, gr, gw = torch.autograd.grad(
                (norm_ref, res_ref),
                (x_in, r_in, w_in),
                (
                    grad_norm.reshape(-1, x2.shape[-1]),
                    grad_res.reshape(-1, r2.shape[-1]),
                ),
            )
        return gx.view(ctx.orig_shape), gr.view(ctx.orig_shape), gw, None


def bi_gemma_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return _BIGemmaRMSNormFn.apply(x, weight, eps)


def bi_fused_add_gemma_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    return _BIFusedAddGemmaRMSNormFn.apply(x, residual, weight, eps)


class _BISigmoidGateMulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gate: torch.Tensor):
        ctx.save_for_backward(x, gate)
        return fused_sigmoid_gate_mul(x.contiguous(), gate.contiguous())

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, gate = ctx.saved_tensors
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            g_in = gate.detach().requires_grad_(True)
            ref = (x_in.float() * torch.sigmoid(g_in.float())).to(x_in.dtype)
            gx, gg = torch.autograd.grad(ref, (x_in, g_in), grad_out)
        return gx, gg


def bi_sigmoid_gate_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return _BISigmoidGateMulFn.apply(x, gate)


# ---------------------------------------------------------------------------
# Fused QK-RMSNorm + MRoPE + Gate (Full Attention)
# ---------------------------------------------------------------------------


def _ref_mrope_interleaved_apply(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    mrope_section: list[int],
) -> torch.Tensor:
    # x: [T, H, D], cos/sin: [3, T, half_rotary]
    half = rotary_dim // 2
    rot_offs = torch.arange(half, device=x.device)
    is_h = (rot_offs % 3 == 1) & (rot_offs < 3 * mrope_section[1])
    is_w = (rot_offs % 3 == 2) & (rot_offs < 3 * mrope_section[2])
    c = torch.where(is_h, cos[1], torch.where(is_w, cos[2], cos[0])).unsqueeze(1)
    s = torch.where(is_h, sin[1], torch.where(is_w, sin[2], sin[0])).unsqueeze(1)
    x_rot = x[..., :rotary_dim].float()
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    o1 = (x1 * c - x2 * s).to(x.dtype)
    o2 = (x2 * c + x1 * s).to(x.dtype)
    if rotary_dim < x.shape[-1]:
        return torch.cat([o1, o2, x[..., rotary_dim:]], dim=-1)
    return torch.cat([o1, o2], dim=-1)


class _BIQKRmsNormRopeGateFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_gate: torch.Tensor,
        k: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        eps: float,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_dim: int,
        mrope_section: tuple[int, ...],
    ):
        q_out, k_out, g_out = fused_qk_rmsnorm_rope_gate(
            q_gate,
            k,
            q_weight.to(q_gate.dtype) if q_weight.dtype != q_gate.dtype else q_weight,
            k_weight.to(k.dtype) if k_weight.dtype != k.dtype else k_weight,
            cos_sin_cache,
            positions,
            eps,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            norm_beta=1.0,
            mrope_section=list(mrope_section),
        )
        ctx.save_for_backward(q_gate, k, q_weight, k_weight, cos_sin_cache, positions)
        ctx.eps = eps
        ctx.num_q_heads = num_q_heads
        ctx.num_kv_heads = num_kv_heads
        ctx.head_dim = head_dim
        ctx.rotary_dim = rotary_dim
        ctx.mrope_section = list(mrope_section)
        return q_out, k_out, g_out

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor, grad_k: torch.Tensor, grad_g: torch.Tensor):
        q_gate, k, q_weight, k_weight, cos_sin_cache, positions = ctx.saved_tensors
        with torch.enable_grad():
            qg_in = q_gate.detach().requires_grad_(True)
            k_in = k.detach().requires_grad_(True)
            qw_in = q_weight.detach().requires_grad_(True)
            kw_in = k_weight.detach().requires_grad_(True)

            T = qg_in.shape[0]
            qg_view = qg_in.view(T, ctx.num_q_heads, 2 * ctx.head_dim)
            q_raw = qg_view[..., : ctx.head_dim]
            g_ref = qg_view[..., ctx.head_dim :].reshape(T, ctx.num_q_heads * ctx.head_dim)
            k_raw = k_in.view(T, ctx.num_kv_heads, ctx.head_dim)

            def _rms_gemma(t, w):
                tf = t.float()
                var = tf.pow(2).mean(dim=-1, keepdim=True)
                return (tf * torch.rsqrt(var + ctx.eps) * (w.float() + 1.0)).to(t.dtype)

            qn = _rms_gemma(q_raw, qw_in)
            kn = _rms_gemma(k_raw, kw_in)
            cs = cos_sin_cache[positions]  # [3, T, rotary_dim]
            cos, sin = cs.chunk(2, dim=-1)
            q_ref = _ref_mrope_interleaved_apply(
                qn, cos, sin, ctx.rotary_dim, ctx.mrope_section
            ).reshape(T, ctx.num_q_heads * ctx.head_dim)
            k_ref = _ref_mrope_interleaved_apply(
                kn, cos, sin, ctx.rotary_dim, ctx.mrope_section
            ).reshape(T, ctx.num_kv_heads * ctx.head_dim)

            gq, gk, gqw, gkw = torch.autograd.grad(
                (q_ref, k_ref, g_ref),
                (qg_in, k_in, qw_in, kw_in),
                (grad_q, grad_k, grad_g),
            )
        return gq, gk, gqw, gkw, None, None, None, None, None, None, None, None


def bi_qk_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope_section: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _BIQKRmsNormRopeGateFn.apply(
        q_gate,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        tuple(mrope_section),
    )


# ---------------------------------------------------------------------------
# FlashAttention-3 BI Forward + ATen Flash Backward
# ---------------------------------------------------------------------------


class _BIFlashAttnVarlenFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_q: int,
        max_k: int,
        scale: float,
    ):
        out, lse = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            max_seqlen_q=max_q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_k,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=scale,
            causal=True,
            fa_version=3,
            num_splits=1,
            return_softmax_lse=True,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.max_q = max_q
        ctx.max_k = max_k
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        rng_state = torch.zeros((2,), dtype=torch.uint64, device=q.device)
        unused = torch.empty(0, device=q.device)
        dq, dk, dv = torch.ops.aten._flash_attention_backward(
            grad_out.contiguous(),
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_q,
            ctx.max_k,
            0.0,
            True,
            rng_state,
            unused,
            scale=ctx.scale,
            window_size_left=-1,
            window_size_right=0,
        )
        return dq, dk, dv, None, None, None, None, None


def bi_flash_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_q: int,
    max_k: int,
    scale: float,
) -> torch.Tensor:
    return _BIFlashAttnVarlenFn.apply(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_q, max_k, scale
    )


class _AllGatherCPFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cp_group):
        ctx.cp_group = cp_group
        cp_size = dist.get_world_size(cp_group)
        parts = [torch.empty_like(x) for _ in range(cp_size)]
        dist.all_gather(parts, x.contiguous(), group=cp_group)
        return tuple(parts)

    @staticmethod
    def backward(ctx, *grad_parts: torch.Tensor):
        cp_group = ctx.cp_group
        stacked = torch.stack([g.contiguous() for g in grad_parts], dim=0)
        grad_local = torch.empty_like(stacked[0])
        dist.reduce_scatter_tensor(grad_local, stacked, group=cp_group)
        return grad_local, None


def bi_all_gather_cp(x: torch.Tensor, cp_group) -> list[torch.Tensor]:
    return list(_AllGatherCPFn.apply(x, cp_group))


# ---------------------------------------------------------------------------
# GatedDeltaNet BI Ops (causal_conv1d_fn + fused_sigmoid_gating + rmsnorm_fn)
# ---------------------------------------------------------------------------


class _BICausalConv1dFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        conv_weight: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
    ):
        # x: [T, dim] when packed, or [batch * seq_len, dim]
        dim = x.shape[-1]
        width = conv_weight.shape[-1]
        if cu_seqlens is None:
            cu_cpu = torch.arange(batch_size + 1, dtype=torch.int32) * seq_len
            cu_dev = cu_cpu.to(device=x.device)
            num_seqs = batch_size
        else:
            cu_dev = cu_seqlens.to(dtype=torch.int32, device=x.device)
            cu_cpu = cu_dev.cpu()
            num_seqs = cu_cpu.numel() - 1

        nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
            cu_cpu, device=x.device
        )
        metadata = SimpleNamespace(
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        # causal_conv1d_fn mutates its input in-place; clone so x stays intact for backward.
        x_t = x.transpose(0, 1).clone()
        w_conv = conv_weight.to(x.dtype) if conv_weight.dtype != x.dtype else conv_weight
        conv_states = torch.zeros(
            (num_seqs + 1, dim, width - 1), dtype=x.dtype, device=x.device
        )
        has_initial_state = torch.zeros((num_seqs,), dtype=torch.bool, device=x.device)
        cache_indices = torch.arange(1, num_seqs + 1, dtype=torch.int32, device=x.device)
        out_t = causal_conv1d_fn(
            x_t,
            w_conv,
            None,
            activation="silu",
            conv_states=conv_states,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=cu_dev,
            metadata=metadata,
        )
        out = out_t.transpose(0, 1).contiguous()
        ctx.save_for_backward(x, conv_weight, cu_cpu)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, conv_weight, cu_cpu = ctx.saved_tensors
        dim, width = conv_weight.shape
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            w_in = conv_weight.detach().requires_grad_(True)
            refs = []
            for i in range(cu_cpu.numel() - 1):
                s = int(cu_cpu[i].item())
                e = int(cu_cpu[i + 1].item())
                if e <= s:
                    continue
                xi = x_in[s:e].transpose(0, 1).unsqueeze(0)
                ci = F.silu(
                    F.conv1d(xi, w_in.unsqueeze(1), padding=width - 1, groups=dim)[
                        :, :, : e - s
                    ]
                )
                refs.append(ci.squeeze(0).transpose(0, 1))
            ref = torch.cat(refs, dim=0)
            gx, gw = torch.autograd.grad(ref, (x_in, w_in), grad_out)
        return gx, gw, None, None, None


def bi_causal_conv1d(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    return _BICausalConv1dFn.apply(x, conv_weight, cu_seqlens, batch_size, seq_len)


class _BIFusedGdnFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A_log: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
    ):
        # q, k: [T_total, nkh, dk]; v: [T_total, nvh, dv]; a, b: [T_total, nvh]
        if cu_seqlens is None:
            cu_dev = (
                torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
                * seq_len
            )
            num_seqs = batch_size
        else:
            cu_dev = cu_seqlens.to(dtype=torch.int32, device=q.device)
            num_seqs = cu_dev.numel() - 1
        nvh = v.shape[1]
        dk = q.shape[2]
        dv = v.shape[2]
        ssm_state = torch.zeros(
            (num_seqs + 1, nvh, dv, dk), dtype=torch.float32, device=q.device
        )
        ssm_indices = torch.arange(1, num_seqs + 1, dtype=torch.int32, device=q.device)
        has_init = torch.zeros((num_seqs,), dtype=torch.bool, device=q.device)
        out, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=v.unsqueeze(0),
            initial_state=ssm_state,
            inplace_final_state=False,
            cu_seqlens=cu_dev,
            ssm_state_indices=ssm_indices,
            has_initial_state=has_init,
            use_qk_l2norm_in_kernel=True,
        )
        out = out.squeeze(0)
        ctx.save_for_backward(A_log, a, b, dt_bias, q, k, v, cu_dev)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if _fla_chunk_gated_delta_rule is None or _fla_l2norm is None:
            raise RuntimeError("FLA chunk_gated_delta_rule is required for GDN backward.")
        A_log, a, b, dt_bias, q, k, v, cu_dev = ctx.saved_tensors
        nkh = q.shape[1]
        nvh = v.shape[1]
        v_per_k = nvh // nkh
        with torch.enable_grad():
            A_in = A_log.detach().requires_grad_(True)
            a_in = a.detach().requires_grad_(True)
            b_in = b.detach().requires_grad_(True)
            dt_in = dt_bias.detach().requires_grad_(True)
            q_in = q.detach().requires_grad_(True)
            k_in = k.detach().requires_grad_(True)
            v_in = v.detach().requires_grad_(True)

            qn = _fla_l2norm(q_in).unsqueeze(0)
            kn = _fla_l2norm(k_in).unsqueeze(0)
            if v_per_k > 1:
                qn = qn.repeat_interleave(v_per_k, dim=2)
                kn = kn.repeat_interleave(v_per_k, dim=2)
            g = (-A_in.float().exp() * F.softplus(a_in.float() + dt_in.float())).unsqueeze(0)
            beta = b_in.sigmoid().unsqueeze(0)
            out_ref, _ = _fla_chunk_gated_delta_rule(
                qn,
                kn,
                v_in.unsqueeze(0),
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                cu_seqlens=cu_dev,
            )
            grads = torch.autograd.grad(
                out_ref.squeeze(0),
                (A_in, a_in, b_in, dt_in, q_in, k_in, v_in),
                grad_out,
            )
        return (*grads, None, None, None)


def bi_fused_gdn(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    return _BIFusedGdnFn.apply(
        A_log, a, b, dt_bias, q, k, v, cu_seqlens, batch_size, seq_len
    )


class _BIRMSNormGatedFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float
    ):
        # x, z: [T, nvh, dv]
        orig_shape = x.shape
        T, nvh, dv = orig_shape
        x2 = x.reshape(T, nvh * dv).contiguous()
        z2 = z.reshape(T, nvh * dv).contiguous()
        out = torch.empty_like(x2)
        w_cast = weight.to(x2.dtype) if weight.dtype != x2.dtype else weight
        fused_gdn_gated_rms_norm(
            x=x2,
            z=z2,
            weight=w_cast,
            out=out,
            eps=eps,
            num_v_heads=nvh,
            head_v_dim=dv,
            norm_before_gate=True,
            activation="silu",
        )
        ctx.save_for_backward(x.reshape(-1, dv).contiguous(), z.reshape(-1, dv).contiguous(), weight)
        ctx.eps = eps
        return out.view(orig_shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x2, z2, weight = ctx.saved_tensors
        orig_shape = grad_out.shape
        dv = x2.shape[-1]
        with torch.enable_grad():
            x_in = x2.detach().requires_grad_(True)
            z_in = z2.detach().requires_grad_(True)
            w_in = weight.detach().requires_grad_(True)
            xf = x_in.float()
            var = xf.pow(2).mean(dim=-1, keepdim=True)
            x_hat = xf * torch.rsqrt(var + ctx.eps)
            zf = z_in.float()
            y = x_hat * w_in.float() * (zf * torch.sigmoid(zf))
            y = y.to(x_in.dtype)
            gx, gz, gw = torch.autograd.grad(
                y, (x_in, z_in, w_in), grad_out.reshape(-1, dv)
            )
        return gx.view(orig_shape), gz.view(orig_shape), gw, None


def bi_rms_norm_gated(
    x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return _BIRMSNormGatedFn.apply(x, z, weight, eps)


# ---------------------------------------------------------------------------
# MoE BI Ops (silu_and_mul, vllm_topk_softmax, TritonExperts, moe_sum)
# ---------------------------------------------------------------------------


class _BISiluAndMulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        d = x.shape[-1] // 2
        out = torch.empty((*x.shape[:-1], d), dtype=x.dtype, device=x.device)
        torch.ops._C.silu_and_mul(out, x)
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            g, u = x_in.chunk(2, dim=-1)
            ref = F.silu(g) * u
            (gx,) = torch.autograd.grad(ref, x_in, grad_out)
        return gx


def bi_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    return _BISiluAndMulFn.apply(x)


class _BITopKSoftmaxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, topk: int, renormalize: bool):
        M = logits.shape[0]
        tw = torch.empty((M, topk), dtype=torch.float32, device=logits.device)
        ti = torch.empty((M, topk), dtype=torch.int32, device=logits.device)
        tei = torch.empty((M, topk), dtype=torch.int32, device=logits.device)
        vllm_topk_softmax(tw, ti, tei, logits, renormalize)
        ctx.save_for_backward(logits, ti)
        ctx.renormalize = renormalize
        ctx.mark_non_differentiable(ti)
        return tw, ti

    @staticmethod
    def backward(ctx, grad_tw: torch.Tensor, grad_ti: torch.Tensor | None):
        del grad_ti
        logits, ti = ctx.saved_tensors
        with torch.enable_grad():
            l_in = logits.detach().requires_grad_(True)
            probs = F.softmax(l_in.float(), dim=-1)
            sel = probs.gather(-1, ti.long())
            if ctx.renormalize:
                sel = sel / sel.sum(dim=-1, keepdim=True)
            (gl,) = torch.autograd.grad(sel, l_in, grad_tw)
        return gl, None, None


def bi_topk_softmax(
    logits: torch.Tensor, topk: int, renormalize: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    return _BITopKSoftmaxFn.apply(logits, topk, renormalize)


class _BITritonExpertsFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        *weights: torch.Tensor,
    ):
        w1_params = weights[:num_experts]
        w2_params = weights[num_experts:]
        num_tokens = x.shape[0]
        top_k_num = topk_ids.shape[1]
        K = x.shape[1]
        ctx.save_for_backward(x, topk_weights, topk_ids)
        ctx.num_experts = num_experts
        ctx.w1_params = w1_params
        ctx.w2_params = w2_params
        if num_tokens == 0:
            return torch.empty((0, top_k_num, K), dtype=x.dtype, device=x.device)

        def _to_local_bf16(w: torch.Tensor) -> torch.Tensor:
            if hasattr(w, "to_local"):
                w = w.to_local()
            return w.detach().to(x.dtype)

        w1 = torch.stack([_to_local_bf16(w) for w in w1_params], dim=0).contiguous()
        w2 = torch.stack([_to_local_bf16(w) for w in w2_params], dim=0).contiguous()
        E, N, _ = w1.shape

        cfg = get_batch_invariant_moe_config(E, N, K, None, None)
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, cfg["BLOCK_SIZE_M"], E
        )
        # Keep the Triton TMA allocator active on this device before fused_moe_kernel
        # (and ensure `num_tokens_post_padded` is on GPU).
        ic1 = torch.empty((num_tokens, top_k_num, N), dtype=x.dtype, device=x.device)
        ic2 = torch.empty(
            (num_tokens * top_k_num, N // 2), dtype=x.dtype, device=x.device
        )
        ic3 = torch.empty((num_tokens, top_k_num, K), dtype=x.dtype, device=x.device)
        compute_type = _compute_dtype_for_triton(x.dtype)
        # Note: w1/w2 are plain contiguous CUDA tensors (from .to_local()) on x.device.
        invoke_fused_moe_triton_kernel(
            x,
            w1,
            ic1,
            None,
            None,
            None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,
            top_k_num,
            cfg,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
        )
        torch.ops._C.silu_and_mul(ic2, ic1.view(-1, N))
        invoke_fused_moe_triton_kernel(
            ic2,
            w2,
            ic3,
            None,
            None,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            True,
            1,
            cfg,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
        )
        del w1, w2, ic1, ic2
        return ic3

    @staticmethod
    def backward(ctx, grad_ic3: torch.Tensor):
        x, topk_weights, topk_ids = ctx.saved_tensors
        E = ctx.num_experts
        w1_params = ctx.w1_params
        w2_params = ctx.w2_params
        grad_w1_list = [torch.zeros_like(w.to_local() if hasattr(w, "to_local") else w) for w in w1_params]
        grad_w2_list = [torch.zeros_like(w.to_local() if hasattr(w, "to_local") else w) for w in w2_params]
        if x.shape[0] == 0:
            def _wrap_dt(g, w):
                if hasattr(w, "to_local"):
                    from torch.distributed.tensor import DTensor
                    return DTensor.from_local(g, w.device_mesh, w.placements, run_check=False)
                return g
            return (
                torch.zeros_like(x),
                torch.zeros_like(topk_weights),
                None,
                None,
                *[_wrap_dt(g, w) for g, w in zip(grad_w1_list, w1_params, strict=True)],
                *[_wrap_dt(g, w) for g, w in zip(grad_w2_list, w2_params, strict=True)],
            )
        grad_x = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        grad_tw = torch.zeros_like(topk_weights)

        flat_ids = topk_ids.reshape(-1)
        flat_tok = (
            torch.arange(x.shape[0], device=x.device)
            .unsqueeze(1)
            .expand_as(topk_ids)
            .reshape(-1)
        )
        flat_slot = (
            torch.arange(topk_ids.shape[1], device=x.device)
            .unsqueeze(0)
            .expand_as(topk_ids)
            .reshape(-1)
        )
        flat_tw = topk_weights.reshape(-1)
        flat_grad = grad_ic3.reshape(-1, grad_ic3.shape[-1])

        order = torch.argsort(flat_ids, stable=True)
        sorted_e = flat_ids[order]
        counts = torch.bincount(sorted_e.long(), minlength=E).tolist()
        sorted_tok = flat_tok[order]
        sorted_slot = flat_slot[order]
        sorted_tw = flat_tw[order]
        sorted_grad = flat_grad[order]

        offset = 0
        for e_idx, cnt in enumerate(counts):
            if cnt == 0:
                continue
            sl = slice(offset, offset + cnt)
            offset += cnt
            tok_e = sorted_tok[sl]
            slot_e = sorted_slot[sl]
            tw_e = sorted_tw[sl]
            g_e = sorted_grad[sl]
            x_e = x[tok_e]
            with torch.enable_grad():
                xe_in = x_e.detach().requires_grad_(True)
                w1_loc = (
                    w1_params[e_idx].to_local()
                    if hasattr(w1_params[e_idx], "to_local")
                    else w1_params[e_idx]
                )
                w2_loc = (
                    w2_params[e_idx].to_local()
                    if hasattr(w2_params[e_idx], "to_local")
                    else w2_params[e_idx]
                )
                w1_in = w1_loc.detach().to(x_e.dtype).requires_grad_(True)
                w2_in = w2_loc.detach().to(x_e.dtype).requires_grad_(True)
                tw_in = tw_e.detach().requires_grad_(True)
                h1 = F.linear(xe_in, w1_in)
                g1, u1 = h1.chunk(2, dim=-1)
                h2 = F.silu(g1) * u1
                out_e = F.linear(h2, w2_in) * tw_in.unsqueeze(-1).to(h2.dtype)
                gxe, gw1e, gw2e, gtwe = torch.autograd.grad(
                    out_e, (xe_in, w1_in, w2_in, tw_in), g_e
                )
            grad_x.index_add_(0, tok_e, gxe.float())
            grad_w1_list[e_idx].copy_(gw1e)
            grad_w2_list[e_idx].copy_(gw2e)
            grad_tw[tok_e, slot_e] = gtwe
        def _wrap_dt_out(g, w):
            if hasattr(w, "to_local"):
                from torch.distributed.tensor import DTensor
                return DTensor.from_local(g, w.device_mesh, w.placements, run_check=False)
            return g
        out_gw1 = [_wrap_dt_out(g, w) for g, w in zip(grad_w1_list, w1_params, strict=True)]
        out_gw2 = [_wrap_dt_out(g, w) for g, w in zip(grad_w2_list, w2_params, strict=True)]
        return grad_x.to(x.dtype), grad_tw, None, None, *out_gw1, *out_gw2


def bi_triton_experts(
    x: torch.Tensor,
    w1_params: tuple[torch.Tensor, ...] | list[torch.Tensor],
    w2_params: tuple[torch.Tensor, ...] | list[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    num_experts = len(w1_params)
    assert len(w2_params) == num_experts
    return _BITritonExpertsFn.apply(
        x, topk_weights, topk_ids, num_experts, *w1_params, *w2_params
    )


class _BIMoeSumFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, slots: torch.Tensor):
        ctx.shape = slots.shape
        out = torch.empty(
            (slots.shape[0], slots.shape[2]), dtype=slots.dtype, device=slots.device
        )
        vllm_ops.moe_sum(slots, out)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return grad_out.unsqueeze(1).expand(ctx.shape)


def bi_moe_sum(slots: torch.Tensor) -> torch.Tensor:
    return _BIMoeSumFn.apply(slots)


# ---------------------------------------------------------------------------
# Vocab Cross-Entropy BI (matches vLLM _topk_log_softmax_kernel)
# ---------------------------------------------------------------------------


class _BIVocabCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, target: torch.Tensor):
        orig_shape = target.shape
        l2 = logits.reshape(-1, logits.shape[-1])
        t1 = target.reshape(-1)
        logprobs = compute_token_logprobs(l2, t1.unsqueeze(-1)).squeeze(-1)
        ctx.save_for_backward(logits, target)
        return (-logprobs).view(orig_shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        logits, target = ctx.saved_tensors
        l2 = logits.reshape(-1, logits.shape[-1])
        t1 = target.reshape(-1)
        g1 = grad_out.reshape(-1, 1)
        grad_l2 = torch.empty_like(l2)
        chunk = 512
        for start in range(0, l2.shape[0], chunk):
            end = min(start + chunk, l2.shape[0])
            sm = F.softmax(l2[start:end].float(), dim=-1)
            rows = torch.arange(end - start, device=l2.device)
            sm[rows, t1[start:end].long()] -= 1.0
            sm.mul_(g1[start:end])
            grad_l2[start:end].copy_(sm.to(l2.dtype))
        return grad_l2.view_as(logits), None


def bi_vocab_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _BIVocabCrossEntropyFn.apply(logits, target)
