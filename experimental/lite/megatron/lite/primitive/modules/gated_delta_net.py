# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen-style Gated DeltaNet primitive.

Context-parallel (CP) modes
---------------------------
- ``headwise`` (default under CP): head-parallel all-to-all. Each rank all-to-alls
  its zigzag sequence shard so it ends up holding the *full* sequence for ``1/cp`` of
  the heads (with the matching slices of conv1d / A_log / dt_bias), runs the ordinary
  full-sequence gated-delta recurrence, then all-to-alls back. Heads are independent,
  so this is numerically **bitwise-identical to CP-off** while sharding the per-head
  state/activation memory across ranks. This mirrors upstream Megatron
  ``linear_cp_mode='headwise'`` (``megatron/core/ssm/gated_delta_net.py``).
  but every rank materialises the full sequence for *all* heads (worst memory).
- ``chunkwise``: FLA ``cp_context`` ring path. Each rank keeps its ``1/cp`` sequence
  shard *and* all heads; the recurrence's per-chunk state is passed around the CP ring
  by the FLA kernel. This shards both sequence and per-head state memory (best memory).
  Chunkwise is a *packing-aware faithful mirror* of upstream Megatron
  ``linear_cp_mode='chunkwise'`` (``megatron/core/ssm/gated_delta_net.py`` @ d1384c2d9):
  the Megatron zigzag CP layout is reshuffled to contiguous-time chunks with a single
  packing-aware all-to-all keyed on the *global* ``cu_seqlens`` (``_resolve_cu_seqlens``
  validates padded lengths and per-sequence ``% cp_size`` divisibility), the FLA
  recurrence runs, then the output is reshuffled back to zigzag. Because the ring
  accumulates chunk state across ranks, chunkwise matches CP-off at the bf16 floor
  rather than bitwise (``headwise`` is bitwise-exact).

An earlier local chunkwise (``sharded``) copy diverged from upstream on the packed/THD
path — it sliced ``cu_seqlens // cp_size`` per sequence and swapped each slice as if it
were unpacked, corrupting multi-sequence / non-``cp``-divisible reshuffles and producing
a large RL train/inference log-prob mismatch. This module restores chunkwise as a
faithful mirror of the upstream packing-aware reshuffle.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# 0-diff: native-baseline switch (C7 layer-3 three-arm comparison) — see
# model/qwen3_5/lite/model.py; contracts stay the default.
_NATIVE_MODE = os.environ.get("MLITE_VLLM_CONTRACTS", "1") == "0"

from megatron.lite.primitive import transformer_engine as te
from megatron.lite.primitive.kernels.jit import jit_fuser
from megatron.lite.primitive.ops.gated_delta_rule import (
    l2norm,
    torch_chunk_gated_delta_rule,
)
from megatron.lite.primitive.parallel import (
    ColumnParallelLinear,
    ParallelState,
    RowParallelLinear,
    all_gather_last_dim_with_grad_reduce,
)
from megatron.lite.primitive.parallel.cp import (
    all_to_all_hidden_shards,
    build_headwise_section_perm,
    contiguous_to_zigzag_chunks,
    get_parameter_local_cp_headwise,
    zigzag_reconstruct_from_cp_parts,
    zigzag_slice_for_cp,
    zigzag_to_contiguous_chunks,
)
from megatron.lite.primitive.parallel.thd import (
    reconstruct_packed_from_cp_parts,
    split_packed_to_cp_local,
)
from megatron.lite.primitive.utils import ensure_divisible


try:
    from fla.modules.convolution import (
        causal_conv1d as _fla_causal_conv1d,  # pyright: ignore[reportMissingImports]
    )
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,  # pyright: ignore[reportMissingImports]
    )

    _HAS_FLA = True
except ImportError:
    _HAS_FLA = False

try:
    from fla.ops.cp import (
        build_cp_context as _fla_build_cp_context,  # pyright: ignore[reportMissingImports]
    )
except ImportError:
    _fla_build_cp_context = None

_CONV_PAD_ALIGNMENT = 4096

_CP_MODES = {"headwise", "chunkwise"}

try:
    import triton  # pyright: ignore[reportMissingImports]
    import triton.language as tl  # pyright: ignore[reportMissingImports]

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

try:
    from flashinfer.norm import (  # pyright: ignore[reportMissingImports]
        gemma_fused_add_rmsnorm as _fi_gemma_fused_add_rmsnorm,
        gemma_rmsnorm as _fi_gemma_rmsnorm,
    )

    _HAS_FI_NORM = True
except Exception:
    _HAS_FI_NORM = False


# 0-diff (qwen35-0diff-lite): vLLM causal-conv contract replication -----------------
#
# The rollout contract owner is vLLM's Triton kernel
# (vllm/model_executor/layers/mamba/ops/causal_conv1d.py).  For a bf16, no-bias,
# silu conv of width W its arithmetic is:
#   acc = 0 (fp32)
#   for j in 0..W-1 (oldest -> newest):
#       acc += bf16(x) * bf16(w)          # product ROUNDED TO BF16, then fp32 add
#   out = bf16( acc / (1 + exp(-acc)) )   # silu in fp32
# The fla 0.6 kernel multiplies in fp32 (no product rounding), which alone
# accounts for the entire ~7e-5 absmean cross-side conv diff.  The torch
# elementwise bf16 mul rounds identically to Triton's, so the accumulation is
# plain torch; the silu, however, must run through Triton — torch.exp/torch.div
# differ from tl.exp/'/' by 1 fp32 ulp on rare values, which flips the bf16
# rounding at exact midpoints (~1e-6 of elements).


if _HAS_TRITON:

    @triton.jit
    def _vllm_conv_silu_kernel(acc_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        acc = tl.load(acc_ptr + offs, mask=mask)
        y = acc / (1 + tl.exp(-acc))
        tl.store(out_ptr + offs, y.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)

    class _VllmConvSilu(torch.autograd.Function):
        """Contract-exact silu (Triton forward, analytic fp32 backward)."""

        @staticmethod
        def forward(ctx, acc: torch.Tensor) -> torch.Tensor:
            ctx.save_for_backward(acc)
            n = acc.numel()
            out = torch.empty(n, device=acc.device, dtype=torch.bfloat16)
            _vllm_conv_silu_kernel[(triton.cdiv(n, 1024),)](
                acc.reshape(-1).contiguous(), out, n, BLOCK=1024
            )
            return out.view_as(acc)

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            (acc,) = ctx.saved_tensors
            s = torch.sigmoid(acc)
            grad = (s + acc * s * (1 - s)) * grad_out.to(torch.float32)
            return grad.to(acc.dtype)


def _vllm_conv1d_silu(
    x: torch.Tensor, weight: torch.Tensor, cu_seqlens: torch.Tensor | None
) -> torch.Tensor:
    """Bit-exact vLLM causal_conv1d (no bias, silu) for bf16 [B, T, D] input.

    ``weight`` is [D, W].  With ``cu_seqlens`` (THD packing, B=1) each sequence
    zero-pads at its own start; otherwise each batch row does.
    """
    B, T, D = x.shape
    W = weight.shape[-1]
    if cu_seqlens is not None:
        assert B == 1, "THD packing requires a single batch row"
        acc = torch.zeros(T, D, device=x.device, dtype=torch.float32)
        pos = torch.arange(T, device=x.device)
        # sequence start of each packed position
        seq_id = torch.searchsorted(cu_seqlens, pos, right=True) - 1
        starts = cu_seqlens[seq_id]
        x0 = x[0]
        for j in range(W):
            idx = pos - (W - 1 - j)
            xj = x0[idx.clamp(min=0)]
            xj = torch.where((idx >= starts)[:, None], xj, torch.zeros_like(xj))
            acc = acc + (xj * weight[:, j]).float()
        return _VllmConvSilu.apply(acc).unsqueeze(0)
    acc = torch.zeros(B, T, D, device=x.device, dtype=torch.float32)
    for j in range(W):
        xj = F.pad(x, (0, 0, W - 1 - j, 0))[:, :T]
        acc = acc + (xj * weight[:, j]).float()
    return _VllmConvSilu.apply(acc)


# 0-diff (qwen35-0diff-lite): vLLM fused_post_conv_prep gating contract ----------------
#
# vLLM computes the g/beta gating in fp32 inside the prep Triton kernel
# (vllm/third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py):
#   x  = fp32(a) + fp32(dt_bias)
#   sp = where(x > 0, x + log(1+exp(-x)), log(1+exp(x)))   # branch softplus
#   sp = where(x <= 20, sp, x)                             # threshold 20
#   g  = -exp(fp32(A_log)) * sp                            # fp32 exp
#   beta = sigmoid(fp32(b))
# mlite's torch formula rounded exp(A_log) to bf16 when A_log is a bf16
# parameter (100% of g elements off, up to 1e-1), used F.softplus's
# log1p(exp(x)) (ulp-level), and torch sigmoid differs from tl.sigmoid by
# 1 fp32 ulp on ~20% of beta values — so the forward must run in Triton.

if _HAS_TRITON:

    @triton.jit
    def _vllm_g_beta_kernel(
        a_ptr, b_ptr, alog_ptr, dtb_ptr, g_ptr, beta_ptr, n,
        HV: tl.constexpr, BH: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        o_h = tl.arange(0, BH)
        mask_h = o_h < HV
        m2 = mask[:, None] & mask_h[None, :]
        a = tl.load(a_ptr + offs[:, None] * HV + o_h[None, :], mask=m2, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offs[:, None] * HV + o_h[None, :], mask=m2, other=0.0).to(tl.float32)
        A_log = tl.load(alog_ptr + o_h, mask=mask_h, other=0.0).to(tl.float32)
        dtb = tl.load(dtb_ptr + o_h, mask=mask_h, other=0.0).to(tl.float32)
        x = a + dtb[None, :]
        sp = tl.where(x > 0, x + tl.log(1.0 + tl.exp(-x)), tl.log(1.0 + tl.exp(x)))
        sp = tl.where(x <= 20.0, sp, x)
        g = -tl.exp(A_log)[None, :] * sp
        beta = tl.sigmoid(b)
        tl.store(g_ptr + offs[:, None] * HV + o_h[None, :], g, mask=m2)
        tl.store(beta_ptr + offs[:, None] * HV + o_h[None, :], beta, mask=m2)

    class _VllmGBeta(torch.autograd.Function):
        """Contract-exact g/beta gating (Triton forward, analytic fp32 backward)."""

        @staticmethod
        def forward(ctx, alpha, beta_in, A_log, dt_bias):
            n, hv = alpha.shape
            g = torch.empty(n, hv, device=alpha.device, dtype=torch.float32)
            beta = torch.empty(n, hv, device=beta_in.device, dtype=torch.float32)
            _vllm_g_beta_kernel[(triton.cdiv(n, 256),)](
                alpha, beta_in, A_log, dt_bias, g, beta, n,
                HV=hv, BH=triton.next_power_of_2(hv), BLOCK=256,
            )
            ctx.save_for_backward(g, beta, alpha, A_log, dt_bias)
            return g, beta

        @staticmethod
        def backward(ctx, dg, dbeta):
            g, beta, alpha, A_log, dt_bias = ctx.saved_tensors
            # g = -exp(A_log) * softplus(x) with x = alpha + dt_bias
            coef = -A_log.float().exp()
            x = alpha.float() + dt_bias.float()
            sig = torch.sigmoid(x)
            grad_common = dg * (coef * sig)  # wrt both alpha and dt_bias
            grad_beta = dbeta * beta * (1.0 - beta)
            return (
                grad_common.to(alpha.dtype),
                grad_beta.to(alpha.dtype),
                (dg * g).to(A_log.dtype),
                grad_common.to(dt_bias.dtype),
            )


def _vllm_g_beta(alpha: torch.Tensor, beta: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor):
    """Bit-exact vLLM gating for contiguous ``[..., HV]`` alpha/beta inputs."""
    shape = beta.shape
    alpha2 = alpha.reshape(-1, shape[-1]).contiguous()
    beta2 = beta.reshape(-1, shape[-1]).contiguous()
    g, b = _VllmGBeta.apply(alpha2, beta2, A_log.contiguous(), dt_bias.contiguous())
    return g.view(shape), b.view(shape)


# 0-diff (qwen35-0diff-lite): vLLM gated output-norm contract -------------------------
#
# vLLM (nvidia variant) runs the fused Triton kernel
# vllm/models/qwen3_5/nvidia/ops/fused_gdn_gated_norm.py: per-head RMS norm,
# fp32 end to end, ONE bf16 rounding:
#   x fp32; xbar = where(mask, x, 0); var = sum(xbar*xbar)/N
#   rstd = rsqrt(var+eps); y = x*rstd*w;  y *= z*sigmoid(z);  store bf16
# launched as grid=(cdiv(M, ROWS_PER_BLOCK), num_v_heads), num_warps=1,
# ROWS_PER_BLOCK = calc_rows_per_block(M) — the kernel below is a verbatim
# port of that source (same signature, strides, constexprs, launch config),
# verified bitwise against the in-engine capture.
#
# Weight convention (the subtle one): the HF checkpoint stores
# linear_attn.norm.weight in FP32 (plain gamma), and vLLM loads it into a
# bf16 parameter, so the kernel consumes fp32(bf16(gamma)).  mlite stores the
# zero-centered bf16(gamma - 1) for te.RMSNorm, from which bf16(gamma) is NOT
# recoverable (double rounding), so load_hf_weights additionally caches the
# raw bf16 value in GDN._vllm_norm_w.  The analytic backward computes the
# gradient w.r.t. gamma and returns it straight to the zero-centered param
# (d(gamma)/d(w) = 1, STE through both bf16 roundings).
#
# mlite's legacy te.RMSNorm path rounds to bf16 BEFORE the gate multiply and
# TE's reduction differs — together 26% of elements at up to 1.6e-2.

if _HAS_TRITON:

    def _vllm_calc_rows_per_block(M: int, device: torch.device) -> int:
        # Port of layernorm_guard.calc_rows_per_block (num_compute_units ==
        # multi_processor_count on CUDA).
        import math

        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        return min(triton.next_power_of_2(math.ceil(M / (2 * sm_count))), 4)

    @triton.jit(do_not_specialize=["M"])
    def _vllm_gated_norm_kernel(
        x_ptr, z_ptr, w_ptr, out_ptr,
        stride_x_tok, stride_z_tok, stride_out_tok,
        M, eps,
        N: tl.constexpr, BLOCK_N: tl.constexpr, ROWS_PER_BLOCK: tl.constexpr,
        NORM_BEFORE_GATE: tl.constexpr, ACTIVATION: tl.constexpr,
    ):
        row_start = tl.program_id(0) * ROWS_PER_BLOCK
        head = tl.program_id(1)
        rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
        cols = tl.arange(0, BLOCK_N)
        row_mask = rows[:, None] < M
        col_mask = cols[None, :] < N
        mask = row_mask & col_mask
        x_ptrs = x_ptr + rows[:, None] * stride_x_tok + head * N + cols[None, :]
        z_ptrs = z_ptr + rows[:, None] * stride_z_tok + head * N + cols[None, :]
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        xbar = tl.where(mask, x, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N
        rstd = tl.rsqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        y = x * rstd[:, None] * w[None, :]
        z = tl.load(z_ptrs, mask=mask, other=0.0).to(tl.float32)
        y = y * (z * tl.sigmoid(z))
        out_ptrs = out_ptr + rows[:, None] * stride_out_tok + head * N + cols[None, :]
        tl.store(out_ptrs, y.to(out_ptr.dtype.element_ty), mask=mask)

    class _VllmGatedNorm(torch.autograd.Function):
        """Contract-exact gated RMSNorm (verbatim Triton forward, fp32 backward).

        ``w_raw`` is the vLLM-convention weight (bf16 gamma); ``w_param`` is
        the trainable zero-centered param that receives the gamma gradient.
        """

        @staticmethod
        def forward(ctx, x, z, w_param, w_raw, eps):
            num_tokens, width = x.shape
            num_v_heads = width // w_raw.shape[0]
            out = torch.empty_like(x)
            block_n = triton.next_power_of_2(w_raw.shape[0])
            rows_per_block = _vllm_calc_rows_per_block(num_tokens, x.device)
            grid = (triton.cdiv(num_tokens, rows_per_block), num_v_heads)
            _vllm_gated_norm_kernel[grid](
                x, z, w_raw, out,
                x.stride(0), z.stride(0), out.stride(0),
                num_tokens, eps,
                N=w_raw.shape[0], BLOCK_N=block_n, ROWS_PER_BLOCK=rows_per_block,
                NORM_BEFORE_GATE=True, ACTIVATION="silu",
                num_warps=min(max(block_n // 256, 1), 8),
            )
            ctx.save_for_backward(x, z, w_raw)
            ctx.eps = eps
            return out

        @staticmethod
        def backward(ctx, dy):
            x, z, w_raw = ctx.saved_tensors
            n = w_raw.shape[0]
            hv = z.shape[-1] // n
            xf = x.float().view(-1, hv, n)
            zf = z.float().view(-1, hv, n)
            wf = w_raw.float()
            dyf = dy.float().view(-1, hv, n)
            var = (xf * xf).sum(-1, keepdim=True) / n
            rstd = torch.rsqrt(var + ctx.eps)
            sig = torch.sigmoid(zf)
            s = zf * sig  # silu(z)
            A = rstd * wf * s * dyf
            # t is sum(dy * g * x) with NO rstd inside: rstd's own gradient
            # path is the rstd^3 factor below (an extra rstd here made the
            # correction term rstd x too large — rstd ~ 30-50 in this model,
            # exploding the backward; caught by C6 grad-vs-autograd probe).
            t = (dyf * (xf * wf * s)).sum(-1, keepdim=True)
            dx = A - (rstd * rstd * rstd) * (xf / n) * t
            dz = dyf * (xf * rstd * wf) * (sig * (1.0 + zf * (1.0 - sig)))
            dw = (dyf * (xf * rstd) * s).sum(dim=(0, 1))
            return (
                dx.view_as(x).to(x.dtype),
                dz.view_as(z).to(z.dtype),
                dw.to(w_raw.dtype),  # STE: d(gamma)/d(zero-centered w) == 1
                None,
                None,
            )


def _vllm_gated_norm(
    x: torch.Tensor,
    gate: torch.Tensor,
    w_param: torch.Tensor,
    w_raw: torch.Tensor,
    eps: float,
):
    """Bit-exact vLLM fused gated RMSNorm for bf16 ``[..., HV, V]`` x/gate."""
    shape = x.shape
    width = shape[-1] * shape[-2]  # HV * V
    x2 = x.reshape(-1, width).contiguous()
    g2 = gate.reshape(-1, width).contiguous()
    return _VllmGatedNorm.apply(x2, g2, w_param, w_raw, eps).view(shape)


# 0-diff (qwen35-0diff-lite): vLLM fused input-norm contract ---------------------------
#
# The rollout side (vllm/models/qwen3_5/nvidia/model.py Qwen3_5RMSNorm) norms
# the exact fp32 sum  s = fp32(x) + fp32(residual)  inside flashinfer's Gemma
# fused add+RMSNorm (Gemma weight: y = x_hat * (w + 1)), writes the residual
# stream back as bf16(s), and feeds the bf16 normed activation to the plain
# gemm.  Once the caller pre-adds the pair in bf16 the exact fp32 sum is lost,
# so the layer must hand the UN-ADDED pair down.  flashinfer is invoked
# directly (same kernel -> bitwise), with an analytic fp32 backward for
# training: y = s * rstd * (w+1) elementwise, res_out = round_bf16(s) treated
# as s (STE through the rounding).

if _HAS_FI_NORM:

    class _VllmGemmaFusedAddRmsNorm(torch.autograd.Function):
        """flashinfer gemma_fused_add_rmsnorm with analytic fp32 backward.

        Inputs are [N, H] bf16 contiguous; the kernel updates ``x`` in place
        (-> normed y) and ``residual`` in place (-> bf16(s)).  Forward clones
        the pre-kernel pair so backward can recover s = fp32(x)+fp32(res).
        """

        @staticmethod
        def forward(ctx, x, residual, weight, eps):
            ctx.save_for_backward(x.clone(), residual.clone(), weight)
            _fi_gemma_fused_add_rmsnorm(x, residual, weight, eps)
            ctx.eps = eps
            return x, residual

        @staticmethod
        def backward(ctx, dy, dres):
            x0, r0, weight = ctx.saved_tensors
            n = x0.shape[-1]
            s = x0.float() + r0.float()
            g = weight.float() + 1.0
            var = (s * s).sum(-1, keepdim=True) / n
            rstd = torch.rsqrt(var + ctx.eps)
            # y = s * rstd * g consumes s (via rstd); res_out = s is the
            # identity, so dres passes straight through with NO rstd
            # correction — folding dres into the correction term (with this
            # model's rstd ~ 30-50) was the backward explosion root cause.
            # Both errors (extra rstd in t, dres in t) verified against a
            # finite-difference oracle: see scratch/c6_backward_probe.py.
            dyf = dy.float()
            A = dyf * rstd * g
            t = (dyf * (s * g)).sum(-1, keepdim=True)
            ds = A - (rstd * rstd * rstd) * (s / n) * t + dres.float()
            dw = (dyf * (s * rstd)).sum(0)
            return ds.to(x0.dtype), ds.to(r0.dtype), dw.to(weight.dtype), None

    class _VllmGemmaRMSNorm(torch.autograd.Function):
        """flashinfer gemma_rmsnorm (no residual) with analytic fp32 backward."""

        @staticmethod
        def forward(ctx, x, weight, eps):
            y = _fi_gemma_rmsnorm(x, weight, eps)
            ctx.save_for_backward(x, weight)
            ctx.eps = eps
            return y

        @staticmethod
        def backward(ctx, dy):
            x, weight = ctx.saved_tensors
            n = x.shape[-1]
            xf, gf = x.float(), weight.float() + 1.0
            var = (xf * xf).sum(-1, keepdim=True) / n
            rstd = torch.rsqrt(var + ctx.eps)
            A = dy.float() * rstd * gf
            t = (dy.float() * (xf * gf)).sum(-1, keepdim=True)
            dx = A - (rstd * rstd * rstd) * (xf / n) * t
            dw = (dy.float() * (xf * rstd)).sum(0)
            return dx.to(x.dtype), dw.to(weight.dtype), None


def _vllm_pair_norm_available() -> bool:
    """Whether the flashinfer Gemma norm kernels (the contract path) are present."""
    return _HAS_FI_NORM


def _vllm_pair_norm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM input-norm contract on an UN-ADDED (x, residual) pair.

    Returns (y, residual_out): y is the bf16 normed activation (Gemma
    x_hat*(w+1) on the exact fp32 sum), residual_out is the updated stream
    bf16(fp32(x)+fp32(residual)).  With ``residual=None`` y = gemma_rmsnorm(x)
    and residual_out is x itself (vLLM starts the stream at the input).
    """
    shape = x.shape
    # clone: contiguous() may return the caller's own tensor and the
    # flashinfer kernel writes x/residual in place.
    x2 = x.reshape(-1, shape[-1]).contiguous().clone()
    w2 = weight.contiguous()
    if residual is None:
        if _HAS_FI_NORM:
            y = _VllmGemmaRMSNorm.apply(x2, w2, eps)
        else:  # pragma: no cover - fallback parity, not contract-exact
            xf = x2.float()
            y = (xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + eps) * (w2.float() + 1.0)).to(
                x2.dtype
            )
        return y.view(shape), x
    res2 = residual.reshape(-1, shape[-1]).contiguous().clone()
    if _HAS_FI_NORM:
        y, res_out = _VllmGemmaFusedAddRmsNorm.apply(x2, res2, w2, eps)
    else:  # pragma: no cover - fallback parity, not contract-exact
        s = x2.float() + res2.float()
        y = (s * torch.rsqrt((s * s).mean(-1, keepdim=True) + eps) * (w2.float() + 1.0)).to(
            x2.dtype
        )
        res_out = s.to(res2.dtype)
    return y.view(shape), res_out.view(shape)


# ------------------------------------------------------------------------------------


class _ReplicatedParameterWithGradReduce(torch.autograd.Function):
    """Keep a replicated TP parameter in sync by reducing its gradient."""

    @staticmethod
    def forward(ctx, param: torch.Tensor, group) -> torch.Tensor:
        ctx.group = group
        return param

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad = grad_output.contiguous()
        dist.all_reduce(grad, group=ctx.group)
        return grad, None


class GatedDeltaNet(nn.Module):
    """Native Gated DeltaNet with head-parallel / chunkwise CP support."""

    def __init__(
        self,
        *,
        hidden_size: int,
        linear_num_key_heads: int,
        linear_key_head_dim: int,
        linear_num_value_heads: int,
        linear_value_head_dim: int,
        linear_conv_kernel_dim: int,
        rms_norm_eps: float,
        ps: ParallelState,
        deterministic: bool = False,
        cp_mode: str = "headwise",
    ):
        super().__init__()
        if cp_mode not in _CP_MODES:
            raise ValueError(
                f"Unsupported GatedDeltaNet CP mode: {cp_mode!r}; expected one of {sorted(_CP_MODES)}."
            )
        self.ps = ps
        self.deterministic = bool(deterministic)
        self.cp_mode = cp_mode
        self.num_k_heads = linear_num_key_heads
        self.num_v_heads = linear_num_value_heads
        self.dk = linear_key_head_dim
        self.dv = linear_value_head_dim
        self.v_heads_per_k_head = ensure_divisible(self.num_v_heads, self.num_k_heads)
        self._replicate_heads = (
            self.num_k_heads < ps.tp_size or self.num_v_heads < ps.tp_size
        )
        if self._replicate_heads:
            if ps.cp_size > 1:
                raise NotImplementedError(
                    "GatedDeltaNet TP head replication is not yet validated with CP>1."
                )
            for name, num_heads in (
                ("linear_num_key_heads", self.num_k_heads),
                ("linear_num_value_heads", self.num_v_heads),
            ):
                if num_heads % ps.tp_size != 0 and ps.tp_size % num_heads != 0:
                    raise ValueError(
                        f"{name} ({num_heads}) must be a multiple or divisor of "
                        f"TP size ({ps.tp_size})."
                    )
            self.num_k_heads_local = self.num_k_heads
            self.num_v_heads_local = self.num_v_heads
        else:
            self.num_k_heads_local = ensure_divisible(self.num_k_heads, ps.tp_size)
            self.num_v_heads_local = ensure_divisible(self.num_v_heads, ps.tp_size)
        self.qk_dim = self.num_k_heads * self.dk
        self.v_dim = self.num_v_heads * self.dv
        self.qk_dim_local = self.num_k_heads_local * self.dk
        self.v_dim_local = self.num_v_heads_local * self.dv
        self.in_proj_dim = self.qk_dim * 2 + self.v_dim * 2 + self.num_v_heads * 2

        self.in_proj = ColumnParallelLinear(
            hidden_size,
            self.in_proj_dim,
            ps,
            bias=False,
            normalization="RMSNorm",
            eps=rms_norm_eps,
            zero_centered_gamma=True,
        )
        self._in_norm_eps = rms_norm_eps  # 0-diff: vLLM input-norm contract eps
        conv_dim_local = (
            ensure_divisible(self.qk_dim * 2 + self.v_dim, ps.tp_size)
            if self._replicate_heads
            else self.qk_dim_local * 2 + self.v_dim_local
        )
        self.conv_dim_local = conv_dim_local
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim_local,
            out_channels=conv_dim_local,
            kernel_size=linear_conv_kernel_dim,
            groups=conv_dim_local,
            bias=False,
            padding=linear_conv_kernel_dim - 1,
        )
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads, dtype=torch.float32)
            if self._replicate_heads
            else torch.ones(self.num_v_heads_local, dtype=torch.float32)
        )
        self.A_log = nn.Parameter(
            torch.zeros(self.num_v_heads, dtype=torch.float32)
            if self._replicate_heads
            else torch.zeros(self.num_v_heads_local, dtype=torch.float32)
        )
        self.norm = te.RMSNorm(self.dv, eps=rms_norm_eps, zero_centered_gamma=True)
        # 0-diff: raw bf16 gamma cache (vLLM loads the fp32 checkpoint value
        # into a bf16 param; bf16(gamma) is not recoverable from the
        # zero-centered param) — filled by the checkpoint post-load hook.
        self.register_buffer(
            "_vllm_norm_w", torch.zeros(self.dv, dtype=torch.bfloat16), persistent=False
        )
        self._vllm_norm_w_loaded = False
        self.o_proj = RowParallelLinear(self.v_dim, hidden_size, ps, bias=False)
        # (global cu_seqlens, FLA cp_context) keyed on (seq_len_global, batch, device)
        # for the non-packed chunkwise path, where both are static across forwards.
        self._chunkwise_cp_context_cache: dict[tuple, tuple[torch.Tensor, object]] = {}

    def sync_tp_replicated_parameters(self) -> None:
        """Broadcast replicated GDN state after from-scratch initialization.

        ``A_log`` and ``dt_bias`` are full copies on every TP rank when the
        number of GDN heads is smaller than TP.  Their gradients are reduced
        in :meth:`_state_parameters_for_tp`, so their initial values must also
        agree.  HF loading already provides identical state; this closes the
        random-initialization path used by correctness and scratch training.
        """
        if not self._replicate_heads or self.ps.tp_group is None or self.ps.tp_size <= 1:
            return
        if not dist.is_initialized() or dist.get_world_size(self.ps.tp_group) <= 1:
            return
        source_rank = dist.get_process_group_ranks(self.ps.tp_group)[0]
        with torch.no_grad():
            for parameter in (self.A_log, self.dt_bias):
                dist.broadcast(parameter.data, src=source_rank, group=self.ps.tp_group)

    # The six ``qkvzba`` sections and the three conv channel sections, in hidden order.
    def _qkvzba_sections(self) -> list[int]:
        return [
            self.qk_dim_local,
            self.qk_dim_local,
            self.v_dim_local,
            self.v_dim_local,
            self.num_v_heads_local,
            self.num_v_heads_local,
        ]

    def _conv_sections(self) -> list[int]:
        return [self.qk_dim_local, self.qk_dim_local, self.v_dim_local]

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(o_proj output, residual stream)`` — both UN-ADDED.

        ``residual`` (when given) is the un-added running stream: the vLLM
        input-norm contract consumes the exact fp32 (x + residual) inside the
        norm, so the caller must not pre-add the pair.
        """
        del position_ids
        is_packed = packed_seq_params is not None
        # 0-diff (qwen35-0diff-lite): vLLM input-norm contract. flashinfer's
        # Gemma fused add+RMSNorm on the UN-ADDED pair plus a plain F.linear
        # gemm — both bitwise-verified against the in-engine capture (the
        # gemm equals vLLM's in_proj_qkvz/in_proj_ba pair). te.LayerNormLinear
        # instead norms the bf16 pre-added stream (exact fp32 sum lost at the
        # add) with TE's own norm arithmetic: 51% of in_proj outputs diverged
        # on captured data. TP>1 needs the deferred-all-reduce norm variant
        # and a column-sharded gemm — not contract-covered yet.
        if (
            _HAS_FI_NORM
            and self.ps.tp_size == 1
            and not self._replicate_heads
            and not _NATIVE_MODE
        ):
            y, residual = _vllm_pair_norm(
                x, residual, self.in_proj.linear.layer_norm_weight, self._in_norm_eps
            )
            qkvzba = (
                F.linear(y.reshape(-1, y.shape[-1]), self.in_proj.linear.weight)
                .view(x.shape[0], x.shape[1], -1)
                .transpose(0, 1)
                .contiguous()
            )
        else:
            if residual is not None:
                x = residual + x
            residual = x
            qkvzba = self.in_proj(x).transpose(0, 1).contiguous()
        if self._replicate_heads:
            qkvzba = all_gather_last_dim_with_grad_reduce(qkvzba, self.ps.tp_group)
        cu_seqlens = self._packed_cu_seqlens(packed_seq_params) if is_packed else None

        headwise = False
        chunkwise = False
        cp_context = None
        reshuffle_cu = None
        cp_div = 1
        if self.ps.cp_size > 1:
            if self.ps.cp_group is None:
                raise RuntimeError("CP>1 requires ParallelState.cp_group.")
            if self.cp_mode == "chunkwise":
                if not is_packed and qkvzba.shape[0] > 1:
                    raise ValueError(
                        "GatedDeltaNet chunkwise CP with SBHD inputs currently requires "
                        "micro_batch_size == 1. Use packed THD input or micro_batch_size=1."
                    )
                cp_context, cu_seqlens = self._build_chunkwise_cp_context(
                    qkvzba, cu_seqlens, is_packed
                )
                # Packed THD reshuffles with the global cu_seqlens (packing-aware); the
                # non-packed path is a plain two-chunk swap (``cu_seqlens is None``).
                reshuffle_cu = cu_seqlens if is_packed else None
                qkvzba = self._chunkwise_reshuffle(qkvzba, reshuffle_cu, to_contiguous=True)
                chunkwise = True
            else:  # headwise
                if not is_packed and qkvzba.shape[0] > 1:
                    raise ValueError(
                        "GatedDeltaNet headwise CP with SBHD inputs currently requires "
                        "micro_batch_size == 1. Use packed THD input or micro_batch_size=1."
                    )
                qkvzba, cu_seqlens = self._headwise_cp2hp(qkvzba, cu_seqlens)
                headwise = True
                cp_div = self.ps.cp_size

        batch, seq_len = qkvzba.shape[:2]

        # Per-rank parameter slices (identity unless headwise).
        if headwise:
            conv_weight = get_parameter_local_cp_headwise(
                self.conv1d.weight,
                dim=0,
                cp_size=self.ps.cp_size,
                cp_rank=self.ps.cp_rank,
                split_sections=self._conv_sections(),
            )
            A_log = get_parameter_local_cp_headwise(
                self.A_log, dim=0, cp_size=self.ps.cp_size, cp_rank=self.ps.cp_rank
            )
            dt_bias = get_parameter_local_cp_headwise(
                self.dt_bias, dim=0, cp_size=self.ps.cp_size, cp_rank=self.ps.cp_rank
            )
        else:
            conv_weight, A_log, dt_bias = None, self.A_log, self.dt_bias

        query, key, value, gate, beta, alpha = self._split_proj(qkvzba, cp_div)
        qkv = torch.cat(
            [
                query.reshape(batch, seq_len, -1),
                key.reshape(batch, seq_len, -1),
                value.reshape(batch, seq_len, -1),
            ],
            dim=-1,
        )
        if self._replicate_heads:
            conv_width = ensure_divisible(qkv.shape[-1], self.ps.tp_size)
            qkv = qkv.narrow(-1, self.ps.tp_rank * conv_width, conv_width).contiguous()

        qkv = self._causal_conv1d(
            qkv,
            seq_len,
            cu_seqlens=cu_seqlens,
            conv_weight=conv_weight,
            cp_div=cp_div,
            cp_context=cp_context,
        )
        if self._replicate_heads:
            qkv = all_gather_last_dim_with_grad_reduce(qkv, self.ps.tp_group)
        query, key, value, gate, beta, alpha = self._prepare_qkv(
            qkv, gate, beta, alpha, batch, seq_len, cp_div
        )
        if self._replicate_heads:
            A_log, dt_bias = self._state_parameters_for_tp()
        # 0-diff (qwen35-0diff-lite): vLLM gating contract (fp32 branch softplus,
        # fp32 exp(A_log), tl.sigmoid) — see _vllm_g_beta above.
        if (
            _HAS_TRITON
            and alpha.is_contiguous()
            and beta.is_contiguous()
            and A_log.is_contiguous()
            and dt_bias.is_contiguous()
        ):
            g, beta = _vllm_g_beta(alpha, beta, A_log, dt_bias)
        else:
            g, beta = self._compute_g_and_beta(A_log, dt_bias, alpha, beta)
        out, _ = self._gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
        )

        # 0-diff (qwen35-0diff-lite): vLLM fused gated-norm contract (fp32
        # through the gate multiply, single bf16 rounding, raw bf16 gamma) —
        # see _vllm_gated_norm above.
        if (
            _HAS_TRITON
            and out.dtype == torch.bfloat16
            and gate.dtype == torch.bfloat16
            and out.shape[-1] == self.dv
            and not _NATIVE_MODE
        ):
            if self._vllm_norm_w_loaded:
                w_raw = self._vllm_norm_w
            else:
                # fallback for weights not loaded via the 0-diff hook (unit
                # tests / scratch init): approximate bf16(gamma) from the
                # zero-centered param — NOT bit-exact.
                w_raw = (self.norm.weight.detach().float() + 1.0).to(torch.bfloat16)
            out = _vllm_gated_norm(
                out, gate, self.norm.weight, w_raw, getattr(self.norm, "eps", 1e-6)
            )
        else:
            out = self._apply_gated_norm(out, gate)
        out = out.reshape(
            batch,
            seq_len,
            self.v_dim if self._replicate_heads else self.v_dim_local // cp_div,
        )
        if self._replicate_heads:
            out_width = ensure_divisible(self.v_dim, self.ps.tp_size)
            out = out.narrow(-1, self.ps.tp_rank * out_width, out_width).contiguous()
        if self.ps.cp_size > 1:
            if chunkwise:
                # Inverse of the pre-recurrence reshuffle: restore the Megatron
                # attention-load-balanced zigzag layout downstream layers expect.
                out = self._chunkwise_reshuffle(out, reshuffle_cu, to_contiguous=False)
            else:  # headwise
                out = self._headwise_hp2cp(out, cu_seqlens)
            batch, seq_len = out.shape[:2]
        out = out.transpose(0, 1).contiguous()
        return self.o_proj(out), residual

    def _state_parameters_for_tp(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return replicated GDN state with a TP gradient all-reduce."""
        A_log = _ReplicatedParameterWithGradReduce.apply(self.A_log, self.ps.tp_group)
        dt_bias = _ReplicatedParameterWithGradReduce.apply(self.dt_bias, self.ps.tp_group)
        return A_log, dt_bias

    # ------------------------------------------------------------------ CP: headwise
    def _headwise_cp2hp(
        self,
        qkvzba: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Zigzag-sharded ``[b, s_local, h]`` -> contiguous full-seq ``[b, s_global, h/cp]``.

        Permutes the hidden dim section-wise so a plain hidden-scatter all-to-all leaves
        each rank with ``1/cp`` heads of every section, then reassembles the full
        sequence from the per-rank shards with the verified zigzag reconstruction.
        """
        cp_size = self.ps.cp_size
        perm = build_headwise_section_perm(
            self._qkvzba_sections(), cp_size, qkvzba.device
        )
        qkvzba = qkvzba.index_select(-1, perm)
        h = qkvzba.shape[-1]
        hpc = h // cp_size
        send_parts = [qkvzba[..., k * hpc : (k + 1) * hpc].contiguous() for k in range(cp_size)]
        recv = all_to_all_hidden_shards(send_parts, self.ps.cp_group)
        if cu_seqlens is None:
            full = zigzag_reconstruct_from_cp_parts(recv, seq_dim=1)
            return full.contiguous(), None
        if qkvzba.shape[0] != 1:
            raise ValueError("Packed THD GatedDeltaNet expects a single packed batch row.")
        parts = [p[0].contiguous() for p in recv]
        full = reconstruct_packed_from_cp_parts(
            parts,
            cu_seqlens_padded=cu_seqlens,
            cp_size=cp_size,
            dim=0,
        )
        return full.unsqueeze(0).contiguous(), cu_seqlens

    def _headwise_hp2cp(
        self,
        out: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        """Inverse of :meth:`_headwise_cp2hp` for the value output.

        Full-seq ``[b, s_global, v/cp]`` (this rank owns value-head shard ``cp_rank``) ->
        zigzag-sharded ``[b, s_local, v]`` with all heads. The value section's per-rank
        blocks are contiguous, so concatenating the received hidden shards in rank order
        restores the natural head order (no inverse permutation required).
        """
        cp_size = self.ps.cp_size
        if cu_seqlens is None:
            send_parts = [
                zigzag_slice_for_cp(out, j, cp_size, seq_dim=1).contiguous()
                for j in range(cp_size)
            ]
            recv = all_to_all_hidden_shards(send_parts, self.ps.cp_group)
            return torch.cat(recv, dim=-1).contiguous()
        if out.shape[0] != 1:
            raise ValueError("Packed THD GatedDeltaNet expects a single packed batch row.")
        base = out[0].contiguous()
        send_parts = [
            split_packed_to_cp_local(
                base,
                cu_seqlens_padded=cu_seqlens,
                cp_size=cp_size,
                cp_rank=j,
                dim=0,
            ).contiguous()
            for j in range(cp_size)
        ]
        recv = all_to_all_hidden_shards(send_parts, self.ps.cp_group)
        return torch.cat(recv, dim=-1).unsqueeze(0).contiguous()

    # ------------------------------------------------------------------ CP: chunkwise
    def _resolve_cu_seqlens(
        self, cu_seqlens: torch.Tensor, total_seq_len: int
    ) -> torch.Tensor:
        """Validate the global (padded) packed cu_seqlens for chunkwise CP.

        Mirrors upstream Megatron ``GatedDeltaNet._resolve_cu_seqlens``: the padded
        cumulative lengths must cover the whole global sequence and every per-sequence
        length must be divisible by ``cp_size`` (chunk boundaries land on CP shards).
        """
        total_cu = int(cu_seqlens[-1].item())
        if total_cu != total_seq_len:
            raise ValueError(
                f"GDN chunkwise: cu_seqlens[-1]={total_cu} does not match "
                f"total_sequence_length={total_seq_len} (global, padding-inclusive)."
            )
        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if (seq_lengths % self.ps.cp_size != 0).any():
            raise ValueError(
                "GDN chunkwise: all per-sequence cu_seqlens lengths must be divisible by "
                f"cp_size={self.ps.cp_size}, but got lengths {seq_lengths.tolist()}."
            )
        return cu_seqlens

    def _build_chunkwise_cp_context(
        self,
        qkvzba: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        is_packed: bool,
    ) -> tuple[object, torch.Tensor]:
        """Build the FLA ring ``cp_context`` and resolve the global cu_seqlens.

        For packed THD the (padded) global cu_seqlens is validated; for the non-packed
        dense case the only cu_seqlens source is the static global sequence length and
        batch size, so both the cu_seqlens and the (allocation-heavy) cp_context are
        cached to keep them stable across forwards.
        """
        if not _HAS_FLA or _fla_build_cp_context is None:
            raise NotImplementedError(
                "GatedDeltaNet chunkwise CP requires the FLA cp_context ring kernel."
            )
        batch, seq_len_local = qkvzba.shape[:2]
        seq_len_global = seq_len_local * self.ps.cp_size
        conv_kernel = self.conv1d.kernel_size[0]
        if is_packed:
            cu = self._resolve_cu_seqlens(cu_seqlens, seq_len_global)
            cp_context = _fla_build_cp_context(
                cu_seqlens=cu,
                group=self.ps.cp_group,
                conv1d_kernel_size=conv_kernel,
            )
            return cp_context, cu

        cache_key = (seq_len_global, batch, qkvzba.device)
        cached = self._chunkwise_cp_context_cache.get(cache_key)
        if cached is None:
            dense_cu = (
                torch.arange(batch + 1, device=qkvzba.device, dtype=torch.long)
                * seq_len_global
            )
            cp_context = _fla_build_cp_context(
                cu_seqlens=dense_cu,
                group=self.ps.cp_group,
                conv1d_kernel_size=conv_kernel,
            )
            cached = (dense_cu, cp_context)
            self._chunkwise_cp_context_cache[cache_key] = cached
        dense_cu, cp_context = cached
        return cp_context, dense_cu

    def _chunkwise_reshuffle(
        self,
        tensor: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        *,
        to_contiguous: bool,
    ) -> torch.Tensor:
        """Reshuffle CP chunks between Megatron zigzag and contiguous-time layout.

        ``cu_seqlens`` (the global packed layout) selects the packing-aware THD swap;
        ``None`` selects the plain two-chunk SBHD swap. Shape is preserved either way.
        """
        swap = zigzag_to_contiguous_chunks if to_contiguous else contiguous_to_zigzag_chunks
        return swap(tensor, self.ps.cp_group, seq_dim=1, cu_seqlens=cu_seqlens)

    # ------------------------------------------------------------------ compute
    def _causal_conv1d(
        self,
        qkv: torch.Tensor,
        seq_len: int,
        *,
        cu_seqlens: torch.Tensor | None,
        conv_weight: torch.Tensor | None,
        cp_div: int,
        cp_context=None,
    ) -> torch.Tensor:
        # ``conv_weight`` is the (headwise) per-rank slice; None means use the full module.
        weight = self.conv1d.weight if conv_weight is None else conv_weight
        groups = self.conv_dim_local // cp_div
        if _HAS_TRITON and cp_context is None:
            # 0-diff (qwen35-0diff-lite): the plain (non-CP) conv runs the vLLM
            # contract replication — the fla kernel's fp32 products diverged by
            # ~7e-5 absmean / 0.125 absmax from the rollout side on identical
            # inputs.  Verified bitwise-identical to vLLM's Triton kernel on
            # the C3 in-engine captures (4 layers x 336 tokens).
            return _vllm_conv1d_silu(qkv, weight.squeeze(1), cu_seqlens)
        if _HAS_FLA and (cp_context is not None or cu_seqlens is not None or not self.deterministic):
            orig_seq_len = qkv.shape[1]
            # Chunkwise CP must not pad conv inputs: padding chunk-local causal-conv
            # inputs would change later chunk numerics across the ring.
            pad_n = 0 if cp_context is not None else (-orig_seq_len % _CONV_PAD_ALIGNMENT)
            conv_input = qkv
            conv_cu_seqlens = cu_seqlens
            if pad_n > 0:
                conv_input = F.pad(qkv, (0, 0, 0, pad_n))
                if conv_cu_seqlens is not None:
                    conv_cu_seqlens = conv_cu_seqlens.clone()
                    conv_cu_seqlens[-1] += pad_n
            kwargs = {}
            if cp_context is not None:
                kwargs["cp_context"] = cp_context
            qkv, _ = _fla_causal_conv1d(
                x=conv_input,
                weight=weight.squeeze(1),
                bias=None,
                activation="silu",
                cu_seqlens=conv_cu_seqlens,
                **kwargs,
            )
            if pad_n > 0:
                qkv = qkv[:, :orig_seq_len, :]
            return qkv
        if cu_seqlens is None and cp_context is None:
            conv_out = F.conv1d(
                qkv.transpose(1, 2),
                weight=weight,
                bias=None,
                padding=self.conv1d.padding[0],
                groups=groups,
            )
            return F.silu(conv_out[:, :, :seq_len].transpose(1, 2))
        raise NotImplementedError("GatedDeltaNet packed THD requires FLA causal conv.")

    def _gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None,
        cp_context=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if _HAS_FLA and (cp_context is not None or cu_seqlens is not None or not self.deterministic):
            kwargs = {}
            if cp_context is not None:
                kwargs["cp_context"] = cp_context
            return _fla_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=False,
                cu_seqlens=cu_seqlens,
                **kwargs,
            )
        if cp_context is not None:
            raise NotImplementedError(
                "GatedDeltaNet chunkwise CP requires the FLA gated delta rule kernel."
            )
        return torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )

    @staticmethod
    def _packed_cu_seqlens(packed_seq_params) -> torch.Tensor:
        cu_seqlens = (
            packed_seq_params.cu_seqlens_q_padded
            if getattr(packed_seq_params, "cu_seqlens_q_padded", None) is not None
            else packed_seq_params.cu_seqlens_q
        )
        if cu_seqlens is None:
            raise ValueError(
                "packed_seq_params must carry cu_seqlens_q for CP GatedDeltaNet."
            )
        return cu_seqlens

    def _split_proj(self, qkvzba: torch.Tensor, cp_div: int):
        qk = self.qk_dim_local // cp_div
        vd = self.v_dim_local // cp_div
        nvh = self.num_v_heads_local // cp_div
        nkh = self.num_k_heads_local // cp_div
        q, k, v, z, b, a = qkvzba.split([qk, qk, vd, vd, nvh, nvh], dim=-1)
        batch, seq_len = qkvzba.shape[:2]
        return (
            q.reshape(batch, seq_len, nkh, self.dk),
            k.reshape(batch, seq_len, nkh, self.dk),
            v.reshape(batch, seq_len, nvh, self.dv),
            z.reshape(batch, seq_len, nvh, self.dv),
            b.reshape(batch, seq_len, nvh),
            a.reshape(batch, seq_len, nvh),
        )

    @jit_fuser
    def _prepare_qkv(
        self, qkv: torch.Tensor, gate, beta, alpha, batch: int, seq_len: int, cp_div: int
    ):
        qk = self.qk_dim_local // cp_div
        nkh = self.num_k_heads_local // cp_div
        query_key, value = qkv.split([2 * qk, self.v_dim_local // cp_div], dim=-1)
        query_key = query_key.reshape(batch, seq_len, 2 * nkh, self.dk)
        value = value.reshape(batch, seq_len, self.num_v_heads_local // cp_div, self.dv)
        query_key = self._l2norm(query_key.contiguous())
        query, key = query_key.split(nkh, dim=2)
        if self.v_heads_per_k_head > 1:
            query = query.repeat_interleave(self.v_heads_per_k_head, dim=2)
            key = key.repeat_interleave(self.v_heads_per_k_head, dim=2)
        return (
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            gate.contiguous(),
            beta.contiguous(),
            alpha.contiguous(),
        )

    @staticmethod
    @jit_fuser
    def _l2norm(x: torch.Tensor) -> torch.Tensor:
        return l2norm(x)

    @staticmethod
    @jit_fuser
    def _compute_g_and_beta(A_log, dt_bias, alpha, beta):
        g = -A_log.exp() * F.softplus(alpha.float() + dt_bias)
        # 0-diff (qwen35-0diff-lite): beta in fp32 to match the inference-side
        # contract (vLLM fused_post_conv_prep computes sigmoid in fp32 and feeds
        # fp32 beta into the chunk kernel; bf16 beta diverged by ~2e-3/element).
        return g, beta.float().sigmoid()

    @jit_fuser
    def _apply_gated_norm(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.reshape(-1, x.shape[-1])
        y = self.norm(x)
        gate = gate.reshape(-1, gate.shape[-1])
        y = y * F.silu(gate.float())
        return y.to(x_dtype)


__all__ = ["GatedDeltaNet"]
