# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoE expert compute: SwiGLU fusions, _AllReduceETP, and Experts."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive import transformer_engine as te
from megatron.lite.primitive.kernels.swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl
from megatron.lite.primitive.modules.lora import (
    LoraConfig,
    SharedGroupedLinearLoRA,
    normalize_lora_config,
)
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.recompute import CheckpointWithoutOutput
from megatron.lite.primitive.utils import ensure_divisible

__all__ = ["Experts", "_AllReduceETP"]


# 0-diff (qwen35-0diff-lite): vLLM Triton fused_moe contract for routed experts
# --------------------------------------------------------------------------------
#
# The rollout side (vLLM, TRITON unquantized MoE backend) computes the routed
# experts as: moe_align_block_size -> Triton GEMM1 (w13) -> silu_and_mul ->
# Triton GEMM2 (w2, router weights applied in-kernel) -> moe_sum over the k
# slots.  Driving vLLM's own ``fused_experts`` entry point with the mlite
# expert weights (stacked [E, 2N, K] / [E, K, N], identical layout) and the
# bit-exact router outputs reproduces the engine's routed output bitwise
# (verified 0 mismatch x 4 layers on the C3 in-engine captures).  Importing
# the vLLM function keeps the Triton binaries identical to the engine's —
# no transcription risk.
#
# ``fused_experts`` is inference-only, so gradients go through a
# mathematically-exact torch recompute (not bitwise; vLLM defines no backward
# reference at all) under :class:`_VllmFusedMoe`.
#
# Validity window: bf16, TP=EP=ETP=1, unquantized.  Other configurations fall
# back to the TE grouped-GEMM path.

_VLLM_FUSED_MOE_IMPORT_ERROR: str | None = ""


def _vllm_fused_experts_fn():
    """Lazily import vLLM's fused_experts; None when unavailable."""
    global _VLLM_FUSED_MOE_IMPORT_ERROR
    if _VLLM_FUSED_MOE_IMPORT_ERROR != "":
        return None
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

        return fused_experts
    except Exception as e:  # pragma: no cover - env-dependent
        _VLLM_FUSED_MOE_IMPORT_ERROR = f"{type(e).__name__}: {e}"
        return None


def stack_expert_weights(grouped: "te.GroupedLinear") -> torch.Tensor:
    """Stack the per-expert ``weight{i}`` params into ``[E, out, in]``."""
    num_experts = getattr(grouped, "num_gemms")
    return torch.stack([getattr(grouped, f"weight{i}") for i in range(num_experts)])


def _torch_grouped_moe(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
) -> torch.Tensor:
    """Differentiable torch recompute of the fused_experts math.

    Used only by the backward: segments the (token, slot) rows by expert and
    runs one small matmul per expert; gradients flow through plain torch ops.
    """
    num_tokens, hidden = x.shape
    k = ids.shape[1]
    num_experts = w13.shape[0]
    flat_ids = ids.view(-1).long()
    xs = x.repeat_interleave(k, dim=0)
    order = torch.argsort(flat_ids, stable=True)
    sorted_ids = flat_ids[order]
    xs_s = xs[order]
    wts_s = weights.view(-1).float()[order]
    counts = torch.bincount(sorted_ids, minlength=num_experts)
    ends = torch.cumsum(counts, 0)
    out_sorted = torch.zeros(
        num_tokens * k, hidden, dtype=torch.float32, device=x.device
    )
    for e in range(num_experts):
        end = int(ends[e])
        beg = int(ends[e - 1]) if e else 0
        if beg == end:
            continue
        c1 = torch.nn.functional.linear(xs_s[beg:end], w13[e])
        gate, up = c1.chunk(2, dim=-1)
        c2 = torch.nn.functional.silu(gate) * up
        c3 = torch.nn.functional.linear(c2, w2[e])
        out_sorted[beg:end] = c3.float() * wts_s[beg:end, None]
    out_orig = torch.zeros_like(out_sorted)
    out_orig[order] = out_sorted
    return (out_orig.view(num_tokens, k, hidden).sum(1)).to(x.dtype)


class _VllmFusedMoe(torch.autograd.Function):
    """Routed experts: bitwise vLLM forward (fused_experts), exact torch backward."""

    @staticmethod
    def forward(ctx, x, w13, w2, weights, ids):
        fused_experts = _vllm_fused_experts_fn()
        assert fused_experts is not None
        out = fused_experts(
            x,
            w13,
            w2,
            weights.to(torch.float32).contiguous(),
            ids.to(torch.int32).contiguous(),
        )
        ctx.save_for_backward(x, w13, w2, weights, ids)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, w13, w2, weights, ids = ctx.saved_tensors
        with torch.enable_grad():
            x_ = x.detach().requires_grad_(True)
            w13_ = w13.detach().requires_grad_(True)
            w2_ = w2.detach().requires_grad_(True)
            wts_ = weights.detach().float().requires_grad_(True)
            out_ = _torch_grouped_moe(x_, w13_, w2_, wts_, ids)
            grads = torch.autograd.grad(out_, (x_, w13_, w2_, wts_), dout)
        return grads[0], grads[1], grads[2], grads[3].to(weights.dtype), None


def vllm_fused_moe(
    x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor
) -> torch.Tensor:
    """Bit-exact vLLM Triton fused_moe for bf16 [T, K] inputs."""
    return _VllmFusedMoe.apply(x, w13, w2, weights, ids)


def vllm_fused_moe_available() -> bool:
    return _vllm_fused_experts_fn() is not None


# 0-diff (qwen35-0diff-epcp): EP>1 fused-MoE contract --------------------------
#
# At EP>1 every rank sees the full (replicated) token stream and routing and
# owns a contiguous expert shard [ep_rank*n_local, (ep_rank+1)*n_local) (the
# checkpoint loader's placement).  Each rank drives vLLM's fused_experts on ITS
# shard with the non-local slots remapped to local expert 0 with weight 0
# (those rows come out exactly zero); the per-slot PRE-SUM intermediate (the
# ops.moe_sum input, [t, k, h]) is all-reduced — every element is non-zero on
# exactly one rank and IEEE x+0 is exact, so the combine is bitwise — and the
# final topk-slot-order sum runs locally through the same vLLM moe_sum kernel
# the EP1 contract uses.  Per-slot bitwise equivalence of the sharded call vs
# the full-experts call (Triton config/tiling invariance) is empirically
# verified: outputs/.../c10mesh/cache3_probe.json (C10_CACHE3_PASS).
#
# The pre-sum intermediate is obtained by capturing the ops.moe_sum input
# around the fused_experts call (the same public entry point as EP1 — no
# transcription).  Backward mirrors _VllmFusedMoe: a masked
# _torch_grouped_moe surrogate per rank; x / routing-weight grads are
# per-shard contributions and are all-reduced (summed) across EP, expert grads
# stay local (FSDP owns the shard params).

_moe_sum_orig = None
_moe_sum_captured: list[torch.Tensor] = []


@contextmanager
def _capture_pre_moe_sum():
    """Capture fused_experts' pre-sum intermediate (the ops.moe_sum input).

    The stub skips vLLM's in-call moe_sum (its summed result is discarded —
    the contract runs the real kernel itself, after the EP combine).
    """
    global _moe_sum_orig
    import vllm._custom_ops as vops

    if _moe_sum_orig is None:
        _moe_sum_orig = vops.moe_sum
    del _moe_sum_captured[:]

    def _stub(inp, out):
        _moe_sum_captured.append(inp.detach().clone())

    vops.moe_sum = _stub
    try:
        yield _moe_sum_captured
    finally:
        vops.moe_sum = _moe_sum_orig


def _vllm_moe_sum(inp: torch.Tensor, out: torch.Tensor) -> None:
    import vllm._custom_ops as vops

    vops.moe_sum(inp, out)


class _VllmFusedMoeEP(torch.autograd.Function):
    """Routed experts at EP>1: bitwise vLLM forward, exact torch backward."""

    @staticmethod
    def forward(ctx, x, w13, w2, weights, ids, ep_group, ep_rank, n_local):
        fused_experts = _vllm_fused_experts_fn()
        assert fused_experts is not None
        mask = (ids // n_local) == ep_rank
        ids_r = torch.where(mask, ids - ep_rank * n_local, torch.zeros_like(ids))
        w_r = weights.to(torch.float32) * mask.to(torch.float32)
        with _capture_pre_moe_sum() as captured:
            fused_experts(x, w13, w2, w_r.contiguous(),
                          ids_r.to(torch.int32).contiguous())
        cache3 = captured[0]
        # EP combine: exactly one rank contributes a non-zero per-slot value.
        dist.all_reduce(cache3, group=ep_group)
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        _vllm_moe_sum(cache3.contiguous(), out)
        ctx.save_for_backward(x, w13, w2, weights, ids)
        ctx.ep_group = ep_group
        ctx.ep_rank = ep_rank
        ctx.n_local = n_local
        return out

    @staticmethod
    def backward(ctx, dout):
        x, w13, w2, weights, ids = ctx.saved_tensors
        mask = (ids // ctx.n_local) == ctx.ep_rank
        ids_r = torch.where(mask, ids - ctx.ep_rank * ctx.n_local,
                            torch.zeros_like(ids))
        with torch.enable_grad():
            x_ = x.detach().requires_grad_(True)
            w13_ = w13.detach().requires_grad_(True)
            w2_ = w2.detach().requires_grad_(True)
            wts_ = (weights.detach().float() * mask.to(torch.float32)).requires_grad_(True)
            out_ = _torch_grouped_moe(x_, w13_, w2_, wts_, ids_r)
            grads = torch.autograd.grad(out_, (x_, w13_, w2_, wts_), dout)
        # x / routing-weight grads are this shard's contribution only: sum
        # them across EP (the surrounding network is replicated, so dout —
        # and every other gradient path — is identical on all ranks).
        gx = grads[0].contiguous()
        gw = grads[3].contiguous()
        dist.all_reduce(gx, group=ctx.ep_group)
        dist.all_reduce(gw, group=ctx.ep_group)
        return gx, grads[1], grads[2], gw.to(weights.dtype), None, None, None, None


def vllm_fused_moe_ep(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    ep_group,
    ep_rank: int,
    n_local: int,
) -> torch.Tensor:
    """Bit-exact vLLM Triton fused_moe for bf16 [T, K] inputs at EP>1."""
    return _VllmFusedMoeEP.apply(x, w13, w2, weights, ids, ep_group, ep_rank, n_local)


@contextmanager
def _expert_nvtx_range(name: str):
    if os.environ.get("MEGATRON_LITE_EP_EXPERT_NVTX") != "1" or not torch.cuda.is_available():
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def swiglu_with_probs(
    y: torch.Tensor, probs: torch.Tensor | None, swiglu_limit: float = 0.0
) -> torch.Tensor:
    """SwiGLU with optional expert probability scaling."""
    if swiglu_limit > 0:
        gate, up = y.chunk(2, dim=-1)
        up = torch.clamp(up.float(), min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate.float(), max=swiglu_limit)
        out = torch.nn.functional.silu(gate) * up
        if probs is not None:
            out = out * probs
        return out.to(dtype=y.dtype)
    if probs is not None:
        return weighted_bias_swiglu_impl(y, bias=None, weights=probs)
    return bias_swiglu_impl(y, bias=None)


class _AllReduceETP(torch.autograd.Function):
    """AllReduce with proper autograd: grad(AllReduce) = AllReduce."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class Experts(nn.Module):

    def __init__(
        self,
        config: Any,
        ps: ParallelState,
        *,
        fp8: bool = False,
        moe_act_recompute: bool = False,
        lora_config: LoraConfig | dict | None = None,
    ):
        super().__init__()
        self.num_local_experts = ensure_divisible(config.num_experts, ps.ep_size)
        self.fp8 = fp8
        self.moe_act_recompute = moe_act_recompute
        self.etp_group = ps.etp_group if ps.etp_size > 1 else None
        self.swiglu_limit = float(getattr(config, "swiglu_limit", 0.0) or 0.0)
        self.fc1 = te.GroupedLinear(
            self.num_local_experts,
            config.hidden_size,
            config.moe_intermediate_size * 2 // ps.etp_size,
            bias=False,
            params_dtype=torch.bfloat16,
        )
        self.fc2 = te.GroupedLinear(
            self.num_local_experts,
            config.moe_intermediate_size // ps.etp_size,
            config.hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,
        )
        lora = normalize_lora_config(lora_config)
        self.fc1_lora: SharedGroupedLinearLoRA | None = None
        self.fc2_lora: SharedGroupedLinearLoRA | None = None
        if lora.enabled and lora.targets_module("linear_fc1"):
            self.fc1_lora = SharedGroupedLinearLoRA(
                self.num_local_experts,
                config.hidden_size,
                config.moe_intermediate_size * 2 // ps.etp_size,
                lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
            )
        if lora.enabled and lora.targets_module("linear_fc2"):
            self.fc2_lora = SharedGroupedLinearLoRA(
                self.num_local_experts,
                config.moe_intermediate_size // ps.etp_size,
                config.hidden_size,
                lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
            )
        if ps.tp_size > 1 and ps.ep_size == 1 and ps.etp_size == 1:
            tp_group = ps.tp_group
            for module in (self.fc1, self.fc2, self.fc1_lora, self.fc2_lora):
                if module is None:
                    continue
                for param in module.parameters():

                    def _ar(grad, g=tp_group):
                        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=g)
                        return grad

                    param.register_hook(_ar)

    def forward(
        self,
        x: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor | None = None,
        tokens_per_expert_list: list[int] | None = None,
    ) -> torch.Tensor:
        m_splits = (
            tokens_per_expert.tolist()
            if tokens_per_expert_list is None
            else list(tokens_per_expert_list)
        )
        pad_mask = None
        if self.fp8:
            x, permuted_probs, m_splits, pad_mask = self._fp8_pad(x, permuted_probs, m_splits)

        etp_real_len = x.shape[0]
        if self.etp_group is not None:
            max_len = torch.tensor([etp_real_len], device=x.device, dtype=torch.int64)
            dist.all_reduce(max_len, op=dist.ReduceOp.MAX, group=self.etp_group)
            max_len = int(max_len.item())
            if etp_real_len < max_len:
                x = torch.cat(
                    [
                        x,
                        torch.zeros(
                            max_len - etp_real_len, x.shape[1], dtype=x.dtype, device=x.device
                        ),
                    ],
                    dim=0,
                )
                if permuted_probs is not None:
                    permuted_probs = torch.cat(
                        [
                            permuted_probs,
                            torch.zeros(
                                max_len - etp_real_len, dtype=permuted_probs.dtype, device=x.device
                            ),
                        ],
                        dim=0,
                    )
                m_splits = list(m_splits)
                m_splits[-1] += max_len - etp_real_len

        probs = permuted_probs.unsqueeze(-1) if permuted_probs is not None else None
        with _expert_nvtx_range("ep_experts.forward"):
            if self.moe_act_recompute and probs is not None:
                act_ckpt = CheckpointWithoutOutput(preserve_rng_state=True)
                fc1_out = self.fc1(x, m_splits)
                if self.fc1_lora is not None:
                    fc1_out = fc1_out + self.fc1_lora(x, m_splits)
                h = act_ckpt.checkpoint(swiglu_with_probs, fc1_out, probs, self.swiglu_limit)
                out = self.fc2(h, m_splits)
                if self.fc2_lora is not None:
                    out = out + self.fc2_lora(h, m_splits)
                act_ckpt.discard_output_and_register_recompute(out)
            else:
                fc1_out = self.fc1(x, m_splits)
                if self.fc1_lora is not None:
                    fc1_out = fc1_out + self.fc1_lora(x, m_splits)
                h = swiglu_with_probs(fc1_out, probs, self.swiglu_limit)
                out = self.fc2(h, m_splits)
                if self.fc2_lora is not None:
                    out = out + self.fc2_lora(h, m_splits)

        if self.etp_group is not None:
            out = _AllReduceETP.apply(out, self.etp_group)
            out = out[:etp_real_len]

        if pad_mask is not None:
            out = out[pad_mask]
        return out

    @staticmethod
    def _fp8_pad(x, permuted_probs, m_splits):
        padded = [(s + 15) // 16 * 16 for s in m_splits]
        if padded == m_splits:
            return x, permuted_probs, m_splits, None
        device, dtype = x.device, x.dtype
        total_padded = sum(padded)
        x_pad = torch.zeros(total_padded, x.size(1), device=device, dtype=dtype)
        mask = torch.zeros(total_padded, dtype=torch.bool, device=device)
        probs_pad = None
        if permuted_probs is not None:
            probs_pad = torch.zeros(total_padded, device=device, dtype=permuted_probs.dtype)
        src_off, dst_off = 0, 0
        for real, pad in zip(m_splits, padded, strict=True):
            x_pad[dst_off : dst_off + real] = x[src_off : src_off + real]
            mask[dst_off : dst_off + real] = True
            if probs_pad is not None:
                probs_pad[dst_off : dst_off + real] = permuted_probs[src_off : src_off + real]
            src_off += real
            dst_off += pad
        return x_pad, probs_pad, padded, mask
