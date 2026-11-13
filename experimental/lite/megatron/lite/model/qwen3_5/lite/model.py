# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3.5 lite native model.

This implementation keeps the lightweight qwen3_moe/lite composition style
and uses Megatron Lite parallel, TE, RoPE, and MoE primitives directly.
"""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.primitive import transformer_engine as te
from megatron.lite.model.qwen3_5.config import Qwen35Config
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.experts import Experts, swiglu_with_probs
from megatron.lite.primitive.modules.fused_ar_norm import (
    ensure_vllm_tp_state,
    tp_all_reduce_,
    tp_col_linear,
    tp_contract_active,
    vllm_fused_ar_norm,
    vllm_fused_ar_norm_available,
)
from megatron.lite.primitive.modules.gated_delta_net import GatedDeltaNet
from megatron.lite.primitive.modules.gated_delta_net import (
    _HAS_FI_NORM as _GDN_HAS_FI_NORM,
)
from megatron.lite.primitive.modules.gated_delta_net import _vllm_pair_norm
from megatron.lite.primitive.modules.gqa import GQAttention as FullAttention
from megatron.lite.primitive.modules.gqa import split_grouped_qkvg as _split_grouped_qkvg
from megatron.lite.primitive.modules.mrope import MultimodalRotaryEmbedding as Qwen35MRoPE
from megatron.lite.primitive.modules.mtp import (
    MTPBlock,
    MTPDecoderLayer,
    MTPLossAutoScaler,
    roll_mtp_tensor_left,
)
from megatron.lite.primitive.modules.router import TopKRouter
from megatron.lite.primitive.ops.cross_entropy import (
    vllm_token_logprobs,
    vllm_token_logprobs_available,
    vocab_parallel_cross_entropy,
)
from megatron.lite.primitive.ops.linear_cross_entropy import linear_cross_entropy
from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy
from megatron.lite.primitive.parallel import (
    ColumnParallelLinear,
    ParallelState,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelOutput,
    build_pipeline_chunk_layout,
    gather_from_sequence_parallel,
    scatter_to_sequence_parallel,
)
from megatron.lite.primitive.parallel.linear import _AllGatherLastDim
from megatron.lite.primitive.utils import build_fp8_recipe

# 0-diff: native-baseline switch (C7 layer-3 three-arm comparison).
# MLITE_VLLM_CONTRACTS=0 disables every vLLM-numerics contract in the model
# so it runs the ORIGINAL TE training numerics; contracts stay the default.
_NATIVE_MODE = os.environ.get("MLITE_VLLM_CONTRACTS", "1") == "0"

_SP_GRAD_SUFFIXES: tuple[str, ...] = (
    ".full_attn.qkv.linear.layer_norm_weight",
    ".full_attn.q_norm.weight",
    ".full_attn.k_norm.weight",
    ".linear_attn.in_proj.linear.layer_norm_weight",
    ".linear_attn.norm.weight",
    ".mlp.gate_up.linear.layer_norm_weight",
    ".mlp_norm.weight",
    ".moe.router.gate.weight",
    ".moe.shared_expert.shared_gate.weight",
    ".enorm.weight",
    ".hnorm.weight",
    ".final_layernorm.weight",
)


def _collect_sp_grad_params(model: nn.Module) -> list[nn.Parameter]:
    params = []
    for name, param in model.named_parameters():
        if any(name.endswith(s) for s in _SP_GRAD_SUFFIXES) or name == "norm.weight":
            params.append(param)
    return params


def _swiglu(x: torch.Tensor) -> torch.Tensor:
    return swiglu_with_probs(x, probs=None)


def _vllm_swiglu(x: torch.Tensor) -> torch.Tensor:
    """0-diff: vLLM silu_and_mul contract for bf16 ``[..., 2d]`` inputs.

    vLLM's act_and_mul kernel (csrc/activation_kernels.cu) computes silu in
    fp32, ROUNDS TO bf16 (packed_silu_kernel's cast_to_packed), then
    multiplies by the up half in fp32 and rounds once more — the same
    double-rounding structure as plain torch ``F.silu(gate) * up``, verified
    bitwise against the in-engine capture (0 mismatch on engine + 2.6M random
    bf16 elements).  The TE bias_swiglu kernel single-rounds and differs
    (~27% of elements, up to 1.6e-2).
    """
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def _qwen_mrope_section(config: Qwen35Config) -> list[int]:
    section = getattr(config, "mrope_section", None)
    if section is not None:
        return list(section)
    rotary_half = max(int(config.rotary_dim // 2), 1)
    base = rotary_half // 3
    return [base, base, rotary_half - 2 * base]


class DenseMLP(nn.Module):
    def __init__(self, config: Qwen35Config, ps: ParallelState):
        super().__init__()
        assert config.intermediate_size is not None
        self.gate_up = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size * 2,
            ps,
            bias=False,
            normalization="RMSNorm",
            eps=config.rms_norm_eps,
            zero_centered_gamma=True,
        )
        self.down = RowParallelLinear(
            config.intermediate_size, config.hidden_size, ps, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(_swiglu(self.gate_up(x)))


class _TPSharedExpert(torch.autograd.Function):
    """Shared expert at TP>1: bitwise engine mirror, torch backward + ARs.

    Forward is the exact plain-torch mirror of the engine's per-rank shared
    expert (Qwen2MoeMLP on the loader's TP shards: gate_up / down GEMMs on
    the [gate_shard; up_shard] / I-column-block weights, sigmoid gate on the
    raw expert_gate logits) — bitwise vs the engine's captured shared_out on
    4/4 ranks (scratch/c10_tp_moe_probe.py, C10_TP_MOE_PASS).  Output is the
    PARTIAL (deferred AR — the combine with the routed partial is reduced by
    the following norm's fused AR kernel).

    Backward recomputes the same chain under autograd and all-reduces dx
    (replicated input) and d(shared_gate) (replicated weight); the gate_up /
    down shard grads stay local.
    """

    @staticmethod
    def forward(ctx, x, w_gate_up, w_down, w_shared_gate, tp_group):
        gate_up = torch.nn.functional.linear(x, w_gate_up)
        gu, gu2 = gate_up.chunk(2, dim=-1)
        act = torch.nn.functional.silu(gu) * gu2
        down = torch.nn.functional.linear(act, w_down)
        gate_logits = torch.nn.functional.linear(x, w_shared_gate)
        out = torch.sigmoid(gate_logits) * down
        ctx.save_for_backward(x, w_gate_up, w_down, w_shared_gate)
        ctx.tp_group = tp_group
        return out

    @staticmethod
    def backward(ctx, dout):
        x, w_gate_up, w_down, w_shared_gate = ctx.saved_tensors
        with torch.enable_grad():
            x_ = x.detach().requires_grad_(True)
            a_ = w_gate_up.detach().requires_grad_(True)
            b_ = w_down.detach().requires_grad_(True)
            g_ = w_shared_gate.detach().requires_grad_(True)
            gate_up = torch.nn.functional.linear(x_, a_)
            gu, gu2 = gate_up.chunk(2, dim=-1)
            act = torch.nn.functional.silu(gu) * gu2
            down = torch.nn.functional.linear(act, b_)
            gate_logits = torch.nn.functional.linear(x_, g_)
            out = torch.sigmoid(gate_logits) * down
            grads = torch.autograd.grad(out, (x_, a_, b_, g_), dout)
        tp_all_reduce_(grads[0], ctx.tp_group)  # dx: replicated input
        tp_all_reduce_(grads[3], ctx.tp_group)  # d(shared_gate): replicated
        return grads[0], grads[1], grads[2], grads[3], None


class SharedExpert(nn.Module):
    _stream: torch.cuda.Stream | None = None

    class _CopyToTPRegion(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, group):
            ctx.group = group
            return x

        @staticmethod
        def backward(ctx, grad):
            group = ctx.group
            if group is not None and dist.get_world_size(group) > 1:
                dist.all_reduce(grad, group=group)
            return grad, None

    class _PlainTELinear(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = te.Linear(
                in_features,
                out_features,
                bias=False,
                params_dtype=torch.bfloat16,
                parallel_mode=None,
                sequence_parallel=False,
                tp_group=None,
                tp_size=1,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    def __init__(
        self,
        config: Qwen35Config,
        ps: ParallelState,
        *,
        use_plain_te_linear: bool = False,
        vllm_tp_contract: bool = False,
    ):
        super().__init__()
        ffn = config.shared_expert_intermediate_size
        if use_plain_te_linear and ps.tp_size == 1:
            self.gate_up = self._PlainTELinear(config.hidden_size, ffn * 2)
            self.down = self._PlainTELinear(ffn, config.hidden_size)
        else:
            self.gate_up = ColumnParallelLinear(config.hidden_size, ffn * 2, ps, bias=False)
            self.down = RowParallelLinear(ffn, config.hidden_size, ps, bias=False)
        self.shared_gate = nn.Linear(config.hidden_size, 1, bias=False)
        self.tp_group = ps.tp_group
        self.use_mcore_overlap_graph = bool(use_plain_te_linear and ps.tp_size == 1)
        # 0-diff (qwen35-0diff-epcp): TP contract — plain GEMMs on the loader
        # shards, PARTIAL output (deferred AR); te wrappers are SP-aware at
        # tp>1 and would all-gather/mis-reduce instead.
        self.vllm_tp_contract = bool(vllm_tp_contract and ps.tp_size > 1)

    @staticmethod
    def _get_stream() -> torch.cuda.Stream:
        if SharedExpert._stream is None:
            SharedExpert._stream = torch.cuda.Stream()
        return SharedExpert._stream

    @staticmethod
    def _set_grad_fn_sequence_sr(tensor: torch.Tensor) -> None:
        grad_fn = getattr(tensor, "grad_fn", None)
        if grad_fn is not None and hasattr(grad_fn, "_set_sequence_nr"):
            grad_fn._set_sequence_nr(torch.iinfo(torch.int).max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.vllm_tp_contract and x.dtype == torch.bfloat16 and not _NATIVE_MODE:
            # 0-diff (qwen35-0diff-epcp): engine mirror on the loader's TP
            # shards (C10_TP_MOE_PASS) — partial output, deferred AR; the
            # backward all-reduces dx / d(shared_gate) inside the Function.
            shape = x.shape
            x2d = x.reshape(-1, shape[-1])
            out = _TPSharedExpert.apply(
                x2d,
                self.gate_up.linear.weight,
                self.down.linear.weight,
                self.shared_gate.weight,
                self.tp_group,
            )
            return out.view(shape)
        gate_val = self.shared_gate(x).sigmoid()
        fc1_input = x
        if self.use_mcore_overlap_graph:
            fc1_input = self._CopyToTPRegion.apply(x, self.tp_group)
            self._set_grad_fn_sequence_sr(fc1_input)
        # 0-diff: vLLM silu_and_mul contract (see _vllm_swiglu).
        act = (
            _vllm_swiglu(self.gate_up(fc1_input))
            if fc1_input.dtype == torch.bfloat16 and not _NATIVE_MODE
            else _swiglu(self.gate_up(fc1_input))
        )
        output = self.down(act)
        if self.use_mcore_overlap_graph:
            self._set_grad_fn_sequence_sr(output)
        return output * gate_val


class MoELayer(nn.Module):
    def __init__(
        self,
        config: Qwen35Config,
        ps: ParallelState,
        *,
        use_deepep: bool,
        router_bias_rate: float,
        fp8: bool,
        moe_act_recompute: bool,
        router_dtype: torch.dtype | None = None,
        preserve_3d_graph: bool = False,
        shared_expert_plain_te: bool = False,
        moe_permute_fusion: bool | None = None,
        vllm_topk_contract: bool = False,
        vllm_fused_moe_contract: bool = False,
        vllm_tp_contract: bool = False,
    ):
        super().__init__()
        if fp8:
            raise NotImplementedError("lite qwen35 MoE fp8 is not implemented yet.")
        self.router = TopKRouter(
            config,
            ps,
            router_bias_rate=router_bias_rate,
            compute_aux_loss=True,
            router_dtype=router_dtype,
            vllm_topk_contract=vllm_topk_contract,
        )
        self.ps = ps
        # 0-diff (qwen35-0diff-epcp): TP>1 plain-TP contract — the loader
        # shards each expert's intermediate dim, the routed call drives
        # fused_experts on the per-rank shards with NO all-reduce (deferred
        # AR in the next norm) and the shared expert runs the engine mirror.
        self.vllm_tp_contract = bool(
            vllm_tp_contract
            and ps.tp_size > 1
            and ps.ep_size == 1
            and ps.etp_size == 1
            and not fp8
        )
        self.experts = Experts(
            config,
            ps,
            fp8=fp8,
            moe_act_recompute=moe_act_recompute,
            expert_tp_size=ps.tp_size if self.vllm_tp_contract else 1,
        )
        # 0-diff (qwen35-0diff-epcp): EP>1 is served by the per-shard
        # vllm_fused_moe_ep contract (bitwise via pre-sum all-reduce + local
        # slot-order moe_sum), so the contract no longer requires ep_size==1.
        self.vllm_fused_moe_contract = bool(
            vllm_fused_moe_contract
            and vllm_topk_contract
            and ps.tp_size == 1
            and ps.etp_size == 1
            and not fp8
        )
        self.dispatcher = TokenDispatcher(
            config.num_experts,
            config.hidden_size,
            ps,
            use_deepep=use_deepep,
            moe_permute_fusion=moe_permute_fusion,
        )
        self.shared_expert = SharedExpert(
            config,
            ps,
            use_plain_te_linear=shared_expert_plain_te,
            vllm_tp_contract=self.vllm_tp_contract,
        )
        self.preserve_3d_graph = bool(preserve_3d_graph)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        x_2d = x.reshape(-1, x.size(-1))
        shared_input = x_2d.view(input_shape) if self.preserve_3d_graph else x_2d
        router_input = x if self.preserve_3d_graph else x_2d

        shared_out = None
        side_stream = None
        if x_2d.is_cuda:
            side_stream = SharedExpert._get_stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                shared_out = self.shared_expert(shared_input)
        scores, indices = self.router(router_input)
        if (
            self.vllm_fused_moe_contract or self.vllm_tp_contract
        ) and x_2d.dtype == torch.bfloat16:
            # 0-diff: drive vLLM's Triton fused_moe (same binaries the
            # rollout engine runs) on the stacked expert weights — bitwise
            # routed output; see experts.vllm_fused_moe.
            from megatron.lite.primitive.modules.experts import (
                stack_expert_weights,
                vllm_fused_moe,
                vllm_fused_moe_available,
                vllm_fused_moe_ep,
                vllm_fused_moe_tp,
            )

            if vllm_fused_moe_available():
                w13 = stack_expert_weights(self.experts.fc1)
                w2 = stack_expert_weights(self.experts.fc2)
                if self.vllm_tp_contract:
                    # 0-diff (qwen35-0diff-epcp): per-rank sharded call, NO
                    # all-reduce (deferred AR; C10_TP_MOE_PASS replay).
                    routed_out = vllm_fused_moe_tp(
                        x_2d.contiguous(), w13, w2, scores, indices,
                        self.ps.tp_group,
                    )
                elif self.ps.ep_size > 1:
                    routed_out = vllm_fused_moe_ep(
                        x_2d.contiguous(), w13, w2, scores, indices,
                        self.ps.ep_group, self.ps.ep_rank,
                        self.experts.num_local_experts,
                    )
                else:
                    routed_out = vllm_fused_moe(
                        x_2d.contiguous(), w13, w2, scores, indices)
                routed_out = routed_out.view(input_shape)
                scores = indices = None
            else:  # pragma: no cover - vllm missing in env
                routed_out = None
        else:
            routed_out = None
        if routed_out is None:
            dispatched, tpe, permuted_probs = self.dispatcher.dispatch(x_2d, scores, indices)
            del scores, indices
            self.dispatcher.wait_dispatch_event()
            expert_out = self.experts(
                dispatched,
                tpe,
                permuted_probs,
                tokens_per_expert_list=getattr(self.dispatcher, "_local_tpe_list", None),
            )
            routed_out = self.dispatcher.combine(expert_out)

        if shared_out is None:
            shared_out = self.shared_expert(shared_input)
        else:
            assert side_stream is not None
            torch.cuda.current_stream().wait_stream(side_stream)
        output = routed_out.view(input_shape)
        if shared_out.shape != input_shape:
            shared_out = shared_out.view(input_shape)
        output += shared_out
        return output.to(x.dtype)


# 0-diff (qwen35-0diff-epcp): CP full-stream contracts on the TIGHT stream -----
#
# The vLLM FA contract (gqa._vllm_fa_contract_forward) has no CP awareness:
# feeding it a zigzag shard as complete sequences corrupts the FA3 varlen call
# (NaN / illegal memory access — C10 mesh probe).  And the CP pack layout pads
# every sequence to a multiple of 2*cp, which the CP-off (TP1, engine-bitwise)
# path never does — running contract GEMMs on either the local shard
# (M = T/4) or the padded full stream (M = T + pad) selects different cuBLAS
# kernels than M = T and corrupts individual rows at 1 ulp (measured on the
# shared expert's plain GEMMs, incl. the N=1 shared gate; see
# c10mesh/DESIGN.md).  The bitwise CP contract therefore gathers the zigzag
# parts, strips the pad rows, and runs the contract body on the TIGHT full
# stream — byte-identical to the CP-off input (same M, same cu, same code
# path), so the layer output is bitwise by construction — then scatters the
# outputs back into this rank's local shard (pad rows zero).
#
# Applied to EVERY contract layer (GDN and FA alike): the whole layer body
# (attention + projections + MoE) runs on the gathered tight stream — the
# GDN in/out projections are plain GEMMs too and hit the same M-dependence
# (the headwise CP path was bitwise on the 31-seq group by luck: M=2520 vs
# 9989 picked the same cuBLAS kernels; the single-seq group, M=84 vs 336,
# did not).  The in-process GDN CP view (headwise all-to-all) is bypassed
# inside the wrapper via ``disable_cp``.  At EP>1 the gathered stream is
# also exactly what vllm_fused_moe_ep needs.
#
# Cost: contract layers run redundantly on every CP rank (~cp x FLOPs);
# recorded as perf debt.
#
# Backward: the scatter's backward all-reduces the per-rank output grads
# into the total full-stream grad (each rank owns a disjoint row set), and
# the gather's backward then just splits the total onto the local shard —
# no second all-reduce (the body ran redundantly, so every rank already
# holds the total; summing again would count it cp times).


def _thd_token_dim(t: torch.Tensor) -> int:
    """Token dim of a THD stream tensor: (1,T,H)/(T,1,H) → 1/0, MRoPE (3,1,T) → 2."""
    if t.dim() == 3 and t.shape[0] == 3:
        return 2
    if t.dim() == 3:
        return 1 if t.shape[0] == 1 else 0
    if t.dim() == 2:
        return 0
    raise ValueError(f"not a THD stream shape: {tuple(t.shape)}")


def _tight_stream_dim(t: torch.Tensor) -> int:
    """Token dim for the model-level tight-stream CP wrappers.

    _thd_token_dim assumes [T, H] for 2-D tensors, but the packer emits
    [1, T] rows for input_ids/labels — their token dim is 1.
    """
    if t.dim() == 2 and t.shape[0] == 1 and t.shape[1] > 1:
        return 1
    return _thd_token_dim(t)


_TIGHT_ROW_CACHE: dict = {}


def _tight_real_rows(cu_pad: torch.Tensor, cu_real: torch.Tensor) -> torch.Tensor:
    """Index of real (non-pad) rows in the padded stream, cached per layout."""
    key = (tuple(cu_pad.tolist()), tuple(cu_real.tolist()))
    idx = _TIGHT_ROW_CACHE.get(key)
    if idx is None:
        rows = []
        for a, r0, r1 in zip(
            cu_pad[:-1].tolist(), cu_real[:-1].tolist(), cu_real[1:].tolist()
        ):
            rows.append(torch.arange(a, a + (r1 - r0), device=cu_pad.device))
        idx = torch.cat(rows)
        _TIGHT_ROW_CACHE[key] = idx
    return idx


def _tight_max_seqlen(cu_real: torch.Tensor) -> int:
    lens = (cu_real[1:] - cu_real[:-1])
    return int(lens.max().item()) if lens.numel() else 0


class _CPGatherTightStream(torch.autograd.Function):
    """All-gather zigzag parts, reconstruct the padded full stream, strip pads."""

    @staticmethod
    def forward(ctx, t, dim, cu_pad, cu_real, cp_size, cp_rank, cp_group):
        from megatron.lite.primitive.parallel.thd import (
            _all_gather_cp_tensor,
            reconstruct_packed_from_cp_parts,
        )

        parts = _all_gather_cp_tensor(t, cp_size=cp_size, cp_group=cp_group)
        full_pad = reconstruct_packed_from_cp_parts(
            parts, cu_seqlens_padded=cu_pad, cp_size=cp_size, dim=dim
        )
        tight = full_pad.index_select(dim, _tight_real_rows(cu_pad, cu_real))
        ctx.meta = (dim, cu_pad, cu_real, cp_size, cp_rank, cp_group)
        return tight

    @staticmethod
    def backward(ctx, grad_tight):
        from megatron.lite.primitive.parallel.thd import split_packed_to_cp_local

        dim, cu_pad, cu_real, cp_size, cp_rank, cp_group = ctx.meta
        shape = list(grad_tight.shape)
        shape[dim] = int(cu_pad[-1].item())
        grad_pad = torch.zeros(
            shape, dtype=grad_tight.dtype, device=grad_tight.device
        )
        grad_pad.index_copy_(dim, _tight_real_rows(cu_pad, cu_real), grad_tight)
        # grad_tight is already the total (the scatter backward all-reduced
        # it and the body ran identically on every rank): only split.
        grad_local = split_packed_to_cp_local(
            grad_pad, cu_seqlens_padded=cu_pad, cp_size=cp_size,
            cp_rank=cp_rank, dim=dim,
        )
        return grad_local, None, None, None, None, None, None


class _CPScatterTightToLocal(torch.autograd.Function):
    """Scatter tight full-stream values into this rank's local shard (pads 0)."""

    @staticmethod
    def forward(ctx, t_tight, dim, cu_pad, cu_real, cp_size, cp_rank, cp_group):
        from megatron.lite.primitive.parallel.thd import split_packed_to_cp_local

        shape = list(t_tight.shape)
        shape[dim] = int(cu_pad[-1].item())
        full_pad = torch.zeros(
            shape, dtype=t_tight.dtype, device=t_tight.device
        )
        full_pad.index_copy_(dim, _tight_real_rows(cu_pad, cu_real), t_tight)
        ctx.meta = (dim, cu_pad, cu_real, cp_size, cp_rank, cp_group)
        return split_packed_to_cp_local(
            full_pad, cu_seqlens_padded=cu_pad, cp_size=cp_size,
            cp_rank=cp_rank, dim=dim,
        )

    @staticmethod
    def backward(ctx, grad_local):
        from megatron.lite.primitive.parallel.thd import (
            reconstruct_packed_from_cp_parts,
        )

        dim, cu_pad, cu_real, cp_size, cp_rank, cp_group = ctx.meta
        parts = [torch.zeros_like(grad_local) for _ in range(cp_size)]
        parts[cp_rank] = grad_local
        grad_pad = reconstruct_packed_from_cp_parts(
            parts, cu_seqlens_padded=cu_pad, cp_size=cp_size, dim=dim
        )
        # Sum the disjoint per-rank contributions into the total tight grad.
        grad_tight = grad_pad.index_select(dim, _tight_real_rows(cu_pad, cu_real))
        dist.all_reduce(grad_tight, group=cp_group)
        return grad_tight, None, None, None, None, None, None


def _cp_tight_meta(packed_seq_params):
    """(cu_pad, cu_real) for the CP contracts; raises when tight cu is absent."""
    cu_pad = packed_seq_params.cu_seqlens_q
    cu_real = getattr(packed_seq_params, "cu_seqlens_q_real", None)
    if cu_real is None:
        raise RuntimeError(
            "CP full-stream contract requires PackedSeqParams.cu_seqlens_q_real "
            "(attached by pack_thd_forward_kwargs)."
        )
    return cu_pad, cu_real


class Qwen35Layer(nn.Module):
    def __init__(
        self,
        config: Qwen35Config,
        ps: ParallelState,
        layer_idx: int,
        *,
        use_deepep: bool = False,
        router_bias_rate: float = 0.0,
        fp8: bool = False,
        moe_act_recompute: bool = False,
        use_thd: bool = False,
        deterministic: bool = False,
        gdn_cp_mode: str = "headwise",
        vllm_tp_contract: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.ps = ps
        # 0-diff (qwen35-0diff-epcp): plain-TP deferred-AR contract — every
        # sublayer boundary norm is the fused AR+Gemma norm sync point on the
        # previous row-parallel partial.  False at TP=1 and in native mode.
        self._vllm_tp_contract = bool(vllm_tp_contract and ps.tp_size > 1)
        self._norm_eps = config.rms_norm_eps
        self._layer_type = config.layer_type_at(layer_idx)
        self.full_attn: FullAttention | None = None
        self.linear_attn: GatedDeltaNet | None = None
        if self._layer_type == "full_attention":
            self.full_attn = FullAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                ps=ps,
                rms_norm_eps=config.rms_norm_eps,
                rope_theta=config.rope_theta,
                rotary_percent=config.partial_rotary_factor,
                use_thd=use_thd,
                output_gate=True,
                zero_centered_gamma=True,
                qkv_layout="mcore",
                mrope_section=_qwen_mrope_section(config),
                # 0-diff (qwen35-0diff-lite): drive vLLM's own FA-path kernels
                # (fused_qk_rmsnorm_rope_gate + FA3 varlen + fused_sigmoid_gate_
                # mul) so the layer is bitwise vs the rollout engine.  Falls
                # back to the TE path when the kernels/inputs are outside the
                # C3-validated window (TP=1, bf16, THD).
                vllm_fa_contract=not _NATIVE_MODE,
                max_position_embeddings=config.max_position_embeddings,
            )
        else:
            self.linear_attn = GatedDeltaNet(
                hidden_size=config.hidden_size,
                linear_num_key_heads=config.linear_num_key_heads,
                linear_key_head_dim=config.linear_key_head_dim,
                linear_num_value_heads=config.linear_num_value_heads,
                linear_value_head_dim=config.linear_value_head_dim,
                linear_conv_kernel_dim=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                ps=ps,
                deterministic=deterministic,
                cp_mode=gdn_cp_mode,
            )
        if config.is_moe:
            self.mlp_norm: nn.Module | None = te.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, zero_centered_gamma=True
            )
            self.moe: MoELayer | None = MoELayer(
                config,
                ps,
                use_deepep=use_deepep,
                router_bias_rate=router_bias_rate,
                fp8=fp8,
                moe_act_recompute=moe_act_recompute,
                router_dtype=torch.float32,
                preserve_3d_graph=deterministic,
                shared_expert_plain_te=deterministic,
                moe_permute_fusion=True,
                # 0-diff (qwen35-0diff-lite): route bit-exactly as the rollout
                # side does (plain bf16 gate GEMM + vLLM topkGating weights);
                # the fp32 TE gate mis-routed 10-14% of tokens per layer.
                vllm_topk_contract=not _NATIVE_MODE,
                # 0-diff: routed experts through vLLM's Triton fused_moe
                # (bitwise vs the rollout engine; TE grouped-GEMM path stays
                # the fallback for TP/EP or non-bf16 configs).
                vllm_fused_moe_contract=not _NATIVE_MODE,
                vllm_tp_contract=self._vllm_tp_contract,
            )
            self.mlp: DenseMLP | None = None
        else:
            self.mlp_norm = None
            self.moe = None
            self.mlp = DenseMLP(config, ps)

    def _contract_cp_full_stream(
        self, x, position_ids, packed_seq_params, residual
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CP wrapper: gather the TIGHT full stream, run the contract layer, scatter.

        The CP-packed psp keeps the GLOBAL padded cu_seqlens (only the
        local_cp_size/cp_rank/cp_group metadata is attached; the stream
        tensors are pre-split to the local zigzag shard).  The contract body
        must instead see the CP-off input exactly: the tight (true-length)
        full stream with a fresh plain psp — see the _CPGatherTightStream
        block above for why anything else breaks per-row bitwise.
        """
        from megatron.lite.primitive.utils.packed_seq import PackedSeqParams

        ps = self.ps
        cp = ps.cp_size
        cu_pad, cu_real = _cp_tight_meta(packed_seq_params)
        gather = _CPGatherTightStream.apply
        xg = gather(x, _thd_token_dim(x), cu_pad, cu_real, cp,
                    ps.cp_rank, ps.cp_group)
        if residual is not None:
            residual = gather(residual, _thd_token_dim(residual),
                              cu_pad, cu_real, cp, ps.cp_rank, ps.cp_group)
        if position_ids is not None:
            position_ids = gather(
                position_ids, _thd_token_dim(position_ids),
                cu_pad, cu_real, cp, ps.cp_rank, ps.cp_group)
        psp_full = PackedSeqParams.from_cu_seqlens(
            cu_real, max_seqlen=_tight_max_seqlen(cu_real)
        )
        h, res_out = Qwen35Layer.forward(
            self, xg, position_ids=position_ids,
            packed_seq_params=psp_full, residual=residual,
            _cp_full_stream=True,
        )
        scatter = _CPScatterTightToLocal.apply
        h = scatter(h, _thd_token_dim(h), cu_pad, cu_real, cp,
                    ps.cp_rank, ps.cp_group)
        if res_out is not None:
            res_out = scatter(res_out, _thd_token_dim(res_out),
                              cu_pad, cu_real, cp, ps.cp_rank, ps.cp_group)
        return h, res_out

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
        residual: torch.Tensor | None = None,
        _cp_full_stream: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(module output, residual stream)``, both UN-ADDED.

        0-diff (qwen35-0diff-lite): vLLM keeps the residual stream un-added
        between sublayers and lets each Gemma fused add+RMSNorm consume the
        exact fp32 (x + residual) inside the kernel. Pre-adding in bf16 loses
        the exact sum and diverged 51% of in_proj outputs on captured data,
        so the model loop now carries the (h, residual) pair.
        """
        if (
            self.ps.cp_size > 1
            and not _cp_full_stream
            and packed_seq_params is not None
            and getattr(packed_seq_params, "cu_seqlens_q_real", None) is not None
            and _GDN_HAS_FI_NORM
            and not _NATIVE_MODE
        ):
            # 0-diff (qwen35-0diff-epcp): CP full-stream contract wrapper —
            # every contract layer (GDN and FA alike) runs its whole body on
            # the gathered TIGHT full stream; see the _CPGatherTightStream
            # block above.
            return self._contract_cp_full_stream(
                x, position_ids, packed_seq_params, residual
            )
        if self.full_attn is not None:
            # 0-diff (qwen35-0diff-lite): vLLM FA-layer input norm contract
            # (flashinfer Gemma fused add+RMSNorm on the fp32 sum; the fused
            # AR variant at TP>1 — see fused_ar_norm), then GQAttention's
            # vLLM FA kernel contract on the post-norm activation.
            # qkv.linear.layer_norm_weight stores the raw
            # input_layernorm.weight (no -1), the convention the kernel wants.
            if self.full_attn.vllm_fa_engages(x, packed_seq_params) and _GDN_HAS_FI_NORM:
                if self._vllm_tp_contract and residual is not None:
                    y, residual = vllm_fused_ar_norm(
                        x,
                        residual,
                        self.full_attn.qkv.linear.layer_norm_weight,
                        self._norm_eps,
                    )
                else:
                    y, residual = _vllm_pair_norm(
                        x,
                        residual,
                        self.full_attn.qkv.linear.layer_norm_weight,
                        self._norm_eps,
                    )
                h = self.full_attn(y, position_ids=position_ids, packed_seq_params=packed_seq_params)
            else:
                # Legacy path: pre-added input into the fused te.LayerNormLinear.
                xs = x if residual is None else residual + x
                h = self.full_attn(xs, position_ids=position_ids, packed_seq_params=packed_seq_params)
                residual = xs
        else:
            assert self.linear_attn is not None
            h, residual = self.linear_attn(
                x,
                position_ids=position_ids,
                packed_seq_params=packed_seq_params,
                residual=residual,
                disable_cp=_cp_full_stream,
            )
        if self.moe is not None:
            assert self.mlp_norm is not None
            # 0-diff (qwen35-0diff-lite): vLLM post-attention norm contract
            # (flashinfer Gemma fused add+RMSNorm on the fp32 sum; the fused
            # AR variant at TP>1 — see fused_ar_norm).
            if (
                _GDN_HAS_FI_NORM
                and (self.ps.tp_size == 1 or self._vllm_tp_contract)
                and not _NATIVE_MODE
            ):
                if self._vllm_tp_contract and residual is not None:
                    y, residual = vllm_fused_ar_norm(
                        h, residual, self.mlp_norm.weight, self._norm_eps
                    )
                else:
                    y, residual = _vllm_pair_norm(
                        h, residual, self.mlp_norm.weight, self._norm_eps
                    )
                return self.moe(y), residual
            xs = h if residual is None else residual + h
            return self.moe(self.mlp_norm(xs)), xs
        assert self.mlp is not None
        xs = h if residual is None else residual + h
        return self.mlp(xs), xs


def _temperature_to_float(temperature: float | torch.Tensor) -> float:
    if isinstance(temperature, torch.Tensor):
        if temperature.numel() != 1:
            raise ValueError("Qwen35Model fused/MTP SFT supports scalar temperature only.")
        return float(temperature.detach().float().item())
    return float(temperature)


def _ensure_mrope_position_ids(position_ids: torch.Tensor | None) -> torch.Tensor | None:
    if position_ids is None:
        return None
    if position_ids.dim() == 2:
        return position_ids.unsqueeze(0).expand(3, -1, -1).contiguous()
    if position_ids.dim() == 3:
        if position_ids.shape[0] == 1:
            return position_ids.expand(3, -1, -1).contiguous()
        if position_ids.shape[0] == 3:
            return position_ids
    raise ValueError("Qwen3.5 MRoPE expects position_ids shape (B,S), (1,B,S), or (3,B,S).")


def _apply_attention_backend_override(backend: str | None) -> None:
    # Direct protocol users pass None.  Restore TE's automatic selection in
    # that case so a previously constructed model cannot leak its choice into
    # Qwen3.5.  The runtime passes an explicit override when one is configured.
    if backend is None:
        backend = "auto"
    env = {
        "auto": ("1", "1", "1"),
        "flash": ("1", "0", "0"),
        "fused": ("0", "1", "0"),
        "unfused": ("0", "0", "1"),
        "local": ("0", "0", "1"),
    }.get(backend)
    if env is None:
        raise ValueError(
            "attention_backend_override must be one of "
            "{'auto', 'flash', 'fused', 'unfused', 'local'}"
        )
    (
        os.environ["NVTE_FLASH_ATTN"],
        os.environ["NVTE_FUSED_ATTN"],
        os.environ["NVTE_UNFUSED_ATTN"],
    ) = env


class Qwen35Model(nn.Module):
    def __init__(
        self,
        config: Qwen35Config,
        train_config,
        ps: ParallelState,
        *,
        vpp_chunk_id: int | None = None,
        router_bias_rate: float = 0.0,
        use_thd: bool = False,
        hf_path: str = "",
        attention_backend_override: str | None = None,
        mtp_enable: bool = False,
        mtp_enable_train: bool = False,
        mtp_detach_encoder: bool = False,
        mount_vision_model: bool = False,
        gdn_cp_mode: str = "headwise",
    ):
        super().__init__()
        _apply_attention_backend_override(attention_backend_override)
        self.config = config
        self.train_config = train_config
        self.ps = ps
        # 0-diff (qwen35-0diff-epcp): plain-TP deferred-AR contract.  At
        # TP>1 (contracts on) build vLLM's in-process TP state FIRST so the
        # fused AR+Gemma-norm sync points and the per-module gates (which
        # read fused_ar_norm.tp_contract_active) see it armed.  Pure-TP
        # meshes only (tp == world): vLLM builds its TP group from the world
        # in rank order, which matches the trainer's tp group exactly there;
        # CP/EP x TP compositions are out of the contract window for now.
        self._vllm_tp_contract = False
        if ps.tp_size > 1 and not _NATIVE_MODE:
            if dist.is_initialized():
                if dist.get_world_size() != ps.tp_size:
                    raise NotImplementedError(
                        "The vLLM plain-TP contract currently supports pure-TP meshes "
                        f"(tp == world_size); got tp={ps.tp_size}, "
                        f"world={dist.get_world_size()}."
                    )
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                if not vllm_fused_ar_norm_available() or not ensure_vllm_tp_state(
                    ps.tp_size, dist.get_rank(), local_rank
                ):
                    raise RuntimeError(
                        "TP>1 with contracts on requires the vLLM fused AR+norm op "
                        "and in-process vLLM TP state (ensure_vllm_tp_state failed)."
                    )
                self._vllm_tp_contract = True
            else:
                # Construction without distributed init (meta/CPU tests):
                # layers still build; the per-forward gates stay off until
                # ensure_vllm_tp_state runs.
                pass
        self.mtp_enable_train = bool(mtp_enable and mtp_enable_train)
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor
        self._input_tensor: torch.Tensor | None = None

        layout = build_pipeline_chunk_layout(
            config.num_hidden_layers, ps, train_config.vpp, vpp_chunk_id
        )
        self.layer_indices = layout.layer_indices
        has_embed = layout.has_embed
        has_head = layout.has_head
        self.pre_process = has_embed
        self.post_process = has_head
        self.share_embeddings_and_output_weights = False
        self.vision_model: nn.Module | None = (
            _build_native_vision_model(hf_path) if has_embed and mount_vision_model else None
        )

        self.embed: VocabParallelEmbedding | None = None
        if has_embed:
            self.embed = VocabParallelEmbedding(config.vocab_size, config.hidden_size, ps)

        recompute_modules = getattr(train_config, "recompute_modules", [])
        moe_act_recompute = "moe_act" in recompute_modules and "moe" not in recompute_modules
        self.layers = nn.ModuleList(
            [
                Qwen35Layer(
                    config,
                    ps,
                    idx,
                    use_deepep=train_config.use_deepep,
                    router_bias_rate=router_bias_rate,
                    fp8=train_config.fp8,
                    moe_act_recompute=moe_act_recompute,
                    use_thd=use_thd,
                    deterministic=getattr(train_config, "deterministic", False),
                    gdn_cp_mode=gdn_cp_mode,
                    vllm_tp_contract=self._vllm_tp_contract,
                )
                for idx in self.layer_indices
            ]
        )

        self.norm: nn.Module | None = None
        self.head: VocabParallelOutput | None = None
        if has_head:
            self.norm = te.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, zero_centered_gamma=True
            )
            self.head = VocabParallelOutput(config.vocab_size, config.hidden_size, ps)
            self._norm_eps = config.rms_norm_eps
            # 0-diff (qwen35-0diff-epcp): at TP>1 the final norm is the fused
            # AR+Gemma norm on the LAST layer's MoE partial (the engine's
            # deferred-AR structure); the plain pair-norm at TP=1 is the
            # C3-validated contract.
            self._vllm_head_norm_contract = (
                _GDN_HAS_FI_NORM
                and (ps.tp_size == 1 or self._vllm_tp_contract)
                and not _NATIVE_MODE
            )

        self.mtp_embed: VocabParallelEmbedding | None = None
        self.mtp: MTPBlock | None = None
        if mtp_enable and config.num_nextn_predict_layers > 0 and self.head is not None:
            mtp_embedding = self.embed
            if mtp_embedding is None:
                mtp_embedding = VocabParallelEmbedding(config.vocab_size, config.hidden_size, ps)
                self.mtp_embed = mtp_embedding

            def make_mtp_layer(layer_idx: int) -> MTPDecoderLayer:
                return MTPDecoderLayer(
                    hidden_size=config.hidden_size,
                    rms_norm_eps=config.rms_norm_eps,
                    ps=ps,
                    embedding=mtp_embedding,
                    transformer_layer=Qwen35Layer(
                        config,
                        ps,
                        config.num_hidden_layers + layer_idx,
                        use_deepep=train_config.use_deepep,
                        router_bias_rate=router_bias_rate,
                        fp8=train_config.fp8,
                        moe_act_recompute=moe_act_recompute,
                        use_thd=use_thd,
                        deterministic=getattr(train_config, "deterministic", False),
                        gdn_cp_mode=gdn_cp_mode,
                        vllm_tp_contract=self._vllm_tp_contract,
                    ),
                    detach_encoder=mtp_detach_encoder,
                )

            self.mtp = MTPBlock(
                num_layers=config.num_nextn_predict_layers,
                repeated_layer=config.mtp_use_repeated_layer,
                layer_factory=make_mtp_layer,
            )

        self.sp_params: list[nn.Parameter] = []
        if ps.tp_size > 1 and not self._vllm_tp_contract:
            # 0-diff (qwen35-0diff-epcp): the TP contract's own backward
            # all-reduces every replicated-param gradient to the TOTAL (dx
            # ARs in the GEMM/norm/MoE contract Functions) — the native SP
            # grad hooks would double-count on top and must stay off.
            self.sp_params = _collect_sp_grad_params(self)

    def set_input_tensor(self, input_tensor):
        if isinstance(input_tensor, list):
            if len(input_tensor) > 1:
                raise ValueError("Qwen35Model expects a single pipeline input tensor.")
            input_tensor = input_tensor[0] if input_tensor else None
        self._input_tensor = input_tensor

    def _head_ce_cp_tight(self, h, residual, labels, temperature,
                          calculate_entropy, packed_seq_params) -> dict:
        """CP wrapper for the head block: gather tight, run CP-off head+CE, scatter.

        Mirrors the CP-off head code exactly (norm contract + head GEMM +
        vLLM token-logprobs contract) on the gathered TIGHT full stream —
        bitwise by construction — then slices the log_probs/entropy back to
        the local shard.  The model-level parameters (final norm, head — and
        embed via its own wrapper) then hold the TOTAL gradient on every
        rank, same invariant as the layer contracts.

        Loss normalization: local token sum over the GLOBAL token count
        (one scalar all-reduce), so each rank's backward reproduces the
        CP-off mean-loss gradient after the scatter backward's all-reduce.
        MTP and the fused-CE kernel path are not CP-covered (fall back to
        the unwrapped path below).
        """
        ps = self.ps
        cp = ps.cp_size
        cu_pad, cu_real = _cp_tight_meta(packed_seq_params)
        gather = _CPGatherTightStream.apply
        scatter = _CPScatterTightToLocal.apply
        hg = gather(h, _tight_stream_dim(h), cu_pad, cu_real, cp,
                    ps.cp_rank, ps.cp_group)
        resg = (gather(residual, _tight_stream_dim(residual), cu_pad, cu_real, cp,
                       ps.cp_rank, ps.cp_group)
                if residual is not None else None)
        lbg = gather(labels, _tight_stream_dim(labels), cu_pad, cu_real, cp,
                     ps.cp_rank, ps.cp_group)
        output: dict = {}
        if (
            self._vllm_head_norm_contract
            and resg is not None
            and hg.dtype == torch.bfloat16
        ):
            hidden_for_head, h_sum = _vllm_pair_norm(
                hg, resg, self.norm.weight, self._norm_eps
            )
            output["hidden_states"] = scatter(
                h_sum, _thd_token_dim(h_sum), cu_pad, cu_real, cp,
                ps.cp_rank, ps.cp_group)
        else:
            if resg is not None:
                hg = resg + hg
            output["hidden_states"] = scatter(
                hg, _thd_token_dim(hg), cu_pad, cu_real, cp,
                ps.cp_rank, ps.cp_group)
            hidden_for_head = self.norm(hg)
        temperature_value = _temperature_to_float(temperature)
        labels_sb = lbg.transpose(0, 1).contiguous()
        logits = self.head(hidden_for_head)
        labels_valid = bool(
            (labels_sb >= 0).all() and (labels_sb < self.config.vocab_size).all()
        )
        entropy = None
        if (
            self._vllm_head_norm_contract
            and temperature_value == 1.0
            and labels_valid
            and vllm_token_logprobs_available()
            and logits.dtype == torch.bfloat16
        ):
            log_probs = vllm_token_logprobs(logits, labels_sb)
            if calculate_entropy:
                entropy = vocab_parallel_entropy(logits, self.ps.tp_group)
        else:
            if temperature_value != 1.0:
                logits = logits / temperature_value
            loss = vocab_parallel_cross_entropy(
                logits, labels_sb, self.ps.tp_group
            )
            log_probs = -loss
            if calculate_entropy:
                entropy = vocab_parallel_entropy(logits, self.ps.tp_group)
        lp_local = scatter(log_probs, _thd_token_dim(log_probs), cu_pad,
                           cu_real, cp, ps.cp_rank, ps.cp_group)
        output["log_probs"] = lp_local.transpose(0, 1).contiguous()
        if entropy is not None:
            ent_local = scatter(entropy, _thd_token_dim(entropy), cu_pad,
                                cu_real, cp, ps.cp_rank, ps.cp_group)
            output["entropy"] = ent_local.transpose(0, 1).contiguous()
        n_global = torch.tensor(
            [float(lp_local.numel())], device=lp_local.device, dtype=torch.float64
        )
        dist.all_reduce(n_global, group=ps.cp_group)
        output["loss"] = (-lp_local.float().sum()) / n_global
        return output

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        temperature: float | torch.Tensor = 1.0,
        use_fused_kernels: bool = False,
        calculate_entropy: bool = False,
    ) -> dict:
        if self.embed is not None:
            assert input_ids is not None
            if (
                self.ps.cp_size > 1
                and packed_seq_params is not None
                and getattr(packed_seq_params, "cu_seqlens_q_real", None) is not None
                and _GDN_HAS_FI_NORM
                and not _NATIVE_MODE
            ):
                # 0-diff (qwen35-0diff-epcp): embedding on the gathered TIGHT
                # full ids — per-rank dW accumulates ALL tokens (the scatter
                # backward all-reduces), matching the CP-off gradient.
                cu_pad, cu_real = _cp_tight_meta(packed_seq_params)
                ids_t = _CPGatherTightStream.apply(
                    input_ids, _tight_stream_dim(input_ids), cu_pad, cu_real,
                    self.ps.cp_size, self.ps.cp_rank, self.ps.cp_group)
                h_t = self.embed(ids_t)
                h = _CPScatterTightToLocal.apply(
                    h_t, _thd_token_dim(h_t), cu_pad, cu_real,
                    self.ps.cp_size, self.ps.cp_rank, self.ps.cp_group)
            else:
                h = self.embed(input_ids)
        else:
            if hidden_states is None:
                hidden_states = self._input_tensor
            assert hidden_states is not None
            h = hidden_states

        if packed_seq_params is not None:
            assert position_ids is not None, "THD path requires caller-supplied MRoPE position_ids."
        elif position_ids is None and input_ids is not None:
            if self.ps.cp_size > 1:
                raise ValueError("CP>1 requires caller-supplied FULL position_ids (3,B,S).")
            batch, seq = input_ids.shape
            pos = torch.arange(seq, device=input_ids.device, dtype=torch.long)
            position_ids = pos.unsqueeze(0).unsqueeze(0).expand(3, batch, seq).contiguous()
        position_ids = _ensure_mrope_position_ids(position_ids)

        fp8_ctx = (
            te.fp8_autocast(enabled=True, fp8_recipe=build_fp8_recipe(self.train_config))
            if self.train_config.fp8
            else nullcontext()
        )
        with fp8_ctx:
            if self.embed is not None and not self._vllm_tp_contract:
                # 0-diff (qwen35-0diff-epcp): the TP contract keeps the FULL
                # replicated token stream on every rank (the engine's
                # plain-TP structure — no sequence sharding).
                h = scatter_to_sequence_parallel(h, self.ps)
            # 0-diff (qwen35-0diff-lite): carry the UN-ADDED (h, residual) pair
            # through the stack; each layer's fused norms consume the exact
            # fp32 sum (vLLM layout). See Qwen35Layer.forward.
            residual = None
            for layer in self.layers:
                h, residual = layer(
                    h,
                    residual=residual,
                    position_ids=position_ids,
                    packed_seq_params=packed_seq_params,
                )

        output: dict = {}
        if self.head is not None:
            assert self.norm is not None
            # 0-diff (qwen35-0diff-lite): the engine's final norm is the
            # flashinfer Gemma fused-add rmsnorm applied to the UN-ADDED
            # pair — the fp32 sum lives only inside the kernel, so bf16-add
            # + te.RMSNorm rounds once more and offsets the weight by the
            # (gamma-1) double rounding.  The contract is bitwise (verified
            # 0/335 logprob mismatch on the C3 capture); residual_out
            # bf16(fp32 sum) equals the plain bf16 add, so the reported
            # hidden_states are unchanged.
            if (
                self._vllm_head_norm_contract
                and residual is not None
                and h.dtype == torch.bfloat16
            ):
                if self._vllm_tp_contract:
                    # 0-diff (qwen35-0diff-epcp): final norm = fused AR norm
                    # on the last layer's MoE partial (deferred-AR sync
                    # point, bitwise vs the engine TP4 dumps).
                    hidden_for_head, h_sum = vllm_fused_ar_norm(
                        h, residual, self.norm.weight, self._norm_eps
                    )
                else:
                    hidden_for_head, h_sum = _vllm_pair_norm(
                        h, residual, self.norm.weight, self._norm_eps
                    )
                output["hidden_states"] = h_sum
            else:
                if residual is not None:
                    h = residual + h
                output["hidden_states"] = h
                hidden_for_head = self.norm(h)
        else:
            if residual is not None:
                h = residual + h
            output["hidden_states"] = h
        if self.head is not None and labels is not None:
            temperature_value = _temperature_to_float(temperature)
            if (
                self.ps.cp_size > 1
                and packed_seq_params is not None
                and getattr(packed_seq_params, "cu_seqlens_q_real", None) is not None
                and not use_fused_kernels
                and _GDN_HAS_FI_NORM
                and not _NATIVE_MODE
            ):
                # 0-diff (qwen35-0diff-epcp): head+CE on the gathered TIGHT
                # full stream (see _head_ce_cp_tight).  MTP and the fused-CE
                # kernel path are not CP-covered and take the unwrapped path.
                return self._head_ce_cp_tight(
                    h, residual, labels, temperature, calculate_entropy,
                    packed_seq_params)
            mtp_result = self._apply_mtp_loss(
                hidden_for_head,
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                loss_mask=loss_mask,
                packed_seq_params=packed_seq_params,
                temperature=temperature_value,
                use_fused_kernels=use_fused_kernels,
            )
            if mtp_result is not None:
                hidden_for_head, mtp_loss = mtp_result
                output["mtp_loss"] = mtp_loss
            labels_sb = labels.transpose(0, 1).contiguous()
            if use_fused_kernels:
                hidden_full = gather_from_sequence_parallel(hidden_for_head, self.ps)
                log_probs, entropy = linear_cross_entropy(
                    hidden_full,
                    self._head_weight_for_fused_ce(hidden_full),
                    labels_sb,
                    temperature_value,
                    self.ps.tp_group,
                )
                output["loss"] = (-log_probs).mean()
                output["log_probs"] = log_probs.transpose(0, 1).contiguous()
                if calculate_entropy:
                    output["entropy"] = entropy.transpose(0, 1).contiguous()
            else:
                if self._vllm_tp_contract:
                    # 0-diff (qwen35-0diff-epcp): TP contract head — plain
                    # column GEMM on the replicated hidden with the per-rank
                    # vocab shard (engine GEMM shape; backward all-reduces
                    # dx), then an autograd-aware all-gather + vocab trim to
                    # the FULL logits every rank consumes (the engine's
                    # logprob path runs on the gathered logits).
                    logits_shard = tp_col_linear(
                        hidden_for_head,
                        self.head.col.linear.weight,
                        self.ps.tp_group,
                    )
                    logits_full = _AllGatherLastDim.apply(
                        logits_shard, self.ps.tp_size, self.ps.tp_group
                    )[..., : self.head.vocab_size]
                    logits = logits_full
                else:
                    logits_shard = logits = self.head(hidden_for_head)
                # 0-diff (qwen35-0diff-lite): vLLM prompt-logprob contract —
                # drive the engine's own Triton chunked log-softmax kernel
                # on the RAW bf16 logits (no temperature; the engine's
                # raw_logprobs mode applies none).  Bitwise vs the engine
                # (0/335 on the C3 capture); torch CE differs by ~1e-7 ulp
                # noise.  The kernel does not clamp ids, so gate on label
                # validity.  Backward is an exact analytic softmax.
                labels_valid = bool(
                    (labels_sb >= 0).all() and (labels_sb < self.config.vocab_size).all()
                )
                if (
                    self._vllm_head_norm_contract
                    and temperature_value == 1.0
                    and labels_valid
                    and vllm_token_logprobs_available()
                    and logits.dtype == torch.bfloat16
                ):
                    log_probs = vllm_token_logprobs(logits, labels_sb)
                    output["loss"] = (-log_probs).mean()
                    output["log_probs"] = log_probs.transpose(0, 1).contiguous()
                    if calculate_entropy:
                        entropy = vocab_parallel_entropy(logits_shard, self.ps.tp_group)
                        output["entropy"] = entropy.transpose(0, 1).contiguous()
                else:
                    if temperature_value != 1.0:
                        logits_shard = logits_shard / temperature_value
                        logits = logits / temperature_value
                    loss = vocab_parallel_cross_entropy(logits_shard, labels_sb, self.ps.tp_group)
                    output["loss"] = loss.mean()
                    output["log_probs"] = (-loss).transpose(0, 1).contiguous()
                    if calculate_entropy:
                        entropy = vocab_parallel_entropy(logits_shard, self.ps.tp_group)
                        output["entropy"] = entropy.transpose(0, 1).contiguous()
        elif self.head is not None:
            if self._vllm_tp_contract:
                logits_shard = tp_col_linear(
                    hidden_for_head,
                    self.head.col.linear.weight,
                    self.ps.tp_group,
                )
                logits = _AllGatherLastDim.apply(
                    logits_shard, self.ps.tp_size, self.ps.tp_group
                )[..., : self.head.vocab_size]
            else:
                logits = self.head(hidden_for_head)
                logits = self.head.gather(logits)
            output["logits"] = logits.transpose(0, 1).contiguous()
        return output

    def _apply_mtp_loss(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        labels: torch.Tensor,
        loss_mask: torch.Tensor | None,
        packed_seq_params,
        temperature: float,
        use_fused_kernels: bool,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.mtp is None or not self.mtp_enable_train:
            return None
        if input_ids is None:
            raise ValueError("MTP training requires input_ids.")
        if loss_mask is None:
            loss_mask = torch.ones_like(labels, dtype=torch.float32)
        else:
            loss_mask = loss_mask.to(dtype=torch.float32)

        mtp_hidden_states = self.mtp(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            packed_seq_params=packed_seq_params,
        )
        mtp_labels = labels.clone()
        mtp_loss_mask = loss_mask.clone()
        mtp_loss_values = []
        for mtp_hidden in mtp_hidden_states:
            mtp_labels, _ = roll_mtp_tensor_left(
                mtp_labels, packed_seq_params=packed_seq_params, dims=-1
            )
            mtp_loss_mask, num_tokens = roll_mtp_tensor_left(
                mtp_loss_mask, packed_seq_params=packed_seq_params, dims=-1
            )
            labels_sb = mtp_labels.transpose(0, 1).contiguous()
            mask_sb = mtp_loss_mask.transpose(0, 1).contiguous()
            if use_fused_kernels:
                mtp_hidden_full = gather_from_sequence_parallel(mtp_hidden, self.ps)
                log_probs, _entropy = linear_cross_entropy(
                    mtp_hidden_full,
                    self._head_weight_for_fused_ce(mtp_hidden_full),
                    labels_sb,
                    temperature,
                    self.ps.tp_group,
                )
                token_loss = -log_probs
            else:
                assert self.head is not None
                logits = self.head(mtp_hidden)
                if temperature != 1.0:
                    logits = logits / temperature
                token_loss = vocab_parallel_cross_entropy(logits, labels_sb, self.ps.tp_group)
            token_loss = token_loss * mask_sb.to(dtype=token_loss.dtype)
            num_tokens = num_tokens.to(dtype=token_loss.dtype).clamp_min(1.0)
            mtp_loss_values.append(token_loss.sum() / num_tokens)
            mtp_loss_scale = self.mtp_loss_scaling_factor / max(len(mtp_hidden_states), 1)
            hidden_states = MTPLossAutoScaler.apply(
                hidden_states, mtp_loss_scale * token_loss / num_tokens
            )

        if not mtp_loss_values:
            return None
        return (
            hidden_states,
            torch.stack([loss.detach().float() for loss in mtp_loss_values]).mean(),
        )

    def _head_weight_for_fused_ce(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.head is not None
        weight = self.head.col.linear.weight
        return (
            weight if weight.dtype == hidden_states.dtype else weight.to(dtype=hidden_states.dtype)
        )


def _iter_auto_model_class_refs(hf_config) -> list[str]:
    auto_map = getattr(hf_config, "auto_map", None) or {}
    preferred_keys = (
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModel",
    )
    refs: list[str] = []
    for key in preferred_keys:
        ref = auto_map.get(key)
        if isinstance(ref, (list, tuple)):
            ref = ref[0] if ref else None
        if isinstance(ref, str) and ref not in refs:
            refs.append(ref)
    return refs


def _resolve_hf_vision_cls(hf_config, hf_path: str) -> type:
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except ImportError as exc:
        raise ImportError(
            "mount_vision_model=True requires transformers with dynamic module support."
        ) from exc

    errors: list[str] = []
    for class_ref in _iter_auto_model_class_refs(hf_config):
        try:
            model_cls = get_class_from_dynamic_module(class_ref, hf_path, trust_remote_code=True)
        except Exception as exc:
            errors.append(f"{class_ref}: {exc}")
            continue
        vision_cls = getattr(model_cls, "HfVisionClass", None)
        if vision_cls is not None:
            return vision_cls
        errors.append(f"{class_ref}: missing HfVisionClass")
    detail = "; ".join(errors) if errors else "config auto_map has no supported AutoModel entry"
    raise RuntimeError(f"Cannot resolve native HF vision class for Qwen3.5: {detail}.")


def _hook_fp32_rotary_emb(module: nn.Module) -> None:
    for submodule in module.modules():
        if hasattr(submodule, "inv_freq") and submodule.inv_freq is not None:
            submodule._inv_freq_fp32_original = submodule.inv_freq.detach().clone().float()

            def _hook(mod, args):
                del args
                if hasattr(mod, "_inv_freq_fp32_original"):
                    mod.inv_freq = mod._inv_freq_fp32_original.to(device=mod.inv_freq.device)

            submodule.register_forward_pre_hook(_hook)


def _hook_vision_params_avg_grad_across_tp(module: nn.Module) -> None:
    for param in module.parameters(recurse=True):
        param.average_gradients_across_tp_domain = True  # type: ignore[assignment]


def _build_native_vision_model(hf_path: str) -> nn.Module | None:
    if not hf_path:
        raise ValueError("mount_vision_model requires hf_path.")
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise ImportError("mount_vision_model=True requires transformers.") from exc

    hf_config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
    vision_config = getattr(hf_config, "vision_config", None)
    if vision_config is None:
        # Text-only checkpoint (no vision tower): nothing to mount -> graceful skip.
        return None
    auto_map = getattr(hf_config, "auto_map", None) or {}
    if auto_map:
        # Remote-code checkpoint: resolve the vision class from its dynamic module.
        hf_vision_cls = _resolve_hf_vision_cls(hf_config, hf_path)
        if hasattr(hf_vision_cls, "_from_config"):
            vision = hf_vision_cls._from_config(vision_config)
        else:
            vision = hf_vision_cls(vision_config)
    else:
        # Native-transformers checkpoint (auto_map=None, e.g. Qwen3.5-35B-A3B): build the vision tower
        # directly from its vision_config via the native AutoModel registry (no remote code).
        from transformers import AutoModel

        vision = AutoModel.from_config(vision_config)
    _hook_fp32_rotary_emb(vision)
    _hook_vision_params_avg_grad_across_tp(vision)
    return vision.to(torch.bfloat16)


__all__ = [
    "DenseMLP",
    "FullAttention",
    "GatedDeltaNet",
    "MoELayer",
    "MTPLossAutoScaler",
    "Qwen35MRoPE",
    "Qwen35Layer",
    "Qwen35Model",
    "SharedExpert",
    "_split_grouped_qkvg",
]
