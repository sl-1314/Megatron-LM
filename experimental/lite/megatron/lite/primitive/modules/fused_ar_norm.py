"""0-diff (qwen35-0diff-epcp): vLLM fused all-reduce + Gemma RMSNorm contract.

The vLLM engine's plain-TP path (the bitwise alignment target for TP>1)
defers every row-parallel all-reduce into the FOLLOWING norm and runs it as
one flashinfer trtllm fused kernel: all_reduce(x_partial) + residual + Gemma
RMSNorm, with the AR result kept in fp32 and normed WITHOUT an intermediate
bf16 rounding (fp32_acc=True).  A decomposed NCCL AR + pair-norm is NOT
bitwise equal (verified offline, 1-2 ulp on ~38% of elements) — the contract
must drive the very same imported op the engine calls.

Module contents:
- ``ensure_vllm_tp_state``: build vLLM's distributed state (distributed
  environment + tensor-model-parallel group) inside a plain trainer process,
  so the imported op finds the group it expects.  Idempotent per process.
  On success (tp_size > 1) it also flips the module-level TP-contract flag
  that the per-module forward gates read (``tp_contract_active``).
- ``vllm_fused_ar_norm_available``: import predicate.
- ``vllm_fused_ar_norm``: the contract entry — forward is the imported
  ``fused_allreduce_gemma_rms_norm`` (bitwise by construction); backward is
  the analytic fp32 Gemma-RMSNorm gradient of the pair-norm contract with
  identity AR-transpose (each partial receives the full ds), which is a
  tolerance-grade surrogate (same pattern as every other 0-diff contract).
  The entry mirrors the engine's own dispatch: when the fused trtllm kernel
  is NOT eligible (num_tokens above the kernel's max for this tp/dtype —
  large packed training batches), it runs the engine's decomposed fallback
  (vLLM's TP all-reduce + the pair-norm contract) instead.
- ``tp_col_linear``: column-parallel GEMM on a REPLICATED input — the TP
  contract's shape for every column projection (engine GEMM shapes, bitwise
  forward; backward all-reduces dx over the TP group, dW stays local).

Empirical anchors (outputs/2026-11-10-qwen35-mlite-0diff/c10mesh):
- scratch/c10_tp_fiarnorm_probe.py: 4-rank outputs bitwise == engine TP4
  layer dumps (absmax=0).
- DESIGN.md "TP line" sections: fused-op semantics and shard rules.
"""

from __future__ import annotations

import os
import threading
from types import SimpleNamespace

import torch

_LOCK = threading.Lock()
_STATE: dict = {"inited": False, "cfg": None, "tp_active": False}

_ERR = ""


def tp_contract_active() -> bool:
    """Whether the TP>1 vLLM contract is armed in this process.

    Set by ``ensure_vllm_tp_state`` (which the model constructor calls at
    TP>1 with contracts enabled).  Module forward gates (GDN / GQA / MoE /
    head) read this together with ``ps.tp_size > 1`` to pick the plain-TP
    deferred-AR contract path; at TP=1 or in native mode it stays False and
    every module takes its existing (TP1-validated) path.
    """
    return _STATE["tp_active"]


def tp_all_reduce_(grad: torch.Tensor, group) -> torch.Tensor:
    """In-place TP all-reduce of a gradient (no-op for 1-rank groups)."""
    import torch.distributed as dist

    if group is not None and dist.is_available() and dist.is_initialized():
        if dist.get_world_size(group) > 1:
            dist.all_reduce(grad, group=group)
    return grad


def _import_op():
    global _ERR
    try:
        from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
            fused_allreduce_gemma_rms_norm,
        )

        return fused_allreduce_gemma_rms_norm
    except Exception as e:  # pragma: no cover - env-dependent
        _ERR = str(e)
        return None


def vllm_fused_ar_norm_available() -> bool:
    return _import_op() is not None


def ensure_vllm_tp_state(tp_size: int, rank: int, local_rank: int) -> bool:
    """Initialize vLLM's TP distributed state in this (trainer) process.

    Mirrors scratch/c10_tp_fiarnorm_probe.py exactly: both
    ``init_distributed_environment`` and ``initialize_model_parallel`` must
    run inside a ``set_current_vllm_config`` context, using a bare
    ``VllmConfig()`` (no ``get_default()``).  Safe to call multiple times —
    the second call is a no-op.  Returns True when the state is ready.
    """
    with _LOCK:
        if _STATE["inited"]:
            return True
        try:
            from vllm.config import VllmConfig, set_current_vllm_config
            from vllm.distributed import (
                init_distributed_environment,
                initialize_model_parallel,
            )
        except Exception as e:  # pragma: no cover - env-dependent
            global _ERR
            _ERR = str(e)
            return False
        if not torch.distributed.is_initialized():
            _ERR = "torch.distributed is not initialized"
            return False
        # A private TCP store for vLLM's own process group, distinct from the
        # trainer's MASTER_ADDR/PORT (the probe recipe).
        port = int(os.environ.get("MLITE_VLLM_TP_INIT_PORT", "29701"))
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(
                world_size=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                local_rank=local_rank,
                distributed_init_method=f"tcp://127.0.0.1:{port}",
                backend="nccl",
            )
            initialize_model_parallel(tensor_model_parallel_size=tp_size)
        # Cache one config instance for the per-call context (construction
        # is not free; the op itself reads the parallel state, not the cfg,
        # but the probe kept the context around the call too).
        _STATE["cfg"] = VllmConfig()
        _STATE["inited"] = True
        if tp_size > 1:
            _STATE["tp_active"] = True
        return True


class _VllmFusedARNorm(torch.autograd.Function):
    """fused_allreduce_gemma_rms_norm with analytic fp32 backward.

    Forward (bitwise contract): y, res_out = op(x_partial, residual, norm)
    where the kernel computes s = AR(x_partial) + residual in fp32 and
    y = gemma_rmsnorm(s) — the engine's deferred-AR sync point.

    Backward (surrogate): the same analytic Gemma-RMSNorm gradient as the
    pair-norm contract (_VllmGemmaFusedAddRmsNorm), evaluated on
    s = fp32(AR(x)) + fp32(residual).  dx is the IDENTITY (no AR): s is the
    sum-replicate of the partials, so every rank's x_partial receives the
    same full ds (Megatron g-op semantics — the plain-TP loss is computed
    on the replicated stream, and an AR of the already-identical ds would
    scale gradients by tp_size per layer boundary).
    """

    @staticmethod
    def forward(ctx, x, residual, weight, eps):
        op = _import_op()
        assert op is not None, f"vllm fused AR+norm unavailable: {_ERR}"
        # The trtllm kernel writes the NEW RESIDUAL (all_reduce(x_partial) +
        # residual) INTO the hidden-states buffer in place and returns
        # (norm_out, hidden_states) — the returned res_out IS the mutated x
        # buffer (vllm's own docstring: "the kernel writes the new residual
        # ... into the hidden_states buffer ... leaving `residual`
        # untouched").  Operate on clones so the caller's tensors stay
        # intact, and save PRE-OP clones: save_for_backward stores tensor
        # OBJECTS, so saving x2 after the call would record the overwritten
        # buffer (s, not the per-rank partial) and the backward would
        # all-reduce the already-summed s (~5x the true sum, rstd off by the
        # same factor — the C10 g0 fused-path grad attenuation bug).
        x2 = x.reshape(-1, x.shape[-1]).contiguous().clone()
        res2 = residual.reshape(-1, residual.shape[-1]).contiguous().clone()
        x_pre = x2.clone()
        res_pre = res2.clone()
        w2 = weight.contiguous()
        norm = SimpleNamespace(weight=w2, variance_epsilon=eps)
        cfg = _STATE["cfg"]
        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(cfg):
            y, res_out = op(x2, res2, norm)
        ctx.save_for_backward(x_pre, res_pre, w2)
        ctx.eps = eps
        ctx.x_shape = x.shape
        ctx.res_shape = residual.shape
        return y.view(x.shape), res_out.view(residual.shape)

    @staticmethod
    def backward(ctx, dy, dres):
        x0, r0, weight = ctx.saved_tensors
        import torch.distributed as dist

        from vllm.distributed import get_tp_group

        group = get_tp_group().device_group
        n = x0.shape[-1]
        dy = dy.reshape(-1, n)
        dres = dres.reshape(-1, n)
        # Surrogate recompute of the kernel's fp32 s (torch AR + add; the
        # kernel's in-kernel fp32 sum has the same value up to fp32 rounding
        # of the AR order — tolerance-grade for gradients only).
        ar = x0.float()
        dist.all_reduce(ar, group=group)
        s = ar + r0.float()
        g = weight.float() + 1.0
        var = (s * s).sum(-1, keepdim=True) / n
        rstd = torch.rsqrt(var + ctx.eps)
        # Same structure as _VllmGemmaFusedAddRmsNorm.backward: dres passes
        # straight through (res_out is the identity of s); no rstd folding.
        dyf = dy.float()
        A = dyf * rstd * g
        t = (dyf * (s * g)).sum(-1, keepdim=True)
        ds = A - (rstd * rstd * rstd) * (s / n) * t + dres.float()
        dw = (dyf * (s * rstd)).sum(0)
        # s = AR(x_partial) + residual: the sum-replicate's transpose gives
        # EVERY rank's x_partial the same full ds (identity, no AR — see the
        # class docstring; an AR here would 4x-scale grads per boundary).
        dx = ds
        return (
            dx.to(x0.dtype).view(ctx.x_shape),
            ds.to(r0.dtype).view(ctx.res_shape),
            dw.to(weight.dtype),
            None,
        )


def _import_eligibility():
    """vLLM's own fused-path predicate (private, same module as the op)."""
    try:
        from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
            _can_use_flashinfer,
        )

        return _can_use_flashinfer
    except Exception:  # pragma: no cover - env-dependent
        return None


class _VllmARIdentity(torch.autograd.Function):
    """vLLM's TP all-reduce (out-of-place, custom-AR-or-NCCL), identity backward.

    The g-op transpose: the forward maps the local partial to the replicated
    sum, so every rank's partial receives the SAME full gradient (no AR —
    an AR of the already-identical ds would scale grads by tp_size).
    """

    @staticmethod
    def forward(ctx, x, ar_fn):
        return ar_fn(x)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def vllm_fused_ar_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deferred-AR norm sync point: (y, new_residual) per rank.

    ``x`` is the LOCAL partial (un-reduced row-parallel output).  Mirrors
    the engine's own dispatch in ``fused_allreduce_gemma_rms_norm``: the
    fused trtllm AR+norm kernel when eligible (num_tokens <= the kernel's
    max_token_num for this tp/dtype), else the engine's decomposed fallback
    (tensor_model_parallel_all_reduce + GemmaRMSNorm pair).  Large packed
    training batches take the fallback — exactly like the engine.
    """
    if not _STATE["inited"]:  # pragma: no cover - defensive
        raise RuntimeError("ensure_vllm_tp_state must run before vllm_fused_ar_norm")
    x_shape, res_shape = x.shape, residual.shape
    x2 = x.reshape(-1, x_shape[-1])
    ok = False
    elig = _import_eligibility()
    if elig is not None:
        try:
            from vllm.distributed import get_tensor_model_parallel_world_size

            tp_size = get_tensor_model_parallel_world_size()
            if tp_size > 1:
                ok = bool(elig(x2.contiguous(), tp_size)[0])
        except Exception:  # pragma: no cover - env-dependent
            ok = False
    if ok:
        return _VllmFusedARNorm.apply(x, residual, weight, eps)
    # Engine decomposed fallback: AR (vLLM's own dispatch, out-of-place,
    # identity backward) + the C7-validated pair-norm contract.  The engine
    # model is the platform overlay vllm/models/qwen3_5/nvidia/model.py:
    # fused_allreduce_gemma_rms_norm's fallback runs ``norm(reduced,
    # residual)`` on ITS Qwen3_5RMSNorm, whose forward is flashinfer's
    # gemma_fused_add_rmsnorm — the same kernel the pair-norm contract
    # drives (the generic ir-native GemmaRMSNorm is NOT what runs here).
    from vllm.distributed import tensor_model_parallel_all_reduce

    from megatron.lite.primitive.modules.gated_delta_net import _vllm_pair_norm

    res2 = residual.reshape(-1, res_shape[-1])
    ar = _VllmARIdentity.apply(x2.contiguous(), tensor_model_parallel_all_reduce)
    y2, res_out2 = _vllm_pair_norm(ar, res2, weight, eps)
    return y2.view(x_shape), res_out2.view(res_shape)


class _TPColMatmul(torch.autograd.Function):
    """Column-parallel GEMM on a REPLICATED input (TP>1 contract).

    Forward: ``y = x @ W_r^T`` with the per-rank weight shard — exactly the
    engine's column GEMM shape (bitwise; the caller hands in the loader
    shard).  No communication.

    Backward: ``dW`` is the shard's local gradient (correct — only this
    rank's partial output flowed through it); ``dx`` is all-reduced over the
    TP group, because the replicated input feeds every rank's shard and the
    true gradient is the sum of all contributions (Megatron f-op).  The
    row-parallel partials this op's chain eventually produces are summed by
    the FOLLOWING norm's forward AR, whose backward is the identity.

    ``kv_replica_reduce`` (GQA kv-head replication, num_kv_heads < tp):
    ranks sharing a kv head (r // replicas) hold the SAME logical k/v weight
    rows, but each rank's local attention only backprops its own q heads —
    its k/v-row dw is a PARTIAL.  The backward all-gathers the tail row
    block [kv_start:] and sums it across the rank's replica peers so every
    replica holds the total dw (keeps replicas bitwise in sync through
    optimizer steps).  Q/gate rows are rank-unique and stay local.
    """

    @staticmethod
    def forward(ctx, x, weight, group, kv_start=-1, replicas=1, tp_rank=0, tp_size=1):
        ctx.group = group
        ctx.x_shape = x.shape
        ctx.kv_start = kv_start
        ctx.replicas = replicas
        ctx.tp_rank = tp_rank
        ctx.tp_size = tp_size
        x2 = x.reshape(-1, x.shape[-1])
        y = torch.nn.functional.linear(x2, weight)
        ctx.save_for_backward(x2, weight)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        dy2 = dy.reshape(-1, dy.shape[-1])
        dw = dy2.t().matmul(x)
        if ctx.kv_start >= 0 and ctx.replicas > 1:
            import torch.distributed as dist

            tail = dw[ctx.kv_start :].contiguous()
            bufs = [torch.empty_like(tail) for _ in range(ctx.tp_size)]
            dist.all_gather(bufs, tail, group=ctx.group)
            my_group = ctx.tp_rank // ctx.replicas
            total = sum(
                b for r, b in enumerate(bufs) if r // ctx.replicas == my_group
            )
            dw = torch.cat([dw[: ctx.kv_start], total.to(dw.dtype)], dim=0)
        dx = dy2.matmul(weight)
        tp_all_reduce_(dx, ctx.group)
        return dx.view(ctx.x_shape), dw, None, None, None, None, None


def tp_col_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    group,
    kv_replica_reduce: tuple[int, int, int, int] | None = None,
):
    """Column GEMM on a replicated input; see ``_TPColMatmul``.

    ``kv_replica_reduce`` = (kv_row_start, replicas, tp_rank, tp_size) arms
    the backward's replica-peer summation of the k/v row block of dw (GQA
    kv-head replication; see the class docstring).
    """
    if kv_replica_reduce is None:
        return _TPColMatmul.apply(x, weight, group)
    kv_start, replicas, tp_rank, tp_size = kv_replica_reduce
    return _TPColMatmul.apply(
        x, weight, group, kv_start, replicas, tp_rank, tp_size
    )
