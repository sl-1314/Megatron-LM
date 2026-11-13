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
  the analytic fp32 Gemma-RMSNorm gradient of the pair-norm contract plus
  the all-reduce transpose (AR backward = AR), which is a tolerance-grade
  surrogate (same pattern as every other 0-diff contract).
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
    s = fp32(AR(x)) + fp32(residual), with dx additionally passed through
    the all-reduce transpose (AR backward = AR, since AR is a sum-replicate).
    """

    @staticmethod
    def forward(ctx, x, residual, weight, eps):
        op = _import_op()
        assert op is not None, f"vllm fused AR+norm unavailable: {_ERR}"
        # The kernel writes the new residual into the hidden-states buffer
        # in place — operate on clones so the caller's tensors (and the
        # autograd graph) stay intact.
        x2 = x.reshape(-1, x.shape[-1]).contiguous().clone()
        res2 = residual.reshape(-1, residual.shape[-1]).contiguous().clone()
        w2 = weight.contiguous()
        norm = SimpleNamespace(weight=w2, variance_epsilon=eps)
        cfg = _STATE["cfg"]
        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(cfg):
            y, res_out = op(x2, res2, norm)
        ctx.save_for_backward(x2, res2, w2)
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
        # AR transpose: each rank's x_partial receives the (replicated) sum
        # of all ranks' ds contributions.
        dx = ds.contiguous()
        dist.all_reduce(dx, group=group)
        return (
            dx.to(x0.dtype).view(ctx.x_shape),
            ds.to(r0.dtype).view(ctx.res_shape),
            dw.to(weight.dtype),
            None,
        )


def vllm_fused_ar_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deferred-AR norm sync point: (y, new_residual) per rank.

    ``x`` is the LOCAL partial (un-reduced row-parallel output); the op
    all-reduces internally.  See the module docstring for the contract.
    """
    if not _STATE["inited"]:  # pragma: no cover - defensive
        raise RuntimeError("ensure_vllm_tp_state must run before vllm_fused_ar_norm")
    y, res_out = _VllmFusedARNorm.apply(x, residual, weight, eps)
    return y, res_out


class _TPColMatmul(torch.autograd.Function):
    """Column-parallel GEMM on a REPLICATED input (TP>1 contract).

    Forward: ``y = x @ W_r^T`` with the per-rank weight shard — exactly the
    engine's column GEMM shape (bitwise; the caller hands in the loader
    shard).  No communication.

    Backward: ``dW`` is the shard's local gradient (correct — only this
    rank's partial output flowed through it); ``dx`` is all-reduced over the
    TP group, because the replicated input feeds every rank's shard and the
    true gradient is the sum of all contributions.  (The output's own
    all-reduce lives in the FOLLOWING norm's ``_VllmFusedARNorm`` backward,
    which returns the AR-transposed gradient to this op.)
    """

    @staticmethod
    def forward(ctx, x, weight, group):
        ctx.group = group
        ctx.x_shape = x.shape
        x2 = x.reshape(-1, x.shape[-1])
        y = torch.nn.functional.linear(x2, weight)
        ctx.save_for_backward(x2, weight)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        dy2 = dy.reshape(-1, dy.shape[-1])
        dw = dy2.t().matmul(x)
        dx = dy2.matmul(weight)
        tp_all_reduce_(dx, ctx.group)
        return dx.view(ctx.x_shape), dw, None


def tp_col_linear(x: torch.Tensor, weight: torch.Tensor, group):
    """Column GEMM on a replicated input; see ``_TPColMatmul``."""
    return _TPColMatmul.apply(x, weight, group)
