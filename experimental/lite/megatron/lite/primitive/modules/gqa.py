# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Grouped Query Attention + Rotary Embedding.

Model-agnostic: takes explicit params instead of model-specific config.
Supports sequence parallel, context parallel, and THD (packed sequences).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from megatron.lite.primitive import transformer_engine as te
from megatron.lite.primitive.modules.gqa_utils import (
    split_grouped_qkvg,
    split_grouped_qkvg_for_tp,
)
from megatron.lite.primitive.modules.lora import LinearLoRA, LoraConfig, normalize_lora_config
from megatron.lite.primitive.modules.fused_ar_norm import (
    tp_all_reduce_,
    tp_col_linear,
    tp_contract_active,
)
from megatron.lite.primitive.modules.mrope import MultimodalRotaryEmbedding
from megatron.lite.primitive.parallel import (
    ColumnParallelLinear,
    ParallelState,
    RowParallelLinear,
    all_gather_last_dim_with_grad_reduce,
)
from megatron.lite.primitive.utils import ensure_divisible
from megatron.lite.primitive.utils.rope import _apply_rotary_pos_emb_bshd, _apply_rotary_pos_emb_thd
from megatron.lite.primitive.utils.rotary import RotaryEmbedding

# Whitelist of MC PackedSeqParams fields accepted by TE DotProductAttention.forward().
# MC-only fields (local_cp_size, cp_group, total_tokens, seq_idx) are excluded.
# Mirror the TE DotProductAttention packed-sequence argument whitelist.
_KEPT_PSP_FIELDS = (
    "qkv_format",
    "cu_seqlens_q",
    "cu_seqlens_kv",
    "cu_seqlens_q_padded",
    "cu_seqlens_kv_padded",
    "max_seqlen_q",
    "max_seqlen_kv",
)


# 0-diff (qwen35-0diff-lite): vLLM FA-path kernel contracts for Qwen3.5
# --------------------------------------------------------------------------------
#
# The rollout side computes each full-attention layer as:
#   qkv GEMM -> fused_qk_rmsnorm_rope_gate (Triton: split + Gemma QK-RMSNorm +
#   partial NeoX MRoPE + gate copy in ONE kernel) -> FA3 varlen (paged in the
#   engine) -> fused_sigmoid_gate_mul (Triton) -> o_proj GEMM.
# The contract imports and drives the engine's own kernel functions, so the
# Triton/CuTe binaries are identical -- no transcription risk.  Verified
# module-by-module bitwise against in-engine captures on the C3 mixed
# [L,L,L,F] proxy (see c3_fa_replay.py): every stage including the whole
# decoder layer h_out matches bit-exactly.
#
# FA3 layout note: the engine's prefill attention runs the PAGED path
# (k/v = KV-cache pool + block_table + scheduler_metadata).  A/B on the
# captures showed the DENSE varlen path (cu_seqlens_k, no block_table) and any
# page size / scheduler_metadata=None all reproduce the engine output
# bit-exactly for this causal prefill shape, so the contract uses the dense
# THD call.  The engine runs the fused kernels with norm_beta=1.0 (Gemma
# (w+1)); mlite's zero-centered RMSNorm weights store the same raw values.
#
# ``positions`` must be (3, T) MRoPE rows (all rows equal for text), matching
# the engine's 2D path so the kernel's HAS_MROPE specialization is the same.
# The cos/sin cache is built on the same device as q (GPU libdevice cos/sin;
# a CPU-built cache differs from the engine's in ~68/16M rows by 1 ulp).
#
# ``fused_qk_rmsnorm_rope_gate``, ``flash_attn_varlen_func`` and
# ``fused_sigmoid_gate_mul`` are inference-only, so gradients go through
# exact fp32 torch recomputes (C6 tolerance-validated), same pattern as
# _VllmFusedMoe.
#
# Validity window: bf16, TP=1, THD (packed_seq_params given), output_gate,
# mrope_section, qkv_layout="mcore".  Other configurations fall back to the
# TE path.

_VLLM_FA_IMPORT_ERROR: str | None = ""


def _vllm_fa_fns():
    """Lazily import vLLM's FA-path kernel entry points; None when unavailable."""
    global _VLLM_FA_IMPORT_ERROR
    if _VLLM_FA_IMPORT_ERROR != "":
        return None
    try:
        from vllm.kernels.triton.fused_sigmoid_gate_mul import fused_sigmoid_gate_mul
        from vllm.model_executor.layers.fused_qk_norm_rope import (
            fused_qk_rmsnorm_rope_gate,
        )
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        return fused_qk_rmsnorm_rope_gate, flash_attn_varlen_func, fused_sigmoid_gate_mul
    except Exception as e:  # pragma: no cover - env-dependent
        _VLLM_FA_IMPORT_ERROR = f"{type(e).__name__}: {e}"
        return None


def vllm_fa_available() -> bool:
    return _vllm_fa_fns() is not None


def vllm_fa_cos_sin_cache(
    rotary_dim: int, theta: float, max_position: int, device: torch.device
) -> torch.Tensor:
    """Bit-exact vLLM rotary cos/sin cache (bf16, [max_position, rotary_dim]).

    Same construction as vLLM's RotaryEmbedding._compute_cos_sin_cache, run on
    ``device`` (GPU libdevice trans; CPU libm differs from the engine's cache
    in a few 1-ulp rows at large positions).
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
    )
    t = torch.arange(max_position, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)


def _torch_qk_norm_rope_gate(
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
    mrope_section: list[int] | None,
    norm_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable fp32 recompute of the fused rope kernel's math (backward)."""
    n_tok = q_gate.shape[0]
    qg = q_gate.view(n_tok, num_q_heads, 2, head_dim)
    q, gate = qg[:, :, 0].float(), qg[:, :, 1].float()
    kf = k.view(n_tok, num_kv_heads, head_dim).float()

    def _gemma_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # Kernel rounds the normed head to the storage dtype before RoPE;
        # autograd passes through the cast (straight-through), keeping the
        # gradient signal while matching the forward's quantization point.
        # NOTE: w is a flat (head_dim,) vector SHARED across heads (the
        # kernel indexes w_ptr by head offsets only), like the HF q_norm.
        var = x.pow(2).mean(dim=-1, keepdim=True)
        y = x * torch.rsqrt(var + eps) * (w.float() + norm_beta)
        return y.to(q_gate.dtype).float()

    q = _gemma_norm(q, q_weight)
    kf = _gemma_norm(kf, k_weight)

    half = rotary_dim // 2
    cos = cos_sin_cache[:, :half].float()
    sin = cos_sin_cache[:, half:rotary_dim].float()
    if mrope_section is not None and positions.dim() == 2:
        # Interleaved row pattern, exactly the kernel's:
        # j%3==1 -> row 1 when j < 3*section[1], j%3==2 -> row 2 when
        # j < 3*section[2], else row 0 (NOT contiguous [s0|s1|s2] blocks).
        jj = torch.arange(half, device=q_gate.device)
        row = torch.zeros(half, dtype=torch.long, device=q_gate.device)
        row = torch.where(
            (jj % 3 == 1) & (jj < 3 * mrope_section[1]),
            torch.ones_like(row),
            row,
        )
        row = torch.where(
            (jj % 3 == 2) & (jj < 3 * mrope_section[2]),
            torch.full_like(row, 2),
            row,
        )
        pos_per_freq = positions.index_select(0, row).transpose(0, 1)  # (T, half)
    else:
        pos_per_freq = positions.reshape(-1)[:n_tok].view(n_tok, 1).expand(n_tok, half)
    # cosf[t, j] = cache[pos[t, j], j] -- advanced indexing with the row index
    # alone would append the cache's column dim (giving (T, half, half)).
    j = torch.arange(half, device=q_gate.device)
    cosf = cos[pos_per_freq, j][:, None]  # (n_tok, 1, half), broadcasts over heads
    sinf = sin[pos_per_freq, j][:, None]

    def _rope(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :half], x[..., half:rotary_dim]
        rest = x[..., rotary_dim:]
        r1 = x1 * cosf - x2 * sinf
        r2 = x2 * cosf + x1 * sinf
        return torch.cat([r1, r2, rest], dim=-1)

    q = _rope(q).to(q_gate.dtype).reshape(n_tok, num_q_heads * head_dim)
    k_out = _rope(kf).to(k.dtype).reshape(n_tok, num_kv_heads * head_dim)
    gate_out = gate.to(q_gate.dtype).reshape(n_tok, num_q_heads * head_dim)
    return q, k_out, gate_out


class _VllmQkNormRopeGate(torch.autograd.Function):
    """fused_qk_rmsnorm_rope_gate: bitwise vLLM forward, exact torch backward."""

    @staticmethod
    def forward(ctx, q_gate, k, q_weight, k_weight, cos_sin, positions, eps,
                num_q_heads, num_kv_heads, head_dim, rotary_dim, mrope_section,
                norm_beta, tp_group=None):
        fn = _vllm_fa_fns()[0]
        q, k_out, gate = fn(
            q_gate, k, q_weight, k_weight, cos_sin, positions, eps,
            num_q_heads, num_kv_heads, head_dim, rotary_dim,
            norm_beta=norm_beta, mrope_section=mrope_section,
        )
        ctx.save_for_backward(q_gate, k, q_weight, k_weight, cos_sin, positions)
        ctx.meta = (eps, num_q_heads, num_kv_heads, head_dim, rotary_dim,
                    mrope_section, norm_beta)
        ctx.tp_group = tp_group
        return q, k_out, gate

    @staticmethod
    def backward(ctx, dq, dk, dgate):
        q_gate, k, q_weight, k_weight, cos_sin, positions = ctx.saved_tensors
        eps, nq, nkv, hd, rot_dim, mrope_section, norm_beta = ctx.meta
        with torch.enable_grad():
            qg = q_gate.detach().requires_grad_(True)
            k_ = k.detach().requires_grad_(True)
            qw = q_weight.detach().requires_grad_(True)
            kw = k_weight.detach().requires_grad_(True)
            q, k_out, gate = _torch_qk_norm_rope_gate(
                qg, k_, qw, kw, cos_sin, positions, eps, nq, nkv, hd,
                rot_dim, mrope_section, norm_beta,
            )
            grads = torch.autograd.grad(
                (q, k_out, gate), (qg, k_, qw, kw),
                (dq.to(q.dtype), dk.to(k_out.dtype), dgate.to(gate.dtype)),
            )
        # 0-diff (qwen35-0diff-epcp): at TP>1 the flat q/k norm weights are
        # replicated but only the LOCAL heads flow through this rank — sum
        # the shard contributions across TP (contract only).
        tp_all_reduce_(grads[2], ctx.tp_group)
        tp_all_reduce_(grads[3], ctx.tp_group)
        return grads[0], grads[1], grads[2], grads[3], None, None, None, None, None, None, None, None, None, None


def _torch_varlen_attn(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale):
    """Differentiable fp32 recompute of causal varlen attention (backward).

    GQA: q heads are grouped per kv head (nq must divide evenly by nkv).
    """
    n_tok, nq, hd = q.shape
    nkv = k.shape[1]
    qpg = nq // nkv
    out = torch.empty_like(q)
    for i in range(cu_seqlens_q.shape[0] - 1):
        qlo, qhi = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        klo, khi = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
        qs = q[qlo:qhi].float().view(-1, nkv, qpg, hd)
        ks = k[klo:khi].float()
        vs = v[klo:khi].float()
        att = torch.einsum("tghd,sgd->tghs", qs, ks) * softmax_scale
        t_q, t_k = qs.size(0), ks.size(0)
        # tril on the 2D (t_q, t_k) matrix, then broadcast over the head dims;
        # torch.tril on a (T,1,T) tensor would batch dim-0 and keep only s==0.
        causal = torch.ones(t_q, t_k, device=q.device, dtype=torch.bool).tril()[:, None, None]
        att = att.masked_fill(~causal, float("-inf"))
        p = torch.softmax(att, dim=-1)
        o = torch.einsum("tghs,sgd->tghd", p, vs)
        out[qlo:qhi] = o.reshape(-1, nq, hd).to(q.dtype)
    return out


class _VllmFA3Varlen(torch.autograd.Function):
    """flash_attn_varlen_func (dense THD, fa_version=3): bitwise vLLM forward,
    exact fp32 torch backward."""

    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                max_seqlen_k, softmax_scale):
        fn = _vllm_fa_fns()[1]
        out = fn(
            q, k, v, max_seqlen_q=max_seqlen_q, cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k, cu_seqlens_k=cu_seqlens_k,
            softmax_scale=softmax_scale, causal=True, window_size=(-1, -1),
            softcap=0.0, fa_version=3,
        )
        ctx.save_for_backward(q, k, v, cu_seqlens_q, cu_seqlens_k)
        ctx.softmax_scale = softmax_scale
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, cu_q, cu_k = ctx.saved_tensors
        with torch.enable_grad():
            q_ = q.detach().requires_grad_(True)
            k_ = k.detach().requires_grad_(True)
            v_ = v.detach().requires_grad_(True)
            out = _torch_varlen_attn(q_, k_, v_, cu_q, cu_k, ctx.softmax_scale)
            grads = torch.autograd.grad(out, (q_, k_, v_), dout.to(out.dtype))
        return grads[0], grads[1], grads[2], None, None, None, None, None


class _VllmSigmoidGateMul(torch.autograd.Function):
    """fused_sigmoid_gate_mul: bitwise vLLM forward, exact fp32 backward."""

    @staticmethod
    def forward(ctx, x, gate):
        fn = _vllm_fa_fns()[2]
        out = fn(x, gate)
        ctx.save_for_backward(x, gate)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, gate = ctx.saved_tensors
        with torch.enable_grad():
            x_ = x.detach().float().requires_grad_(True)
            g_ = gate.detach().float().requires_grad_(True)
            out = x_ * torch.sigmoid(g_)
            grads = torch.autograd.grad(out, (x_, g_), dout.float())
        return grads[0].to(x.dtype), grads[1].to(gate.dtype)


class GQAttention(nn.Module):
    """Grouped Query Attention with TE DotProductAttention.

    Model-agnostic: all architecture params passed explicitly.
    """

    _cp_stream: torch.cuda.Stream | None = None

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        ps: ParallelState,
        *,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
        rotary_percent: float = 1.0,
        use_thd: bool = False,
        output_gate: bool = False,
        use_fp32_rope: bool = False,
        zero_centered_gamma: bool = False,
        qkv_layout: str = "flat",
        lora_config: LoraConfig | dict | None = None,
        mrope_section: list[int] | None = None,
        vllm_fa_contract: bool = False,
        max_position_embeddings: int | None = None,
    ):
        super().__init__()
        self.num_heads_local = ensure_divisible(num_attention_heads, ps.tp_size)
        self._replicate_kv = num_key_value_heads < ps.tp_size
        if self._replicate_kv:
            ensure_divisible(ps.tp_size, num_key_value_heads)
            self.num_kv_heads_local = 1
        else:
            self.num_kv_heads_local = ensure_divisible(num_key_value_heads, ps.tp_size)
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.ps = ps
        self._output_gate = output_gate
        self._use_fp32_rope = use_fp32_rope
        self._qkv_eps = rms_norm_eps
        self._qkv_zero_centered_gamma = zero_centered_gamma
        if qkv_layout not in {"flat", "mcore"}:
            raise ValueError(f"Unsupported qkv_layout={qkv_layout!r}")
        self._qkv_layout = qkv_layout
        self._mrope_section = list(mrope_section) if mrope_section is not None else None
        self._vllm_fa_contract = vllm_fa_contract
        self._max_position_embeddings = max_position_embeddings
        self._rope_theta = rope_theta
        self._rotary_percent = rotary_percent
        self._vllm_cos_sin: torch.Tensor | None = None
        if vllm_fa_contract:
            # Validity window of the C3 bitwise verification (see module docstring).
            if not (output_gate and qkv_layout == "mcore" and mrope_section is not None):
                raise ValueError(
                    "vllm_fa_contract requires output_gate, qkv_layout='mcore' and mrope_section."
                )
            if not use_thd:
                raise ValueError("vllm_fa_contract is THD-only (packed_seq_params path).")
            if ps.tp_size != 1 and not tp_contract_active():
                # 0-diff (qwen35-0diff-epcp): TP>1 is contract-covered only
                # after ensure_vllm_tp_state armed the plain-TP deferred-AR
                # contract (model constructor); native TP>1 keeps the raise.
                raise ValueError(
                    "vllm_fa_contract supports TP=1 only (TP>1 requires the TP contract)."
                )
            if max_position_embeddings is None:
                raise ValueError("vllm_fa_contract requires max_position_embeddings.")

        # Declaration order follows MC's `SelfAttention` submodule order
        # (linear_proj → linear_qkv → q_layernorm → k_layernorm). `named_
        # parameters()` iterates in registration order, and MC's
        # `DistributedDataParallel` partitions gradient buckets by that order.
        # Mismatched order would put bucket boundaries in different places,
        # producing different per-rank fp32 master shard layouts and
        # non-bitwise step-1 divergence.
        self.proj = RowParallelLinear(num_attention_heads * head_dim, hidden_size, ps, bias=False)
        q_cols = num_attention_heads * (2 if output_gate else 1)
        qkv_size = (q_cols + 2 * num_key_value_heads) * head_dim
        if ps.tp_size > 1 and tp_contract_active():
            # 0-diff (qwen35-0diff-epcp): TP contract — with KV head
            # replication (num_kv_heads < tp) the per-rank qkv rows are NOT
            # qkv_size/tp (the local layout is [q_local, gate_local, k, v]
            # per replicated kv head, N=2560 at 16q/2kv/tp4).  Size the
            # (bypassed) te param so the loader's per-rank shard fills it
            # exactly; at tp=1 / non-replicated tp this equals qkv_size.
            local_rows = (
                num_attention_heads // ps.tp_size * (2 if output_gate else 1)
                + 2 * self.num_kv_heads_local
            ) * head_dim
            qkv_size = local_rows * ps.tp_size
        self.qkv = ColumnParallelLinear(
            hidden_size,
            qkv_size,
            ps,
            bias=False,
            normalization="RMSNorm",
            eps=rms_norm_eps,
            zero_centered_gamma=zero_centered_gamma,
        )
        self.q_norm = te.RMSNorm(head_dim, eps=rms_norm_eps, zero_centered_gamma=zero_centered_gamma)
        self.k_norm = te.RMSNorm(head_dim, eps=rms_norm_eps, zero_centered_gamma=zero_centered_gamma)

        lora = normalize_lora_config(lora_config)
        self.qkv_lora: LinearLoRA | None = None
        self.proj_lora: LinearLoRA | None = None
        if lora.enabled and lora.targets_module("linear_qkv"):
            self.qkv_lora = LinearLoRA(
                hidden_size,
                self.qkv.local_out,
                lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
                sequence_parallel_input=self.qkv.use_sp,
                tp_group=ps.tp_group,
                rank_partition_size=ps.tp_size,
                rank_partitioned_a=ps.tp_size > 1,
                a_tensor_model_parallel=ps.tp_size > 1,
                b_tensor_model_parallel=ps.tp_size > 1,
            )
        if lora.enabled and lora.targets_module("linear_proj"):
            self.proj_lora = LinearLoRA(
                self.proj.local_in,
                hidden_size,
                lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
                tp_group=ps.tp_group,
                tp_rank=ps.tp_rank,
                sequence_parallel_scatter_output=self.proj.use_sp,
                input_parallel_reduce=ps.tp_size > 1,
                output_partition_size=ps.tp_size,
                output_partitioned_b=ps.tp_size > 1,
                a_tensor_model_parallel=ps.tp_size > 1,
                b_tensor_model_parallel=ps.tp_size > 1,
            )

        if self._mrope_section is None:
            self.rotary = RotaryEmbedding(
                kv_channels=head_dim,
                rotary_percent=rotary_percent,
                rotary_interleaved=False,
                rotary_base=int(rope_theta),
                use_cpu_initialization=False,
                cp_group=ps.cp_group if ps.cp_size > 1 else None,
            )
        else:
            self.rotary = MultimodalRotaryEmbedding(
                kv_channels=head_dim,
                rotary_percent=rotary_percent,
                rotary_base=rope_theta,
                cp_group=ps.cp_group if ps.cp_size > 1 else None,
            )
        cp_kwargs = {}
        if ps.cp_size > 1:
            if GQAttention._cp_stream is None:
                GQAttention._cp_stream = torch.cuda.Stream()
            cp_kwargs = dict(
                cp_group=ps.cp_group,
                cp_global_ranks=ps.cp_global_ranks,
                cp_stream=GQAttention._cp_stream,
            )

        self.core_attn = te.DotProductAttention(
            num_attention_heads=self.num_heads_local,
            kv_channels=head_dim,
            num_gqa_groups=self.num_kv_heads_local,
            attention_dropout=0.0,
            attn_mask_type="causal",
            qkv_format="thd" if use_thd else "sbhd",
            **cp_kwargs,
        )

    def vllm_fa_engages(self, x: torch.Tensor, packed_seq_params) -> bool:
        """Whether the vLLM FA contract path will run for this call.

        The contract path consumes the POST pair-norm input (see the module
        docstring and Qwen35Layer's FA branch); the caller must branch on this
        BEFORE pre-adding the residual, the way the legacy path requires.
        """
        return (
            self._vllm_fa_contract
            and packed_seq_params is not None
            and x.dim() == 3
            and x.dtype == torch.bfloat16
            and (self.ps.tp_size == 1 or tp_contract_active())
            and _vllm_fa_fns() is not None
        )

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor | None = None, packed_seq_params=None
    ) -> torch.Tensor:
        if self.vllm_fa_engages(x, packed_seq_params):
            return self._vllm_fa_contract_forward(x, position_ids, packed_seq_params)
        qkv = self.qkv(x)
        if self.qkv_lora is not None:
            qkv = qkv + self.qkv_lora(self._qkv_lora_input(x))
        if self._replicate_kv:
            qkv = all_gather_last_dim_with_grad_reduce(qkv, self.ps.tp_group)
        q, gate, k, v = self._split_qkv(qkv)

        is_thd = packed_seq_params is not None
        if is_thd:
            q, k, v = q.squeeze(1), k.squeeze(1), v.squeeze(1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE uses the local unfused rotate-half helpers.
        if self._use_fp32_rope:
            orig_dtype = q.dtype
            q, k = q.float(), k.float()
        if self._mrope_section is not None:
            if position_ids is None:
                raise ValueError("MRoPE attention requires position_ids.")
            # Packed THD position_ids are already CP-local after protocol.forward
            # prepares the VERL batch; avoid slicing MRoPE frequencies twice.
            freqs = self.rotary(position_ids, self._mrope_section, packed_seq=is_thd)
            if is_thd:
                q = _apply_rotary_pos_emb_bshd(q[:, None], freqs).squeeze(1)
                k = _apply_rotary_pos_emb_bshd(k[:, None], freqs).squeeze(1)
            else:
                q = _apply_rotary_pos_emb_bshd(q, freqs)
                k = _apply_rotary_pos_emb_bshd(k, freqs)
        elif is_thd:
            max_q = getattr(packed_seq_params, "max_seqlen_q", None)
            max_kv = getattr(packed_seq_params, "max_seqlen_kv", None)
            if max_q is None or max_kv is None:
                seq_len_for_rope = int(packed_seq_params.cu_seqlens_q[-1])
            else:
                seq_len_for_rope = int(max(max_q, max_kv))
            # Packed THD uses max per-sequence padded length, not total packed tokens.
            # The THD apply helper handles CP-zigzag frequency slicing per sequence.
            freqs = self.rotary(seq_len_for_rope, packed_seq=True)
            q = _apply_rotary_pos_emb_thd(
                q,
                packed_seq_params.cu_seqlens_q,
                freqs,
                rotary_interleaved=False,
                mscale=1.0,
                cp_group=self.ps.cp_group,
            )
            k = _apply_rotary_pos_emb_thd(
                k,
                packed_seq_params.cu_seqlens_kv,
                freqs,
                rotary_interleaved=False,
                mscale=1.0,
                cp_group=self.ps.cp_group,
            )
        else:
            # q is CP-zigzag pre-sliced; rotary needs FULL seq len,
            # its internal get_pos_emb_on_this_cp_rank re-slices to local len.
            local_seq_len = q.size(0)
            seq_len_for_rope = local_seq_len * self.ps.cp_size
            freqs = self.rotary(seq_len_for_rope)
            q = _apply_rotary_pos_emb_bshd(q, freqs, rotary_interleaved=False, mscale=1.0)
            k = _apply_rotary_pos_emb_bshd(k, freqs, rotary_interleaved=False, mscale=1.0)
        if self._use_fp32_rope:
            q, k = q.to(orig_dtype), k.to(orig_dtype)

        if is_thd:
            psp_kwargs = {
                k: getattr(packed_seq_params, k)
                for k in _KEPT_PSP_FIELDS
                if getattr(packed_seq_params, k, None) is not None
            }
            attn_out = self.core_attn(
                q,
                k,
                v,
                core_attention_bias_type="no_bias",
                attn_mask_type="padding_causal",
                **psp_kwargs,
            )
            attn_out = attn_out.reshape(attn_out.size(0), 1, -1)
        else:
            attn_out = self.core_attn(q, k, v, core_attention_bias_type="no_bias")
            if attn_out.dim() > x.dim():
                shape = attn_out.shape
                attn_out = attn_out.reshape(*shape[:-2], self.num_heads_local * self.head_dim)

        if gate is not None:
            gate_fp32 = gate.reshape(attn_out.shape).float().sigmoid()
            attn_out = (attn_out.float() * gate_fp32).to(attn_out.dtype)
        output = self.proj(attn_out)
        if self.proj_lora is not None:
            output = output + self.proj_lora(attn_out)
        return output

    def _vllm_fa_contract_forward(
        self, x: torch.Tensor, position_ids: torch.Tensor | None, packed_seq_params
    ) -> torch.Tensor:
        """vLLM FA-path kernel contract (bitwise vs the rollout engine).

        ``x`` must already be the post pair-norm activation (Qwen35Layer's FA
        branch applies ``_vllm_pair_norm`` with ``qkv.linear.layer_norm_weight``
        and passes the un-added residual stream through).  Layout math mirrors
        checkpoint._merge/_unmerge_full_attn_qkvg: the mcore qkv activation is
        per-kv-group [q|gate|k|v]; the vLLM kernels want per-head interleaved
        [q|gate] followed by flat k and v.  The GEMM itself runs on the mcore
        weight via plain F.linear (row permutations of a GEMM output are
        bitwise-safe: each output element is an independent dot product).
        """
        n_tok = x.shape[1] if x.shape[0] == 1 else x.shape[0]
        # THD arrives as (1, T, H) when multiple sequences are packed and
        # (T, 1, H) for a single sequence (pack_nested_thd layout); the token
        # stream itself is identical.
        x2d = x.reshape(n_tok, -1)
        nq, nkv, hd = self.num_heads_local, self.num_kv_heads_local, self.head_dim
        qpg = nq // nkv
        _tpc = self.ps.tp_size > 1 and tp_contract_active()

        if _tpc:
            # 0-diff (qwen35-0diff-epcp): TP contract — column GEMM on the
            # replicated post-norm input with the per-rank shard (engine GEMM
            # shape, bitwise); backward all-reduces dx over TP.  With kv-head
            # replication the k/v tail rows of dw are partial (each replica
            # rank sees only its own q heads) — arm the replica-peer dw sum.
            kv_reduce = None
            if self._replicate_kv:
                kv_reduce = (
                    2 * self.num_heads_local * self.head_dim,
                    self.ps.tp_size // self.num_kv_heads,
                    self.ps.tp_rank,
                    self.ps.tp_size,
                )
            qkv = tp_col_linear(
                x2d, self.qkv.linear.weight, self.ps.tp_group,
                kv_replica_reduce=kv_reduce,
            )
        else:
            qkv = torch.nn.functional.linear(x2d, self.qkv.linear.weight)
        mc = qkv.view(n_tok, nkv, 2 * qpg + 2, hd)
        q_ = mc[:, :, :qpg].reshape(n_tok, nq, hd)
        gate_ = mc[:, :, qpg : 2 * qpg].reshape(n_tok, nq, hd)
        k_ = mc[:, :, 2 * qpg].reshape(n_tok, nkv * hd)
        v_ = mc[:, :, 2 * qpg + 1].reshape(n_tok, nkv * hd)
        q_gate = torch.stack([q_, gate_], dim=2).reshape(n_tok, nq * 2 * hd)

        pos = position_ids
        if pos is None:
            raise ValueError("vllm_fa_contract requires position_ids (MRoPE).")
        if pos.dim() == 3:  # (3, 1, T) THD batch
            pos = pos[:, 0]
        if pos.dim() != 2 or pos.shape[0] != 3:
            raise ValueError(f"vllm_fa_contract requires (3, T) positions, got {tuple(pos.shape)}")

        if self._vllm_cos_sin is None or self._vllm_cos_sin.device != x2d.device:
            rotary_dim = int(hd * self._rotary_percent)
            self._vllm_cos_sin = vllm_fa_cos_sin_cache(
                rotary_dim, self._rope_theta, self._max_position_embeddings, x2d.device
            )

        # split + Gemma QK-RMSNorm + partial NeoX MRoPE + gate copy (one Triton
        # kernel).  mlite's zero-centered weights store the engine's raw
        # values; the kernel adds norm_beta itself.
        q_r, k_r, gate_out = _VllmQkNormRopeGate.apply(
            q_gate,
            k_,
            self.q_norm.weight,
            self.k_norm.weight,
            self._vllm_cos_sin,
            pos,
            self._qkv_eps,
            nq,
            nkv,
            hd,
            int(hd * self._rotary_percent),
            self._mrope_section,
            1.0,
            self.ps.tp_group if _tpc else None,
        )

        # FA3 varlen, dense THD call: cu_seqlens are the PADDED per-seq offsets
        # that match the physical (1, T) token layout; causal masking keeps
        # true-token rows bitwise unaffected by the pad tail.
        cu_q = packed_seq_params.cu_seqlens_q
        max_len = int(packed_seq_params.max_seqlen_q)
        attn_out = _VllmFA3Varlen.apply(
            q_r.view(n_tok, nq, hd),
            k_r.view(n_tok, nkv, hd),
            v_.view(n_tok, nkv, hd),
            cu_q,
            cu_q,
            max_len,
            max_len,
            hd**-0.5,
        )

        out_sg = _VllmSigmoidGateMul.apply(
            attn_out.reshape(n_tok, nq * hd), gate_out
        )
        out = torch.nn.functional.linear(out_sg, self.proj.linear.weight)
        return out.reshape(x.shape)

    def _qkv_lora_input(self, x: torch.Tensor) -> torch.Tensor:
        linear = self.qkv.linear
        if not hasattr(linear, "layer_norm_weight"):
            return x
        weight = linear.layer_norm_weight
        if self._qkv_zero_centered_gamma:
            weight = weight + 1
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_norm = x.float() * torch.rsqrt(variance + self._qkv_eps)
        return (x_norm * weight.float()).to(x.dtype)

    def _split_qkv(self, qkv: torch.Tensor):
        nq, nkv, hd = self.num_heads_local, self.num_kv_heads_local, self.head_dim
        lead = qkv.shape[:-1]
        if self._qkv_layout == "mcore":
            if self._replicate_kv:
                if self._output_gate:
                    return split_grouped_qkvg_for_tp(
                        qkv,
                        num_heads=self.num_heads,
                        num_kv_heads=self.num_kv_heads,
                        head_dim=hd,
                        tp_rank=self.ps.tp_rank,
                        tp_size=self.ps.tp_size,
                    )

                replicas_per_kv_head = ensure_divisible(
                    self.ps.tp_size, self.num_kv_heads
                )
                q_heads_per_group = ensure_divisible(self.num_heads, self.num_kv_heads)
                group_width = (q_heads_per_group + 2) * hd
                kv_group_rank = self.ps.tp_rank // replicas_per_kv_head
                q_rank_in_group = self.ps.tp_rank % replicas_per_kv_head
                qkv = qkv.narrow(-1, kv_group_rank * group_width, group_width)
                qkv = qkv.view(*lead, 1, group_width)
                q, k, v = qkv.split([q_heads_per_group * hd, hd, hd], dim=-1)
                q = q.reshape(*lead, q_heads_per_group, hd)
                q_start = q_rank_in_group * nq
                return q[..., q_start : q_start + nq, :], None, k, v

            q_per_group = ensure_divisible(nq, nkv)
            if self._output_gate:
                return split_grouped_qkvg(qkv, num_heads=nq, num_kv_heads=nkv, head_dim=hd)

            qkv = qkv.view(*lead, nkv, (q_per_group + 2) * hd)
            q = qkv[..., : q_per_group * hd].reshape(*lead, nq, hd)
            k = qkv[..., q_per_group * hd : (q_per_group + 1) * hd]
            v = qkv[..., (q_per_group + 1) * hd : (q_per_group + 2) * hd]
            return q, None, k, v

        if self._output_gate:
            q_block = qkv[..., : nq * 2 * hd].reshape(*lead, nq, 2 * hd)
            kv_block = qkv[..., nq * 2 * hd :].reshape(*lead, 2 * nkv, hd)
            q = q_block[..., :hd]
            gate = q_block[..., hd:]
            k = kv_block[..., :nkv, :]
            v = kv_block[..., nkv:, :]
            return q, gate, k, v
        qkv = qkv.view(*lead, nq + 2 * nkv, hd)
        # Match MCore SelfAttention's split path: keep q/k/v as views instead
        # of inserting copy nodes, since those copy nodes alter backward
        # accumulation at the qkv boundary under TP/CP.
        q = qkv[..., :nq, :]
        k = qkv[..., nq : nq + nkv, :]
        v = qkv[..., nq + nkv :, :]
        return q, None, k, v
