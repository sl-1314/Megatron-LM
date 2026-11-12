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
from megatron.lite.primitive.utils import build_fp8_recipe

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
        self, config: Qwen35Config, ps: ParallelState, *, use_plain_te_linear: bool = False
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
        gate_val = self.shared_gate(x).sigmoid()
        fc1_input = x
        if self.use_mcore_overlap_graph:
            fc1_input = self._CopyToTPRegion.apply(x, self.tp_group)
            self._set_grad_fn_sequence_sr(fc1_input)
        # 0-diff: vLLM silu_and_mul contract (see _vllm_swiglu).
        act = (
            _vllm_swiglu(self.gate_up(fc1_input))
            if fc1_input.dtype == torch.bfloat16
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
        self.experts = Experts(config, ps, fp8=fp8, moe_act_recompute=moe_act_recompute)
        self.vllm_fused_moe_contract = bool(
            vllm_fused_moe_contract
            and vllm_topk_contract
            and ps.tp_size == 1
            and ps.ep_size == 1
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
        self.shared_expert = SharedExpert(config, ps, use_plain_te_linear=shared_expert_plain_te)
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
        if self.vllm_fused_moe_contract and x_2d.dtype == torch.bfloat16:
            # 0-diff: drive vLLM's Triton fused_moe (same binaries the
            # rollout engine runs) on the stacked expert weights — bitwise
            # routed output; see experts.vllm_fused_moe.
            from megatron.lite.primitive.modules.experts import (
                stack_expert_weights,
                vllm_fused_moe,
                vllm_fused_moe_available,
            )

            if vllm_fused_moe_available():
                w13 = stack_expert_weights(self.experts.fc1)
                w2 = stack_expert_weights(self.experts.fc2)
                routed_out = vllm_fused_moe(x_2d.contiguous(), w13, w2, scores, indices)
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
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.ps = ps
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
                vllm_fa_contract=True,
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
                vllm_topk_contract=True,
                # 0-diff: routed experts through vLLM's Triton fused_moe
                # (bitwise vs the rollout engine; TE grouped-GEMM path stays
                # the fallback for TP/EP or non-bf16 configs).
                vllm_fused_moe_contract=True,
            )
            self.mlp: DenseMLP | None = None
        else:
            self.mlp_norm = None
            self.moe = None
            self.mlp = DenseMLP(config, ps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(module output, residual stream)``, both UN-ADDED.

        0-diff (qwen35-0diff-lite): vLLM keeps the residual stream un-added
        between sublayers and lets each Gemma fused add+RMSNorm consume the
        exact fp32 (x + residual) inside the kernel. Pre-adding in bf16 loses
        the exact sum and diverged 51% of in_proj outputs on captured data,
        so the model loop now carries the (h, residual) pair.
        """
        if self.full_attn is not None:
            # 0-diff (qwen35-0diff-lite): vLLM FA-layer input norm contract
            # (flashinfer Gemma fused add+RMSNorm on the fp32 sum, TP=1 only),
            # then GQAttention's vLLM FA kernel contract on the post-norm
            # activation.  qkv.linear.layer_norm_weight stores the raw
            # input_layernorm.weight (no -1), the convention the kernel wants.
            if self.full_attn.vllm_fa_engages(x, packed_seq_params) and _GDN_HAS_FI_NORM:
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
            )
        if self.moe is not None:
            assert self.mlp_norm is not None
            # 0-diff (qwen35-0diff-lite): vLLM post-attention norm contract
            # (flashinfer Gemma fused add+RMSNorm on the fp32 sum), TP=1 only.
            if _GDN_HAS_FI_NORM and self.ps.tp_size == 1:
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
            self._vllm_head_norm_contract = (
                _GDN_HAS_FI_NORM and ps.tp_size == 1
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
                    ),
                    detach_encoder=mtp_detach_encoder,
                )

            self.mtp = MTPBlock(
                num_layers=config.num_nextn_predict_layers,
                repeated_layer=config.mtp_use_repeated_layer,
                layer_factory=make_mtp_layer,
            )

        self.sp_params: list[nn.Parameter] = []
        if ps.tp_size > 1:
            self.sp_params = _collect_sp_grad_params(self)

    def set_input_tensor(self, input_tensor):
        if isinstance(input_tensor, list):
            if len(input_tensor) > 1:
                raise ValueError("Qwen35Model expects a single pipeline input tensor.")
            input_tensor = input_tensor[0] if input_tensor else None
        self._input_tensor = input_tensor

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
            if self.embed is not None:
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
                logits = self.head(hidden_for_head)
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
                        entropy = vocab_parallel_entropy(logits, self.ps.tp_group)
                        output["entropy"] = entropy.transpose(0, 1).contiguous()
                else:
                    if temperature_value != 1.0:
                        logits = logits / temperature_value
                    loss = vocab_parallel_cross_entropy(logits, labels_sb, self.ps.tp_group)
                    output["loss"] = loss.mean()
                    output["log_probs"] = (-loss).transpose(0, 1).contiguous()
                    if calculate_entropy:
                        entropy = vocab_parallel_entropy(logits, self.ps.tp_group)
                        output["entropy"] = entropy.transpose(0, 1).contiguous()
        elif self.head is not None:
            logits = self.head(hidden_for_head)
            output["logits"] = self.head.gather(logits).transpose(0, 1).contiguous()
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
