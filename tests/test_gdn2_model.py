"""GDN-2 torch 基线的 CPU 语义与 checkpoint ABI 测试。"""
from __future__ import annotations

import copy
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ascend_fla.models import (  # noqa: E402
    GDN2Config,
    GDN2ForCausalLM,
    GDN2LayerCache,
    GDN2NPUGraphDecodeRunner,
    GatedDeltaNet2,
    checkpoint_state_dict,
    generate_tokens,
)
from ascend_fla.models.gdn2 import RMSNorm, SwiGLU  # noqa: E402


def _tiny_config(**overrides) -> GDN2Config:
    values = dict(
        vocab_size=32,
        padding_multiple=8,
        block_size=32,
        n_layer=2,
        n_embd=16,
        intermediate_size=24,
        num_heads=2,
        num_v_heads=2,
        head_dim=4,
        conv_size=3,
    )
    values.update(overrides)
    return GDN2Config(**values)


def _rel_l2(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-30))


def test_release_config_parameter_count() -> None:
    model = GDN2ForCausalLM(GDN2Config.gdn2_1_3b(), device="meta")
    assert model.reported_parameter_count == 1_302_638_112
    assert model.parameter_count == 1_450_096_416
    assert len(model.state_dict()) == 399


def test_mixed_mlp_pack_is_explicit_and_nonpersistent() -> None:
    swiglu = SwiGLU(
        2304, 6208, False, device="meta", dtype=torch.bfloat16
    ).eval().pack_for_inference()
    norm = RMSNorm(2304, 1e-5, device="meta", dtype=torch.bfloat16).eval()
    parameter_count = sum(parameter.numel() for parameter in swiglu.parameters())
    state_keys = set(swiglu.state_dict())

    swiglu.enable_mixed_decode(norm)
    assert swiglu.mlp_backend == "cce"
    assert tuple(swiglu.paired_w12_weight.shape) == (97, 128, 2304)
    assert swiglu.paired_w12_weight.dtype == torch.bfloat16
    assert swiglu.paired_w12_weight.device.type == "meta"
    assert sum(parameter.numel() for parameter in swiglu.parameters()) == parameter_count
    assert set(swiglu.state_dict()) == state_keys
    assert "paired_w12_weight" not in swiglu.state_dict()

    # Idempotence must not allocate/register another derived layout.
    original = swiglu.paired_w12_weight
    assert swiglu.enable_mixed_decode(norm) is swiglu
    assert swiglu.paired_w12_weight is original
    with pytest.raises(ValueError, match="NPU tensor"):
        swiglu.forward_with_norm(
            torch.empty((1, 1, 2304), dtype=torch.bfloat16), norm
        )


def test_checkpoint_loader_rejects_ambiguous_mixed_mlp_configuration() -> None:
    with pytest.raises(ValueError, match="packed-inference"):
        GDN2ForCausalLM.from_checkpoint(
            "not-read.pth", projection_layout="canonical", mlp_backend="cce"
        )
    with pytest.raises(ValueError, match="core_backend='cce'"):
        GDN2ForCausalLM.from_checkpoint(
            "not-read.pth",
            projection_layout="packed-inference",
            core_backend="torch",
            mlp_backend="cce",
        )


def test_one_step_scan_matches_closed_form() -> None:
    config = _tiny_config(n_layer=1, use_short_conv=False, use_qk_l2norm=False)
    mixer = GatedDeltaNet2(config, layer_idx=0)
    torch.manual_seed(0)
    shape_k = (1, 1, config.num_heads, config.head_k_dim)
    shape_v = (1, 1, config.num_v_heads, config.head_v_dim)
    q, k = torch.randn(shape_k), torch.randn(shape_k)
    v = torch.randn(shape_v)
    g = -torch.rand(shape_k)
    b = torch.sigmoid(torch.randn(shape_k))
    w = torch.sigmoid(torch.randn(shape_v))
    state = torch.randn(1, config.num_v_heads, config.head_k_dim, config.head_v_dim)

    got, got_state = mixer._scan(q, k, v, g, b, w, state)
    decayed = state * torch.exp(g[:, 0]).unsqueeze(-1)
    erase = torch.matmul((b[:, 0] * k[:, 0]).unsqueeze(-2), decayed).squeeze(-2)
    want_state = decayed + k[:, 0].unsqueeze(-1) * (w[:, 0] * v[:, 0] - erase).unsqueeze(-2)
    want = torch.matmul(
        (q[:, 0] * config.attention_scale).unsqueeze(-2), want_state
    ).squeeze(-2).unsqueeze(1)
    assert torch.allclose(got_state, want_state, atol=1e-6, rtol=1e-6)
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-6)


def test_streaming_cache_matches_full_sequence() -> None:
    torch.manual_seed(0)
    model = GDN2ForCausalLM(_tiny_config()).eval()
    tokens = torch.tensor([[1, 7, 3, 9, 2]])
    with torch.inference_mode():
        full = model(tokens)
        cache = None
        pieces = []
        for index in range(tokens.shape[1]):
            logits, cache = model(tokens[:, index:index + 1], cache=cache, return_cache=True)
            pieces.append(logits)
    stepped = torch.cat(pieces, dim=1)
    assert _rel_l2(stepped, full) < 1e-6
    assert cache is not None and len(cache) == model.config.n_layer
    assert isinstance(cache[0], GDN2LayerCache)
    assert cache[0].offset == tokens.shape[1]
    assert tuple(cache[0]["recurrent_state"].shape) == (1, 2, 4, 4)
    assert tuple(cache[0]["conv_state"][0].shape) == (1, 8, 3)


def test_checkpoint_loader_is_strict_and_prefix_compatible(tmp_path: pathlib.Path) -> None:
    torch.manual_seed(0)
    config = _tiny_config()
    original = GDN2ForCausalLM(config)
    wrapped = {f"_forward_module.module.{key}": value for key, value in original.state_dict().items()}
    path = tmp_path / "checkpoint.pth"
    torch.save(
        {"model": wrapped, "optimizer": {"ignored": True}, "trained_tokens": 95_000_000_000},
        path,
    )

    loaded = GDN2ForCausalLM.from_checkpoint(
        path, config=config, device="cpu", dtype=torch.float32, strict=True
    )
    for key, value in original.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key
    assert loaded.checkpoint_metadata == {"trained_tokens": 95_000_000_000}

    packed = GDN2ForCausalLM.from_checkpoint(
        path,
        config=config,
        device="cpu",
        dtype=torch.float32,
        strict=True,
        projection_layout="packed-inference",
    )
    assert packed.projection_layout == "packed-inference"
    assert packed.parameter_count == original.parameter_count
    tokens = torch.tensor([[1, 3, 7]])
    original.eval()
    with torch.inference_mode():
        assert _rel_l2(packed(tokens), original(tokens)) < 1e-6


def test_litgpt_parameter_names() -> None:
    keys = GDN2ForCausalLM(_tiny_config(n_layer=1)).state_dict().keys()
    expected = {
        "lm_head.weight",
        "transformer.wte.weight",
        "transformer.h.0.norm_1.weight",
        "transformer.h.0.attn.A_log",
        "transformer.h.0.attn.dt_bias",
        "transformer.h.0.attn.q_conv1d.weight",
        "transformer.h.0.attn.f_proj.0.weight",
        "transformer.h.0.attn.g_proj.1.bias",
        "transformer.h.0.attn.o_norm.weight",
        "transformer.h.0.mlp.swiglu.w3.weight",
        "transformer.ln_f.weight",
    }
    assert expected <= set(keys)


def test_checkpoint_normalization_rejects_duplicate_keys() -> None:
    tensor = torch.zeros(1)
    with pytest.raises(ValueError, match="重复 key"):
        checkpoint_state_dict({"weight": tensor, "module.weight": tensor})


def test_packed_layout_matches_canonical_and_uses_packed_cache() -> None:
    torch.manual_seed(0)
    canonical = GDN2ForCausalLM(_tiny_config()).eval()
    packed = copy.deepcopy(canonical).eval().pack_for_inference()
    assert canonical.projection_layout == "canonical"
    assert packed.projection_layout == "packed-inference"
    assert packed.parameter_count == canonical.parameter_count
    assert not hasattr(packed.transformer["h"][0].attn, "q_proj")

    prompt = torch.tensor([[1, 7, 3, 9]])
    step = torch.tensor([[2]])
    with torch.inference_mode():
        canonical_logits, canonical_cache = canonical(prompt, return_cache=True)
        packed_logits, packed_cache = packed(prompt, return_cache=True)
        canonical_step, canonical_next = canonical(
            step, cache=canonical_cache, return_cache=True
        )
        packed_step, packed_next = packed(step, cache=packed_cache, return_cache=True)

    # CPU/NPU BLAS 可以因输出列数变化选择不同规约路径；语义判 relative-L2 数值预算。
    # 真实 A5 同 dtype 对照由 benchmarks/verify_gdn2_packed.py 单独验，bitwise 只作诊断。
    assert _rel_l2(packed_logits, canonical_logits) < 1e-6
    assert _rel_l2(packed_step, canonical_step) < 1e-6
    for canonical_layer, packed_layer in zip(canonical_cache, packed_cache):
        assert canonical_layer is not None and packed_layer is not None
        assert _rel_l2(packed_layer.recurrent_state, canonical_layer.recurrent_state) < 1e-6
        assert isinstance(canonical_layer.conv_state, tuple)
        assert isinstance(packed_layer.conv_state, torch.Tensor)
        assert torch.allclose(
            packed_layer.conv_state,
            torch.cat(canonical_layer.conv_state, dim=1),
            atol=1e-6,
            rtol=1e-6,
        )
        assert packed_layer.offset == canonical_layer.offset == prompt.shape[1]
    assert packed_next[0] is not None and canonical_next[0] is not None
    assert packed_next[0].offset == canonical_next[0].offset == prompt.shape[1] + 1


def test_packed_inference_configures_norms_packs_mlp_and_caches_decay() -> None:
    torch.manual_seed(7)
    model = GDN2ForCausalLM(_tiny_config(), dtype=torch.bfloat16).eval()
    with torch.no_grad():
        for block in model.transformer["h"]:
            block.attn.A_log.copy_(torch.linspace(-0.5, 0.5, block.attn.num_heads))

    native_norms = [model.transformer["ln_f"]]
    fused_decode_norms = []
    for block in model.transformer["h"]:
        native_norms.extend((block.norm_1, block.norm_2))
        fused_decode_norms.append(block.attn.o_norm)
    native_values = [module.weight.detach().clone() for module in native_norms]
    fused_decode_values = [
        module.weight.detach().float().clone() for module in fused_decode_norms
    ]
    original_parameter_count = model.parameter_count
    expected_decay = [
        (-block.attn.A_log.detach().float().exp().view(block.attn.num_heads, 1))
        .expand(block.attn.num_heads, block.attn.head_k_dim)
        .contiguous()
        for block in model.transformer["h"]
    ]

    model.pack_for_inference()

    assert model.parameter_count == original_parameter_count
    for module, expected in zip(native_norms, native_values):
        assert module.weight.dtype == torch.bfloat16
        assert module._use_native_npu_inference
        assert torch.equal(module.weight, expected)
    for module, expected in zip(fused_decode_norms, fused_decode_values):
        assert module.weight.dtype == torch.float32
        assert torch.equal(module.weight, expected)
    for block, expected in zip(model.transformer["h"], expected_decay):
        swiglu = block.mlp.swiglu
        assert swiglu.projection_layout == "packed-inference"
        assert not hasattr(swiglu, "w1")
        assert not hasattr(swiglu, "w2")
        assert tuple(swiglu.w12_weight.shape) == (
            2 * model.config.intermediate_size,
            model.config.n_embd,
        )
        assert block.attn._decay_rate.dtype == torch.float32
        assert torch.equal(block.attn._decay_rate, expected)
        assert "_decay_rate" not in block.attn.state_dict()


def test_real_shape_packed_bf16_cce_routes_one_token_through_fused_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config(
        n_layer=1,
        num_heads=16,
        num_v_heads=16,
        head_dim=128,
        use_short_conv=False,
    )
    model = GDN2ForCausalLM(
        config, core_backend="cce", dtype=torch.bfloat16
    ).eval()
    model.pack_for_inference()
    calls: list[tuple[torch.Size, ...]] = []

    def fake_fused_decode(*args, initial_state=None, block_dim=8, **kwargs):
        del kwargs
        q, *rest = args
        calls.append(tuple(tensor.shape for tensor in (q, *rest)))
        assert block_dim == 8
        assert q.dtype == torch.bfloat16
        assert all(tensor.is_contiguous() for tensor in args)
        state = initial_state
        if state is None:
            state = torch.zeros(1, 16, 128, 128)
        return torch.zeros_like(q), state

    monkeypatch.setattr(
        "ascend_fla.ops.gdn2.fused_decode_gdn2", fake_fused_decode
    )
    with torch.inference_mode():
        _, cache = model(torch.tensor([[1]]), return_cache=True)
    assert len(calls) == 1
    assert calls[0][:7] == (torch.Size([1, 1, 16, 128]),) * 7
    assert calls[0][7:] == (
        torch.Size([16, 128]),
        torch.Size([16, 128]),
        torch.Size([128]),
    )
    assert cache[0] is not None
    assert cache[0].recurrent_state.dtype == torch.float32


def test_real_shape_packed_bf16_cce_routes_cached_qkv_through_short_conv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config(
        n_layer=1,
        num_heads=16,
        num_v_heads=16,
        head_dim=128,
        conv_size=4,
        use_short_conv=True,
    )
    mixer = GatedDeltaNet2(
        config, layer_idx=0, core_backend="cce", dtype=torch.bfloat16
    ).eval()
    mixer.pack_for_inference()
    conv_state = torch.randn(1, 6144, 4, dtype=torch.bfloat16)
    hidden_states = torch.randn(1, 1, config.n_embd, dtype=torch.bfloat16)
    calls: list[tuple[torch.Size, torch.Size, torch.Size]] = []

    def fake_short_conv(x, cache, weight, *, block_dim=8, **kwargs):
        del kwargs
        calls.append((x.shape, cache.shape, weight.shape))
        assert block_dim == 8
        return torch.zeros_like(x), cache.clone()

    monkeypatch.setattr(
        "ascend_fla.ops.gdn2.fused_short_conv_decode_gdn2", fake_short_conv
    )
    with torch.inference_mode():
        projected = mixer._project_packed(hidden_states, conv_state, True)

    assert calls == [
        (
            torch.Size([1, 1, 6144]),
            torch.Size([1, 6144, 4]),
            torch.Size([6144, 1, 4]),
        )
    ]
    assert isinstance(projected[-1], torch.Tensor)
    assert torch.equal(projected[-1], conv_state)


def test_backend_and_packed_training_are_explicitly_gated() -> None:
    with pytest.raises(ValueError, match="不能静默回退"):
        GDN2ForCausalLM(_tiny_config(), core_backend="fused_recurrent")

    model = GDN2ForCausalLM(_tiny_config())
    with pytest.raises(RuntimeError, match="eval"):
        model.pack_for_inference()
    model.eval().pack_for_inference()
    with pytest.raises(RuntimeError, match="不可求导"):
        model(torch.tensor([[1]]))


def test_typed_cache_accepts_legacy_mapping_but_rejects_layout_mismatch() -> None:
    torch.manual_seed(0)
    canonical = GDN2ForCausalLM(_tiny_config()).eval()
    with torch.inference_mode():
        _, cache = canonical(torch.tensor([[1, 2]]), return_cache=True)
        legacy = [dict(layer) if layer is not None else None for layer in cache]
        got, next_cache = canonical(torch.tensor([[3]]), cache=legacy, return_cache=True)
        want, _ = canonical(torch.tensor([[3]]), cache=cache, return_cache=True)
    assert torch.equal(got, want)
    assert next_cache[0] is not None and next_cache[0].offset == 3

    packed = copy.deepcopy(canonical).eval().pack_for_inference()
    with pytest.raises(TypeError, match="packed-inference"):
        with torch.inference_mode():
            packed(torch.tensor([[3]]), cache=cache, return_cache=True)


class _ScriptedCausalLM(torch.nn.Module):
    def __init__(self, next_tokens: list[int], vocab_size: int = 16) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = type("Config", (), {"vocab_size": vocab_size})()
        self.next_tokens = next_tokens
        self.seen_lengths: list[int] = []

    def forward(self, input_ids, cache=None, use_cache=False, return_cache=False):
        del use_cache
        self.seen_lengths.append(input_ids.shape[1])
        token = self.next_tokens[len(self.seen_lengths) - 1]
        logits = torch.full((1, input_ids.shape[1], self.config.vocab_size), -10.0)
        logits[:, -1, token] = 10.0
        next_cache = (
            {"step": len(self.seen_lengths)}
            if cache is None
            else {"step": cache["step"] + 1}
        )
        return (logits, next_cache) if return_cache else logits


@pytest.mark.parametrize("decode_backend", ["eager", "npu-graph"])
def test_generation_prefills_once_then_uses_single_token_cache(
    decode_backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ScriptedCausalLM([4, 5, 6])
    captured: list[dict[str, object]] = []

    class FakeGraphRunner:
        def __init__(self, model, initial_cache, *, sample_input, warmup):
            self.model = model
            self.cache = initial_cache
            captured.append(
                {
                    "sample_input": sample_input.clone(),
                    "warmup": warmup,
                }
            )

        def step(self, token):
            logits, self.cache = self.model(
                token, cache=self.cache, use_cache=True, return_cache=True
            )
            return logits

    if decode_backend == "npu-graph":
        monkeypatch.setattr(
            "ascend_fla.models.generation.GDN2NPUGraphDecodeRunner",
            FakeGraphRunner,
        )
    result = generate_tokens(
        model,
        torch.tensor([[1, 8, 9]], dtype=torch.long),
        max_new_tokens=3,
        temperature=0,
        eos_token_id=None,
        decode_backend=decode_backend,
        graph_warmup=2,
    )
    assert result.token_ids.tolist() == [[1, 8, 9, 4, 5, 6]]
    assert result.prompt_tokens == 3
    assert result.generated_tokens == 3
    assert not result.stopped_on_eos
    assert result.decode_backend == decode_backend
    assert result.timings is not None
    assert result.timings.decode_model_calls == 2
    assert model.seen_lengths == [3, 1, 1]
    if decode_backend == "npu-graph":
        assert len(captured) == 1
        assert captured[0]["sample_input"].tolist() == [[4]]
        assert captured[0]["warmup"] == 2
    else:
        assert captured == []


def test_generation_rejects_unknown_decode_backend() -> None:
    with pytest.raises(ValueError, match="不能静默回退"):
        generate_tokens(
            _ScriptedCausalLM([4]),
            torch.tensor([[1]], dtype=torch.long),
            max_new_tokens=1,
            decode_backend="automatic",
        )


def test_npu_graph_runner_rejects_non_npu_model_before_capture() -> None:
    class FakeGraphModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
            self.lm_head = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
            self.core_backend = "cce"
            self.projection_layout = "packed-inference"
            self.config = type(
                "Config",
                (),
                {
                    "num_heads": 16,
                    "num_v_heads": 16,
                    "head_dim": 128,
                    "conv_size": 4,
                },
            )()

    with pytest.raises(ValueError, match="NPU 模型"):
        GDN2NPUGraphDecodeRunner._validate_model(FakeGraphModel().eval())


def test_generation_stops_on_eos_without_extra_model_call() -> None:
    model = _ScriptedCausalLM([2])
    result = generate_tokens(
        model,
        torch.tensor([[1, 8]], dtype=torch.long),
        max_new_tokens=8,
        temperature=0,
        eos_token_id=2,
    )
    assert result.token_ids.tolist() == [[1, 8, 2]]
    assert result.generated_tokens == 1
    assert result.stopped_on_eos
    assert model.seen_lengths == [2]


def test_graph_generation_stops_on_first_eos_without_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_graph(*args, **kwargs):
        raise AssertionError(f"首 token 已是 EOS，不应捕获 graph：{args} {kwargs}")

    monkeypatch.setattr(
        "ascend_fla.models.generation.GDN2NPUGraphDecodeRunner",
        unexpected_graph,
    )
    model = _ScriptedCausalLM([2])
    result = generate_tokens(
        model,
        torch.tensor([[1, 8]], dtype=torch.long),
        max_new_tokens=8,
        temperature=0,
        eos_token_id=2,
        decode_backend="npu-graph",
    )
    assert result.token_ids.tolist() == [[1, 8, 2]]
    assert result.timings is not None
    assert result.timings.decode_setup_seconds == 0
    assert result.timings.decode_model_calls == 0


def test_generation_sampling_is_seeded() -> None:
    first_model = _ScriptedCausalLM([4, 5, 6])
    second_model = _ScriptedCausalLM([4, 5, 6])
    prompt = torch.tensor([[1]], dtype=torch.long)
    first = generate_tokens(
        first_model, prompt, max_new_tokens=3, temperature=0.8, top_k=4, seed=7
    )
    second = generate_tokens(
        second_model, prompt, max_new_tokens=3, temperature=0.8, top_k=4, seed=7
    )
    assert torch.equal(first.token_ids, second.token_ids)
