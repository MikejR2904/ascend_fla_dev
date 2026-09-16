"""GDN-2 的固定形状、可递推 torch_npu Graph decode runner。

NPU Graph replay 要求所有输入输出地址固定。模型普通 eager decode 每步返回一组新 cache，
因此本 runner 在图尾把新 recurrent/conv cache 拷回固定输入缓冲区。下一次 replay 便从更新后
的 state 开始，数学上仍是正常的自回归递推。

首版刻意只声明真实模型已经验证过的窄契约：B=1、T=1、BF16、packed-inference、CCE、NPU。
不满足时直接报错，不回退 eager。
"""
from __future__ import annotations

from collections.abc import Sequence
import time

import torch
from torch import nn

from ..layers.gdn2 import GDN2LayerCache

__all__ = ["GDN2NPUGraphDecodeRunner"]


class GDN2NPUGraphDecodeRunner:
    """Capture and replay one stateful GDN-2 decode step.

    ``step()`` returns one static logits tensor whose contents are replaced by the
    next replay.  Callers must consume or copy it before calling ``step()`` again.
    The cache tensors exposed by :attr:`cache` likewise have stable addresses and
    are updated in place.  A runner is single-stream/single-caller; synchronize
    externally before sharing tensors with another stream.
    """

    def __init__(
        self,
        model: nn.Module,
        initial_cache: Sequence[GDN2LayerCache | None],
        *,
        sample_input: torch.Tensor | None = None,
        warmup: int = 3,
    ) -> None:
        if warmup < 1:
            raise ValueError(f"NPU Graph capture 至少需要一次 T=1 warmup，收到 {warmup}")
        self.model = model
        self.device = self._validate_model(model)
        validated_cache, offset = self._validate_cache(initial_cache)
        self._offset = offset

        if sample_input is None:
            sample_input = torch.zeros((1, 1), dtype=torch.long, device="cpu")
        self._validate_token(sample_input)
        self._static_input = torch.empty(
            (1, 1), dtype=torch.long, device=self.device
        )
        self._static_input.copy_(sample_input)

        setup_started = time.perf_counter()
        with torch.inference_mode():
            # Prompt execution does not initialize all T=1 kernels.  Warm the exact
            # decode route on a side-effect-free cache before capture so lazy setup
            # and first-use workspace allocation cannot enter the graph.
            warmup_started = time.perf_counter()
            for _ in range(warmup):
                warmup_output = model(
                    self._static_input,
                    cache=validated_cache,
                    return_cache=True,
                )
            torch.npu.synchronize(self.device)
            self.warmup_seconds = time.perf_counter() - warmup_started
            del warmup_output

            self._input_cache = self._clone_cache(validated_cache)
            self._graph = torch.npu.NPUGraph()
            capture_started = time.perf_counter()
            with torch.npu.graph(self._graph):
                self._logits, self._output_cache = model(
                    self._static_input,
                    cache=self._input_cache,
                    return_cache=True,
                )
                self._copy_cache_(self._input_cache, self._output_cache)
            self.capture_seconds = time.perf_counter() - capture_started

            # Capture records work but does not promise to execute it.  Restoring the
            # known prompt cache makes this independent of capture implementation.
            self._copy_cache_(self._input_cache, validated_cache)
            torch.npu.synchronize(self.device)
        self.setup_seconds = time.perf_counter() - setup_started

    @staticmethod
    def _validate_model(model: nn.Module) -> torch.device:
        if model.training:
            raise RuntimeError("NPU Graph decode 要求 model.eval()")
        if getattr(model, "core_backend", None) != "cce":
            raise ValueError("NPU Graph decode 只支持 core_backend='cce'，不静默回退")
        if getattr(model, "projection_layout", None) != "packed-inference":
            raise ValueError(
                "NPU Graph decode 只支持 projection_layout='packed-inference'，不静默回退"
            )
        config = getattr(model, "config", None)
        expected = {
            "num_heads": 16,
            "num_v_heads": 16,
            "head_dim": 128,
            "conv_size": 4,
        }
        actual = {name: getattr(config, name, None) for name in expected}
        if actual != expected:
            raise ValueError(f"NPU Graph decode 的 GDN-2 定尺不匹配：{actual} != {expected}")
        try:
            parameter = next(model.parameters())
        except StopIteration as error:
            raise ValueError("NPU Graph decode 要求模型至少有一个参数") from error
        if parameter.device.type != "npu":
            raise ValueError(f"NPU Graph decode 要求 NPU 模型，收到 {parameter.device}")
        lm_head = getattr(model, "lm_head", None)
        if getattr(getattr(lm_head, "weight", None), "dtype", None) != torch.bfloat16:
            raise ValueError("NPU Graph decode 首版只支持 BF16 模型权重")
        if not hasattr(torch, "npu") or not hasattr(torch.npu, "NPUGraph"):
            raise RuntimeError("当前 torch_npu 未提供 torch.npu.NPUGraph")
        return parameter.device

    def _validate_cache(
        self, cache: Sequence[GDN2LayerCache | None]
    ) -> tuple[list[GDN2LayerCache], int]:
        expected_layers = getattr(getattr(self.model, "config", None), "n_layer", None)
        if len(cache) != expected_layers:
            raise ValueError(f"graph cache 应有 {expected_layers} 层，收到 {len(cache)}")
        result: list[GDN2LayerCache] = []
        offsets: set[int] = set()
        for index, layer in enumerate(cache):
            if not isinstance(layer, GDN2LayerCache):
                raise TypeError(f"graph cache 第 {index} 层不是 GDN2LayerCache")
            if not isinstance(layer.conv_state, torch.Tensor):
                raise TypeError(f"graph cache 第 {index} 层必须是 packed conv tensor")
            expected_state = (1, 16, 128, 128)
            expected_conv = (1, 3 * 16 * 128, 4)
            if (
                tuple(layer.recurrent_state.shape) != expected_state
                or layer.recurrent_state.dtype != torch.float32
            ):
                raise ValueError(
                    f"graph recurrent cache 第 {index} 层应为 FP32 {expected_state}，收到 "
                    f"{layer.recurrent_state.dtype} {tuple(layer.recurrent_state.shape)}"
                )
            if (
                tuple(layer.conv_state.shape) != expected_conv
                or layer.conv_state.dtype != torch.bfloat16
            ):
                raise ValueError(
                    f"graph conv cache 第 {index} 层应为 BF16 {expected_conv}，收到 "
                    f"{layer.conv_state.dtype} {tuple(layer.conv_state.shape)}"
                )
            for name, tensor in (
                ("recurrent_state", layer.recurrent_state),
                ("conv_state", layer.conv_state),
            ):
                if tensor.device != self.device:
                    raise ValueError(
                        f"graph cache 第 {index} 层 {name} 应在 {self.device}，收到 {tensor.device}"
                    )
                if not tensor.is_contiguous():
                    raise ValueError(
                        f"graph cache 第 {index} 层 {name} 必须 contiguous，收到 {tensor.stride()}"
                    )
            offsets.add(layer.offset)
            result.append(layer)
        if len(offsets) != 1:
            raise ValueError(f"graph cache 各层 offset 必须一致，收到 {sorted(offsets)}")
        return result, offsets.pop()

    def _validate_token(self, token: torch.Tensor) -> None:
        if tuple(token.shape) != (1, 1) or token.dtype != torch.long:
            raise ValueError(
                f"graph decode token 必须是 torch.long [1,1]，收到 "
                f"{token.dtype} {tuple(token.shape)}"
            )
        if token.device.type not in ("cpu", "npu"):
            raise ValueError(f"graph decode token 只接受 CPU/NPU tensor，收到 {token.device}")
        if token.device.type == "npu" and token.device != self.device:
            raise ValueError(f"graph decode token 应在 {self.device}，收到 {token.device}")
        if token.device.type == "cpu":
            token_id = int(token.item())
            vocab_size = getattr(getattr(self.model, "config", None), "vocab_size", 0)
            if not 0 <= token_id < vocab_size:
                raise ValueError(f"graph decode token 应位于 [0,{vocab_size})，收到 {token_id}")

    @staticmethod
    def _clone_cache(cache: Sequence[GDN2LayerCache]) -> list[GDN2LayerCache]:
        return [
            GDN2LayerCache(
                recurrent_state=layer.recurrent_state.clone(),
                conv_state=layer.conv_state.clone(),  # type: ignore[union-attr]
                offset=layer.offset,
            )
            for layer in cache
        ]

    @staticmethod
    def _copy_cache_(
        destination: Sequence[GDN2LayerCache],
        source: Sequence[GDN2LayerCache | None],
    ) -> None:
        if len(destination) != len(source):
            raise ValueError(f"cache 层数不一致：{len(destination)} != {len(source)}")
        for index, (destination_layer, source_layer) in enumerate(
            zip(destination, source)
        ):
            if source_layer is None or not isinstance(source_layer.conv_state, torch.Tensor):
                raise TypeError(f"graph 输出 cache 第 {index} 层不是 packed tensor cache")
            assert isinstance(destination_layer.conv_state, torch.Tensor)
            destination_layer.recurrent_state.copy_(source_layer.recurrent_state)
            destination_layer.conv_state.copy_(source_layer.conv_state)

    @property
    def cache(self) -> list[GDN2LayerCache]:
        """固定地址、内容随 replay 原地推进的当前 cache。"""
        return self._input_cache

    @property
    def offset(self) -> int:
        return self._offset

    @torch.inference_mode()
    def reset(self, cache: Sequence[GDN2LayerCache | None]) -> None:
        """把已有 graph 重置到另一个同形状 prompt cache，不重新捕获。"""
        validated, offset = self._validate_cache(cache)
        self._copy_cache_(self._input_cache, validated)
        torch.npu.synchronize(self.device)
        self._offset = offset
        for layer in self._input_cache:
            layer.offset = offset

    @torch.inference_mode()
    def step(self, token: torch.Tensor) -> torch.Tensor:
        """Consume one token, advance both caches in place, and return static logits."""
        self._validate_token(token)
        self._static_input.copy_(token)
        self._graph.replay()
        self._offset += 1
        for layer in self._input_cache:
            layer.offset = self._offset
        return self._logits

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GDN2NPUGraphDecodeRunner(device={self.device}, offset={self.offset}, "
            f"capture_seconds={self.capture_seconds:.6f})"
        )
