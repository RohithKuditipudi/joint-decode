from __future__ import annotations

from dataclasses import dataclass
from math import prod

from joint_decode.config import JointDecodeSamplingConfig

VLLM_TPU_ENV_VARS: dict[str, str] = {
    "MARIN_VLLM_MODE": "native",
    "VLLM_TARGET_DEVICE": "tpu",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    "VLLM_TPU_SKIP_PRECOMPILE": "1",
}


@dataclass(frozen=True)
class TpuPlacement:
    visible_chips: tuple[int, ...]
    chips_per_process_bounds: tuple[int, int, int]
    tensor_parallel_size: int

    def __post_init__(self) -> None:
        if not self.visible_chips:
            raise ValueError("visible_chips must be non-empty")
        if len(set(self.visible_chips)) != len(self.visible_chips):
            raise ValueError(f"visible_chips must be unique, got {self.visible_chips}")
        if any(chip < 0 for chip in self.visible_chips):
            raise ValueError(f"visible_chips must be non-negative, got {self.visible_chips}")
        if any(size <= 0 for size in self.chips_per_process_bounds):
            raise ValueError(f"chips_per_process_bounds must be positive, got {self.chips_per_process_bounds}")
        bounds_volume = prod(self.chips_per_process_bounds)
        if bounds_volume != len(self.visible_chips):
            raise ValueError(
                f"chips_per_process_bounds={self.chips_per_process_bounds} has volume "
                f"{bounds_volume}, expected {len(self.visible_chips)}"
            )
        if not 1 <= self.tensor_parallel_size <= len(self.visible_chips):
            raise ValueError(
                f"tensor_parallel_size must be in [1, {len(self.visible_chips)}], got {self.tensor_parallel_size}"
            )


@dataclass(frozen=True)
class JointDecodeModelConfig:
    model_path: str
    placement: TpuPlacement
    max_model_len: int
    gpu_memory_utilization: float | None
    enable_prefix_caching: bool
    # Halve the RPA-kernel block sizes. Required for delphi-shaped models
    # (otherwise vmem error); harms perf on standard models, so default off.
    apply_rpa_block_size_patch: bool = False


@dataclass(frozen=True)
class JointDecodeConfig:
    model_a: JointDecodeModelConfig
    model_b: JointDecodeModelConfig
    sampling: JointDecodeSamplingConfig
    # Root for worker caches: one JAX compilation cache shared by the pair,
    # one vLLM assets cache per side.
    cache_dir: str
