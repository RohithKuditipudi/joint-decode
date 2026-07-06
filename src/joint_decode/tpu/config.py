from __future__ import annotations

from dataclasses import dataclass

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
class JointDecodeModelConfig:
    model_path: str
    chip_index: int
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
    # one vLLM assets cache per chip.
    cache_dir: str
