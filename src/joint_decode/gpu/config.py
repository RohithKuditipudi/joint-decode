from __future__ import annotations

from dataclasses import dataclass

from joint_decode.config import JointDecodeSamplingConfig

VLLM_GPU_ENV_VARS: dict[str, str] = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_LOGGING_STREAM": "ext://sys.stderr",
}


@dataclass(frozen=True)
class JointDecodeModelConfig:
    model_path: str
    gpu_index: int
    max_model_len: int
    gpu_memory_utilization: float | None
    enable_prefix_caching: bool
    enforce_eager: bool


@dataclass(frozen=True)
class JointDecodeConfig:
    model_a: JointDecodeModelConfig
    model_b: JointDecodeModelConfig
    sampling: JointDecodeSamplingConfig
