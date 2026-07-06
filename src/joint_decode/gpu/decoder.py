from __future__ import annotations

import functools
import json
import logging
import os
import subprocess
import sys

from joint_decode.config import GenerateOutput
from joint_decode.coordinator import JointDecoder, SelectTokens
from joint_decode.gpu.config import VLLM_GPU_ENV_VARS, JointDecodeConfig

logger = logging.getLogger(__name__)


def joint_decoder(config: JointDecodeConfig, *, select_token: SelectTokens) -> JointDecoder:
    return JointDecoder(
        config.sampling,
        max_model_len_a=config.model_a.max_model_len,
        max_model_len_b=config.model_b.max_model_len,
        select_token=select_token,
        spawn_worker=functools.partial(_spawn_worker, config),
    )


def run_joint_decode(
    config: JointDecodeConfig,
    prompts_a: list[str],
    prompts_b: list[str],
    *,
    select_token: SelectTokens,
) -> list[GenerateOutput]:
    with joint_decoder(config, select_token=select_token) as decoder:
        return decoder.generate(prompts_a, prompts_b)


def _spawn_worker(
    config: JointDecodeConfig,
    *,
    side: str,
    decision_env: dict[str, str],
    max_num_seqs: int,
    max_num_batched_tokens: int,
) -> subprocess.Popen:
    model_config = config.model_a if side == "a" else config.model_b
    max_tokens = config.sampling.max_tokens_a if side == "a" else config.sampling.max_tokens_b

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(model_config.gpu_index)
    env.update(decision_env)
    for key, value in VLLM_GPU_ENV_VARS.items():
        env[key] = value

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "joint_decode.gpu.worker",
        "--model-path",
        model_config.model_path,
        "--max-tokens",
        str(max_tokens),
        "--max-model-len",
        str(model_config.max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--seed",
        str(config.sampling.seed),
    ]
    if model_config.gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(model_config.gpu_memory_utilization)]
    if model_config.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    if model_config.enforce_eager:
        cmd.append("--enforce-eager")
    if config.sampling.stop:
        cmd += ["--stop", json.dumps(list(config.sampling.stop))]

    logger.info("spawning joint-decode worker %s on CUDA_VISIBLE_DEVICES=%s", side, model_config.gpu_index)
    return subprocess.Popen(
        cmd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
