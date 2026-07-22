from __future__ import annotations

import subprocess

from joint_decode.config import JointDecodeSamplingConfig
from joint_decode.tpu import decoder
from joint_decode.tpu.config import JointDecodeConfig, JointDecodeModelConfig, TpuPlacement


def test_spawn_worker_passes_exact_tpu_placement(monkeypatch, tmp_path):
    placement = TpuPlacement(
        visible_chips=(3, 2, 1, 0),
        chips_per_process_bounds=(2, 2, 1),
        tensor_parallel_size=2,
    )
    model = JointDecodeModelConfig(
        model_path="/model",
        placement=placement,
        max_model_len=8192,
        gpu_memory_utilization=None,
        enable_prefix_caching=False,
    )
    config = JointDecodeConfig(
        model_a=model,
        model_b=model,
        sampling=JointDecodeSamplingConfig(
            max_tokens_a=64,
            max_tokens_b=64,
            top_k_a=16,
            top_k_b=16,
            barrier_timeout_s=60,
            seed=7,
            stop=(),
        ),
        cache_dir=str(tmp_path),
    )
    captured: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(decoder.subprocess, "Popen", fake_popen)

    decoder._spawn_worker(
        config,
        side="a",
        decision_env={"RERANK_TOKEN_DECISION_SIDE": "a"},
        max_num_seqs=4,
        max_num_batched_tokens=8196,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--tensor-parallel-size") + 1] == "2"

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    env = kwargs["env"]
    assert env["TPU_VISIBLE_CHIPS"] == "3,2,1,0"
    assert env["TPU_PROCESS_BOUNDS"] == "1,1,1"
    assert env["TPU_CHIPS_PER_PROCESS_BOUNDS"] == "2,2,1"
    assert env["VLLM_ASSETS_CACHE"] == str(tmp_path / "assets_a")
    assert kwargs["stdin"] is subprocess.PIPE
    assert kwargs["stdout"] is subprocess.PIPE
