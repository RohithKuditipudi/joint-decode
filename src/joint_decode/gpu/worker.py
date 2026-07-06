from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

from joint_decode.gpu.config import VLLM_GPU_ENV_VARS
from joint_decode.worker_loop import engine_max_live_requests, run_worker_loop, validate_engine

logger = logging.getLogger(__name__)


def main() -> None:
    import torch.distributed as dist

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--stop", default=None)
    args = parser.parse_args()

    try:
        run_worker(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_worker(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s [worker pid=%(process)d] %(message)s",
    )
    for key, value in VLLM_GPU_ENV_VARS.items():
        os.environ[key] = value

    from vllm import LLM
    from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config

    from joint_decode.gpu.logits_processor import JointDecodeLogitsProcessor

    kwargs: dict[str, Any] = {
        "model": args.model_path,
        "trust_remote_code": True,
        "seed": args.seed,
        "tensor_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enforce_eager": args.enforce_eager,
        "async_scheduling": False,
        "logits_processors": [JointDecodeLogitsProcessor],
    }
    if args.gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization

    llm = LLM(**kwargs)
    stop = list(json.loads(args.stop)) if args.stop is not None else None
    engine = llm.llm_engine
    validate_engine(engine, args.max_num_seqs, args.max_num_batched_tokens)
    run_worker_loop(
        llm,
        max_tokens=args.max_tokens,
        stop=stop,
        max_live_requests=engine_max_live_requests(engine, get_max_concurrency_for_kv_cache_config),
    )


if __name__ == "__main__":
    main()
