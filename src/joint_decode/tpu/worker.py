from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

from joint_decode.tpu.config import VLLM_TPU_ENV_VARS
from joint_decode.tpu.decision_client import DecisionClient
from joint_decode.worker_loop import (
    _scheduler,
    engine_max_live_requests,
    run_worker_loop,
    validate_engine,
)

logger = logging.getLogger(__name__)

_V5_RPA_VMEM_LIMIT_BYTES = 64_000_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--apply-rpa-block-size-patch", action="store_true")
    parser.add_argument("--stop", default=None)
    args = parser.parse_args()
    run_worker(args)


def run_worker(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s [worker pid=%(process)d] %(message)s",
    )
    for key, value in VLLM_TPU_ENV_VARS.items():
        os.environ.setdefault(key, value)
    if os.environ.get("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION") != "1":
        raise RuntimeError("joint decode on TPU requires VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1")
    if args.apply_rpa_block_size_patch:
        _patch_rpa_kernel_block_sizes()

    from tpu_inference.runner import token_decision
    from vllm import LLM
    from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config

    kwargs: dict[str, Any] = {
        "model": args.model_path,
        "trust_remote_code": True,
        "load_format": "runai_streamer",
        "seed": args.seed,
        "tensor_parallel_size": args.tensor_parallel_size,
        "data_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": args.enable_prefix_caching,
        "async_scheduling": False,
    }
    if args.gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization

    llm = LLM(**kwargs)
    stop = list(json.loads(args.stop)) if args.stop is not None else None
    engine = llm.llm_engine
    validate_engine(engine, args.max_num_seqs, args.max_num_batched_tokens)
    _validate_tpu_engine(engine)

    token_decision.register(
        DecisionClient(
            decision_url=os.environ["RERANK_TOKEN_DECISION_URL"],
            side=os.environ["RERANK_TOKEN_DECISION_SIDE"],
            timeout=float(os.environ["RERANK_TOKEN_DECISION_TIMEOUT"]),
            eos_token_id=_eos_token_id(llm.get_tokenizer()),
        ),
        top_k=int(os.environ["RERANK_TOKEN_DECISION_TOP_K"]),
    )
    run_worker_loop(
        llm,
        max_tokens=args.max_tokens,
        stop=stop,
        max_live_requests=engine_max_live_requests(engine, get_max_concurrency_for_kv_cache_config),
    )


def _validate_tpu_engine(engine: Any) -> None:
    parallel_config = engine.vllm_config.parallel_config
    if parallel_config.data_parallel_size != 1:
        raise RuntimeError("joint decode on TPU requires data_parallel_size=1")
    scheduler_module = type(_scheduler(engine)).__module__
    if not scheduler_module.startswith("vllm.v1.core.sched"):
        raise RuntimeError(f"joint decode requires vLLM's v1 scheduler, got {scheduler_module}")
    if (engine.vllm_config.additional_config or {}).get("enable_continue_decode"):
        raise RuntimeError(
            "joint decode requires enable_continue_decode to be unset: the continue-decode "
            "loop samples multiple tokens per step on-device, bypassing the token-decision callback"
        )


def _eos_token_id(tokenizer: Any) -> int:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("joint decode requires the tokenizer to define eos_token_id")
    return int(eos_id)


def _patch_rpa_kernel_block_sizes() -> None:
    import tpu_inference.kernels.ragged_paged_attention.v3.kernel as rpa_kernel

    original = rpa_kernel.get_default_block_sizes
    if getattr(original, "_joint_decode_patched", False):
        return

    def patched_get_default_block_sizes(*args: Any, **kwargs: Any) -> dict[str, int]:
        sizes = dict(original(*args, **kwargs))
        case = kwargs.get("case")
        if case is not rpa_kernel.RpaCase.DECODE:
            q_dtype = args[0]
            kv_dtype = args[1]
            actual_num_q_heads = args[2]
            actual_num_kv_heads = args[3]
            head_dim = args[4]
            page_size = args[5]
            sizes["bq_sz"] = max(1, sizes["bq_sz"] // 2)
            sizes["bq_csz"] = max(1, sizes["bq_csz"] // 2)
            sizes["bkv_sz"] = max(page_size, sizes["bkv_sz"] // 2)
            sizes["bkv_csz"] = max(page_size, sizes["bkv_csz"] // 2)
            if rpa_kernel.get_tpu_version() == 5:
                estimated_vmem_bytes = rpa_kernel.get_vmem_estimate_bytes(
                    actual_num_kv_heads=actual_num_kv_heads,
                    actual_num_q_heads_per_kv_head=actual_num_q_heads // actual_num_kv_heads,
                    actual_head_dim=head_dim,
                    bq_sz=sizes["bq_sz"],
                    bkv_sz=sizes["bkv_sz"],
                    q_dtype=q_dtype,
                    kv_dtype=kv_dtype,
                )
                if estimated_vmem_bytes > _V5_RPA_VMEM_LIMIT_BYTES:
                    sizes["bq_sz"] = max(1, sizes["bq_sz"] // 2)
                    sizes["bkv_sz"] = max(page_size, sizes["bkv_sz"] // 2)
                    sizes["bq_csz"] = min(sizes["bq_csz"], sizes["bq_sz"])
                    sizes["bkv_csz"] = min(sizes["bkv_csz"], sizes["bkv_sz"])
        return sizes

    patched_get_default_block_sizes._joint_decode_patched = True  # type: ignore[attr-defined]
    rpa_kernel.get_default_block_sizes = patched_get_default_block_sizes


if __name__ == "__main__":
    main()
