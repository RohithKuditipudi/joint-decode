"""Coordinator-driven worker run loop, shared by all backends.

The backend worker entry point constructs and validates the engine, then hands
it to run_worker_loop. The spawning JointDecoder provides the
RERANK_TOKEN_DECISION_* env vars.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Iterable, Iterator
from typing import Any

from joint_decode.decision import post_decision
from joint_decode.ipc import emit_ipc
from joint_decode.runtime_state import WorkerCommands, runtime_state

logger = logging.getLogger(__name__)


def run_worker_loop(llm: Any, *, max_tokens: int, stop: list[str] | None, max_live_requests: int) -> None:
    from vllm import SamplingParams, TokensPrompt

    side = os.environ["RERANK_TOKEN_DECISION_SIDE"]
    decision_url = os.environ["RERANK_TOKEN_DECISION_URL"]
    timeout = float(os.environ["RERANK_TOKEN_DECISION_TIMEOUT"])
    tokenizer = llm.get_tokenizer()
    eos_id = tokenizer.eos_token_id
    engine = llm.llm_engine

    emit_ipc({"kind": "handshake", "max_live_requests": max_live_requests})
    commands = _commands(sys.stdin)
    for message in commands:
        command = message.get("command")
        if command == "shutdown":
            break
        if command != "process_chunk":
            raise RuntimeError(f"unknown joint-decode worker command: {command!r}")

        runtime_state.reset()
        request_ids: list[str] = message["request_ids"]
        prompts: list[str] = message["prompts"]
        token_ids_by_rid: dict[str, list[int]] = {
            rid: tokenizer.encode(prompt)
            for rid, prompt in zip(request_ids, prompts, strict=True)
        }
        emit_ipc(
            {
                "kind": "plan",
                "prompt_tokens": {rid: len(token_ids) for rid, token_ids in token_ids_by_rid.items()},
            }
        )
        initial_admit = _read_start(commands)

        live: set[str] = set()
        pending_admits: list[str] = []
        text_results: dict[str, str] = {}
        finish_reasons: dict[str, str] = {}

        def admit(rids: list[str]) -> None:
            for rid in rids:
                if rid in live:
                    raise RuntimeError(f"coordinator admitted already-live rid={rid}")
                engine.add_request(
                    request_id=rid,
                    prompt=TokensPrompt(prompt_token_ids=token_ids_by_rid[rid]),
                    params=SamplingParams(
                        max_tokens=max_tokens,
                        ignore_eos=False,
                        stop=stop,
                        stop_token_ids=[eos_id] if eos_id is not None else None,
                        extra_args={"joint_decode_rid": rid},
                    ),
                )
                live.add(rid)
                runtime_state.live_rids.add(rid)

        admit(initial_admit)

        run_done = False
        while not run_done:
            pending_admits.extend(_drain_worker_commands().admit)
            if pending_admits and _local_decision_boundary(live):
                admits = pending_admits
                pending_admits = []
                admit(admits)

            if not live:
                response = post_decision(
                    decision_url,
                    {
                        "kind": "control",
                        "side": side,
                    },
                    timeout=timeout,
                )
                if response.get("abort"):
                    raise RuntimeError(str(response["abort"]))
                if response.get("done"):
                    run_done = True
                    continue
                pending_admits.extend(str(rid) for rid in response.get("admit") or [])
                continue

            _set_held_request_ids(engine, live)
            finished: list[dict[str, Any]] = []
            for output in engine.step():
                if not output.finished:
                    continue
                rid = output.request_id
                if rid not in live:
                    continue
                completion = output.outputs[0]
                text_results[rid] = completion.text
                finish_reasons[rid] = completion.finish_reason or "unknown"
                live.remove(rid)
                runtime_state.live_rids.discard(rid)
                finished.append(
                    {
                        "rid": rid,
                        "finish_reason": completion.finish_reason,
                        "stop_reason": getattr(completion, "stop_reason", None),
                    }
                )
            if finished:
                post_decision(
                    decision_url,
                    {
                        "kind": "finish",
                        "side": side,
                        "finished": finished,
                    },
                    timeout=timeout,
                )
            pending_admits.extend(_drain_worker_commands().admit)

        emit_ipc(
            {
                "kind": "result",
                "results": text_results,
                "finish_reasons": finish_reasons,
            }
        )


def _commands(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        yield json.loads(line)


def _read_start(commands: Iterator[dict[str, Any]]) -> list[str]:
    start = next(commands, None)
    if start is None or start.get("command") != "start":
        raise RuntimeError(f"expected start command after plan, got {start!r}")
    return [str(rid) for rid in start["initial_admit"]]


def _drain_worker_commands() -> WorkerCommands:
    commands = runtime_state.drain_commands()
    if commands.abort is not None:
        raise RuntimeError(commands.abort)
    return commands


def _local_decision_boundary(live: set[str]) -> bool:
    return not any(runtime_state.pending_tokens.get(rid) for rid in live)


def _set_held_request_ids(engine: Any, live: set[str]) -> None:
    scheduler = _scheduler(engine)
    pending_tokens = runtime_state.pending_tokens
    decode_live: dict[str, str] = {}
    for internal_id, request in scheduler.requests.items():
        assert request.sampling_params is not None
        extra_args = request.sampling_params.extra_args
        assert extra_args is not None
        rid = str(extra_args["joint_decode_rid"])

        if rid not in live:
            continue
        if request.num_computed_tokens >= request.num_prompt_tokens:
            decode_live[rid] = internal_id

    busy = any(pending_tokens.get(rid) for rid in decode_live)
    scheduler.held_request_ids = {
        internal_id
        for rid, internal_id in decode_live.items()
        if busy and not pending_tokens.get(rid)
    }


def _scheduler(engine: Any) -> Any:
    engine_core = getattr(engine, "engine_core", None)
    inner_core = getattr(engine_core, "engine_core", None)
    scheduler = getattr(inner_core, "scheduler", None)
    if scheduler is None:
        raise RuntimeError("vLLM engine does not expose engine.engine_core.engine_core.scheduler")
    return scheduler


def engine_max_live_requests(engine: Any, concurrency_fn: Any) -> int:
    """Largest live window that can never exhaust KV cache, per vLLM's own
    max-model-len concurrency accounting. Staying under it means vLLM never
    preempts, which is what keeps the two engines' decode sets identical."""
    scheduler = _scheduler(engine)
    max_concurrency = concurrency_fn(engine.vllm_config, scheduler.kv_cache_config)
    return min(int(max_concurrency), scheduler.scheduler_config.max_num_seqs)


def validate_engine(engine: Any, max_num_seqs: int, max_num_batched_tokens: int) -> None:
    scheduler = _scheduler(engine)
    if not hasattr(scheduler, "held_request_ids"):
        raise RuntimeError("vLLM scheduler does not expose held_request_ids")
    scheduler_config = scheduler.scheduler_config
    if scheduler_config.async_scheduling is not False:
        raise RuntimeError("joint decode requires async_scheduling=False")
    if scheduler_config.enable_chunked_prefill is not False:
        raise RuntimeError("joint decode requires enable_chunked_prefill=False")
    if scheduler_config.long_prefill_token_threshold != 0:
        raise RuntimeError("joint decode requires long_prefill_token_threshold=0")
    if scheduler_config.max_num_seqs != max_num_seqs:
        raise RuntimeError(
            f"vLLM max_num_seqs mismatch: scheduler={scheduler_config.max_num_seqs} expected={max_num_seqs}"
        )
    scheduled_tokens = getattr(scheduler, "max_num_scheduled_tokens", max_num_batched_tokens)
    if scheduled_tokens < max_num_batched_tokens:
        raise RuntimeError(
            f"vLLM max_num_scheduled_tokens={scheduled_tokens} is below max_num_batched_tokens={max_num_batched_tokens}"
        )
