from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

from joint_decode.decision import resolve_undecided, split_pending


@dataclass
class RequestState:
    rid: str
    output_token_ids: list[int]


class JointDecodeLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config: Any, device: torch.device, is_pin_memory: bool) -> None:
        del is_pin_memory
        self.device = device
        self.decision_url = _required_env("RERANK_TOKEN_DECISION_URL")
        self.side = _required_env("RERANK_TOKEN_DECISION_SIDE")
        self.top_k = int(_required_env("RERANK_TOKEN_DECISION_TOP_K"))
        self.timeout = float(_required_env("RERANK_TOKEN_DECISION_TIMEOUT"))
        if self.top_k < 1:
            raise ValueError("RERANK_TOKEN_DECISION_TOP_K must be >= 1")
        if self.timeout <= 0:
            raise ValueError("RERANK_TOKEN_DECISION_TIMEOUT must be > 0")
        self.eos_token_id = _eos_token_id(vllm_config)
        self._rows: dict[int, RequestState] = {}
        self._batch_size = 0

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        extra_args = sampling_params.extra_args
        if not extra_args or "joint_decode_rid" not in extra_args:
            raise ValueError("SamplingParams.extra_args must include joint_decode_rid")

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return

        for index in batch_update.removed:
            self._rows.pop(index, None)

        for index, params, _, output_token_ids in batch_update.added:
            extra_args = params.extra_args
            if not extra_args or "joint_decode_rid" not in extra_args:
                raise ValueError("SamplingParams.extra_args must include joint_decode_rid")
            self._rows[index] = RequestState(
                rid=str(extra_args["joint_decode_rid"]),
                output_token_ids=output_token_ids,
            )

        for source, dest, direction in batch_update.moved:
            source_state = self._rows.get(source)
            dest_state = self._rows.get(dest)
            if source_state is None:
                raise RuntimeError(f"vLLM moved missing source row {source} -> {dest}")
            self._rows[dest] = source_state
            if direction == MoveDirectionality.SWAP:
                if dest_state is None:
                    raise RuntimeError(f"vLLM swapped source row {source} with empty row {dest}")
                self._rows[source] = dest_state
            elif direction == MoveDirectionality.UNIDIRECTIONAL:
                self._rows.pop(source, None)
            else:
                raise RuntimeError(f"unknown vLLM move direction {direction!r}")

        self._batch_size = batch_update.batch_size
        for index in tuple(self._rows):
            if index >= self._batch_size:
                del self._rows[index]

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"expected 2D logits tensor, got shape={tuple(logits.shape)}")
        rids_by_row: list[str] = []
        for row in range(logits.shape[0]):
            state = self._rows.get(row)
            if state is None:
                raise RuntimeError(f"missing request state for logits row {row}")
            rids_by_row.append(state.rid)

        forced_by_rid, undecided = split_pending(rids_by_row)
        if undecided:
            k = min(self.top_k, logits.shape[-1])
            top_values, top_indices = torch.topk(logits, k=k, dim=-1)
            top_values_cpu = top_values.detach().cpu()
            top_indices_cpu = top_indices.detach().cpu()
            undecided_set = set(undecided)
            topk_by_rid: dict[str, list[dict[str, int | float]]] = {}
            for row, rid in enumerate(rids_by_row):
                if rid not in undecided_set:
                    continue
                topk_by_rid[rid] = [
                    {
                        "token_id": int(token_id),
                        "logit": float(logit),
                    }
                    for token_id, logit in zip(
                        top_indices_cpu[row].tolist(),
                        top_values_cpu[row].tolist(),
                        strict=True,
                    )
                ]
            forced_by_rid.update(
                resolve_undecided(
                    topk_by_rid,
                    decision_url=self.decision_url,
                    side=self.side,
                    timeout=self.timeout,
                    eos_token_id=self.eos_token_id,
                )
            )

        forced = torch.tensor(
            [forced_by_rid[rid] for rid in rids_by_row],
            dtype=torch.long,
            device=logits.device,
        ).unsqueeze(-1)
        out = torch.full_like(logits, -float("inf"))
        out.scatter_(-1, forced, 0.0)
        return out


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"{name} must be set")
    return value


def _eos_token_id(vllm_config: Any) -> int:
    eos_token_id = vllm_config.model_config.hf_config.eos_token_id
    if isinstance(eos_token_id, list):
        if not eos_token_id:
            raise RuntimeError("model EOS token list must not be empty")
        eos_token_id = eos_token_id[0]
    if eos_token_id is None:
        raise RuntimeError("model config must define eos_token_id")
    return int(eos_token_id)
