from __future__ import annotations

from typing import Any

from joint_decode.decision import resolve_undecided, split_pending
from joint_decode.runtime_state import runtime_state


class DecisionClient:
    """Per-step token-decision callback registered with the tpu-inference
    runner hook; mirrors the GPU logits processor's protocol logic."""

    def __init__(self, *, decision_url: str, side: str, timeout: float, eos_token_id: int) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        self.decision_url = decision_url
        self.side = side
        self.timeout = timeout
        self.eos_token_id = eos_token_id

    def __call__(
        self,
        req_ids: list[str],
        topk_by_rid: dict[str, list[dict[str, Any]]],
    ) -> dict[str, int]:
        unknown = sorted(set(req_ids) - runtime_state.live_rids)
        if unknown:
            raise RuntimeError(
                f"token decision callback received unknown request ids {unknown}; "
                "is VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 set?"
            )
        forced, undecided = split_pending(list(req_ids))
        if undecided:
            forced.update(
                resolve_undecided(
                    {rid: topk_by_rid[rid] for rid in undecided},
                    decision_url=self.decision_url,
                    side=self.side,
                    timeout=self.timeout,
                    eos_token_id=self.eos_token_id,
                )
            )
        return forced
