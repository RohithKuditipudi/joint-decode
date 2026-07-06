"""Per-step token decision logic, shared by every interception backend.

The GPU logits processor and the TPU decision callback are thin wrappers over
these two functions: consume queued pending tokens locally, post one decode
round for the remaining rows, and apply the coordinator's response.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from joint_decode.runtime_state import runtime_state


def split_pending(rids: list[str]) -> tuple[dict[str, int], list[str]]:
    """Consume one queued pending token per rid; return (forced, undecided)."""
    forced: dict[str, int] = {}
    undecided: list[str] = []
    for rid in rids:
        pending = runtime_state.pending_tokens.get(rid)
        if pending:
            forced[rid] = pending.pop(0)
            if not pending:
                runtime_state.pending_tokens.pop(rid, None)
        else:
            undecided.append(rid)
    return forced, undecided


def resolve_undecided(
    topk_by_rid: dict[str, list[dict[str, int | float]]],
    *,
    decision_url: str,
    side: str,
    timeout: float,
    eos_token_id: int,
) -> dict[str, int]:
    """Post one decode round for the given rows and apply the response.

    Returns a forced token for every posted rid. Multi-token tails are queued
    in runtime state, admits are published to the worker loop, force-stopped
    rids are forced to EOS, and an abort or an undecided row raises.
    """
    request_ids = list(topk_by_rid)
    response = post_decision(
        decision_url,
        {
            "kind": "decode",
            "side": side,
            "request_ids": request_ids,
            "topk": topk_by_rid,
        },
        timeout=timeout,
    )

    abort = response.get("abort")
    if abort:
        runtime_state.publish_commands(abort=str(abort))
        raise RuntimeError(str(abort))
    runtime_state.publish_commands(admit=response.get("admit") or [])

    forced: dict[str, int] = {}
    for rid, token_list in (response.get("tokens") or {}).items():
        tokens = [token_list] if isinstance(token_list, int) else list(token_list)
        if not tokens:
            raise RuntimeError(f"coordinator returned an empty token list for rid={rid}")
        forced[rid] = int(tokens.pop(0))
        if tokens:
            runtime_state.pending_tokens[rid] = [int(token) for token in tokens]
        else:
            runtime_state.pending_tokens.pop(rid, None)

    decoded_rids = set(request_ids)
    for rid in response.get("force_stop") or []:
        if rid in decoded_rids:
            forced[rid] = eos_token_id
            runtime_state.pending_tokens.pop(rid, None)

    missing = decoded_rids - set(forced)
    if missing:
        raise RuntimeError(f"coordinator did not return tokens or force_stop for rids={sorted(missing)}")
    return forced


def post_decision(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"joint-decode decision request failed: {exc}") from exc
