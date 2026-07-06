from __future__ import annotations

import random

import pytest

from joint_decode.selection import select_avg_logits, select_top_rank


def _topk(pairs: list[tuple[int, float]]) -> list[dict[str, float | int]]:
    return [{"token_id": token_id, "logit": logit} for token_id, logit in pairs]


def test_select_top_rank_prefers_highest_b_rank_among_a_topk() -> None:
    a = _topk([(1, 5.0), (2, 4.0), (3, 3.0)])
    b = _topk([(9, 9.0), (3, 8.0), (2, 7.0)])

    assert select_top_rank(a, b, rng=random.Random(0)) == 3


def test_select_top_rank_falls_back_to_a_top1_without_overlap() -> None:
    a = _topk([(1, 5.0), (2, 4.0)])
    b = _topk([(8, 9.0), (9, 8.0)])

    assert select_top_rank(a, b, rng=random.Random(0)) == 1


def test_select_top_rank_rejects_empty_a() -> None:
    with pytest.raises(ValueError, match="empty top-k"):
        select_top_rank([], _topk([(1, 1.0)]), rng=random.Random(0))


def test_select_avg_logits_greedy_argmax_at_zero_temperature() -> None:
    a = _topk([(1, 10.0), (2, 0.0)])
    b = _topk([(2, 10.0), (1, 0.0)])

    tokens_a, tokens_b = select_avg_logits(a, b, advisor_weight=1.0, temperature=0.0, rng=random.Random(0))

    assert tokens_a == tokens_b == [2]
