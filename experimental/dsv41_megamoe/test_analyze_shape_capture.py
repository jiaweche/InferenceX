import json

import pytest

from experimental.dsv41_megamoe.analyze_shape_capture import _load_events, summarize


def _event(*, raw_m, padded_m, topk_ids, per_expert_m):
    return {
        "component": "backbone",
        "phase": "decode",
        "layer": "model.layers.0.ffn.experts",
        "raw_m": raw_m,
        "padded_m": padded_m,
        "mega_padded_m": 1 if raw_m == 1 else 2,
        "topk": 2,
        "expert_count": 4,
        "invalid_route_count": 0,
        "raw_topk_ids_captured": True,
        "topk_ids": topk_ids,
        "per_expert_m": per_expert_m,
    }


def test_summarize_aggregates_bucket_and_expert_frequencies():
    summary = summarize(
        [
            _event(
                raw_m=2,
                padded_m=4,
                topk_ids=[[0, 1], [1, 2]],
                per_expert_m=[1, 2, 1, 0],
            ),
            _event(
                raw_m=1,
                padded_m=1,
                topk_ids=[[1, 3]],
                per_expert_m=[0, 1, 0, 1],
            ),
        ]
    )

    backbone = summary["components"]["backbone"]
    assert summary["event_count"] == 2
    assert backbone["aggregate_route_count"] == 6
    assert backbone["raw_topk_event_count"] == 2
    assert backbone["aggregate_expert_routes"] == [1, 3, 1, 1]
    assert backbone["top_experts"][0] == {
        "expert": 1,
        "routes": 3,
        "route_fraction": 0.5,
    }
    assert {(row["raw_m"], row["event_fraction"]) for row in backbone["buckets"]} == {
        (1, 0.5),
        (2, 0.5),
    }
    assert {row["raw_m"]: row["route_fraction"] for row in backbone["buckets"]} == {
        1: pytest.approx(1 / 3),
        2: pytest.approx(2 / 3),
    }


def test_summarize_rejects_incomplete_route_counts():
    event = _event(
        raw_m=1,
        padded_m=1,
        topk_ids=[[0, 1]],
        per_expert_m=[1, 0, 0, 0],
    )

    with pytest.raises(ValueError, match="do not cover all routes"):
        summarize([event])


def test_summarize_accepts_count_only_nonrepresentative_layer():
    event = _event(
        raw_m=1,
        padded_m=1,
        topk_ids=[[0, 1]],
        per_expert_m=[1, 1, 0, 0],
    )
    event["raw_topk_ids_captured"] = False
    event["topk_ids"] = None

    summary = summarize([event])

    assert summary["components"]["backbone"]["aggregate_expert_routes"] == [1, 1, 0, 0]


def test_load_events_filters_and_persists_exact_time_window(tmp_path):
    events = []
    for timestamp in (10, 20, 30):
        event = _event(
            raw_m=1,
            padded_m=1,
            topk_ids=[[0, 1]],
            per_expert_m=[1, 1, 0, 0],
        )
        event["time_ns"] = timestamp
        events.append(event)
    capture = tmp_path / "capture.jsonl"
    capture.write_text("".join(json.dumps(event) + "\n" for event in events))
    filtered = tmp_path / "filtered.jsonl"

    loaded = _load_events(
        capture,
        start_ns=15,
        end_ns=30,
        filtered_output=filtered,
    )

    assert [event["time_ns"] for event in loaded] == [20]
    assert json.loads(filtered.read_text())["time_ns"] == 20
