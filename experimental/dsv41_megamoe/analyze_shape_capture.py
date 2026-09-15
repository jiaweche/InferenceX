#!/usr/bin/env python3
"""Validate and summarize DeepSeek V4.1 MoE routing JSONL."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_events(
    path: Path,
    *,
    start_ns: int | None = None,
    end_ns: int | None = None,
    filtered_output: Path | None = None,
) -> list[dict[str, Any]]:
    events = []
    output = filtered_output.open("w") if filtered_output is not None else None
    try:
        with path.open() as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                timestamp = int(event["time_ns"])
                if start_ns is not None and timestamp < start_ns:
                    continue
                if end_ns is not None and timestamp >= end_ns:
                    continue
                events.append(event)
                if output is not None:
                    output.write(line)
    finally:
        if output is not None:
            output.close()
    if not events:
        raise ValueError(f"{path}: capture contains no events")
    return events


def _validate_event(event: dict[str, Any], index: int) -> None:
    raw_m = int(event["raw_m"])
    padded_m = int(event["padded_m"])
    topk = int(event["topk"])
    expert_count = int(event["expert_count"])
    topk_ids = event["topk_ids"]
    per_expert_m = event["per_expert_m"]
    if raw_m < 0 or padded_m < raw_m:
        raise ValueError(f"event {index}: invalid M pair {raw_m}/{padded_m}")
    raw_ids_captured = bool(
        event.get("raw_topk_ids_captured", topk_ids is not None)
    )
    if raw_ids_captured:
        if len(topk_ids) != raw_m or any(len(row) != topk for row in topk_ids):
            raise ValueError(f"event {index}: raw top-k shape disagrees with M/topk")
    elif topk_ids is not None:
        raise ValueError(f"event {index}: uncaptured raw top-k IDs must be null")
    if len(per_expert_m) != expert_count:
        raise ValueError(f"event {index}: per-expert vector has wrong length")
    if int(event["invalid_route_count"]) != 0:
        raise ValueError(f"event {index}: invalid expert IDs were captured")
    if sum(int(value) for value in per_expert_m) != raw_m * topk:
        raise ValueError(f"event {index}: per-expert counts do not cover all routes")


def _coefficient_of_variation(values: list[int]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, event in enumerate(events):
        _validate_event(event, index)
        components[str(event["component"])].append(event)

    output: dict[str, Any] = {
        "schema_version": 1,
        "event_count": len(events),
        "components": {},
    }
    for component, component_events in sorted(components.items()):
        expert_count = int(component_events[0]["expert_count"])
        aggregate_expert_routes = [0] * expert_count
        buckets: dict[tuple[Any, ...], dict[str, Any]] = {}
        for event in component_events:
            if int(event["expert_count"]) != expert_count:
                raise ValueError(f"{component}: inconsistent expert count")
            per_expert_m = [int(value) for value in event["per_expert_m"]]
            aggregate_expert_routes = [
                total + current
                for total, current in zip(aggregate_expert_routes, per_expert_m)
            ]
            key = (
                event["phase"],
                int(event["raw_m"]),
                int(event["padded_m"]),
                int(event["mega_padded_m"]),
                int(event["topk"]),
            )
            bucket = buckets.setdefault(
                key,
                {
                    "phase": key[0],
                    "raw_m": key[1],
                    "padded_m": key[2],
                    "mega_padded_m": key[3],
                    "topk": key[4],
                    "event_count": 0,
                    "route_count": 0,
                    "max_expert_m_observed": 0,
                    "max_expert_fraction_observed": 0.0,
                    "sum_max_expert_m": 0,
                    "sum_max_expert_fraction": 0.0,
                    "sum_active_experts": 0,
                },
            )
            max_expert_m = max(per_expert_m, default=0)
            max_expert_fraction = (
                max_expert_m / int(event["raw_m"]) if int(event["raw_m"]) else 0.0
            )
            bucket["event_count"] += 1
            bucket["route_count"] += int(event["raw_m"]) * int(event["topk"])
            bucket["max_expert_m_observed"] = max(
                bucket["max_expert_m_observed"], max_expert_m
            )
            bucket["max_expert_fraction_observed"] = max(
                bucket["max_expert_fraction_observed"], max_expert_fraction
            )
            bucket["sum_max_expert_m"] += max_expert_m
            bucket["sum_max_expert_fraction"] += max_expert_fraction
            bucket["sum_active_experts"] += sum(value > 0 for value in per_expert_m)

        total_routes = sum(aggregate_expert_routes)
        bucket_rows = []
        for bucket in sorted(
            buckets.values(),
            key=lambda row: (
                row["phase"],
                row["raw_m"],
                row["padded_m"],
            ),
        ):
            count = bucket.pop("event_count")
            sum_max = bucket.pop("sum_max_expert_m")
            sum_max_fraction = bucket.pop("sum_max_expert_fraction")
            sum_active = bucket.pop("sum_active_experts")
            bucket_rows.append(
                {
                    **bucket,
                    "event_count": count,
                    "event_fraction": count / len(component_events),
                    "route_fraction": (
                        bucket["route_count"] / total_routes if total_routes else 0.0
                    ),
                    "mean_max_expert_m": sum_max / count,
                    "mean_max_expert_fraction": sum_max_fraction / count,
                    "mean_active_experts": sum_active / count,
                }
            )

        top_experts = sorted(
            enumerate(aggregate_expert_routes),
            key=lambda pair: (-pair[1], pair[0]),
        )[:10]
        output["components"][component] = {
            "event_count": len(component_events),
            "raw_topk_event_count": sum(
                bool(event.get("raw_topk_ids_captured"))
                for event in component_events
            ),
            "layers": sorted({str(event["layer"]) for event in component_events}),
            "expert_count": expert_count,
            "aggregate_route_count": total_routes,
            "aggregate_route_cv": _coefficient_of_variation(aggregate_expert_routes),
            "top_experts": [
                {
                    "expert": expert,
                    "routes": routes,
                    "route_fraction": routes / total_routes if total_routes else 0.0,
                }
                for expert, routes in top_experts
            ],
            "buckets": bucket_rows,
            "aggregate_expert_routes": aggregate_expert_routes,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--start-ns", type=int)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--filtered-output", type=Path)
    args = parser.parse_args()
    if (args.start_ns is None) != (args.duration_seconds is None):
        parser.error("--start-ns and --duration-seconds must be provided together")
    if args.duration_seconds is not None and args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")
    end_ns = (
        None
        if args.start_ns is None
        else args.start_ns + int(args.duration_seconds * 1_000_000_000)
    )

    summary = summarize(
        _load_events(
            args.capture,
            start_ns=args.start_ns,
            end_ns=end_ns,
            filtered_output=args.filtered_output,
        )
    )
    if args.start_ns is not None:
        summary["window"] = {
            "start_ns": args.start_ns,
            "end_ns": end_ns,
            "duration_seconds": args.duration_seconds,
        }
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
