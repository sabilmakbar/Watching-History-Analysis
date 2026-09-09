from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


SOURCE_FILENAME = "part-00000-0dd5f001-57d7-4ec1-a3f0-53ea28bba9c3-c000.csv"
EXPECTED_SOURCE_SHA256 = (
    "83ba4a0ae4e965e61ca44c4c3e444218d07cc8b8b3cec4a2029ab51b6a12c09c"
)
EXPECTED_FIELD_COUNT = 41
TIMESTAMP_COLUMNS = ("play_time", "end_time")
NUMERIC_COLUMNS = (
    "average_bitrate",
    "total_bytes",
    "buffer_duration",
    "play_duration",
)
BOOLEAN_COLUMNS = ("is_login", "completed", "has_ad", "autoplay", "is_premium")
COMPLETION_CONTENT_TYPES = ("catchup", "vod")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_source_hash(
    path: Path, expected_sha256: str = EXPECTED_SOURCE_SHA256
) -> str:
    actual_sha256 = file_sha256(path)
    assert actual_sha256 == expected_sha256, (
        f"source SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
    )
    return actual_sha256


def _coerce_columns(frame: pd.DataFrame) -> dict[str, dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, dict[str, int]]] = {
        "timestamps": {},
        "numeric": {},
        "booleans": {},
    }

    for column in TIMESTAMP_COLUMNS:
        raw = frame[column].str.strip()
        present = raw.ne("")
        parsed = pd.to_datetime(raw.where(present), errors="coerce", utc=True)
        counts["timestamps"][column] = {
            "missing": int((~present).sum()),
            "parse_failures": int((present & parsed.isna()).sum()),
        }
        frame[column] = parsed

    for column in NUMERIC_COLUMNS:
        raw = frame[column].str.strip()
        present = raw.ne("")
        parsed = pd.to_numeric(raw.where(present), errors="coerce")
        counts["numeric"][column] = {
            "missing": int((~present).sum()),
            "parse_failures": int((present & parsed.isna()).sum()),
        }
        frame[column] = parsed

    for column in BOOLEAN_COLUMNS:
        raw = frame[column].str.strip().str.lower()
        present = raw.ne("")
        parsed = raw.where(present).map({"true": True, "false": False}).astype("boolean")
        counts["booleans"][column] = {
            "missing": int((~present).sum()),
            "parse_failures": int((present & parsed.isna()).sum()),
        }
        frame[column] = parsed

    return counts


def load_events(source_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_sha256 = assert_source_hash(source_path)
    accepted: list[list[Any]] = []
    rejected: list[dict[str, Any]] = []

    with source_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        header = next(reader)
        assert len(header) == EXPECTED_FIELD_COUNT, (
            f"expected {EXPECTED_FIELD_COUNT} header fields, found {len(header)}"
        )

        logical_record_number = 0
        while True:
            source_line_start = reader.line_num + 1
            try:
                row = next(reader)
            except StopIteration:
                break

            logical_record_number += 1
            if len(row) != len(header):
                rejected.append(
                    {
                        "logical_record_number": logical_record_number,
                        "source_line_start": source_line_start,
                        "reason": f"expected {len(header)} fields, found {len(row)}",
                    }
                )
                continue
            accepted.append([logical_record_number, *row])

    frame = pd.DataFrame(accepted, columns=["logical_record_number", *header])
    coercion_counts = _coerce_columns(frame)
    quality: dict[str, Any] = {
        "source": {
            "filename": source_path.name,
            "sha256": source_sha256,
            "field_count": len(header),
        },
        "records": {
            "parsed": logical_record_number,
            "accepted": len(frame),
            "rejected": len(rejected),
        },
        "rejected_records": rejected,
        "coercion": coercion_counts,
    }
    assert_row_accounting(quality)
    return frame, quality


def assert_row_accounting(quality: dict[str, Any]) -> None:
    records = quality["records"]
    assert records["parsed"] == records["accepted"] + records["rejected"], (
        "parsed record count must equal accepted plus rejected"
    )
    assert records["rejected"] == len(quality["rejected_records"]), (
        "rejected count must match rejected record details"
    )
    assert all(
        record["logical_record_number"] > 0
        and record["source_line_start"] > 1
        and record["reason"]
        for record in quality["rejected_records"]
    ), "each rejected record must identify its logical record, source line, and reason"


def _completion_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        frame["content_type"].isin(COMPLETION_CONTENT_TYPES)
        & frame["completed"].notna()
    ]


def _completion_by(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    records = []
    for value, group in frame.groupby(column, dropna=False, sort=True):
        denominator = len(group)
        completed = int(group["completed"].sum())
        records.append(
            {
                column: "(missing)" if pd.isna(value) or value == "" else value,
                "eligible_events": denominator,
                "completed_events": completed,
                "completion_rate": round(completed / denominator, 6),
            }
        )
    return records


def _audience_context(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    eligible = _completion_rows(frame)
    records = []
    for value in (False, True, pd.NA):
        if pd.isna(value):
            all_rows = frame[frame[column].isna()]
            eligible_rows = eligible[eligible[column].isna()]
            label: bool | str = "(missing)"
        else:
            all_rows = frame[frame[column].eq(value).fillna(False)]
            eligible_rows = eligible[eligible[column].eq(value).fillna(False)]
            label = value
        if all_rows.empty:
            continue
        denominator = len(eligible_rows)
        completed = int(eligible_rows["completed"].sum()) if denominator else 0
        records.append(
            {
                "value": label,
                "events": len(all_rows),
                "eligible_events": denominator,
                "completed_events": completed,
                "completion_rate": (
                    round(completed / denominator, 6) if denominator else None
                ),
            }
        )
    return records


def build_eda(frame: pd.DataFrame, source_sha256: str) -> dict[str, Any]:
    daily_counts = frame["play_time"].dt.strftime("%Y-%m-%d").value_counts().sort_index()
    hourly_counts = frame["play_time"].dt.hour.value_counts()
    platform_content = (
        frame.groupby(["platform", "content_type"], dropna=False, sort=True)
        .size()
        .reset_index(name="events")
    )
    eligible = _completion_rows(frame)
    percentiles = (0.01, 0.25, 0.5, 0.75, 0.95, 0.99)

    return {
        "source_sha256": source_sha256,
        "daily_volume": [
            {"date_utc": date, "events": int(count)}
            for date, count in daily_counts.items()
        ],
        "hourly_volume": [
            {"hour_utc": hour, "events": int(hourly_counts.get(hour, 0))}
            for hour in range(24)
        ],
        "platform_content_mix": [
            {
                "platform": "(missing)" if pd.isna(row.platform) else row.platform,
                "content_type": (
                    "(missing)" if pd.isna(row.content_type) else row.content_type
                ),
                "events": int(row.events),
            }
            for row in platform_content.itertuples(index=False)
        ],
        "completion": {
            "denominator": "non-missing completed values for vod and catchup only",
            "eligible_events": len(eligible),
            "by_content_type": _completion_by(eligible, "content_type"),
            "by_category_name": _completion_by(eligible, "category_name"),
            "by_platform": _completion_by(eligible, "platform"),
        },
        "audience_context": {
            column: _audience_context(frame, column)
            for column in ("is_login", "is_premium", "has_ad")
        },
        "playback_quality_percentiles": {
            column: {
                "present": int(frame[column].notna().sum()),
                "missing": int(frame[column].isna().sum()),
                **{
                    f"p{int(percentile * 100):02d}": round(
                        float(frame[column].quantile(percentile)), 6
                    )
                    for percentile in percentiles
                },
            }
            for column in NUMERIC_COLUMNS
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_analysis(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    frame, quality = load_events(root / SOURCE_FILENAME)
    eda = build_eda(frame, quality["source"]["sha256"])
    _write_json(root / "results" / "data_quality.json", quality)
    _write_json(root / "results" / "eda.json", eda)
    return quality, eda
