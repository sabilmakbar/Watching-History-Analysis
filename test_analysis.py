from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from pathlib import Path

from watch_history.analysis import (
    EXPECTED_SOURCE_SHA256,
    SOURCE_FILENAME,
    assert_row_accounting,
    assert_source_hash,
    build_analysis,
    file_sha256,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / SOURCE_FILENAME
RESULT_FILES = (ROOT / "results" / "data_quality.json", ROOT / "results" / "eda.json")


def expect_assertion(action, message: str) -> None:
    try:
        action()
    except AssertionError:
        return
    raise AssertionError(message)


def result_hashes() -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in RESULT_FILES
    }


def main() -> None:
    before = file_sha256(SOURCE)
    assert before == EXPECTED_SOURCE_SHA256
    expect_assertion(
        lambda: assert_source_hash(SOURCE, "0" * 64),
        "a deliberately wrong source hash did not fail",
    )

    quality, eda = build_analysis(ROOT)
    first_hashes = result_hashes()
    quality_again, eda_again = build_analysis(ROOT)
    assert result_hashes() == first_hashes
    assert quality_again == quality
    assert eda_again == eda
    assert file_sha256(SOURCE) == before

    assert quality["records"] == {
        "parsed": 106823,
        "accepted": 106811,
        "rejected": 12,
    }
    assert len(quality["rejected_records"]) == 12
    assert all(
        set(record) == {"logical_record_number", "source_line_start", "reason"}
        for record in quality["rejected_records"]
    )
    assert quality["coercion"]["numeric"]["average_bitrate"]["missing"] == 4946
    for kind in ("timestamps", "numeric", "booleans"):
        assert all(
            counts["parse_failures"] == 0
            for counts in quality["coercion"][kind].values()
        )

    tampered = copy.deepcopy(quality)
    tampered["records"]["accepted"] += 1
    expect_assertion(
        lambda: assert_row_accounting(tampered),
        "a deliberately broken accounting equation did not fail",
    )

    assert set(eda) == {
        "source_sha256",
        "daily_volume",
        "hourly_volume",
        "platform_content_mix",
        "completion",
        "audience_context",
        "playback_quality_percentiles",
    }
    assert sum(row["events"] for row in eda["daily_volume"]) == 106811
    assert sum(row["events"] for row in eda["platform_content_mix"]) == 106811
    assert len(eda["hourly_volume"]) == 24
    assert eda["completion"]["eligible_events"] == 67874
    assert sum(
        row["eligible_events"] for row in eda["completion"]["by_category_name"]
    ) == 67874
    assert set(eda["audience_context"]) == {"is_login", "is_premium", "has_ad"}

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = {item.split(">=")[0].lower() for item in project["project"]["dependencies"]}
    assert dependencies == {"pandas", "plotly", "scikit-learn"}
    for path in RESULT_FILES:
        json.loads(path.read_text(encoding="utf-8"))

    print("foundation checks passed")


if __name__ == "__main__":
    main()
