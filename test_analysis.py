from __future__ import annotations

import copy
import hashlib
import json
import re
import tomllib
from pathlib import Path

from watch_history.analysis import (
    EXPECTED_SOURCE_SHA256,
    FORBIDDEN_MODEL_FEATURES,
    MODEL_FEATURES,
    PLOTLY_CDN,
    REPORT_TITLE,
    SOURCE_FILENAME,
    assert_model_features,
    assert_row_accounting,
    assert_source_hash,
    build_analysis,
    file_sha256,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / SOURCE_FILENAME
RESULT_FILES = (
    ROOT / "results" / "data_quality.json",
    ROOT / "results" / "eda.json",
    ROOT / "results" / "model_metrics.json",
    ROOT / "docs" / "index.html",
)
REQUIRED_EDA_SECTIONS = (
    "daily-volume",
    "hourly-volume",
    "platform-mix",
    "content-mix",
    "completion-rates",
    "audience-context",
    "playback-quality",
)


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


def assert_report_order(report: str) -> None:
    model_position = report.index('<section id="model">')
    assert all(
        report.index(f'<section id="{section}">') < model_position
        for section in REQUIRED_EDA_SECTIONS
    ), "every EDA section must appear before model evidence"


def main() -> None:
    before = file_sha256(SOURCE)
    assert before == EXPECTED_SOURCE_SHA256
    assert before == "83ba4a0ae4e965e61ca44c4c3e444218d07cc8b8b3cec4a2029ab51b6a12c09c"
    expect_assertion(
        lambda: assert_source_hash(SOURCE, "0" * 64),
        "a deliberately wrong source hash did not fail",
    )

    quality, eda, model = build_analysis(ROOT)
    first_hashes = result_hashes()
    quality_again, eda_again, model_again = build_analysis(ROOT)
    assert result_hashes() == first_hashes
    assert quality_again == quality
    assert eda_again == eda
    assert model_again == model
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

    expected_features = (
        "playback_location",
        "platform",
        "referrer_group",
        "content_type",
        "category_name",
        "player_name",
        "autoplay",
        "is_login",
        "is_premium",
        "os_name",
        "browser_name",
        "hour",
        "day_of_week",
    )
    expected_forbidden = {
        "completed",
        "play_duration",
        "end_time",
        "total_bytes",
        "buffer_duration",
        "average_bitrate",
        "bitrate_range",
        "has_ad",
    }
    assert MODEL_FEATURES == expected_features
    assert set(FORBIDDEN_MODEL_FEATURES) == expected_forbidden
    assert tuple(model["features"]) == expected_features
    assert not set(model["features"]).intersection(FORBIDDEN_MODEL_FEATURES)
    assert model["label"] == "completed"
    assert model["eligible_content_types"] == ["catchup", "vod"]
    split = model["split"]
    assert split["sort_order"] == ["play_time", "logical_record_number"]
    assert split["train_rows"] == int(67874 * 0.8)
    assert split["train_rows"] + split["test_rows"] == 67874
    dummy = model["models"]["dummy_prior"]
    logistic = model["models"]["logistic_regression"]
    assert logistic["roc_auc"] >= 0.65
    assert logistic["pr_auc"] - dummy["pr_auc"] >= 0.10
    assert 0 <= logistic["brier_score"] <= 1
    assert model["top_coefficients"]["positive"]
    assert model["top_coefficients"]["negative"]
    expect_assertion(
        lambda: assert_model_features(MODEL_FEATURES + ("completed",)),
        "a deliberately leaked model feature did not fail",
    )

    report = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    figure_count = report.count("Plotly.newPlot(")
    assert figure_count >= 7
    assert report.count('"responsive": true') == figure_count
    assert PLOTLY_CDN == "https://cdn.plot.ly/plotly-4.0.0.min.js"
    assert re.findall(r'<script[^>]+src="([^"]+)"', report) == [PLOTLY_CDN]
    assert report.count(PLOTLY_CDN) == 1
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in report
    assert f"<h1>{REPORT_TITLE}</h1>" in report
    assert_report_order(report)
    model_position = report.index('<section id="model">')
    assert all(
        report.index(f'<section id="{section}">') > model_position
        for section in ("method", "findings", "caveats")
    )
    report_lower = report.lower()
    for phrase in (
        "denominator",
        "utc",
        "robust outlier treatment",
        "observational associations",
        "provenance remains limited",
        "without a local server",
    ):
        assert phrase in report_lower
    broken_order = '<section id="model"></section>' + "".join(
        f'<section id="{section}"></section>' for section in REQUIRED_EDA_SECTIONS
    )
    expect_assertion(
        lambda: assert_report_order(broken_order),
        "a deliberately broken report order did not fail",
    )

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = {item.split(">=")[0].lower() for item in project["project"]["dependencies"]}
    assert dependencies == {"pandas", "plotly", "scikit-learn"}
    for path in RESULT_FILES:
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for heading in ("Setup", "Build", "Validation", "Results", "Source", "Publishing"):
        assert f"## {heading}" in readme
    assert "https://sabilmakbar.github.io/Watching-History-Analysis/" in readme
    assert "upstream publisher and license" in readme.lower()

    print("analysis checks passed")


if __name__ == "__main__":
    main()
