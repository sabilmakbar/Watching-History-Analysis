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
TOPIC_STORIES = {
    "viewing-time-story": ("daily-volume-chart", "hourly-volume-chart"),
    "platform-content-story": ("platform-mix-chart", "content-mix-chart"),
    "completion-story": ("completion-platform-chart", "completion-category-chart"),
    "audience-story": ("audience-context-chart",),
    "quality-story": ("playback-quality-chart",),
    "model-story": ("model-metrics-chart", "model-coefficients-chart"),
}


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


def assert_topic_story(report: str, story_id: str, chart_ids: tuple[str, ...]) -> None:
    start = report.index(f'<article id="{story_id}"')
    end = report.index("</article>", start)
    story = report[start:end]
    question = story.index('class="topic-question"')
    evidence = story.index('class="topic-evidence"')
    implication = story.index('class="topic-implication"')
    charts = [story.index(f'id="{chart_id}"') for chart_id in chart_ids]
    assert question < min(charts) <= max(charts) < evidence < implication
    assert "<strong>Question:</strong>" in story
    assert "<strong>Evidence:</strong>" in story
    assert "<strong>Next question:</strong>" in story


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
    assert figure_count == 10
    assert report.count('"responsive": true') == figure_count
    assert PLOTLY_CDN == "https://cdn.plot.ly/plotly-4.0.0.min.js"
    assert re.findall(r'<script[^>]+src="([^"]+)"', report) == [PLOTLY_CDN]
    assert report.count(PLOTLY_CDN) == 1
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in report
    assert f"<h1>{REPORT_TITLE}</h1>" in report
    assert_report_order(report)
    executive_position = report.index('<section id="executive-summary">')
    first_eda_position = report.index('<section id="daily-volume">')
    model_position = report.index('<section id="model">')
    findings_position = report.index('<section id="findings">')
    method_position = report.index('<section id="method">')
    caveats_position = report.index('<section id="caveats">')
    assert executive_position < first_eda_position
    assert model_position < findings_position < method_position < caveats_position
    assert "<h2>Problem and decision context</h2>" in report
    assert "<h3>Data and approach</h3>" in report
    assert "<h2>What we found</h2>" in report
    assert '<ol class="executive-findings">' in report
    assert report.count('class="topic-question"') == len(TOPIC_STORIES)
    assert report.count('class="topic-evidence"') == len(TOPIC_STORIES)
    assert report.count('class="topic-implication"') == len(TOPIC_STORIES)
    for story_id, chart_ids in TOPIC_STORIES.items():
        assert_topic_story(report, story_id, chart_ids)
    quality_story = report[
        report.index('<article id="quality-story"') : report.index(
            "</article>", report.index('<article id="quality-story"')
        )
    ]
    assert "viewing-experience signal" in quality_story
    assert "fixed at 300,000 from the median through p99" in quality_story
    assert "1,035 at p95" in quality_story
    findings = report[findings_position:method_position]
    assert findings.count("<li>") == 6
    for evidence in (
        "57.8%",
        "82.1%",
        "13:00–14:00 UTC",
        "32.1%",
        "23.7%",
        "31.2%",
        "24.0%",
        "35.0%",
        "21,283",
        "19.7%",
        "21,846",
        "0.218",
        "0.365",
        "0.172",
        "0.158",
        "300,000",
        "1,035",
    ):
        assert evidence in findings
    report_lower = report.lower()
    for phrase in (
        "denominator",
        "utc",
        "robust outlier treatment",
        "observational associations",
        "provenance remains limited",
        "without a local server",
        "event-level associations from a short export",
        "not causal or user-level effects",
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
    for evidence in ("57.8%", "82.1%", "32.1%", "23.7%", "0.218", "0.365"):
        assert evidence in readme
    assert "event-level associations from a short export" in readme

    print("analysis checks passed")


if __name__ == "__main__":
    main()
