from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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
MODEL_FEATURES = (
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
FORBIDDEN_MODEL_FEATURES = (
    "completed",
    "play_duration",
    "end_time",
    "total_bytes",
    "buffer_duration",
    "average_bitrate",
    "bitrate_range",
    "has_ad",
)
PLOTLY_CDN = "https://cdn.plot.ly/plotly-4.0.0.min.js"
REPORT_TITLE = "Watching History Analysis"
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9")


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


def assert_model_features(features: tuple[str, ...] = MODEL_FEATURES) -> None:
    assert features == MODEL_FEATURES, "model feature list must match the approved list"
    assert not set(features).intersection(FORBIDDEN_MODEL_FEATURES), (
        "model features include a post-start or outcome-derived field"
    )


def _model_metrics(labels: pd.Series, probabilities: Any) -> dict[str, float]:
    return {
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 6),
        "pr_auc": round(float(average_precision_score(labels, probabilities)), 6),
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 6),
    }


def build_completion_model(frame: pd.DataFrame) -> dict[str, Any]:
    assert_model_features()
    eligible = _completion_rows(frame).sort_values(
        ["play_time", "logical_record_number"], kind="stable"
    )
    eligible = eligible.assign(
        hour=eligible["play_time"].dt.hour,
        day_of_week=eligible["play_time"].dt.day_name(),
    )
    features = eligible[list(MODEL_FEATURES)].astype("object")
    features = features.where(features.notna(), float("nan"))
    labels = eligible["completed"].astype(int)
    split_at = int(len(eligible) * 0.8)
    train_x, test_x = features.iloc[:split_at], features.iloc[split_at:]
    train_y, test_y = labels.iloc[:split_at], labels.iloc[split_at:]

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(train_x, train_y)
    dummy_probabilities = dummy.predict_proba(test_x)[:, 1]

    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20),
            ),
        ]
    )
    model = Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [("categorical", categorical, list(MODEL_FEATURES))]
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=500, random_state=42),
            ),
        ]
    )
    model.fit(train_x, train_y)
    model_probabilities = model.predict_proba(test_x)[:, 1]

    encoded_features = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = [
        {
            "feature": str(feature).removeprefix("categorical__"),
            "coefficient": round(float(coefficient), 6),
        }
        for feature, coefficient in zip(
            encoded_features, model.named_steps["classifier"].coef_[0], strict=True
        )
    ]
    positive = sorted(
        (row for row in coefficients if row["coefficient"] > 0),
        key=lambda row: (-row["coefficient"], row["feature"]),
    )[:10]
    negative = sorted(
        (row for row in coefficients if row["coefficient"] < 0),
        key=lambda row: (row["coefficient"], row["feature"]),
    )[:10]

    return {
        "label": "completed",
        "eligible_content_types": list(COMPLETION_CONTENT_TYPES),
        "features": list(MODEL_FEATURES),
        "split": {
            "strategy": "chronological 80/20 holdout",
            "sort_order": ["play_time", "logical_record_number"],
            "train_rows": len(train_x),
            "test_rows": len(test_x),
            "train_label_rate": round(float(train_y.mean()), 6),
            "test_label_rate": round(float(test_y.mean()), 6),
            "train_start_utc": eligible.iloc[0]["play_time"].isoformat(),
            "train_end_utc": eligible.iloc[split_at - 1]["play_time"].isoformat(),
            "test_start_utc": eligible.iloc[split_at]["play_time"].isoformat(),
            "test_end_utc": eligible.iloc[-1]["play_time"].isoformat(),
        },
        "models": {
            "dummy_prior": _model_metrics(test_y, dummy_probabilities),
            "logistic_regression": _model_metrics(test_y, model_probabilities),
        },
        "top_coefficients": {"positive": positive, "negative": negative},
    }


def _style_figure(figure: go.Figure, height: int = 390) -> go.Figure:
    figure.update_layout(
        template=None,
        height=height,
        margin={"l": 60, "r": 24, "t": 72, "b": 58},
        paper_bgcolor="white",
        plot_bgcolor="#f7f8fa",
        font={"family": "Inter, system-ui, sans-serif", "color": "#18212b"},
        hoverlabel={"bgcolor": "white"},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    figure.update_xaxes(gridcolor="#dfe3e8", zeroline=False)
    figure.update_yaxes(gridcolor="#dfe3e8", zeroline=False)
    return figure


def _figure_html(figure: go.Figure, div_id: str) -> str:
    return figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False},
        div_id=div_id,
    )


def _mix_totals(rows: list[dict[str, Any]], column: str) -> tuple[list[str], list[int]]:
    totals: dict[str, int] = {}
    for row in rows:
        label = str(row[column])
        totals[label] = totals.get(label, 0) + row["events"]
    ordered = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    return [item[0] for item in ordered], [item[1] for item in ordered]


def build_report(eda: dict[str, Any], model: dict[str, Any]) -> str:
    figures: dict[str, str] = {}

    daily = eda["daily_volume"]
    figure = go.Figure(
        go.Scatter(
            x=[row["date_utc"] for row in daily],
            y=[row["events"] for row in daily],
            mode="lines+markers",
            name="Events",
            line={"color": COLORS[0], "width": 3},
            hovertemplate="%{x}<br>%{y:,} events<extra></extra>",
        )
    )
    figure.update_layout(
        title="Daily playback volume", xaxis_title="Date (UTC)", yaxis_title="Events"
    )
    figures["daily"] = _figure_html(_style_figure(figure), "daily-volume-chart")

    hourly = eda["hourly_volume"]
    figure = go.Figure(
        go.Bar(
            x=[row["hour_utc"] for row in hourly],
            y=[row["events"] for row in hourly],
            name="Events",
            marker={"color": COLORS[1], "pattern": {"shape": "/"}},
            hovertemplate="%{x}:00 UTC<br>%{y:,} events<extra></extra>",
        )
    )
    figure.update_layout(
        title="Playback by hour",
        xaxis_title="Hour of day (UTC)",
        yaxis_title="Events",
    )
    figures["hourly"] = _figure_html(_style_figure(figure), "hourly-volume-chart")

    platforms, platform_events = _mix_totals(eda["platform_content_mix"], "platform")
    figure = go.Figure(
        go.Bar(
            x=platforms,
            y=platform_events,
            marker={"color": COLORS[2], "pattern": {"shape": "."}},
            text=[f"{value:,}" for value in platform_events],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:,} events<extra></extra>",
        )
    )
    figure.update_layout(
        title="Platform mix", xaxis_title="Platform", yaxis_title="Events"
    )
    figures["platform"] = _figure_html(_style_figure(figure), "platform-mix-chart")

    content_types, content_events = _mix_totals(
        eda["platform_content_mix"], "content_type"
    )
    figure = go.Figure(
        go.Bar(
            x=content_types,
            y=content_events,
            marker={"color": COLORS[3], "pattern": {"shape": "x"}},
            text=[f"{value:,}" for value in content_events],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:,} events<extra></extra>",
        )
    )
    figure.update_layout(
        title="Content mix", xaxis_title="Content type", yaxis_title="Events"
    )
    figures["content"] = _figure_html(_style_figure(figure), "content-mix-chart")

    completion_platform = sorted(
        eda["completion"]["by_platform"], key=lambda row: row["completion_rate"]
    )
    figure = go.Figure(
        go.Bar(
            x=[row["completion_rate"] for row in completion_platform],
            y=[row["platform"] for row in completion_platform],
            orientation="h",
            marker={"color": COLORS[0], "pattern": {"shape": "/"}},
            text=[
                f"{row['completion_rate']:.1%} · n={row['eligible_events']:,}"
                for row in completion_platform
            ],
            textposition="outside",
            hovertemplate="%{y}<br>%{x:.1%} completed<extra></extra>",
        )
    )
    figure.update_layout(
        title="Completion rate by platform",
        xaxis_title="Completion rate",
        yaxis_title="Platform",
    )
    figure.update_xaxes(
        tickformat=".0%",
        range=[
            0,
            max(row["completion_rate"] for row in completion_platform) * 1.35,
        ],
    )
    figures["completion_platform"] = _figure_html(
        _style_figure(figure), "completion-platform-chart"
    )

    completion_category = sorted(
        eda["completion"]["by_category_name"],
        key=lambda row: row["completion_rate"],
    )
    figure = go.Figure(
        go.Bar(
            x=[row["completion_rate"] for row in completion_category],
            y=[row["category_name"] for row in completion_category],
            orientation="h",
            marker={"color": COLORS[1], "pattern": {"shape": "."}},
            customdata=[row["eligible_events"] for row in completion_category],
            hovertemplate="%{y}<br>%{x:.1%} completed<br>n=%{customdata:,}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Completion rate by content category",
        xaxis_title="Completion rate",
        yaxis_title="Category",
    )
    figure.update_xaxes(tickformat=".0%")
    figures["completion_category"] = _figure_html(
        _style_figure(figure, 520), "completion-category-chart"
    )

    figure = go.Figure()
    context_labels = {
        "is_login": "Logged in",
        "is_premium": "Premium",
        "has_ad": "Ad present",
    }
    for index, (column, rows) in enumerate(eda["audience_context"].items()):
        usable = [row for row in rows if row["completion_rate"] is not None]
        figure.add_bar(
            name=context_labels[column],
            x=[
                "No"
                if row["value"] is False
                else "Yes"
                if row["value"] is True
                else "Missing"
                for row in usable
            ],
            y=[row["completion_rate"] for row in usable],
            marker={
                "color": COLORS[index],
                "pattern": {"shape": ("/", ".", "x")[index]},
            },
            text=[
                f"{row['completion_rate']:.1%}<br>n={row['eligible_events']:,}"
                for row in usable
            ],
            hovertemplate=(
                f"{context_labels[column]}: %{{x}}<br>"
                "%{y:.1%} completed<extra></extra>"
            ),
        )
    figure.update_layout(
        title="Audience context and completion",
        xaxis_title="Flag value",
        yaxis_title="Completion rate",
        barmode="group",
    )
    figure.update_yaxes(tickformat=".0%")
    figures["audience"] = _figure_html(
        _style_figure(figure), "audience-context-chart"
    )

    quality = eda["playback_quality_percentiles"]
    percentile_keys = ("p01", "p25", "p50", "p75", "p95", "p99")
    quality_labels = {
        "average_bitrate": "Average bitrate",
        "total_bytes": "Total bytes",
        "buffer_duration": "Buffer duration",
        "play_duration": "Play duration",
    }
    figure = make_subplots(
        rows=2, cols=2, subplot_titles=list(quality_labels.values())
    )
    percentile_labels = ("1st", "25th", "50th", "75th", "95th", "99th")
    for index, (column, label) in enumerate(quality_labels.items()):
        row, col = divmod(index, 2)
        figure.add_trace(
            go.Scatter(
                x=percentile_labels,
                y=[quality[column][key] for key in percentile_keys],
                mode="lines+markers",
                name=label,
                line={"color": COLORS[index], "width": 3},
                hovertemplate=(
                    "%{x} percentile<br>%{y:,.2f} source units<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row + 1,
            col=col + 1,
        )
    figure.update_layout(title="Playback quality, central 98% percentile view")
    figures["quality"] = _figure_html(
        _style_figure(figure, 620), "playback-quality-chart"
    )

    metric_names = ("roc_auc", "pr_auc", "brier_score")
    metric_labels = ("ROC-AUC", "PR-AUC", "Brier score")
    figure = go.Figure()
    model_labels = (
        ("dummy_prior", "Dummy prior"),
        ("logistic_regression", "Logistic Regression"),
    )
    for index, (key, label) in enumerate(model_labels):
        figure.add_bar(
            name=label,
            x=metric_labels,
            y=[model["models"][key][metric] for metric in metric_names],
            marker={
                "color": COLORS[index],
                "pattern": {"shape": ("/", ".")[index]},
            },
            text=[
                f"{model['models'][key][metric]:.3f}" for metric in metric_names
            ],
            textposition="outside",
            hovertemplate=f"{label}<br>%{{x}}: %{{y:.3f}}<extra></extra>",
        )
    figure.update_layout(
        title="Chronological holdout metrics (lower is better for Brier)",
        yaxis_title="Score",
        barmode="group",
    )
    figure.update_yaxes(range=[0, 1])
    figures["metrics"] = _figure_html(
        _style_figure(figure), "model-metrics-chart"
    )

    coefficient_rows = model["top_coefficients"]["negative"][:8] + list(
        reversed(model["top_coefficients"]["positive"][:8])
    )
    figure = go.Figure(
        go.Bar(
            x=[row["coefficient"] for row in coefficient_rows],
            y=[row["feature"] for row in coefficient_rows],
            orientation="h",
            marker={
                "color": [
                    COLORS[3] if row["coefficient"] < 0 else COLORS[2]
                    for row in coefficient_rows
                ],
                "pattern": {
                    "shape": [
                        "x" if row["coefficient"] < 0 else "/"
                        for row in coefficient_rows
                    ]
                },
            },
            text=[f"{row['coefficient']:+.2f}" for row in coefficient_rows],
            hovertemplate="%{y}<br>coefficient %{x:+.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Strongest signed Logistic Regression coefficients",
        xaxis_title="Log-odds coefficient",
        yaxis_title="Encoded start-time feature",
    )
    figures["coefficients"] = _figure_html(
        _style_figure(figure, 600), "model-coefficients-chart"
    )

    busiest_day = max(daily, key=lambda row: row["events"])
    busiest_hour = max(hourly, key=lambda row: row["events"])
    logistic = model["models"]["logistic_regression"]
    dummy_metrics = model["models"]["dummy_prior"]
    split = model["split"]
    total_events = sum(platform_events)
    platform_totals = dict(zip(platforms, platform_events, strict=True))
    hourly_totals = {row["hour_utc"]: row["events"] for row in hourly}
    completion_platforms = {
        row["platform"]: row for row in eda["completion"]["by_platform"]
    }
    completion_categories = {
        row["category_name"]: row
        for row in eda["completion"]["by_category_name"]
    }
    audience = {
        column: {row["value"]: row for row in rows}
        for column, rows in eda["audience_context"].items()
    }
    web_mobile_share = platform_totals["web-mobile"] / total_events
    mobile_share = (
        platform_totals["web-mobile"] + platform_totals["app-android"]
    ) / total_events
    android_completion = completion_platforms["app-android"]
    web_completion = completion_platforms["web-mobile"]
    platform_completion_gap = (
        android_completion["completion_rate"] - web_completion["completion_rate"]
    ) * 100
    news_completion = completion_categories["News"]
    entertainment_completion = completion_categories["Entertainment"]
    logged_in = audience["is_login"][True]
    logged_out = audience["is_login"][False]
    ad_present = audience["has_ad"][True]
    ad_absent = audience["has_ad"][False]
    average_bitrate = quality["average_bitrate"]
    buffer_duration = quality["buffer_duration"]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{REPORT_TITLE} | Interactive Report</title>
  <script src="{PLOTLY_CDN}"></script>
  <style>
    :root {{ color-scheme: light; --ink: #18212b; --muted: #586574; --panel: #fff; --line: #dfe3e8; --accent: #0072b2; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #f2f5f7; color: var(--ink); font-family: Inter, system-ui, sans-serif; line-height: 1.6; }}
    header, main, footer {{ width: min(1120px, calc(100% - 32px)); margin: auto; }}
    header {{ padding: 64px 0 32px; }}
    h1 {{ margin: 0 0 8px; font-size: clamp(2.2rem, 7vw, 4.8rem); line-height: 1; letter-spacing: -.04em; }}
    h2 {{ margin-top: 0; font-size: clamp(1.5rem, 4vw, 2.2rem); }}
    .eyebrow {{ color: var(--accent); font-weight: 750; letter-spacing: .08em; text-transform: uppercase; }}
    .lede {{ max-width: 760px; color: var(--muted); font-size: 1.15rem; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 28px; }}
    .stat, section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 8px 28px rgba(24,33,43,.05); }}
    .stat {{ padding: 18px; }} .stat strong {{ display: block; font-size: 1.7rem; }} .stat span, .note {{ color: var(--muted); }}
    section {{ margin: 18px 0; padding: clamp(20px, 4vw, 38px); }}
    .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 470px), 1fr)); gap: 18px; }}
    .chart-grid section {{ margin: 0; }}
    .callout {{ border-left: 5px solid var(--accent); padding-left: 18px; }}
    .executive-copy {{ max-width: 850px; color: var(--muted); }}
    .executive-findings {{ display: grid; gap: 14px; padding-left: 28px; }}
    .executive-findings li {{ padding: 4px 4px 14px 10px; border-bottom: 1px solid var(--line); }}
    .executive-findings li:last-child {{ border-bottom: 0; }}
    .executive-findings li::marker {{ color: var(--accent); font-size: 1.2rem; font-weight: 800; }}
    .executive-findings strong {{ display: block; color: var(--ink); font-size: 1.05rem; }}
    .implication {{ color: var(--muted); }}
    code {{ overflow-wrap: anywhere; }}
    footer {{ padding: 28px 0 64px; color: var(--muted); }}
    @media (max-width: 620px) {{ header {{ padding-top: 38px; }} section {{ border-radius: 12px; }} }}
  </style>
</head>
<body>
<header>
  <p class="eyebrow">Interactive exploratory analysis</p>
  <h1>{REPORT_TITLE}</h1>
  <p class="lede">An executive view of when playback demand concentrates, where completion differs, and how much start-time context can predict later completion across 106,811 accepted events.</p>
  <div class="stats">
    <div class="stat"><strong>106,811</strong><span>accepted events</span></div>
    <div class="stat"><strong>12</strong><span>malformed records rejected</span></div>
    <div class="stat"><strong>67,874</strong><span>completion-eligible events</span></div>
    <div class="stat"><strong>{logistic['roc_auc']:.3f}</strong><span>Logistic Regression ROC-AUC</span></div>
  </div>
</header>
<main>
  <section id="executive-summary">
    <h2>Problem and decision context</h2>
    <p class="executive-copy">The source repository held a large anonymized playback export but no analysis explaining where viewing demand concentrates, which contexts are associated with completion, or whether those patterns generalize to later events. This report turns the export into an auditable portfolio artifact and highlights where product or data-quality investigation would be most useful.</p>
    <h3>Data and approach</h3>
    <p class="executive-copy">The pipeline parsed 106,823 logical records, accepted 106,811 structurally valid events, and recorded 12 rejections without changing the source. Exploratory analysis covers time, platform, content, audience context, completion, and robust playback-quality percentiles. A leakage-controlled benchmark then compares a prior dummy with Logistic Regression on 67,874 labeled VOD and catch-up events using an 80/20 chronological holdout.</p>
    <p class="callout">The export covers only 1–16 February 2020 in UTC, with partial first and last dates. Treat the results as evidence for investigation within this sample.</p>
  </section>
  <section id="data-trust">
    <h2>Data trust</h2>
    <p>Every parsed logical record is accounted for. The build accepts rows with exactly 41 fields, rejects 12 malformed records with their logical and physical source positions, and verifies the source SHA-256 before analysis.</p>
    <p class="note">All times are UTC. The original repository does not record the upstream publisher or license, so this repository is the available source and provenance remains limited.</p>
  </section>
  <div class="chart-grid">
    <section id="daily-volume"><h2>Daily volume</h2><p>Volume peaks on {busiest_day['date_utc']} at {busiest_day['events']:,} events.</p>{figures['daily']}</section>
    <section id="hourly-volume"><h2>Hourly volume</h2><p>The busiest hour begins at {busiest_hour['hour_utc']:02d}:00 UTC with {busiest_hour['events']:,} events.</p>{figures['hourly']}</section>
    <section id="platform-mix"><h2>Platform mix</h2><p>Hover to compare the devices through which playback began.</p>{figures['platform']}</section>
    <section id="content-mix"><h2>Content mix</h2><p>Livestream events appear here but are excluded from completion rates because their labels are missing.</p>{figures['content']}</section>
  </div>
  <section id="completion-rates">
    <h2>Completion rates</h2>
    <p class="callout">Denominator: only VOD and catch-up events with a defined <code>completed</code> label. Every hover label reports or corresponds to its eligible event count.</p>
    <div class="chart-grid"><div>{figures['completion_platform']}</div><div>{figures['completion_category']}</div></div>
  </section>
  <section id="audience-context">
    <h2>Audience context</h2>
    <p>Login, premium, and ad flags describe observed playback context. Differences are observational associations and do not establish causes.</p>
    {figures['audience']}
  </section>
  <section id="playback-quality">
    <h2>Playback quality</h2>
    <p>Robust outlier treatment uses the 1st through 99th percentiles so extreme raw tails do not hide the main distribution. Values retain the source data's units; 4,946 average-bitrate values are missing.</p>
    {figures['quality']}
  </section>
  <section id="model">
    <h2>Model evidence</h2>
    <p>The benchmark uses {split['train_rows']:,} earlier eligible events for training and {split['test_rows']:,} later events for testing. The test period starts {split['test_start_utc']}; equal timestamps are ordered by retained logical record number.</p>
    <p>Logistic Regression reaches ROC-AUC {logistic['roc_auc']:.3f} and PR-AUC {logistic['pr_auc']:.3f}, compared with dummy PR-AUC {dummy_metrics['pr_auc']:.3f}. Brier score measures probability error and is better when lower.</p>
    {figures['metrics']}
    <p>Coefficients are signed observational associations conditional on the encoded inputs. They are not causal effects, and their scale is in model log-odds.</p>
    {figures['coefficients']}
  </section>
  <section id="findings">
    <h2>What we found</h2>
    <ol class="executive-findings">
      <li><strong>Playback is concentrated on two mobile surfaces.</strong> Web mobile contributes {web_mobile_share:.1%} of all events; together with the Android app it contributes {mobile_share:.1%}. <span class="implication">Experience and reliability work on these two surfaces reaches most events in this export, though event share is not the same as unique-user share.</span></li>
      <li><strong>Demand peaks around 13:00–14:00 UTC.</strong> The 13:00 and 14:00 hours contain {hourly_totals[13]:,} and {hourly_totals[14]:,} events respectively. The first and last dates are partial windows, so their lower daily totals should not be read as demand drops. <span class="implication">Operational monitoring can focus on the midday UTC peak, while date-to-date comparisons should exclude or annotate the boundary days.</span></li>
      <li><strong>Android-app completion is {platform_completion_gap:.1f} percentage points higher than mobile web.</strong> The Android app completes at {android_completion['completion_rate']:.1%} across {android_completion['eligible_events']:,} eligible events versus {web_completion['completion_rate']:.1%} across {web_completion['eligible_events']:,} on mobile web. <span class="implication">The gap is large enough to justify a journey-level comparison, but this export cannot identify whether platform, audience, or content mix explains it.</span></li>
      <li><strong>Login status separates completion; ad presence does not.</strong> Logged-in events complete at {logged_in['completion_rate']:.1%} versus {logged_out['completion_rate']:.1%} for logged-out events, while ad-present and ad-absent rates are effectively flat at {ad_present['completion_rate']:.1%} and {ad_absent['completion_rate']:.1%}. <span class="implication">Login is useful for segmentation and diagnosis here; the ad flag provides no comparable event-level completion signal.</span></li>
      <li><strong>Content category is a major context for completion.</strong> News completes at {news_completion['completion_rate']:.1%} on {news_completion['eligible_events']:,} eligible events versus {entertainment_completion['completion_rate']:.1%} on {entertainment_completion['eligible_events']:,} for Entertainment. <span class="implication">Platform comparisons should control for content mix before they drive product conclusions.</span></li>
      <li><strong>Start-time context adds useful, limited predictive signal.</strong> Logistic Regression raises chronological holdout PR-AUC from {dummy_metrics['pr_auc']:.3f} to {logistic['pr_auc']:.3f} and improves Brier score from {dummy_metrics['brier_score']:.3f} to {logistic['brier_score']:.3f}. <span class="implication">The model is a credible benchmark for ranking and probability quality, not evidence that it is ready for operational decisions.</span></li>
      <li><strong>Playback-quality fields need a data dictionary before operational use.</strong> Average bitrate is fixed at {average_bitrate['p50']:,.0f} from the median through the 99th percentile, while buffer duration jumps from a median of {buffer_duration['p50']:,.0f} to {buffer_duration['p95']:,.0f} at p95. <span class="implication">The apparent bitrate ceiling and large buffer tail may reflect source units, defaults, or capping and should be resolved upstream before thresholds are set.</span></li>
    </ol>
    <p class="callout">These are event-level associations from a short export, not causal or user-level effects.</p>
  </section>
  <section id="method">
    <h2>Method</h2>
    <p>The build validates the source, parses timestamps in UTC, coerces numeric and boolean fields, and creates aggregate results without writing event-level cleaned data. The completion model uses only playback-start context, an 80/20 chronological split, most-frequent categorical imputation, one-hot encoding with rare levels grouped below 20 observations, and Logistic Regression with a prior-probability dummy baseline.</p>
  </section>
  <section id="caveats">
    <h2>Caveats</h2>
    <p>The data covers a short 2020 window, identifiers are anonymized, and the repository does not establish the original publisher or license. Results describe associations in this export; they do not support causal claims, personalized recommendations, or broad population inference.</p>
  </section>
</main>
<footer>Built deterministically from the repository source file. Charts support hover, zoom, and legend controls without a local server.</footer>
</body>
</html>
"""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_analysis(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    frame, quality = load_events(root / SOURCE_FILENAME)
    eda = build_eda(frame, quality["source"]["sha256"])
    model = build_completion_model(frame)
    _write_json(root / "results" / "data_quality.json", quality)
    _write_json(root / "results" / "eda.json", eda)
    _write_json(root / "results" / "model_metrics.json", model)
    report_path = root / "docs" / "index.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(eda, model), encoding="utf-8")
    return quality, eda, model
