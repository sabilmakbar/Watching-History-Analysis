# Spec: Interactive viewing-history analysis

| Field | Value |
|---|---|
| Mode | feature |
| Status | signed-off |
| Date | 2026-09-10 |

## Measurements (freeze-exempt)

| Measure | Command | Value | As of |
|---|---|---|---|
| source physical lines | `wc -l part-00000-0dd5f001-57d7-4ec1-a3f0-53ea28bba9c3-c000.csv` | 106825 | 2026-09-10 |
| source columns | `awk -F, 'NR==1 {print NF}' part-00000-0dd5f001-57d7-4ec1-a3f0-53ea28bba9c3-c000.csv` | 41 | 2026-09-10 |
| source SHA-256 | `shasum -a 256 part-00000-0dd5f001-57d7-4ec1-a3f0-53ea28bba9c3-c000.csv` | `83ba4a0ae4e965e61ca44c4c3e444218d07cc8b8b3cec4a2029ab51b6a12c09c` | 2026-09-10 |
| parsed / accepted / rejected events | `uv run python scripts/build_analysis.py --summary` | 106823 / 106811 / 12 (direct probe) | 2026-09-10 |
| event time span | `uv run python scripts/build_analysis.py --summary` | 2020-02-01 through 2020-02-16 UTC (direct probe) | 2026-09-10 |
| repeat-history coverage | `uv run python scripts/build_analysis.py --summary` | 102985 accepted-row watchers; median 1 event; 45 with at least 5 events (direct probe) | 2026-09-10 |
| chronological model probe | `uv run python scripts/build_analysis.py --summary` | LogReg ROC-AUC 0.689, PR-AUC 0.365; dummy PR-AUC 0.218 (without `has_ad`) | 2026-09-10 |
| accepted-row numeric missingness | `uv run python scripts/build_analysis.py --summary` | `average_bitrate`: 4946 missing; all present modeled numeric values parse | 2026-09-10 |

## Problem & intent

The public repository contains a large playback-event export but no analysis,
documentation, or runnable project. The user wants exploratory data analysis and
interactive data visualization to be the main portfolio artifact, with a small
predictive section where the data supports it. Done means a reproducible pipeline,
an interactive GitHub Pages report, and an honest account of data quality,
observational limits, and model leakage controls.

## Ground truth

- `.gitattributes:1` and the repository tree — the current project contains only
  line-ending configuration and the root CSV; it has no README, environment, code,
  tests, results, or Pages site.
- `part-00000-0dd5f001-57d7-4ec1-a3f0-53ea28bba9c3-c000.csv:1` — the source has
  hashed event/content/watcher identifiers plus playback context, device, quality,
  content, and completion fields; column count is in Measurements.
- The direct CSV profile found structurally malformed records; the apparent invalid
  numeric payload belongs to one field-shifted rejected record. All non-empty
  accepted numeric and timestamp values parse successfully; `average_bitrate`
  missingness is recorded in Measurements.
- Completion is populated for video-on-demand and catch-up events but absent for
  livestream events, so completion modeling cannot use livestream rows.
- Repeat-history coverage in Measurements is too sparse for validated user
  recommendations or viewer clustering.
- The repository and commit history do not name the original publisher or license.

## User scenarios

### S1 — Trust the analysis inputs (priority: P1)
As a reviewer, I want explicit row accounting and quality checks, so that every
chart rests on inspectable cleaning rules.
**Independent test:** run the analysis self-check and inspect the quality artifact.
**Acceptance:**
- Given the original CSV, when the loader runs, then it preserves the source file,
  accepts structurally valid rows, and records each rejected logical record number,
  physical starting line, and reason.
- Given numeric and timestamp fields, when cleaning runs, then invalid values become
  missing values and the quality artifact reports their counts.

### S2 — Explore viewing and playback behavior (priority: P1)
As a portfolio reviewer, I want interactive views of the event data, so that I can
inspect patterns rather than read static summary claims.
**Independent test:** build and open `docs/index.html`.
**Acceptance:**
- Given accepted events, when the report builds, then interactive Plotly views show
  event volume over time, hour-of-day use, platform/content mix, completion rates,
  audience context, and robust playback-quality distributions.
- Given a chart, when a viewer hovers, zooms, or toggles a legend item, then Plotly
  exposes details without a server process.
- Given quality outliers and missing completion for livestreams, when the report is
  read, then it explains the display treatment and denominator for each result.

### S3 — Test completion propensity without leakage (priority: P2)
As an analyst, I want a compact predictive benchmark, so that the report shows how
start-time context generalizes to later events.
**Independent test:** run the build and inspect metrics plus the model section.
**Acceptance:**
- Given eligible video-on-demand and catch-up events, when training runs, then a
  prior-probability dummy model and Logistic Regression use an 80/20 chronological
  split and report ROC-AUC, PR-AUC, and Brier score.
- Given the model feature list, when the leakage test runs, then no post-start or
  outcome-derived field appears.
- Given the fitted Logistic Regression, when the report builds, then it displays the
  strongest positive and negative coefficients as associations, not causes.

### S4 — Reproduce and publish the portfolio artifact (priority: P1)
As the repository owner, I want one documented build command and a Pages-ready file,
so that the public repository stands alone.
**Independent test:** run setup, build, and self-check from a clean environment.
**Acceptance:**
- Given `uv sync`, when `uv run python scripts/build_analysis.py` runs, then it writes
  deterministic result artifacts and `docs/index.html`.
- Given the README, when a reviewer opens the repository, then it explains the data,
  method, commands, findings, caveats, and public report link.
- Given the committed report, when the coordinator enables GitHub Pages from
  `main:/docs`, then `https://sabilmakbar.github.io/Watching-History-Analysis/`
  returns HTTP 200 and contains the report title.

## Requirements

- **FR-001** — Use a uv-managed Python project with direct dependencies limited to
  pandas, scikit-learn, and Plotly.
- **FR-002** — Load the root CSV without modifying it; reject rows whose parsed field
  count differs from the header. Record the one-based logical data-record number,
  one-based physical source starting line, and reason. Retain logical record number
  on accepted rows for deterministic ordering.
- **FR-003** — Coerce timestamps, numeric fields, and boolean fields explicitly; save
  row accounting and missingness to `results/data_quality.json`.
- **FR-004** — Produce aggregates for time, platform, content, completion, audience
  context, and playback quality; use log/percentile-aware views where raw tails would
  hide the distribution.
- **FR-005** — Generate a responsive `docs/index.html` with viewport metadata and
  Plotly `responsive: true` figures, narrative findings, method notes, and caveats;
  load Plotly from the exact pinned URL
  `https://cdn.plot.ly/plotly-4.0.0.min.js`.
- **FR-006** — Model completion only for rows with a defined label in `vod` or
  `catchup`; split by event time, earlier 80% for training and later 20% for test.
- **FR-007** — Model inputs are exactly: `playback_location`, `platform`,
  `referrer_group`, `content_type`, `category_name`, `player_name`, `autoplay`,
  `is_login`, `is_premium`, `os_name`, `browser_name`, derived UTC `hour`, and
  derived UTC `day_of_week`. `has_ad` is excluded because the source does not prove
  it is known at prediction time.
- **FR-008** — Compare `DummyClassifier(strategy="prior")` with one-hot Logistic
  Regression; save metrics and top coefficients to `results/model_metrics.json`.
- **FR-009** — Provide one runnable `test_analysis.py` covering source immutability,
  row accounting, outputs, leakage exclusions, metric gates, and required report
  sections.
- **FR-010** — Add a README with setup, build, validation, result summary, Pages link,
  and the known provenance limitation.

## Success criteria

- **SC-001** — Accepted plus rejected parsed records equals total parsed records; each
  rejected record has a row number and non-empty reason.
- **SC-002** — The report contains at least seven interactive Plotly figures and all
  S2 topics; it has viewport metadata and responsive Plotly configuration; it renders
  without a local server.
- **SC-003** — Logistic Regression test ROC-AUC is at least 0.65 and PR-AUC exceeds
  the dummy model by at least 0.10.
- **SC-004** — The leakage test positively asserts the complete allowed feature list
  and asserts `completed`, `play_duration`, `end_time`, `total_bytes`,
  `buffer_duration`, `average_bitrate`, `bitrate_range`, and `has_ad` are outside it.
- **SC-005** — `uv sync`, the report build, and `uv run python test_analysis.py` run
  cleanly; repeated builds produce identical tracked artifacts.
- **SC-006** — The source CSV SHA-256 remains equal to the source hash in Measurements.
- **SC-007** — After coordinator deployment, the public Pages URL returns HTTP 200
  and contains the report title.

## Mode section

### Feature: regression surface
Existing behavior at risk: the root CSV is the repository's only substantive asset
and is currently unpinned. Story 001 must add a SHA-256 assertion before changing any
other tracked content. Integration points touched: root CSV read path, new uv project,
new `src/`, `scripts/`, `results/`, `docs/`, README, and GitHub Pages configuration.
The coordinator owns the Pages setting change after verified commits are pushed;
deployment source is `main:/docs`.

## Out of scope

- Personalized recommendations, collaborative filtering, and watcher clustering.
- Causal claims that buffering, ads, or platform choice change completion.
- Livestream completion modeling, because the source does not populate that label.
- Deep learning, hyperparameter tuning, notebooks, dashboards requiring a server,
  and external data enrichment.
- Rewriting, renaming, or moving the source CSV.
- Claiming an upstream publisher or license not recorded in the repository.

## Assumptions & open items

- ASSUMED: all displayed times remain UTC because the source gives UTC timestamps and
  no viewer timezone.
- ASSUMED: the public GitHub repository remains the report's stated data source.

## Complexity ledger

| Deviation | Justification |
|---|---|
| Plotly dependency and CDN runtime | User requested interactive HTML; hover, zoom, and legend filtering justify it. |
| Two sequential stories | Data trust must be independently verified before modeling and report publication. |

## Change log

| Date | Section | Change | Reason | Scope change? (needs re-sign-off) |
|---|---|---|---|---|
