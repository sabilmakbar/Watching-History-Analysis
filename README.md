# Watching History Analysis

An executive, interactive analysis of 106,811 accepted playback events from a
short February 2020 export. It asks where playback demand concentrates, which
contexts are associated with completion, and how much information available at
playback start predicts later completion. The responsive Plotly report pairs each
measured result with its practical meaning and limits.

View the published report at
[sabilmakbar.github.io/Watching-History-Analysis](https://sabilmakbar.github.io/Watching-History-Analysis/).
The committed [docs/index.html](docs/index.html) also opens directly in a browser
without a local server.

## Setup

Install [uv](https://docs.astral.sh/uv/), then create the locked environment:

```sh
uv sync
```

The project uses Python 3.11 or newer. Its direct dependencies are pandas,
scikit-learn, and Plotly.

## Build

Regenerate every result artifact and the report:

```sh
uv run python scripts/build_analysis.py
```

The build verifies the source file hash, accounts for malformed records, creates
aggregate EDA results, trains the chronological completion benchmark, and writes
the self-contained report shell to `docs/index.html`. Plotly is loaded from the
pinned CDN when the report opens.

## Validation

Run the full self-check:

```sh
uv run python test_analysis.py
```

The check covers source immutability, row accounting, coercion, leakage controls,
model metric gates, report content and ordering, interactivity hooks, the pinned
Plotly URL, and byte-identical repeated builds.

## Results

The pipeline parsed 106,823 logical records, accepted 106,811, and recorded 12
malformed records with their source positions. Its main findings are:

1. Web mobile accounts for 57.8% of events; together with the Android app it
   accounts for 82.1%, concentrating the largest experience and reliability surface.
2. Viewing peaks at 13:00–14:00 UTC. The first and last dates are partial windows,
   so their lower totals should not be interpreted as demand changes.
3. Android-app completion is 32.1% versus 23.7% on mobile web. Logged-in completion
   is 31.2% versus 24.0%, while ad-present and ad-absent completion are effectively
   flat at 24.8% and 24.9%.
4. News completes at 35.0% on 21,283 eligible events versus 19.7% on 21,846 for
   Entertainment, showing why platform comparisons must consider content mix.
5. On the chronological holdout, Logistic Regression improves PR-AUC from 0.218 to
   0.365 and Brier score from 0.172 to 0.158. This is a useful benchmark, not an
   operational model.
6. Average bitrate stays at 300,000 from the median through p99, while buffer
   duration rises from a median of 1 to 1,035 at p95. Source units and possible
   capping need investigation before operational use.

Completion modeling uses 67,874 labeled VOD and catch-up events, with the first 80%
by playback time for training and the last 20% for testing. All differences are
event-level associations from a short export, not causal or user-level effects.

Machine-readable outputs are in `results/data_quality.json`, `results/eda.json`,
and `results/model_metrics.json`.

## Source

The analysis uses the anonymized CSV committed at the repository root. Its pinned
SHA-256 is
`83ba4a0ae4e965e61ca44c4c3e444218d07cc8b8b3cec4a2029ab51b6a12c09c`.
The original repository does not identify the upstream publisher and license, so
this repository is the available source and the provenance is limited. Times are
reported in UTC.

## Publishing

GitHub Pages should serve the `main` branch from `/docs`. Once enabled in the
repository settings, the public URL is:

<https://sabilmakbar.github.io/Watching-History-Analysis/>
