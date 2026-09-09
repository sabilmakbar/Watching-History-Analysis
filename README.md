# Watching History Analysis

An interactive exploration of 106,811 accepted playback events from a short
February 2020 export. The main artifact is a responsive Plotly report covering
viewing volume, platform and content mix, completion, audience context, playback
quality, and a compact completion benchmark.

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

- 106,823 logical records were parsed: 106,811 accepted and 12 malformed records
  rejected with their source positions.
- Completion modeling uses 67,874 labeled VOD and catch-up events. The first 80%
  by playback time trains the models; the last 20% tests them.
- Logistic Regression reaches 0.690 ROC-AUC and 0.365 PR-AUC on the chronological
  holdout, versus 0.218 PR-AUC for the prior-probability dummy model.
- Completion and coefficient differences are observational associations, not
  causal effects.

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
