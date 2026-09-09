# Decision spine: Watching History Analysis

## AD-1 — Interactive EDA is the primary product
- **Decision:** The project centers on a responsive Plotly HTML report; predictive
  modeling is a compact supporting section.
- **Binds:** Most report space and validation cover data quality, viewing patterns,
  content, audience context, and playback quality.
- **Prevents:** turning the repository into a model leaderboard with token EDA.
- **Rule:** `docs/index.html` contains the required interactive EDA views before any
  model result is presented.
- **Decided:** 2026-09-10, by user
- **Supersedes:** —

## AD-2 — Prediction uses playback-start information only
- **Decision:** The completion model uses fields known when playback begins and a
  chronological holdout.
- **Binds:** `play_duration`, `end_time`, `total_bytes`, `buffer_duration`,
  `average_bitrate`, `bitrate_range`, `completed`, and `has_ad` are excluded from
  model inputs.
- **Prevents:** inflated scores caused by information recorded after the outcome.
- **Rule:** the model feature list is explicit and a test asserts every excluded
  field is absent.
- **Decided:** 2026-09-10, by user-approved direction
- **Supersedes:** —

## AD-3 — No personalized recommendation claim
- **Decision:** The report analyzes event-level behavior and does not build user
  recommendations or viewer clusters.
- **Binds:** hashed watcher identifiers are used only for aggregate coverage checks.
- **Prevents:** claims based on histories that are too sparse to validate.
- **Rule:** no result artifact contains a watcher-level recommendation or segment.
- **Decided:** 2026-09-10, by user-approved direction
- **Supersedes:** —

## AD-4 — Preserve the source data and state its limited provenance
- **Decision:** Keep the root CSV byte-identical and cite this public repository as
  the available source; do not assert an external license or original publisher.
- **Binds:** cleaning happens in memory and rejected-row output contains row numbers
  and reasons rather than rewritten source data.
- **Prevents:** silent source mutation and unsupported provenance claims.
- **Rule:** the self-check asserts the source CSV SHA-256 and README states that the
  upstream source and license are not recorded in the original repository.
- **Decided:** 2026-09-10, by user-approved direction
- **Supersedes:** —

## Memlog

- 2026-09-10 — AD-1 decided
- 2026-09-10 — AD-2 decided
- 2026-09-10 — AD-3 decided
- 2026-09-10 — AD-4 decided
