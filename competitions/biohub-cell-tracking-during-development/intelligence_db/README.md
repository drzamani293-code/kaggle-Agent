# Biohub Kaggle Intelligence Database

A reproducible, local experiment-intelligence store for the Kaggle
**Biohub – Cell Tracking During Development** competition. It tracks our
experiments, public-leaderboard scores, commands, run diagnostics, the
public resources we studied, distilled lessons, and data-driven next
actions.

**Data policy:** only public resources, our own Kaggle run logs, and our own
diagnostics. No hidden competition data, no Kaggle submission API, no private
scraping.

## Layout

```
intelligence_db/
├── README.md                 # this file
├── schema.sql                # DuckDB schema (6 tables)
├── intelligence.duckdb       # generated DB (gitignored; rebuild from seeds)
├── data/
│   ├── seed_experiments.json # our own runs: experiments + stats + decisions + lessons
│   └── public_sources.json   # public GitHub resources we consulted
├── scripts/
│   ├── _common.py            # shared paths + idempotent DB loaders
│   ├── init_intelligence_db.py   # build DB from seeds, regenerate reports
│   ├── update_from_report.py     # ingest a Kaggle run report (idempotent)
│   ├── generate_reports.py       # regenerate all Markdown reports from the DB
│   └── query_db.py               # read-only CLI queries
├── reports/
│   ├── score_timeline.md     # chronological scores + deltas
│   ├── experiment_matrix.md  # variant / command / postprocess / stats / score
│   ├── next_actions.md       # data-driven recommendation (branches on M19-C)
│   └── lessons_learned.md    # evidence-linked findings
└── tests/
    └── test_intelligence_db.py
```

## Requirements

```bash
pip install duckdb
```

The `.duckdb` file is a rebuildable artifact and is gitignored; the JSON
seeds under `data/` are the source of truth.

## Quick start

```bash
cd competitions/biohub-cell-tracking-during-development/intelligence_db

# Build the database from the seeds and generate all reports.
python scripts/init_intelligence_db.py --fresh

# Read-only queries.
python scripts/query_db.py best
python scripts/query_db.py timeline
python scripts/query_db.py pending
python scripts/query_db.py experiment M19_A_SAFE_DIVISIONS_PRUNE

# Ingest a real Kaggle run report (idempotent), then reports auto-refresh.
python scripts/update_from_report.py \
    --experiment-id M19_C_FULL_CHAIN_PENDING \
    --submission-report /kaggle/working/milestone19_reference_submission_report.json \
    --conversion-report /kaggle/working/milestone19_geff_conversion.json \
    --public-score 0.881

# Score-only update once Kaggle shows the number (metadata is preserved):
python scripts/update_from_report.py --experiment-id M19_C_FULL_CHAIN_PENDING --public-score 0.881

# Run the tests.
python tests/test_intelligence_db.py
```

## Tables

| Table | Purpose |
|-------|---------|
| `experiments` | one row per run: milestone/variant, command knobs, postprocess ops, score, status |
| `submission_stats` | submission shape + topology + validity/fallback per experiment |
| `postprocess_stats` | divisions/gaps/synthetic-nodes/prune/edge-removed + before/after counts |
| `public_sources` | public GitHub resources we studied + extracted ideas + relevance |
| `decisions` | recorded observation → recommendation → next variant, with risk |
| `lessons` | distilled, evidence-linked findings |

## Idempotency

Every loader is delete-then-insert by key, so `init_intelligence_db.py` and
`update_from_report.py` can be run repeatedly without duplicating rows. A
score-only `update_from_report.py` preserves previously-loaded metadata
(created_at, notes, raw_file, and prior stats) instead of nulling it.

## Current snapshot (seed)

- M16 baseline `0.874` → M17-C det-0.985 `0.874` → M18-C edge-prune `0.874`
  → **M19-A safe_divisions_prune `0.877`** (first mover) → M19-C full_chain
  **pending**.
- Best scored: `M19_A_SAFE_DIVISIONS_PRUNE` @ `0.877` while M19-C is pending.
- Reports regenerate the recommendation automatically once M19-C scores
  (branch on `> / == / < 0.877`).

## Notes

- `created_at` encodes run order (not necessarily exact wall-clock time).
- Scores are our own public-leaderboard results; stats are our own run
  diagnostics. The runner files under `../` are **not** modified by anything
  here.
