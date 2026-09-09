from __future__ import annotations

import argparse
import json
from pathlib import Path

from watch_history.analysis import build_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic analysis outputs.")
    parser.add_argument("--summary", action="store_true", help="print row and time summary")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    quality, eda = build_analysis(root)
    if args.summary:
        print(
            json.dumps(
                {
                    "records": quality["records"],
                    "date_range_utc": [
                        eda["daily_volume"][0]["date_utc"],
                        eda["daily_volume"][-1]["date_utc"],
                    ],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
