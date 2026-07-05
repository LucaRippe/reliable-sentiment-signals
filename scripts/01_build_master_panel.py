#!/usr/bin/env python3
"""Build the master earnings-call panel (transcripts + returns + controls)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.config import load_config, resolve_path
from data.panel import build_master_panel, save_panel
from data.transcripts import load_transcripts
from data.universe import filter_sp500_point_in_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N transcripts (useful for smoke tests)",
    )
    parser.add_argument(
        "--skip-controls",
        action="store_true",
        help="Skip momentum/size/SUE controls (faster smoke test)",
    )
    parser.add_argument(
        "--transcripts-only",
        action="store_true",
        help="Only download and filter transcripts; skip returns merge",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.transcripts_only:
        dataset_name = config["data"]["transcripts"]["dataset"]
        lookback_years = config["project"]["lookback_years"]
        index_name = config["data"]["universe"]["index"]

        transcripts = load_transcripts(dataset_name, lookback_years)
        if args.limit is not None:
            transcripts = transcripts.head(args.limit)
        transcripts = filter_sp500_point_in_time(transcripts, index=index_name)

        output_dir = resolve_path(config["data"]["paths"]["interim"], config)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "transcripts_sp500_pit.parquet"
        transcripts.to_parquet(output_path, index=False)
        print(f"Saved {len(transcripts):,} transcripts -> {output_path}")
        return

    panel = build_master_panel(
        config,
        limit=args.limit,
        skip_controls=args.skip_controls,
    )
    output_path = save_panel(panel, config)
    print(f"Saved master panel with {len(panel):,} rows -> {output_path}")
    print(panel[["symbol", "call_date", "ret_t5"]].head())


if __name__ == "__main__":
    main()
