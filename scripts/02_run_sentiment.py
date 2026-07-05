#!/usr/bin/env python3
"""Zero-shot LLM sentiment + MC-uncertainty pipeline over the master panel."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from data.config import load_config
from nlp.inference import load_model
from nlp.runner import run_panel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("02_run_sentiment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "master_panel.parquet",
        help="Input panel file (.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Directory for output files",
    )
    parser.add_argument(
        "--section",
        choices=["full", "prepared_remarks", "qa"],
        default="full",
        help="Which part of the transcript to classify",
    )
    parser.add_argument(
        "--mode",
        choices=["answer_only", "generate_explanation"],
        default="answer_only",
        help="Classification mode",
    )
    parser.add_argument(
        "--explanation-order",
        choices=["answer_first", "rationale_first"],
        default="answer_first",
        help="JSON field order (only relevant for generate_explanation)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="MC samples per call (default: from config)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: from config)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Max tokens per sample (default: from config based on mode)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save checkpoint every N rows (default: from config)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows (smoke test)",
    )
    parser.add_argument(
        "--output-format",
        choices=["parquet", "json", "csv", "xlsx"],
        default="parquet",
        help="Output file format",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cuda, cpu, or specific (e.g. cuda:0)",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16"],
        default="auto",
        help="Model weight dtype",
    )
    return parser.parse_args()


def save_result(panel: pd.DataFrame, output_path: Path, fmt: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        # majority_samples is a list — store as string for parquet compatibility
        out = panel.copy()
        if "majority_samples" in out.columns:
            out["majority_samples"] = out["majority_samples"].apply(
                lambda x: str(x) if x is not None else None
            )
        out.to_parquet(output_path, index=False)
    elif fmt == "json":
        panel.to_json(output_path, orient="records", indent=2, force_ascii=False)
    elif fmt == "csv":
        out = panel.copy()
        if "majority_samples" in out.columns:
            out["majority_samples"] = out["majority_samples"].apply(
                lambda x: str(x) if x is not None else None
            )
        out.to_csv(output_path, index=False)
    elif fmt == "xlsx":
        out = panel.copy()
        if "majority_samples" in out.columns:
            out["majority_samples"] = out["majority_samples"].apply(
                lambda x: str(x) if x is not None else None
            )
        out.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unknown format {fmt!r}")
    logger.info("Saved %d rows -> %s", len(panel), output_path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    nlp_cfg = config.get("nlp", {})
    mc_cfg = nlp_cfg.get("monte_carlo", {})

    model_name = nlp_cfg.get("model", "Qwen/Qwen2.5-7B-Instruct")
    n_samples = args.n_samples or mc_cfg.get("n_samples", 20)
    temperature = args.temperature if args.temperature is not None else mc_cfg.get("temperature", 0.7)
    checkpoint_every = args.checkpoint_every or mc_cfg.get("checkpoint_every", 100)

    if args.max_new_tokens is not None:
        max_new_tokens = args.max_new_tokens
    elif args.mode == "answer_only":
        max_new_tokens = mc_cfg.get("max_new_tokens_answer_only", 5)
    else:
        max_new_tokens = mc_cfg.get("max_new_tokens_explanation", 200)

    logger.info("Loading panel from %s", args.input)
    panel = pd.read_parquet(args.input)
    if args.limit is not None:
        panel = panel.head(args.limit).copy()
        logger.info("Limiting to %d rows.", args.limit)

    logger.info("Loading model %s (device=%s, dtype=%s)", model_name, args.device, args.dtype)
    model, tokenizer = load_model(model_name, device=args.device, dtype=args.dtype)

    checkpoint_dir = PROJECT_ROOT / "data" / "interim"
    result = run_panel(
        panel=panel,
        model=model,
        tokenizer=tokenizer,
        section=args.section,
        mode=args.mode,
        explanation_order=args.explanation_order,
        n_samples=n_samples,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=checkpoint_dir,
    )

    suffix = f"{args.section}_{args.mode}"
    if args.mode == "generate_explanation":
        suffix += f"_{args.explanation_order}"
    output_filename = f"panel_with_sentiment_{suffix}.{args.output_format}"
    output_path = args.output_dir / output_filename

    save_result(result, output_path, args.output_format)

    # Print summary statistics
    if "majority_answer" in result.columns:
        counts = result["majority_answer"].value_counts(dropna=False)
        logger.info("Sentiment distribution:\n%s", counts.to_string())
        if "majority_agreement" in result.columns:
            logger.info(
                "Mean agreement: %.3f  Mean confidence: %.3f  Mean entropy: %.3f",
                result["majority_agreement"].mean(),
                result["confidence"].mean() if "confidence" in result.columns else float("nan"),
                result["entropy"].mean() if "entropy" in result.columns else float("nan"),
            )


if __name__ == "__main__":
    main()
