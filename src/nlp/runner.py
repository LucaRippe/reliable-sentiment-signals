"""Main inference loop: iterates over the panel, checkpoints, and saves results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from nlp.inference import batch_mc_sample
from nlp.metrics import compute_sample_metrics
from nlp.prompt import build_messages, select_content

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "majority_answer",
    "majority_samples",
    "majority_agreement",
    "entropy",
    "confidence",
    "representative_text",
]


def _checkpoint_path(output_dir: Path, section: str, mode: str) -> Path:
    return output_dir / f"sentiment_checkpoint_{section}_{mode}.parquet"


def _load_checkpoint(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.warning("Could not load checkpoint %s: %s", path, e)
    return None


def _save_checkpoint(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(path, index=False)


def run_panel(
    panel: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    section: str,
    mode: str,
    explanation_order: str = "answer_first",
    n_samples: int = 20,
    temperature: float = 0.7,
    max_new_tokens: int = 5,
    checkpoint_every: int = 100,
    checkpoint_dir: Path | None = None,
) -> pd.DataFrame:
    """Run the full MC-sentiment pipeline over a panel DataFrame.

    Parameters
    ----------
    panel:
        The master panel (one row per earnings call), must have
        ``content`` and ``structured_content`` columns.
    model, tokenizer:
        Loaded from :func:`~nlp.inference.load_model`.
    section:
        ``"full"``, ``"prepared_remarks"``, or ``"qa"``.
    mode:
        ``"answer_only"`` or ``"generate_explanation"``.
    explanation_order:
        ``"answer_first"`` or ``"rationale_first"``; only for explain mode.
    n_samples:
        Number of MC samples per call.
    temperature:
        Sampling temperature.
    max_new_tokens:
        Max new tokens per sample.
    checkpoint_every:
        Save intermediate results every N rows.
    checkpoint_dir:
        Directory for checkpoint files. Defaults to ``data/interim``.

    Returns
    -------
    pd.DataFrame
        Original panel with six new columns appended.
    """
    chk_dir = checkpoint_dir or Path("data/interim")
    chk_path = _checkpoint_path(chk_dir, section, mode)

    # Resume from checkpoint
    checkpoint_df = _load_checkpoint(chk_path)
    if checkpoint_df is not None and len(checkpoint_df) > 0:
        already_done = set(zip(checkpoint_df["symbol"], checkpoint_df["call_date"].astype(str)))
        logger.info("Resuming from checkpoint: %d rows already processed.", len(already_done))
    else:
        already_done = set()
        checkpoint_df = None

    results: list[dict] = list(checkpoint_df.to_dict("records")) if checkpoint_df is not None else []
    pending = panel[
        ~panel.apply(
            lambda r: (r["symbol"], str(r["call_date"])) in already_done, axis=1
        )
    ].copy()

    logger.info(
        "Processing %d rows (section=%s, mode=%s, n_samples=%d)",
        len(pending), section, mode, n_samples,
    )

    for i, (_, row) in enumerate(tqdm(pending.iterrows(), total=len(pending), desc="Sentiment")):
        try:
            text = select_content(row, section)
            messages = build_messages(text, mode, explanation_order)
            samples = batch_mc_sample(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                n_samples=n_samples,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            metrics = compute_sample_metrics(samples, mode=mode, explanation_order=explanation_order)
        except Exception as e:
            logger.warning("Row %s/%s failed: %s", row["symbol"], row["call_date"], e)
            metrics = {col: None for col in OUTPUT_COLUMNS}

        record = {
            "symbol": row["symbol"],
            "call_date": row["call_date"],
            **metrics,
        }
        results.append(record)

        if (i + 1) % checkpoint_every == 0:
            _save_checkpoint(results, chk_path)
            logger.info("Checkpoint saved (%d rows).", len(results))

    # Final checkpoint
    _save_checkpoint(results, chk_path)

    results_df = pd.DataFrame(results)
    # Merge back onto the full original panel
    merged = panel.merge(
        results_df[["symbol", "call_date"] + OUTPUT_COLUMNS],
        on=["symbol", "call_date"],
        how="left",
    )
    return merged
