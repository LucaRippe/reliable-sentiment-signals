"""Aggregate N Monte-Carlo samples into majority-vote uncertainty metrics."""

from __future__ import annotations

from collections import Counter
from typing import Any

from nlp.parse import VALID_LETTERS, extract_answer_letter, extract_answer_token_prob


def compute_sample_metrics(
    samples: list[dict[str, Any]],
    mode: str,
    explanation_order: str = "answer_first",
) -> dict[str, Any]:
    """Compute majority-vote metrics from N Monte-Carlo samples.

    Parameters
    ----------
    samples:
        List of N dicts as returned by :func:`~nlp.inference.batch_mc_sample`.
        Each dict has keys ``raw_text``, ``answer_token_logits``,
        ``answer_token_id``.
    mode:
        ``"answer_only"`` or ``"generate_explanation"``.
    explanation_order:
        ``"answer_first"`` or ``"rationale_first"``.

    Returns
    -------
    dict with keys:
        majority_answer     : str | None   — A / B / C (plurality winner)
        majority_samples    : list[str]    — all N extracted letters (None → "?")
        majority_agreement  : float        — fraction of samples == majority_answer
        entropy             : float | None — mean token entropy (majority samples only)
        confidence          : float | None — mean token confidence (majority samples only)
        representative_text : str | None   — last raw_text among majority samples;
                                             only set when mode == "generate_explanation"
    """
    parsed_letters: list[str] = []
    for s in samples:
        letter = extract_answer_letter(
            s["raw_text"], mode=mode, explanation_order=explanation_order
        )
        parsed_letters.append(letter if letter in VALID_LETTERS else "?")

    # Majority vote (excluding parse failures)
    valid_letters = [lt for lt in parsed_letters if lt != "?"]
    if not valid_letters:
        return {
            "majority_answer": None,
            "majority_samples": parsed_letters,
            "majority_agreement": 0.0,
            "entropy": None,
            "confidence": None,
            "representative_text": None,
        }

    counter = Counter(valid_letters)
    majority_answer = counter.most_common(1)[0][0]
    majority_agreement = counter[majority_answer] / len(parsed_letters)

    # Collect indices of majority samples
    majority_indices = [
        i for i, letter in enumerate(parsed_letters) if letter == majority_answer
    ]

    # Entropy and confidence over majority samples
    entropies: list[float] = []
    confidences: list[float] = []
    representative_text: str | None = None

    for i in majority_indices:
        s = samples[i]
        logits = s.get("answer_token_logits")
        token_id = s.get("answer_token_id")
        if logits is not None and token_id is not None:
            conf, ent = extract_answer_token_prob(logits, token_id)
            confidences.append(conf)
            entropies.append(ent)
        if mode == "generate_explanation":
            representative_text = s["raw_text"]

    mean_entropy = float(sum(entropies) / len(entropies)) if entropies else None
    mean_confidence = float(sum(confidences) / len(confidences)) if confidences else None

    return {
        "majority_answer": majority_answer,
        "majority_samples": parsed_letters,
        "majority_agreement": majority_agreement,
        "entropy": mean_entropy,
        "confidence": mean_confidence,
        "representative_text": representative_text,
    }
