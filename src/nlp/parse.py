"""Parsing utilities: extract answer letter and token-level probability / entropy."""

from __future__ import annotations

import json
import re

import numpy as np
import torch

VALID_LETTERS = {"A", "B", "C"}
_LETTER_RE = re.compile(r"\b([ABC])\b")


def extract_answer_letter(
    text: str,
    mode: str,
    explanation_order: str = "answer_first",
) -> str | None:
    """Extract the classification letter (A/B/C) from generated text.

    Parameters
    ----------
    text:
        Raw generated text from the model.
    mode:
        ``"answer_only"`` or ``"generate_explanation"``.
    explanation_order:
        ``"answer_first"`` or ``"rationale_first"``; only used when
        ``mode == "generate_explanation"``.

    Returns
    -------
    str | None
        One of ``"A"``, ``"B"``, ``"C"``, or ``None`` if parsing fails.
    """
    text = text.strip()

    if mode == "answer_only":
        # Expect exactly one letter as the first meaningful character.
        for ch in text:
            if ch in VALID_LETTERS:
                return ch
        return None

    # generate_explanation: parse JSON
    letter = _extract_from_json(text, explanation_order)
    if letter is not None:
        return letter

    # Fallback: first standalone A/B/C in text
    match = _LETTER_RE.search(text)
    return match.group(1) if match else None


def _extract_from_json(text: str, explanation_order: str) -> str | None:
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    # Try to find a JSON object
    brace_match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if not brace_match:
        return None
    try:
        obj = json.loads(brace_match.group(0))
    except json.JSONDecodeError:
        # Try ast literal_eval as fallback for single-quoted JSON
        import ast
        try:
            obj = ast.literal_eval(brace_match.group(0))
        except Exception:
            return None

    answer = obj.get("answer", "")
    if isinstance(answer, str):
        for ch in answer.strip().upper():
            if ch in VALID_LETTERS:
                return ch
    return None


def extract_answer_token_prob(
    logits_vec: torch.Tensor | np.ndarray,
    chosen_token_id: int,
) -> tuple[float, float]:
    """Compute confidence (prob of chosen token) and entropy from a logit vector.

    Parameters
    ----------
    logits_vec:
        1-D logit vector over the full vocabulary for the answer token position.
    chosen_token_id:
        Token-id that was actually generated (the first token of the answer).

    Returns
    -------
    (confidence, entropy)
        confidence : probability of the chosen token under the softmax distribution.
        entropy    : Shannon entropy (nats) of the full softmax distribution.
    """
    if isinstance(logits_vec, np.ndarray):
        logits_vec = torch.from_numpy(logits_vec).float()
    else:
        logits_vec = logits_vec.float()

    probs = torch.softmax(logits_vec, dim=-1)
    confidence = float(probs[chosen_token_id].item())

    # Clamp to avoid log(0)
    probs_clamped = probs.clamp(min=1e-12)
    entropy = float(-torch.sum(probs_clamped * torch.log(probs_clamped)).item())

    return confidence, entropy
