"""Prompt construction and content selection for earnings-call sentiment classification."""

from __future__ import annotations

import ast
import re
from typing import Any

import numpy as np

SECTION_FULL = "full"
SECTION_PREPARED = "prepared_remarks"
SECTION_QA = "qa"

MODE_ANSWER_ONLY = "answer_only"
MODE_GENERATE_EXPLANATION = "generate_explanation"

ORDER_ANSWER_FIRST = "answer_first"
ORDER_RATIONALE_FIRST = "rationale_first"

LABEL_MAP = {"A": "Bearish", "B": "Neutral", "C": "Bullish"}

_SYSTEM_MESSAGE = (
    "You are a senior financial analyst specializing in equity research. "
    "Your task is to classify the forward-looking business outlook expressed "
    "in an earnings call transcript. Focus on management's tone, guidance, "
    "and forward-looking statements — not on historical results alone. "
    "Use exactly the output format requested."
)

_QA_BOUNDARY_RE = re.compile(
    r"question.and.answer|q&a session|open.{0,10}for questions|questions from",
    re.IGNORECASE,
)


def _parse_structured_content(raw: Any) -> list[dict[str, str]]:
    if isinstance(raw, list):
        return [r if isinstance(r, dict) else ast.literal_eval(str(r)) for r in raw]
    if isinstance(raw, np.ndarray):
        out = []
        for item in raw:
            if isinstance(item, dict):
                out.append(item)
            else:
                try:
                    out.append(ast.literal_eval(str(item)))
                except Exception:
                    out.append({"speaker": "Unknown", "text": str(item)})
        return out
    try:
        parsed = ast.literal_eval(str(raw))
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return [{"speaker": "Unknown", "text": str(raw)}]


def _find_qa_start_index(turns: list[dict[str, str]]) -> int:
    for i, turn in enumerate(turns):
        text = turn.get("text", "")
        if _QA_BOUNDARY_RE.search(text):
            return i
    return len(turns)


def _turns_to_text(turns: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"[{t.get('speaker', 'Unknown')}]: {t.get('text', '')}" for t in turns
    )


def select_content(row: Any, section: str) -> str:
    """Return the relevant text section from a panel row.

    Parameters
    ----------
    row:
        A pandas Series or dict-like with at least ``content`` and
        ``structured_content`` columns.
    section:
        One of ``"full"``, ``"prepared_remarks"``, or ``"qa"``.

    Returns
    -------
    str
        Plain text ready to be inserted into the prompt.
    """
    if section == SECTION_FULL:
        return str(row["content"])

    raw = row.get("structured_content") if hasattr(row, "get") else row["structured_content"]
    turns = _parse_structured_content(raw)
    qa_idx = _find_qa_start_index(turns)

    if section == SECTION_PREPARED:
        relevant = turns[:qa_idx] if qa_idx > 0 else turns
    elif section == SECTION_QA:
        relevant = turns[qa_idx:] if qa_idx < len(turns) else turns
    else:
        raise ValueError(f"Unknown section {section!r}. Use 'full', 'prepared_remarks', or 'qa'.")

    return _turns_to_text(relevant)


def build_messages(
    text: str,
    mode: str,
    explanation_order: str = ORDER_ANSWER_FIRST,
) -> list[dict[str, str]]:
    """Build HuggingFace-style chat messages for the classification task.

    Parameters
    ----------
    text:
        Transcript text (already selected by ``select_content``).
    mode:
        ``"answer_only"`` or ``"generate_explanation"``.
    explanation_order:
        ``"answer_first"`` or ``"rationale_first"``; only relevant for
        ``generate_explanation`` mode.

    Returns
    -------
    list[dict]
        ``[{"role": "system", "content": ...}, {"role": "user", "content": ...}]``
    """
    if mode == MODE_ANSWER_ONLY:
        user_content = (
            f"{text}\n\n"
            "Based on the earnings call above, classify the company's "
            "forward-looking business outlook.\n\n"
            "Answer with exactly one letter — no punctuation, no explanation:\n"
            "A = Bearish\n"
            "B = Neutral\n"
            "C = Bullish\n\n"
            "Answer:"
        )
    elif mode == MODE_GENERATE_EXPLANATION:
        if explanation_order == ORDER_ANSWER_FIRST:
            format_instruction = (
                'Respond with a JSON object in exactly this format:\n'
                '{"answer": "<A|B|C>", "rationale": "<one concise sentence>"}\n\n'
                "Where answer is:\n"
                "A = Bearish\nB = Neutral\nC = Bullish\n\n"
                "JSON:"
            )
        elif explanation_order == ORDER_RATIONALE_FIRST:
            format_instruction = (
                'Respond with a JSON object in exactly this format:\n'
                '{"rationale": "<one concise sentence>", "answer": "<A|B|C>"}\n\n'
                "Where answer is:\n"
                "A = Bearish\nB = Neutral\nC = Bullish\n\n"
                "JSON:"
            )
        else:
            raise ValueError(
                f"Unknown explanation_order {explanation_order!r}. "
                "Use 'answer_first' or 'rationale_first'."
            )
        user_content = (
            f"{text}\n\n"
            "Based on the earnings call above, classify the company's "
            "forward-looking business outlook.\n\n"
            + format_instruction
        )
    else:
        raise ValueError(
            f"Unknown mode {mode!r}. Use 'answer_only' or 'generate_explanation'."
        )

    return [
        {"role": "system", "content": _SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]
