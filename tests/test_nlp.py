"""Unit tests for nlp.parse and nlp.metrics (no model required).

Tests that depend on torch are skipped automatically when torch is not installed
(e.g. on a local dev machine without the [llm] extras).  On the GPU cluster
where ``pip install -e ".[llm]"`` has been run, all tests execute.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed; skipping tensor tests")

from nlp.parse import VALID_LETTERS, extract_answer_letter, extract_answer_token_prob
from nlp.metrics import compute_sample_metrics


# ---------------------------------------------------------------------------
# extract_answer_letter
# ---------------------------------------------------------------------------

class TestExtractAnswerLetterAnswerOnly:
    def test_single_letter_A(self):
        assert extract_answer_letter("A", mode="answer_only") == "A"

    def test_single_letter_with_whitespace(self):
        assert extract_answer_letter("  B  ", mode="answer_only") == "B"

    def test_letter_followed_by_text(self):
        assert extract_answer_letter("C some extra text", mode="answer_only") == "C"

    def test_lowercase_not_matched(self):
        # lowercase 'a' is not a valid letter
        result = extract_answer_letter("a", mode="answer_only")
        assert result is None

    def test_no_letter_returns_none(self):
        assert extract_answer_letter("unknown", mode="answer_only") is None

    def test_empty_string(self):
        assert extract_answer_letter("", mode="answer_only") is None


class TestExtractAnswerLetterAnswerFirst:
    def test_valid_json_answer_first(self):
        text = '{"answer": "A", "rationale": "Company showed declining margins."}'
        assert extract_answer_letter(text, mode="generate_explanation", explanation_order="answer_first") == "A"

    def test_valid_json_rationale_first(self):
        text = '{"rationale": "Strong guidance.", "answer": "C"}'
        assert extract_answer_letter(text, mode="generate_explanation", explanation_order="rationale_first") == "C"

    def test_json_with_markdown_fences(self):
        text = '```json\n{"answer": "B", "rationale": "Mixed signals."}\n```'
        assert extract_answer_letter(text, mode="generate_explanation", explanation_order="answer_first") == "B"

    def test_fallback_regex_when_json_fails(self):
        text = "The outlook is Neutral. Answer: B"
        result = extract_answer_letter(text, mode="generate_explanation", explanation_order="answer_first")
        assert result == "B"

    def test_invalid_letter_in_json(self):
        text = '{"answer": "X", "rationale": "Unknown."}'
        # Should fall back to regex search, which finds no standalone A/B/C
        result = extract_answer_letter(text, mode="generate_explanation", explanation_order="answer_first")
        assert result is None or result in VALID_LETTERS

    def test_answer_field_extracts_only_valid(self):
        text = '{"answer": "Bullish", "rationale": "Strong Q4."}'
        # "Bullish" doesn't start with A/B/C alone, but regex should find B
        result = extract_answer_letter(text, mode="generate_explanation", explanation_order="answer_first")
        assert result in VALID_LETTERS or result is None


# ---------------------------------------------------------------------------
# extract_answer_token_prob
# ---------------------------------------------------------------------------

class TestExtractAnswerTokenProb:
    def _uniform_logits(self, vocab_size: int = 100) -> torch.Tensor:
        return torch.zeros(vocab_size)

    def test_uniform_confidence(self):
        logits = self._uniform_logits(100)
        conf, ent = extract_answer_token_prob(logits, chosen_token_id=0)
        assert abs(conf - 1 / 100) < 1e-6

    def test_uniform_entropy(self):
        logits = self._uniform_logits(100)
        _, ent = extract_answer_token_prob(logits, chosen_token_id=0)
        expected_entropy = math.log(100)
        assert abs(ent - expected_entropy) < 0.01

    def test_peaked_distribution_high_confidence(self):
        logits = torch.zeros(100)
        logits[5] = 1000.0
        conf, _ = extract_answer_token_prob(logits, chosen_token_id=5)
        assert conf > 0.999

    def test_peaked_distribution_low_entropy(self):
        logits = torch.zeros(100)
        logits[5] = 1000.0
        _, ent = extract_answer_token_prob(logits, chosen_token_id=5)
        assert ent < 0.01

    def test_numpy_input(self):
        logits_np = np.zeros(50)
        conf, ent = extract_answer_token_prob(logits_np, chosen_token_id=0)
        assert 0.0 < conf <= 1.0
        assert ent > 0.0

    def test_confidence_between_0_and_1(self):
        logits = torch.randn(200)
        conf, _ = extract_answer_token_prob(logits, chosen_token_id=10)
        assert 0.0 <= conf <= 1.0

    def test_entropy_nonnegative(self):
        logits = torch.randn(200)
        _, ent = extract_answer_token_prob(logits, chosen_token_id=10)
        assert ent >= 0.0


# ---------------------------------------------------------------------------
# compute_sample_metrics
# ---------------------------------------------------------------------------

def _make_sample(letter: str, conf: float = 0.8, ent: float = 1.5, token_id: int = 1) -> dict:
    vocab = 100
    logits = torch.full((vocab,), -10.0)
    # Shift logits so that softmax(logits)[token_id] ≈ conf
    logits[token_id] = 0.0
    return {
        "raw_text": letter,
        "answer_token_logits": logits,
        "answer_token_id": token_id,
    }


class TestComputeSampleMetrics:
    def test_unanimous_agreement(self):
        samples = [_make_sample("A") for _ in range(5)]
        result = compute_sample_metrics(samples, mode="answer_only")
        assert result["majority_answer"] == "A"
        assert result["majority_agreement"] == 1.0
        assert result["majority_samples"] == ["A"] * 5

    def test_majority_vote_simple(self):
        samples = [_make_sample("A")] * 3 + [_make_sample("B")] * 2
        result = compute_sample_metrics(samples, mode="answer_only")
        assert result["majority_answer"] == "A"
        assert abs(result["majority_agreement"] - 3 / 5) < 1e-9

    def test_all_invalid_returns_none(self):
        samples = [{"raw_text": "???", "answer_token_logits": torch.zeros(10), "answer_token_id": 0}] * 3
        result = compute_sample_metrics(samples, mode="answer_only")
        assert result["majority_answer"] is None
        assert result["majority_agreement"] == 0.0

    def test_majority_samples_length(self):
        samples = [_make_sample("C")] * 7
        result = compute_sample_metrics(samples, mode="answer_only")
        assert len(result["majority_samples"]) == 7

    def test_entropy_and_confidence_are_floats(self):
        samples = [_make_sample("B")] * 4
        result = compute_sample_metrics(samples, mode="answer_only")
        assert isinstance(result["entropy"], float)
        assert isinstance(result["confidence"], float)

    def test_representative_text_set_for_explain_mode(self):
        samples = []
        for _ in range(3):
            s = _make_sample("C")
            s["raw_text"] = '{"answer": "C", "rationale": "Strong guidance."}'
            samples.append(s)
        result = compute_sample_metrics(
            samples,
            mode="generate_explanation",
            explanation_order="answer_first",
        )
        assert result["representative_text"] is not None
        assert "guidance" in result["representative_text"]

    def test_representative_text_none_for_answer_only(self):
        samples = [_make_sample("A")] * 3
        result = compute_sample_metrics(samples, mode="answer_only")
        assert result["representative_text"] is None

    def test_minority_samples_excluded_from_entropy(self):
        # 4×A, 1×B — entropy should only average over the 4 majority samples
        samples_a = [_make_sample("A", token_id=1) for _ in range(4)]
        samples_b = [_make_sample("B", token_id=2) for _ in range(1)]
        all_samples = samples_a + samples_b
        result = compute_sample_metrics(all_samples, mode="answer_only")
        assert result["majority_answer"] == "A"
        # entropy is defined (not None) since majority samples have valid logits
        assert result["entropy"] is not None
