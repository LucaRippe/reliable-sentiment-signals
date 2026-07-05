"""Model loading and Monte-Carlo sampling with logit capture."""

from __future__ import annotations

from typing import Any

import torch

SampleResult = dict[str, Any]


def load_model(
    model_name: str,
    device: str = "auto",
    dtype: str = "auto",
) -> tuple[Any, Any]:
    """Load model and tokenizer from HuggingFace Hub.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier, e.g. ``"Qwen/Qwen2.5-7B-Instruct"``.
    device:
        ``"auto"``, ``"cuda"``, ``"cpu"``, or a specific device string.
    dtype:
        ``"auto"``, ``"bf16"``, or ``"fp16"``.

    Returns
    -------
    (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "auto": "auto",
    }
    torch_dtype = dtype_map.get(dtype, "auto")
    device_map = device if device != "auto" else "auto"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def single_sample(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_new_tokens: int = 5,
) -> SampleResult:
    """Run one forward pass and return generated text plus answer-token logits.

    The model is called with ``output_scores=True`` so we can recover the
    raw logit vector over the full vocabulary for the **first generated token**
    (i.e. the position where the answer letter A/B/C should appear).

    Parameters
    ----------
    model, tokenizer:
        Loaded from :func:`load_model`.
    messages:
        Chat messages as returned by :func:`~nlp.prompt.build_messages`.
    temperature:
        Sampling temperature. Use > 0 for stochastic generation.
    max_new_tokens:
        Maximum number of tokens to generate.

    Returns
    -------
    dict with keys:
        ``raw_text``           – decoded generated tokens only (not the prompt).
        ``answer_token_logits``– 1-D ``torch.Tensor`` (vocab_size,) for token 0.
        ``answer_token_id``    – int, the actual token id chosen at position 0.
    """
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output.sequences[0, input_len:]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # output.scores is a tuple of (vocab_size,) tensors, one per generated step
    answer_token_logits = output.scores[0][0].cpu()
    answer_token_id = int(generated_ids[0].item())

    return {
        "raw_text": raw_text,
        "answer_token_logits": answer_token_logits,
        "answer_token_id": answer_token_id,
    }


def batch_mc_sample(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    n_samples: int,
    temperature: float = 0.7,
    max_new_tokens: int = 5,
) -> list[SampleResult]:
    """Run N independent forward passes (Monte-Carlo sampling).

    Parameters
    ----------
    n_samples:
        Number of stochastic samples to draw.

    Returns
    -------
    list of N dicts, each with ``raw_text``, ``answer_token_logits``,
    ``answer_token_id``.
    """
    results = []
    for _ in range(n_samples):
        result = single_sample(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        results.append(result)
    return results
