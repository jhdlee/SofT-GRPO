"""Privileged OPD prompt templates.

These functions render only teacher-side user content.  Student prompting stays
in the released rollout path and must never call them.
"""

from __future__ import annotations

from typing import Optional, Union

from .config import PromptTemplate


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def render_sdft_prompt(
    original_user_content: str,
    gold_cot: str,
    gold_answer: str,
) -> str:
    """Render the Self-Distillation Fine-Tuning demonstration prompt."""

    _require_text(original_user_content, "original_user_content")
    _require_text(gold_cot, "gold_cot")
    _require_text(gold_answer, "gold_answer")
    return (
        f"\n{original_user_content}\n\n"
        "This is an example for a response to the question:\n"
        f"{gold_cot}\n"
        f"The final answer is: \\boxed{{{gold_answer}}}\n\n"
        "Now answer with a response of your own, including the thinking process.\n"
    )


def render_sdpg_prompt(original_user_content: str, gold_solution: str, gold_answer: str) -> str:
    """Render the SDPG hint-and-alternative-solution prompt."""

    _require_text(original_user_content, "original_user_content")
    _require_text(gold_solution, "gold_solution")
    _require_text(gold_answer, "gold_answer")
    return (
        f"\n{original_user_content}\n\n"
        f"[Hint] The correct answer is {gold_answer}. A common way to solve this is:\n"
        f"{gold_solution}\n\n"
        f"[Instruction] If possible, derive the answer {gold_answer} using an alternative, equally rigorous "
        "mathematical approach to the one provided above. Otherwise, improve the given reasoning by making it "
        "clearer, more complete, and logically sound. Do NOT state that you were given the answer or reference."
    )


def render_privileged_prompt(
    original_user_content: str,
    gold_cot: str,
    template: Union[PromptTemplate, str] = PromptTemplate.SDPG,
    gold_answer: Optional[str] = None,
) -> str:
    """Render one of the supported privileged teacher prompts."""

    try:
        template_type = PromptTemplate(template)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown privileged prompt template: {template!r}") from exc
    if template_type is PromptTemplate.SDFT:
        if gold_answer is None:
            raise ValueError("gold_answer is required for the sdft prompt")
        return render_sdft_prompt(original_user_content, gold_cot, gold_answer)
    if gold_answer is None:
        raise ValueError("gold_answer is required for the sdpg prompt")
    return render_sdpg_prompt(original_user_content, gold_cot, gold_answer)
