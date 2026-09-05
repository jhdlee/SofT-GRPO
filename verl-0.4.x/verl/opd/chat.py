"""Explicit training chat rendering contracts shared by data and OPD replay."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


QWEN3_TRAINING_PROFILE = "qwen3-training-benchmark-v1"
QWEN3_THINK_TOKEN_IDS = (151667, 151668)
QWEN3_ASSISTANT_HEADER = "<|im_start|>assistant\n"
QWEN3_THINK_OPENER = "<think>\n"


def validate_prompt_profile(profile: str | None) -> None:
    if profile not in (None, QWEN3_TRAINING_PROFILE):
        raise ValueError(f"unsupported training prompt profile: {profile!r}")


def validate_training_reasoning_tokens(tokenizer: Any, profile: str | None = None) -> tuple[int, int]:
    validate_prompt_profile(profile)
    observed = []
    for marker in ("<think>", "</think>"):
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids, skip_special_tokens=False) != marker:
            raise RuntimeError(f"{marker!r} must round-trip as one native tokenizer token")
        observed.append(int(ids[0]))
    if observed[0] == observed[1]:
        raise RuntimeError("thinking delimiters share a token ID")
    if profile == QWEN3_TRAINING_PROFILE and tuple(observed) != QWEN3_THINK_TOKEN_IDS:
        raise RuntimeError("Qwen3 reasoning token IDs differ from the pinned profile")
    return tuple(observed)


def render_training_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    profile: str | None = None,
) -> str:
    """Preserve legacy rendering; append exactly one explicit Qwen3 opener."""

    validate_prompt_profile(profile)
    kwargs = {"add_generation_prompt": True, "tokenize": False}
    if profile == QWEN3_TRAINING_PROFILE:
        kwargs["enable_thinking"] = True
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if profile == QWEN3_TRAINING_PROFILE:
        if not isinstance(rendered, str) or not rendered.endswith(QWEN3_ASSISTANT_HEADER):
            raise RuntimeError("Qwen3 template must end in its native assistant header before the fixed opener")
        rendered += QWEN3_THINK_OPENER
    return rendered
