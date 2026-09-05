"""Student filtering and privileged replay share an explicit Qwen3 opener."""

import ast
from pathlib import Path
from types import SimpleNamespace

import datasets
import pytest
from torch import nn

from verl.opd.chat import QWEN3_TRAINING_PROFILE, render_training_prompt, validate_training_reasoning_tokens
from verl.opd.config import OPDConfig
from verl.opd.prompts import render_privileged_prompt
from verl.opd.replay import PrivilegedReplay


class Tokenizer:
    def __init__(self, header="<|im_start|>assistant\n"):
        self.header = header
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return messages[0]["content"] + self.header

    def encode(self, text, add_special_tokens=False):
        if text in ("<think>", "</think>"):
            return [151667 if text == "<think>" else 151668]
        return [ord(char) for char in text] + ([151667] if "<think>" in text else [])

    def decode(self, ids, skip_special_tokens=False):
        return {151667: "<think>", 151668: "</think>"}[ids[0]]


def test_legacy_rendering_keeps_original_template_call_and_output():
    tokenizer = Tokenizer(header="legacy <think>")
    messages = [{"role": "user", "content": "question"}]
    assert render_training_prompt(tokenizer, messages) == "questionlegacy <think>"
    assert tokenizer.calls == [(messages, {"add_generation_prompt": True, "tokenize": False})]


def test_qwen_renderer_has_exact_header_opener_and_atomic_ids():
    tokenizer = Tokenizer()
    text = render_training_prompt(tokenizer, [{"role": "user", "content": "question"}], QWEN3_TRAINING_PROFILE)
    assert text == "question<|im_start|>assistant\n<think>\n"
    assert tokenizer.calls[0][1]["enable_thinking"] is True
    assert validate_training_reasoning_tokens(tokenizer, QWEN3_TRAINING_PROFILE) == (151667, 151668)
    with pytest.raises(RuntimeError, match="assistant header"):
        render_training_prompt(Tokenizer(header="<|im_start|>assistant\n<think>\n"), [{"role": "user", "content": "q"}], QWEN3_TRAINING_PROFILE)
    with pytest.raises(ValueError, match="prompt profile"):
        OPDConfig.from_mapping({"prompt_profile": "unregistered"})


def test_privileged_replay_uses_same_qwen_renderer_without_changing_sdpg_content():
    tokenizer = Tokenizer()
    config = OPDConfig.from_mapping({"prompt_profile": QWEN3_TRAINING_PROFILE})
    replay = PrivilegedReplay(nn.Linear(1, 1), tokenizer, config)
    extra = {"opd_original_user_content": "Question", "opd_gold_cot": "Reasoning", "opd_gold_answer": "4"}
    ids = replay._prompt_ids(extra)
    content = render_privileged_prompt(original_user_content="Question", gold_cot="Reasoning", gold_answer="4", template="sdpg")
    assert content.startswith("\nQuestion")
    assert tokenizer.calls[-1][0] == [{"role": "user", "content": content}]
    assert ids == tokenizer.encode(render_training_prompt(tokenizer, [{"role": "user", "content": content}], QWEN3_TRAINING_PROFILE), add_special_tokens=False)


def test_dataset_filter_measures_qwen_fixed_opener_tokens(tmp_path, monkeypatch):
    # Execute the real dataset method without unrelated GPU/Ray dependencies.
    source = Path(__file__).resolve().parents[2] / "verl/utils/dataset/rl_dataset.py"
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")
    tree = ast.parse(source.read_text())
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in ("_dataset_filter_kwargs", "_read_files_and_tokenize")]
    namespace = {"datasets": datasets}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), "exec"), namespace)
    tokenizer = Tokenizer()
    messages = [{"role": "user", "content": "short"}]
    length = len(tokenizer.encode(render_training_prompt(tokenizer, messages, QWEN3_TRAINING_PROFILE), add_special_tokens=False))
    parquet = tmp_path / "prompts.parquet"
    datasets.Dataset.from_list([{"prompt": messages}]).to_parquet(str(parquet))
    obj = SimpleNamespace(data_files=[str(parquet)], filter_overlong_prompts=True, tokenizer=tokenizer, prompt_key="prompt", num_workers=1, prompt_profile=QWEN3_TRAINING_PROFILE, max_prompt_length=length - 1)
    namespace["_read_files_and_tokenize"](obj)
    assert len(obj.dataframe) == 0
    obj.max_prompt_length = length
    namespace["_read_files_and_tokenize"](obj)
    assert len(obj.dataframe) == 1
