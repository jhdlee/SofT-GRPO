"""Behavioral checks for native-soft dispatch without GPU/Ray dependencies."""

import __future__
import ast
import asyncio
import copy
import hashlib
import importlib.util
import json
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn.utils.rnn import pad_sequence


VERL_ROOT = Path(__file__).resolve().parents[2]
ROLLOUT = VERL_ROOT / "verl/workers/rollout/sglang_rollout"
SGLANG = VERL_ROOT.parent / "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang"
spec = importlib.util.spec_from_file_location("opd_request_dispatch_tested", ROLLOUT / "request_dispatch.py")
dispatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dispatch)


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)


SEED_NS = {"hashlib": hashlib, "json": json, "MAX_SEED": (1 << 63) - 1}
load_definitions(SGLANG / "srt/sampling/stateless_random.py", {"_stable_component", "derive_seed", "derive_parallel_seed"}, SEED_NS)
load_definitions(ROLLOUT / "deterministic_sampling.py", {
    "_python_scalar", "prompt_fingerprint", "derive_request_seed", "build_request_sampling_params", "expand_parallel_seeds",
}, SEED_NS)


def native_output(seed):
    token = 20 + seed % 50
    tokens = [token, 90, 71, 2]  # latent, first close, categorical answer, EOS
    return {
        "text": "native completion",
        "meta_info": {
            "seed_for_test": seed,
            "native_field_must_survive": {"boundary": 1},
            "output_token_logprobs": [(-0.25, token_id, None) for token_id in tokens],
            "output_topk_idx_list": [[token, 8, 9, 10, 11]] + [[token_id, 0, 0, 0, 0] for token_id in tokens[1:]],
            "output_topk_gumbel_list": [[0.5, 0.2, -0.3, 0.1, -0.1]] + [[0.0] * 5 for _ in tokens[1:]],
            "output_topk_gumbel_noise_list": [[0.8, -0.4, 0.1, 0.3, -0.2]] + [[0.0] * 5 for _ in tokens[1:]],
            "output_topk_retained_mask_list": [[True, False, True, True, False]] + [[True, False, False, False, False] for _ in tokens[1:]],
            "output_topk_prob_list": [[0.8, 0.01, 0.1, 0.08, 0.01]] + [[1.0, 0.0, 0.0, 0.0, 0.0] for _ in tokens[1:]],
        },
    }


class Engine:
    def __init__(self):
        self.calls = []
        self.shutdown_calls = 0
        self.flush_calls = 0

    async def async_generate(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        assert kwargs["return_logprob"] is True
        ids = kwargs["input_ids"]
        params = kwargs["sampling_params"]
        if isinstance(ids[0], list):
            rows = params if isinstance(params, list) else [params] * len(ids)
            result = []
            for row in rows:
                for sample in range(row.get("n", 1)):
                    seed = row.get("seed", 0)
                    if row.get("n", 1) > 1:
                        seed = SEED_NS["derive_parallel_seed"](seed, sample)
                    result.append(native_output(seed))
            return result
        await asyncio.sleep(0)
        return native_output(params.get("seed", 0))

    def shutdown(self):
        self.shutdown_calls += 1

    async def flush_cache(self):
        self.flush_calls += 1


def request_arguments(group_size=8):
    ids = [[3, 4], [5, 6]]
    params, bases = SEED_NS["build_request_sampling_params"](
        {"n": group_size, "max_new_tokens": 8192, "temperature": 1.0},
        root_seed=11, rollout_iteration=2, example_identities=["a", "b"], prompt_token_ids=ids,
    )
    return {
        "queue_size": 3, "input_ids": ids, "image_data": [None, None], "sampling_params": params,
        "expanded_sampling_seeds": SEED_NS["expand_parallel_seeds"](bases, group_size),
    }


@pytest.mark.parametrize("group_size", [1, 8])
def test_dispatchers_preserve_seed_expansion_order_and_native_metadata(group_size):
    kwargs = request_arguments(group_size)
    original = copy.deepcopy(kwargs)
    outputs = {}
    for mode in dispatch.DISPATCH_MODES:
        engine = Engine()
        outputs[mode], timing = asyncio.run(dispatch.dispatch_generation(engine, mode=mode, **kwargs))
        assert timing["request_count"] == 2 * group_size
        assert timing["engine_generation_seconds"] >= 0
        assert engine.shutdown_calls == 0
        if mode == "bounded_async":
            assert len(engine.calls) == 2 * group_size
            assert all(call["sampling_params"]["n"] == 1 for call in engine.calls)
        elif mode == "expanded_batch":
            assert len(engine.calls[0]["input_ids"]) == 2 * group_size
            assert all(row["n"] == 1 for row in engine.calls[0]["sampling_params"])
    assert outputs["legacy_batch"] == outputs["expanded_batch"] == outputs["bounded_async"]
    assert [row["meta_info"]["seed_for_test"] for row in outputs["bounded_async"]] == kwargs["expanded_sampling_seeds"]
    assert kwargs == original


def test_bounded_queue_refills_behind_slow_first_request_and_restores_order():
    class StragglerEngine(Engine):
        async def async_generate(self, **kwargs):
            index = kwargs["input_ids"][0]
            self.started.append(index)
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                if index == 0:
                    await self.release_first.wait()
                if index == 2:
                    assert 0 not in self.completed
                    self.release_first.set()
                await asyncio.sleep(0)
                self.completed.append(index)
                return native_output(index)
            finally:
                self.active -= 1

    async def run():
        engine = StragglerEngine()
        engine.started, engine.completed = [], []
        engine.active = engine.peak = 0
        engine.release_first = asyncio.Event()
        output, timing = await asyncio.wait_for(dispatch.dispatch_generation(
            engine, mode="bounded_async", queue_size=2,
            input_ids=[[index] for index in range(5)], image_data=[None] * 5,
            sampling_params={"n": 1}, expanded_sampling_seeds=list(range(5)),
        ), timeout=2)
        assert engine.peak == timing["frontend_peak_pending"] == 2
        assert engine.active == 0
        assert engine.completed.index(1) < engine.completed.index(0)
        assert [item["meta_info"]["seed_for_test"] for item in output] == list(range(5))

    asyncio.run(run())


def test_legacy_dispatch_preserves_engine_managed_seed_without_opd_seed_metadata():
    engine = Engine()
    outputs, _ = asyncio.run(dispatch.dispatch_generation(
        engine, mode="legacy_batch", queue_size=2,
        input_ids=[[3, 4]], image_data=[None], sampling_params={"n": 8, "seed": 41},
        expanded_sampling_seeds=None,
    ))
    assert engine.calls[0]["sampling_params"] == {"n": 8, "seed": 41}
    assert len(outputs) == 8


def test_partial_failure_retires_backend_before_cancelling_siblings_and_forbids_reuse():
    class FailingEngine(Engine):
        def shutdown(self):
            super().shutdown()
            self.events.append("shutdown")

        async def async_generate(self, **kwargs):
            try:
                if kwargs["input_ids"][0] == 3:
                    await asyncio.sleep(0)
                    raise ValueError("injected scheduler failure")
                await asyncio.Event().wait()
            finally:
                self.events.append("finished")

    engine = FailingEngine()
    engine.events = []
    kwargs = request_arguments(1)
    with pytest.raises(ValueError, match="scheduler failure"):
        asyncio.run(dispatch.dispatch_generation(engine, mode="bounded_async", **kwargs))
    assert engine.shutdown_calls == 1
    assert engine.events[-2:] == ["shutdown", "finished"]
    with pytest.raises(RuntimeError, match="poisoned"):
        asyncio.run(dispatch.dispatch_generation(engine, mode="legacy_batch", **kwargs))
    assert engine.shutdown_calls == 1


def test_caller_cancellation_retires_engine_and_leaves_no_frontend_tasks():
    class WaitingEngine(Engine):
        async def async_generate(self, **kwargs):
            self.started.set()
            self.active += 1
            try:
                await asyncio.Event().wait()
            finally:
                self.active -= 1

    async def run():
        engine = WaitingEngine()
        engine.active = 0
        engine.started = asyncio.Event()
        task = asyncio.create_task(dispatch.dispatch_generation(
            engine, mode="bounded_async", **request_arguments(1),
        ))
        await engine.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert engine.active == 0
        assert engine.shutdown_calls == 1

    asyncio.run(run())


@pytest.mark.parametrize("mode", dispatch.DISPATCH_MODES)
def test_invalid_completion_fails_entire_rollout(mode):
    class BadEngine(Engine):
        async def async_generate(self, **kwargs):
            return []

    engine = BadEngine()
    with pytest.raises(RuntimeError, match="completion"):
        asyncio.run(dispatch.dispatch_generation(engine, mode=mode, **request_arguments(1)))
    assert engine.shutdown_calls == 1


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_invalid_queue_and_engine_bounds_are_rejected(value):
    with pytest.raises(ValueError):
        dispatch.validate_dispatch_options("bounded_async", value, None)
    with pytest.raises(ValueError):
        dispatch.validate_dispatch_options("bounded_async", 32, value)


def test_warmup_cap_is_positive_bounded_and_does_not_mutate_metadata():
    assert dispatch.benchmark_response_cap({}, 8192) == 8192
    metadata = {"benchmark_max_new_tokens": 8}
    assert dispatch.benchmark_response_cap(metadata, 8192) == 8
    assert metadata == {"benchmark_max_new_tokens": 8}
    for value in [True, 0, -1, 8193]:
        with pytest.raises(ValueError):
            dispatch.benchmark_response_cap({"benchmark_max_new_tokens": value}, 8192)


class LocalDist:
    @staticmethod
    def is_initialized():
        return False

    @staticmethod
    def get_rank():
        return 0


def test_remote_engine_failure_poisoning_reaches_other_ranks():
    class RemoteFailure:
        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size():
            return 2

        @staticmethod
        def all_gather_object(results, local):
            assert local is None
            results[:] = [None, "ValueError: remote scheduler failed"]

    engine = Engine()
    with pytest.raises(RuntimeError, match="rank 1.*remote scheduler"):
        dispatch.check_collective_error(engine, RemoteFailure, None, "generation")
    assert engine.shutdown_calls == 1


class Config(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class DataProto:
    def __init__(self, batch=None, non_tensor_batch=None, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = {} if non_tensor_batch is None else non_tensor_batch
        self.meta_info = {} if meta_info is None else meta_info


def load_adapter():
    namespace = {
        "torch": torch, "F": torch.nn.functional, "np": np, "asyncio": asyncio, "time": time,
        "contextmanager": contextmanager, "pad_sequence": pad_sequence,
        "TensorDict": lambda values, batch_size: values, "DataProto": DataProto,
        "dist": LocalDist, "broadcast_pyobj": lambda data, **kwargs: data,
        **{name: getattr(dispatch, name) for name in (
            "benchmark_response_cap", "check_collective_error", "dispatch_generation",
            "positive_integer", "validate_dispatch_options",
        )},
        "os": SimpleNamespace(environ={}),
        **{name: SEED_NS[name] for name in ("build_request_sampling_params", "expand_parallel_seeds")},
    }
    load_definitions(VERL_ROOT / "verl/utils/torch_functional.py", {
        "pad_sequence_to_length", "pad_sequence_to_length_1", "get_response_mask",
    }, namespace)
    load_definitions(ROLLOUT / "sglang_rollout.py", {"_post_process_outputs", "_pre_process_inputs"}, namespace)
    tree = ast.parse((ROLLOUT / "sglang_rollout.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SGLangRollout")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {
        "_batch_level_generate_sequences", "_assemble_single_turn_outputs", "update_sampling_params", "_init_inference_engine",
    }]
    for method in methods:
        if method.name != "update_sampling_params":
            method.decorator_list = []
    adapter = ast.ClassDef(name="Adapter", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[adapter], type_ignores=[]))
    exec(compile(module, "adapter_under_test", "exec", flags=__future__.annotations.compiler_flag), namespace)
    result = namespace["Adapter"]()
    result.config = Config(
        dispatch_mode="legacy_batch", async_queue_size=3, deterministic_sampling=True,
        response_length=8, free_cache_engine=True, gumbel_softmax_temperature=0.1,
        val_kwargs=SimpleNamespace(top_k=5, top_p=0.95, temperature=0.6),
    )
    result.sampling_params = {"n": 8, "max_new_tokens": 8, "temperature": 1.0}
    result.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    result.pad_token_id = 0
    result._rank = result._tp_rank = 0
    result._device_mesh_cpu = {"tp": SimpleNamespace(get_group=lambda: None, mesh=torch.tensor([0]))}
    result._engine = Engine()
    result._test_namespace = namespace
    return result


def prompts(*, warmup=None, validate=False):
    result = DataProto(
        batch={
            "input_ids": torch.tensor([[3, 4], [5, 6]]),
            "attention_mask": torch.ones(2, 2, dtype=torch.long),
            "position_ids": torch.tensor([[0, 1], [0, 1]]),
        },
        non_tensor_batch={
            "index": np.array([4, 9]), "raw_prompt_ids": np.array([[3, 4], [5, 6]], dtype=object),
            "rollout_sample_index": np.array([0, 1]),
        },
        meta_info={"eos_token_id": 2, "rollout_seed": 11, "rollout_iteration": 2, "validate": validate},
    )
    if warmup is not None:
        result.meta_info["benchmark_max_new_tokens"] = warmup
    return result


@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("group_size", [1, 8])
def test_real_adapter_preserves_every_replay_tensor_and_validation_expansion(validate, group_size):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = {}
        for mode in dispatch.DISPATCH_MODES:
            adapter = load_adapter()
            adapter.config.dispatch_mode = mode
            adapter.sampling_params["n"] = group_size
            result[mode] = adapter._batch_level_generate_sequences(prompts(validate=validate))
            expected_count = 2 if validate else 2 * group_size
            assert result[mode].batch["responses"].shape == (expected_count, 8)
            assert adapter._engine.flush_calls == 1
            assert adapter.sampling_params["n"] == group_size
            assert result[mode].meta_info["rollout_timing"]["tensor_assembly_seconds"] >= 0
            assert result[mode].meta_info["rollout_timing"]["dispatch_mode"] == mode
        reference = result["legacy_batch"]
        assert set(reference.batch) == {
            "prompts", "responses", "input_ids", "rollout_log_probs", "rollout_topk_ids",
            "rollout_topk_gumbels", "attention_mask", "position_ids", "gumbel_temperature", "rollout_sampling_seed",
            "rollout_topk_retained_mask", "rollout_topk_gumbel_noise", "rollout_topk_probs", "rollout_rank",
        }
        for other in (result["expanded_batch"], result["bounded_async"]):
            for key, value in reference.batch.items():
                assert torch.equal(value, other.batch[key]), key
            for key, value in reference.non_tensor_batch.items():
                assert np.array_equal(value, other.non_tensor_batch[key]), key
        assert reference.batch["rollout_topk_retained_mask"].dtype == torch.bool
        assert reference.batch["rollout_topk_retained_mask"][0, 2].tolist() == [True, False, True, True, False]
        assert reference.batch["rollout_topk_retained_mask"][0, :2, 0].all()
        assert reference.batch["rollout_topk_gumbel_noise"][0, 2].tolist() == pytest.approx([0.8, -0.4, 0.1, 0.3, -0.2])
        assert reference.batch["rollout_topk_probs"][0, 2].tolist() == pytest.approx([0.8, 0.01, 0.1, 0.08, 0.01])
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def test_adapter_malformed_replay_is_collective_failure_without_cache_flush():
    class MissingReplay(Engine):
        async def async_generate(self, **kwargs):
            outputs = await super().async_generate(**kwargs)
            outputs[0]["meta_info"].pop("output_topk_gumbel_list")
            return outputs

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        adapter = load_adapter()
        adapter._engine = MissingReplay()
        with pytest.raises(RuntimeError, match="collective tensor assembly.*output_topk_gumbel_list"):
            adapter._batch_level_generate_sequences(prompts())
        assert adapter._engine.shutdown_calls == 1
        assert adapter._engine.flush_calls == 0
    finally:
        loop.close()
        asyncio.set_event_loop(None)


@pytest.mark.parametrize("field", ["output_topk_retained_mask_list", "output_topk_gumbel_noise_list", "output_topk_prob_list"])
def test_required_filter_metadata_failure_retires_entire_rollout(field):
    class MissingFilter(Engine):
        async def async_generate(self, **kwargs):
            outputs = await super().async_generate(**kwargs)
            for output in outputs:
                output["meta_info"].pop(field)
            return outputs

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        adapter = load_adapter()
        adapter.config.require_retained_support = True
        adapter._engine = MissingFilter()
        with pytest.raises(RuntimeError, match="collective tensor assembly"):
            adapter._batch_level_generate_sequences(prompts())
        assert adapter._engine.shutdown_calls == 1
        assert adapter._engine.flush_calls == 0
    finally:
        loop.close()
        asyncio.set_event_loop(None)


@pytest.mark.parametrize("mode", dispatch.DISPATCH_MODES)
def test_actual_engine_transitions_reject_outstanding_dispatch(mode):
    tree = ast.parse((ROLLOUT / "sglang_rollout.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AsyncEngine")
    methods = [node for node in cls.body if isinstance(node, ast.AsyncFunctionDef)]
    module = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(
        name="Transitions", bases=[], keywords=[], body=methods, decorator_list=[],
    )], type_ignores=[]))
    namespace = {"require_idle_engine": dispatch.require_idle_engine}
    exec(compile(module, "actual_engine_transitions", "exec", flags=__future__.annotations.compiler_flag), namespace)

    class BlockingEngine(namespace["Transitions"], Engine):
        async def async_generate(self, **kwargs):
            self.started.set()
            await self.finish.wait()
            return await Engine.async_generate(self, **kwargs)

    async def run():
        engine = BlockingEngine()
        engine.started, engine.finish = asyncio.Event(), asyncio.Event()
        kwargs = request_arguments(group_size=1)
        task = asyncio.create_task(dispatch.dispatch_generation(engine, mode=mode, **kwargs))
        await engine.started.wait()
        try:
            for transition in (engine.release_memory_occupation, engine.resume_memory_occupation,
                               engine.flush_cache, lambda: engine.update_weights_from_tensor([])):
                with pytest.raises(RuntimeError, match="requests remain outstanding"):
                    await transition()
            with pytest.raises(RuntimeError, match="requests remain outstanding"):
                await dispatch.dispatch_generation(engine, mode=mode, **kwargs)
        finally:
            engine.finish.set()
            await task
        dispatch.require_idle_engine(engine)
        assert engine.shutdown_calls == 0

    asyncio.run(run())


@pytest.mark.parametrize("result", [(False, "invalid tensor"), None, (True, "updated")])
def test_engine_rejected_weight_acknowledgment_cannot_be_ignored(result):
    tree = ast.parse((ROLLOUT / "sglang_rollout.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AsyncEngine")
    method = next(node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "update_weights_from_tensor")
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {
        "require_idle_engine": dispatch.require_idle_engine,
        "UpdateWeightsFromTensorReqInput": lambda **kwargs: SimpleNamespace(**kwargs),
        "MultiprocessingSerializer": SimpleNamespace(serialize=lambda value: value),
    }
    exec(compile(module, "actual_weight_transfer", "exec", flags=__future__.annotations.compiler_flag), namespace)

    async def update(*args):
        return result

    engine = SimpleNamespace(server_args=SimpleNamespace(tp_size=1),
                             tokenizer_manager=SimpleNamespace(update_weights_from_tensor=update))
    if result and result[0] is True:
        assert asyncio.run(namespace["update_weights_from_tensor"](engine, [])) == result
    else:
        with pytest.raises(RuntimeError, match="rejected weight transfer"):
            asyncio.run(namespace["update_weights_from_tensor"](engine, []))


@pytest.mark.parametrize("success", [False, True])
def test_native_scheduler_busy_ack_prevents_memory_pause(success):
    tree = ast.parse((ROLLOUT / "sglang_rollout.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AsyncEngine")
    methods = [node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name in {"flush_cache", "release_memory_occupation"}]
    module = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    namespace = {"require_idle_engine": dispatch.require_idle_engine,
                 "ReleaseMemoryOccupationReqInput": lambda: None}
    exec(compile(module, "actual_memory_transition", "exec", flags=__future__.annotations.compiler_flag), namespace)
    released = []

    async def flush():
        return SimpleNamespace(success=success)

    async def release(*args):
        released.append(True)

    engine = SimpleNamespace(tokenizer_manager=SimpleNamespace(
        flush_cache=flush, release_memory_occupation=release,
    ))
    engine.flush_cache = lambda: namespace["flush_cache"](engine)
    if success:
        asyncio.run(namespace["release_memory_occupation"](engine))
        assert released == [True]
    else:
        with pytest.raises(RuntimeError, match="scheduler requests may remain outstanding"):
            asyncio.run(namespace["release_memory_occupation"](engine))
        assert not released


@pytest.mark.parametrize("failing_rank", [0, 1])
def test_broadcast_failure_keeps_local_and_remote_ranks_in_identical_collective_phases(failing_rank):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        adapter = load_adapter()
        stages = []

        class PeerDist(LocalDist):
            calls = 0

            @staticmethod
            def is_initialized():
                return True

            @staticmethod
            def get_world_size():
                return 2

            @classmethod
            def all_gather_object(cls, results, local):
                cls.calls += 1
                results[:] = [local, None]
                if cls.calls == 3 and failing_rank == 1:
                    # This healthy rank completed its TP=1 broadcast; its peer
                    # failed before assembly at exactly the same global phase.
                    assert local is None
                    results[1] = "MemoryError: injected broadcast allocation failure"
                elif cls.calls == 3:
                    assert "injected broadcast" in local
                elif cls.calls == 4:
                    assert "output broadcast" in local
                    results[1] = "RuntimeError: collective output broadcast failed"

        def check(engine, distributed, error, stage):
            stages.append(stage)
            dispatch.check_collective_error(engine, distributed, error, stage)

        adapter._test_namespace["dist"] = PeerDist
        adapter._test_namespace["check_collective_error"] = check
        if failing_rank == 0:
            def failed_broadcast(**kwargs):
                raise MemoryError("injected broadcast allocation failure")

            adapter._test_namespace["broadcast_pyobj"] = failed_broadcast
        adapter._assemble_single_turn_outputs = lambda *args: pytest.fail("assembly ran after a broadcast failure")
        with pytest.raises(RuntimeError, match="collective output broadcast") as failure:
            adapter._batch_level_generate_sequences(prompts())

        # Exercise the real next context-exit phase too: both ranks must arrive
        # here once, after the same broadcast error collective, before return.
        path = VERL_ROOT / "verl/workers/sharding_manager/fsdp_sglang.py"
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FSDPSGLangShardingManager")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__exit__")
        method.decorator_list = []
        namespace = {"check_collective_error": check, "poison_engine": dispatch.poison_engine, "dist": PeerDist}
        module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
        manager = SimpleNamespace(inference_engine=adapter._engine, device_mesh=None)
        with pytest.raises(RuntimeError, match="collective sharding context"):
            namespace["__exit__"](manager, RuntimeError, failure.value, None)
        assert stages == ["preparation", "generation", "output broadcast", "sharding context"]
        assert PeerDist.calls == 4
        assert manager._opd_poisoned
        assert adapter._engine.shutdown_calls == 1
        assert adapter._engine.flush_calls == 0
    finally:
        loop.close()
        asyncio.set_event_loop(None)


@pytest.mark.parametrize("options", [{}, {"max_running_requests": 16, "engine_context_length": 12000}])
def test_engine_limits_are_forwarded_only_when_requested(options):
    adapter = load_adapter()
    adapter._tp_size = 1
    adapter.visible_devices_set = {0}
    adapter.config.update(
        dtype="bfloat16", gpu_memory_utilization=0.6, load_format="dummy_dtensor", max_model_len=12000, **options,
    )
    adapter._test_namespace["AsyncEngine"] = lambda **kwargs: SimpleNamespace(options=kwargs)
    adapter._init_inference_engine(False, "model-under-test", None)
    actual = adapter._engine.options
    if options:
        assert actual["max_running_requests"] == 16
        assert actual["context_length"] == 12000
    else:
        assert "max_running_requests" not in actual
        assert "context_length" not in actual


def test_adapter_warmup_does_not_shorten_following_measured_batch():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        adapter = load_adapter()
        adapter._batch_level_generate_sequences(prompts(warmup=4))
        adapter._batch_level_generate_sequences(prompts())
        caps = [call["sampling_params"][0]["max_new_tokens"] for call in adapter._engine.calls]
        assert caps == [4, 8]
        assert adapter.sampling_params["max_new_tokens"] == 8
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def test_sharding_failure_skips_release_and_poison_prevents_next_enter():
    path = VERL_ROOT / "verl/workers/sharding_manager/fsdp_sglang.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FSDPSGLangShardingManager")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {
        "__enter__", "__exit__", "_guard_stage", "_finish_stage", "_require_idle", "_prepare_weights",
    }]
    for node in methods:
        node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(
        name="Manager", bases=[], keywords=[], body=methods, decorator_list=[],
    )], type_ignores=[]))
    namespace = {
        "poison_engine": dispatch.poison_engine,
        "check_collective_error": dispatch.check_collective_error,
        "dist": LocalDist,
        "time": time, "asyncio": asyncio, "require_idle_engine": dispatch.require_idle_engine,
    }
    exec(compile(module, "sharding_under_test", "exec", flags=__future__.annotations.compiler_flag), namespace)
    manager = namespace["Manager"]()
    manager.device_mesh = None
    manager.inference_engine = Engine()
    with pytest.raises(RuntimeError, match="failed rollout"):
        manager.__exit__(RuntimeError, RuntimeError("failed rollout"), None)
    assert manager.inference_engine.shutdown_calls == 1
    with pytest.raises(RuntimeError, match="poisoned"):
        manager.__enter__()
