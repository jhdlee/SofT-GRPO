# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging
import os
import time
from collections import OrderedDict

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from dataclasses import asdict

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import package_version as vllm_package_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.device import get_torch_device
from verl.utils.fsdp_utils import fsdp_version, layered_summon_lora_params, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.torch_functional import check_device_is_available
from verl.utils.vllm_utils import TensorLoRARequest, VLLMHijack, is_version_ge, patch_vllm_moe_model_weight_loader

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FSDPVLLMShardingManager(BaseShardingManager):
    @check_device_is_available()
    def __init__(self, module: FSDP, inference_engine: LLM, model_config, full_params: bool = False, device_mesh: DeviceMesh = None, offload_param: bool = False, load_format: str = "dummy_hf", layered_summon: bool = True):
        self.module = module
        # For AsyncLLM, inference_engine and model_runner are defer initialized in vLLMAsyncRollout.load_model
        self.inference_engine = inference_engine
        self._frozen_batch_guard = bool(getattr(inference_engine, "_qwen_frozen_batch_guard", False))
        self._opd_rng_switched = False
        self.last_rollout_timing = {}
        # self.model_runner = inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner if inference_engine else None

        if "vllm_v_0_6_3" in str(type(self.inference_engine)) or "vllm_v_0_5_4" in str(type(self.inference_engine)):
            # vLLM <= v0.6.3
            self.model_runner = self.inference_engine.llm_engine.model_executor.worker.model_runner if self.inference_engine else None
        else:
            # vLLM > v0.6.3
            self.model_runner = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner if self.inference_engine else None

        self.model_config = model_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        self.load_format = load_format
        self.layered_summon = layered_summon

        # Full params
        self.full_params = full_params
        if full_params and fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig())
        elif fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = get_torch_device().get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

        self.base_sync_done: bool = "dummy" not in load_format
        if is_version_ge(pkg="vllm", minver="0.7.3"):
            VLLMHijack.hijack()

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __enter__(self):
        entered_at = time.perf_counter()
        if self._frozen_batch_guard:
            from verl.opd.vllm_lifecycle import require_idle_vllm
            self.last_rollout_timing = {}
            self._guard_stage("entry readiness", lambda: require_idle_vllm(self.inference_engine))
            return self._enter_native(entered_at)
        def __collect_lora_params() -> OrderedDict:
            """
            collect lora params or full params if base model is not ready in vllm
            work with if isinstance(self.module._fsdp_wrapped_module, PeftModel)
            """
            from peft.utils.save_and_load import get_peft_model_state_dict

            lora_params = OrderedDict()
            peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
            if fsdp_version(self.module) > 0:
                if self.layered_summon:
                    if not self.base_sync_done:
                        raise ValueError("To use layered_summon, you must make sure base-model is preloaded in vllm, e.g. let rollout.load_format=safetensors")
                    lora_params = layered_summon_lora_params(self.module)
                else:
                    with FSDP.summon_full_params(self.module, writeback=False):
                        if self.base_sync_done:
                            lora_params = get_peft_model_state_dict(peft_model)
                            lora_params = {name: param.full_tensor().detach().cpu() if hasattr(param, "full_tensor") else param.detach().cpu() for name, param in lora_params.items()}
                        else:
                            model = peft_model.base_model.model
                            orig_dev = "cpu" if "cpu" in next(model.parameters()).device else "cuda"
                            model = model.to("cpu")
                            for name, param in model.state_dict().items():
                                if any(x in name for x in ["_flat_param", "lora_"]):
                                    continue
                                name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
                                lora_params[name] = param.full_tensor().detach().cpu() if hasattr(param, "full_tensor") else param.detach().cpu()
                            model = model.to(orig_dev)
                    torch.cuda.empty_cache()
            else:
                if self.base_sync_done:
                    lora_params = get_peft_model_state_dict(peft_model)
                else:
                    model = peft_model.base_model.model
                    orig_dev = "cpu" if "cpu" in next(model.parameters()).device else "cuda"
                    model = model.to("cpu")
                    for name, param in model.state_dict().items():
                        if any(x in name for x in ["_flat_param", "lora_"]):
                            continue
                        name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
                        lora_params[name] = param.detach().cpu()
                    model = model.to(orig_dev)
            return lora_params

        # NOTE: Basically, we only need `get_torch_device().empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        get_torch_device().empty_cache()

        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        if self.offload_param:
            load_fsdp_model_to_gpu(self.module)

        peft_config = None
        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if hasattr(peft_model, "peft_config"):
            peft_config = peft_model.peft_config.get("default", None)
            params = __collect_lora_params()
        else:
            params = self.module.state_dict()
            from verl.opd.qwen_lora import has_qwen_lora, qwen_lora_config
            if has_qwen_lora(self.module):
                from verl.opd.qwen_weight_export import dense_rollout_weights
                params = dense_rollout_weights(params, qwen_lora_config(self.module))
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)

        # Copy, not share memory
        load_format = "hf" if self.full_params else "dtensor"
        transfer_started = time.perf_counter()

        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            self.inference_engine.sync_model_weights(params, load_format=load_format)
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            del params
        else:
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["weights"])
            else:
                self.inference_engine.wake_up()

            # update model params
            self.update_params(params, peft_config=peft_config)
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            del params
            if self.offload_param:
                offload_fsdp_model_to_cpu(self.module)
            get_torch_device().empty_cache()

            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["kv_cache"])

        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.gen_random_states)
            self._opd_rng_switched = True
        if self._frozen_batch_guard:
            get_torch_device().synchronize()
            self.last_rollout_timing = {"weight_transfer_seconds": time.perf_counter() - transfer_started,
                                       "sharding_enter_seconds": time.perf_counter() - entered_at}

    def _enter_native(self, entered_at):
        """Prepare TP1 native weights through matched, fail-closed phases.

        Tensor materialization has its own checked collectives. It must not
        sit inside an outer guard, and vLLM's local loader must receive only
        dense tensors so a loader failure cannot compete with a peer's gather.
        """
        from verl.opd.qwen_lora import has_qwen_lora, qwen_lora_config
        from verl.opd.qwen_weight_export import dense_rollout_weights

        def prepare_local():
            from verl.opd.qwen_lora import validate_qwen_lora_frozen
            # vllm_version selects legacy vendored adapters and remains None
            # for modern vLLM. Authenticate the installed package separately.
            if self.tp_size != 1 or vllm_package_version != "0.8.5":
                raise ValueError(
                    "guarded native vLLM entry requires TP1 and vLLM 0.8.5 "
                    f"(observed TP{self.tp_size}, package {vllm_package_version!r})"
                )
            get_torch_device().empty_cache()
            if self.offload_param:
                load_fsdp_model_to_gpu(self.module)
            if has_qwen_lora(self.module):
                validate_qwen_lora_frozen(self.module)
            return "tags" in inspect.signature(self.inference_engine.wake_up).parameters

        tagged_wakeup = self._guard_stage("entry local preparation", prepare_local)
        state = self._guard_stage("state dict", self.module.state_dict)

        def actor_configuration():
            model = getattr(self.module, "_fsdp_wrapped_module", self.module)
            if hasattr(model, "peft_config") or not getattr(model, "_opd_qwen_replay_arithmetic", False):
                raise ValueError("guarded vLLM transfer requires a native dense or merged-LoRA actor")
            return qwen_lora_config(self.module) if has_qwen_lora(self.module) else None

        config = self._guard_stage("native actor configuration", actor_configuration)
        transfer_started = time.perf_counter()
        params = dense_rollout_weights(state, config, stage=self._guard_stage)
        del state
        self._guard_stage("wake weights", lambda: self.inference_engine.wake_up(tags=["weights"])
                          if tagged_wakeup else self.inference_engine.wake_up())

        def update_local():
            if any(hasattr(value, "full_tensor") or hasattr(value, "local_shards") for value in params.values()):
                raise ValueError("native vLLM local weight loading received a distributed tensor")
            self.update_params(params, peft_config=None)

        self._guard_stage("dense weight update", update_local)

        def install_arithmetic():
            from verl.opd.qwen_vllm_arithmetic import install_qwen_vllm_replay_arithmetic, verify_qwen_vllm_weights
            from verl.opd.qwen_vllm_attention import install_qwen_vllm_attention, qwen_vllm_attention_telemetry
            identity = install_qwen_vllm_replay_arithmetic(
                self.model_runner.model, self.model_config, backend="native_fa3_v2",
                tensor_parallel_size=self.tp_size,
                engine_config=self.inference_engine.llm_engine.get_vllm_config(),
            )
            identity.update(verify_qwen_vllm_weights(self.model_runner.model, params))
            attention = [install_qwen_vllm_attention(layer.self_attn.attn.impl, telemetry_limit=32 if index == 0 else 0)
                         for index, layer in enumerate(self.model_runner.model.model.layers)]
            qwen_vllm_attention_telemetry(self.model_runner.model, reset=True)
            identity.update(attention="opd_fa3_one_split", attention_layers=attention)
            self.inference_engine._opd_qwen_vllm_arithmetic = identity

        # Eager hooks live on the actual external-launcher worker model. This
        # matched phase finishes on every rank before any new requests start.
        self._guard_stage("native rollout arithmetic", install_arithmetic)
        del params
        self._guard_stage("actor offload", lambda: offload_fsdp_model_to_cpu(self.module) if self.offload_param else None)
        self._guard_stage("entry cache release", lambda: get_torch_device().empty_cache())
        self._guard_stage("wake KV cache", lambda: self.inference_engine.wake_up(tags=["kv_cache"]) if tagged_wakeup else None)

        def switch_rng():
            if self.device_mesh is not None:
                self.torch_random_states = get_torch_device().get_rng_state()
                # Restore even when set_rng_state fails after touching state.
                self._opd_rng_switched = True
                get_torch_device().set_rng_state(self.gen_random_states)

        self._guard_stage("entry RNG switch", switch_rng)
        self._guard_stage("entry synchronization", lambda: get_torch_device().synchronize())
        self.last_rollout_timing = {"weight_transfer_seconds": time.perf_counter() - transfer_started,
                                   "sharding_enter_seconds": time.perf_counter() - entered_at,
                                   "categorical_arithmetic": dict(self.inference_engine._opd_qwen_vllm_arithmetic)}

    def _guard_stage(self, stage, operation):
        from verl.opd.vllm_lifecycle import finish_vllm_stage
        result, error = None, None
        try:
            result = operation()
        except BaseException as failure:
            error = failure
        try:
            finish_vllm_stage(self.inference_engine, torch.distributed, error, stage)
        except BaseException:
            if self._opd_rng_switched:
                get_torch_device().set_rng_state(self.torch_random_states)
                self._opd_rng_switched = False
            raise
        return result

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        if self._frozen_batch_guard:
            from verl.opd.vllm_lifecycle import require_idle_vllm

            def completed():
                if exc_value is not None:
                    raise exc_value
                require_idle_vllm(self.inference_engine)

            # This is the sole generation completion collective. Generation
            # and TP1 postprocessing contain no competing world collectives.
            # Even a fast healthy rank must wait here before sleeping its engine.
            self._guard_stage("rollout completion", completed)
            from verl.opd.qwen_vllm_attention import qwen_vllm_attention_telemetry
            self.last_rollout_timing["categorical_attention"] = self._guard_stage(
                "attention telemetry", lambda: qwen_vllm_attention_telemetry(self.model_runner.model))
            started = time.perf_counter()
            self._guard_stage("memory release", self._release_after_rollout)
            self.last_rollout_timing["release_memory_seconds"] = time.perf_counter() - started
            return
        self._release_after_rollout()

    def _release_after_rollout(self):
        # TODO(ZSL): check this
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            self.inference_engine.offload_model_weights()
        else:
            self.inference_engine.sleep(level=1)

        self.module.train()

        # add empty cache after each compute
        get_torch_device().empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
        self._opd_rng_switched = False

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            group = vllm_ps.get_tensor_model_parallel_group()
        else:
            group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]

    def update_params(self, updated_params, peft_config=None):
        model = self.model_runner.model
        if peft_config:
            if self.base_sync_done:
                lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
                lora_reqest = TensorLoRARequest(
                    lora_name=f"{lora_int_id}",
                    lora_int_id=lora_int_id,
                    lora_path="simon_lora_path",
                    peft_config=asdict(peft_config),
                    lora_tensors=updated_params,
                )
                self.inference_engine.llm_engine.add_lora(lora_reqest)
                logger.info(f"vLLM load weights, loaded_params: {len(updated_params)}")
                return
            else:

                def replace_lora_wrapper(k):
                    stacked_params = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                    if any([k.endswith(f"{s}.weight") for s in stacked_params]):
                        return k.replace(".weight", ".base_layer.weight")
                    if any([k.endswith(f"{s}.bias") for s in stacked_params]):
                        return k.replace(".bias", ".base_layer.bias")
                    return k

                updated_params = {replace_lora_wrapper(k): v for k, v in updated_params.items()}

        patch_vllm_moe_model_weight_loader(model)
        device = get_torch_device().current_device()  # used when fsdp2 set cpu_offload_policy
        loaded_params = model.load_weights(((name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param) for name, param in updated_params.items()))

        self.base_sync_done = True
        logger.info(f"vLLM load weights, loaded_params: {len(loaded_params) if loaded_params else -1}")
