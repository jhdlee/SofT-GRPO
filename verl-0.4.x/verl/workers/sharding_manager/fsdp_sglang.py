# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import asyncio
import logging
import os
import time

import torch
import torch.distributed as dist
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang.srt.utils import MultiprocessingSerializer
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.torch_functional import check_device_is_available
from verl.workers.rollout.sglang_rollout.request_dispatch import (
    check_collective_error,
    poison_engine,
    require_idle_engine,
)

from .base import BaseShardingManager

# from vllm.distributed import parallel_state as sglang_ps
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor


class FSDPSGLangShardingManager(BaseShardingManager):
    @check_device_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: Engine,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        self._opd_poisoned = None
        self._opd_rng_switched = False
        self.last_rollout_timing = {}

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
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    @GPUMemoryLogger(role="FSDPSGLangShardingManager enter", logger=logger)
    def __enter__(self):
        self.last_rollout_timing = {}
        entered_at = time.perf_counter()
        self._guard_stage("entry readiness", self._require_idle)
        loop = self._guard_stage("entry event loop", asyncio.get_event_loop)
        params = self._prepare_weights(readiness_checked=True)
        loop.run_until_complete(self.update_weights(params))
        del params
        self._finish_entry()
        self.last_rollout_timing["sharding_enter_seconds"] = time.perf_counter() - entered_at

    def _require_idle(self):
        if self._opd_poisoned:
            raise RuntimeError(f"rollout sharding manager is poisoned: {self._opd_poisoned}")
        require_idle_engine(self.inference_engine)

    def _finish_stage(self, error, stage):
        """Retire every manager at one matched phase boundary on any failure.

        Callers must not wrap a sequence of these phases in another failure
        collective: a rank catching an earlier failure could otherwise match
        that outer collective with a peer's next inner phase.
        """
        try:
            check_collective_error(self.inference_engine, dist, error, stage)
        except BaseException as failure:
            self._opd_poisoned = f"{type(failure).__name__}: {failure}"
            poison_engine(self.inference_engine, self._opd_poisoned)
            if getattr(self, "_opd_rng_switched", False):
                self._opd_rng_switched = False
                torch.cuda.set_rng_state(self.torch_random_states)
            raise

    def _guard_stage(self, stage, operation):
        result, error = None, None
        try:
            result = operation()
        except BaseException as failure:
            error = failure
        self._finish_stage(error, stage)
        return result

    def _prepare_weights(self, readiness_checked=False):
        if not readiness_checked:
            self._guard_stage("entry readiness", self._require_idle)

        def prepare_local():
            from verl.opd.qwen_lora import has_qwen_lora, validate_qwen_lora_frozen
            torch.cuda.empty_cache()
            log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
            if self.offload_param:
                load_fsdp_model_to_gpu(self.module)
            if has_qwen_lora(self.module):
                validate_qwen_lora_frozen(self.module)

        # Finish local readiness on every DP rank before FSDP state_dict can
        # enter its own collectives. Fatal process/NCCL loss still requires the
        # process supervisor; Python/RPC failures must never skip a phase.
        self._guard_stage("entry local preparation", prepare_local)
        params = self._guard_stage("state dict", self.module.state_dict)

        from verl.opd.qwen_lora import has_qwen_lora, qwen_lora_config
        if has_qwen_lora(self.module):
            from verl.opd.qwen_weight_export import dense_rollout_weights
            return dense_rollout_weights(params, qwen_lora_config(self.module), stage=self._guard_stage)

        def transfer_local():
            log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)
            device = torch.cuda.current_device()
            return {k: v.to(device, non_blocking=True) if fsdp_version(self.module) == 2 else v for k, v in params.items()}

        return self._guard_stage("state dict transfer", transfer_local)

    def _finish_entry(self):
        def finish_local():
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            if self.offload_param:
                offload_fsdp_model_to_cpu(self.module)
            torch.cuda.empty_cache()
            log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)
            if self.device_mesh is not None:
                self.torch_random_states = torch.cuda.get_rng_state()
                self._opd_rng_switched = True
                torch.cuda.set_rng_state(self.gen_random_states)

        self._guard_stage("entry completion", finish_local)

    @GPUMemoryLogger(role="FSDPSGLangShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            # Include failures in TP postprocessing, outside the adapter, so
            # healthy ranks cannot enter normal memory release on their own.
            check_collective_error(self.inference_engine, dist, exc_value, "sharding context")
        except BaseException as error:
            # async_generate cancellation leaves scheduler work alive. A failed
            # outer batch must never release/reuse that engine through the
            # ordinary inference-to-training memory transition.
            self._opd_poisoned = f"{type(error).__name__}: {error}"
            poison_engine(self.inference_engine, self._opd_poisoned)
            if self.device_mesh is not None:
                torch.cuda.set_rng_state(self.torch_random_states)
            raise
        log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.release_memory())
        except BaseException as error:
            self._opd_poisoned = f"{type(error).__name__}: {error}"
            poison_engine(self.inference_engine, self._opd_poisoned)
            if self.device_mesh is not None:
                torch.cuda.set_rng_state(self.torch_random_states)
            raise
        log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        self._opd_rng_switched = False

    async def update_weights(self, params):
        self._guard_stage("weight update readiness", self._require_idle)
        resumed_at = time.perf_counter()
        resume_error = None
        try:
            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                await self.inference_engine.resume_memory_occupation()
        except BaseException as error:
            resume_error = error
        self.last_rollout_timing["memory_resume_seconds"] = time.perf_counter() - resumed_at
        self._finish_stage(resume_error, "memory resume")
        synchronized_at = time.perf_counter()

        named_tensors = self._guard_stage("weight inventory preparation", lambda: list(params.items()))

        def validate_inventory():
            names = [name for name, _ in named_tensors]
            if dist.is_initialized():
                inventories = [None] * dist.get_world_size()
                dist.all_gather_object(inventories, names)
                if any(inventory != names for inventory in inventories):
                    raise RuntimeError("rollout ranks have different ordered weight inventories")

        self._guard_stage("weight inventory", validate_inventory)
        load_format = None
        for tensor_index, (name, tensor) in enumerate(named_tensors):
            materialized = self._guard_stage(
                f"weight materialization {name}", lambda: _preprocess_tensor_for_update_weights(tensor),
            )

            def prepare_transfer():
                tp_mesh = self.device_mesh["infer_tp"]
                serialized = MultiprocessingSerializer.serialize(materialized)
                gathered = [None] * tp_mesh.mesh.size()[0] if tp_mesh.get_local_rank() == 0 else None
                return serialized, gathered, tp_mesh.mesh.tolist()[0], tp_mesh.get_group()

            serialized_tensor, gathered_serialized_tensors, destination, group = self._guard_stage(
                f"weight serialization {name}", prepare_transfer,
            )
            # The serialization guard completes globally before any TP group
            # enters gather_object; the next guard completes before any RPC.
            self._guard_stage(
                f"weight gather {name}", lambda: dist.gather_object(
                    obj=serialized_tensor, object_gather_list=gathered_serialized_tensors,
                    dst=destination, group=group,
                ),
            )
            update_error = None
            try:
                if self.device_mesh["infer_tp"].get_local_rank() == 0:
                    await self.inference_engine.update_weights_from_tensor(
                        named_tensors=[(name, LocalSerializedTensor(values=gathered_serialized_tensors))],
                        load_format=load_format,
                        flush_cache=tensor_index == len(named_tensors) - 1,
                    )
            except BaseException as error:
                update_error = error
            self._finish_stage(update_error, f"weight update {name}")
        self.last_rollout_timing["weight_sync_seconds"] = time.perf_counter() - synchronized_at

    async def release_memory(self):
        self._guard_stage("memory release readiness", self._require_idle)
        released_at = time.perf_counter()
        release_error = None
        try:
            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                await self.inference_engine.release_memory_occupation()
        except BaseException as error:
            release_error = error
        self.last_rollout_timing["memory_release_seconds"] = time.perf_counter() - released_at
        self._finish_stage(release_error, "memory release")

    @GPUMemoryLogger(role="FSDPSGLangShardingManager enter", logger=logger)
    async def wake_up(self):
        self.last_rollout_timing = {}
        entered_at = time.perf_counter()
        params = self._prepare_weights()
        await self.update_weights(params)
        del params
        self._finish_entry()
        self.last_rollout_timing["sharding_enter_seconds"] = time.perf_counter() - entered_at

    @GPUMemoryLogger(role="FSDPSGLangShardingManager exit", logger=logger)
    async def sleep(self):
        log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
        await self.release_memory()
        log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        self._opd_rng_switched = False

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        group = self.device_mesh["infer_tp"].get_group()

        all_gather_data_proto(data=data, process_group=group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]
