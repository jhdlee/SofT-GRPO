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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import hydra
import ray

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    started_local_ray = not ray.is_initialized()
    if started_local_ray:
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
            num_cpus=config.ray_init.num_cpus,
            include_dashboard=False,
        )

    try:
        runner = TaskRunner.remote()
        ray.get(runner.run.remote(config))
        # create a timeline trace file to analyze the performance
        timeline_json_file = config.ray_init.get("timeline_json_file", None)
        if timeline_json_file:
            ray.timeline(filename=timeline_json_file)
    finally:
        # The two-GPU gate intentionally runs several independent cases in one
        # allocation.  Do not leave a previous driver's workers attached to the
        # next case (or orphaned when a case fails).
        if started_local_ray:
            ray.shutdown()


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf, open_dict

        from verl.utils.fs import copy_to_local

        OmegaConf.resolve(config)

        # Keep ``algorithm.opd`` as the one public source of truth while making
        # an independent, fully resolved copy available to the colocated
        # actor/rollout/ref workers (which receive only actor_rollout_ref).
        from verl.opd import OPDConfig
        from verl.trainer.ppo.opd_driver import RolloutIntegrityConfig

        resolved_opd = OmegaConf.to_container(config.algorithm.opd, resolve=True)
        OPDConfig.from_mapping(resolved_opd)
        resolved_integrity = OmegaConf.to_container(config.trainer.rollout_integrity, resolve=True)
        integrity_config = RolloutIntegrityConfig.from_mapping(resolved_integrity)
        with open_dict(config):
            config.actor_rollout_ref.opd = OmegaConf.create(resolved_opd)
            config.actor_rollout_ref.rollout_integrity = OmegaConf.create(resolved_integrity)

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)  # used for multimodal LLM, could be none

        if integrity_config.enabled:
            for tag in ("<think>", "</think>"):
                tag_ids = tokenizer.encode(tag, add_special_tokens=False)
                if len(tag_ids) != 1:
                    raise RuntimeError(f"rollout integrity requires atomic {tag!r}; got token IDs {tag_ids}")
            rollout = config.actor_rollout_ref.rollout
            incompatible = []
            if not bool(rollout.get("deterministic_sampling", False)):
                incompatible.append(
                    "deterministic_sampling must be true for exact request-local replay"
                )
            if bool(rollout.get("enable_soft_thinking", True)):
                if rollout.name != "sglang":
                    incompatible.append("native-soft rollout.name must be sglang")
                if not rollout.add_noise_gumbel_softmax:
                    incompatible.append("add_noise_gumbel_softmax must be true")
                if not rollout.noise_gumbel or not rollout.noise_on_logits:
                    incompatible.append("shifted-Gumbel logit noise must be active")
                if rollout.add_noise_dirichlet or rollout.noise_gaussian or rollout.noise_on_inputs:
                    incompatible.append("alternative continuous-noise paths must be disabled")
                if rollout.top_k != 5 or rollout.after_thinking_top_k != 5:
                    incompatible.append("thinking and answer top_k must both be 5")
                if float(rollout.temperature) != 1.0 or float(rollout.after_thinking_temperature) != 1.0:
                    incompatible.append("thinking and answer LM temperatures must both be 1.0")
                if float(rollout.gumbel_softmax_temperature) != 0.1:
                    incompatible.append("gumbel_softmax_temperature must be 0.1")
                if not config.actor_rollout_ref.model.use_remove_padding:
                    incompatible.append("native-soft replay requires model.use_remove_padding=true")
            else:
                if rollout.name != "vllm":
                    incompatible.append("categorical hard rollout.name must be vllm")
                if float(rollout.temperature) != 1.0:
                    incompatible.append("categorical hard temperature must be 1.0")
                if float(rollout.top_p) != 1.0:
                    incompatible.append("categorical hard top_p must be 1.0")
                if int(rollout.top_k) != -1:
                    incompatible.append("categorical hard top_k must be unrestricted (-1)")
                if any(
                    bool(rollout.get(name, False))
                    for name in (
                        "add_noise_gumbel_softmax",
                        "add_noise_dirichlet",
                        "noise_gumbel",
                        "noise_gaussian",
                        "noise_on_inputs",
                        "noise_on_logits",
                    )
                ):
                    incompatible.append("categorical hard rollout must disable continuous-noise paths")
            if incompatible:
                raise RuntimeError("rollout integrity configuration failed: " + "; ".join(incompatible))

        # vllm early verify
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
