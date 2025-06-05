import os
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
from torch.utils.data import Sampler
from transformers import Trainer
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    get_parameter_names,
    has_length,
    is_sagemaker_mp_enabled,
    logger,
)
import time
from transformers import TrainerCallback
from transformers import is_apex_available
if is_apex_available():
    from apex import amp

class TimeProfilerCallback(TrainerCallback):
    def __init__(self):
        self.step_start_time = None
        self.data_loading_time = 0.0
        self.forward_time = 0.0
        self.backward_time = 0.0

    def on_step_begin(self, args, state, control, **kwargs):
        # 记录步骤开始时间（数据加载结束时刻）
        self.step_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        # 计算总耗时
        step_duration = time.time() - self.step_start_time

        # 输出耗时分析（保留3位小数）
        if torch.distributed.get_rank() == 0:
            print(
                f"\n[Step {state.global_step}] "
                f"Data Loading: {self.data_loading_time:.10f}s | "
                f"Forward: {self.forward_time:.10f}s | "
                f"Backward: {self.backward_time:.10f}s | "
                f"Total: {step_duration:.3f}s"
            )

        # 重置计时器
        self.data_loading_time = 0.0
        self.forward_time = 0.0
        self.backward_time = 0.0

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [
        mm_indices[i]
        for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)
    ]
    lang_shuffle = [
        lang_indices[i]
        for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)
    ]
    megabatch_size = world_size * batch_size
    mm_megabatches = [
        mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)
    ]
    lang_megabatches = [
        lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)
    ]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [
        indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)
    ]
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches
    ]
    megabatches = [
        split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches
    ]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)


class VITATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "mm_projector" in name or "vision_tower" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {p.data_ptr(): p.numel() for p in module.parameters()}.values()
                        )
                        logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2 ** 20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["mm_projector", "vision_resampler"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(
                self.model.named_parameters(), keys_to_match
            )

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        else:
            super(VITATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(VITATrainer, self)._save(output_dir, state_dict)

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], *args, **kwargs
    ) -> torch.Tensor:
        #tr_loss_step = super().training_step(model, inputs, *args, **kwargs)
        #return tr_loss_step
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        # 获取回调函数实例
        callback = self._get_time_profiler_callback()

        # 记录数据加载耗时（从上一step结束到数据加载完成）
        data_loading_start = time.time()
        inputs = self._prepare_inputs(inputs)
        if callback:
            callback.data_loading_time = time.time() - data_loading_start
        #import pdb; pdb.set_trace()
        # 前向传播计时
        forward_start = time.time()
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)
        if callback:
            callback.forward_time = time.time() - forward_start

        # 反向传播计时
        backward_start = time.time()
        if self.control.should_evaluate:
            inputs["use_cache"] = False
        with self.compute_loss_context_manager():
            loss, output = super().compute_loss(model, inputs, return_outputs=True)
        if self.control.should_evaluate or self.state.global_step % self.state.logging_steps == 0:
            prefix = "eval_" if self.control.should_evaluate else ""
            # Dirty hack, evaluation set is not shuffled, and tasks in a batch occurs consecutively. 
            # Use the task of the first item to roughly represent the batch
            task = output["tasks"][0]
            suffix = f"_{task}" if self.control.should_evaluate else ""
            logs = {
                f"{prefix}loss": round(loss.item(), 4), 
            }
            if self.control.should_evaluate:
                logs[f"{prefix}loss{suffix}"] = round(loss.item(), 4)
            if output[f"loss_text"] is not None:
                logs[f"{prefix}loss_text{suffix}"] = round(output["loss_text"].item(), 4)
            for i, audio_loss in enumerate(output["loss_audios"]):
                logs[f"{prefix}audio_loss_{i}{suffix}"] = round(audio_loss.item(), 4)
            for i, state_loss in enumerate(output["loss_states"]):
                logs[f"{prefix}state_loss_{i}{suffix}"] = round(state_loss.item(), 4)
            self.log(logs)
            


        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        #import pdb; pdb.set_trace()

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)
        if callback:
            callback.backward_time = time.time() - backward_start
        
        return loss.detach() / self.args.gradient_accumulation_steps

    def _get_time_profiler_callback(self):
        # 从回调列表中查找 TimeProfilerCallback 实例
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, TimeProfilerCallback):
                return callback
        return None

    def compute_loss(self, model, inputs, return_outputs=False):
        
        if self.control.should_evaluate:
            inputs["use_cache"] = False
        loss, output = super().compute_loss(model, inputs, return_outputs=True)

        if self.control.should_evaluate or self.state.global_step % self.state.logging_steps == 0:
            prefix = "eval_" if self.control.should_evaluate else ""
            # Dirty hack, evaluation set is not shuffled, and tasks in a batch occurs consecutively. 
            # Use the task of the first item to roughly represent the batch
            task = output["tasks"][0]
            suffix = f"_{task}" if self.control.should_evaluate else ""
            logs = {
                f"{prefix}loss": round(loss.item(), 4), 
            }
            if self.control.should_evaluate:
                logs[f"{prefix}loss{suffix}"] = round(loss.item(), 4)
            if output[f"loss_text"] is not None:
                logs[f"{prefix}loss_text{suffix}"] = round(output["loss_text"].item(), 4)
            for i, audio_loss in enumerate(output["loss_audios"]):
                logs[f"{prefix}audio_loss_{i}{suffix}"] = round(audio_loss.item(), 4)
            for i, state_loss in enumerate(output["loss_states"]):
                logs[f"{prefix}state_loss_{i}{suffix}"] = round(state_loss.item(), 4)
            self.log(logs)
        
        return (loss, output) if return_outputs else loss
