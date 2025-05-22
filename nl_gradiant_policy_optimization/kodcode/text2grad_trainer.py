import inspect
import math
import os
import re
import time
import typing
import warnings
from contextlib import nullcontext
from typing import Callable, List, Optional, Union

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, gather_object, is_deepspeed_available, DummyOptim, DummyScheduler
from datasets import Dataset
from huggingface_hub import whoami
from packaging import version
from torch.optim import Adam
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

from trl.core import (
    WANDB_PADDING,
    PPODecorators,
    clip_by_value,
    convert_to_scalar,
    entropy_from_logits,
    flatten_dict,
    logprobs_from_logits,
    masked_mean,
    masked_var,
    masked_whiten,
    set_seed,
    stack_dicts,
    stats_to_np,
)

from accelerate.utils import is_npu_available, is_xpu_available

from trl.models import (
    SUPPORTED_ARCHITECTURES,
    PreTrainedModelWrapper,
    create_reference_model,
    unwrap_model_for_generation,
)
from trl.trainer import AdaptiveKLController, BaseTrainer, FixedKLController, PPOConfig, RunningMoments
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model

if is_deepspeed_available():
    import deepspeed


MODEL_CARD_TEMPLATE = """---
license: apache-2.0
tags:
- trl
- ppo
- transformers
- reinforcement-learning
---

# {model_name}

This is a [TRL language model](https://github.com/huggingface/trl) that has been fine-tuned with reinforcement learning to
 guide the model outputs according to a value, function, or human feedback. The model can be used for text generation.

## Usage

To use this model for inference, first install the TRL library:

```bash
python -m pip install trl
```

You can then generate text as follows:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="{model_id}")
outputs = generator("Hello, my llama is cute")
```

If you want to use the model for training or to obtain the outputs from the value head, load the model as follows:

```python
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

tokenizer = AutoTokenizer.from_pretrained("{model_id}")
model = AutoModelForCausalLMWithValueHead.from_pretrained("{model_id}")


inputs = tokenizer("Hello, my llama is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
```
"""


class Text2GradTrainer(BaseTrainer):
    """
    The Text2GradTrainer uses Proximal Policy Optimization to optimise language models.
    Note, this trainer is heavily inspired by the original OpenAI learning to summarize work here:
    https://github.com/openai/summarize-from-feedback

    Attributes:
        **config** (`PPOConfig`) -- Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more
            details.
        **model** (`PreTrainedModelWrapper`) -- Model to be optimized, Hugging Face transformer model with a value head.
            Check the documentation of `PreTrainedModelWrapper` for more details.
        **ref_model** (`PreTrainedModelWrapper`, *optional*) -- Reference model to be used for KL penalty, Hugging Face
            transformer model with a casual language modelling head. Check the documentation of `PreTrainedModelWrapper`
            for more details. If no reference model is provided, the trainer will create a reference model with the same
             architecture as the model to be optimized with shared layers.
        **tokenizer** (`PreTrainedTokenizerBase`) -- Tokenizer to be used for encoding the
            data. Check the documentation of `transformers.PreTrainedTokenizer` and
            `transformers.PreTrainedTokenizerFast` for more details.
        **dataset** (Union[`torch.utils.data.Dataset`, `datasets.Dataset`], *optional*) -- PyTorch dataset or Hugging
            Face dataset. This is used to create a PyTorch dataloader. If no dataset is provided, the dataloader must be
             created outside the trainer users needs to design their own dataloader and make sure the batch
            size that is used is the same as the one specified in the configuration object.
        **optimizer** (`torch.optim.Optimizer`, *optional*) -- Optimizer to be used for training. If no optimizer is
            provided, the trainer will create an Adam optimizer with the learning rate specified in the configuration
            object.
        **data_collator** (DataCollatorForLanguageModeling, *optional*) -- Data collator to be used for training and
            passed along the dataloader
        **num_shared_layers** (int, *optional*) -- Number of layers to be shared between the model and the reference
            model, if no reference model is passed. If no number is provided, all the layers will be shared.
        **lr_scheduler** (`torch.optim.lr_scheduler`, *optional*) -- Learning rate scheduler to be used for training.
    """

    _tag_names = ["trl", "ppo"]

    def __init__(
            self,
            config: Optional[PPOConfig] = None,
            model: Optional[PreTrainedModelWrapper] = None,
            ref_model: Optional[PreTrainedModelWrapper] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            data_collator: Optional[typing.Callable] = None,
            num_shared_layers: Optional[int] = None,
            lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        Initialize PPOTrainer.

        Args:
            config (`PPOConfig`):
                Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more details.
            model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a value head.
            ref_model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a casual language modelling head. Used for KL penalty
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                Hugging Face tokenizer
            dataset (Optional[Union[`torch.utils.data.Dataset`, `datasets.Dataset`]]):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
                will be preprocessed by removing the columns that are not used by the model. If none is passed,
                a warning will be raised in a multi-GPU setting.
            optimizer (Optional[`torch.optim.Optimizer`]):
                Optimizer used for training. If `None`, the `Adam` is used as default.
            data_collator (Optional[function]):
                Data collator function.
            num_shared_layers (Optional[int]):
                Number of shared layers between the model and the reference model. If `None`, all layers are shared.
                used only if `ref_model` is `None`.
            lr_scheduler (Optional[`torch.optim.lr_scheduler`]):
                Learning rate scheduler used for training.
        """
        super().__init__(config)

        # initial seed for reproducible experiments
        set_seed(config.seed)

        # Step 0: check positional arguments validity
        if not isinstance(config, PPOConfig):
            raise ValueError(f"config must be a PPOConfig, got {type(config)}")
        if not isinstance(tokenizer, (PreTrainedTokenizerBase)):
            raise ValueError(
                f"tokenizer must be a PreTrainedTokenizerBase like a PreTrainedTokenizer or a PreTrainedTokenizerFast, got {type(tokenizer)}"
            )
        if not isinstance(model, (SUPPORTED_ARCHITECTURES)):
            raise ValueError(
                f"model must be a PreTrainedModelWrapper, got {type(model)} - supported architectures are: {SUPPORTED_ARCHITECTURES}"
            )
        # Step 1: Initialize Accelerator
        self.accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_config=ProjectConfiguration(**config.project_kwargs),
            **config.accelerator_kwargs,
        )

        config.world_size = self.accelerator.num_processes
        config.global_backward_batch_size = config.backward_batch_size * config.world_size
        config.global_batch_size = config.batch_size * config.world_size

        self.model = model
        self.model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
        self.is_peft_model = getattr(self.model, "is_peft_model", False)
        self.is_peft_model = True
        config.is_encoder_decoder = self.is_encoder_decoder
        config.is_peft_model = self.is_peft_model

        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"
        self.accelerator.init_trackers(
            config.tracker_project_name,
            config=dict(trl_ppo_trainer_config=config.to_dict()) if not is_using_tensorboard else config.to_dict(),
            init_kwargs=config.tracker_kwargs,
        )
        self.is_using_text_environment = getattr(config, "use_text_environment", False)

        if isinstance(ref_model, SUPPORTED_ARCHITECTURES):
            self.ref_model = ref_model
            if num_shared_layers is not None:
                warnings.warn(
                    "num_shared_layers is ignored when ref_model is provided. Two different models are used for the "
                    "model and the reference model and no layers are shared.",
                    UserWarning,
                )
        elif ref_model is None and not self.is_peft_model:
            self.ref_model = create_reference_model(self.model, num_shared_layers=num_shared_layers)
        elif self.is_peft_model:
            self.ref_model = None
        else:
            raise ValueError(
                f"ref_model must be a PreTrainedModelWrapper or `None`, got {type(ref_model)} - supported "
                f"architectures are: {SUPPORTED_ARCHITECTURES} "
            )
        self.optional_peft_ctx = (
            self.accelerator.unwrap_model(self.model).pretrained_model.disable_adapter
            if self.is_peft_model
            else nullcontext
        )

        if not (isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast)):
            raise ValueError(
                "tokenizer must be a transformers.PreTrainedTokenizer or transformers.PreTrainedTokenizerFast"
            )
        self.tokenizer = tokenizer

        if dataset is not None and not (isinstance(dataset, torch.utils.data.Dataset) or isinstance(dataset, Dataset)):
            raise ValueError("dataset must be a torch.utils.data.Dataset or datasets.Dataset")
        elif dataset is None:
            warnings.warn(
                "No dataset is provided. Make sure to set config.batch_size to the correct value before training.",
                UserWarning,
            )
        self.dataset = dataset
        self._signature_columns = None
        if self.dataset is not None:
            self.dataloader = self.prepare_dataloader(self.dataset, data_collator)
        elif self.dataset is None and self.accelerator.num_processes > 1:
            warnings.warn(
                "No dataset is provided. In a multi-GPU setting, this will lead to an error. You should"
                " prepare your dataloader yourself with `dataloader = ppo_trainer.accelerator.prepare(dataloader)`"
                " and using `torch.utils.data.DataLoader`, or pass a dataset to the `PPOTrainer`. Please "
                " refer to the documentation for more details.",
                UserWarning,
            )
            self.dataloader = None
        else:
            self.dataloader = None

        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        if optimizer is None:
            if not isinstance(optimizer, DummyOptim):
                lr = optimizer.param_groups[0]['lr']
                weight_decay = optimizer.param_groups[0].get('weight_decay', 0.0)
                beta1 = optimizer.param_groups[0].get('betas', (0.9, 0.999))[0]
                beta2 = optimizer.param_groups[0].get('betas', (0.9, 0.999))[1]
                epsilon = optimizer.param_groups[0].get('eps', 1e-8)

                self.optimizer = DummyOptim(
                    params=model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=(beta1, beta2),
                    eps=epsilon
                )
            else:
                self.optimizer = optimizer
        else:
            self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler
        current_torch_version = torch.__version__
        if current_torch_version > '2.0':
            is_torch_greater_2_0 = True
        else:
            is_torch_greater_2_0 = False
        if self.lr_scheduler is not None:
            lr_scheduler_class = (
                torch.optim.lr_scheduler._LRScheduler
                if not is_torch_greater_2_0
                else torch.optim.lr_scheduler.LRScheduler
            )

            if not isinstance(self.lr_scheduler, lr_scheduler_class):
                raise ValueError(
                    "lr_scheduler must be a torch.optim.lr_scheduler._LRScheduler or torch.optim.lr_scheduler.LRScheduler (for torch >= 2.0)"
                )

        if self.config.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(self.config.init_kl_coef, self.config.target, self.config.horizon)
        else:
            self.kl_ctl = FixedKLController(self.config.init_kl_coef)

        is_deepspeed_used = self.accelerator.distributed_type == "DEEPSPEED" and hasattr(
            self.accelerator.state, "deepspeed_plugin"
        )

        print(f"Creating DummyOptim for DeepSpeed compatibility (distributed: {self.accelerator.num_processes > 1})")
        if optimizer is not None:
            try:
                lr = optimizer.param_groups[0]['lr']
                weight_decay = optimizer.param_groups[0].get('weight_decay', 0.0)
                betas = optimizer.param_groups[0].get('betas', (0.9, 0.999))
                eps = optimizer.param_groups[0].get('eps', 1e-8)

                print(f"Creating DummyOptim with params: lr={lr}, weight_decay={weight_decay}")
                self.optimizer = DummyOptim(
                    params=filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=betas,
                    eps=eps
                )
            except Exception as e:
                print(f"Error creating DummyOptim from optimizer: {str(e)}")
                print("Creating default DummyOptim")
                self.optimizer = DummyOptim(
                    params=filter(lambda p: p.requires_grad, model.parameters()),
                    lr=config.learning_rate,
                    weight_decay=0.0
                )
        else:
            print("No optimizer provided, creating default DummyOptim")
            self.optimizer = DummyOptim(
                params=filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.learning_rate,
                weight_decay=0.0
            )

        if lr_scheduler is not None:
            try:
                print("Creating DummyScheduler")
                self.lr_scheduler = DummyScheduler(
                    self.optimizer,
                    total_num_steps=config.total_ppo_epochs * config.batch_size,
                    warmup_num_steps=0
                )
            except Exception as e:
                print(f"Error creating DummyScheduler: {str(e)}")
                self.lr_scheduler = None
        else:
            self.lr_scheduler = None

        self.current_step = 0

        print("Preparing with accelerator")
        if self.lr_scheduler is not None:
            (
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.dataloader,
            ) = self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.dataloader,
            )
        else:
            (
                self.model,
                self.optimizer,
                self.dataloader, 
            ) = self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.dataloader,
            )

        self.current_device = self.accelerator.device
        self.is_distributed = self.accelerator.num_processes > 1
        self.is_deepspeed = is_deepspeed_used

    def _filter_kwargs(self, kwargs, target_func):
        """
        filter the keyword arguments that are supported by the target function.

        Args:
            kwargs (dict):
                Keyword arguments
            target_func (function):
                Target function
        """
        return {k: v for k, v in kwargs.items() if k in inspect.signature(target_func).parameters.keys()}

    def prepare_dataloader(self, dataset: Union[torch.utils.data.Dataset, Dataset], data_collator=None):
        """
        Prepare the dataloader for training.

        Args:
            dataset (Union[`torch.utils.data.Dataset`, `datasets.Dataset`]):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
                will be preprocessed by removing the columns that are not used by the model.
            data_collator (Optional[function]):
                Data collator function.

        Returns:
            `torch.utils.data.DataLoader`: PyTorch dataloader
        """
        if isinstance(dataset, Dataset):
            dataset = self._remove_unused_columns(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
        )
        return dataloader

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            self._signature_columns += ["label", "query", "response"]

    def _remove_unused_columns(self, dataset: "Dataset"):
        if not self.config.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"],
                columns=columns,
                format_kwargs=dataset.format["format_kwargs"],
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def generate(
            self,
            query_tensor: Union[torch.Tensor, List[torch.Tensor]],
            length_sampler: Optional[Callable] = None,
            batch_size: int = 4,
            return_prompt: bool = True,
            generate_ref_response: bool = False,
            **generation_kwargs,
    ):
        """
        Generate response with the model given the query tensor.
        call the `generate` method of the model.

        Args:
            query_tensor (`torch.LongTensor`):
                A tensor of shape (`seq_len`) containing query tokens or a list of tensors of shape (`seq_len`).
            length_sampler (`Callable`, *optional*):
                Callable that returns the number of newly generated tokens.
            batch_size (`int`, *optional*):
                Batch size used for generation, defaults to `4`.
            return_prompt (`bool`, *optional*):
                If set to `False` the prompt is not returned but only the newly generated tokens, defaults to `True`.
            generate_ref_response (`bool`, *optional*):
                If set to `True` the reference response is also generated, defaults to `False`.
            generation_kwargs (dict[str, Any]):
                Keyword arguments for generation.

        Returns:
            `torch.LongTensor`: A tensor of shape (`batch_size`, `gen_len`) containing response tokens.
        """
        if generate_ref_response:
            ref_model = self.model if self.is_peft_model else self.ref_model
        if isinstance(query_tensor, List):
            response = self._generate_batched(
                self.model,
                query_tensor,
                length_sampler=length_sampler,
                batch_size=batch_size,
                return_prompt=return_prompt,
                **generation_kwargs,
            )
            if generate_ref_response:
                ref_response = self._generate_batched(
                    ref_model,
                    query_tensor,
                    length_sampler=length_sampler,
                    batch_size=batch_size,
                    return_prompt=return_prompt,
                    **generation_kwargs,
                )

        else:
            if len(query_tensor.shape) == 2:
                raise ValueError(
                    "query_tensor must be a tensor of shape (`seq_len`) or a list of tensors of shape (`seq_len`)"
                )

            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()

            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                response = unwrapped_model.generate(input_ids=query_tensor.unsqueeze(dim=0), **generation_kwargs)

            if generate_ref_response:
                with unwrap_model_for_generation(
                        ref_model, self.accelerator, is_peft_model=self.is_peft_model
                ) as unwrapped_model:
                    ref_response = unwrapped_model.generate(
                        input_ids=query_tensor.unsqueeze(dim=0), **generation_kwargs
                    )

            if not return_prompt and not self.is_encoder_decoder:
                response = response[:, query_tensor.shape[0]:]
                if generate_ref_response:
                    ref_response = ref_response[:, query_tensor.shape[0]:]

        if generate_ref_response:
            return response, ref_response
        return response

    def _generate_batched(
            self,
            model: PreTrainedModelWrapper,
            query_tensors: List[torch.Tensor],
            length_sampler: Optional[Callable] = None,
            batch_size: int = 4,
            return_prompt: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            remove_padding: bool = True,
            **generation_kwargs,
    ):
        outputs = []
        padding_side_default = self.tokenizer.padding_side
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = "right"  

        batch_size = min(len(query_tensors), batch_size)

        def clean_special_tokens(output_tensor):
            """Helper function to clean special tokens and get clean text"""
            special_tokens = {
                '<|start_header_id|>': '',
                '<|end_header_id|>': '',
                '<|eot_id|>': '',
                'assistant': '',
                'human': ''
            }

            output_ids = output_tensor.tolist()

            text = self.tokenizer.decode(output_ids)

            for token, replacement in special_tokens.items():
                text = text.replace(token, replacement)


            return self.tokenizer.encode(text, return_tensors='pt')[0]

        for i in range(0, len(query_tensors), batch_size):
            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()

            end_index = min(len(query_tensors), i + batch_size)

            batch = query_tensors[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch, "attention_mask": batch_mask}

            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            ).to(self.current_device)

            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                generations = unwrapped_model.generate(**padded_inputs, **generation_kwargs)

            for generation, mask in zip(generations, padded_inputs["attention_mask"]):
                if not return_prompt and not self.is_encoder_decoder:
                    output = generation[len(mask):]  # remove prompt

                if remove_padding:
                    output = clean_special_tokens(output)

                    if self.tokenizer.eos_token_id in output:
                        eos_positions = (output == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]

                        if len(eos_positions) > 0:
                            first_eos_pos = eos_positions[0].item()
                            output = output[:first_eos_pos]  

                            if len(output) > 0:
                                clean_text = self.tokenizer.decode(output)
                                if clean_text.strip():
                                    print(f"Cleaned text: {clean_text}")

                outputs.append(output)

        self.tokenizer.padding_side = padding_side_default
        return outputs

    def _step_safety_checker(
            self,
            batch_size: int,
            queries: List[torch.LongTensor],
            responses: List[torch.LongTensor],
            scores: List[torch.FloatTensor],
            masks: Optional[List[torch.LongTensor]] = None,
    ):
        """
        Check if the input data is valid for training.

        Args:
            batch_size (int):
                Batch size from the config file.
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            masks (List[`torch.LongTensor`], *optional*):
                list of optional tensors containing the masks of shape (`query_length` + `response_length`)
        Returns:
            `tuple`: The input processed data.
        """
        for name, tensor_list in zip(["queries", "responses", "scores"], [queries, responses, scores]):
            if not isinstance(tensor_list, list):
                raise ValueError(f"{name} must be a list of tensors - got {type(tensor_list)}")
            if not isinstance(tensor_list[0], torch.Tensor):
                raise ValueError(f"Elements in {name} must be tensors - got {type(tensor_list[0])}")
            if batch_size is not None and len(tensor_list) != batch_size:
                raise ValueError(
                    f"Batch size ({batch_size}) does not match number of examples - but got {len(tensor_list)} for: {name}"
                )

        queries = [tensor.to(self.current_device) for tensor in queries]
        responses = [tensor.to(self.current_device) for tensor in responses]
        scores = [tensor.to(self.current_device) for tensor in scores]
        masks = [tensor.to(self.current_device) for tensor in masks] if masks is not None else None

        for i, score in enumerate(scores):
            if score.dim() > 1:
                raise ValueError(f"Scores must be 1-dimensional - got {score.dim()} for {score}")
            elif score.dim() == 1:
                scores[i] = score.squeeze()

        return queries, responses, scores, masks

    @PPODecorators.empty_device_cache()
    def step(
            self,
            queries: List[torch.LongTensor],
            responses: List[torch.LongTensor],
            scores: List[torch.FloatTensor],
            words: List[List[str]] = None,
            response_masks: Optional[List[torch.LongTensor]] = None,
            mask_loss: str = "",
    ):
        """
        Run a PPO optimisation step given a list of queries, model responses, and rewards.
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            bs = len(responses)
            self.config.batch_size = bs
            queries, responses, scores, response_masks = self._step_safety_checker(
                bs, queries, responses, scores, response_masks
            )
            scores = [score.to(self.current_device) for score in scores]

            if self.config.use_score_scaling:
                scores_mean, scores_std = self.running.update(scores)
                tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
                score_scaling_factor = self.running.std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
                if self.config.use_score_norm:
                    scores = (scores - self.running.mean.to(**tensor_to_kwargs)) / score_scaling_factor
                else:
                    scores /= score_scaling_factor
                del tensor_to_kwargs, score_scaling_factor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if self.config.score_clip is not None:
                scores_dtype = scores.dtype
                scores = torch.clip(scores.float(), -self.config.score_clip, self.config.score_clip).to(dtype=scores_dtype)
                del scores_dtype
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if hasattr(self, "highest_reward"):
                if self.compare_step % self.config.compare_steps == 0:
                    curr_mean_reward = scores.mean()
                    if curr_mean_reward > self.highest_reward:
                        self.highest_reward = curr_mean_reward
                        self.push_to_hub(**self.push_to_hub_kwargs)
                    del curr_mean_reward
                self.compare_step += 1

            timing = dict()
            t0 = time.time()
            t = time.time()

            model_inputs = self.prepare_model_inputs(queries, responses)

            if self.is_distributed:
                pad_first = self.tokenizer.padding_side == "right"

                model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                    model_inputs["input_ids"],
                    dim=1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
                model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                    model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
                )
                if self.is_encoder_decoder:
                    model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                        model_inputs["decoder_input_ids"],
                        dim=1,
                        pad_index=self.tokenizer.pad_token_id,
                        pad_first=pad_first,
                    )
                    model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
                        model_inputs["decoder_attention_mask"],
                        dim=1,
                        pad_index=0,
                        pad_first=pad_first,
                    )
                del pad_first
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            model_inputs_names = list(model_inputs.keys())
            full_kl_penalty = self.config.kl_penalty == "full"

            with torch.no_grad():
                all_logprobs, logits_or_none, values, masks, all_tokens = self.batched_forward_pass(
                    self.model,
                    queries,
                    responses,
                    model_inputs,
                    response_masks=response_masks,
                    return_logits=full_kl_penalty,
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                with self.optional_peft_ctx():
                    ref_logprobs, ref_logits_or_none, _, _, _ = self.batched_forward_pass(
                        self.model if self.is_peft_model else self.ref_model,
                        queries,
                        responses,
                        model_inputs,
                        return_logits=full_kl_penalty,
                    )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            timing["time/ppo/forward_pass"] = time.time() - t

            scores, answers_tokens, answer_indices_tokens, total_skip_words_nums = self.rematch_scores(scores, words,
                                                                                                       all_tokens)
            del answers_tokens, answer_indices_tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.no_grad():
                t = time.time()
                if full_kl_penalty:
                    active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
                    ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

                    rewards, non_score_reward, kls = self.compute_rewards(
                        scores, active_full_logprobs, ref_full_logprobs, masks, all_tokens, words
                    )
                    del active_full_logprobs, ref_full_logprobs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    rewards, non_score_reward, kls = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks,
                                                                          all_tokens, words)
                timing["time/ppo/compute_rewards"] = time.time() - t

                t = time.time()
                values, advantages, returns = self.compute_advantages(values, rewards, masks)
                timing["time/ppo/compute_advantages"] = time.time() - t

                del rewards
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            batch_dict = {
                "queries": queries,
                "responses": responses,
                "logprobs": all_logprobs.to(torch.float32),
                "values": values.to(torch.float32),
                "masks": masks,
                "advantages": advantages,
                "returns": returns,
            }

            del all_logprobs, values, advantages, returns
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if full_kl_penalty:
                batch_dict.update({
                    "ref_logits_or_none": ref_logits_or_none.to(torch.float32)
                })
                del ref_logits_or_none
            else:
                batch_dict.update({
                    "ref_logprobs": ref_logprobs.to(torch.float32),
                })
                del ref_logprobs

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            batch_dict.update(model_inputs)
            del model_inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            t = time.time()
            all_stats = []
            early_stop = False

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            loss_ps = []
            loss_vs = []

            for _ in range(self.config.ppo_epochs):
                if early_stop:
                    break

                b_inds = np.random.permutation(bs)
                for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                    backward_batch_end = backward_batch_start + self.config.backward_batch_size
                    backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                    for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                        mini_batch_end = mini_batch_start + self.config.mini_batch_size
                        mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]

                        mini_batch_dict = {
                            "logprobs": batch_dict["logprobs"][mini_batch_inds],
                            "values": batch_dict["values"][mini_batch_inds],
                            "masks": batch_dict["masks"][mini_batch_inds],
                            "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                            "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                            "advantages": batch_dict["advantages"][mini_batch_inds],
                            "returns": batch_dict["returns"][mini_batch_inds],
                        }

                        for k in model_inputs_names:
                            mini_batch_dict[k] = batch_dict[k][mini_batch_inds]

                        # Modified to handle DeepSpeed ZeRO stage 2 compatibility
                        if hasattr(self, 'is_deepspeed') and self.is_deepspeed:
                            # For DeepSpeed, don't use accumulate context manager
                            model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                            logprobs, logits, vpreds, _, _ = self.batched_forward_pass(
                                self.model,
                                mini_batch_dict["queries"],
                                mini_batch_dict["responses"],
                                model_inputs,
                                return_logits=True,
                            )
                            train_stats, loss_p, loss_v = self.train_minibatch(
                                mini_batch_dict["logprobs"],
                                mini_batch_dict["values"],
                                logprobs,
                                logits,
                                vpreds,
                                mini_batch_dict["masks"],
                                mini_batch_dict["advantages"],
                                mini_batch_dict["returns"],
                                mask_loss
                            )

                            # Clean up temporary variables
                            del logprobs, logits, vpreds, model_inputs
                        else:
                            # For non-DeepSpeed, use the original accumulate context manager
                            with self.accelerator.accumulate(self.model):
                                model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                                logprobs, logits, vpreds, _, _ = self.batched_forward_pass(
                                    self.model,
                                    mini_batch_dict["queries"],
                                    mini_batch_dict["responses"],
                                    model_inputs,
                                    return_logits=True,
                                )
                                train_stats, loss_p, loss_v = self.train_minibatch(
                                    mini_batch_dict["logprobs"],
                                    mini_batch_dict["values"],
                                    logprobs,
                                    logits,
                                    vpreds,
                                    mini_batch_dict["masks"],
                                    mini_batch_dict["advantages"],
                                    mini_batch_dict["returns"],
                                    mask_loss
                                )

                                # Clean up temporary variables
                                del logprobs, logits, vpreds, model_inputs

                        # Clean up mini_batch_dict to free memory
                        del mini_batch_dict
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        all_stats.append(train_stats)
                        loss_ps.append(loss_p)
                        loss_vs.append(loss_v)

                    # Clear cache after each backward batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # typically, early stopping is done at the epoch level
                if self.config.early_stopping:
                    policykl = train_stats["policy/policykl"]
                    early_stop = self._early_stop(policykl)
                    if early_stop:
                        break

            # Save a copy of advantages for the final calculation
            advantages_for_return = batch_dict["advantages"].clone()

            # Clean up batch_dict as it's no longer needed
            # Save references to needed variables before deleting batch_dict
            saved_logprobs = batch_dict.get("logprobs", None)
            saved_ref_logprobs = batch_dict.get("ref_logprobs", None)
            del batch_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            timing["time/ppo/optimize_step"] = time.time() - t

            t = time.time()
            train_stats = stack_dicts(all_stats)
            # Clean up all_stats
            del all_stats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            train_stats["skip_word_percentage"] = float(sum(total_skip_words_nums) / len(total_skip_words_nums))
            # reshape advantages/ratios such that they are not averaged.
            train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
            train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
            train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

            stats = self.record_step_stats(
                scores=scores,
                logprobs=saved_logprobs,
                ref_logprobs=saved_ref_logprobs,
                non_score_reward=non_score_reward,
                train_stats=train_stats,
                kl_coef=self.kl_ctl.value,
                masks=masks,
                queries=queries,
                responses=responses,
                kls=kls,
            )

            # Clean up variables no longer needed
            del saved_logprobs, saved_ref_logprobs, non_score_reward, train_stats, masks, kls
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Gather/Reduce stats from all processes
            if self.is_distributed:
                stats = self.gather_stats(stats)
            stats = stats_to_np(stats)
            timing["time/ppo/calc_stats"] = time.time() - t
            stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

            self.kl_ctl.update(
                stats["objective/kl"],
                self.config.batch_size * self.accelerator.num_processes,
            )

            # Log the total ppo time
            timing["time/ppo/total"] = time.time() - t0
            stats.update(timing)

            # post-process stats for tensorboard and other loggers
            if self.config.log_with != "wandb":
                stats = convert_to_scalar(stats)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Final cache clearing at the end of step
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Calculate average advantages for return
            avg_advantages = sum(sum(advantages_for_return)) / (advantages_for_return.shape[0] * advantages_for_return.shape[1])
            del advantages_for_return
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return stats, sum(loss_ps) / len(loss_ps), sum(loss_vs) / len(loss_vs), avg_advantages
        except Exception as e:
            # 确保所有进程都知道发生了错误
            if self.is_distributed:
                import torch.distributed as dist
                dist.barrier()
            print(f"Error in step: {e}")
            raise

    def _early_stop(self, policykl):
        r"""
        Handles the early stopping logic. If the policy KL is greater than the target KL, then the gradient is zeroed and
        the optimization step is skipped.
        This also handles the multi-gpu case where the policy KL is averaged across all processes.

        Args:
            policy_kl (torch.Tensor):
                the policy KL

        Returns:
            `bool`: whether to early stop or not
        """
        early_stop = False
        if not self.config.early_stopping:
            return early_stop

        if not self.is_distributed and policykl > 1.5 * self.config.target_kl:
            self.optimizer.zero_grad()
            early_stop = True
        elif self.is_distributed:
            import torch.distributed as dist

            # Wait for all processes to finish
            dist.barrier()

            # all gather the policykl
            dist.all_reduce(policykl, dist.ReduceOp.SUM)
            policykl /= self.accelerator.num_processes

            if policykl > 1.5 * self.config.target_kl:
                self.optimizer.zero_grad()
                early_stop = True
        return early_stop

    def gather_stats(self, stats):
        import torch.distributed as dist

        # 确保所有进程都到达这里
        dist.barrier()

        for k, v in stats.items():
            if isinstance(v, torch.Tensor):
                # 添加错误处理
                try:
                    dist.all_reduce(v.to(self.accelerator.device), dist.ReduceOp.SUM)
                    v /= self.accelerator.num_processes
                except Exception as e:
                    print(f"Error in gather_stats for key {k}: {e}")
                    raise
                stats[k] = v
        return stats

    def prepare_model_inputs(self, queries: torch.Tensor, responses: torch.Tensor):
        if self.is_encoder_decoder:
            input_data = self.data_collator(
                [{"input_ids": q, "attention_mask": torch.ones_like(q)} for q in queries]
            ).to(self.current_device)

            decoder_inputs = self.data_collator(
                [{"input_ids": r, "attention_mask": torch.ones_like(r)} for r in responses]
            ).to(self.current_device)

            input_data["decoder_input_ids"] = decoder_inputs["input_ids"]
            input_data["decoder_attention_mask"] = decoder_inputs["attention_mask"]
        else:
            input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]  # 将question与answer部分拼接在一起
            input_data = self.data_collator(
                [{"input_ids": ids, "attention_mask": torch.ones_like(ids)} for ids in input_ids]
            ).to(self.current_device)

        input_data.pop("labels", None)  # we don't want to compute LM losses
        return input_data

    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
            self,
            model: PreTrainedModelWrapper,
            queries: torch.Tensor,
            responses: torch.Tensor,
            model_inputs: dict,
            return_logits: bool = False,
            response_masks: Optional[torch.Tensor] = None,
    ):
        """
        Calculate model outputs in multiple batches.
        """
        # Clear cache at the beginning
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 此处的mask指示出有效的tokens的位置，用于后面求取lengthsampler
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []
        all_tokens = []

        model.eval()

        # 添加检查，确保bs > 0
        if bs == 0:
            raise ValueError("Empty batch received in batched_forward_pass")

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs: (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs: (i + 1) * fbs]
            response_batch = responses[i * fbs: (i + 1) * fbs]

            # 检查当前批次是否为空
            if len(query_batch) == 0 or len(response_batch) == 0:
                print(f"Warning: Empty batch at index {i}, skipping")
                continue

            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs: (i + 1) * fbs]

            # 运行模型前检查输入
            for k, v in input_kwargs.items():
                if isinstance(v, torch.Tensor) and v.numel() == 0:
                    print(f"Warning: Empty tensor for {k} at batch {i}")

            logits, _, values = model(**input_kwargs)

            # 检查logits是否为空
            if logits.numel() == 0:
                print(f"Warning: Empty logits at batch {i}, skipping")
                continue

            # 取出logits对应的token
            token_ids = input_kwargs["input_ids"]
            token_ids_list = token_ids.tolist()
            # 遍历每个样本的token ID序列
            for sample_token_ids in token_ids_list:
                import re
                # 将每个token ID转换为对应的单词或子词
                tokens = [re.sub(r"[''‛']", "'", self.tokenizer.decode(token_id).strip()) for token_id in
                          sample_token_ids]
                all_tokens.append(tokens)  # 得到q+a对应的tokens序列

            if self.is_encoder_decoder:  # 不进入
                input_ids = input_kwargs["decoder_input_ids"]
                attention_mask = input_kwargs["decoder_attention_mask"]
            else:  # 进入
                input_ids = input_kwargs["input_ids"]
                attention_mask = input_kwargs["attention_mask"]

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])  # 计算每个tokens的对数概率
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            if logprobs.numel() == 0:
                print(f"Warning: Empty logprobs at batch {i}, skipping")
                continue

            for k in range(len(masks)):
                mask = masks[k]
                tokens = all_tokens[k]
                comma_indices = [i for i, char in enumerate(tokens) if (char in [',', '.', '?', '!']) and mask[i] == 1]
                # Set these indices to 0 in mask
                for idx in comma_indices:
                    mask[idx] = torch.tensor(0., device=mask.device)

                masks[k] = mask

            for j in range(len(query_batch)):
                if self.is_encoder_decoder:
                    # Decoder sentence starts always in the index 1 after padding in the Enc-Dec Models
                    start = 1
                    end = attention_mask[j, :].sum() - 1
                else:
                    start = len(query_batch[j]) - 1  # logprobs starts from the second query token
                    if attention_mask[j, 0] == 0:  # offset left padding
                        start += attention_mask[j, :].nonzero()[0]
                    end = start + len(response_batch[j])
                    if response_masks is not None:
                        response_masks_batch[j] = torch.cat(
                            (torch.zeros_like(query_batch[j]), response_masks_batch[j])
                        )[1:]
                    tokens = all_tokens[j]

                    for i in range(start, end):
                        if tokens[i] == '<|eot_id|>':
                            # 找到EOS后，将end调整到EOS后一个位置
                            end = i + 1 if i < len(tokens) - 1 else i
                            break
                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

            del input_kwargs, query_batch, response_batch
            if response_masks is not None:
                del response_masks_batch
            del token_ids, token_ids_list
            # Clear cache after each batch
            if torch.cuda.is_available() and i % 2 == 1:  
                torch.cuda.empty_cache()

        # 最终检查，确保列表不为空
        if not all_logprobs:
            raise ValueError("No valid batches processed, all_logprobs is empty")

        # Final cache clearing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
            all_tokens
        )

    @PPODecorators.empty_device_cache()
    def train_minibatch(
            self,
            old_logprobs: torch.FloatTensor,
            values: torch.FloatTensor,
            logprobs: torch.FloatTensor,
            logits: torch.FloatTensor,
            vpreds: torch.FloatTensor,
            mask: torch.LongTensor,
            advantages: torch.FloatTensor,
            returns: torch.FloatTensor,
            mask_loss: str = ""
    ):
        """
        Train one PPO minibatch
        """
        # Clear cache at the beginning
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model.train()
        loss_p, loss_v, train_stats = self.loss(
            old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns
        )
        del old_logprobs, values, logprobs, logits, vpreds

        # Explicit memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if mask_loss:
            loss = loss_v if mask_loss == "loss_p" else loss_p
        else:
            loss = loss_p + loss_v
        self.accelerator.backward(loss)
        if self.config.max_grad_norm is not None:
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model_params, self.config.max_grad_norm)
        self.optimizer.step()
        # we call optimizer.zero_grad() every time and let `accelerator` handle accumulation
        # see https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation#the-finished-code
        self.optimizer.zero_grad()

        # Final cache clearing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return train_stats, loss_p.item(), loss_v.item()

    def merge_words(self, word_list):
        new_word_list = []
        new_word_indices = []

        for i, word in enumerate(word_list):
            if 'ġ' not in word:
                if i > 0:
                    new_word_list[-1] += word
                    new_word_indices[-1].append(i)
                else:
                    new_word_list.append(word)
                    new_word_indices.append([i])
            else:
                new_word_list.append(word)
                new_word_indices.append([i])

        for i in range(len(new_word_list)):
            new_word_list[i] = new_word_list[i].replace('ġ', '')
            new_word_list[i] = new_word_list[i].replace(',', '')

        return new_word_list, new_word_indices

    def get_word_score_by_indices(self, scores, word_indices):
        word_scores = []
        for indice_list in word_indices:
            word_score = max([scores[i] for i in indice_list])
            word_scores.append(word_score)
        return word_scores


    def assign_token_rewards(self, token_list, word_list, score_list):
        def is_special_char(token):
            """检查是否为特殊字符"""
            special_chars = ['<|eot_id|>', '<|start_header_id|>',
                             '<|end_header_id|>']
            return (token in special_chars or token.startswith('<') or
                    not any(c.isalnum() for c in token if c not in ['', ' ']))

        print("\n=== Starting new matching process ===")
        print(f"Token list: {token_list}")

        token_index = 0
        word_index = 0
        last_match_token_ind = 0
        max_attempts = 6
        token_score = []
        total_skip_words_num = 0

        # Filter out empty words
        filtered_word_list = []
        filtered_score_list = []
        for i, word in enumerate(word_list):
            if word.strip():  # Only keep non-empty words
                filtered_word_list.append(word)
                filtered_score_list.append(score_list[i])

        word_list = filtered_word_list
        score_list = filtered_score_list
        while token_index <= len(token_list) - 1:
            if word_index >= len(word_list):
                print(f"Reached end of word list. Filling remaining tokens with 0")
                token_score.extend([torch.tensor(0., device=score_list[0].device)] *
                                   (len(token_list) - len(token_score)))
                break

            flag_matched = False
            i_matched = 0
            wid_matched = -1
            attempts = 0
            initial_token_index = token_index

            #print(f"\nTrying to match word: {word_list[word_index]}")
            #print(f"Starting from token index: {token_index}")

            while attempts < max_attempts:
                for i in range(1, min(10, len(token_list) - token_index)):
                    # Include all tokens in concatenation, including special characters
                    actual_tokens = token_list[token_index:token_index + i]
                    concatenated_tokens = ''.join(actual_tokens)
                    current_word = word_list[word_index].replace("'","'").replace("'","'")  # Fixed the invalid comma character

                    if len(concatenated_tokens) > len(current_word) + 2:
                        continue

                    #print(f"Attempt {attempts + 1}: {actual_tokens} ({concatenated_tokens}) -> {current_word}")

                    if concatenated_tokens.lower() == current_word.lower():
                        flag_matched = True
                        i_matched = i
                        wid_matched = word_index
                        #print(f"Match found! Length: {i}")
                        break

                if flag_matched:
                    break
                else:
                    attempts += 1
                    token_index += 1
                    if token_index == len(token_list):
                        #print("Reached end of token list")
                        break

                if attempts >= max_attempts:
                    total_skip_words_num += 1
                    print(f"Failed to match word: {word_list[word_index]} after {max_attempts} attempts")
                    word_index += 1
                    token_index = initial_token_index
                    break

            if flag_matched:
                # Fill unmatched tokens with 0
                zeros_count = token_index - last_match_token_ind
                if zeros_count > 0:
                    #print(f"Filling {zeros_count} unmatched tokens with 0")
                    token_score.extend([torch.tensor(0., device=score_list[wid_matched].device)] * zeros_count)

                # Assign scores - special characters get 0, other tokens get the word score
                #print(f"Assigning scores for matched tokens:")
                for j in range(i_matched):
                    current_token = token_list[token_index + j]
                    if is_special_char(current_token):
                        #print(f"  {current_token}: 0 (special char)")
                        token_score.append(torch.tensor(0., device=score_list[wid_matched].device))
                    else:
                        #print(f"  {current_token}: {float(score_list[wid_matched])} (matched)")
                        token_score.append(score_list[wid_matched])

                last_match_token_ind = token_index + i_matched
                token_index += i_matched
                word_index += 1

        # Pad remaining tokens with 0
        if len(token_score) != len(token_list):
            padding_count = len(token_list) - len(token_score)
            #print(f"\nPadding final {padding_count} tokens with 0")
            token_score.extend([torch.tensor(0., device=score_list[0].device)] * padding_count)

        print("\n=== Final Results ===")
        #print(f"Total skipped words: {total_skip_words_num}")
        #print(f"Final score length: {len(token_score)}")
        #print(f"Scores: {[float(s) for s in token_score]}")

        return token_score, total_skip_words_num


    def rematch_scores(self, scores, words, all_tokens):
        # 该函数利用从reward model处获得的word-level的得分变成token-level的score
        # 使用嵌套的列表推导式将每个字符串转换为小写

        print(f'收到的score：{scores}')

        for i in range(len(words)):
            words[i] = [word.lower() for word in words[i]]

        # process all_tokens
        # 使用嵌套的列表推导式将每个字符串转换为小写,并覆写原来的all_tokens
        for i in range(len(all_tokens)):
            all_tokens[i] = [token.lower() for token in all_tokens[i]]

        answer_indices_tokens = []
        for sublist in all_tokens:
            found_index = -1
            for i in range(len(sublist) - 2):
                if (sublist[i] == '<|start_header_id|>' and
                        sublist[i + 1] == 'assistant' and
                        sublist[i + 2] == '<|end_header_id|>'):
                    # 找到header后的第一个非空token位置
                    j = i + 3
                    while j < len(sublist) and sublist[j] == '':
                        j += 1
                    found_index = j
                    break
            answer_indices_tokens.append(found_index)

        answers_tokens = []
        answers_words = words
        answers_scores = scores

        for i in range(len(words)):
            if answer_indices_tokens[i] != -1:
                answers_tokens.append(all_tokens[i][answer_indices_tokens[i]:])
            else:
                answers_tokens.append([])

        token_scores = []
        total_skip_words_nums = []

        for i, token_list in enumerate(answers_tokens):
            word_list = answers_words[i]
            score_list = answers_scores[i]

            # Fix: Check if score_list is empty or if it's a 0-d tensor
            if isinstance(score_list, torch.Tensor) and score_list.dim() == 0:
                # Handle 0-dimensional tensor case
                default_device = score_list.device
                token_non_score = [torch.tensor(0., device=default_device)] * max(0, answer_indices_tokens[i])
                token_score = [torch.tensor(0., device=default_device)] * len(token_list)
                total_skip_words_num = 0
            elif len(score_list) == 0 or answer_indices_tokens[i] == -1:
                default_device = score_list[0].device if len(score_list) > 0 else torch.device('cpu')
                token_non_score = [torch.tensor(0., device=default_device)] * max(0, answer_indices_tokens[i])
                token_score = [torch.tensor(0., device=default_device)] * len(token_list)
                total_skip_words_num = 0
            else:
                # Normal case - create zero tensors with the same device as the first score
                token_non_score = [torch.tensor(0., device=score_list[0].device)] * answer_indices_tokens[i]
                token_score, total_skip_words_num = self.assign_token_rewards(token_list, word_list, score_list)

            total_skip_words_nums.append(float(total_skip_words_num / max(1, len(word_list))))  # Avoid division by zero

            token_scores.append(token_non_score + token_score)

        return token_scores, answers_tokens, answer_indices_tokens, total_skip_words_nums

    def compute_rewards(
            self,
            scores: torch.FloatTensor,
            logprobs: torch.FloatTensor,
            ref_logprobs: torch.FloatTensor,
            masks: torch.LongTensor,
            all_tokens: List[List[str]],
            words: List[List[str]] = None
    ):
        """
        Compute per token rewards from scores and KL-penalty.

        Args:
            scores (`torch.FloatTensor`):
                Scores from the reward model, shape (`batch_size`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            ref_logprobs (`torch.FloatTensor`):
                Log probabilities of the reference model, shape (`batch_size`, `response_length`)

        Returns:
            `torch.FloatTensor`: Per token rewards, shape (`batch_size`, `response_length`)
            `torch.FloatTensor`: Non score rewards, shape (`batch_size`, `response_length`)
            `torch.FloatTensor`: KL penalty, shape (`batch_size`, `response_length`)
        """
        rewards, non_score_rewards, kls = [], [], []

        for i in range(len(scores)):
            scores[i] = scores[i][1:]

        for score, logprob, ref_logprob, mask in zip(scores, logprobs, ref_logprobs, masks):
            assert len(logprob) == len(score)
            # compute KL penalty (from difference in logprobs)
            kl = self._kl_penalty(logprob, ref_logprob)  # kl=0
            kls.append(kl)
            non_score_reward = -self.kl_ctl.value * kl  # 0.2*0=0
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            # last_non_masked_index = mask.nonzero()[-1]
            non_masked_indices = mask.nonzero()

            for index in non_masked_indices:
                reward[index] += score[index]

            # reward is preference model score + KL penalty
            # reward[last_non_masked_index] += score
            rewards.append(reward)
        return torch.stack(rewards), torch.stack(non_score_rewards), torch.stack(kls)

    def _kl_penalty(self, logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor) -> torch.FloatTensor:
        if self.config.kl_penalty == "kl":
            return logprob - ref_logprob

        if self.config.kl_penalty == "abs":
            return (logprob - ref_logprob).abs()

        if self.config.kl_penalty == "mse":
            return 0.5 * (logprob - ref_logprob).square()

        if self.config.kl_penalty == "full":
            return F.kl_div(ref_logprob, logprob, log_target=True, reduction="none").sum(-1)

        raise NotImplementedError

    def compute_advantages(
            self,
            values: torch.FloatTensor,
            rewards: torch.FloatTensor,
            mask: torch.FloatTensor,
    ):
        """
        Compute advantages and returns.

        Args:
            values: Value function predictions
            rewards: Rewards
            mask: Mask

        Returns:
            values: Value function predictions
            advantages: Advantages
            returns: Returns
        """
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[-1]

        values = values * mask
        rewards = rewards * mask

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
            lastgaelam = lastgaelam * mask[:, t]
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
        returns = advantages + values

        valid_advantages = advantages[mask.bool()]
        advantages = masked_whiten(advantages, mask)
        valid_advantages = advantages[mask.bool()]
        advantages = advantages.detach()

        return values, advantages, returns

    def loss(
            self,
            old_logprobs: torch.FloatTensor,
            values: torch.FloatTensor,
            logits: torch.FloatTensor,
            vpreds: torch.FloatTensor,
            logprobs: torch.FloatTensor,
            mask: torch.LongTensor,
            advantages: torch.FloatTensor,
            returns: torch.FloatTensor,
    ):
        """
        Calculate policy and value losses.

        Args:
            old_logprobs: Log probabilities of the model, shape (batch_size, response_length)
            values: Values of the value head, shape (batch_size, response_length)
            logits: Logits of the model, shape (batch_size, response_length, vocab_size)
            vpreds: Values of the value head, shape (batch_size, response_length)
            logprobs: Log probabilities of the model, shape (batch_size, response_length)
            mask: Mask tensor, shape (batch_size, response_length)
            advantages: Advantages tensor, shape (batch_size, response_length)
            returns: Returns tensor, shape (batch_size, response_length)
        """
        vpredclipped = clip_by_value(
            vpreds,
            values - self.config.cliprange_value,
            values + self.config.cliprange_value,
        )

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), mask)

        logprob_diff = logprobs - old_logprobs
        logprob_diff = torch.clamp(logprob_diff, -20, 20)
        ratio = torch.exp(logprob_diff)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), mask)

        loss = pg_loss + self.config.vf_coef * vf_loss

        avg_ratio = masked_mean(ratio, mask).item()
        if avg_ratio > self.config.ratio_threshold:
            warnings.warn(
                f"The average ratio of batch ({avg_ratio:.2f}) exceeds threshold {self.config.ratio_threshold:.2f}. Skipping batch."
            )
            pg_loss = pg_loss * 0.0
            vf_loss = vf_loss * 0.0
            loss = loss * 0.0

        entropy = masked_mean(entropy_from_logits(logits), mask)
        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)

        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)
        value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)

        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach()),
            policy=dict(
                entropy=entropy.detach(),
                approxkl=approxkl.detach(),
                policykl=policykl.detach(),
                clipfrac=pg_clipfrac.detach(),
                advantages=advantages.detach(),
                advantages_mean=masked_mean(advantages, mask).detach(),
                ratio=ratio.detach(),
            ),
            returns=dict(mean=return_mean.detach(), var=return_var.detach()),
            val=dict(
                vpred=masked_mean(vpreds, mask).detach(),
                error=masked_mean((vpreds - returns) ** 2, mask).detach(),
                clipfrac=vf_clipfrac.detach(),
                mean=value_mean.detach(),
                var=value_var.detach(),
            ),
        )
        return pg_loss, self.config.vf_coef * vf_loss, flatten_dict(stats)

    def record_step_stats(self, kl_coef: float, **data):
        """
        Record training step statistics.
        """
        mask = data.pop("masks")


        stats = {}


        kls = data.pop("kls")
        kl_list = ((kls) * mask).sum(axis=-1)
        mean_kl = kl_list.mean()
        mean_entropy = (-data["logprobs"] * mask).sum(axis=-1).mean()

        mean_non_score_reward = masked_mean(
            data["non_score_reward"], mask
        )

        masked_scores = []
        for i, scores in enumerate(data["scores"]):
            mask_i = mask[i]
            mask_score = [score for j, score in enumerate(scores) if j < len(mask_i) and mask_i[j] == 1]

            mask_score = torch.tensor(mask_score, dtype=torch.float32)
            

            masked_scores.append(mask_score)

        batch_mean_scores = [scores.mean() for scores in masked_scores]
        mean_scores = torch.stack(batch_mean_scores).mean()
        batch_std_scores = [scores.std() for scores in masked_scores]
        std_scores = torch.stack(batch_std_scores).mean()

        if mean_kl.item() < -1.0:
            warnings.warn(
                f"KL divergence is starting to become negative: {mean_kl.item():.2f} - this might be a precursor for failed training."
                " sometimes this happens because the generation kwargs are not correctly set. Please make sure"
                " that the generation kwargs are set correctly, or review your training hyperparameters."
            )

        stats.update({
            "objective/kl": mean_kl,
            "objective/kl_dist": kl_list,
            "objective/logprobs": data["logprobs"],
            "objective/ref_logprobs": data["ref_logprobs"],
            "objective/kl_coef": kl_coef,
            "objective/entropy": mean_entropy,
            "ppo/mean_non_score_reward": mean_non_score_reward,
            "ppo/mean_scores": mean_scores,
            "ppo/std_scores": std_scores,
        })

        return stats

    def log_stats(
            self,
            wandb_step: int,
            stats: dict,
            batch: dict,
            rewards: List[torch.FloatTensor],
            columns_to_log: typing.Iterable[str] = ("question", "response"),
    ):
        """
        A function that logs all the training stats. Call it at the end of each epoch.

        Args:
            stats (dict[str, Any]):
                A dictionary of training stats.
            batch (dict[str, Any]):
                A dictionary of batch data, this contains the queries and responses.
            rewards (`List[torch.FloatTensor]`):
                A tensor of rewards.
        """

        if not isinstance(rewards, torch.Tensor):
            rewards = [reward.float().mean() for reward in rewards]  # TODO: add mask for rewards
            rewards = torch.tensor(rewards).to(self.current_device)
        rewards = self.accelerator.gather(rewards).flatten()

        if self.config.log_with == "wandb":
            import wandb

            if any(column_to_log not in batch.keys() for column_to_log in columns_to_log):
                raise ValueError(f"Columns to log {columns_to_log} are not present in the batch {batch.keys()}.")

            batch_list = [batch[column_to_log] for column_to_log in columns_to_log]
            if self.is_distributed:
                gathered_batch_list = []
                for b in batch_list:
                    flattened = gather_object(b)
                    gathered_batch_list.append(flattened)
                batch_list = gathered_batch_list

        if self.accelerator.is_main_process:
            logs = {}

            if "query" not in batch.keys() and "response" not in batch.keys():
                warnings.warn(
                    "The game logs will not be logged because the batch does not contain the keys 'query' and "
                    "'response'. "
                )
            elif self.config.log_with == "wandb":
                table_rows = [list(r) for r in zip(*batch_list, rewards.cpu().tolist())]
                logs.update({"game_log": wandb.Table(columns=[*columns_to_log, "reward"], rows=table_rows)})

            logs.update(stats)

            for k, v in logs.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                    logs[k] = v.float()

            logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy().item()
            logs["env/reward_std"] = torch.std(rewards).cpu().numpy().item()
            logs["env/reward_dist"] = rewards.cpu().numpy()

            if self.config.log_with == "tensorboard" or self.config.log_with == "wandb":
                self.current_step += 1

            self.accelerator.log(
                logs,
                step=wandb_step,
            )

    def create_model_card(self, path: str, model_name: Optional[str] = "TRL Model") -> None:
        """Creates and saves a model card for a TRL model.

        Args:
            path (`str`): The path to save the model card to.
            model_name (`str`, *optional*): The name of the model, defaults to `TRL Model`.
        """
        try:
            user = whoami()["name"]
        except Exception:
            warnings.warn("Cannot retrieve user information assuming you are running in offline mode.")
            return

        if not os.path.exists(path):
            os.makedirs(path)

        model_card_content = MODEL_CARD_TEMPLATE.format(model_name=model_name, model_id=f"{user}/{path}")
        with open(os.path.join(path, "README.md"), "w", encoding="utf-8") as f:
            f.write(model_card_content)

    def _save_pretrained(self, save_directory: str) -> None:
        self.accelerator.unwrap_model(self.model).save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        self.create_model_card(save_directory)

    def _show_tokens(self, tokens, masks):
        from rich import print
        from rich.text import Text

        text = Text()

        for _i, (token, mask) in enumerate(zip(tokens, masks)):
            if mask == 1:
                text.append(self.tokenizer.decode(token.item()), style="black on deep_sky_blue1")
                text.append(" ")
            else:
                text.append(self.tokenizer.decode(token.item()), style="black on cyan3")
                text.append(" ")
        print(text)

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepspeed_plugin.deepspeed_config
        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model