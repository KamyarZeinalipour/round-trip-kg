# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
Self-Supervised Fine-Tuning with SFTTrainer + LoRA + DeepSpeed
===============================================================

Fine-tunes LLMs on self-generated high-fidelity synthetic data
produced by the round-trip pipeline. Uses PEFT LoRA for parameter-
efficient training and DeepSpeed ZeRO-2 for memory efficiency.

Usage (with Accelerate + DeepSpeed):
    accelerate launch --config_file configs/deepspeed_config.yaml \\
        train.py --model_name_or_path <MODEL> --dataset_name <DATASET> ...

See ``training/scripts/run_training.sh`` for complete examples.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments, set_seed
from trl import SFTTrainer

from utils import create_and_prepare_model, create_datasets


# ---------------------------------------------------------------------------
#  Argument dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ModelArguments:
    """Arguments for model configuration and PEFT/LoRA setup."""

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or HuggingFace model identifier."}
    )
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={
            "help": "chatml | zephyr | none. Use 'none' if the dataset is "
            "already formatted with the chat template."
        },
    )
    lora_alpha: Optional[int] = field(default=32)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=16)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj,embed_tokens,lm_head",
        metadata={"help": "Comma-separated list of LoRA target modules."},
    )
    use_nested_quant: Optional[bool] = field(default=False)
    bnb_4bit_compute_dtype: Optional[str] = field(default="float16")
    bnb_4bit_quant_type: Optional[str] = field(default="nf4")
    use_flash_attn: Optional[bool] = field(
        default=False, metadata={"help": "Enable FlashAttention-2 for training."}
    )
    use_peft_lora: Optional[bool] = field(
        default=False, metadata={"help": "Enable PEFT LoRA for training."}
    )
    use_8bit_qunatization: Optional[bool] = field(default=False)
    use_4bit_qunatization: Optional[bool] = field(default=False)
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient checkpointing reentrant mode."},
    )


@dataclass
class DataTrainingArguments:
    """Arguments for dataset configuration."""

    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "HuggingFace dataset name or local path."},
    )
    packing: Optional[bool] = field(default=False)
    dataset_text_field: str = field(default="text")
    max_seq_length: Optional[int] = field(default=512)
    append_concat_token: Optional[bool] = field(default=False)
    add_special_tokens: Optional[bool] = field(default=False)
    splits: Optional[str] = field(
        default="train,test",
        metadata={"help": "Comma-separated list of dataset splits."},
    )
    train_fraction: float = field(
        default=1.0,
        metadata={"help": "Fraction of training data to use (0.0–1.0)."},
    )


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main(model_args, data_args, training_args):
    set_seed(training_args.seed)

    # Prepare model and tokenizer
    model, peft_config, tokenizer = create_and_prepare_model(model_args)

    # Gradient checkpointing
    model.config.use_cache = training_args.gradient_checkpointing
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": model_args.use_reentrant
        }

    # Load datasets
    train_dataset, eval_dataset = create_datasets(
        tokenizer, data_args, training_args, apply_chat_template=True
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        packing=data_args.packing,
        dataset_kwargs={
            "append_concat_token": data_args.append_concat_token,
            "add_special_tokens": data_args.add_special_tokens,
        },
        dataset_text_field=data_args.dataset_text_field,
        max_seq_length=data_args.max_seq_length,
    )
    trainer.accelerator.print(f"{trainer.model}")

    if model_args.use_peft_lora:
        trainer.model.print_trainable_parameters()

    # Train
    trainer.train()

    # Save model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)
