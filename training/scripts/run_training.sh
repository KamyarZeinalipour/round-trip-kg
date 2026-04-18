#!/usr/bin/env bash
# ============================================================================
#  Self-Supervised Fine-Tuning with LoRA + DeepSpeed ZeRO-2
# ============================================================================
#
#  Trains LLaMA models on self-generated high-fidelity synthetic data
#  produced by the round-trip pipeline.
#
#  Models:   LLaMA-2-7B, LLaMA-3.2-3B, LLaMA-3.2-1B
#  Datasets: Hosted on HuggingFace Hub (Kamyar-zeinalipour/*)
#
#  Usage:
#    chmod +x run_training.sh && ./run_training.sh
#
# ============================================================================
set -euo pipefail

export CUDA_VISIBLE_DEVICES="0"
LAUNCH_CMD="accelerate launch --config_file ../configs/deepspeed_config.yaml ../train.py"

# ---------- Common hyperparameters (Table in Appendix B) ----------
COMMON_ARGS=(
  --seed 100
  --chat_template_format "none"
  --add_special_tokens False
  --append_concat_token False
  --splits "train,test"
  --max_seq_len 512
  --num_train_epochs 3
  --logging_steps 5
  --log_level "info"
  --logging_strategy "steps"
  --evaluation_strategy "epoch"
  --save_strategy "epoch"
  --push_to_hub False
  --bf16 True
  --packing True
  --learning_rate 1e-4
  --lr_scheduler_type "cosine"
  --weight_decay 1e-4
  --warmup_ratio 0.0
  --max_grad_norm 1.0
  --per_device_train_batch_size 4
  --per_device_eval_batch_size 4
  --gradient_accumulation_steps 2
  --gradient_checkpointing True
  --use_reentrant False
  --dataset_text_field "content"
  --use_flash_attn True
  --use_peft_lora True
  --lora_r 16
  --lora_alpha 32
  --lora_dropout 0.1
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj,embed_tokens,lm_head"
)

# ---------- Models & datasets ----------
MODELS=(
  "NousResearch/Llama-2-7b-chat-hf"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
)

declare -A DATASETS=(
  ["NousResearch/Llama-2-7b-chat-hf"]="Kamyar-zeinalipour/llama2_kg"
  ["meta-llama/Llama-3.2-1B-Instruct"]="Kamyar-zeinalipour/llama1b_kg"
  ["meta-llama/Llama-3.2-3B-Instruct"]="Kamyar-zeinalipour/llama3b_kg"
)

# ---------- Data fractions for efficiency ablation (Fig. 5) ----------
FRACTIONS=(1 0.25 0.5 0.75)

for MODEL in "${MODELS[@]}"; do
  DATASET="${DATASETS[$MODEL]}"
  SANITIZED_MODEL=${MODEL//\//_}

  for FRAC in "${FRACTIONS[@]}"; do
    OUTDIR="outputs/${SANITIZED_MODEL}_frac${FRAC}"
    mkdir -p "$OUTDIR"

    echo "==> Training $MODEL on $DATASET with ${FRAC} of data → $OUTDIR"
    $LAUNCH_CMD \
      --model_name_or_path "$MODEL" \
      --dataset_name     "$DATASET" \
      --train_fraction   "$FRAC" \
      --output_dir       "$OUTDIR" \
      "${COMMON_ARGS[@]}"
  done
done
