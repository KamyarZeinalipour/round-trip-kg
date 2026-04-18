"""
Inference with Fine-Tuned LoRA Models
======================================

Runs inference on the held-out evaluation set using models fine-tuned
with LoRA adapters. Iterates over all model × fraction combinations.

Usage:
    python inference_finetuned.py
"""

import ast
import gc
import os
import time

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
#  Chat template for Llama-2
# ---------------------------------------------------------------------------
DEFAULT_LLAMA2_CHAT_TEMPLATE = (
    "{% for message in messages %}\n"
    "{% if message['role'] == 'system' %}\n"
    "{{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}\n"
    "{% elif message['role'] == 'user' %}\n"
    "{{ '[INST]' + message['content'] + '[/INST]\\n' }}\n"
    "{% elif message['role'] == 'assistant' %}\n"
    "{{ message['content'] + '[/INST]\\n' }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n"
    "{% endif %}\n"
    "{% endfor %}"
)

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
MODELS = [
    "NousResearch/Llama-2-7b-chat-hf",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

FRACTIONS = [0.25, 0.5, 0.75, 1]
INFERENCE_DIR = "inference_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generation hyperparameters (Appendix B.3)
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.9
REPETITION_PENALTY = 1.1

os.makedirs(INFERENCE_DIR, exist_ok=True)


def extract_conversation(message_str: str, tokenizer) -> str:
    """Parse a stringified message list and apply the chat template."""
    try:
        message_list = ast.literal_eval(message_str)
    except Exception as e:
        print(f"Parsing error: {e}")
        return ""
    if message_list:
        return tokenizer.apply_chat_template(
            message_list, tokenize=False, add_generation_prompt=True
        )
    return ""


def load_peft_model(base_name: str, adapter_dir: str):
    """Load a base model and merge a LoRA adapter."""
    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(
        model, adapter_dir, device_map="auto"
    ).merge_and_unload()
    peft_model.eval()
    return peft_model


def generate(model, tokenizer, prompt: str) -> str:
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    out_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        do_sample=True,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(out_ids, skip_special_tokens=False)[0]


def unload_model(model):
    """Free GPU memory."""
    try:
        model.cpu()
    except Exception:
        pass
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for model_name in MODELS:
        print(f"\n{'='*60}\n  Model: {model_name}\n{'='*60}")
        df = pd.read_csv("kg_triple_gen_test_formated.csv")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if "llama-2" in model_name.lower():
            tokenizer.chat_template = DEFAULT_LLAMA2_CHAT_TEMPLATE

        df["prompt"] = [
            extract_conversation(m, tokenizer) for m in df["messages"]
        ]

        for frac in FRACTIONS:
            sanitized = model_name.replace("/", "_")
            adapter_dir = f"outputs/{sanitized}_frac{frac}"

            print(f"\n--- {model_name} @ {int(frac*100)}% fraction ---")
            model = load_peft_model(model_name, adapter_dir)
            model.to(DEVICE)

            results = []
            for idx, row in df.iterrows():
                print(f"  [{idx+1}/{len(df)}]", end="\r")
                try:
                    out = generate(model, tokenizer, row["prompt"])
                except Exception as e:
                    print(f"\n  Error at row {idx}: {e}. Retrying...")
                    time.sleep(30)
                    out = generate(model, tokenizer, row["prompt"])
                results.append(out)

            out_df = df.copy()
            out_df["output"] = results
            out_path = os.path.join(INFERENCE_DIR, f"{sanitized}_frac{frac:.2f}.csv")
            out_df.to_csv(out_path, index=False)
            print(f"  Saved → {out_path}")

            unload_model(model)

    print("\nAll inference complete.")
