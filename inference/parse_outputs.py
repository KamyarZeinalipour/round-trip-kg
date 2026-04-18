"""
Output Parser
==============

Extracts the model-generated content from raw inference outputs
by removing chat template markers and special tokens.

Handles both Llama-2 (`[INST]...[/INST]`) and Llama-3.x
(`<|end_header_id|>...<|eot_id|>`) output formats.

Usage:
    python parse_outputs.py --input_dir inference_results --output_dir inference_results/parsed
"""

import argparse
import glob
import os
import re

import pandas as pd


def clean_llama3_output(output: str, prompt: str = "") -> str:
    """Extract assistant response from Llama-3.x format."""
    match = re.search(
        r"assistant<\|end_header_id\|>(.*?)<\|eot_id\|>", output, flags=re.DOTALL
    )
    return match.group(1).strip() if match else ""


def extract_llama2_inst(text: str) -> str:
    """Extract content between [/INST] markers (Llama-2 format)."""
    sections = re.findall(r"\[/INST\](.*?)\[/INST\]", text, re.DOTALL)
    if sections:
        return sections[0].strip()
    fallback = re.search(r"\[/INST\](.*?)</s>", text, re.DOTALL)
    return fallback.group(1).strip() if fallback else ""


def extract_llama2_base(text: str) -> str:
    """Extract content between [/INST] and </s> (Llama-2 base)."""
    match = re.search(r"\[/INST\](.*?)</s>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_file(csv_path: str, output_dir: str) -> None:
    """Parse a single inference output CSV."""
    filename = os.path.basename(csv_path)
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  Skipping {csv_path}: {e}")
        return

    if "output" not in df.columns:
        print(f"  Skipping {csv_path}: no 'output' column.")
        return

    # Select parser based on model type in filename
    if "Llama-2-7b-chat-hf_base" in filename:
        df["output_parsed"] = df["output"].apply(
            lambda t: extract_llama2_base(t) if pd.notnull(t) else ""
        )
    elif "base" in filename or "meta-llama" in filename:
        if "prompt" in df.columns:
            df["output_parsed"] = df.apply(
                lambda r: clean_llama3_output(r["output"], r.get("prompt", ""))
                if pd.notnull(r["output"])
                else "",
                axis=1,
            )
        else:
            print(f"  Skipping {csv_path}: needs 'prompt' column.")
            return
    else:
        df["output_parsed"] = df["output"].apply(
            lambda t: extract_llama2_inst(t) if pd.notnull(t) else ""
        )

    out_path = os.path.join(output_dir, filename)
    df.to_csv(out_path, index=False)
    print(f"  Parsed → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse raw model outputs.")
    parser.add_argument(
        "--input_dir", default="inference_results", help="Directory with raw CSVs."
    )
    parser.add_argument(
        "--output_dir",
        default="inference_results/parsed",
        help="Directory for parsed CSVs.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files to parse.\n")

    for csv_file in csv_files:
        parse_file(csv_file, args.output_dir)

    print("\nAll files parsed.")
