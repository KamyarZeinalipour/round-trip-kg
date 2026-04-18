"""
Automatic Evaluation Metrics
==============================

Computes BERTScore (XLM-RoBERTa-base), ROUGE-L, and BLEU-4
between extracted KG triples and gold-standard references.

Usage:
    python evaluate.py --input_dir inference_results/parsed --output_dir eval_results
"""

import argparse
import os

import pandas as pd
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer


def evaluate_file(input_path: str, output_dir: str) -> dict:
    """Evaluate a single parsed inference CSV.

    Computes per-instance and mean scores for:
      - BERTScore (P, R, F1) with ``xlm-roberta-base``
      - BLEU-1 through BLEU-4
      - ROUGE-1, ROUGE-2, ROUGE-L (P, R, F1)

    Parameters
    ----------
    input_path : str
        CSV with ``triple`` (reference) and ``output_parsed`` (hypothesis) columns.
    output_dir : str
        Directory to save per-instance result CSVs.

    Returns
    -------
    dict
        Mean metric values.
    """
    df = pd.read_csv(input_path)
    refs = df["triple"].astype(str).tolist()
    hyps = df["output_parsed"].astype(str).tolist()

    # --- BERTScore ---
    P, R, F1 = bert_score(
        hyps, refs,
        model_type="xlm-roberta-base",
        lang="en",
        rescale_with_baseline=True,
    )

    # --- BLEU ---
    smooth = SmoothingFunction().method1
    bleu_scores = {f"bleu{n}": [] for n in range(1, 5)}
    weights_map = {
        "bleu1": (1, 0, 0, 0),
        "bleu2": (0.5, 0.5, 0, 0),
        "bleu3": (1 / 3, 1 / 3, 1 / 3, 0),
        "bleu4": (0.25, 0.25, 0.25, 0.25),
    }

    # --- ROUGE ---
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_data = {
        f"{m}_{s}": []
        for m in ("rouge1", "rouge2", "rougeL")
        for s in ("precision", "recall", "f1")
    }

    for ref, hyp in zip(refs, hyps):
        ref_tok, hyp_tok = ref.split(), hyp.split()
        for key, w in weights_map.items():
            bleu_scores[key].append(
                sentence_bleu([ref_tok], hyp_tok, weights=w, smoothing_function=smooth)
            )
        scores = scorer.score(ref, hyp)
        for metric_name in ("rouge1", "rouge2", "rougeL"):
            s = scores[metric_name]
            rouge_data[f"{metric_name}_precision"].append(s.precision)
            rouge_data[f"{metric_name}_recall"].append(s.recall)
            rouge_data[f"{metric_name}_f1"].append(s.fmeasure)

    # --- Assemble results ---
    results = pd.DataFrame(
        {
            "bert_precision": P.tolist(),
            "bert_recall": R.tolist(),
            "bert_f1": F1.tolist(),
            **bleu_scores,
            **rouge_data,
        }
    )

    name_root = os.path.splitext(os.path.basename(input_path))[0]
    results.to_csv(os.path.join(output_dir, f"{name_root}_results.csv"), index=False)

    return {f"{col}_mean": results[col].mean() for col in results.columns}


def main():
    parser = argparse.ArgumentParser(description="Evaluate parsed model outputs.")
    parser.add_argument(
        "--input_dir",
        default="inference_results/parsed",
        help="Directory with parsed CSVs.",
    )
    parser.add_argument(
        "--output_dir", default="eval_results", help="Directory for evaluation results."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary = []

    for fname in sorted(os.listdir(args.input_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        print(f"Evaluating {fname} ...")
        metrics = evaluate_file(
            os.path.join(args.input_dir, fname), args.output_dir
        )
        metrics["file_name"] = fname
        summary.append(metrics)

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(args.output_dir, "summary_metrics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
