#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Round-Trip Pipeline: Structure-to-Text-and-Back Loop
=====================================================

Implements the iterative round-trip generation pipeline described in:

    "From Graph to Text and Back: Semantic Fidelity in
     Automated Industrial Knowledge Graphs"
    (ACL 2026 — Industry Track)

The pipeline alternates between:
  • Stage 1  –  KG → Text  (verbalization via LLM₁)
  • Stage 2  –  Text → KG  (reconstruction via LLM₂)
and uses a combined round-trip similarity score to dynamically
adjust decoding parameters (temperature, top-p) over N cycles.
"""

import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from rouge_score import rouge_scorer
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# 1.  Embedding-Based Similarity Calculator
# ---------------------------------------------------------------------------
class EmbeddingSimilarity:
    """Compute cosine similarity between two texts using a HuggingFace
    encoder model (default: multilingual-e5-large-instruct)."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    @torch.no_grad()
    def compute(self, text_a: str, text_b: str) -> float:
        """Return cosine similarity ∈ [-1, 1] between *text_a* and *text_b*."""
        emb_a = self._encode(text_a)
        emb_b = self._encode(text_b)
        return torch.nn.functional.cosine_similarity(emb_a, emb_b).item()

    def _encode(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        if torch.cuda.is_available():
            tokens = {k: v.cuda() for k, v in tokens.items()}
        return self.model(**tokens).last_hidden_state.mean(dim=1)


# ---------------------------------------------------------------------------
# 2.  LLM Generator (Chat-Style)
# ---------------------------------------------------------------------------
class LLMGenerator:
    """Wrapper around a causal LM for chat-style text generation."""

    def __init__(self, model_name: str, custom_chat_template: str | None = None):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        if torch.cuda.is_available():
            self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if custom_chat_template is not None:
            self.tokenizer.chat_template = custom_chat_template

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.5,
        top_p: float = 1.0,
        max_new_tokens: int = 256,
    ) -> tuple[str, str]:
        """Generate a response for a list of chat *messages*.

        Returns
        -------
        full_output : str
            The complete decoded output (including the prompt portion).
        prompt_text : str
            The decoded prompt that was fed to the model.
        """
        input_dict = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        prompt_text = self.tokenizer.decode(input_dict["input_ids"][0])

        self.model.eval()
        outputs = self.model.generate(
            **input_dict,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=50,
            top_p=top_p,
            do_sample=True,
            no_repeat_ngram_size=4,
            repetition_penalty=1.0,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        full_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
        return full_output, prompt_text

    def evaluate_similarity(self, original: str, generated: str) -> float:
        """Ask the LLM to rate similarity between two triple sets (0–1)."""
        messages = [
            {"role": "system", "content": "You are an expert evaluator."},
            {
                "role": "user",
                "content": (
                    f"Compare the following original triple set:\n'{original}'\n"
                    f"with the generated triple set:\n'{generated}'\n"
                    "and provide a similarity score between 0 and 1."
                ),
            },
        ]
        result, _ = self.generate(messages, temperature=0.1, top_p=0.9)
        match = re.search(r"(\d+(\.\d+)?)", result)
        return float(match.group(0)) if match else 0.5


# ---------------------------------------------------------------------------
# 3.  Prompt Templates
# ---------------------------------------------------------------------------
class PromptTemplates:
    """Build chat-formatted message lists for the two pipeline stages."""

    @staticmethod
    def kg_to_text_first(triples: str) -> list[dict]:
        """KG → Text  (first pass, no reference paragraph)."""
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant skilled in generating "
                    "descriptive paragraphs from knowledge-graph triples."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Using the following knowledge-graph triples, craft a detailed "
                    "paragraph of at least 70 words that elaborates on the information "
                    f"provided. Return only the generated text.\n{triples}"
                ),
            },
        ]

    @staticmethod
    def kg_to_text_subsequent(triples: str, reference: str) -> list[dict]:
        """KG → Text  (subsequent passes, with diversity reference)."""
        return [
            {
                "role": "system",
                "content": (
                    "You are a creative assistant skilled in crafting original "
                    "content based on provided data."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Using the knowledge-graph triples provided below, write a clear "
                    "and original paragraph of at least 70 words. The paragraph must "
                    "begin with the subject of the triples and must include all "
                    "information accurately from the triples — nothing should be left "
                    "out or misrepresented. While the paragraph should differ from the "
                    "reference, it need not be drastically different. Return only the "
                    f"paragraph.\n\nReference Paragraph:\n{reference}\n\n"
                    f"Knowledge-Graph Triples:\n{triples}"
                ),
            },
        ]

    @staticmethod
    def text_to_kg(paragraph: str) -> list[dict]:
        """Text → KG  (triple extraction)."""
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant specialised in extracting "
                    "structured data."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Extract exactly five knowledge-graph triples from the text below "
                    "in the form (Entity -> Relation -> Object). Return only the "
                    f"extracted triples.\n\nTEXT:\n{paragraph}"
                ),
            },
        ]

    @staticmethod
    def extract_model_output(full_text: str, prompt_text: str) -> str:
        """Strip the prompt portion from the full generated text."""
        if full_text.startswith(prompt_text):
            cleaned = full_text[len(prompt_text) :].strip()
        else:
            cleaned = full_text
        # Detect structured triple output
        m = re.search(r"(Triple 1:.*)", cleaned, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Remove common chat markers
        cleaned = re.sub(r"\[System\]:.*", "", cleaned)
        cleaned = re.sub(r"\[User\]:.*", "", cleaned)
        return cleaned.strip()


# ---------------------------------------------------------------------------
# 4.  Lexical Similarity (ROUGE)
# ---------------------------------------------------------------------------
class LexicalSimilarity:
    """ROUGE-based lexical similarity utilities."""

    _scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    @classmethod
    def compute(cls, reference: str, hypothesis: str) -> dict:
        """Return ROUGE-1 / -2 / -L score dicts."""
        return cls._scorer.score(reference, hypothesis)

    @staticmethod
    def preprocess(text: str) -> str:
        """Normalize triple-formatted text for a fairer lexical comparison."""
        noise = ["subject", "predicate", "object", "Triple"]
        text = re.sub(r"[;:!\-.]", " ", text)
        text = re.sub(
            r"\b(?:" + "|".join(noise) + r")\b", "", text, flags=re.IGNORECASE
        )
        return " ".join(text.split())


# ---------------------------------------------------------------------------
# 5.  Round-Trip Pipeline Controller
# ---------------------------------------------------------------------------
class RoundTripPipeline:
    """Iterative structure-to-text-and-back loop (Algorithm 1 in the paper).

    Parameters
    ----------
    text_model : str
        HuggingFace model name/path for KG → Text generation.
    triple_model : str
        HuggingFace model name/path for Text → KG extraction.
    initial_triples : str
        The seed KG triples in textual form.
    cycles : int
        Number of round-trip iterations *N*.
    initial_temperature, initial_top_p : float
        Starting decoding parameters.
    temperature_delta, top_p_delta : float
        Fixed step sizes for the adaptive controller.
    tau_low : float
        Lower fidelity threshold; scores below this trigger parameter reduction.
    tau_high : float
        Upper fidelity threshold; scores at or above this trigger parameter increase.
    tau_select : float
        Selection threshold; only (S, T̂) pairs with fidelity ≥ τ_select are retained.
    alpha : float
        Balance weight between semantic and lexical similarity
        (default 0.7 as per paper).
    custom_chat_template : str | None
        Optional Jinja chat template override.
    """

    def __init__(
        self,
        text_model: str,
        triple_model: str,
        initial_triples: str,
        *,
        cycles: int = 100,
        initial_temperature: float = 0.8,
        initial_top_p: float = 0.8,
        temperature_delta: float = 0.05,
        top_p_delta: float = 0.05,
        tau_low: float = 0.5,
        tau_high: float = 0.9,
        tau_select: float = 0.85,
        alpha: float = 0.7,
        custom_chat_template: str | None = None,
    ):
        self.text_model_name = text_model
        self.triple_model_name = triple_model
        self.initial_triples = initial_triples
        self.cycles = cycles
        self.temperature = initial_temperature
        self.top_p = initial_top_p
        self.temperature_delta = temperature_delta
        self.top_p_delta = top_p_delta
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.tau_select = tau_select
        self.alpha = alpha
        self.custom_chat_template = custom_chat_template

    # ---- similarity -------------------------------------------------------
    def _round_trip_similarity(self, emb_score: float, rouge_l: float) -> float:
        """sim(S, Ŝ, α) = α · cos(φ(S), φ(Ŝ)) + (1−α) · sim_lex(S, Ŝ)"""
        return self.alpha * emb_score + (1 - self.alpha) * rouge_l

    # ---- main loop --------------------------------------------------------
    def run(self) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
        """Execute the iterative round-trip pipeline (Algorithm 1).

        Returns
        -------
        metrics_df : pd.DataFrame
            Per-cycle metrics for analysis.
        selected : list[tuple[str, str]]
            List of ``(S, T̂)`` pairs that passed the τ_select threshold.
        """
        text_llm = LLMGenerator(self.text_model_name, self.custom_chat_template)
        triple_llm = LLMGenerator(self.triple_model_name, self.custom_chat_template)
        emb_sim = EmbeddingSimilarity()

        current_text = self.initial_triples
        generated_texts: list[str] = []
        selected: list[tuple[str, str]] = []
        records: list[dict] = []

        for i in range(self.cycles):
            record = {
                "cycle": i + 1,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }

            # --- Step 1: Structure-to-Text Generation ---
            if i == 0:
                messages = PromptTemplates.kg_to_text_first(self.initial_triples)
                task = "text"
            elif i % 2 == 0:
                messages = PromptTemplates.kg_to_text_subsequent(
                    self.initial_triples, generated_texts[-1]
                )
                task = "text"
            else:
                messages = PromptTemplates.text_to_kg(current_text)
                task = "triple"

            # --- generate ---
            llm = text_llm if task == "text" else triple_llm
            full_output, prompt_text = llm.generate(
                messages, temperature=self.temperature, top_p=self.top_p
            )
            extracted = PromptTemplates.extract_model_output(full_output, prompt_text)
            record["task"] = task
            record["extracted_output"] = extracted

            # --- Step 2/3: Evaluate fidelity ---
            if task == "text":
                generated_texts.append(extracted)
                # Compare with all previous text generations
                for j, prev in enumerate(generated_texts[:-1]):
                    rouge = LexicalSimilarity.compute(prev, extracted)
                    emb = emb_sim.compute(prev, extracted)
                    record[f"text_emb_sim_{j+1}"] = emb
                    record[f"text_rougeL_{j+1}"] = rouge["rougeL"].fmeasure
            else:
                # Compare reconstructed triples against original
                prep_orig = LexicalSimilarity.preprocess(self.initial_triples)
                prep_gen = LexicalSimilarity.preprocess(extracted)
                rouge = LexicalSimilarity.compute(prep_orig, prep_gen)
                emb = emb_sim.compute(prep_orig, prep_gen)
                llm_sim = triple_llm.evaluate_similarity(prep_orig, prep_gen)
                fidelity = self._round_trip_similarity(emb, rouge["rougeL"].fmeasure)

                record["triple_emb_sim"] = emb
                record["triple_rougeL"] = rouge["rougeL"].fmeasure
                record["triple_llm_sim"] = llm_sim
                record["round_trip_sim"] = fidelity

                # --- Step 4: Adaptive Parameter Adjustment (Algorithm 1) ---
                if fidelity < self.tau_low:
                    # Low fidelity → decrease randomness to recover accuracy
                    self.temperature = max(0.0, self.temperature - self.temperature_delta)
                    self.top_p = max(0.0, self.top_p - self.top_p_delta)
                elif fidelity >= self.tau_high:
                    # High fidelity → increase randomness to promote diversity
                    self.temperature = min(1.0, self.temperature + self.temperature_delta)
                    self.top_p = min(1.0, self.top_p + self.top_p_delta)
                # else: τ_low ≤ fidelity < τ_high → keep parameters unchanged

                # --- Selection: retain high-fidelity pairs ---
                if fidelity >= self.tau_select:
                    selected.append((self.initial_triples, generated_texts[-1]))
                    record["selected"] = True
                else:
                    record["selected"] = False

            records.append(record)
            current_text = extracted
            time.sleep(0.5)

        return pd.DataFrame(records), selected


# ---------------------------------------------------------------------------
# 6.  CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the round-trip KG ↔ Text generation pipeline."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="CSV file with a 'prompt' column containing KG triples.",
    )
    parser.add_argument(
        "--text_model",
        type=str,
        default="microsoft/Phi-3.5-mini-instruct",
        help="HuggingFace model for KG → Text.",
    )
    parser.add_argument(
        "--triple_model",
        type=str,
        default="microsoft/Phi-3.5-mini-instruct",
        help="HuggingFace model for Text → KG.",
    )
    parser.add_argument("--cycles", type=int, default=100, help="Number of iterations.")
    parser.add_argument("--tau_low", type=float, default=0.5, help="Low fidelity threshold.")
    parser.add_argument("--tau_high", type=float, default=0.9, help="High fidelity threshold.")
    parser.add_argument("--tau_select", type=float, default=0.85, help="Selection threshold.")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory.")
    args = parser.parse_args()

    input_df = pd.read_csv(args.input_csv)
    for idx, row in input_df.iterrows():
        prompt = row.get("prompt")
        if prompt is None:
            print(f"Row {idx} has no 'prompt' column — skipping.")
            continue
        print(f"\n{'='*60}\n  Processing Input {idx}\n{'='*60}")
        pipeline = RoundTripPipeline(
            text_model=args.text_model,
            triple_model=args.triple_model,
            initial_triples=prompt,
            cycles=args.cycles,
            tau_low=args.tau_low,
            tau_high=args.tau_high,
            tau_select=args.tau_select,
        )
        results, selected = pipeline.run()
        out_path = f"{args.output_dir}/output_{idx}.csv"
        results.to_csv(out_path, index=False)
        print(f"  → Metrics saved to {out_path}")
        print(f"  → {len(selected)} high-fidelity pairs selected (τ_select={args.tau_select})")

        if selected:
            sel_path = f"{args.output_dir}/selected_{idx}.csv"
            sel_df = pd.DataFrame(selected, columns=["triples", "text"])
            sel_df.to_csv(sel_path, index=False)
            print(f"  → Selected pairs saved to {sel_path}")

