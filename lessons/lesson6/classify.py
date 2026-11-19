"""Inference: load the fine-tuned model and classify a clause with one constrained next-token step."""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lesson2"))
from ledgar import TOP, label_token, names  # noqa: E402

MODEL_DIR = HERE.parent / "lesson5" / "legalgpt-ft"


def load(path=MODEL_DIR):
    "The fine-tuned tokenizer, the model in eval mode (dropout off) and {label token id: clause type name}."
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path).eval()
    label_id_to_name = {tok.convert_tokens_to_ids(label_token(i)): names[i] for i in TOP}
    return tok, model, label_id_to_name


@torch.no_grad()
def label_scores(model, tok, clause, label_id_to_name):
    "The model's raw score for each clause type as the very next token after the clause."
    logits = model(**tok(clause.strip() + " ", return_tensors="pt")).logits[0, -1]
    return {label_id_to_name[t]: logits[t].item() for t in label_id_to_name}


def classify(model, tok, clause, label_id_to_name):
    "One forward pass, argmax over only the six label tokens: the answer is always a valid clause type."
    scores = label_scores(model, tok, clause, label_id_to_name)
    return max(scores, key=scores.get)
