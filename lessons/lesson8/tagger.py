"""LegalClauseTagger: the whole tagger behind one method, classify_clause()."""
import sys
from pathlib import Path

import torch.nn.functional as F
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lesson6"))
from classify import MODEL_DIR, label_scores, load  # noqa: E402


class LegalClauseTagger:
    "Load the fine-tuned model and tag a contract clause with its type."

    def __init__(self, path=MODEL_DIR):
        self.tok, self.model, self.label_id_to_name = load(path)

    def classify_clause(self, clause):
        scores = label_scores(self.model, self.tok, clause, self.label_id_to_name)
        return max(scores, key=scores.get)

    def classify_with_confidence(self, clause):
        "The clause type and a confidence: the softmax over the six label tokens (they add up to 1)."
        scores = label_scores(self.model, self.tok, clause, self.label_id_to_name)
        probs = F.softmax(torch.tensor(list(scores.values())), dim=0)
        best = int(probs.argmax())
        return list(scores)[best], float(probs[best])
