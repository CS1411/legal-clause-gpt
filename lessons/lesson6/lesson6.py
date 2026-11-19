"""Lesson 6: ask it to tag a clause. Needs legalgpt-ft from Lesson 5."""
import sys
from pathlib import Path

from classify import classify, load

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson2"))
from ledgar import heldout, names  # noqa: E402

if __name__ == "__main__":
    tok, model, label_id_to_name = load()
    val = heldout(seed=1, n=6)  # held-out validation clauses, shuffled reproducibly
    print(f"{'TRUE':22s}{'PREDICTED':22s}CLAUSE")
    hits = 0
    for e in val:
        pred = classify(model, tok, e["text"], label_id_to_name)
        hits += pred == names[e["label"]]
        print(f"{names[e['label']]:22s}{pred:22s}{e['text'][:60]}...")
    print(f"\n{hits} of {len(val)} correct")
    assert all(classify(model, tok, e["text"], label_id_to_name) in set(label_id_to_name.values()) for e in val)
