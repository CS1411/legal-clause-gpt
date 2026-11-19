"""Lesson 7: how good is it? Accuracy on held-out clauses, then which clause types get confused. Needs Lesson 5."""
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
for lesson in ("lesson2", "lesson6"):
    sys.path.insert(0, str(HERE.parent / lesson))
from classify import classify, load  # noqa: E402
from ledgar import TOP, heldout, names  # noqa: E402

if __name__ == "__main__":
    tok, model, label_id_to_name = load()
    val = heldout(seed=2, n=120)
    preds = [classify(model, tok, e["text"], label_id_to_name) for e in val]
    trues = [names[e["label"]] for e in val]
    acc = sum(p == t for p, t in zip(preds, trues)) / len(val)
    print(f"clause-type accuracy on {len(val)} held-out clauses: {acc:.1%} (random = {1 / len(TOP):.1%})")

    print("\nper type (correct / clauses):")
    for t in (names[i] for i in TOP):
        n = sum(1 for x in trues if x == t)
        print(f"  {t:18s} {sum(1 for p, x in zip(preds, trues) if x == t and p == t):3d} / {n}")

    confusions = Counter((t, p) for p, t in zip(preds, trues) if p != t)
    print("\nmost common confusions (true -> predicted):")
    for (t, p), n in confusions.most_common(5):
        print(f"  {t:18s} -> {p:18s} x{n}")
    print("\na few misses:")
    for p, t, e in [x for x in zip(preds, trues, val) if x[0] != x[1]][:5]:
        print(f"  true={t:18s} pred={p:18s} :: {e['text'][:55]}...")
    assert acc > 0.5  # far above the 16.7% of random guessing
