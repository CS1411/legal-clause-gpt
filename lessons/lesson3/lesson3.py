"""Lesson 3: classification as generation. Every labelled clause becomes one string ending in its label token."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson2"))
from ledgar import TOP, balanced_subset, ds, label_token, names  # noqa: E402


def reframe(example):
    "clause text + the label as a single token appended at the end: what the model learns to produce."
    return example["text"].strip() + " " + label_token(example["label"])


if __name__ == "__main__":
    train = balanced_subset(ds["train"], per_class=100, seed=0)  # 600 clauses, 100 per type
    print("reframed training examples (note each ends with its label token):\n")
    for e in train.select(range(4)):
        text = reframe(e)
        print(f"  ...{text[-60:]}")
    print(f"\ntraining subset: {len(train)} clauses ({len(TOP)} types x 100, balanced)")
    counts = {names[t]: sum(1 for lab in train["label"] if lab == t) for t in TOP}
    print("per type:", counts)
    assert len(train) == 600 and set(counts.values()) == {100}
    assert all(reframe(e).endswith(">") for e in train.select(range(20)))
