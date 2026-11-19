"""LEDGAR (part of LexGLUE): real contract clauses from public EDGAR filings, each labelled with one of 100 types.

Shared by every lesson. The data is downloaded once from the Hugging Face Hub and cached.
"""
import random
from collections import Counter, defaultdict

from datasets import load_dataset

ds = load_dataset("coastalcph/lex_glue", "ledgar")
names = ds["train"].features["label"].names               # the 100 clause-type names
TOP = [lab for lab, _ in Counter(ds["train"]["label"]).most_common(6)]  # the 6 most common types
topset = set(TOP)


def balanced_subset(split, per_class, seed):
    "per_class clauses of each of the 6 commonest types, shuffled: equal counts, so guessing the majority label cannot win."
    picked = defaultdict(list)
    for i, lab in enumerate(split["label"]):
        if lab in topset and len(picked[lab]) < per_class:
            picked[lab].append(i)
        if all(len(picked[t]) >= per_class for t in TOP):
            break
    flat = [i for t in TOP for i in picked[t]]
    random.Random(seed).shuffle(flat)
    return split.select(flat)


def label_token(label_index):
    "The special token for a clause type, for example <Governing Laws>."
    return f"<{names[label_index]}>"


def heldout(seed, n):
    "n validation clauses of the 6 types, shuffled reproducibly. Lesson 6 uses seed 1 (6 clauses); Lesson 7 seed 2 (120)."
    return ds["validation"].filter(lambda e: e["label"] in topset).shuffle(seed=seed).select(range(n))
