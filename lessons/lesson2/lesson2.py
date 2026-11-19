"""Lesson 2: meet LEDGAR. Load the dataset, look at the clause types and a few real clauses."""
from ledgar import TOP, ds, names

if __name__ == "__main__":
    print("splits:", {k: len(v) for k, v in ds.items()})
    print("number of clause types:", len(names))
    print("\nthe 6 most common types (what we'll train on):", [names[i] for i in TOP])
    print("\nsample clauses:")
    for e in ds["train"].select(range(3)):
        print(f"  [{names[e['label']]}] {e['text'][:90]}...")
    assert (len(ds["train"]), len(ds["validation"]), len(names)) == (60000, 10000, 100)
