"""Lesson 1: the task. Tag a contract clause with its type, by turning tagging into text generation.

No model here: this lesson only shows the shape of the problem that the next lessons solve.
"""

EXAMPLES = [
    ("This Agreement shall be governed by the laws of the State of Delaware.", "Governing Laws"),
    ("All notices under this Agreement shall be in writing and sent by certified mail.", "Notices"),
    ("This Agreement may be executed in counterparts, each of which shall be deemed an original.", "Counterparts"),
]


def as_generation_example(clause, label):
    "GPT-2 only continues text, so the label is glued to the end of the clause as one special token."
    return f"{clause.strip()} <{label}>"


if __name__ == "__main__":
    print("A contract is a stack of clauses, and every clause has a type a reviewer must recognise.\n")
    for clause, label in EXAMPLES:
        print(f"{label:16s} <- {clause}")
    print("\nGPT-2 is a generator, not a classifier, so each example becomes ONE string for it to learn:\n")
    for clause, label in EXAMPLES:
        print("  " + as_generation_example(clause, label))
    print("\nAt test time we give it only the clause, and the label token it generates is the prediction.")
    assert as_generation_example(*EXAMPLES[0]).endswith("<Governing Laws>")
