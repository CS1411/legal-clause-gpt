"""Lesson 8: ship your clause tagger behind one method. Needs legalgpt-ft from Lesson 5."""
from tagger import LegalClauseTagger

CLAUSES = [
    ("Governing Laws", "This Agreement shall be governed by and construed in accordance with the laws of the State of New York."),
    ("Notices", "All notices under this Agreement shall be in writing and sent by certified mail to the addresses below."),
    ("Counterparts", "This Agreement may be executed in counterparts, each of which shall be deemed an original."),
]

if __name__ == "__main__":
    tagger = LegalClauseTagger()
    for expected, clause in CLAUSES:
        label, confidence = tagger.classify_with_confidence(clause)
        assert label == tagger.classify_clause(clause)
        print(f"{label:16s} {confidence:6.1%}  :: {clause[:62]}...")
        assert label == expected
