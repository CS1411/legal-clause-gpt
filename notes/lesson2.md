# Lesson 2 - Meet LEDGAR

- LEDGAR is part of the LexGLUE benchmark: real clauses from public EDGAR contract filings, each labelled with one of 100 clause types. Splits: 60,000 train, 10,000 validation, 10,000 test.
- The six most common types are Governing Laws, Notices, Counterparts, Entire Agreements, Severability and Survival. Training uses only these, so it finishes in minutes on a laptop CPU; the same code scales to all 100 types on a GPU.
- The dataset is downloaded once from the Hugging Face Hub and cached. `lessons/lesson2/ledgar.py` loads it and is shared by every later lesson (`balanced_subset`, `heldout`, `label_token`).
- Code: `lessons/lesson2/lesson2.py`.
