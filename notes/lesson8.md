# Lesson 8 - Ship your clause tagger

- `LegalClauseTagger` wraps everything behind one method: `classify_clause(clause)` loads the fine-tuned model once and returns the clause type. That is the shape a contract-review product would call.
- A stretch idea from the course is also built: `classify_with_confidence(clause)` returns the type and a confidence, the softmax over the six label-token scores (the six values sum to 1). The three example clauses (governing law, notices, counterparts) are each tagged correctly with about 100% confidence.
- Confidence here is only the model's relative preference among the six types, not a calibrated probability, and it is high even on easy examples. Do not read it as a measure of correctness on unusual clauses.
- Project brief (a separate repository): reframe the data as clause text plus label, extend the tokenizer and resize embeddings, fine-tune with the Trainer, evaluate honestly on a held-out split with a look at the confusions, and add your own touch such as more clause types, a bigger subset or `gpt2` instead of `distilgpt2`.
- Code: `lessons/lesson8/tagger.py` and `lesson8.py`.
