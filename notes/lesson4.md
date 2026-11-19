# Lesson 4 - Teach the tokenizer new words

- Three one-line steps: add the six label tokens so each is a single unsplittable token, resize the model's embeddings so there is a row for each, and give GPT-2 a pad token.
- Before: `<Governing Laws>` is 6 tokens (`<`, `G`, `over`, `ning`, ` Laws`, `>`). After: 1 token. Predicting one token is a single step that cannot derail halfway, unlike spelling six pieces.
- A language model has an input embedding table and an output head, both sized to the vocabulary. Adding 6 tokens grows both from 50,257 to 50,263 rows. The new rows start random and are learned during fine-tuning, which is how a label token acquires its meaning.
- No legal words are added: byte-pair encoding can spell any string from about 50,000 pieces, so an unseen word like "indemnification" just costs a few extra tokens. Only the clause types become atomic tokens.
- GPT-2 has no pad token, so end-of-text is reused. Batches must be rectangles, so short clauses are padded and an attention mask (1 real, 0 padding) tells the model to ignore the filler.
- Environment note: transformers 5 initialises new embedding rows to the same mean vector by default, which makes the six labels identical to the model. `resize_token_embeddings(..., mean_resizing=False)` keeps them distinct random rows and reproduces the course results.
- Code: `lessons/lesson4/lesson4.py`.
