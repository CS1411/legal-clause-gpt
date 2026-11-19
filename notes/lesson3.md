# Lesson 3 - Classification as generation

- GPT-2 only predicts the next token, so the label is made the next token: each training example is the clause text followed by its label token, for example `... laws of the State of New York. <Governing Laws>`.
- The model never "classifies": it continues the text, and the continuation happens to be the answer.
- The training set is a balanced subset: 100 clauses for each of the 6 types, 600 in total, so the model cannot score well by always guessing the commonest label.
- `<Governing Laws>` is not a word GPT-2 knows yet. The tokenizer would shatter it into pieces, which Lesson 4 fixes.
- Code: `lessons/lesson3/lesson3.py`.
