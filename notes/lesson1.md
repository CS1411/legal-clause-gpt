# Lesson 1 - The task: tag a clause

- A commercial contract is a stack of clauses (governing law, notices, counterparts, indemnity...), and a reviewer has to recognise each one. That recognition is a classification task and the core of contract-review tools.
- The approach is domain adaptation: take a model that already knows English (GPT-2) and keep training it on contract text so it learns the vocabulary and patterns of that field.
- GPT-2 is a generator, not a classifier, so tagging is turned into generation: the clause's label is glued to the end of its text as one special token, and the model learns to produce it.
- At test time the model gets only the clause, and the label token it generates is the prediction.
- Code: `lessons/lesson1/lesson1.py` (no model, just the shape of the problem).
