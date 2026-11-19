# Legal-Clause GPT

Working through the askThili drill **Fine-Tune GPT-2 to Tag Legal Contract Clauses**: fine-tune a pretrained GPT-2 on real contract clauses from the LEDGAR dataset so it tags a clause with its type, for example answering `<Governing Laws>` for a governing-law clause.

Papers and data: [LEDGAR (Tuggener et al., 2020)](https://aclanthology.org/2020.lrec-1.155/), [LexGLUE (Chalkidis et al., 2022)](https://arxiv.org/abs/2110.00976) and the [lex_glue dataset on Hugging Face](https://huggingface.co/datasets/coastalcph/lex_glue).

## Progress

- [x] Lesson 1 - The task: tag a clause
- [x] Lesson 2 - Meet LEDGAR
- [x] Lesson 3 - Classification as generation
- [x] Lesson 4 - Teach the tokenizer new words
- [ ] Lesson 5 - Fine-tune with the Trainer
- [ ] Lesson 6 - Ask it to tag a clause
- [ ] Lesson 7 - How good is it?
- [ ] Lesson 8 - Ship your clause tagger
- [ ] Completion test
- [ ] Project

## Layout

- `lessons/lessonN/` code for each lesson
- `notes/` short notes per lesson
