# Legal-Clause GPT

Fine-tuning a pretrained GPT-2 (distilgpt2) to tag real contract clauses with their type, built while working through the askThili drill *Fine-Tune GPT-2 to Tag Legal Contract Clauses*. The trick is **classification as generation**: each clause gets its label glued on as a special token (`... laws of the State of New York. <Governing Laws>`), the tokenizer and embeddings are extended to match, and the model learns to produce the label.

```console
$ python lessons/lesson8/lesson8.py
Governing Laws   100.0%  :: This Agreement shall be governed by and construed in accordanc...
Notices          100.0%  :: All notices under this Agreement shall be in writing and sent ...
Counterparts     100.0%  :: This Agreement may be executed in counterparts, each of which ...
```

Data and papers: [LEDGAR (Tuggener et al., 2020)](https://aclanthology.org/2020.lrec-1.155/), [LexGLUE (Chalkidis et al., 2022)](https://arxiv.org/abs/2110.00976) and the [lex_glue dataset on Hugging Face](https://huggingface.co/datasets/coastalcph/lex_glue).

## Progress

- [x] Lesson 1 - The task: tag a clause
- [x] Lesson 2 - Meet LEDGAR
- [x] Lesson 3 - Classification as generation
- [x] Lesson 4 - Teach the tokenizer new words
- [x] Lesson 5 - Fine-tune with the Trainer
- [x] Lesson 6 - Ask it to tag a clause
- [x] Lesson 7 - How good is it?
- [x] Lesson 8 - Ship your clause tagger
- [ ] Completion test
- [ ] Project

## Results

All figures are from the runs saved as `output.txt` next to each lesson.

| Lesson | Result |
|---|---|
| 2 | LEDGAR: 60,000 train, 10,000 validation and 10,000 test clauses, 100 clause types. The 6 commonest are used |
| 3 | A balanced training subset: 100 clauses for each of the 6 types, 600 in total |
| 4 | `<Governing Laws>` goes from 6 tokens to 1; the embedding table grows from 50,257 to 50,263 rows |
| 5 | Full fine-tune of distilgpt2 (82M parameters), 3 epochs on 600 clauses, about 9 minutes on a CPU. Loss falls from 2.37 to 0.17 |
| 6 | 6 of 6 held-out clauses tagged correctly |
| 7 | **99.2% accuracy (119 of 120) on held-out clauses**, against 16.7% for random guessing. The single miss: a Survival clause predicted as Severability |
| 8 | `LegalClauseTagger.classify_clause()` behind one method, plus a confidence score |

Treat the 99.2% as one run on 120 clauses. The course page reports 95.8% for the same 120 clauses from an unseeded run; one clause is 0.8 percentage points, so this gap is within what a different seed and initialisation can produce.

## Run it

Requires Python 3.11 or newer and git. Lessons 4 to 8 download distilgpt2 and the LEDGAR dataset from the Hugging Face Hub on first use (no keys needed).

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cd lessons/lesson2 && python lesson2.py     # load and explore LEDGAR
cd ../lesson3 && python lesson3.py          # reframe clauses as generation
cd ../lesson4 && python lesson4.py          # extend the tokenizer, resize the embeddings
cd ../lesson5 && python lesson5.py          # fine-tune, about 9 minutes on CPU, writes legalgpt-ft/
cd ../lesson6 && python lesson6.py          # tag held-out clauses
cd ../lesson7 && python lesson7.py          # accuracy and confusions
cd ../lesson8 && python lesson8.py          # the LegalClauseTagger
```

Lessons 6 to 8 load `lessons/lesson5/legalgpt-ft/`, so run Lesson 5 first. That folder is git-ignored.

## Code layout

```text
lessons/
  lesson1  the task, as text
  lesson2  ledgar.py: dataset, balanced subset, held-out samples (shared by every lesson)
  lesson3  clause + label token reframing
  lesson4  extend the tokenizer and resize the model
  lesson5  the fine-tune with a loss on the label token only
  lesson6  classify.py: constrained next-token classification
  lesson7  accuracy, per-type counts, confusions
  lesson8  tagger.py: LegalClauseTagger
notes/     short notes for every lesson
```

## Environment note

With transformers 5, resizing the embeddings initialises every new row to the same mean vector, which makes the six label tokens identical to the model. The code passes `resize_token_embeddings(..., mean_resizing=False)` to keep them distinct random rows, as in earlier versions. Library versions are pinned in `requirements.txt`.

## Honest notes

This repository was built with AI assistance (Claude) in a working session with the repository owner. The commits were made after the code was finished, so the dates show when the files were committed, not how long each step took. The results above were produced by running this code, not copied from the course page.
