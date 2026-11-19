# Lesson 5 - Fine-tune with the Trainer

- Five stages: rebuild the balanced 600-clause subset, rebuild the extended tokenizer and resized model, encode every clause, map the encoder over the data, then run the Hugging Face `Trainer`.
- The key trick is the loss mask. Each clause is encoded as `clause <Label>`, padded to 96 tokens, and every target is set to -100 except the single label-token position. PyTorch's cross-entropy ignores -100, so the model is graded on one question (given this clause, which label?) instead of re-typing text it was already given.
- Settings: 3 epochs, batch size 8, learning rate 5e-4, a fixed seed. All 82M parameters train (this is full fine-tuning). About 9 minutes on a CPU.
- Result: the loss falls from 2.37 to 0.17 and the mean over the run is 0.79. The numbers differ from the course page's run (which was not seeded) because new embedding rows and batching are random, but the shape is the same: a fast fall once the label tokens are learnable.
- The model and tokenizer are saved to `lessons/lesson5/legalgpt-ft/` for Lessons 6-8. That folder is git-ignored.
- Code: `lessons/lesson5/lesson5.py`.
