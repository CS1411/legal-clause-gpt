# Lesson 6 - Ask it to tag a clause

- Inference is constrained next-token prediction. The prompt is the clause plus a trailing space, matching the shape seen in training. One forward pass gives a score for every vocabulary token at the last position.
- Instead of an argmax over all 50,263 tokens, only the six label-token scores are compared. The answer is therefore always a valid clause type and cannot drift into ordinary English, which turns the generator back into a classifier.
- `.eval()` switches dropout off for stable results, and `torch.no_grad()` skips gradient bookkeeping.
- Six held-out validation clauses (`shuffle(seed=1)`): all 6 correct, the same six clauses as the course page.
- Code: `lessons/lesson6/classify.py` (`load`, `label_scores`, `classify`) and `lesson6.py`.
