# Lesson 7 - How good is it?

- Accuracy is measured on 120 held-out validation clauses (`shuffle(seed=2)`), compared with 16.7% for random guessing among 6 types.
- Result from this repo's run: **99.2% (119 of 120)**. Per type: Governing Laws 17/17, Notices 20/20, Counterparts 28/28, Entire Agreements 22/22, Severability 16/16, Survival 16/17. The one miss is a Survival clause predicted as Severability.
- The course page reports 95.8% for the same 120 clauses from its own run. Both are single runs, and a 120-clause sample means one clause is 0.8 percentage points, so the difference is within what a different seed and initialisation can produce. It should not be read as one recipe being better.
- Where models like this slip is among boilerplate look-alikes (Severability, Entire Agreements, Survival often read alike), which is also where a human reviewer slows down.
- Limits: 6 of the 100 clause types, a balanced training subset, and a small test sample. Scale up the types, the data and the model for something production-like.
- Code: `lessons/lesson7/lesson7.py`, which also prints per-type counts and the confusions.
