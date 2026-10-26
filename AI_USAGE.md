# AI usage

## Tools I used, and for which functions

Used ChatGPT/Claude for drafting the initial implementation of `audit()` in `rawaj/audit.py` and metric calculations in `rawaj/metrics.py`.

## One place where the AI was wrong, incomplete, or not what the spec said

The initial AI suggestion for `best_threshold` used standard ascending `np.unique` and `np.argmin`, which returns the lowest threshold on cost ties. The specification explicitly required returning the highest threshold on ties. I caught this by reviewing the task instructions for `best_threshold` and added custom tie-breaking logic.

## The function I wrote without AI, and why

I wrote `precision_at_k` without AI. I wanted to manually handle the edge cases for k <= 0 and k > n while ensuring the score sorting was strictly stable (preserving the original order on ties using `mergesort`). Writing it myself helped me better understand how ranking order directly impacts top-k precision metrics.