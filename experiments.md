# Experiments

## Objective and protocol

Optimize mean pinball loss over quantile levels 0.05, 0.25, 0.50, 0.75, 0.95 for the MakeFun four-hour, minute-level horizon. Evaluation is causal rolling origin: p00 through p04 are predicted using only each checkpoint's input history; labels are sliced from the next checkpoint's recorded history. Reports give percentage points (pp).

## Baselines

The official random-walk prior uses sigma 0.0025 per square-root minute and normal quantile multipliers from the kit. It scored 0.392518 pp on the five fully labeled transitions available locally. A modest fixed sigma reduction and a recent-volatility adaptive sigma both underperformed the empirical candidates in exact testing/screening.

## Candidate search

We screened empirical-history parameters using every tenth horizon minute (a screening approximation): price bandwidth 0.15, 720-minute recency half-life, 30-minute-spaced origins, equal local/peer pooling, and 60% or 80% empirical blend. Four origins with 80% blend had screening loss 0.369856 pp; 12 origins with 80% blend had 0.361497 pp. The latter exceeded the 8-second practice-case runtime when evaluated at every minute, so the entry uses four origins.

The chosen four-origin entry was then evaluated at every minute on p00–p04 and obtained 0.360483 pp, versus 0.392518 pp for the baseline on those same cases. The five-transition sample comprises 24,960 market-minute outcomes. The official six-case practice score includes p05; its future tail was not present in the kit, so no six-case result is claimed. The kit baseline's published six-case result is 0.363589 pp, but a direct full-sample comparison is not available.

## Limitations

- Practice cases are from the NVIDIA CEO market family, whereas the final is the FAA attribution family. The bounty explicitly cautions that practice performance does not indicate final performance.
- Peer market pooling may be imperfect when outcomes are not comparable.
- The available labeled sample is only five rolling transitions; this can overstate generalization.
- The four-origin choice trades some screening score for runtime headroom. Run the official grader to measure actual sandbox runtime and final score.


