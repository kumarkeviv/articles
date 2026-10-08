# Results

## 1. Routing accuracy, 900 complaints

| Model | Correct | Accuracy | 95% margin | Invalid answers |
|---|---|---|---|---|
| Nimble 9B | 694 / 900 | 77.1% | ±2.7% | 0 |
| Qwen3.5-9B, reasoning off | 687 / 900 | 76.3% | ±2.8% | 0 |

Paired: both right 641, only Nimble 53, only Qwen 46, both wrong 160 (same wrong answer from both: 137). McNemar exact p = 0.547.

## 2. Per team (100 complaints each)

| Team | Nimble 9B | Qwen3.5-9B, reasoning off | Qwen3.5-9B, reasoning on |
|---|---|---|---|
| Credit reporting | 97 | 88 | 99 |
| Debt collection | 53 | 66 | 56 |
| Mortgage | 87 | 93 | 94 |
| Credit card | 75 | 79 | 77 |
| Bank account | 84 | 82 | 79 |
| Student loan | 91 | 91 | 94 |
| Vehicle loan | 72 | 78 | 79 |
| Personal loan | 61 | 49 | 63 |
| Money transfer | 74 | 61 | 75 |

Times each model answered Credit reporting (100 complaints belong there): Nimble 9B 196, Qwen3.5-9B, reasoning off 147, Qwen3.5-9B, reasoning on 177.

## 3. Shared misses (both models gave the same wrong team)

| Labelled as | Both models said | Complaints |
|---|---|---|
| Debt collection | Credit reporting | 26 |
| Money transfer | Bank account | 13 |
| Credit card | Credit reporting | 12 |
| Vehicle loan | Credit reporting | 11 |
| Personal loan | Vehicle loan | 8 |
| Personal loan | Debt collection | 7 |
| Money transfer | Credit card | 6 |
| Bank account | Credit card | 6 |

Nimble and Qwen (reasoning off) gave the same answer for 778 complaints, 82.4% of them right. Nimble at 0.95 or more: 67% of complaints, 89.2% right. Nimble at 0.95 or more and Qwen agrees: 65%, 90.3% right.

## 4. Share routed automatically at a required accuracy

Most confident answers first, counted from 30 answers upward.

| Required accuracy | Nimble 9B | Qwen3.5-9B, reasoning off | Qwen3.5-9B, reasoning on |
|---|---|---|---|
| 85% | 76% | 74% | 0% |
| 90% | 63% | 44% | 0% |
| 95% | 43% | 19% | 0% |

## 5. Stated probability and hit rate

Share correct, number of answers in brackets.

| Stated probability | Nimble 9B | Qwen3.5-9B, reasoning off | Qwen3.5-9B, reasoning on |
|---|---|---|---|
| 0.0 to 0.2 | none | 0% (1) | none |
| 0.2 to 0.4 | 20% (5) | 35% (31) | none |
| 0.4 to 0.6 | 46% (52) | 48% (69) | none |
| 0.6 to 0.8 | 46% (78) | 60% (100) | 0% (1) |
| 0.8 to 1.0 | 83% (765) | 83% (699) | 80% (899) |
| 0.99 or more | 96% (356) | none | 80% (883) |

## 6. Speed, 900 complaints

| Model | Median s | Slowest 5% s | Total min | Median input tokens |
|---|---|---|---|---|
| Nimble 9B | 2.25 | 3.22 | 35.4 | 502 |
| Qwen3.5-9B, reasoning off | 2.07 | 3.03 | 32.5 | 331 |

Time against input tokens, fitted line:

| Model | Fixed part s | Reading rate tokens/s | Requests under 1 s (cache hits) |
|---|---|---|---|
| Nimble 9B | 0.12 | 235 | 1 |
| Qwen3.5-9B, reasoning off | 0.69 | 241 | 0 |

## 7. With reasoning: all 900 complaints

| Model | Correct | Accuracy | 95% margin | Median s | Mean s | Longest s | Total |
|---|---|---|---|---|---|---|---|
| Nimble 9B | 694 / 900 | 77.1% | ±2.7% | 2.2 | 2.4 | 4 | 35 min |
| Qwen3.5-9B, reasoning off | 687 / 900 | 76.3% | ±2.8% | 2.1 | 2.2 | 4 | 33 min |
| Qwen3.5-9B, reasoning on | 716 / 900 | 79.6% | ±2.6% | 49.1 | 82.4 | 591 | 20 h 35 min |

Qwen3.5-9B, reasoning on vs Nimble 9B: only first right 42, only second right 20, McNemar exact p = 0.007.

Qwen3.5-9B, reasoning on vs Qwen3.5-9B, reasoning off: only first right 53, only second right 24, McNemar exact p = 0.001.

Nimble 9B vs Qwen3.5-9B, reasoning off: only first right 53, only second right 46, McNemar exact p = 0.547.

Total time with reasoning on is 35 times Nimble's and 38 times that of reasoning off.

Reasoning time per complaint: fastest 29 s, median 49 s, slowest 5% above 265 s, longest 591 s. Over 1 minute: 330. Over 3 minutes: 108. Over 5 minutes: 29.

Reasoning time when the answer was right: median 45 s. When wrong: median 89 s. Median output: 862 tokens. Longest: 10232 tokens, 591 s, answer wrong.

Run from 2026-10-07 21:44:01 to 2026-10-08 18:12:34. Failed requests: 0.

Share routed automatically. Last column: reasoning model ranked by reasoning length, shortest first.

| Required accuracy | Nimble 9B | Qwen3.5-9B, reasoning off | Qwen3.5-9B, reasoning on | Reasoning on, ranked by length |
|---|---|---|---|---|
| 85% | 76% | 74% | 0% | 87% |
| 90% | 63% | 44% | 0% | 67% |
| 95% | 43% | 19% | 0% | 21% |

Cascade: the first model answers everything, answers below the cut-off go to the reasoning model. Time includes the first model.

| First model | Escalate below | Escalated | Accuracy | Time | Time vs reasoning on everything |
|---|---|---|---|---|---|
| Nimble 9B | never | 0% | 77.1% | 35 min | 3% |
| Nimble 9B | 0.50 | 2% | 77.4% | 1 h 47 min | 9% |
| Nimble 9B | 0.60 | 6% | 78.0% | 3 h 08 min | 15% |
| Nimble 9B | 0.70 | 10% | 78.1% | 4 h 57 min | 24% |
| Nimble 9B | 0.80 | 15% | 78.7% | 6 h 47 min | 33% |
| Nimble 9B | 0.90 | 25% | 78.9% | 9 h 56 min | 48% |
| Nimble 9B | 0.95 | 33% | 79.6% | 12 h 29 min | 61% |
| Nimble 9B | 0.99 | 60% | 79.6% | 16 h 58 min | 82% |
| Qwen3.5-9B, reasoning off | never | 0% | 76.3% | 33 min | 3% |
| Qwen3.5-9B, reasoning off | 0.50 | 7% | 77.7% | 3 h 38 min | 18% |
| Qwen3.5-9B, reasoning off | 0.60 | 11% | 78.0% | 5 h 07 min | 25% |
| Qwen3.5-9B, reasoning off | 0.70 | 18% | 78.2% | 7 h 33 min | 37% |
| Qwen3.5-9B, reasoning off | 0.80 | 22% | 78.3% | 9 h 06 min | 44% |
| Qwen3.5-9B, reasoning off | 0.90 | 40% | 79.4% | 13 h 07 min | 64% |
| Qwen3.5-9B, reasoning off | 0.95 | 74% | 79.3% | 17 h 53 min | 87% |
| Qwen3.5-9B, reasoning off | 0.99 | 100% | 79.6% | 21 h 08 min | 103% |
| Reasoning on for every complaint | | 100% | 79.6% | 20 h 35 min | 100% |

The 299 complaints where Nimble is below 0.95: reasoning on right 180, Nimble right 158. The other 601: reasoning on right 536, Nimble right 536.

## 8. What a cache hit changes (8 complaints per case, 20 for the short input)

| Case | Median s | Tokens |
|---|---|---|
| Nimble: new complaint, one question | 2.57 | 580 |
| Nimble: identical request again | 0.10 | 580 |
| Nimble: same complaint, a different question | 1.82 | 412 |
| Nimble: first 100 characters of a new complaint, three teams | 1.06 | 228 |
| Nimble: same complaint with a sentence added at the end | 2.54 | 582 |
| Nimble: same complaint with a word added at the start | 2.56 | 582 |
| Nimble: new complaint, three questions in one request | 3.35 | 1870 |
| Qwen: instructions first, then a new complaint | 2.38 | 406 |
| Qwen: identical request again | 0.64 | 429 |
| Qwen: complaint first, then instructions | 2.21 | 360 |

## 9. Machine time

Sum of request times per result file.

| Run | Requests | Time |
|---|---|---|
| Nimble 9B | 900 | 35 min |
| Qwen3.5-9B, reasoning off | 900 | 33 min |
| Qwen3.5-9B, reasoning on | 900 | 20 h 35 min |
| Qwen3.5-9B, reasoning off, letter answers | 900 | 33 min |
| All runs | | 22 h 17 min |

## 10. Qwen with a probability for all nine teams, 900 complaints

Reasoning off. Qwen answers with one letter per team. The probabilities of the nine letters are scaled to add up to 1.

Accuracy 680 / 900, 75.6% (±2.8%). Against Nimble 9B: only Nimble right 62, only Qwen right 48, McNemar exact p = 0.215.

Share of the probability that was on the nine letters before scaling: median 83%.

| Required accuracy | Nimble 9B | Qwen, team name | Qwen, letter |
|---|---|---|---|
| 85% | 76% | 74% | 74% |
| 90% | 63% | 44% | 58% |
| 95% | 43% | 19% | 16% |

The same shares with a 95% interval, from 2,000 resamples of the complaints:

| Required accuracy | Nimble 9B | Qwen, team name | Qwen, letter |
|---|---|---|---|
| 90% | 56% to 71% | 25% to 64% | 45% to 68% |
| 95% | 34% to 56% | 12% to 32% | 0% to 45% |

Nimble routes more than Qwen, letter, at 95% in 99% of the resamples.

Qwen, letter, without scaling to 1: 65% at 85%, 49% at 90%, 29% at 95%.

Chance that a right answer has a higher probability than a wrong one (0.5 is no better than a coin): Nimble 9B 0.80, Qwen, team name 0.75, Qwen, letter 0.78, Qwen, reasoning on 0.50.

Qwen, team name: the 168 most confident answers by team: Mortgage 84, Debt collection 38, Bank account 21. Highest probability 0.990.

| Stated probability | Nimble 9B | Qwen, letter |
|---|---|---|
| 0.0 to 0.2 | none | none |
| 0.2 to 0.4 | 20% (5) | 34% (35) |
| 0.4 to 0.6 | 46% (52) | 45% (73) |
| 0.6 to 0.8 | 46% (78) | 54% (97) |
| 0.8 to 1.0 | 83% (765) | 84% (695) |
| 0.99 or more | 96% (356) | 93% (314) |
