# Frozen public-information calibration

Fit dates are October 1–2; interval and probability calibration dates are October 3–11. October 13/26/30 were declared before inspecting their actuals. Route/event-kind empirical delay and issued-forecast-error laws, route shrinkage, a receipt-available peer feature and fitted mixture weights were frozen before the new-date readout. Endpoint predictions use schedule minus 5 minutes; transfer predictions use the traveler's observed incoming arrival and forecasts received by that clock. No same-day network-wide actual stream is assumed.

The public peer feature does not reveal the October 25 shock: actual minus prior mean averages +114.4 s, while the received peer forecast feature averages +3.0 s. Residuals average +147.0/+126.1/+62.0 s at 07/08/09 h. Departure and arrival both shift, and earlier diagnostics find little first-to-last growth within a trip. Full route/kind/hour/source splits are in `calibration_diagnosis.json` and `calibration_confirmation_diagnosis.json`.

The table compares the **same October 1–2 fit split**, including separately frozen nine-date interval offsets. Widths and MAE are seconds. Transfer Brier scores use the same measured critical states and an explicit 12-minute patience rule.

| New date | Operating endpoints / transfer states | Evolving coverage / width / MAE | Public mixture coverage / width / MAE | Brier Evolving → mixture |
|---|---:|---:|---:|---:|
| Oct 13 | 1,038 / 2,620 | 97.98% / 501.2 / 57.9 | 98.07% / 387.0 / 51.6 | .07289 → .05473 |
| Oct 26 | 1,044 / 2,728 | 89.94% / 501.4 / 104.5 | 93.20% / 401.9 / 99.2 | .11424 → .10657 |
| Oct 30 | 1,044 / 2,702 | 96.26% / 501.4 / 68.3 | 97.80% / 393.4 / 66.5 | .08137 → .07458 |

Across 3,126 operating endpoints, coverage is 3,012/3,126 versus 2,961/3,126, mean width 394.1 versus 501.4 s, interval score 476.2 versus 641.4, and MAE 72.43 versus 76.97 s. Date-average Brier is .07863 versus .08950. These are three paired date observations; endpoints and transfer pairs are dependent, so their counts do not supply thousands of independent replications.

A later freeze produced the incoming-known shrinkage and heavy-tail candidates, independently confirmed on October 16/20/31. The frozen shrinkage results below use schedule-selected transfer pairs, rather than the endpoint population above.

| Later independent date / pairs | Shrinkage coverage / width / MAE | Brier Evolving → shrinkage |
|---|---:|---:|
| Oct 16 / 2,898 | 98.83% / 395.9 s / 49.4 s | .06601 → .05061 |
| Oct 20 / 2,900 | 98.45% / 395.9 s / 49.7 s | .06740 → .05082 |
| Oct 31 / 2,900 | 97.48% / 395.9 s / 66.2 s | .08341 → .08062 |

These are six predeclared dates across **two successive confirmation stages**. October 13/26/30 are independent for the first frozen public mixture, but were already observed when the later conditional candidates were designed; they are not independent evidence for those later candidates. The two stages use different scoring populations and are not pooled as six same-model replications. Full later controls and tradeoffs are in [the conditional confirmation record](calibration_relative_confirmation_RESULTS.md).

Negative results remain: the previously examined October 25 date still covers only 858/1,020 = 84.12% under the candidate, versus 85.39% for the original envelope. Its Brier deteriorates .13556 → .14352. The independently frozen date-balanced isotonic maps worsen the candidate's Brier on Oct 26/30 (.11105/.07703 versus .10657/.07458); they are retained as a failed comparator. Direct Gaussian conditioning on uncorrected public forecasts also undercovers; October 27 raw coverage is 41.71%. The historical October 2/6/9 fit is retained separately in `calibration_secondary_reference.json`, rather than conflated with this matched split.

That historical three-date fit has raw coverage 95.28/81.23/90.33%, width 367.5 s, MAE 58.0/100.9/67.6 s and Brier .07271/.11072/.08062 on Oct 13/26/30. The new candidate's calibrated interval is slightly wider than that raw reference and covers more; its MAE and Brier are lower on all three dates. No October 1–2 calibration offset is transplanted onto the historical fit.

For the interval statement, let each day contribute the empirical 90th-percentile nonconformity of its eligible observed operating endpoints. The offset is the largest of nine calibration-day scores. Conditional on the fixed fitted predictor, **exchangeable calibration and new day blocks** give probability at least 9/10 that the new day's empirical endpoint coverage is at least 90%, by the rank of its score among ten exchangeable scores. Within-day independence is unnecessary. This is a date-level tolerance statement, not simultaneous coverage of every endpoint and not a calibration theorem for the underlying CDF or boarding probability. The quantile-envelope construction follows [conformalized quantile regression](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html); the day-block rank argument above states this experiment's actual target.

Limitations: the observed October shift and calendar effects leave day exchangeability uncertain. The fixed 16-station SBB cohort uses small remotely filtered archive prefixes; unresolved outcomes remain explicit, but scores condition on observed REAL or explicit canceled endpoints. All six confirmation dates have 524 planned connections and two missing endpoints; October 13 also has six explicit canceled endpoints and October 16 one SKIPPED. Marginal smoothed CDFs do not define a coherent ordered joint-trip model and have not replaced the routing belief. The original joint Evolving October 25 endpoint coverage remains 85.39%; the independent-date passes do not close that failure or give every-date coverage. Interval envelopes do not modify decision probabilities. The favorable new-date result therefore supports this scoped marginal predictor, not a full-city or universal calibration claim. Online actual-label adaptation would require a documented observation stream; a retrospective full-day REAL file does not provide one.

Reproduce from existing small artifacts: `python3 -m unittest experiments.reconstruction_v5.calibration_test`; then run `calibration_run.py`, `calibration_probability.py`, `calibration_confirm.py`, `calibration_secondary.py`, and `calibration_diagnose.py` (also with `--confirmation`) in this directory. `calibration_prepare.py` recreates public receipts from existing local archives; no new download is needed. The confirmation evaluator only loads frozen fits, offsets and probability maps and contains no refitting step.

The later [joint candidate](joint_day_v2_RESULTS.md) implements a common finite day-offset mixture of ordered trip beliefs, with a matching-marginal independent-trip offset control. It is a separate development experiment, not an extension of the six independent-date claims above. Oct25 prior endpoint coverage reaches 92.65% at 602.4 s width, while incoming-known risk and nominal policy results remain mixed. In the completed computation-charged comparison, shared rollout is 74.17 s slower than the inexpensive Gaussian comparator on six common journeys. The earlier empirical CDFs were not silently substituted into old conditional branch values.
