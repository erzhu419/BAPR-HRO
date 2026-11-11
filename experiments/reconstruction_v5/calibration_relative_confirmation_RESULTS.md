# Frozen incoming-known confirmation

The October 16/20/31 protocol was saved before retrieving actual outcomes. Models, β=.1989459021, Cauchy scale 43.5290349 s and both menu-specific interval offsets remained frozen. The server read a 1 MiB compressed range per date; local transfer was 42,290 bytes of filtered gzip plus three small audits. All 524 scheduled edges/date remain represented. October 16 has 1,045 REAL endpoints, two missing and one SKIPPED; October 20/31 each have 1,046 REAL and two missing. This is the fixed 16-station SBB morning cohort, not a complete-city source.

Primary results use schedule-selected pairs and known incoming arrivals; widths, interval scores and MAE are seconds. The same rider arrival and receipt-available RT information is available to every comparator.

| Date / primary pairs | Evolving coverage / width / Brier | Shrinkage coverage / width / Brier | Heavy-tail coverage / width / Brier |
|---|---:|---:|---:|
| Oct 16 / 2,898 | 98.41% / 495.7 / .06601 | 98.83% / 395.9 / .05061 | 99.14% / 459.8 / .05750 |
| Oct 20 / 2,900 | 99.41% / 495.6 / .06740 | 98.45% / 395.9 / .05082 | 98.28% / 462.3 / .06230 |
| Oct 31 / 2,900 | 96.97% / 495.6 / .08341 | 97.48% / 395.9 / .08062 | 97.17% / 474.2 / .08282 |

Shrinkage interval scores are 405.4/457.3/431.3 versus Evolving 516.4/540.6/545.2; MAE is 49.4/49.7/66.2 versus 55.2/52.7/69.2. It improves these scores and Brier against Evolving on all three new dates. Against the already frozen public mixture, however, shrinkage intervals are wider and interval scores worse on every date; Brier improves on two dates and worsens on October 31 (.08062 versus .07924). Date-average Brier is .06068 versus mixture .06175 and Evolving .07227. Three dates do not establish a general superiority result.

The heavy-tail law's MAE is 55.5/61.1/72.5, worse than Evolving each day; coefficient-one relative Brier also deteriorates every day. Actual-current menu and selected-action readouts, censor counts, incoming-delay strata and numerical diagnostics are preserved in `calibration_relative_confirmation_summary.json`, `calibration_relative_confirmation_results.json.gz` and `calibration_relative_confirmation_diagnosis.json`. None is a deployed journey-policy comparison.

Limitations: the original joint Evolving model's October 25 endpoint coverage remains 85.39%. Shrinkage and heavy-tail still cover only 86.84% and 88.97% of October 25 primary pairs. Their new-date passes do not close that failure, establish every-date coverage, or repair the joint continuation belief. All scored new-date pairs have REAL target departures, so this confirmation does not validate cancellation probabilities. No further dates or parameter tuning follow this readout. See `calibration_CONTINUATION_INTERFACE.md` for the same-joint-law requirement before connecting a probability to conditional tail values.

Reproduce the small source join with `calibration_relative_prepare_confirmation.py`, then evaluate with `calibration_relative_confirm.py` and diagnose its stored output with `calibration_relative_diagnose.py --input experiments/reconstruction_v5/calibration_relative_confirmation_results.json.gz`. The evaluator contains no refitting step. API: `model.predict(incoming_source, outgoing_source, observed_arrival)` returns `mean`, `cdf`, `quantile`, `cancel_probability` and `board_probability`; these are local second-clock predictions, not an `EvolvingTripBelief` replacement.
