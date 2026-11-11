# Rider-observed relative delay

October 1–2 alone fit the distribution of departure delay minus the rider's observed incoming arrival delay, after frozen route/event-kind centering. The incoming coefficient is fixed at one, route-pair residual laws shrink to pooled residuals with 20 pseudo-observations, and the kernel is the existing 17.32 s. October 3–11 calibrate each method's interval envelope separately. The prior mixture and its independent-confirmation conclusions remain unchanged.

Primary pairs use **scheduled** incoming arrival plus official station transfer and different-trip scheduled departure within ±5 min; this menu is fixed before actual outcomes. After the incoming leg is measured, only its arrival and RT receipts available by that clock enter the predictor. Both Evolving controls explicitly receive that arrival; their different-trip independence prevents cross-trip updating. Secondary pairs reproduce the earlier actual-ready ±5 min critical-state menu. Unknown outgoing outcomes remain candidates, including during greedy probability selection, and are censored after selection.

| Development date | Primary REAL / known outcomes | Evolving coverage / width / Brier | Relative coverage / width / Brier |
|---|---:|---:|---:|
| Oct 12 | 2,900 / 2,900 | 98.38% / 495.6 s / .06862 | 98.59% / 594.6 s / .07840 |
| Oct 25 | 2,765 / 2,796 | 86.08% / 500.1 s / .13211 | 90.96% / 594.9 s / .14113 |
| Oct 27 | 2,856 / 2,859 | 91.49% / 495.4 s / .11188 | 96.36% / 595.0 s / .11759 |

The coverage threshold is reached on October 25, but this is **not a demonstrated predictive improvement**: relative raw coverage is 71.93%, its added envelope is 127.4 s, interval score 1,140.1 versus 969.1, and MAE 139.6 versus 122.4 s. Direct Evolving+public forecasts already obtains 91.93% at width 559.0 s and interval score 913.9, though its Brier .13801 also deteriorates. The original frozen mixture covers 86.15% at 367.5 s with Brier .13849 on these primary states.

The shared-location mechanism is visible in averages: October 25 incoming/outgoing delays average 181.4/196.7 s; their difference averages 15.3 s, close to October 12's 18.2 s. Individual pairs are weakly correlated (.037), however, and relative-delay SD rises from 93.9 to 227.6 s. Following one incoming delay at coefficient one adds substantial individual noise. The October 25 incoming-delay >180 s subgroup covers only 86.49%, so the aggregate result does not establish coverage for every observed incoming delay.

Selection matters. October 25's actual-current menu adds 627 and removes 798 pairs; observed catch fraction changes 53.00% → 76.79%. Relative secondary coverage is 91.36% at 586.7 s, Brier .14237 versus Evolving .13556. On the primary greedy-selected subset, relative Brier is .07329 versus .09686, but the methods choose different actions and censor 3 versus 6 unknown selected outcomes; this cannot overturn the all-pair negative result or establish journey-policy benefit.

Theory: a common additive day location cancels only with equal incoming/outgoing loadings. Remaining relative residuals must be exchangeable under the same timetable/observation-selection mechanism. Because the residual contains minus the incoming idiosyncratic error, cancellation does not imply its distribution is independent of the observed incoming delay. Nine-date calibration supplies the same date-level tolerance statement described in `calibration_RESULTS.md`; it supplies neither arbitrary conditional coverage nor calibrated boarding probabilities. Cancellation remains a separate earlier route marginal, and all clock interval scores condition on REAL operating endpoints.

Reproduce: run `calibration_relative_run.py`, then `calibration_relative_diagnose.py`. Fit, offsets, complete rows and menu counts are retained in the correspondingly named frozen/summary/results/diagnosis artifacts. Nine receipt, probability, relative-shift and censoring tests pass across `calibration_test.py` and `calibration_relative_test.py`. No core-router or manuscript file was changed.
