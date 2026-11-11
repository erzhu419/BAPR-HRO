# Fixed shrinkage and heavy-tail alternatives

Two additional candidates were fixed after the coefficient-one diagnostic, with parameters fitted only on October 1–2. Neither the earlier public-mixture candidate nor its independent confirmation was changed. Shrinkage fits the zero-intercept route-centered slope, clipped to [0,1]: β = .1989459021. The common-day alternative uses δ ~ Cauchy(0,43.5290349 s), where scale is the unscaled training incoming-residual MAD; independent incoming/outgoing role KDEs condition the local day posterior on the rider-observed arrival. Route shrinkage remains 20 and the kernel 17.32 s. Development dates remain October 12/25/27; the new confirmation outcomes are excluded from all fitting and calibration.

| October 25 fixed timetable pairs | Coverage | Mean width | Interval score | MAE | Brier |
|---|---:|---:|---:|---:|---:|
| Evolving, same incoming information | 86.08% | 500.1 s | 969.1 | 122.4 s | .13211 |
| Frozen public mixture | 86.15% | 367.5 s | 875.6 | 120.5 s | .13849 |
| Relative, coefficient one | 90.96% | 594.9 s | 1,140.1 | 139.6 s | .14113 |
| Training-fitted shrinkage | 86.84% | 396.5 s | 868.5 | 113.5 s | .12917 |
| Heavy-tail day effect | 88.97% | 500.4 s | 1,082.8 | 128.5 s | .13160 |

Shrinkage improves sharpness, MAE and proper scores over Evolving on October 25, but still misses the coverage target. The heavy-tail candidate also misses 90% and worsens interval score. The actual-current secondary menu gives shrinkage 87.31%/389.8 s/Brier .13060 and heavy-tail 89.62%/495.2 s/Brier .13472. Coefficient one reaches coverage by sacrificing other scores. None establishes a combined coverage, sharpness and probability-calibration repair, and no offset or Cauchy scale was enlarged to reach a development threshold.

Observable incoming strata retain failures: October 25's >180 s incoming-delay group has coverage/Brier 87.49%/.1238 under shrinkage and 88.21%/.1475 under the heavy-tail law. These are selected-pair/date-block empirical readouts, not guarantees conditional on every incoming value. The heavy-tail model assumes common equal-load day location and independent role noises given that location; separate empirical route noise already contains earlier day variability. Unknown and canceled outcomes retain the coefficient-one experiment's treatment and menus. No same-day all-network actual stream is used.

Numerical checks compare the fixed 7.5 s delta grid with 3.75 s on the smallest, median and largest training anchors: maximum interval endpoint difference .053 s and CDF difference .000177. One-second KDE tables are reused. Adaptive grid limits contain the fixed [-300,3840] s base, the observed anchor, and all training likelihood centers plus nine bandwidths. The largest unnormalized-tail-bound/grid-evidence ratio is 3.71e-16; evidence is approximate quadrature, so this is a numerical diagnostic rather than a certified posterior coverage claim. Gaussian-KDE interpolation, learned slope, heavy-tail anchor response, cancellation mass and target-future blindness have dedicated passing tests.

Reproduce the development readout with `calibration_relative_variants.py`, then `calibration_relative_diagnose.py --input experiments/reconstruction_v5/calibration_relative_variants_results.json.gz` from the repository root. The `*_frozen_fit.json`, `*_frozen.pkl`, `*_frozen_offsets.json`, `*_numerics.json`, `*_summary.json` and compressed full results preserve all controls and negative results.

That predeclared confirmation is now complete without refitting: shrinkage coverage is 98.83/98.45/97.48%, width 395.9 s and Brier .05061/.05082/.08062 on October 16/20/31. Evolving Brier is .06601/.06740/.08341. The frozen public mixture remains competitive at .05337/.05263/.07924 and has narrower intervals and lower interval scores on every date. Heavy-tail coverage is 99.14/98.28/97.17%, width 459.8/462.3/474.2 s and Brier .05750/.06230/.08282; its MAE is worse than Evolving all three days. Coefficient-one Brier deteriorates all three days.

See [the fixed confirmation results](calibration_relative_confirmation_RESULTS.md) and [the continuation interface](calibration_CONTINUATION_INTERFACE.md). The earlier independent October 13/26/30 dates confirm the earlier public mixture; they were already seen before these new variants and cannot be added to this candidate's independent count. Shrinkage improves local risk estimates against Evolving on three new dates, but the October 25 shortfall and the coherent-joint integration remain open. No further dates or parameter tuning were performed.
