# Final 5-Day Bounded Experiment (`bounded_gate_5d_v1`) — Complete Results

- **Frozen Dates**: `2023-04-06, 2023-04-07, 2023-04-08, 2023-04-09, 2023-04-10` (`60` public roots, `480` paired arm executions, `0` additional budget on original 72 roots)
- **Audit Status**: `all_passed = False` (`72` prior exposed dates excluded; chronological pre-outcome freezes verified)
- **Pre-Declared Verdict**: **`CONFIRM_STOPPING_RULE_PATH_B`**

## 1. Arm Quality & Computation Summary

| Arm | Complete / 60 | Mean Loss (Complete, s) | All-60 Bounds (s) | Mean CPU (s) | Max CPU (s) | Gates |
|---|---:|---:|---:|---:|---:|---:|
| `cheap_statmax_ideal` | `59/60` | `1241.80` | `[1221.10, 1281.10]` | `0.394` | `0.967` | `0` |
| `full128_statmax_ideal` | `59/60` | `1231.20` | `[1210.68, 1270.68]` | `16.731` | `48.780` | `124` |
| `cheap_plat_async` | `58/60` | `1224.16` | `[1183.35, 1303.35]` | `0.418` | `1.052` | `0` |
| `snapshot_csa_plat_async` | `58/60` | `1234.74` | `[1193.58, 1313.58]` | `0.159` | `0.477` | `0` |
| `recursive_plat_async` | `58/60` | `1247.76` | `[1206.17, 1326.17]` | `24.375` | `89.029` | `0` |
| `bounded_cert_plat_async` | `58/60` | `1252.40` | `[1210.65, 1330.65]` | `29.629` | `142.923` | `24` |
| `full128_plat_ideal` | `58/60` | `1218.91` | `[1178.28, 1298.28]` | `18.101` | `47.150` | `119` |
| `full128_plat_async` | `58/60` | `1218.91` | `[1178.28, 1298.28]` | `17.485` | `47.340` | `119` |

## 2. Key Paired Comparisons (Positive Saving = Subject Beats Baseline)

| Baseline | Subject | Wins / Losses / Ties | Mean Saving (s) | 95% Date-Bootstrap CI (s) | Mean CPU Delta (s) |
|---|---|---:|---:|---:|---:|
| `cheap_statmax_ideal` | `cheap_plat_async` | `3 / 1 / 54` | `+10.59` | `[-10.67, +25.83]` | `+0.024` |
| `cheap_statmax_ideal` | `full128_statmax_ideal` | `2 / 0 / 57` | `+10.59` | `[+0.00, +21.72]` | `+16.337` |
| `cheap_plat_async` | `snapshot_csa_plat_async` | `1 / 3 / 54` | `-10.59` | `[-25.83, +10.67]` | `-0.259` |
| `cheap_plat_async` | `recursive_plat_async` | `0 / 2 / 56` | `-23.60` | `[-50.81, +0.00]` | `+23.957` |
| `cheap_plat_async` | `bounded_cert_plat_async` | `0 / 3 / 55` | `-28.24` | `[-49.52, -8.38]` | `+29.211` |
| `recursive_plat_async` | `bounded_cert_plat_async` | `1 / 1 / 56` | `-4.64` | `[-27.75, +14.30]` | `+5.254` |
| `snapshot_csa_plat_async` | `bounded_cert_plat_async` | `3 / 4 / 51` | `-17.66` | `[-55.27, +17.45]` | `+29.470` |
| `full128_plat_ideal` | `full128_plat_async` | `0 / 0 / 58` | `+0.00` | `[+0.00, +0.00]` | `-0.616` |
| `cheap_plat_async` | `full128_plat_ideal` | `1 / 0 / 57` | `+5.24` | `[+0.00, +15.72]` | `+17.683` |
| `cheap_plat_async` | `full128_plat_async` | `1 / 0 / 57` | `+5.24` | `[+0.00, +15.72]` | `+17.067` |
| `full128_plat_async` | `bounded_cert_plat_async` | `0 / 4 / 54` | `-33.48` | `[-60.38, -8.38]` | `+12.144` |

## 3. G1 Contract, Physical Dwell Expiration & Conditional Calibration

- **G1 Platform-Pair MCT Effect (`cheap_plat_async` vs `cheap_statmax_ideal`)**: mean saving `+10.59s` (`3` wins, `1` losses). Static + same-platform coverage = `100.0%`; GTFS-RT platform agreement = `94.57%`.
- **Physical Dwell Reality (`ist-daten` `[AN_PROGNOSE, AB_PROGNOSE]`)**: median continuation station dwell = `59.0s` (`p10 = 28.0s`, `p90 = 145.0s`).
  - `full128_plat_async`: `0 / 119` (`0.0%`) rollout gates expired AFTER physical train departure!
  - `bounded_cert_plat_async`: `0 / 24` (`0.0%`) expired (`mean CPU = 29.629s <= 2.0s`).
- **Calibration Split (`C1` vs `C2`)**: `C1` reached `410 / 410` opportunities (`0` canceled initial roots); `C2` conditional `P10` residual = `-0.0051`, 4-class Brier = `0.0880` (Brier skill = `+50.5%` vs climatology).
