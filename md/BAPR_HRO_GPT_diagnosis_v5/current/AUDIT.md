# Read-only audit of 86ba89c: current V30/V31 and finite-offer residual

Audited commit: `86ba89c327e5230f1290e8678bebcb589b2465a9`.

## 1. What V31 actually changes

`experiments/reconstruction_v31/model.py` changes the conditional elapsed-time transition kernel by adding physical station-segment identity to the existing route/event-kind/scheduled-duration group. Its fixed shrinkage is n/(n+20), preserving the existing route law as the other component. Stable physical stop IDs are mapped across date-local graph IDs. The same July25 records and eligibility rules are used. Initial/service laws, public information, 20s grid, fixed route menus, goal, and cheap control implementations are unchanged.

`run.py:admission` is explicitly a development model-retention gate. It requires CRPS nonworsening overall and on each developed date, target-appearance Brier nonworsening, no label/support loss, and origin single-target mean loss/arrivals nonworsening. It does not require any changed decision, any strictly positive algorithmic gain, or G2 passage. A tie in all actual policy outcomes is therefore an expected possible retention result, not a gate implementation bug.

The new model is connected to actual policy scoring: `run.py` invokes the existing V28 `run_root` with the new belief and scores do change. The V31 saved data also include the strongest combined policy, though its protocol and headline retention focus on one-step and single-target. Neither a new BOARD policy nor a new WAIT optimizer is implemented here.

## 2. Independent reproduction of action-margin diagnosis

Input: saved V28 `route_empirical20` policy traces and saved V31 policy traces; align on root, public clock, visible set and seen set. Exclude direct boarding without an unseen candidate. Raw output: `action_margins.json`.

| Policy | All decisions | Substantive BOARD/WAIT comparisons | Median absolute V31 BOARD−WAIT score | Mean absolute change from V28 | Actual choice changes | Internal target changes |
|---|---:|---:|---:|---:|---:|---:|
| one_step_nominal | 111 | 93 | 313.58794 s | 22.10863 s | 0 | 0 |
| single_target_nominal | 166 | 148 | 399.37990 s | 27.78798 s | 0 | 4 |
| combined_wait_nominal | 166 | 148 | 400.35554 s | 27.44829 s | 0 | 4 |

This exactly reproduces the attachment's 93/148 rows and extends it to the required strongest combined baseline. The gap between average action-value perturbation and existing margins explains why CRPS improvement need not improve travel loss. This is verified by every unchanged choice, not inferred only from medians.

Two fixed-action integrated pair orderings flip in V31 diagnostics, both pairs chosen by `original20`, in Monday OD0 transfer histories at clocks31256 and32916. This is not a new deployed rolling-policy improvement: the empirical rolling policy already waits toward another available target. The six retained V29 discrepancies still have two actual wins and four losses. Fixed-action diagnostic rankings and actual rolling decisions must remain separate.

## 3. Newly executed natural-small-menu probe

The audit adds an exploratory calculation, not a new repository stage or confirmatory experiment. The probe uses the newest segment law and **every naturally occurring nonempty frozen V25 menu with K≤8**. It selects no roots by realized outcome or predicted gap. This yields61 roots:42 origins and19 conditional first-transfer roots. Six empty menus and six Monday OD0 origin menus of K17/18 are outside this exact computation.

For each root, evaluate the exact finite-offer public-event DP and independently integrate the existing first-offer, one-step, single-target, and strongest combined policies on the full supported independent departure laws. No finite sampled world library is used; no future event is revealed to the policy. Two terminal-value definitions are separated:

1. The deployed clock-nominal BOARD marks.
2. Complete conditional expected costs of the **same fixed first path and same fixed first-offer tail continuation**, using `TransferOpportunityScores.board_score(...).exact`. This remains a restricted continuation, not adaptive full-network BOARD.

The second version holds BOARD costs identical across its cheap controls and exact WAIT DP, so differences isolate repeated-wait optimization rather than an information or objective mismatch. On actual off-grid event clocks it uses the repository's supported linear score projection, explicitly not an exact continuous posterior claim.

Implementation: `probe_small.py`. Raw results: `probe_segment_nominal_k1-8.json` and `probe_segment_full_k1-8.json`. Full laws remain frozen; actual records enter only the replay after action rules have been defined. The independent integration reproduces the DP recursion within5.7e−10s. Deployed threshold formula checks are within7.3e−12s. The production CDF formula is preserved in deployed cheap thresholds, rather than silently replacing it by an idealized cheap rule. A residual discrepancy from exact first-policy integration was specifically investigated on the selected root below; its cause is not generalized to all uninvestigated roots.

### Exact model gaps: combined policy expected cost minus optimal finite-offer DP cost

| Population | n | Nominal mean gap | Full BOARD-cost mean gap | Nominal maximum | Full BOARD-cost maximum |
|---|---:|---:|---:|---:|---:|
| Origin roots | 42 | 0.555942s | 0.435317s | 5.536835s | 4.322015s |
| Conditional transfer roots | 19 | 0 | 0 | 0 | 0 |
| All small nonempty roots | 61 | 0.382780s | 0.299726s | 5.536835s | 4.322015s |

All61 actual trajectories have the same selected first vehicle, tail, loss, arrival status and physical status under combined and exact DP. Also changing nominal to complete conditional BOARD costs changes no actual action or loss for one-step, single-target, combined or DP on these61 roots. **There is no actual paired gain in this new probe.**

The only substantive nonzero root gaps occur on the six Monday OD2 K8 origins:

| Start clock | Nominal combined gap | Full BOARD-cost combined gap | Actual combined / DP loss |
|---:|---:|---:|---:|
| 27600 | 0.108653s | 0.407989s | 1319 / 1319s |
| 28500 | 5.235279s | 4.072501s | 1476 / 1476s |
| 29400 | 5.159113s | 4.082102s | 1389 / 1389s |
| 30300 | 5.323844s | 3.888860s | 1391 / 1391s |
| 31200 | 5.536835s | 4.322015s | 1428 / 1428s |
| 32100 | 1.984171s | 1.507638s | 1488 / 1488s |

Five of the old six SundayOD0 roots have gaps about0.000336s nominal /0.000439s full; the other old root has zero. The other49 roots have zero combined root gap. Thus saying "single-target is globally optimal everywhere" would overstate the evidence, but planning a deep WAIT reconstruction for the current empirical application would also overstate these small, unrealized gains.

The largest gap over all mathematically supported masks/clocks is13.535951s nominal and11.753751s full, with some states not reached by the compared policy from the root. These conditional maxima must not be substituted for root-level expected performance.

## 4. Interpretation and what to do

There is a real nonanticipative finite-model mechanism in existing natural menus: a portfolio of future opportunities can justify rejecting an offer that beats every fixed single-target waiting plan and the first-future-offer plan. Yet the current magnitude is a few seconds on one developed day/OD block and zero paired actual gain. This is a candidate mechanism, not evidence for an OR-grade algorithmic contribution or model calibration.

Before claiming that full subset DP is needed, add the cheap two-target rolling control, allowing the lower of existing combined value and exact restricted-two-target wait value. This can be computed through all O(K²) pairs of tiny DPs while reusing the current duration/offer laws. **This audit executed that control on the selected largest full-cost-gap root, MondayOD2 start31200: its root gap to exact DP is0.000000s.** It absorbs the entire4.322015s residual on this root. No blanket statement about all remaining roots is inferred from that one selected mechanism check. This is a stronger cheap baseline and diagnostic, not evidence justifying a large DP. Raw result: `residual_full_2023-07-17_2_31200_origin.json`; code: `diagnose_residual.py`.

### 4.1 One fully attributed public decision gap

This is a positive-probability **model public state**, not an actually encountered archive trajectory. It is chosen after observing the residual to explain mechanism, so cannot provide independent validation.

- Root `2023-07-17_2_31200_origin`; public clock31440 (08:44); visible departure17351; seen departures17351 and17058. Six remaining opportunities are16619,16097,17512,18213,17521,17908.
- Current fixed continuation17351 has a transfer; future16619 is a direct14-connection first path;17512 has a different downstream transfer. These are pre-existing natural frozen options, not constructed synthetic clocks.
- Conditional expected absolute arrival costs: current BOARD32956.55056; best fixed-target WAIT32989.74832; first-future-offer WAIT33014.25844; combined WAIT32989.74832; exact repeated WAIT32907.24965.
- Combined therefore boards; exact DP waits. Best pair16619/17512 yields WAIT32916.24118, already sufficient to reverse the cheap decision to WAIT. At each subsequent public offer, the pair control may reconsider all pairs; it is not an irrevocable commitment to this pair.
- The aggregated public state's probability under combined is0.003942617400252421. Boarding's conditional regret is49.3009085493s. Its root-level contribution is0.1943746199s.
- Sum of occupancy×Bellman action loss over1342 positive-contribution public states is4.32201502389485s, matching independently integrated root gap4.322015023892163s. Their total occupancy mass is0.12353086940332812. Thus the residual is genuine premature boarding because of future-option complementarity, not a clairvoyant lower-bound artifact.
- Root expected arrival: combined32690.0012215; full DP32685.6792065; rolling pair32685.6792065. Actual archive: direct16619 appears at31292 (08:41:32), and all three policies board immediately, yielding1428s loss. The adverse delayed-direct state above does not occur in this actual trajectory.

The pair's conditional WAIT value remains about9s above the full DP's at that one state; nevertheless it crosses the correct action boundary, and its subsequent rolling decisions produce exact root-optimal value. This is a useful demonstration of why matching a sophisticated value function is unnecessary when a cheap approximation already produces the right decisions.

### 4.2 Focused numerical audit of the deployed first-offer formula

The production `FullMenuOpportunity.wait_score` subtracts two accumulated CDF values and divides by each candidate's conditional survival. A discrepancy between that formula and independently integrated first-policy values cannot simply be dismissed as numerical without checking. The selected full-cost root was therefore traversed over **all positive-probability public states reachable under combined**.

The largest first-value discrepancy is34.894375871s; the largest resulting combined threshold difference is13.307210235s. However, replacing the first-value formula by its exact first-policy recursion changes **zero actions**, with changed state probability mass0 and changed Bellman action-loss contribution0. Thus the entire4.322015s root residual and pair-control result remain unchanged.

For every checked reachable discrepancy exceeding1e−5s, the same first-policy value was also computed by summing original event probabilities in the interval directly, avoiding accumulated-CDF subtraction. The maximum direct-sum versus independent-recursion difference is1.4552e−11s. This confirms floating-point CDF cancellation rather than a policy-semantic discrepancy **for this root**.

Largest example: clock32280, visible16619, remaining candidates17512/18213. Candidate17512's survival is5.6030526191e−17; the public state's probability is2.0371469447e−38. Production first-value33546.162990945, direct sum and recursion33581.057366816; single-value33559.47020118, BOARD33533.96449472. Both versions of combined BOARD. Earlier aggregate probe `max_deployed_first_formula_recursion_difference` tested only deterministic representative masks; the full reachable-state check naturally discovers a larger maximum.

This numerical issue deserves a small stable-probability implementation fix and an adversarial-tail regression test if that formula is used operationally. It is not the cause of the present selected-root decision gap and does not justify another G2 research cycle. No claim is made about all other roots' CDF discrepancies without equivalent checking.

## 5. Runtime and exact numerical ledger

These are local audit wall times, not deployable online performance benchmarks. Complete61-root nominal run94.916s; complete61-root full-BOARD-cost run141.642s. Selected-root full-value pair-control/Bellman probe18.210s; focused numerical rerun42.248s. The saved aggregate JSON timestamps are immediately before final save/print (94.8885/141.6253s), explaining millisecond differences. The introductory one-root timing check took5.849s and is not another independent experiment.

Exact aggregate model gaps, means computed across frozen roots (not a statistical estimator with independent-root confidence intervals):

- Nominal combined: all61 mean0.3827799039963942s;42origin mean0.5559422415185725s;19transfer mean0; maximum root5.536834637277934s; maximum supported-state13.535951100588136s.
- Full BOARD-cost combined: all61 mean0.2997262732179017s;42origin mean0.4353167301498096s;19transfer mean0; maximum root4.322015023892163s; maximum supported-state11.753751334261324s.
- Nominal independent DP integration error maximum5.6752469391e−10s; full maximum4.1109160520e−10s. Production first-wait threshold vectorization maximum checked error7.2759576142e−12s in both runs.
- Actual combined vsDP agreement61/61 on first action, tail, loss, arrival indicator, and status in each run. Nominal vsfull actual agreements also61/61 for each of one-step, single, combined andDP.

No conclusion here covers K17/18 roots, omitted correlated menus, adaptive alighting/BOARD, different information contracts, or new independent dates. No repeated windows or model states are counted as independent actual observations. The old G2 threshold for a policy lower-bound certificate is not the same quantity as a planner's gain over combined.

The V30/V31 work is technically coherent as model diagnostics. Its inability to pass G2 follows from (i) retention tests targeting prediction rather than decision contribution; (ii) already large BOARD/WAIT margins; (iii) nearly exhausted restricted WAIT opportunity after strong cheap baselines; and (iv) current menus/continuations omitting richer decisions. This does not show a broad execution failure.
