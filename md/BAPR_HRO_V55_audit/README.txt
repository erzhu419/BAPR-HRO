BAPR-HRO V55 audit — a2b82be9
================================

Start with deliverable/V55_diagnosis_and_plan.pdf (Chinese, complete diagnosis and plan).
The identical plain-text content is in deliverable/V55_diagnosis_and_plan.txt.
deliverable/NEXT_STEPS.json is a proposed, unexecuted next-stage specification.
build_report.py regenerates the PDF using reportlab.

Scope
-----
Pinned source commit: a2b82be9c3fc6b61e2a5826d7cd8691d39ff4491.
Original repository: https://github.com/erzhu419/BAPR-HRO
Audit date: 2026-11-15.
The source repository was not modified. The entire repository is not bundled.

Independent evidence
--------------------
metrics/:
  Read-only recomputation of 4400 archived V53–V55 queries and 576 V51 journey receipts.
  metrics.json, paired_raw_recomputed.csv, optimal_cpu_attribution.json and recompute.py.

theory_code/:
  Theory/implementation audit, original-counter classification and two original-cap cProfiles.
  Targeted V55 testing: 156 passed. Full 1412-test suite was not rerun.
  Profiles include instrumentation cost and are diagnostic, not new speed benchmarks.

strategy/:
  136 original gates, 13 changed actions / 11 roots; developmental, selected-bank summaries.
  12 changed gates map into the original 88-state panel; one additional root15 gate is separate.
  Five roots × two first actions × live/frozen future remote messages = 20 new diagnostic replays.
  Three negative roots × 128 original worlds × two arms = 768 reproduced loss/status outcomes.
  Root4 retrospective menu/observation probe = 3 additional replays.
  The root4 expanded action arm changes the action/observation domain and is not a general
  new policy or a same-domain V55 comparison.

The final PDF incorporates the last root4 probe. Individual sub-audits may discuss
earlier provisional explanations; their raw evidence and qualifications are preserved.

Reproduction
------------
Check out the exact source commit in a separate clone/worktree. Install its declared
dependencies. Set a task-specific path variable; do not change HOME.

  export BAPR_HRO_REPO=/absolute/path/to/BAPR-HRO-at-a2b82be9
  export OPENBLAS_NUM_THREADS=1
  export OMP_NUM_THREADS=1

Run from this audit package directory. Scripts write their derived outputs beside themselves;
copy the package first if you want to preserve the exact supplied receipts.

  python3 metrics/recompute.py
  python3 theory_code/counter_audit.py
  python3 strategy/extract_existing_banks.py
  python3 strategy/extract_actual_mechanisms.py
  python3 strategy/coverage_checks.py

Optional diagnostic replays (recorded outputs are already included):

  python3 strategy/frozen_live_probe.py
  python3 strategy/reconstruct_catch_events.py
  python3 strategy/root4_menu_probe.py
  python3 theory_code/profile_queries.py

These are not the original full V55 experiment runners. Timing is environment dependent.
The original V55 reproduction commands remain in the pinned repository RESULTS.md.

Important interpretation
------------------------
- A wide interval is uncertainty about a value, not a worse travel-loss outcome.
- A first-action optimum certificate is not a certificate that cheap's full policy is optimal.
- An improved feasible policy can witness a gap without solving full V*.
- New simulation seeds at old states are not external validation on new operational days.
- Future-message and menu probes are case-specific, retrospective diagnostics.
- Existing replay uses declared station transfer bounds and an ideal logical publisher;
  it is not a real hard-wall-time or observed-passenger experiment.
- Nothing in this package certifies that G2 has passed.

SHA-256 hashes of packaged evidence are in manifest.json.

