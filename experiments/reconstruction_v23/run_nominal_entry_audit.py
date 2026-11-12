"""Isolate the pre-entry departure mass on each frozen menu and grid."""

from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import replace
from math import fsum
from pathlib import Path
import json

from experiments.reconstruction_v13.factored_run import CASES
from experiments.reconstruction_v15.run_attribution_banks import prepare
from experiments.reconstruction_v18.run_policy_pair import root_menu, support_additions
from experiments.reconstruction_v22.run_offer_audit import GRIDS, OfferAudit, WAIT
from src.public_wait_recourse import through_itineraries


def compare(case):
    graph, belief, root, goal, deadline, plan, parameters = prepare(case)
    original, extras = root_menu(graph, root, goal, deadline, plan)
    extras.update(support_additions(graph, belief, parameters, root, goal, deadline, original, extras))
    trips = {belief._trip_by_cid[cid] for cid in original.keys() | through_itineraries(graph, extras).keys()}
    out = []
    for step in GRIDS:
        law = belief if step is None else belief.with_grid_step(step, trips=trips)
        audit = OfferAudit(law, parameters, plan, root, goal, deadline, original, extras)
        corrected = audit.nominal
        old = copy(corrected)
        # A window start before all fitted support reproduces the original denominator.
        old.root = replace(root, local_since=-10**9)
        _, old.nominal_wait = old._compile_marks(law, parameters)
        old.root = root
        old_waits, switches = [], []
        rows = 0
        for history, mass in audit.first_offers():
            rows += 1
            tick = history[-1][0]
            state = audit.state(tick, history)
            phase = tick-root.time
            visible = tuple(old.option_index[cid] for cid in state.visible_connections)
            seen = old._seen(state)
            before = old._nominal(state, visible, seen)
            after = corrected._nominal(state, visible, seen)
            if before is not None:
                assert before == after
                continue
            active = tuple(cid for cid in audit.cids if old.option_index[cid] not in seen)
            action = WAIT if after is None else ('board', old.options[after])
            old_waits.append((tick, active, action, mass))
            if after is not None:
                switches.append(mass)
        audit.nominal = old
        before = [audit.value('nominal', WAIT, t, active) for t, active, _, _ in old_waits]
        audit.continue_waiting.cache_clear()
        audit.nominal = corrected
        after = [audit.value('nominal', action, t, active) for t, active, action, _ in old_waits]
        audit.later_offers.cache_clear()
        audit.continue_waiting.cache_clear()
        pre_entry = {}
        for cid in audit.cids:
            departures, weights, cancel = law.event_distribution(cid, 'departure', parameters)
            mass = (1-cancel)*weights[departures < root.local_since].sum()
            if mass > .001:
                pre_entry[str(cid)] = float(mass)
        out.append(dict(grid_step=step, histories=rows, pre_entry_mass=pre_entry,
                        old_wait_mass=fsum(w[3] for w in old_waits), changed_histories=len(switches),
                        changed_mass=fsum(switches), loss_reduction=fsum(w[3]*(a-b) for w,a,b in
                                                                        zip(old_waits,before,after,strict=True))))
    return dict(case=case, root_time=root.time, local_since=root.local_since, grids=out)


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=3) as workers:
        results = list(workers.map(compare, CASES))
    Path(__file__).with_name('nominal_entry_audit.json').write_text(json.dumps(results, indent=2) + '\n')
