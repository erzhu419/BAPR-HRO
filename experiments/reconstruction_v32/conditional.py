"""Shared-trip posterior control on the frozen overlap-platform catalogue.

Only public departures during the current platform visit condition the law.
These are forced-arrival platform probes, not complete root policies.
"""

from copy import copy
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v25.run import departure, physical_path, public_offers, route_outcome
from experiments.reconstruction_v31.model import SegmentDurationBelief
from experiments.reconstruction_v32.journey import TailMenu
from experiments.reconstruction_v32.run import actual_tail, board_first
from src.evolving_trip_belief import _ChainLaw
from src.full_menu_opportunity import FullMenuOpportunity
from src.plan_waiting_controls import SingleTargetOpportunity, CombinedWaitingOpportunity, TwoTargetOpportunity


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'reconstruction_v25'


class UnsupportedHistory(ValueError):
    pass


def departure_evidence(law, slot, entry, clock, observed=None):
    """Return a frozen-support law and receipt for one public departure slot.

    Off-grid observations use the declared adjacent-active-state posterior
    interpolation. Its likelihood is desired atom mass / prior atom mass,
    so unequal prior atom masses cannot bias the interpolation weights.
    """
    times, canceled, prior = law.times[slot], law.canceled[slot], law.marginal(slot)
    evidence = list(law.emissions)
    if observed is None:
        keep = canceled | (times < entry) | (times > clock)
        survival = float(prior[keep].sum())
        if survival <= 0:
            raise UnsupportedHistory('No frozen probability for the public nonoccurrence history')
        evidence[slot] = np.where(keep, evidence[slot], -np.inf)
        receipt = dict(kind='local_nonoccurrence', entry=entry, through=clock, prior_mass=survival)
    else:
        if not entry <= observed <= clock:
            raise ValueError('A departure receipt must belong to this public visit prefix')
        active = np.flatnonzero(~canceled & (prior > 0))
        position = int(np.searchsorted(times[active], observed))
        if position < len(active) and times[active[position]] == observed:
            indices, weights = np.array([active[position]]), np.array([1.])
            mode = 'on_grid'
        elif 0 < position < len(active):
            indices = active[position-1:position+1]
            right = (observed-times[indices[0]])/(times[indices[1]]-times[indices[0]])
            weights, mode = np.array([1.-right, right]), 'adjacent_active_projection'
        else:
            raise UnsupportedHistory('Observed departure lies outside frozen operating support')
        likelihood = np.full(len(prior), -np.inf)
        likelihood[indices] = np.log(weights)-np.log(prior[indices])
        evidence[slot] = evidence[slot]+likelihood
        receipt = dict(kind='observed_departure', clock=observed, mode=mode,
            support_clocks=times[indices].tolist(), posterior_weights=weights.tolist())
    result = _ChainLaw(law.times, law.initial, law._transition, tuple(evidence), law.canceled, law.stochastic)
    return result, receipt


def condition_platform(belief, parameters, cids, entry, clock, observed):
    """Copy the frozen belief interface, retaining one conditioned law/trip."""
    laws, receipts = {}, []
    for cid in cids:
        trip, slot = belief._positions[cid, 'departure']
        law = laws[trip] if trip in laws else belief._law(trip, parameters)
        laws[trip], receipt = departure_evidence(law, slot, entry, clock, observed.get(cid))
        receipts.append(dict(cid=cid, trip=trip, **receipt))
    conditioned = copy(belief)
    conditioned._law = lambda trip, params: laws[trip] if trip in laws else belief._law(trip, params)
    return conditioned, receipts


def conditioned_tail(menu, parameters, metadata, entry):
    ready = entry+menu.graph.stops[menu.hub['stop']].min_transfer_time
    offers = public_offers(metadata, menu.cids, entry, ready, menu.deadline)
    trace = []
    if offers is None:
        return None, 'unresolved_tail_departure', trace
    for public in offers:
        tick = perf_counter()
        # Only receipt IDs exposed by this public prefix are dereferenced.
        observed = {cid: departure(metadata[cid])[0] for cid in public['seen']}
        try:
            belief, evidence = condition_platform(menu.belief, parameters, menu.cids,
                                                   entry, public['clock'], observed)
        except UnsupportedHistory as error:
            trace.append(dict(**public, choice=None, status='unsupported_public_history', reason=str(error)))
            return None, 'unsupported_public_history', trace
        current = TailMenu(menu.graph, belief, parameters, menu.hub, menu.deadline)
        reference = FullMenuOpportunity(current, current.ranks, entry, ready, menu.deadline)
        control = TwoTargetOpportunity(CombinedWaitingOpportunity(SingleTargetOpportunity(reference)))
        choice, receipt = control.choose(public['visible'], public['seen'], public['clock'])
        trace.append(dict(**public, choice=choice, receipt=receipt, evidence=evidence,
                          condition_compile_choose_wall=perf_counter()-tick))
        if choice is None:
            continue
        score, index = current.best(choice, public['clock'])
        if index is None:
            return None, 'unsupported_tail_board_clock', trace
        option = current.compilers[index].options[choice]
        trace[-1].update(variant=index, first_path=option.first_path, board_score=score)
        arrival, status, tail = route_outcome(menu.graph, metadata, option, public['clock'], menu.deadline)
        trace[-1]['final_tail'] = tail
        return arrival, status, trace
    return menu.deadline, 'no_tail_board', trace


def main():
    started = perf_counter()
    selection = json.loads((HERE/'selection_full_hubs.json').read_text())
    fit = json.loads((SOURCE/'model_fit.json').read_text())
    segment = json.loads((HERE.parent/'reconstruction_v31/segment_fit.json').read_text())
    data, _ = load_city(SOURCE/'city_transfer_endpoints.json.gz')
    days, rows = {}, []
    for root in selection['selected']:
        targets = [(stage, hub) for stage, hub in enumerate(root['hubs']) if hub['root_tail_overlap']]
        if not targets:
            continue
        day = root['day']
        if day not in days:
            graph, _, ids, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
            belief = SegmentDurationBelief(graph, duration_fit=fit['duration_fit'], segment_fit=segment,
                stop_identities={v: k for k, v in ids.items()}, public_occurrences=False,
                grid_step_seconds=20, **fit['increments'])
            parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                          for c in graph.connections}
            days[day] = graph, metadata, belief, parameters
        graph, metadata, belief, parameters = days[day]
        board, board_status = board_first(graph, metadata, root)
        for stage, hub in targets:
            tick = perf_counter()
            entry, status = ((board, board_status) if board_status != 'boarded' else
                            physical_path(metadata, hub['prefix'], board, root['deadline']))
            row = dict(root_id=root['root_id'], day=day, stage=stage, stop=hub['stop'],
                overlapping_trips=hub['root_tail_overlap'], entry=entry, prefix_status=status)
            if status == 'arrived':
                menu = TailMenu(graph, belief, parameters, hub, root['deadline'])
                outcomes = {}
                for method in ('fixed_pair', 'conditioned_pair'):
                    begin = perf_counter()
                    arrival, state, trace = (actual_tail(menu, metadata, entry, 'pair')
                        if method == 'fixed_pair' else conditioned_tail(menu, parameters, metadata, entry))
                    outcomes[method] = dict(arrival=arrival, loss=None if arrival is None else arrival-entry,
                        status=state, trace=trace, wall=perf_counter()-begin)
                row['outcomes'] = outcomes
                complete = all(v['loss'] is not None for v in outcomes.values())
                row['paired_gain'] = outcomes['fixed_pair']['loss']-outcomes['conditioned_pair']['loss'] if complete else None
                traces = [{event['clock']: (event['choice'], event.get('variant')) for event in value['trace']}
                          for value in outcomes.values()]
                row['changed_common_decisions'] = [clock for clock in traces[0].keys() & traces[1].keys()
                                                   if traces[0][clock] != traces[1][clock]]
            row['wall'] = perf_counter()-tick
            rows.append(row)
            print(root['root_id'], stage, status, row.get('paired_gain'), flush=True)
    paired = [r for r in rows if r.get('paired_gain') is not None]
    summary = dict(overlap_hubs=len(rows), prefix_complete=sum(r['prefix_status'] == 'arrived' for r in rows),
        paired_complete=len(paired), wins=sum(r['paired_gain'] > 0 for r in paired),
        harms=sum(r['paired_gain'] < 0 for r in paired), ties=sum(r['paired_gain'] == 0 for r in paired),
        mean_gain=float(np.mean([r['paired_gain'] for r in paired])) if paired else None,
        changed_hubs=sum(bool(r.get('changed_common_decisions')) for r in rows),
        total_wall=perf_counter()-started)
    output = dict(scope='Forced-arrival overlap platforms; public departure posterior probe, not a complete root policy.',
                  information='Own fixed prefix arrival and local root departures since entry; no remote nonoccurrence evidence.',
                  summary=summary, rows=rows)
    (HERE/'conditional.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
