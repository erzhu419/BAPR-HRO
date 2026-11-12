"""Local grid sensitivity on the already selected near-critical actual windows."""

from collections import Counter
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v22.run_natural_screen import (
    PUBLICATION, ROOT, SOURCES, exact_shadow, fixed_arrival, replay_direct, summarize_shadow,
)
from src.evolving_trip_belief import EvolvingTripBelief
from src.recourse_itinerary_router import trip_key


HERE = Path(__file__).resolve().parent
STEPS = (60, 20, 10)


def summarize(rows):
    paired = [r for r in rows if r['replay']['status']=='paired']
    gains = [r['replay']['gain'] for r in paired]
    legacy_deltas = [r['gain_exact_vs_legacy_nominal'] for r in rows if r['gain_exact_vs_legacy_nominal'] is not None]
    return dict(windows=len(rows),statuses=dict(Counter(r['replay']['status'] for r in rows)),
                paired=len(paired),changed=sum(r['replay']['action_changed'] for r in paired),
                improved=sum(g>0 for g in gains),harmed=sum(g<0 for g in gains),
                mean_gain=mean(gains) if gains else None,
                different_from_legacy_nominal=sum(r['changed_nominal_from_legacy'] for r in rows),
                different_exact_from_legacy_nominal=sum(r['changed_exact_from_legacy_nominal'] for r in rows),
                exact_vs_legacy_nominal_pairs=len(legacy_deltas),
                exact_vs_legacy_nominal_gain=mean(legacy_deltas) if legacy_deltas else None,
                exact_vs_nominal=summarize_shadow(rows))


def run_day(data, day, fit, cohort):
    graph, actual, *_ = dated_graph(data,day,'gtfs_station_max',True)
    base = EvolvingTripBelief(graph,minute=60,public_occurrences=False,**fit['increments'])
    trips = {trip_key(graph.connections[option[2]]) for row in cohort for option in row['pair']}
    params = {c.route:fit['route_priors'].get(c.route,fit['route_priors']['pooled']) for c in graph.connections}
    outcomes = {}
    for step in STEPS:
        started = perf_counter()
        belief = base.with_grid_step(step,trips=trips)
        rows = []
        for row in cohort:
            prefix, deadline, start = row['prefix'], row['deadline'], row['start']
            entry, status = (fixed_arrival(graph,actual,prefix,start+PUBLICATION,deadline)
                             if prefix else (start,'arrived'))
            pair, mct = row['pair'], row['mct']
            if status=='arrived':
                distribution = belief.event_distribution(pair[1][2],'departure',params)
                replay = replay_direct(graph,actual,distribution,pair,entry,mct,deadline)
            else:
                replay = dict(status='prefix_'+status)
            shadow = exact_shadow(graph,belief,params,actual,pair,entry,mct,prefix,deadline,replay)
            legacy = row['replay']
            exact_choice = shadow.get('exact_first', replay.get('comparison'))
            known = legacy['status']==replay['status']==shadow['status']=='paired'
            rows.append(dict(origin=row['origin'],goal=row['sensitivity_goal'],stop=row['stop'],start=start,
                             incoming=row['incoming'], pair=[p[2] for p in pair], replay=replay, exact_shadow=shadow,
                             changed_nominal_from_legacy=('comparison' in replay and 'comparison' in legacy and
                                                          replay['comparison']!=legacy['comparison']),
                             changed_exact_from_legacy_nominal=(shadow['status'] in ('paired','unresolved_tail')
                                                                and 'comparison' in legacy
                                                                and exact_choice!=legacy['comparison']),
                             gain_exact_vs_legacy_nominal=(replay['gain']+shadow['gain']-legacy['gain']
                                                           if known else None)))
        outcomes[str(step)] = dict(rows=rows, wall_seconds=perf_counter()-started, relevant_trips=len(trips),
                                  summary=summarize(rows))
    return outcomes


def main():
    frozen = json.loads((HERE/'natural_screen.json').read_text())
    rows = [r for r in frozen['reordering_sensitivity']['rows'] if r['class']=='near_critical']
    assert len(rows)==frozen['reordering_sensitivity']['exact_vs_fitted_nominal']['windows']
    by_day, fit = {}, None
    for source in SOURCES:
        data, _ = load_city(ROOT/source)
        if fit is None:
            fit = fit_training(data,retain_no_pickup=True)
        for day in data['holdout_dates']:
            cohort = [row for row in rows if row['day']==day]
            by_day[day] = run_day(data,day,fit,cohort)
            print(day,{step:result['wall_seconds'] for step,result in by_day[day].items()},flush=True)
    summary = {str(step):summarize([row for value in by_day.values() for row in value[str(step)]['rows']])
               for step in STEPS}
    output = dict(status='complete', source='natural_screen.json',step_seconds=STEPS,
                  scope='Same outcome-blind near-critical pairs, actual platform clock and shared fixed tails. '
                        'Only the two option trips per case are needed. The explicit steps use bounded ordered Gaussian cell masses. '
                        'The explicit 60s law is distinct from the legacy point-weight law.',
                  summary=summary,by_day=by_day)
    (HERE/'natural_grid.json').write_text(json.dumps(output,indent=2)+'\n')
    print(summary,flush=True)


if __name__=='__main__':
    main()
