"""Frozen-rule actual transfer across all V25 origins and conditional roots."""

from collections import Counter
import gzip
import json
from pathlib import Path
from time import perf_counter

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v24.run import options_from, ranks_from
from experiments.reconstruction_v25.run import physical_path, public_offers, execute
from experiments.reconstruction_v28.summarize import summarize
from src.empirical_duration_belief import EmpiricalDurationBelief
from src.evolving_trip_belief import EvolvingTripBelief
from src.full_menu_opportunity import FullMenuOpportunity
from src.plan_waiting_controls import SchedulePriority, SingleTargetOpportunity, CombinedWaitingOpportunity
from src.transfer_opportunity import TransferOpportunityScores


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'reconstruction_v25'
LAWS = ('original20', 'route_empirical20')
METHODS = ('full_first_offer', 'one_step_nominal', 'schedule_priority', 'single_target_nominal', 'combined_wait_nominal')
FIELDS = ('action', 'tail', 'loss', 'arrived', 'status')


def assert_same(current, previous, label):
    if any(current.get(field) != previous.get(field) for field in FIELDS):
        raise AssertionError((label, 'Frozen control outcome changed', current, previous))


def run_root(graph, metadata, belief, params, root):
    tick = perf_counter()
    options, ranks = options_from(root['full_options']), ranks_from(graph, root['full_options'])
    entry, status = root['start'], 'arrived'
    if root['prefix']:
        entry, status = physical_path(metadata, root['prefix'], root['start']+30, root['deadline'])
    row = {key: root[key] for key in ('root_id', 'day', 'root_kind', 'start', 'deadline')}
    row['menu_size'] = len(options)
    compilation = dict.fromkeys(METHODS, 0.)
    if status != 'arrived':
        loss = None if entry is None else root['deadline']-root['start']
        row.update(outcomes={m: dict(action=None, tail=None, loss=loss, arrived=False,
            status='prefix_'+status, trace=[], online_wall=0., physical_replay_wall=0.) for m in METHODS},
            compilation_wall=compilation, shared_replay_preparation_wall=perf_counter()-tick,
            nominal_compile_wall=0.)
        return row
    ready = entry+(max(30, graph.stops[root['stop']].min_transfer_time) if root['prefix'] else 30)
    offers = public_offers(metadata, tuple(options), entry, ready, root['deadline'])
    shared = perf_counter()-tick
    tick = perf_counter()
    priority = SchedulePriority({o['first_cid']: o['scheduled_arrival'] for o in root['full_options']})
    compilation['schedule_priority'] = perf_counter()-tick
    controls = dict(full_first_offer=None, one_step_nominal=None, schedule_priority=priority,
                    single_target_nominal=None, combined_wait_nominal=None)
    nominal_compile = 0.
    if options:
        tick = perf_counter()
        compiler = TransferOpportunityScores(graph, belief, params, root['deadline'], options)
        one = FullMenuOpportunity(compiler, ranks, entry, ready, root['deadline'])
        nominal_compile = perf_counter()-tick
        controls['one_step_nominal'] = one
        tick = perf_counter()
        single = SingleTargetOpportunity(one)
        compilation['single_target_nominal'] = perf_counter()-tick
        controls['single_target_nominal'] = single
        tick = perf_counter()
        controls['combined_wait_nominal'] = CombinedWaitingOpportunity(single)
        compilation['combined_wait_nominal'] = perf_counter()-tick
    outcomes = {m: execute(graph, metadata, root, options, ranks, offers, controller)
                for m, controller in controls.items()}
    row.update(entry=entry, ready=ready, outcomes=outcomes, compilation_wall=compilation,
        shared_replay_preparation_wall=shared, nominal_compile_wall=nominal_compile)
    return row


def describe(rows):
    summary = summarize(rows)
    summary['menu_sizes'] = dict(Counter(r['menu_size'] for r in rows))
    summary['wall'] = dict(shared_replay_preparation=sum(r['shared_replay_preparation_wall'] for r in rows),
        nominal_compile=sum(r['nominal_compile_wall'] for r in rows),
        incremental_compile={m: sum(r['compilation_wall'][m] for r in rows) for m in METHODS},
        online={m: sum(r['outcomes'][m]['online_wall'] for r in rows) for m in METHODS},
        physical_replay={m: sum(r['outcomes'][m]['physical_replay_wall'] for r in rows) for m in METHODS})
    summary['wall']['root_control_total'] = {
        m: sum(r['outcomes'][m]['online_wall']+r['compilation_wall'][m]
            +(r['nominal_compile_wall'] if m in ('one_step_nominal','single_target_nominal','combined_wait_nominal') else 0.)
            +(r['compilation_wall']['single_target_nominal'] if m == 'combined_wait_nominal' else 0.)
            for r in rows) for m in METHODS}
    return summary


def main():
    started = perf_counter()
    selection = json.loads((SOURCE/'selection.json').read_text())
    fit = json.loads((SOURCE/'model_fit.json').read_text())
    construction = {r['root_id'] for r in json.loads((HERE.parent/'reconstruction_v26'/'selection.json').read_text())['roots']}
    with gzip.open(SOURCE/'results.json.gz', 'rt') as handle:
        v25 = json.load(handle)
    with gzip.open(HERE.parent/'reconstruction_v27'/'results.json.gz', 'rt') as handle:
        v27 = json.load(handle)
    if not v25['complete'] or not v27['complete']:
        raise AssertionError('Frozen source stages are incomplete')
    previous = {r['root_id']: r for r in v25['results']}
    local = {(r['law'], r['root_id']): r for r in v27['results']}
    roots = [r for d in selection['by_day'].values() for r in d['roots']]
    if (len(roots), sum(r['root_kind'] == 'origin' for r in roots), len(construction)) != (73, 48, 6):
        raise AssertionError('V25 complete population changed')
    data, _ = load_city(SOURCE/'city_transfer_endpoints.json.gz')
    loading = perf_counter()-started
    results, day_preparation = [], {}
    for day, group in selection['by_day'].items():
        tick = perf_counter()
        graph, _, _, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
        graph_wall = perf_counter()-tick
        tick = perf_counter()
        params = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
        kwargs = dict(public_occurrences=False, grid_step_seconds=20, **fit['increments'])
        models = dict(original20=EvolvingTripBelief(graph, **kwargs),
            route_empirical20=EmpiricalDurationBelief(graph, duration_fit=fit['duration_fit'], **kwargs))
        day_preparation[day] = dict(graph=graph_wall, model_parameters_and_beliefs=perf_counter()-tick)
        for root in group['roots']:
            for law, model in models.items():
                row = run_root(graph, metadata, model, params, root)
                row.update(law=law, construction=root['root_id'] in construction)
                old = previous[root['root_id']]['outcomes']
                old_method = 'original_full_nominal' if law == 'original20' else 'empirical_full_nominal'
                for current, before in (('full_first_offer', 'full_first_offer'), ('one_step_nominal', old_method)):
                    assert_same(row['outcomes'][current], old[before], (root['root_id'], law, current))
                if row['construction']:
                    for method in METHODS[2:]:
                        assert_same(row['outcomes'][method], local[law, root['root_id']]['actual_new'][method],
                                    ('V27 construction reproduction', root['root_id'], law, method))
                results.append(row)
                print(law, root['root_id'], {m: (o['loss'], o['status']) for m, o in row['outcomes'].items()}, flush=True)
            with gzip.open(HERE/'results.json.gz', 'wt') as handle:
                json.dump(dict(complete=False, results=results), handle, separators=(',', ':'))
    report = dict(loading_wall=loading, day_preparation_wall=day_preparation, by_law={})
    for law in LAWS:
        selected = [r for r in results if r['law'] == law]
        origins = [r for r in selected if r['root_kind'] == 'origin']
        report['by_law'][law] = dict(
            origin=describe(origins),
            construction_origin=describe([r for r in origins if r['construction']]),
            remaining_origin=describe([r for r in origins if not r['construction']]),
            first_transfer=describe([r for r in selected if r['root_kind'] == 'first_transfer']),
            by_day={day: {kind: describe([r for r in selected if r['day'] == day and r['root_kind'] == kind])
                          for kind in ('origin', 'first_transfer')} for day in selection['by_day']},
            by_menu_size={kind: {label: describe([r for r in selected if r['root_kind'] == kind and lo <= r['menu_size'] <= hi])
                         for label, lo, hi in (('empty',0,0),('1-5',1,5),('6-10',6,10),('11-18',11,18))}
                         for kind in ('origin', 'first_transfer')})
    report.update(total_wall=perf_counter()-started, v25_cells_reproduced=146,
                  v25_control_outcomes_reproduced=292, v27_construction_cells_reproduced=12,
                  v27_control_outcomes_reproduced=36, new_worlds=0)
    with gzip.open(HERE/'results.json.gz', 'wt') as handle:
        json.dump(dict(complete=True, results=results, summary=report), handle, separators=(',', ':'))
    (HERE/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
