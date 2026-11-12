"""One frozen diagnostic pass; no model fitting or new policy evaluation."""

from collections import Counter
import gzip
import json
from pathlib import Path
from time import perf_counter

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v24.run import options_from, ranks_from
from experiments.reconstruction_v30.forecast import forecast_board, forecast_target
from experiments.reconstruction_v30.metrics import score_forecast
from experiments.reconstruction_v30.selection import build_day_histories, actual_option
from src.empirical_duration_belief import EmpiricalDurationBelief
from src.evolving_trip_belief import EvolvingTripBelief
from src.full_menu_opportunity import FullMenuOpportunity
from src.plan_waiting_controls import SingleTargetOpportunity
from src.transfer_opportunity import TransferOpportunityScores


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'reconstruction_v25'
LAWS = ('original20', 'route_empirical20')


def reference_tail(graph, option, clock):
    """A public/schedule-only reference, never the retrospectively best tail."""
    if not option.tails:
        return None
    first, last = (graph.connections[option.first_path[i]] for i in (0, -1))
    scheduled_ready = clock+last.arr_time-first.dep_time+graph.stops[last.arr_stop].min_transfer_time
    candidates = [cid for cid in option.tails if graph.connections[cid].dep_time >= scheduled_ready]
    return min(candidates, key=lambda cid: (graph.connections[cid].dep_time,
        graph.connections[option.tails[cid][-1]].arr_time-graph.connections[cid].dep_time, cid), default=None)


def duration_support(compiler, cid, fit):
    """Describe the actual grouped laws along this frozen option."""
    groups = {}
    option = compiler.options[cid]
    for path in (option.first_path, *option.tails.values()):
        trip, start, end = compiler._path_slots(path)
        slots = compiler.belief._trips[trip]
        route = compiler.graph.connections[path[0]].route
        for k in range(start, end):
            left, right = slots[k:k+2]
            kind = ('travel' if right.kind == 'arrival' else 'gap'
                if compiler.graph.connections[left.cid].arr_stop != compiler.graph.connections[right.cid].dep_stop else 'dwell')
            delta = right.scheduled-left.scheduled
            key = (route, kind, delta)
            row = fit['route_groups'].get(route, {}).get(kind, {}).get(str(delta))
            source = 'route'
            if row is None:
                row, source = fit['groups'].get(kind, {}).get(str(delta)), 'pooled'
            if row is None:
                row, source = fit['residuals'][kind], 'residual'
            groups[key] = dict(route=route, kind=kind, scheduled=delta, source=source, n=row['n'])
    return list(groups.values())


def write_result(result):
    with gzip.open(HERE/'results.json.gz', 'wt') as handle:
        json.dump(result, handle, separators=(',', ':'), allow_nan=False)


def main():
    started = perf_counter()
    selection = json.loads((SOURCE/'selection.json').read_text())
    fit = json.loads((SOURCE/'model_fit.json').read_text())
    v29 = json.loads((HERE.parent/'reconstruction_v29/selection.json').read_text())['cases']
    v29_scores = json.loads((HERE.parent/'reconstruction_v29/results.json').read_text())['results']
    old = {(r['root_id'], r['law']): r for r in v29_scores}
    data, _ = load_city(SOURCE/'city_transfer_endpoints.json.gz')
    result = dict(complete=False, histories=[], coverage=[], forecasts=[], pairs=[],
        wall=dict(loading=perf_counter()-started, by_day={}, nominal_preparation=0.),
        checks=dict(board_costs=0, v29_costs=0, v29_actual_routes=0))
    frozen_pairs = []
    for day, group in selection['by_day'].items():
        tick = perf_counter()
        graph, _, _, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
        histories, coverage = build_day_histories(graph, metadata, group['roots'], v29)
        result['histories'].extend(histories)
        result['coverage'].extend(coverage)
        params = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
        kwargs = dict(public_occurrences=False, grid_step_seconds=20, **fit['increments'])
        models = dict(original20=EvolvingTripBelief(graph, **kwargs),
            route_empirical20=EmpiricalDurationBelief(graph, duration_fit=fit['duration_fit'], **kwargs))
        result['wall']['by_day'][day] = perf_counter()-tick
        for history in histories:
            root, clock = history['root'], history['clock']
            tick = perf_counter()
            compilers, singles, candidates = {}, {}, {}
            ranks = ranks_from(graph, root['full_options'])
            for law, model in models.items():
                compiler = TransferOpportunityScores(graph, model, params, root['deadline'], options_from(root['full_options']))
                single = SingleTargetOpportunity(FullMenuOpportunity(compiler, ranks, history['entry'], history['ready'], root['deadline']))
                compilers[law], singles[law] = compiler, single
                if 'first_offer' in history['tags']:
                    _, receipt = single.choose(history['visible'], history['seen'], clock)
                    candidates.setdefault((receipt['best_visible'], receipt['target']), []).append(law)
            if 'v29_disagreement' in history['tags']:
                candidates.setdefault((history['frozen_current_cid'], history['frozen_target_cid']), []).append('v29_frozen')
            result['wall']['nominal_preparation'] += perf_counter()-tick
            frozen_pairs.extend(dict(history_id=history['history_id'], current=cur, target=tar, selectors=selectors)
                                for (cur, tar), selectors in candidates.items())
            (HERE/'selection.json').write_text(json.dumps(dict(histories=result['histories'],
                coverage=result['coverage'], pairs=frozen_pairs), indent=2)+'\n')
            local = {}
            for current, target in candidates:
                for role, cid in (('board', current), ('target', target)):
                    if cid is None or (role, cid) in local:
                        continue
                    tick = perf_counter()
                    actual = actual_option(graph, metadata, root, cid, history['entry'], history['ready'], clock, mode=role)
                    labels_wall = perf_counter()-tick
                    option = compilers[LAWS[0]].options[cid]
                    ref = reference_tail(graph, option, clock) if role == 'board' else None
                    local[role, cid] = {}
                    for law, compiler in compilers.items():
                        tick = perf_counter()
                        forecast = (forecast_board(compiler, cid, clock) if role == 'board' else
                            forecast_target(compiler, cid, history['entry'], clock, history['ready']))
                        forecast_wall = perf_counter()-tick
                        tick = perf_counter()
                        if role == 'board' and forecast['status'] == 'supported':
                            expected = compiler.board_score(cid, clock)
                            if expected is None or abs(forecast['cost']-expected.exact) > 1e-6:
                                raise AssertionError('Distribution does not reproduce the existing BOARD value')
                            result['checks']['board_costs'] += 1
                        if 'v29_disagreement' in history['tags'] and cid == history['frozen_'+('current' if role == 'board' else 'target')+'_cid']:
                            expected = old[root['root_id'], law]['scores']['integrated']['current' if role == 'board' else 'wait']
                            if forecast['status'] != 'supported' or abs(forecast['cost']-expected) > 1e-6:
                                raise AssertionError('V29 fixed-action cost changed')
                            result['checks']['v29_costs'] += 1
                        validation_wall = perf_counter()-tick
                        row = dict(forecast_id=f"{history['history_id']}/{role}/{cid}/{law}",
                            history_id=history['history_id'], root_id=root['root_id'], day=day,
                            root_kind=root['root_kind'], tags=history['tags'], law=law, role=role, cid=cid,
                            clock=clock, deadline=root['deadline'], trip=str(graph.connections[cid].trip_id),
                            route=graph.connections[cid].route, forecast=forecast, actual=actual,
                            scores=score_forecast(forecast, actual, root['deadline'], ref), reference_tail=ref,
                            duration_groups=duration_support(compiler, cid, fit['duration_fit']) if law == 'route_empirical20' else None,
                            wall=dict(forecast=forecast_wall, labels=labels_wall if law == LAWS[0] else 0., score_validation=validation_wall))
                        local[role, cid][law] = row
                        result['forecasts'].append(row)
            for (current, target), selectors in candidates.items():
                actual_board = local['board', current][LAWS[0]]['actual']['cost']
                actual_target = None if target is None else local['target', target][LAWS[0]]['actual']['cost']
                pair = dict(history_id=history['history_id'], root_id=root['root_id'], day=day,
                    root_kind=root['root_kind'], tags=history['tags'], selectors=selectors,
                    clock=clock, deadline=root['deadline'], current_cid=current, target_cid=target,
                    actual_board=actual_board, actual_target=actual_target,
                    actual_gain=None if actual_board is None or actual_target is None else actual_board-actual_target, by_law={})
                for law, compiler in compilers.items():
                    board = local['board', current][law]
                    wait = None if target is None else local['target', target][law]
                    times, mass, cancel = (compiler.departure_distribution(target) if target is not None else (None, None, None))
                    nominal_gain = None
                    if target is not None:
                        unseen = (times < history['entry']) | (times > clock)
                        denominator = cancel+mass[unseen].sum()
                        if denominator > 0:
                            legal = (times > clock) & (times >= history['ready']) & (times < root['deadline']) & (mass > 0)
                            wait_nominal = root['deadline']-sum(p*(root['deadline']-compiler.clock_nominal_score(target, int(t)))
                                for t, p in zip(times[legal], mass[legal]))/denominator
                            nominal_gain = compiler.clock_nominal_score(current, clock)-wait_nominal
                    pair['by_law'][law] = dict(board_id=board['forecast_id'], target_id=None if wait is None else wait['forecast_id'],
                        nominal_gain=None if nominal_gain is None else float(nominal_gain),
                        integrated_gain=None if wait is None or board['forecast']['status'] != 'supported' or wait['forecast']['status'] != 'supported'
                            else board['forecast']['cost']-wait['forecast']['cost'])
                if 'v29_frozen' in selectors:
                    case = next(c for c in v29 if c['root']['root_id'] == root['root_id'])
                    if any(value is None or value-root['start'] != case[key]['loss']
                           for value, key in ((actual_board, 'actual_one_step'), (actual_target, 'actual_single'))):
                        raise AssertionError('V29 two-win/four-loss realized routes changed')
                    result['checks']['v29_actual_routes'] += 2
                result['pairs'].append(pair)
            print(history['history_id'], len(candidates), dict(Counter(r['actual']['status'] for r in result['forecasts']
                if r['history_id'] == history['history_id'] and r['law'] == LAWS[0])), flush=True)
            write_result(result)
    result['complete'] = True
    result['wall']['total'] = perf_counter()-started
    write_result(result)
    from experiments.reconstruction_v30.summarize import summarize
    summary = summarize(result)
    (HERE/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(histories=len(result['histories']), forecasts=len(result['forecasts']),
        pairs=len(result['pairs']), checks=result['checks'], wall=result['wall']), indent=2), flush=True)


if __name__ == '__main__':
    main()
