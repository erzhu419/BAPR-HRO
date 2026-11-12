"""Integrate executed conditional adaptive continuations back into root G.

The independent quadrature pool is never passed to evaluated decisions. Its
only output to the root chain is a mean and standard error for each entry.
All root controls share the same executable continuation policy and G table.
"""

import gzip
import json
from time import perf_counter

import numpy as np

from experiments.reconstruction_v32.journey import TripWorlds, paired_statistics
from experiments.reconstruction_v33.execute import Executor, DrawnWorld, ArchivedWorld
from experiments.reconstruction_v33.history import ConditionalEngine
from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v33.planner import Planner
from experiments.reconstruction_v33.run import HERE, SOURCE, setup_day, jsonable
from experiments.reconstruction_v4.run_city_replay import load_city


class JointTailPlanner(Planner):
    def __init__(self, graph, model, params, deadline, root, root_method, store, protocol, index):
        super().__init__(graph, model, params, deadline)
        self.root, self.root_method, self.store = root, root_method, store
        self.protocol, self.index = protocol, index
        self.top_platforms = {platform: stage for stage, platform in enumerate(root.exits.values())}

    def compile_ride(self, ride, history, method='one_step_rollout'):
        return super().compile_ride(ride, history, self.root_method if ride is self.root else 'one_step_rollout')

    def first_offer_values(self, platform, history, ready, method='one_step_rollout'):
        if platform not in self.top_platforms:
            return super().first_offer_values(platform, history, ready, 'one_step_rollout')
        clocks = np.asarray(ready)-self.graph.stops[platform.stop].min_transfer_time
        key = self.structure(platform), history.signature(self.relevant_trips(platform)), tuple(clocks)
        if key not in self.store['values']:
            began = perf_counter()
            stage = self.top_platforms[platform]
            count = self.protocol['joint_tail_quadrature_worlds']
            seed = self.protocol['joint_tail_seed']+1000*self.index+10*stage
            paths = [p for ride in platform.rides.values() for p in ride.paths()]
            bank = TripWorlds(history.as_belief(), self.parameters, paths, count, seed)
            means, errors = np.full(len(clocks), self.deadline), np.zeros(len(clocks))
            supported = 0
            for k, entry in enumerate(clocks):
                if entry >= self.deadline:
                    continue
                values = []
                for world in range(count):
                    planner = Planner(self.graph, self.model, self.parameters, self.deadline)
                    execute = Executor(self.graph, planner, self.deadline,
                                       method='one_step_rollout', waiting='pair', trace=False)
                    arrival, _ = execute.platform(platform, float(entry), history.clone(), DrawnWorld(bank, world))
                    values.append(arrival)
                means[k] = np.mean(values)
                errors[k] = np.std(values, ddof=1)/np.sqrt(count)
                supported += 1
            row = dict(stage=stage, stop=platform.stop, entry_clocks=clocks, means=means,
                       standard_errors=errors, quadrature_worlds=count, seed=seed,
                       evaluated_entries=supported, wall=perf_counter()-began)
            self.store['values'][key] = means
            self.store['receipts'].append(row)
            print(self.index, 'joint G', stage, supported, 'seconds', round(row['wall'], 2), flush=True)
        return self.store['values'][key]


def main():
    protocol = json.loads((HERE/'protocol.json').read_text())
    selection = json.loads((HERE.parent/'reconstruction_v32/selection_full_hubs.json').read_text())
    eligible = [(i, r) for i, r in enumerate(selection['selected'])
                if len(r['hubs']) >= 3 and any(h['root_tail_overlap'] for h in r['hubs'])]
    selected = sorted(eligible, key=lambda item: (sum(len(h['options']) for h in item[1]['hubs']), item[0]))[:2]
    (HERE/'joint_selection.json').write_text(json.dumps(dict(rule=protocol['joint_tail_cases'],
        eligible_root_indices=[i for i, _ in eligible], selected_root_indices=[i for i, _ in selected],
        selection_uses_actual=False), indent=2)+'\n')
    fit = json.loads((SOURCE/'model_fit.json').read_text())
    segment = json.loads((HERE.parent/'reconstruction_v31/segment_fit.json').read_text())
    data, _ = load_city(SOURCE/'city_transfer_endpoints.json.gz')
    rows = []
    (HERE/'joint_roots').mkdir(exist_ok=True)
    for index, root in selected:
        output = HERE/'joint_roots'/f'{index:02}.json.gz'
        if output.exists():
            with gzip.open(output, 'rt') as handle:
                rows.append(json.load(handle))
            continue
        began = perf_counter()
        graph, metadata, model, params, applied = setup_day(data, root['day'], fit, segment)
        ride = root_ride(graph, root)
        store = dict(values={}, receipts=[])
        methods = ('nominal', 'rolling_fixed', 'one_step_rollout', 'optimal')
        # Precompile before accessing any actual outcome or evaluation world.
        empty = ConditionalEngine(model, params)
        for method in methods:
            planner = JointTailPlanner(graph, model, params, root['deadline'], ride, method, store, protocol, index)
            planner.compile_ride(ride, empty, method)
        compile_wall = perf_counter()-began
        bank = TripWorlds(model, params, ride.paths(), protocol['evaluation_worlds'],
                          protocol['evaluation_seed']+1000*index)
        actual, sampled = {}, {}
        for method in methods:
            planner = JointTailPlanner(graph, model, params, root['deadline'], ride, method, store, protocol, index)
            actual[method] = Executor(graph, planner, root['deadline'], method='one_step_rollout', waiting='pair').run(
                ride, root['start'], ConditionalEngine(model, params), ArchivedWorld(metadata))
            samples = []
            for world in range(bank.count):
                planner = JointTailPlanner(graph, model, params, root['deadline'], ride, method, store, protocol, index)
                outcome = Executor(graph, planner, root['deadline'], method='one_step_rollout', waiting='pair', trace=False).run(
                    ride, root['start'], ConditionalEngine(model, params), DrawnWorld(bank, world))
                if outcome['loss'] is None:
                    raise AssertionError(('Original-law evaluation unsupported', index, world, outcome))
                samples.append(outcome['loss'])
            sampled[method] = samples
        row = dict(root_id=root['root_id'], root_index=index, actual=actual, sampled=sampled,
            paired={m: paired_statistics(sampled['one_step_rollout'], values) for m, values in sampled.items()},
            G=store['receipts'], compile_wall=compile_wall, total_wall=perf_counter()-began,
            supplement_applied=applied, scope='Exact root chain for independently estimated executed conditional-tail G; finite quadrature uncertainty retained.')
        with gzip.open(output, 'wt') as handle:
            json.dump(row, handle, default=jsonable)
        rows.append(row)
    summary = dict(roots=len(rows), rows=[{k:v for k,v in r.items() if k not in ('G',)} for r in rows])
    (HERE/'joint_summary.json').write_text(json.dumps(summary, default=jsonable, indent=2)+'\n')


if __name__ == '__main__':
    main()
