"""Standard current-action rollout at every structurally optional alighting."""

import gzip
import json
from pathlib import Path
from time import perf_counter, process_time

from experiments.reconstruction_v33.run import jsonable
from experiments.reconstruction_v34.conditional_reference import restore_snapshot
from experiments.reconstruction_v34.execute import Executor
from experiments.reconstruction_v34.gated_rollout import conditional_action_check
from experiments.reconstruction_v36.forecast import SingleMessageHistory


def has_legal_continue(ride, cid):
    """A later retained exit, including the destination, exists on this vehicle."""
    return any(later in ride.exits for later in ride.path[ride.path.index(cid)+1:])


def restore_single_snapshot(model, parameters, fit, public_state, clock):
    parsed = restore_snapshot(model, parameters, public_state, clock)
    history = SingleMessageHistory(model, parameters, fit=fit)
    for name in ('_records', '_forecasts', '_endpoint_statuses', '_trip_statuses', '_service_statuses'):
        setattr(history, name, getattr(parsed, name))
    history.message_clock = float(clock)
    return history


def node_structure(node):
    """Plain recursive menu data; node object identity is not a cache key."""
    if hasattr(node, 'path'):
        return dict(kind='ride', first_cid=node.first_cid, path=list(node.path),
            exits=[dict(cid=cid, next=None if node.exits[cid] is None else node_structure(node.exits[cid]))
                   for cid in node.path if cid in node.exits])
    return dict(kind='platform', stop=node.stop,
        rides=[dict(cid=cid, next=node_structure(ride)) for cid, ride in sorted(node.rides.items())])


def current_diagnostics(planner, ride, cid, clock, history):
    base, base_receipt = planner.compile_ride(ride, history, 'rolling_fixed').decision(cid, clock)
    nominal, nominal_receipt = planner.compile_ride(ride, history, 'nominal').decision(cid, clock)
    base, nominal = ('alight' if value else 'continue' for value in (base, nominal))
    return dict(base=base, alternative='continue' if base == 'alight' else 'alight',
                nominal_diagnostic=nominal, cheap_agreed=base == nominal,
                cheap_receipts=dict(base=base_receipt, nominal=nominal_receipt))


def onboard_state(ride, cid, clock, history, deadline, *, root_index, day, count, diagnostics):
    """Semantic state includes every raw message that later reselection may use.

    Seed and call number belong to the paid receipt, not the semantic state.
    Existing matching states reuse that receipt's original random stream.
    """
    value = dict(root_index=root_index, day=day, deadline=deadline,
        node=node_structure(ride), kind='onboard', cid=cid, clock=clock,
        public_history=history.public_state(),
        actions={key: diagnostics[key] for key in ('base', 'alternative', 'nominal_diagnostic')},
        count=count, continuation='combined_rolling; fixed received messages; new local observations',
        frozen_model=dict(physical='../reconstruction_v34/model_fit.json',
                          emission='../reconstruction_v35/fitted_emission.json',
                          conditioner='reconstruction_v36.SingleMessageHistory'))
    # JSON normalizes only tuple/list containers for equality with saved receipts.
    return json.loads(json.dumps(value, default=jsonable, allow_nan=False))


def evaluate_onboard_actions(graph, planner, ride, cid, clock, history, *, root_index, day,
                             call_index, count=128, seed_base=37050, diagnostics=None):
    if not has_legal_continue(ride, cid):
        raise ValueError('Current arrival has no later retained exit for CONTINUE')
    tick, cpu = perf_counter(), process_time()
    if diagnostics is None:
        diagnostics = current_diagnostics(planner, ride, cid, clock, history)
    state = onboard_state(ride, cid, clock, history, planner.deadline,
        root_index=root_index, day=day, count=count, diagnostics=diagnostics)
    setup_cost = dict(wall=perf_counter()-tick, process_cpu=process_time()-cpu)
    seed = seed_base+1000*root_index+call_index
    # The common evaluator's historical labels are renamed below. Its first
    # action here is the alternative, which need not be the nominal action.
    result = conditional_action_check(graph, planner.model, planner.parameters,
        planner.deadline, ride, dict(kind='onboard', cid=cid, clock=clock), history,
        diagnostics['alternative'], diagnostics['base'], count=count, seed=seed)
    result['rows'] = dict(alternative=result['rows']['nominal'], base=result['rows']['combined'])
    costs = result['cost']['action_continuations']
    result['cost']['action_continuations'] = dict(alternative=costs['nominal'], base=costs['combined'])
    result['actions'] = state['actions']
    result['paired']['orientation'] = 'alternative action loss minus base action loss; negative favors alternative'
    result.update(state=state, gate_call_index=call_index, cheap_agreed=diagnostics['cheap_agreed'],
                  cheap_receipts=diagnostics['cheap_receipts'], diagnostic_setup_cost=setup_cost)
    return result


class ReceiptStore:
    """Reuse the first matching paid semantic state; never select by its value."""
    def __init__(self, root_index, reference_directory, policy_directory):
        self.root_index = root_index
        self.reference_directory = Path(reference_directory)
        self.policy_directory = Path(policy_directory)

    def obtain(self, state, call_index, evaluate):
        pattern = f'{self.root_index:03}_*.json.gz'
        for directory in (self.reference_directory, self.policy_directory):
            for path in sorted(directory.glob(pattern)):
                with gzip.open(path, 'rt') as handle:
                    receipt = json.load(handle)
                if receipt['state'] == state:
                    return receipt, dict(reused=True, receipt_source=str(path))
        receipt = evaluate()
        if receipt['state'] != state:
            raise ValueError('Computed receipt differs from the requested public state')
        self.policy_directory.mkdir(parents=True, exist_ok=True)
        path = self.policy_directory/f'{self.root_index:03}_{call_index:03}.json.gz'
        with gzip.open(path, 'xt') as handle:
            json.dump(receipt, handle, default=jsonable, allow_nan=False)
        return receipt, dict(reused=False, receipt_source=str(path))


class OnboardRolloutExecutor(Executor):
    def __init__(self, graph, planner, deadline, *, root_index, day, store, count=128, seed_base=37050):
        super().__init__(graph, planner, deadline, method='rolling_fixed', waiting='combined')
        self.root_index, self.day, self.store = root_index, day, store
        self.count, self.seed_base, self.eligible_calls = count, seed_base, 0
        self.gates = []
        self.reused_evaluation_cpu, self.reused_evaluation_wall = 0., 0.

    def choose_onboard(self, ride, cid, clock, history):
        diagnostic = current_diagnostics(self.planner, ride, cid, clock, history)
        if not has_legal_continue(ride, cid):
            return diagnostic['base'] == 'alight', dict(method='standard_onboard_rollout',
                triggered=False, reason='no_later_retained_exit', **diagnostic)
        call = self.eligible_calls
        self.eligible_calls += 1
        state = onboard_state(ride, cid, clock, history, self.deadline,
            root_index=self.root_index, day=self.day, count=self.count, diagnostics=diagnostic)
        proposed_seed = self.seed_base+1000*self.root_index+call
        receipt, reuse = self.store.obtain(state, call, lambda: evaluate_onboard_actions(
            self.graph, self.planner, ride, cid, clock, history, root_index=self.root_index,
            day=self.day, call_index=call, count=self.count, seed_base=self.seed_base,
            diagnostics=diagnostic))
        if reuse['reused']:
            self.reused_evaluation_cpu += receipt['cost']['total_process_cpu']
            self.reused_evaluation_wall += receipt['cost']['total_wall']
        gate = dict(receipt, context=state, source_evaluation_call_index=receipt['gate_call_index'],
                    gate_call_index=call, proposed_seed=proposed_seed, **reuse)
        self.gates.append(gate)
        return gate['selected_action'] == 'alight', dict(method='standard_onboard_rollout', triggered=True,
            gate_call_index=call, seed=gate['seed'], proposed_seed=proposed_seed,
            reused=gate['reused'], receipt_source=gate['receipt_source'],
            actions=gate['actions'], cheap_agreed=gate['cheap_agreed'], paired=gate['paired'],
            selected_action=gate['selected_action'], cost=gate['cost'])
