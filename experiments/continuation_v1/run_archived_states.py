"""Freeze three real archived policy histories and compare private rollout values.

Only the legacy actor prefixes use the archived realized vehicle scene. Private
counterfactual banks use fitted parameters and the captured public TripBelief;
no realized future is supplied to sampling or tail-policy action selection.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import copy
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.continuation_v1.historical_router import HistoricalContinuationRouter as ContinuationRouter
from src.simulator import JourneyScenario, RegimeSchedule, VehicleEvent
from src.simulate_bandit import simulate_bandit_journey
from src.trip_belief import OUTCOMES

SAMPLES = 8
SEED_BASE = 81403


class TargetCaptured(Exception):
    pass


class CapturingActor(ContinuationRouter):
    """Legacy action selection with a read-only observation/state checkpoint."""
    def __init__(self, graph, priors, target):
        super().__init__(graph, priors, legacy=True)
        self.online_mean = self.online_cancel = False
        self.target = target
        self.received_messages = []
        self.capture = None

    def observe_public_messages(self, stop, time, messages, seen_before):
        super().observe_public_messages(stop, time, messages, seen_before)
        self.received_messages.extend(vars(message).copy() for message in messages)

    def select_connection(self, current_stop, current_time, rng, top_k=5):
        index = len(self.decisions)
        target_index = self.target['decision_index']
        if index == target_index:
            expected_state = self.target['state']
            if (current_stop, current_time) != (expected_state['stop'], expected_state['time']):
                raise AssertionError('Legacy actor did not reach the archived target state')
            belief = self.trip_belief.copy()
            context = self.information_state
            result = super().select_connection(current_stop, current_time, rng, top_k)
            if result is None or result[0].connection_id != self.target['archived_decision']['selected']:
                raise AssertionError('Legacy actor selected a different archived target action')
            self.capture = {
                'belief': belief, 'context': context,
                'cached_result': copy.deepcopy(self.cached_result),
                'fresh_menu': set(self._fresh_menu),
                'initial_menu': set(self.initial_menu),
                'route_beliefs': copy.deepcopy(self.route_beliefs),
                'initial_cancels': self._initial_cancels.copy(),
            }
            raise TargetCaptured()
        if index > target_index:
            raise AssertionError('Target capture was missed')
        return super().select_connection(current_stop, current_time, rng, top_k)


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [json_value(item) for item in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarize_belief(belief):
    return {
        'known': [{'trip_key': list(key), 'canceled': value[0], 'delay': value[1]}
                  for key, value in sorted(belief.known.items())],
        'allowed_outcomes': [{'trip_key': list(key), 'outcomes': OUTCOMES[mask].tolist()}
                             for key, mask in sorted(belief._allowed.items())],
        'unknown_unlisted_trip_rule': 'All 71 outcomes allowed before plug-in weighting.',
    }


def validate_prefix(actor, archived_record):
    if len(actor.decisions) != actor.target['decision_index'] + 1:
        raise AssertionError('Captured prefix has the wrong decision count')
    for actual, expected in zip(actor.decisions, archived_record['decisions']):
        for field in ('stop', 'time', 'selected', 'candidates', 'refresh_only_ids', 'retained_only_ids'):
            if actual[field] != expected[field]:
                raise AssertionError(f'Legacy decision prefix differs in {field}')
        if abs(actual['score'] - expected['score']) > 1e-7:
            raise AssertionError('Legacy decision prefix score differs')
    expected_messages = archived_record['published_messages'][:len(actor.received_messages)]
    if actor.received_messages != expected_messages:
        raise AssertionError('Public message prefix differs from the archive')


def make_counterfactual(actor, conditioning, rescue, seed):
    capture = actor.capture
    router = ContinuationRouter(actor.graph, actor.priors, conditioning=conditioning,
                                rescue=rescue, samples=SAMPLES, seed=seed)
    router.online_mean = router.online_cancel = False
    router.origin, router.destination, router.departure = actor.origin, actor.destination, actor.departure
    router.journey_deadline = actor.journey_deadline
    router.cached_result = capture['cached_result']
    router._fresh_menu = capture['fresh_menu'].copy()
    router.initial_menu = capture['initial_menu'].copy()
    router.route_beliefs = copy.deepcopy(capture['route_beliefs'])
    router._initial_cancels = capture['initial_cancels'].copy()
    router._version = router._valued_version = 0
    router.trip_belief = capture['belief'].copy()
    router.information_state = capture['context']
    context = router.information_state
    router.set_execution_context(context.current_trip, context.last_arrival, context.unavailable_ids)
    return router


def analytic_known_action(actor, cid):
    """Availability and immediate physical cost from public trip facts only."""
    context, belief = actor.capture['context'], actor.capture['belief']
    connection = actor.graph.connections[cid]
    key = belief._trip_by_cid[cid]
    if key not in belief.known:
        return None
    canceled, delay = belief.known[key]
    ready = max(context.time, context.last_arrival + (
        0 if context.current_trip is None or connection.trip_id == context.current_trip
        else actor.graph.stops[context.stop].min_transfer_time))
    if canceled:
        kind, end, board = 'cancel', max(ready, connection.dep_time), False
    else:
        departure, arrival = connection.dep_time + delay, connection.arr_time + delay
        if departure < ready:
            kind, end, board = 'missed', ready, False
        elif departure > ready + 12:
            kind, end, board = 'wait_late', ready + 12, False
        else:
            kind, end, board = 'board', arrival, True
    output = {'derived_from_public_trip_key': list(key), 'known_canceled': canceled,
              'known_delay': delay, 'ready_time': ready,
              'first_attempt_kind': kind, 'first_attempt_failure_probability': float(not board),
              'first_action_end': end, 'first_action_elapsed': end - context.time,
              'terminal_failure': end >= actor.journey_deadline,
              'original_journey_elapsed_at_action_end': end - actor.departure}
    if not canceled:
        output['known_departure'], output['known_arrival'] = departure, arrival
    if board:
        output['destination_arrival_before_this_leg_arrives_probability'] = 0.0
        # A same-trip complete continuation is a physically executable policy,
        # without claiming that every later leg is in the fixed five-label tail.
        legs = sorted((c for c in actor.graph.connections if c.trip_id == connection.trip_id
                       and c.dep_time >= connection.dep_time), key=lambda c: (c.dep_time, c.id))
        stop, clock, path = context.stop, ready, []
        for leg in legs:
            if leg.dep_stop != stop or leg.dep_time + delay < clock:
                continue
            path.append(leg.id)
            stop, clock = leg.arr_stop, leg.arr_time + delay
            if stop == actor.destination:
                output['same_trip_physical_policy'] = {
                    'connections': path, 'destination_arrival': clock,
                    'journey_loss': min(actor.max_time, clock - actor.departure),
                    'scope': 'Known-trip physical continuation, not claimed to stay inside every future five-label menu.'}
                break
    return output


def main():
    diagnosis_path = Path(__file__).with_name('archived_panel_diagnosis.json')
    diagnoses = json.loads(diagnosis_path.read_text())['results']
    panel = json.loads((ROOT / 'experiments/reconstruction_v2/matched_corrected_results.json').read_text())
    with (ROOT / 'data/zurich_wide.pkl').open('rb') as handle:
        base = pickle.load(handle)
    names = {stop.name: sid for sid, stop in base.stops.items()}
    outputs = []
    for index, target in enumerate(diagnoses):
        case = next(case for case in panel['cases'] if case['spec'] == target['case_spec'])
        archived_record = next(record for record in case['records'] if record['method'] == 'refresh/offline')
        scene = JourneyScenario({int(cid): VehicleEvent(**event) for cid, event in case['world'].items()})
        actor = CapturingActor(copy.deepcopy(base), case['priors'], target)
        started = perf_counter()
        try:
            simulate_bandit_journey(actor.graph, actor,
                                    names[case['spec'][2]], names[case['spec'][3]], 490,
                                    RegimeSchedule([(0, 'generated')]),
                                    np.random.default_rng(9000 + case['spec'][4]),
                                    max_time=120, scenario=scene)
        except TargetCaptured:
            pass
        if actor.capture is None:
            raise AssertionError('Actor finished without reaching the target state')
        validate_prefix(actor, archived_record)
        context = actor.capture['context']
        state_output = {
            'case_spec': case['spec'], 'target_connection': target['archived_decision']['selected'],
            'archived_decision_index': target['decision_index'],
            'archived_selected_score': target['archived_decision']['score'],
            'legacy_actor_prefix_match': True,
            'legacy_prefix_decisions': actor.decisions,
            'public_receipt_log': actor.polls,
            'received_messages': actor.received_messages,
            'captured_context': json_value(context),
            'captured_trip_belief': summarize_belief(actor.capture['belief']),
            'frozen_parameters': actor._parameters(),
            'prefix_replay_ms': (perf_counter() - started) * 1000,
            'analytic_public_known_selected_action': analytic_known_action(actor, target['archived_decision']['selected']),
            'private_seed': SEED_BASE + index,
            'private_samples': SAMPLES,
            'comparisons': [],
        }
        for conditioning in (False, True):
            for rescue in (False, True):
                router = make_counterfactual(actor, conditioning, rescue, SEED_BASE + index)
                started = perf_counter()
                chosen = router.select_connection(context.stop, context.time,
                                                  np.random.default_rng(0), top_k=5)
                ids = list(router._scores)
                if ids != target['archived_decision']['candidates']:
                    raise AssertionError('Counterfactual valuation changed the archived five-candidate menu')
                row = {'conditioning': conditioning, 'rescue': rescue,
                       'selected_connection': chosen[0].connection_id if chosen else None,
                       'selected_score': chosen[1] if chosen else None,
                       'candidate_values': copy.deepcopy(router._scores),
                       'rollout_steps': router.rollout_steps,
                       'valuation_ms': (perf_counter() - started) * 1000}
                posterior_delays, posterior_probs = router._initial_belief.posterior(
                    state_output['target_connection'], router._parameters_now)
                row['target_private_trip_posterior'] = [
                    {'delay': int(delay), 'probability': float(probability)}
                    for delay, probability in zip(posterior_delays, posterior_probs)
                    if probability > 0]
                if 'omitted_connection' in target:
                    forced_cid = target['omitted_connection']['original_id']
                    label = next(label for label in router.cached_result.stop_labels[context.stop]
                                 if label.connection_id == forced_cid)
                    forced_score = router._score(label, context.time)
                    row['forced_root_action_outside_top5'] = {
                        'connection': forced_cid, 'score': forced_score,
                        'values': router._scores[forced_cid],
                        'status': 'Counterfactual root action only; tail policy and its five-candidate menus remain unchanged.'}
                state_output['comparisons'].append(json_value(row))
                print(json.dumps({'target': state_output['target_connection'], 'conditioning': conditioning,
                                  'rescue': rescue, 'selected': row['selected_connection'],
                                  'score': row['selected_score']}, ensure_ascii=False), flush=True)
        outputs.append(state_output)
        destination = Path(__file__).with_name('archived_states_results.json')
        destination.write_text(json.dumps({
            'scope': 'Three captured archived policy histories; state-level private plug-in rollout comparisons on the full 2000-connection graph.',
            'actor': 'Original refresh/offline action rule; public-history hooks do not affect its legacy decisions.',
            'frozen_tail': 'Same cached label pool, same five-candidate screen, nominal causal tail policy, frozen fitted mean/cancel parameters.',
            'sampling': '8 private model draws per state/control, the same seed across conditioning controls and identical deterministic sampling inputs across rescue controls; archived future is not supplied.',
            'timing_scope': 'valuation_ms and rollout_steps cover the five offered root actions; the additional forced-706 action was evaluated separately and its extra compute was not profiled.',
            'limits': 'Private sample frequencies are plug-in approximation probabilities, not real-world calibration. Unconditioned controls deliberately ignore consistency with earlier public history. No state values are called causal optima, and no journey improvement is inferred.',
            'cases': outputs}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(destination.relative_to(ROOT)), 'captured_histories': len(outputs)}))


if __name__ == '__main__':
    main()
