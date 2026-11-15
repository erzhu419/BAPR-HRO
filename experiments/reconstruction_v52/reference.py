"""Ordinary causal Bellman reference under a fixed root physical law.

Public observations branch before choices. Same-trip event indicators are
conditioned jointly; choices never see unobserved private endpoint states.
This changes the policy class, unlike the V49 fixed-pi evaluator.
"""
from collections import defaultdict
from time import perf_counter, process_time

import numpy as np

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v33.execute import Executor
from experiments.reconstruction_v34.gated_rollout import freeze_snapshot
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import PhysicalDecisionEngine, PhysicalHistory
from experiments.reconstruction_v43.reference import RootConditionalLaw
from experiments.reconstruction_v50.execute import ObservationContract
from src.evolving_trip_belief import _ChainLaw


class ConsistentP0History(PhysicalHistory):
    """Give the unchanged cheap scorer the same P0 conditioning as Bellman."""

    @classmethod
    def from_history(cls, history, root_law):
        result = history.clone()
        result.__class__ = cls
        result._root_conditional = root_law
        return result

    def law(self, trip):
        law = self._root_conditional.law(trip, self)
        self._physical_bases[trip] = self._root_conditional.roots[trip]
        return law


def point(clock):
    return float(clock), float(clock), True


def choose_bounds(values):
    """The minimum of valid action bounds bounds the causal optimum."""
    return min(v[0] for v in values.values()), min(v[1] for v in values.values()), all(v[2] for v in values.values())


class CausalReference:
    MODES = ('optimal', 'cheap_p0', 'cheap_working', 'one_step_p0', 'two_step_p0')
    IMPLEMENTATIONS = ('chronological', 'mass_terminal')

    def __init__(self, graph, model, parameters, deadline, history, observation_contract=None,
                 *, max_states=2000, max_process_cpu=2., keep_decisions=False,
                 implementation='chronological'):
        if implementation not in self.IMPLEMENTATIONS:
            raise ValueError(implementation)
        self.implementation = implementation
        self.graph, self.model, self.parameters = graph, model, parameters
        self.deadline = float(deadline)
        self.max_states, self.max_process_cpu = max_states, max_process_cpu
        self.cpu_start, self.wall_start = process_time(), perf_counter()
        self.snapshot = freeze_snapshot(history, history.message_clock)
        self.root_law = RootConditionalLaw(self.snapshot, ())
        self.observations = observation_contract
        self.keep_decisions = keep_decisions
        self.memo, self.structure_cache, self.grids = {}, {}, {}
        self.stats = dict(states=0, cache_hits=0, chance_outcomes=0, decisions=0,
            joint_trip_conditionings=0, same_trip_observation_groups=0,
            iterative_no_event_steps=0,
            terminal_integrations=0, terminal_support_outcomes=0,
            budget_unknowns=0, unsupported_histories=0, root_law_preparations=0)
        self.reasons = defaultdict(int)
        self.records = []
        self.planner = MatchedContinuationPlanner(graph, model, parameters, deadline)
        self.engine = PhysicalDecisionEngine(self.planner, variant='frontier')
        self.policy = Executor(graph, self.planner, deadline, method='rolling_fixed', waiting='combined', trace=False)
        self.mode, self._state_limit, self._cpu_limit = 'optimal', max_states, max_process_cpu

    def _prepare(self, node):
        trips = set(node.trips(self.model))
        trips.update(self.observations.trips(self.model))
        for trip in sorted(trips):
            if trip not in self.root_law.roots:
                self.root_law.roots[trip] = self.snapshot.law(trip)
                self.stats['root_law_preparations'] += 1

    def _structure(self, node):
        if node not in self.structure_cache:
            if hasattr(node, 'path'):
                value = ('ride', tuple(node.path), tuple((cid, None if tail is None else self._structure(tail))
                         for cid, tail in node.exits.items()))
            else:
                value = ('platform', node.stop, tuple((cid, self._structure(ride)) for cid, ride in node.rides.items()))
            self.structure_cache[node] = value
        return self.structure_cache[node]

    def _history_key(self, history):
        # The received soft emissions and public notices are fixed for this
        # query. Only causal hard records change inside its decision tree.
        return tuple(sorted(history._records.items()))

    def _available(self):
        return ((self._state_limit is None or self.stats['states'] < self._state_limit)
                and (self._cpu_limit is None or process_time()-self.cpu_start < self._cpu_limit))

    def _unknown(self, clock, reason='budget'):
        self.stats['budget_unknowns' if reason == 'budget' else 'unsupported_histories'] += 1
        self.reasons[reason] += 1
        return min(float(clock), self.deadline), self.deadline, False

    def _node(self, key, clock, build):
        if key in self.memo:
            self.stats['cache_hits'] += 1
            return self.memo[key]
        if not self._available():
            return self._unknown(clock)
        self.stats['states'] += 1
        try:
            result = build()
        except UnsupportedHistory as error:
            result = self._unknown(clock, str(error))
        self.memo[key] = result
        return result

    def _expect(self, outcomes, clock):
        lower = upper = consumed = 0.
        complete = True
        for mass, continuation in outcomes:
            if mass <= 0:
                continue
            if not self._available():
                remainder = max(0., 1.-consumed)
                self.stats['budget_unknowns'] += 1
                self.reasons['budget_chance_remainder'] += 1
                return lower+remainder*min(clock, self.deadline), upper+remainder*self.deadline, False
            lo, hi, exact = continuation()
            lower += mass*lo
            upper += mass*hi
            consumed += mass
            complete &= exact
            self.stats['chance_outcomes'] += 1
        return max(min(clock, self.deadline), lower), min(self.deadline, upper), complete

    def _score_history(self, history):
        if self.mode == 'cheap_working':
            return history
        return ConsistentP0History.from_history(history, self.root_law)

    def _chance_items(self, probabilities):
        items = enumerate(probabilities)
        if self.implementation == 'mass_terminal':
            return sorted(((i,p) for i,p in items if p > 0), key=lambda row:(-float(row[1]),row[0]))
        return items

    def _terminal_arrival(self, conditional, slot, probabilities, when):
        # This slot is absorbing: its observed time cannot affect another
        # choice, so integrate its unchanged conditional marginal directly.
        if not self._available():
            return self._unknown(when)
        positive = probabilities > 0
        times = conditional.times[slot][positive]
        failed = conditional.canceled[slot][positive] | (times < when) | (times >= self.deadline)
        value = float(np.dot(probabilities[positive], np.where(failed, self.deadline, times)))
        self.stats['terminal_integrations'] += 1
        self.stats['terminal_support_outcomes'] += int(np.count_nonzero(positive))
        return max(min(when, self.deadline), value), min(self.deadline, value), True

    def _choose(self, options, clock, depth, history, row, cheap_choice):
        self.stats['decisions'] += 1
        optimize = self.mode == 'optimal' or depth > 0
        if optimize:
            next_depth = depth if self.mode == 'optimal' else depth-1
            values = {action:call(next_depth) for action, call in options.items()}
            result = choose_bounds(values)
            selected = min(values, key=lambda action: values[action][1])
        else:
            selected = cheap_choice(self._score_history(history))
            values = {selected:options[selected](depth)}
            result = values[selected]
        if self.keep_decisions:
            self.records.append(dict(**row, clock=float(clock),
                public_history=history.public_state(), action_intervals={str(a):list(v[:2]) for a,v in values.items()},
                selected=selected, optimized=optimize, complete=result[2]))
        return result

    def _ride(self, ride, position, history, clock, depth):
        if position >= len(ride.path) or clock >= self.deadline:
            return point(self.deadline)
        cid = ride.path[position]
        key = ('ride', self._structure(ride), position, self._history_key(history), float(clock), depth)

        def build():
            trip, slot = self.model._positions[cid, 'departure']
            law = self.root_law.law(trip, history)
            marginal = law.marginal(slot)

            def departure(index):
                when = float(law.times[slot][index])
                if law.canceled[slot][index] or when < clock or when >= self.deadline:
                    return point(self.deadline)
                h = history.clone()
                h.observe(cid, 'departure', when)
                arrival_slot = self.model._positions[cid, 'arrival'][1]
                conditional = self.root_law.law(trip, h)
                probabilities = conditional.marginal(arrival_slot)
                if self.implementation == 'mass_terminal' and cid in ride.exits and ride.exits[cid] is None:
                    return self._terminal_arrival(conditional, arrival_slot, probabilities, when)

                def arrived(i):
                    arrival = float(conditional.times[arrival_slot][i])
                    if conditional.canceled[arrival_slot][i] or arrival < when or arrival >= self.deadline:
                        return point(self.deadline)
                    observed = h.clone()
                    observed.observe(cid, 'arrival', arrival)
                    if cid in ride.exits:
                        if ride.exits[cid] is None:
                            return point(arrival)
                        return self._onboard(ride, position, observed, arrival, depth)
                    return self._ride(ride, position+1, observed, arrival, depth)

                return self._expect(((float(p), lambda i=i:arrived(i))
                    for i,p in self._chance_items(probabilities) if p > 0), when)

            return self._expect(((float(p), lambda i=i:departure(i))
                for i,p in self._chance_items(marginal) if p > 0), clock)

        return self._node(key, clock, build)

    def _onboard(self, ride, position, history, clock, depth):
        cid = ride.path[position]
        key = ('onboard', self._structure(ride), position, self._history_key(history), float(clock), depth)
        options = dict(alight=lambda d:self._platform(ride.exits[cid], clock, history, depth=d),
                       continue_=lambda d:self._ride(ride, position+1, history, clock, d))
        options['continue'] = options.pop('continue_')

        def cheap(h):
            decision, _ = self.engine.onboard(ride, cid, clock, h, method='rolling_fixed')
            return 'alight' if decision else 'continue'

        return self._node(key, clock, lambda:self._choose(options, clock, depth, history,
            dict(kind='onboard', cid=cid), cheap))

    def _joint_trip_outcomes(self, trip, cids, clock, history):
        """Exact simultaneous-departure subset law within one physical trip."""
        law = self.root_law.law(trip, history)
        if len(cids) > 1:
            self.stats['same_trip_observation_groups'] += 1
        outcomes = [(1., (), law)]
        for cid in cids:
            slot = self.model._positions[cid, 'departure'][1]
            next_rows = []
            for prior, visible, conditional in outcomes:
                if not self._available():
                    return None
                event = (~conditional.canceled[slot]) & (conditional.times[slot] == clock)
                probability = float(conditional.marginal(slot)[event].sum())
                for yes, mass in ((False, 1.-probability), (True, probability)):
                    if mass <= 0:
                        continue
                    keep = event if yes else ~event
                    emissions = [row.copy() for row in conditional.emissions]
                    emissions[slot] = np.where(keep, emissions[slot], -np.inf)
                    child = _ChainLaw(conditional.times, conditional.initial, conditional._transition,
                                      emissions, conditional.canceled, conditional.stochastic)
                    self.stats['joint_trip_conditionings'] += 1
                    next_rows.append((prior*mass, (*visible, cid) if yes else visible, child))
            outcomes = next_rows
        return sorted(((p, visible) for p, visible, _ in outcomes), key=lambda r:(-r[0],r[1]))

    def _joint_outcomes(self, active, clock, history):
        grouped = defaultdict(list)
        for cid in active:
            trip, slot = self.model._positions[cid, 'departure']
            law = self.root_law.law(trip, history)
            if np.any((law.times[slot] == clock) & ~law.canceled[slot] & (law.marginal(slot) > 0)):
                grouped[trip].append(cid)
        groups = []
        for trip, cids in sorted(grouped.items()):
            rows = self._joint_trip_outcomes(trip, cids, clock, history)
            if rows is None:
                return None
            groups.append(rows)

        def combine(i, probability, visible):
            if i == len(groups):
                yield probability, visible
                return
            for mass, seen in groups[i]:
                yield from combine(i+1, probability*mass, (*visible, *seen))

        return combine(0, 1., ())

    def _platform(self, platform, entry, history, *, depth, after=float('-inf')):
        if entry >= self.deadline:
            return point(self.deadline)
        cids = tuple(self.observations.cids_at(platform.stop))
        if not set(platform.rides).issubset(cids):
            raise ValueError('Every legal action must be in the common observation contract')
        grid_key = self._structure(platform), float(entry)
        if grid_key not in self.grids:
            times = set()
            for cid in cids:
                trip, slot = self.model._positions[cid, 'departure']
                law = self.root_law.roots[trip]
                times.update(float(t) for t,cancel,p in zip(law.times[slot],law.canceled[slot],law.marginal(slot),strict=True)
                             if not cancel and p > 0 and entry <= t < self.deadline)
            self.grids[grid_key] = tuple(sorted(times))
        times = self.grids[grid_key]
        ready = entry+self.graph.stops[platform.stop].min_transfer_time
        initial_seen = tuple(cid for cid in cids if any(r[:3] == (cid, 'departure', 'observed')
                         and entry <= r[3] <= after for r in history._records.get(self.model._positions[cid,'departure'][0],())))

        def observe(j, seen, h, d, visible):
            clock=times[j]
            observed=h.clone()
            observed.observe_platform(cids,entry,clock,{cid:clock for cid in visible})
            next_seen=tuple(sorted((*seen,*visible)))
            available=tuple(cid for cid in platform.rides if cid in visible)
            if clock<ready or not available:
                return event(j+1,next_seen,observed,d)
            options={'wait':lambda dd:event(j+1,next_seen,observed,dd)}
            options.update({cid:(lambda dd,cid=cid:self._ride(platform.rides[cid],0,observed,clock,dd))
                            for cid in available})

            def cheap(score_history):
                chosen,_=self.engine.platform(platform,entry,ready,score_history,available,
                    tuple(cid for cid in next_seen if cid in platform.rides),clock,
                    choose=self.policy.choose,method='rolling_fixed')
                return 'wait' if chosen is None else chosen

            return self._choose(options,clock,d,observed,
                dict(kind='platform',stop=platform.stop,entry=float(entry),ready=float(ready),
                     visible=list(available),observed_visible=list(visible)),cheap)

        def event(j,seen,h,d):
            # Every recursive event call below observes at least one new
            # vehicle. A long finite clock support with no event is evaluated
            # as an iterative affine chain, not a Python recursion chain.
            chain=[]
            while True:
                if j==len(times) or not set(platform.rides).difference(seen):
                    tail=point(self.deadline)
                    break
                clock=times[j]
                key=('event',grid_key,j,seen,self._history_key(h),d)
                if key in self.memo:
                    self.stats['cache_hits']+=1
                    tail=self.memo[key]
                    break
                if not self._available():
                    tail=self._unknown(clock)
                    break
                self.stats['states']+=1
                lower=upper=consumed=no_event=0.
                complete=True
                try:
                    outcomes=self._joint_outcomes(tuple(cid for cid in cids if cid not in seen),clock,h)
                    if outcomes is None:
                        tail=self._unknown(clock)
                        self.memo[key]=tail
                        break
                    for mass,visible in outcomes:
                        if mass<=0:
                            continue
                        if not self._available():
                            remainder=max(0.,1.-consumed)
                            lower+=remainder*clock
                            upper+=remainder*self.deadline
                            complete=False
                            self.stats['budget_unknowns']+=1
                            self.reasons['budget_chance_remainder']+=1
                            break
                        if visible:
                            lo,hi,exact=observe(j,seen,h,d,visible)
                            lower+=mass*lo
                            upper+=mass*hi
                            complete &= exact
                        else:
                            no_event+=mass
                        consumed+=mass
                        self.stats['chance_outcomes']+=1
                except UnsupportedHistory as error:
                    tail=self._unknown(clock,str(error))
                    self.memo[key]=tail
                    break
                chain.append((key,clock,lower,upper,no_event,complete))
                if no_event==0:
                    tail=point(self.deadline)
                    break
                h=h.clone()
                h.observe_platform(cids,entry,clock,{})
                j+=1
                self.stats['iterative_no_event_steps']+=1
            for key,clock,lower,upper,mass,complete in reversed(chain):
                tail=(max(clock,lower+mass*tail[0]),min(self.deadline,upper+mass*tail[1]),
                      complete and (mass==0 or tail[2]))
                self.memo[key]=tail
            return tail

        start = int(np.searchsorted(times, max(float(after), float(entry)-1.), side='right')) if after != float('-inf') else 0
        return event(start, initial_seen, history, depth)

    def solve(self, node, row, *, mode='optimal'):
        if mode not in self.MODES:
            raise ValueError(mode)
        self.mode = mode
        self.observations = self.observations or ObservationContract.from_menu(node)
        self._prepare(node)
        clock = float(row['clock'])
        depth = {'optimal':-1, 'cheap_p0':0, 'cheap_working':0, 'one_step_p0':0, 'two_step_p0':1}[mode]
        history = self.snapshot
        if row['kind'] == 'onboard':
            position = node.path.index(row['cid'])
            options = dict(alight=lambda:self._platform(node.exits[row['cid']], clock, history, depth=depth),
                           continue_=lambda:self._ride(node, position+1, history, clock, depth))
            options['continue'] = options.pop('continue_')
            root_choice, _ = self.engine.onboard(node, row['cid'], clock, self._score_history(history), method='rolling_fixed')
            cheap_action = 'alight' if root_choice else 'continue'
        elif row['kind'] == 'platform_start':
            if mode in ('one_step_p0','two_step_p0'):
                depth += 1
            options = {'policy':lambda:self._platform(node, row.get('entry', clock), history, depth=depth)}
            cheap_action = 'policy'
        elif row['kind'] == 'platform':
            entry = float(row['entry']); ready = entry+self.graph.stops[node.stop].min_transfer_time
            visible = tuple(row['visible']) if clock >= ready else ()
            if not set(visible).issubset(node.rides):
                raise ValueError('Root visible action is outside the frozen menu')
            options = {'wait':lambda:self._platform(node, entry, history, depth=depth, after=clock)}
            options.update({cid:(lambda cid=cid:self._ride(node.rides[cid], 0, history, clock, depth)) for cid in visible})
            chosen, _ = self.engine.platform(node, entry, ready, self._score_history(history), visible,
                tuple(int(cid) for cid in row.get('observed', ())), clock, choose=self.policy.choose, method='rolling_fixed')
            cheap_action = 'wait' if chosen is None else chosen
        else:
            raise ValueError('Unsupported root decision kind')
        actions = {}
        for i,(action, call) in enumerate(options.items()):
            # Give each forced root action a declared share before ordinary
            # recursive evaluation. Unused previous shares remain available.
            self._state_limit = None if self.max_states is None else self.max_states*(i+1)//len(options)
            self._cpu_limit = None if self.max_process_cpu is None else self.max_process_cpu*(i+1)/len(options)
            actions[action] = call()
        best = min(actions, key=lambda a:actions[a][1])
        optimize_root = mode in ('optimal','one_step_p0','two_step_p0')
        value = choose_bounds(actions) if optimize_root else actions[cheap_action]
        competitors = [v[0] for a,v in actions.items() if a != best]
        certified = best if competitors and actions[best][1] < min(competitors)-1e-9 else None

        def receipt(bound):
            lo, hi, complete = bound
            return dict(lower=lo-clock, upper=hi-clock, complete=bool(complete),
                        mean=lo-clock if complete else None)

        result = dict(mode=mode, implementation=self.implementation,
            target='Causal finite-menu optimum' if mode=='optimal' else 'Declared finite-lookahead or fixed cheap policy',
            physical_law='Root P0 conditioned on causal physical observations',
            scoring_law='Working W reselection' if mode=='cheap_working' else 'Consistent root P0',
            value=receipt(value), actions={str(a):receipt(v) for a,v in actions.items()},
            cheap_action=cheap_action, selected_action=best if optimize_root else cheap_action,
            certified_action=certified, clock=clock, deadline=self.deadline,
            counters=dict(self.stats), reasons=dict(self.reasons),
            limits=dict(states=self.max_states, process_cpu=self.max_process_cpu),
            future_public_frames=0, population_scope=True,
            public_history=self.snapshot.public_state(), observation_contract=self.observations.describe(),
            cost=dict(process_cpu=process_time()-self.cpu_start, wall=perf_counter()-self.wall_start),
            scope='Ordinary bounded Bellman recursion. A nonzero unresolved interval or implementation cost does not establish an intrinsic barrier for optimized DP.')
        if {'alight','continue'}.issubset(actions):
            a,c=actions['alight'],actions['continue']
            result['difference_interval']=[a[0]-c[1],a[1]-c[0]]
        if self.keep_decisions:
            result['decisions']=self.records
        return result
