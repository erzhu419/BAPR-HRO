"""Resume root work and refine screened branches within one cumulative budget."""
from time import process_time, perf_counter

from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v52.reference import choose_bounds
from experiments.reconstruction_v53.bounds import commitment_upper_bound
from experiments.reconstruction_v53.priority import ImpactReference


class RefiningReference(ImpactReference):
    def _set_cutoff(self, cutoff):
        cutoff = float(cutoff)
        if cutoff < self.impact_cutoff:
            # Conditional intervals remain valid. Only the assertion that
            # all work at the old screening level was processed expires.
            self.settled_reach.clear()
        self.impact_cutoff = cutoff

    def _policy_value(self, actions, cheap_action):
        return (choose_bounds(actions) if self.mode in
                ('optimal', 'one_step_p0', 'two_step_p0') else actions[cheap_action])

    def _solve_actions(self, options, clock, cheap_action, initial_bounds=None):
        initial_bounds = initial_bounds or {}
        actions = {}
        settled = {}
        passes = []
        self.refinement = dict(initial_cutoff=self.impact_cutoff, passes=passes,
                               threshold_refinements=0, reflow_calls=0)

        def evaluate(action, phase):
            before = actions.get(action)
            counters = dict(self.stats)
            cpu = process_time()-self.cpu_start
            value = options[action]()
            bound = initial_bounds.get(action, {})
            value = self._intersect(value, bound.get('lower', min(clock, self.deadline)), before)
            value = self._intersect(value, min(clock, self.deadline),
                                    (min(clock, self.deadline), bound.get('upper', self.deadline), False))
            actions[action] = value
            interrupted = self.stats['budget_unknowns']-counters['budget_unknowns']
            unsupported = self.stats['unsupported_histories']-counters['unsupported_histories']
            screened = self.stats['impact_pruned_branches']-counters['impact_pruned_branches']
            available = self._available()
            settled[action] = self.impact_cutoff if not interrupted and available else None
            reason = ('complete' if value[2] else 'unsupported' if unsupported else
                      'state_limit' if self._state_limit is not None and
                      self.stats['states'] >= self._state_limit else
                      'cpu_limit' if not available else
                      'budget_remainder' if interrupted else 'screening_settled')
            passes.append(dict(phase=phase, action=str(action), cutoff=self.impact_cutoff,
                cpu_start=cpu, cpu_end=process_time()-self.cpu_start,
                states_start=counters['states'], states_end=self.stats['states'],
                state_limit=self._state_limit, cpu_limit=self._cpu_limit,
                budget_unknowns=interrupted, impact_pruned_branches=screened,
                unsupported_histories=unsupported, stop_reason=reason,
                before=None if before is None else list(before), after=list(value)))

        # Preserve V53's first-pass action order and cumulative shares.
        for index, action in enumerate(options):
            self._state_limit = (None if self.max_states is None else
                                 self.max_states*(index+1)//len(options))
            self._cpu_limit = (None if self.max_process_cpu is None else
                               self.max_process_cpu*(index+1)/len(options))
            evaluate(action, 'initial')
        self.refinement.update(first_pass_actions={str(a):list(v) for a,v in actions.items()},
            first_pass_value=list(self._policy_value(actions, cheap_action)),
            first_pass_cpu=process_time()-self.cpu_start, first_pass_states=self.stats['states'])
        # This is a return of unused shares, never a fresh budget.
        self._state_limit, self._cpu_limit = self.max_states, self.max_process_cpu
        stop = None
        while self._available():
            target = self._policy_value(actions, cheap_action)
            if target[2] or target[1]-target[0] <= 1e-8:
                stop = 'complete' if target[2] else 'numerical_zero_width'
                break
            if self.mode in ('cheap_p0', 'cheap_working'):
                candidates = [cheap_action]
            else:
                best_upper = min(v[1] for v in actions.values())
                candidates = [a for a,v in actions.items()
                              if not v[2] and v[0] <= best_upper+1e-9]
                candidates.sort(key=lambda a: -(actions[a][1]-actions[a][0]))
            if not candidates:
                stop = 'no_unresolved_value_action'
                break
            action = candidates[0]
            if settled.get(action) == self.impact_cutoff:
                if self.impact_cutoff == 0:
                    stop = 'unscreened_unresolved'
                    break
                next_cutoff = self.impact_cutoff/10
                self._set_cutoff(0. if next_cutoff < 1e-6 else next_cutoff)
                self.refinement['threshold_refinements'] += 1
            evaluate(action, 'reflow')
            self.refinement['reflow_calls'] += 1
            if passes[-1]['unsupported_histories']:
                stop = 'unsupported'
                break
        if stop is None:
            stop = ('state_limit' if self.max_states is not None and
                    self.stats['states'] >= self.max_states else 'cpu_limit')
        self.refinement.update(final_stop=stop, final_cutoff=self.impact_cutoff)
        return actions

    def solve(self, node, row, *, mode='optimal'):
        if mode not in self.MODES:
            raise ValueError(mode)
        self.mode = mode
        self.observations = self.observations or ObservationContract.from_menu(node)
        self._prepare(node)
        clock = float(row['clock'])
        depth = {'optimal':-1, 'cheap_p0':0, 'cheap_working':0,
                 'one_step_p0':0, 'two_step_p0':1}[mode]
        history = self.snapshot
        if row['kind'] == 'onboard':
            position = node.path.index(row['cid'])
            options = {'alight':lambda:self._platform(node.exits[row['cid']], clock, history, depth=depth),
                       'continue':lambda:self._ride(node, position+1, history, clock, depth)}
            specifications = {'alight':(node.exits[row['cid']], None, None, float('-inf')),
                              'continue':(node, position+1, None, float('-inf'))}
            root_choice, _ = self.engine.onboard(node, row['cid'], clock,
                self._score_history(history), method='rolling_fixed')
            cheap_action = 'alight' if root_choice else 'continue'
        elif row['kind'] == 'platform_start':
            if mode in ('one_step_p0', 'two_step_p0'):
                depth += 1
            options = {'policy':lambda:self._platform(node, row.get('entry', clock), history, depth=depth)}
            specifications = {'policy':(node, None, None, float('-inf'))}
            cheap_action = 'policy'
        elif row['kind'] == 'platform':
            entry = float(row['entry'])
            ready = entry+self.graph.stops[node.stop].min_transfer_time
            visible = tuple(row['visible']) if clock >= ready else ()
            if not set(visible).issubset(node.rides):
                raise ValueError('Root visible action is outside the frozen menu')
            options = {'wait':lambda:self._platform(node, entry, history, depth=depth, after=clock)}
            options.update({cid:(lambda cid=cid:self._ride(node.rides[cid], 0, history, clock, depth))
                            for cid in visible})
            specifications = {'wait':(node, None, ready, clock)}
            specifications.update({cid:(node.rides[cid], 0, clock, float('-inf')) for cid in visible})
            chosen, _ = self.engine.platform(node, entry, ready, self._score_history(history), visible,
                tuple(int(cid) for cid in row.get('observed', ())), clock,
                choose=self.policy.choose, method='rolling_fixed')
            cheap_action = 'wait' if chosen is None else chosen
        else:
            raise ValueError('Unsupported root decision kind')
        initial_bounds = {}
        for action, (target, position, ready, after) in specifications.items():
            initial_bounds[action] = dict(lower=self._lower(target, history, clock, position),
                                          upper=self.deadline, commitment=None)
        if mode == 'optimal':
            ceiling = None if self.max_process_cpu is None else max(0., self.max_process_cpu*.15)
            for action, (target, position, ready, after) in specifications.items():
                receipt = commitment_upper_bound(self.graph, self.root_law, self.deadline,
                    target, history, clock, position=position, ready=ready, after=after,
                    max_routes=16, available=lambda:self._available() and
                    (ceiling is None or process_time()-self.cpu_start < ceiling))
                initial_bounds[action].update(upper=receipt['upper'], commitment=receipt)
        actions = self._solve_actions(options, clock, cheap_action, initial_bounds)
        value = self._policy_value(actions, cheap_action)
        best = min(actions, key=lambda a:actions[a][1])
        competitors = [v[0] for a,v in actions.items() if a != best]
        certified = best if competitors and actions[best][1] < min(competitors)-1e-9 else None

        def receipt(bound):
            lo, hi, complete = bound
            return dict(lower=lo-clock, upper=hi-clock, complete=bool(complete),
                        mean=lo-clock if complete else None)

        result = dict(mode=mode, implementation='refining_conditional',
            target='Causal finite-menu optimum' if mode=='optimal' else 'Declared finite-lookahead or fixed cheap policy',
            physical_law='Root P0 conditioned on causal physical observations',
            scoring_law='Working W reselection' if mode=='cheap_working' else 'Consistent root P0',
            value=receipt(value), actions={str(a):receipt(v) for a,v in actions.items()},
            cheap_action=cheap_action, selected_action=best if mode in
                ('optimal','one_step_p0','two_step_p0') else cheap_action,
            certified_action=certified, clock=clock, deadline=self.deadline,
            counters=dict(self.stats), reasons=dict(self.reasons), root_refinement=self.refinement,
            root_bounds={str(a):b for a,b in initial_bounds.items()},
            impact_cutoff=self.impact_cutoff,
            impact_scope='Adaptive local work threshold; skipped mass retains full intervals, not a global error guarantee.',
            limits=dict(states=self.max_states, process_cpu=self.max_process_cpu),
            future_public_frames=0, population_scope=True,
            public_history=history.public_state(), observation_contract=self.observations.describe(),
            cost=dict(process_cpu=process_time()-self.cpu_start, wall=perf_counter()-self.wall_start),
            scope='Same causal targets. Cumulative root-share return and finer screening; ordinary reference repair, no novelty claim.')
        if {'alight','continue'}.issubset(actions):
            a,c = actions['alight'],actions['continue']
            result['difference_interval'] = [a[0]-c[1],a[1]-c[0]]
        if self.keep_decisions:
            result['decisions'] = self.records
        return result
