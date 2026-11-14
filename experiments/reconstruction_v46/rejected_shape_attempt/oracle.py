"""Causal full-policy evaluation of independent-root one-transfer tails.

P0 supplies physical probabilities, W supplies the unchanged policy's scores.
WAIT marks, observed BOARD scores and actual physical terminal values remain
separate. Unsupported structures and exhausted budgets retain bounded mass.
"""
from time import perf_counter, process_time

import numpy as np

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v34.gated_rollout import freeze_snapshot
from experiments.reconstruction_v39.state import raw_history_key
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import PhysicalDecisionEngine, posterior_transition
from experiments.reconstruction_v43.reference import RootConditionalLaw, departure_distribution
from experiments.reconstruction_v33.nodes import Ride


def terminal(ride):
    return set(ride.exits) == {ride.path[-1]} and ride.exits[ride.path[-1]] is None


def structural(model, platform):
    """Admit the checked mixed one-transfer class, not shared-current roots."""
    roots = {cid: model._positions[cid, 'departure'][0] for cid in platform.rides}
    reasons = []
    if len(set(roots.values())) != len(roots):
        reasons.append('current_root_trips_not_distinct')
    root_set = set(roots.values())
    for cid, ride in platform.rides.items():
        positions = [model._positions[c, kind] for c in ride.path for kind in ('departure', 'arrival')]
        if (any(trip != roots[cid] for trip, _ in positions)
                or any(b[1] != a[1]+1 for a,b in zip(positions, positions[1:]))):
            reasons.append(f'noncontiguous_ride:{cid}')
        for stop_cid, tail in ride.exits.items():
            if tail is None:
                if stop_cid != ride.path[-1]:
                    reasons.append(f'early_destination:{cid}')
                continue
            if root_set & tail.trips(model):
                reasons.append(f'root_trip_reappears_in_tail:{cid}:{stop_cid}')
            if any(not terminal(child) for child in tail.rides.values()):
                reasons.append(f'more_than_one_transfer:{cid}:{stop_cid}')
            leaf = structural(model, tail) if all(terminal(child) for child in tail.rides.values()) else None
            if leaf and not leaf['eligible']:
                reasons.extend(leaf['reasons'])
    return dict(eligible=not reasons, reasons=reasons, candidates=len(roots),
                terminal_candidates=sum(terminal(ride) for ride in platform.rides.values()))


def point(value):
    return np.array([float(value), float(value), 0.])


def unknown(clock, deadline):
    return np.array([min(float(clock), deadline), float(deadline), 1.])


def receipt(value, reason=None):
    lo, hi, mass = map(float, value)
    return dict(lower=lo, upper=hi, mean=lo if mass == 0 else None,
                unresolved_probability=mass, reason=reason)


class FullPolicyOracle:
    def __init__(self, graph, model, parameters, deadline, root_history, *,
                 max_compiles=None, max_process_cpu=None):
        self.graph, self.model, self.parameters = graph, model, parameters
        self.deadline = float(deadline)
        self.cpu_start, self.wall_start = process_time(), perf_counter()
        self.max_compiles, self.max_process_cpu = max_compiles, max_process_cpu
        self._compile_limit, self._cpu_limit = max_compiles, max_process_cpu
        self.snapshot = freeze_snapshot(root_history, root_history.message_clock)
        self.root_law = RootConditionalLaw(self.snapshot, ())
        self.planner = MatchedContinuationPlanner(graph, model, parameters, deadline)
        self.engine = PhysicalDecisionEngine(self.planner, variant='frontier')
        self.platform_cache, self.ride_cache, self.board_cache = {}, {}, {}
        self.stats = dict(charged_compiles=0, platform_solves=0, wait_compiles=0,
            board_compiles=0, onboard_compiles=0, platform_hits=0, ride_hits=0,
            board_hits=0, positive_event_branches=0, budget_refusals=0,
            structure_refusals=0, unsupported_scores=0, physical_transition_cells=0)

    def _prepare(self, node):
        for trip in node.trips(self.model):
            if trip not in self.root_law.roots:
                self.root_law.roots[trip] = self.snapshot.law(trip)

    def _spend(self, kind):
        if ((self._compile_limit is not None and self.stats['charged_compiles'] >= self._compile_limit)
                or (self._cpu_limit is not None and process_time()-self.cpu_start >= self._cpu_limit)):
            self.stats['budget_refusals'] += 1
            return False
        self.stats['charged_compiles'] += 1
        self.stats[kind] += 1
        return True

    def _key(self, node, history):
        return self.planner.structure(node), raw_history_key(history, node.trips(self.model))[0]

    def _board(self, ride, departure, history):
        h = history.clone()
        h.observe(ride.first_cid, 'departure', departure)
        key = self._key(ride, h), float(departure)
        if key in self.board_cache:
            self.stats['board_hits'] += 1
            return self.board_cache[key]
        if not self._spend('board_compiles'):
            return None
        try:
            value = self.planner.compile_ride(ride, h, 'rolling_fixed').score(departure)
        except UnsupportedHistory:
            self.stats['unsupported_scores'] += 1
            value = None
        self.board_cache[key] = value
        return value

    def ride_marks(self, ride, history=None):
        """All root-departure physical values, computed in one backward pass."""
        self._prepare(ride)
        history = self.snapshot if history is None else history
        key = self._key(ride, history)
        if key in self.ride_cache:
            self.stats['ride_hits'] += 1
            return self.ride_cache[key]
        trip, first = self.model._positions[ride.first_cid, 'departure']
        last = self.model._positions[ride.path[-1], 'arrival'][1]
        law = self.root_law.law(trip, history)
        by_slot = {self.model._positions[cid, kind][1]:(cid,kind)
                   for cid in ride.path for kind in ('departure','arrival')}
        following = None
        for slot in range(last, first-1, -1):
            clocks, canceled = law.times[slot], law.canceled[slot]
            if following is None:
                values = np.tile(point(self.deadline), (len(clocks),1))
            else:
                matrix = posterior_transition(law, slot)
                self.stats['physical_transition_cells'] += matrix.size
                values = matrix @ following
            cid, kind = by_slot[slot]
            active = (~canceled) & (clocks < self.deadline) & (law.marginal(slot) > 0)
            values[~active] = point(self.deadline)
            if kind == 'arrival' and cid in ride.exits:
                tail = ride.exits[cid]
                if tail is None:
                    values[active,0] = values[active,1] = clocks[active]
                    values[active,2] = 0.
                else:
                    for i in np.flatnonzero(active):
                        clock = float(clocks[i])
                        if not self._spend('onboard_compiles'):
                            values[i] = unknown(clock,self.deadline)
                            continue
                        h = history.clone()
                        h.observe(cid,'arrival',clock)
                        try:
                            alight, _ = self.engine.onboard(ride,cid,clock,h,method='rolling_fixed')
                        except UnsupportedHistory:
                            self.stats['unsupported_scores'] += 1
                            values[i] = unknown(clock,self.deadline)
                            continue
                        if alight:
                            values[i] = self._platform(tail,clock,history)
            following = values
        result = (law.times[first].copy(), following.copy())
        if not np.any(following[:,2]>0):
            self.ride_cache[key] = result
        return result

    def _platform(self, platform, entry, history):
        key = self._key(platform,history),float(entry)
        if key in self.platform_cache:
            self.stats['platform_hits'] += 1
            return self.platform_cache[key]
        if entry >= self.deadline:
            return point(self.deadline)
        check = structural(self.model,platform)
        if not check['eligible']:
            self.stats['structure_refusals'] += 1
            return unknown(entry,self.deadline)
        if not self._spend('platform_solves') or not self._spend('wait_compiles'):
            return unknown(entry,self.deadline)
        ready = entry+self.graph.stops[platform.stop].min_transfer_time
        try:
            reference = self.planner.platform_reference(platform,history,entry,ready,'rolling_fixed')
        except UnsupportedHistory:
            self.stats['unsupported_scores'] += 1
            return unknown(entry,self.deadline)
        cids = tuple(reference.cids)
        physical = {cid:self.root_law.departure_distribution(cid,history) for cid in cids}
        times = np.array(sorted({float(t) for source in (reference.distributions,physical)
            for cid in cids for t,p in zip(*source[cid][:2],strict=True)
            if p > 0 and entry <= t < self.deadline}))
        nt,nc = len(times),len(cids)
        hazards = []
        for source in (reference.distributions,physical):
            hazard = np.zeros((nt,nc))
            for i,cid in enumerate(cids):
                t,p,cancel = source[cid]
                tails = np.r_[np.cumsum(p[::-1])[::-1],0.]
                absent = cancel+p[t<entry].sum()
                survival = absent+tails[np.searchsorted(t,times,side='left')]
                selected = (p>0)&(t>=entry)&(t<self.deadline)
                np.add.at(hazard[:,i],np.searchsorted(times,t[selected]),p[selected])
                np.divide(hazard[:,i],survival,out=hazard[:,i],where=survival>0)
            hazards.append(hazard)
        working_hazard,physical_hazard = hazards
        wait_marks = np.full((nt,nc),self.deadline)
        board = np.full((nt,nc),np.nan)
        truth = np.tile(unknown(entry,self.deadline),(nt,nc,1))
        single = np.full((nt,nc),self.deadline)
        for i,cid in enumerate(cids):
            clocks,p,marks = reference.marks[cid]
            # Only these unobserved marks enter the original hypothetical WAIT.
            wait_marks[np.searchsorted(times,clocks),i] = marks
            gains = np.r_[np.cumsum((p*(self.deadline-marks))[::-1])[::-1],0.]
            root_times,root_mass,cancel = reference.distributions[cid]
            tails = np.r_[np.cumsum(root_mass[::-1])[::-1],0.]
            survival = cancel+root_mass[root_times<entry].sum()+tails[np.searchsorted(root_times,times,side='right')]
            gain = gains[np.searchsorted(clocks,times,side='right')]
            single[:,i] -= np.divide(gain,survival,out=np.zeros_like(gain),where=survival>0)
            ride = platform.rides[cid]
            truth_times,true_values = self.ride_marks(ride,history)
            physical_times,mass,_ = physical[cid]
            for t in physical_times[(mass>0)&(physical_times>=ready)&(physical_times<self.deadline)]:
                j = int(np.searchsorted(times,t))
                value = self._board(ride,float(t),history)
                if value is not None:
                    board[j,i] = value
                index = np.flatnonzero(truth_times == t)
                # One operating state exists at each exact physical clock.
                operating = ~self.root_law.law(self.model._positions[cid,'departure'][0],history).canceled[
                    self.model._positions[cid,'departure'][1]]
                index = index[operating[index]]
                if len(index) != 1:
                    raise ValueError('Physical departure mark has no unique operating state')
                truth[j,i] = true_values[index[0]]
        nstates = 1<<nc
        first = np.full((nt+1,nstates),self.deadline)
        values = np.tile(point(self.deadline),(nt+1,nstates,1))
        bits = [[i for i in range(nc) if mask&(1<<i)] for mask in range(nstates)]
        subsets=[]
        for active in bits:
            outcomes=np.array([0],dtype=int)
            for i in active:outcomes=np.r_[outcomes,outcomes|(1<<i)]
            subsets.append(outcomes)
        rank = sorted(range(nc),key=lambda i:reference.ranks[cids[i]])
        for j in range(nt-1,-1,-1):
            legal = times[j]>=ready
            for mask in range(nstates):
                wp,pp = np.ones(1),np.ones(1)
                for i in bits[mask]:
                    a,b=working_hazard[j,i],physical_hazard[j,i]
                    wp=np.r_[wp*(1-a),wp*a]
                    pp=np.r_[pp*(1-b),pp*b]
                departed=subsets[mask];remaining=mask^departed
                first_cost=first[j+1,remaining].copy()
                local=values[j+1,remaining].copy()
                for k,seen in enumerate(departed):
                    if not legal or not seen:continue
                    selected_rank=next(i for i in rank if seen&(1<<i))
                    first_cost[k]=wait_marks[j,selected_rank]
                    if pp[k]<=0:continue
                    self.stats['positive_event_branches'] += 1
                    visible=bits[seen]
                    if any(not np.isfinite(board[j,i]) for i in visible):
                        local[k]=unknown(times[j],self.deadline)
                        continue
                    best=min(visible,key=lambda i:(board[j,i],reference.ranks[cids[i]]))
                    target=min((single[j,i] for i in bits[remaining[k]]),default=self.deadline)
                    waiting=min(target,first[j+1,remaining[k]])
                    if board[j,best]<=waiting+1e-9:
                        local[k]=truth[j,best]
                first[j,mask]=wp@first_cost
                values[j,mask]=pp@local
        if values[0,-1,2] == 0:
            self.platform_cache[key]=values[0,-1].copy()
        return values[0,-1]

    def platform(self, platform, entry, history=None):
        self._prepare(platform)
        history=self.snapshot if history is None else history
        check=structural(self.model,platform)
        result=receipt(self._platform(platform,float(entry),history),
            None if check['eligible'] else ';'.join(check['reasons']))
        return dict(**result,structure=check,scope='P0 expectation of original complete pi',
                    counters=dict(self.stats),cost=self.cost())

    def forced(self,node,row):
        """Both current actions, in absolute time, with all incomplete mass."""
        self._prepare(node)
        clock=float(row['clock']);cid=row['cid']
        from experiments.reconstruction_v45.prefix import structure_gate
        check=structure_gate(self.model,node,row)
        allocation={}
        if not check['eligible']:
            actions={a:receipt(unknown(clock,self.deadline),'head structure not admitted') for a in ('alight','continue')}
        else:
            # Reserve at least half of both declared resources for CONTINUE.
            self._compile_limit=None if self.max_compiles is None else self.max_compiles//2
            self._cpu_limit=None if self.max_process_cpu is None else self.max_process_cpu/2
            began_cpu,began_count=process_time(),self.stats['charged_compiles']
            actions={'alight':receipt(self._platform(node.exits[cid],clock,self.snapshot))}
            allocation['alight']=dict(charged_compiles=self.stats['charged_compiles']-began_count,
                process_cpu=process_time()-began_cpu,initial_compile_limit=self._compile_limit,
                initial_cpu_limit=self._cpu_limit)
            self._compile_limit,self._cpu_limit=self.max_compiles,self.max_process_cpu
            began_cpu,began_count=process_time(),self.stats['charged_compiles']
            remaining=node.path[node.path.index(cid)+1:]
            if not remaining:
                value=point(self.deadline)
            else:
                ride=Ride(remaining[0],remaining,{c:node.exits[c] for c in remaining if c in node.exits})
                clocks,marks=self.ride_marks(ride,self.snapshot)
                trip,slot=self.model._positions[remaining[0],'departure']
                mass=self.root_law.roots[trip].marginal(slot)
                value=mass@marks
            actions['continue']=receipt(value)
            allocation['continue']=dict(charged_compiles=self.stats['charged_compiles']-began_count,
                process_cpu=process_time()-began_cpu,global_compile_limit=self.max_compiles,
                global_cpu_limit=self.max_process_cpu)
        a,c=actions['alight'],actions['continue']
        interval=[a['lower']-c['upper'],a['upper']-c['lower']]
        return dict(actions=actions,difference_interval=interval,
            certified_action='alight' if interval[1]<-1e-9 else 'continue' if interval[0]>1e-9 else None,
            complete=all(value['mean'] is not None for value in actions.values()),
            root_structure=check,clock=clock,deadline=self.deadline,
            counters=dict(self.stats),cost=self.cost(),budget_allocation=allocation,
            scope='Exact or bounded original full-pi population values under fixed P0')

    def cost(self):
        return dict(process_cpu=process_time()-self.cpu_start,wall=perf_counter()-self.wall_start,
                    planner=dict(self.planner.stats),root_law=dict(self.root_law.stats))
