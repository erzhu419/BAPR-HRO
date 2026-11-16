"""Development execution contract with commands issued before departure.

Platform control is a new common policy: publish either first legal offer or a
named target, then consume the instruction at a departure event. A command that
finishes after that event cannot board retrospectively. Arrival information and
future messages are not added. Platform-pair transfer times are static planned
values with the declared same-platform assumption.
"""

from bisect import bisect_right
from math import inf, isfinite, nextafter
from time import perf_counter, process_time

import numpy as np

from experiments.bounded_gate_5d_v1.pair_contract import (
    BoundedArmExecutor as LegacyBoundedArmExecutor,
    DayPlatformIndex, PairDecisionEngine, PairMatchedPlanner, PairMenuOpportunity,
    PairPlatform, PairRecursivePlanner,
    PairSnapshotExecutor as LegacyPairSnapshotExecutor,
    decorate_menu_with_mct,
)
from experiments.g2_reset_v1.certified_gate import should_invoke_rollout
from experiments.reconstruction_v34.execute import MissingEndpoint
from experiments.reconstruction_v37.controller import has_legal_continue
from experiments.reconstruction_v44.matched import choose as choose_point_estimate


def choose_command(planner, platform, history, entry, clock, seen, *, csa=None):
    """Choose a future first-offer/target instruction from public information."""
    rides = {cid: ride for cid, ride in platform.rides.items() if cid not in seen}
    ready = {cid: max(float(clock), float(entry) + platform.mct_for(cid)) for cid in rides}
    ranks = planner.ranks(platform)
    if not rides:
        return dict(mode='wait', target=None, ranks={}, ready_by_cid={}), dict(values={})
    view = PairPlatform(platform.stop, rides, incoming_cid=platform.incoming_cid,
                        transfer_times={cid: ready[cid]-entry for cid in rides},
                        default_mct=platform.default_mct)
    if csa is not None:
        snapshot = csa._snapshot(view, history, clock)
        values = {}
        for cid, ride in rides.items():
            departure = snapshot['means'].get((cid, 'departure'))
            if departure is not None and departure >= ready[cid] and departure < planner.deadline:
                values[cid] = csa._scan(snapshot, departure, ride=ride, position=0)
        target = min(values, key=lambda cid: (values[cid], ranks[cid]), default=None)
        mode = 'target' if target is not None and values[target] < planner.deadline else 'wait'
        receipt = dict(method='preissued_mean_target', values=values)
    else:
        compiler = planner.platform_compiler(view, history, 'rolling_fixed')
        reference = PairMenuOpportunity(compiler, {cid:ranks[cid] for cid in rides},
                                        entry, ready, planner.deadline)
        # The current departure, if observed, is already in seen. Initial entry
        # may coincide with an offer only for the ideal zero-latency reference.
        first = reference.wait_score(nextafter(float(clock), -inf), ())
        values = {}
        for cid in rides:
            # The reference has already removed canceled/zero-probability
            # support and compiled exactly these legal boarding values.
            _, mass, scores = reference.marks[cid]
            values[cid] = float(planner.deadline - mass @ (planner.deadline-scores))
        target = min(values, key=lambda cid: (values[cid], ranks[cid]))
        mode = 'target' if values[target] < first-1e-9 else 'first'
        if min(first, values[target]) >= planner.deadline:
            mode = 'wait'
        receipt = dict(method='preissued_first_or_target', first_offer=first, values=values)
    command = dict(mode=mode, target=target if mode=='target' else None,
                   ranks={cid:ranks[cid] for cid in rides}, ready_by_cid=ready)
    return command, receipt


def command_choice(command, visible, clock, *, ideal=False):
    """Consume a previously issued instruction, with no new planning."""
    if command is None or command['mode']=='wait':
        return None
    issued = command['issued_at']
    if issued > clock or (issued == clock and not ideal):
        return None
    eligible = [cid for cid in visible if cid in command['ranks']
                and command['ready_by_cid'][cid] <= clock]
    if command['mode']=='target':
        return command['target'] if command['target'] in eligible else None
    return min(eligible, key=lambda cid:command['ranks'][cid], default=None)


class PreissuedPlatformMixin:
    """One platform policy for archived journeys and zero-latency worlds."""
    command_latency = False

    def platform(self, platform, entry, history, world, *, after=-inf):
        if not isinstance(platform, PairPlatform):
            platform = PairPlatform(platform.stop, platform.rides,
                                    default_mct=self.graph.stops[platform.stop].min_transfer_time)
        cids = (self.observation_contract.cids_at(platform.stop)
                if self.observation_contract is not None else tuple(platform.rides))
        departures = {cid:world.event(cid, 'departure') for cid in cids}
        events = sorted({t for t,canceled,_ in departures.values()
                         if not canceled and entry <= t < self.deadline and t > after})
        frames = (getattr(history, 'frames', ()) if self.command_latency else ())
        messages = sorted({f['receipt_seconds'] for f in frames
                           if entry <= f['receipt_seconds'] < self.deadline})
        event_pos = 0
        command = None
        now = max(float(entry), getattr(self, '_controller_available_at', float(entry)))
        ideal = not self.command_latency or self.clock_contract=='ideal'

        def observe(clock):
            observed = {cid:t for cid,(t,canceled,_) in departures.items()
                        if not canceled and entry <= t <= clock}
            history.observe_platform(cids, entry, clock, observed)
            return observed, self.observation_evidence(history, clock)

        def consume_through(limit):
            nonlocal event_pos
            while event_pos < len(events) and events[event_pos] <= limit:
                clock = events[event_pos]; event_pos += 1
                observed,evidence = observe(clock)
                visible = tuple(cid for cid in platform.rides
                                if not departures[cid][1] and departures[cid][0]==clock
                                and entry+platform.mct_for(cid)<=clock)
                selected = command_choice(command, visible, clock, ideal=ideal)
                self.record(dict(kind='platform', stop=platform.stop, entry=entry,
                    ready=platform.min_ready(entry,self.deadline), clock=clock,
                    visible=visible, observed=observed, choice=selected,
                    command=command, receipt=dict(method='consume_preissued_command'), **evidence))
                self.counts['platform'] += bool(visible)
                if selected is not None:
                    return selected,clock
            return None

        while now < self.deadline:
            # While the controller was occupied, only an already-issued command
            # could act. Initial departures before first publication are lost.
            hit = consume_through(nextafter(now,-inf) if ideal and command is None else now)
            if hit is not None:
                return self.ride(platform.rides[hit[0]], hit[1], history, world)
            query_cpu_start = process_time()
            observed,evidence = observe(now)
            seen = {cid for cid in observed if cid in platform.rides}
            if ideal and command is None:
                # A zero-time reference can issue at initial entry itself.
                seen -= {cid for cid,t in observed.items() if t==now}
            proposed,receipt = choose_command(self.planner, platform, history, entry, now,
                                               seen, csa=getattr(self,'csa',None))
            cpu = process_time()-query_cpu_start
            issued = now + cpu if self.command_latency and not ideal else now
            proposed['issued_at'] = issued
            if self.command_latency:
                self._controller_available_at = issued
                self.decision_costs.append(dict(kind='platform_command',clock=now,process_cpu=cpu))
                self.states.append(dict(kind='platform_command',clock=now,entry=entry,
                    public_history=history.public_state(),selected_action=proposed,
                    root_index=self.root_index,deadline=self.deadline))
                self.publications.append(dict(kind='platform_command',opens_at=now,
                    issued_at=issued,base_cpu_seconds=cpu,command=proposed,
                    scope='Command applies only to departures after publication.'))
            # A previously issued command remains live during computation.
            before_events = event_pos
            hit = consume_through(nextafter(issued,-inf) if ideal else issued)
            if hit is not None:
                return self.ride(platform.rides[hit[0]], hit[1], history, world)
            command = proposed
            self.record(dict(kind='platform_command',clock=now,issued_at=issued,
                             entry=entry,command=command,receipt=receipt,**evidence))
            if ideal:
                hit = consume_through(issued)
                if hit is not None:
                    return self.ride(platform.rides[hit[0]],hit[1],history,world)
            information_arrived = (event_pos>before_events or
                bisect_right(messages,issued)>bisect_right(messages,now))
            if information_arrived and issued>now:
                # Coalesce all arrivals of information during this computation
                # into one update at completion, never replay past queries.
                now=issued
                continue
            future = []
            if event_pos < len(events):
                future.append(events[event_pos])
            j = bisect_right(messages, issued)
            if j < len(messages):
                future.append(messages[j])
            if not future:
                break
            now = min(future)
        return self.deadline,'no_legal_offer'


class PairSnapshotExecutor(PreissuedPlatformMixin, LegacyPairSnapshotExecutor):
    """Model-world continuation with the same commands and zero compute delay."""

    def forced(self,node,row,action,history,world):
        from experiments.reconstruction_v34.conditional_reference import ProjectionIncompatible

        requested=action
        began,cpu=perf_counter(),process_time()
        try:
            if row['kind']=='onboard':
                position=node.path.index(row['cid'])
                if position+1==len(node.path):
                    action='alight'
                else:
                    next_cid=node.path[position+1]
                    departure,canceled,_=world.event(next_cid,'departure')
                    if canceled:
                        raise ProjectionIncompatible(
                            f'{next_cid}:canceled_departure_has_no_door_closing_endpoint')
                    if row['clock']>=departure:
                        action='continue'
            outcome=super().forced(node,row,action,history,world)
        except ProjectionIncompatible as error:
            action=None
            outcome=dict(loss=None,status='projection_incompatible:'+str(error),
                trace=self.trace,decisions=self.counts,compute_wall=self.compute_wall,
                total_wall=perf_counter()-began,total_process_cpu=process_time()-cpu)
        outcome.update(requested_root_action=requested,executed_root_action=action)
        return outcome

    def ride(self,ride,ready,history,world):
        from experiments.reconstruction_v34.conditional_reference import ProjectionIncompatible

        previous,vehicle=ready,None
        for idx,cid in enumerate(ride.path):
            departure,canceled,identity=world.event(cid,'departure')
            if canceled or departure>=self.deadline:
                return self.deadline,'service_or_deadline_failure'
            if departure<previous:
                raise ProjectionIncompatible(f'{cid}:departure={departure}<known_previous={previous}')
            if vehicle is not None and identity!=vehicle:
                raise ValueError('A sampled ride changed its physical vehicle')
            vehicle=identity
            arrival,canceled,arrival_identity=world.event(cid,'arrival')
            if canceled:return self.deadline,'service_failure'
            if arrival_identity!=vehicle:
                raise ValueError('Sampled arrival changed its physical vehicle')
            if arrival<departure:
                raise ProjectionIncompatible(f'{cid}:arrival={arrival}<departure={departure}')
            if arrival>=self.deadline:return self.deadline,'deadline'
            history.observe(cid,'departure',departure);history.observe(cid,'arrival',arrival)
            previous=arrival
            if cid not in ride.exits:continue
            platform=ride.exits[cid]
            if platform is None:return arrival,'arrived'
            if idx+1==len(ride.path):
                self.record(dict(kind='onboard',cid=cid,clock=arrival,choice='alight',
                                 receipt=dict(method='mandatory_menu_exit')))
                return self.platform(platform,arrival,history,world)
            next_departure,next_canceled,_=world.event(ride.path[idx+1],'departure')
            if next_canceled:
                raise ProjectionIncompatible(
                    f'{ride.path[idx+1]}:canceled_departure_has_no_door_closing_endpoint')
            if arrival>=next_departure:
                self.record(dict(kind='controller_busy',cid=cid,clock=arrival,
                                 departure_deadline=next_departure,choice='continue'))
                continue
            alight,receipt=self.engine.onboard(ride,cid,arrival,history,method=self.method)
            self.counts['onboard']+=1
            self.record(dict(kind='onboard',cid=cid,clock=arrival,
                             choice='alight' if alight else 'continue',receipt=receipt))
            if alight:return self.platform(platform,arrival,history,world)
        return self.deadline,'exhausted_ride'


class BoundedArmExecutor(PreissuedPlatformMixin, LegacyBoundedArmExecutor):
    command_latency = True

    def _publish_action(self, *, opens_at, closes_at, legal, fallback, base_cpu,
                        compute_fn=None, forced_on_total_expiry=None):
        ideal = self.clock_contract=='ideal'
        base_end = opens_at if ideal else opens_at+base_cpu
        background_cpu = 0.0
        candidate = fallback
        if compute_fn is not None:
            tick = process_time();candidate = compute_fn();background_cpu = process_time()-tick
        end = opens_at if ideal else base_end+background_cpu
        base_valid = ideal or base_end < closes_at
        timely = ideal or end < closes_at
        accepted = compute_fn is not None and timely
        # Serial onboard contract: the base action is a proposal, not a live
        # fallback. Submit once after all requested work completes. If doors
        # have closed, remaining onboard is the physical default.
        chosen = candidate if timely else forced_on_total_expiry
        consumed_at = end if timely else closes_at
        self._controller_available_at = end
        self._last_consumed_clock = consumed_at
        result = dict(kind='onboard_publication',clock_contract=self.clock_contract,
            opens_at=opens_at,closes_at=closes_at,window_seconds=closes_at-opens_at,
            base_cpu_seconds=base_cpu,background_cpu_seconds=background_cpu,
            total_cpu_seconds=base_cpu+background_cpu,fallback_expired=not base_valid,
            background_submitted=compute_fn is not None,background_accepted=accepted,
            background_rejection_reason=None if accepted or compute_fn is None else 'deadline_expired',
            consumed=dict(action=chosen,consumed_wall=consumed_at),
            controller_available_at=end,fallback_published=False,
            base_proposal_timely=base_valid,final_proposal_timely=timely,
            execution_contract='serial_onboard_proposal; no preissued alight fallback')
        self.publications.append(result)
        return chosen,result

    def choose_onboard(self,node,cid,clock,history):
        from experiments.trc_revision_v1.rollout import evaluate_pair_rollout, choose_paired_screen
        has_cont = has_legal_continue(node,cid)
        dwell = self._current_dwell_info
        state = self._record_state(node,history,kind='onboard',cid=cid,clock=float(clock),
            legal_actions=['alight','continue'],has_continuation=has_cont,
            raw_dwell_seconds=dwell['raw_dwell_seconds'],dwell_closes_at=dwell['closes_at'])
        tick = getattr(self,'_onboard_cpu_start',process_time())
        if self.csa is not None:
            alight,receipt = self.csa.onboard(node,cid,clock,history)
        else:
            alight,receipt = self.engine.onboard(node,cid,clock,history,method='rolling_fixed')
        base_cpu = process_time()-tick
        base = 'alight' if alight else 'continue'
        compute = None;screen = None
        if has_cont and self.controller in ('full128','bounded_cert','paired_screen'):
            key = 'full128' if self.controller=='full128' else 'bounded_cert'
            hp = self.hyperparams[key]
            invoke = self.controller=='full128'
            if not invoke:
                invoke,screen = should_invoke_rollout(receipt,deadline=self.deadline,
                    ambiguity_margin_seconds=hp['ambiguity_margin_seconds'])
            remaining = float(hp['max_process_cpu_seconds'])-base_cpu
            if invoke and remaining>0:
                def compute():
                    gate = evaluate_pair_rollout(self.graph,self.planner.model,self.planner.parameters,
                        self.deadline,node,dict(kind='onboard',cid=cid,clock=clock),history,
                        self.observation_contract,root_index=self.root_index,
                        count=hp['worlds_per_query'],max_process_cpu=remaining,seed=hp['seed'],
                        rollout_planner_kind='matched' if self.controller=='full128' else 'recursive',
                        exact_combined_max_width=hp.get('exact_combined_max_width',6))
                    if self.controller=='full128':
                        selected=choose_point_estimate(gate,base);normal=None
                    else:
                        selected,normal=choose_paired_screen(gate,base,z_alpha=hp['z_alpha'],
                                                            min_effect_seconds=hp['min_effect_seconds'])
                    gate.update(cid=cid,clock=clock,state_index=len(self.states)-1,
                        base_action=base,base_receipt=receipt,proposed_action=selected,
                        normal_screen=normal,gate_screen=screen)
                    self.gates.append(gate);state['gate_index']=len(self.gates)-1
                    return selected
            elif invoke:
                state['rollout_skipped']='base_exhausted_query_allowance'
        selected,pub=self._publish_action(opens_at=clock,closes_at=dwell['closes_at'],
            legal=('alight','continue'),fallback=base,base_cpu=base_cpu,compute_fn=compute,
            forced_on_total_expiry='continue' if has_cont else 'alight')
        state.update(selected_action=selected,base_action=base)
        if 'gate_index' in state:
            self.gates[state['gate_index']].update(selected_action=selected,
                publication_accepted=pub['background_accepted'])
        self.decision_costs.append(dict(kind='onboard',clock=float(clock),
            process_cpu=base_cpu+pub['background_cpu_seconds']))
        return selected=='alight',dict(arm=self.arm_config['label'],base=base,
                                       selected=selected,scoring=receipt)

    def ride(self,ride,ready,history,world):
        previous,vehicle=ready,None
        for idx,cid in enumerate(ride.path):
            departure,canceled,identity=world.event(cid,'departure')
            if canceled or departure<previous or departure>=self.deadline:
                return self.deadline,'service_or_clock_failure'
            if identity is None or vehicle is not None and identity!=vehicle:
                raise MissingEndpoint(f'{cid}:inconsistent_vehicle_identity')
            vehicle=identity
            arrival,canceled,arrival_identity=world.event(cid,'arrival')
            if canceled:return self.deadline,'service_failure'
            if arrival_identity!=vehicle or arrival<departure:
                raise MissingEndpoint(f'{cid}:inconsistent_arrival_identity_or_clock')
            if arrival>=self.deadline:return self.deadline,'deadline'
            history.observe(cid,'departure',departure);history.observe(cid,'arrival',arrival)
            previous=arrival
            if cid not in ride.exits:continue
            platform=ride.exits[cid]
            if platform is None:
                self.record(dict(kind='destination',cid=cid,clock=arrival));return arrival,'arrived'
            has_cont=idx+1<len(ride.path)
            if not has_cont:
                # This menu already commits to alighting at its final exit;
                # no fictitious terminal door interval or new query is needed.
                evidence=self.decision_history(history,arrival)
                self._record_state(ride,history,kind='onboard',cid=cid,clock=float(arrival),
                    legal_actions=['alight'],has_continuation=False,selected_action='alight',
                    base_action='alight',decision_type='mandatory_menu_exit')
                self.record(dict(kind='onboard',cid=cid,clock=arrival,arrival_clock=arrival,
                    consumed_at=arrival,choice='alight',receipt=dict(method='mandatory_menu_exit'),
                    **evidence))
                return self.platform(platform,arrival,history,world)
            next_dep,next_canceled,_=world.event(ride.path[idx+1],'departure')
            if next_canceled:
                raise MissingEndpoint(
                    f'{ride.path[idx+1]}:canceled_departure_has_no_door_closing_endpoint')
            closes_at=float(next_dep)
            start=float(arrival)
            available=getattr(self,'_controller_available_at',start)
            if available>start or start>=closes_at:
                self.record(dict(kind='controller_busy',cid=cid,clock=arrival,
                                 available_at=available,departure_deadline=closes_at,choice='continue'))
                continue
            self._current_dwell_info=dict(raw_dwell_seconds=None if next_dep is None else next_dep-arrival,
                                         opens_at=start,closes_at=closes_at)
            self._onboard_cpu_start=process_time()
            evidence=self.decision_history(history,start)
            alight,receipt=self.choose_onboard(ride,cid,start,history)
            self.counts['onboard']+=1
            self.record(dict(kind='onboard',cid=cid,clock=start,arrival_clock=arrival,
                consumed_at=self._last_consumed_clock,choice='alight' if alight else 'continue',
                receipt=receipt,**evidence))
            if alight:
                return self.platform(platform,self._last_consumed_clock,history,world)
        return self.deadline,'exhausted_ride'
