"""Read saved V51 traces only; no new policy replay or fitted-model estimates.

Run with BAPR_HRO_REPO pointing at the pinned a2b82be9 checkout.  Graph creation
merely maps saved connection IDs to names and the archived physical endpoint.
"""
import gzip
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ['BAPR_HRO_REPO']).resolve()
sys.path.insert(0, str(REPO))
from experiments.reconstruction_v34.discover import context, read

HERE = Path(__file__).resolve().parent
ROOTS = (4, 6, 18, 32, 62)


def main():
    data = read(REPO/'experiments/reconstruction_v34/data.json.gz')
    graphs, rows = {}, []
    for root in ROOTS:
        arms = {}
        for label in ('cheap', 'full128'):
            relative = f'experiments/reconstruction_v51/roots/{root:03}_{label}.json.gz'
            j = read(REPO/relative)
            if j['day'] not in graphs:
                graphs[j['day']] = context(data, j['day'])
            graph, _, _, _, metadata, _ = graphs[j['day']]
            events = []
            for ti, t in enumerate(j['outcome']['trace']):
                x = {k: t[k] for k in ('kind', 'cid', 'stop', 'entry', 'ready',
                      'clock', 'choice', 'visible', 'observed_visible') if k in t}
                x['trace_index'] = ti
                if 'cid' in t:
                    c = graph.connections[t['cid']]
                    x.update(arrival_station=graph.stops[c.arr_stop].name,
                             route=c.route, trip=c.trip_id)
                if 'stop' in t:
                    x['station'] = graph.stops[t['stop']].name
                    x['mct'] = graph.stops[t['stop']].min_transfer_time
                    x['visible_services'] = [dict(cid=cid,
                        route=graph.connections[cid].route,
                        trip=graph.connections[cid].trip_id,
                        to_station=graph.stops[graph.connections[cid].arr_stop].name)
                        for cid in t.get('visible', [])]
                    if t['kind'] == 'observation' and t['clock'] < t['ready']:
                        x['seconds_before_ready'] = t['ready'] - t['clock']
                receipt = t.get('message_receipt', {})
                rs = receipt.get('receipts', [])
                x['remote'] = dict(frames_consumed=receipt.get('frames_consumed', 0),
                    first_frame=receipt.get('first_frame_index'),
                    next_frame=receipt.get('next_frame_index'),
                    publication_times=sorted({r['receipt_seconds'] for r in rs}),
                    changed_endpoint_occurrences=sum(len(r.get('changed_events', [])) for r in rs),
                    cleared_endpoint_occurrences=sum(len(r.get('cleared_events', [])) for r in rs))
                # Save only forecasts/observations on the physical trip(s) used by
                # either actual arm. No summary calls these predictive errors.
                events.append(x)
            arms[label] = dict(source=relative, loss=j['outcome']['loss'],
                status=j['outcome']['status'], events=events,
                gates=[{k:g[k] for k in ('state_index','cid','clock','base_action','selected_action')}
                       for g in j['gates']],
                root_scoring=j['outcome']['trace'][0].get('receipt', {}).get('scoring', {}),
                first_public_history=j['states'][0]['public_history'])
        cheap, full = arms['cheap'], arms['full128']
        assert cheap['events'][0]['clock'] == full['events'][0]['clock']
        assert cheap['first_public_history'] == full['first_public_history']
        assert all(g['base_action'] == g['selected_action'] for g in full['gates'][1:])
        actual_trip_ids = {e['trip'] for a in arms.values() for e in a['events'] if 'trip' in e}
        actual_trip_ids.update(v['trip'] for a in arms.values() for e in a['events']
                               for v in e.get('visible_services', []) if e.get('choice') == v['cid'])
        for a in arms.values():
            a['first_public_history_on_actual_trips'] = [r for r in a.pop('first_public_history')
                if r['trip'] in actual_trip_ids and r['evidence'] in ('observed','forecast')]
        common_through = []
        # A through-vehicle seen at the later hub before ready is observable in
        # the stored full trace, and its actual traversal is in cheap's history.
        cheap_trips = {e['trip'] for e in cheap['events'] if 'trip' in e}
        for e in full['events']:
            if 'seconds_before_ready' not in e:
                continue
            for service in e.get('visible_services', []):
                if service['trip'] in cheap_trips:
                    common_through.append(dict(station=e['station'], cid=service['cid'],
                        trip=service['trip'], departure=e['clock'], ready=e['ready'],
                        seconds_before_ready=e['seconds_before_ready'], mct=e['mct']))
        rows.append(dict(root_index=root, day=j['day'], origin=j['origin'], goal=j['goal'],
            actual_full_gain=cheap['loss']-full['loss'],
            first_history_identical=True, later_rollout_overrides=0,
            through_services_missed_at_later_transfer=common_through, arms=arms))
    out = dict(commit='a2b82be9c3fc6b61e2a5826d7cd8691d39ff4491',
        scope='Saved actual V51 traces; descriptive mechanisms, not causal message ablations, population model bias, or new replays.', rows=rows)
    (HERE/'actual_mechanisms.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    for r in rows:
        print(json.dumps({k:r[k] for k in ('root_index','actual_full_gain',
            'later_rollout_overrides','through_services_missed_at_later_transfer')},ensure_ascii=False))


if __name__ == '__main__':
    main()
