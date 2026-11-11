"""Replay two fixed, measured Zurich rail itineraries from the Ist-Daten subset."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import pickle
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_data_audit.run_realized_subset import latest_direct_forecasts


INCOMING, TRANSFER, STAY_ONBOARD = 975, 1, 976
START_SECONDS = 505 * 60


def clock_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def replay(row):
    service_date, measured, buffer = row
    legs = {cid: measured['legs'][str(cid)] for cid in (INCOMING, TRANSFER, STAY_ONBOARD)}
    events = {(cid, kind): event for cid, leg in legs.items() for kind, event in leg.items()}
    first = legs[INCOMING]
    hub_arrival = first['arrival']['actual_seconds']
    decisions = {'origin': START_SECONDS, 'hub': hub_arrival}
    windows = {(checkpoint, key): (time - 3600, time)
               for checkpoint, time in decisions.items() for key in events
               if checkpoint == 'origin' or key[0] != INCOMING}
    forecasts = latest_direct_forecasts(service_date, events, windows)

    alternatives = {}
    for name, cid in (('transfer_to_s20', TRANSFER), ('stay_on_s11', STAY_ONBOARD)):
        leg = legs[cid]
        same_vehicle = first['arrival']['official_journey'] == leg['departure']['official_journey']
        if same_vehicle != (cid == STAY_ONBOARD):
            raise ValueError('Measured journey identifiers disagree with the itinerary')
        ready = hub_arrival if same_vehicle else hub_arrival + buffer
        departure = leg['departure']['actual_seconds']
        arrival = leg['arrival']['actual_seconds']
        caught = ready <= departure
        alternatives[name] = {
            'connection_id': cid, 'official_journey': leg['departure']['official_journey'],
            'same_vehicle': same_vehicle, 'buffer_seconds': 0 if same_vehicle else buffer,
            'ready_to_depart': clock_time(ready), 'actual_departure': clock_time(departure),
            'actual_arrival': clock_time(arrival), 'departure_slack_seconds': departure - ready,
            'arrived': caught, 'elapsed_seconds': arrival - START_SECONDS if caught else None,
        }

    snapshots = {checkpoint: {'time': clock_time(time), 'direct_forecasts': []}
                 for checkpoint, time in decisions.items()}
    for (checkpoint, (cid, kind)), forecast in sorted(forecasts.items()):
        snapshots[checkpoint]['direct_forecasts'].append({
            'connection_id': cid, 'event_kind': kind, 'receipt': clock_time(forecast['receipt_seconds']),
            'predicted_event_time': clock_time(forecast['forecast_seconds']),
        })
    boarded = first['departure']['actual_seconds']
    if boarded < START_SECONDS:
        raise ValueError('Measured first leg departed before the passenger was ready')
    return service_date, {
        'public_history': snapshots,
        'shared_first_leg': {'connection_id': INCOMING, 'official_journey': first['departure']['official_journey'],
                             'ready': clock_time(START_SECONDS), 'actual_board': clock_time(boarded),
                             'actual_hub_arrival': clock_time(hub_arrival)},
        'fixed_itineraries': alternatives,
    }


def main():
    measured = json.loads((HERE / 'realized_subset.json').read_text())
    with (ROOT / 'data/zurich_wide.pkl').open('rb') as handle:
        graph = pickle.load(handle)
    incoming, transfer, stay = (graph.connections[cid] for cid in (INCOMING, TRANSFER, STAY_ONBOARD))
    if not incoming.arr_stop == transfer.dep_stop == stay.dep_stop or transfer.arr_stop != stay.arr_stop:
        raise ValueError('Itinerary does not share the same hub and destination')
    buffer = graph.stops[incoming.arr_stop].min_transfer_time * 60
    rows = [(date, data, buffer) for date, data in measured['dates'].items()]
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(replay, rows))
    output = {
        'scope': 'Retrospective replay of two fixed itineraries on two dated, measured SBB worlds. '
                 'Only the specified services are evaluated; no citywide policy comparison.',
        'actual_source': 'realized_subset.json', 'forecast_source': 'local GTFS-RT snapshot archive',
        'forecast_lookback_seconds': 3600,
        'stops': [graph.stops[sid].name for sid in (incoming.dep_stop, incoming.arr_stop, transfer.arr_stop)],
        'results': results,
    }
    path = HERE / 'verified_itinerary.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(results, ensure_ascii=False))


if __name__ == '__main__':
    main()
