"""Reachable Oct28 extra-run identity ambiguity must remain visible."""
from copy import deepcopy
import csv
import gzip
import json
from pathlib import Path

from experiments.reconstruction_v6.calibration_source_completion import supplemental_exact_rows
from experiments.reconstruction_data_audit.build_dated_replay import attach_outcomes


def oct28_sample():
    root = Path(__file__).resolve().parents[1] / 'experiments/reconstruction_v6'
    with gzip.open(root / 'calibration_selection_full_source_actuals.json.gz', 'rt') as handle:
        connections = json.load(handle)['dates']['2023-10-28']['connections']
    connection = next(row for row in connections if row['id'] == 8)
    path = root / 'calibration_confirmation_sources/rail_v6_confirmation_chunked_2023-10-28.csv.gz'
    with gzip.open(path, 'rt') as handle:
        rows = [row for row in csv.DictReader(handle, delimiter=';')
                if row['LINIEN_ID'] == connection['vehicle_number'] and row['BPUIC']
                in (connection['dep_stop'], connection['arr_stop'])]
    return deepcopy(connection), rows


def test_actual_oct28_extra_run_can_be_scored_without_changing_the_strict_join():
    connection, rows = oct28_sample()
    strict = attach_outcomes('2023-10-28', [connection], rows)
    assert strict['connection_outcome_counts'] == {'UNRESOLVED': 1}
    augmented, used = supplemental_exact_rows('2023-10-28', [connection], rows)
    sensitivity = attach_outcomes('2023-10-28', [connection], augmented)
    assert sensitivity['connection_outcome_counts'] == {'MEASURED': 1}
    assert connection['departure']['official_journey'] == '85:11:20226:004'
    assert used and all(row['ZUSATZFAHRT_TF'] == 'true' for row in rows)


def test_regular_records_take_priority_and_competing_extra_runs_stay_ambiguous():
    connection, extras = oct28_sample()
    regular = [dict(row, ZUSATZFAHRT_TF='false', FAHRT_BEZEICHNER='regular') for row in extras]
    combined, used = supplemental_exact_rows('2023-10-28', [connection], regular + extras)
    assert not used
    attach_outcomes('2023-10-28', [connection], combined)
    assert connection['outcome'] == 'MEASURED'
    assert connection['departure']['official_journey'] == 'regular'
    competing = [dict(row, FAHRT_BEZEICHNER='different_extra') for row in extras]
    combined, _ = supplemental_exact_rows('2023-10-28', [connection], extras + competing)
    attach_outcomes('2023-10-28', [connection], combined)
    assert connection['departure']['status'] == 'AMBIGUOUS_ACTUAL'
    assert connection['outcome'] == 'UNRESOLVED'
