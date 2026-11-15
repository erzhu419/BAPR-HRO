"""Source completion gates and daily-only execution, using synthetic results."""
from concurrent.futures import ThreadPoolExecutor

from experiments.decision_calibration_v2 import evaluate_ready
from experiments.decision_gap_v1.panel import read, write


def test_payload_files_do_not_signal_completion_before_both_receipts(tmp_path):
    day = '2023-03-09'
    for folder, suffix in (('actual', 'csv'), ('messages', 'json')):
        (tmp_path/folder).mkdir()
        (tmp_path/folder/f'{day}.{suffix}.gz').touch()
    assert not evaluate_ready.inputs_ready(tmp_path, day)
    (tmp_path/'actual'/f'{day}.csv.audit.json').touch()
    assert not evaluate_ready.inputs_ready(tmp_path, day)
    (tmp_path/'messages'/f'{day}.json.audit.json').touch()
    assert evaluate_ready.inputs_ready(tmp_path, day)


def test_complete_results_are_reused_errors_retained_and_no_early_bootstrap(tmp_path, monkeypatch):
    dates = [f'2023-03-{day:02}' for day in range(9, 23)]
    write(tmp_path/'date_manifest.json', dict(dates_frozen=True, frozen_at_utc='date_freeze',
                                           selected_dates=dates))
    write(tmp_path/'selection.json', dict(frozen_at_utc='menu_freeze'))
    for day in dates:
        for folder, suffix in (('actual', 'csv'), ('messages', 'json')):
            write(tmp_path/folder/f'{day}.{suffix}.audit.json', {})

    def result(day):
        return dict(day=day, roots=[dict(initial_state=dict(status='observed'))],
            events=[dict(model=dict(status='resolved'), actual=dict(status='label_unresolved'))],
            cost=dict(process_cpu=.1, wall=.2))

    write(tmp_path/'results'/f'{dates[0]}.json.gz', result(dates[0]))
    calls = []

    def evaluate(directory, day):
        calls.append(day)
        if day == dates[3]:
            raise ValueError('synthetic source inconsistency')
        write(directory/'results'/f'{day}.json.gz', result(day))

    monkeypatch.setattr(evaluate_ready, 'evaluate_day', evaluate)
    monkeypatch.setattr(evaluate_ready, 'ProcessPoolExecutor', ThreadPoolExecutor)
    audit = evaluate_ready.run(tmp_path)
    assert set(calls) == set(dates[1:]) and len(calls) == 13
    assert audit['days'][dates[0]]['reused_complete_result']
    assert len(audit['days']) == 13 and set(audit['errors']) == {dates[3]}
    assert audit['status'] == 'failed' and audit['max_workers'] == 4
    assert not audit['bootstrap_summary_run'] and not (tmp_path/'summary.json').exists()
    saved = read(tmp_path/'evaluation_run.json')
    assert saved['waiting_for_sources'] == saved['evaluating_dates'] == []
