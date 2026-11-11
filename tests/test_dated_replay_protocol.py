"""Replay checkpoints must stay attached to their actual inputs."""
from copy import deepcopy
import gzip
import json

import pytest

from experiments.reconstruction_v4.run_dated_replay import freeze_replay_inputs


@pytest.mark.parametrize('field', ['data', 'forecasts', 'training', 'settings'])
def test_changed_replay_input_rejects_resume(tmp_path, field):
    output = tmp_path / 'result.json'
    inputs = dict(data={'event': 120}, forecasts={'receipt': 90}, training={'mean': 1.},
                  settings={'budget': 64}, sources=['input.json'])
    freeze_replay_inputs(output, **inputs)
    freeze_replay_inputs(output, **inputs)
    changed = deepcopy(inputs)
    changed[field]['changed'] = True
    with pytest.raises(ValueError, match='Inputs or protocol changed'):
        freeze_replay_inputs(output, **changed)


def test_unfrozen_replay_receipts_require_a_new_output(tmp_path):
    output = tmp_path / 'result.json'
    cases = tmp_path / 'result_cases'
    cases.mkdir()
    (cases / '2023-10-25_q00_trained_nominal.json').write_text('{}')
    with pytest.raises(ValueError, match='lack a frozen protocol'):
        freeze_replay_inputs(output, data={}, forecasts={}, training={}, settings={}, sources=[])


def test_source_text_and_library_versions_are_frozen(tmp_path):
    source = tmp_path / 'policy.py'
    source.write_text('value = 1\n')
    output = tmp_path / 'result.json'
    inputs = dict(data={}, forecasts={}, training={}, settings={}, sources=[str(source)])
    path = freeze_replay_inputs(output, **inputs)
    with gzip.open(path, 'rt') as handle:
        frozen = json.load(handle)
    assert frozen['source_text'][str(source)] == 'value = 1\n'
    assert set(frozen['libraries']) == {'numpy', 'scipy'}
    source.write_text('value = 2\n')
    with pytest.raises(ValueError, match='Inputs or protocol changed'):
        freeze_replay_inputs(output, **inputs)
