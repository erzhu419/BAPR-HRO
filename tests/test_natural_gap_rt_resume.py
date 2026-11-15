"""Resume retains the first public mapping and does not decode completed days."""
import sys

from experiments.natural_gap_v2 import rt_decode


def test_resume_reuses_complete_day_and_preserves_mapping_time(monkeypatch,tmp_path):
    day = '2023-03-23'
    receipt = dict(dates=[day],date_freeze_utc='date-freeze',menu_freeze_utc='menu-freeze',
                   written_at_utc='first-mapping-before-first-decode')
    summary = dict(source_parse_complete=True,identity_mapping_complete=True,sampled_snapshots=17)
    value = dict(day=day,frames=[],summary=summary)
    files = {
        'date_manifest.json':dict(dates_frozen=True,selected_dates=[day],frozen_at_utc='date-freeze'),
        'selection.json':dict(frozen_at_utc='menu-freeze'),
        'scheduled.json.gz':dict(dates={day:dict(connections=[])}),
        'rt_mapping_receipt.json':receipt,
        'rt_public_identity_mapping.json.gz':dict(days={day:dict(rows=[])}),
        'rt_decode_attempts.json':dict(attempts=[dict(source_processing_started_at_utc='first-attempt',workers=2)]),
        f'messages/{day}.json.gz':value,
        f'messages/{day}.audit.json':dict(date=day,output_complete=True,summary=summary),
    }
    for name,record in files.items():
        rt_decode.write(tmp_path/name,record)

    def prohibited(*args,**kwargs):
        raise AssertionError('Completed source must not be decoded or remapped again')

    monkeypatch.setattr(rt_decode,'decode_day',prohibited)
    monkeypatch.setattr(rt_decode,'audit_day',prohibited)
    monkeypatch.setattr(sys,'argv',['rt_decode','--directory',str(tmp_path),'--workers','6'])
    rt_decode.main()
    assert rt_decode.read(tmp_path/'rt_mapping_receipt.json') == receipt
    assert rt_decode.read(tmp_path/'messages.json.gz')['dates'][day] == value
    output = rt_decode.read(tmp_path/'rt_decode_summary.json')
    assert output['source_processing_started_at_utc'] == 'first-attempt'
    assert output['resumed_complete_dates'] == [day] and output['newly_decoded_dates'] == []
