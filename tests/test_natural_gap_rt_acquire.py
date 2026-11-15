"""A short read must retry before untouched archive work fills its worker slot."""
import json
import sys

from experiments.natural_gap_v2 import rt_acquire


def test_failed_partial_range_resumes_before_next_date(monkeypatch,tmp_path):
    days = [dict(date=day,filename=day+'.tar.bz2',size_bytes=3,url='unused')
            for day in ('2023-03-23','2023-03-24')]
    rt_acquire.write(tmp_path/'rt_source_metadata.json',dict(candidate_days=days))
    rt_acquire.write(tmp_path/'rt_metadata_rule.json',dict(frozen_at_utc='before any bytes'))
    calls = []

    def download(source,start,end,path):
        calls.append(source['date'])
        if len(calls) == 1:
            path.write_bytes(b'ab')
            raise ValueError('short read')
        if source['date'] == days[0]['date']:
            assert path.read_bytes() == b'ab'
            with path.open('ab') as handle:
                handle.write(b'c')
        else:
            path.write_bytes(b'xyz')
        return 1

    def clocks(source,archive,rule,report):
        return dict(date=source['date'],sampled_capture_count=17,missing_bins=[])

    monkeypatch.setattr(rt_acquire,'download_range',download)
    monkeypatch.setattr(rt_acquire,'capture_metadata',clocks)
    monkeypatch.setattr(sys,'argv',['rt_acquire','--directory',str(tmp_path),'--connections','1'])
    rt_acquire.main()
    assert calls == [days[0]['date'],days[0]['date'],days[1]['date']]
    assert (tmp_path/'cache/rt'/days[0]['filename']).read_bytes() == b'abc'
    audit = json.loads((tmp_path/'rt_acquisition.json').read_text())
    assert len(audit['errors']) == 1 and audit['unfinished_dates'] == []
