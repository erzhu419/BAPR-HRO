import json

import pytest

from experiments.validation_identity_v1 import static_acquire


def test_download_records_each_fully_written_range(tmp_path, monkeypatch):
    body=b'abcdefghijkl'
    monkeypatch.setattr(static_acquire,'BLOCK_BYTES',4)
    monkeypatch.setattr(static_acquire,'WORKERS',3)
    monkeypatch.setattr(static_acquire,'fetch_range',lambda url,start,length,size:body[start:start+length])
    path=tmp_path/'official.zip'
    result=static_acquire.download_archive('official-source',len(body),path)
    rows=[json.loads(line) for line in path.with_suffix('.blocks.jsonl').read_text().splitlines()]
    assert path.read_bytes()==body
    assert {(row['start'],row['bytes']) for row in rows}=={(0,4),(4,4),(8,4)}
    assert all(row['status']=='complete' for row in rows)
    assert result['downloaded_bytes']==12


def test_failed_range_resume_keeps_successes_without_refetching(tmp_path, monkeypatch):
    body=b'abcdefghijkl'
    calls=[]
    monkeypatch.setattr(static_acquire,'BLOCK_BYTES',4)
    monkeypatch.setattr(static_acquire,'WORKERS',3)

    def fetch(url,start,length,size):
        calls.append(start)
        if start==4 and calls.count(4)==1:
            raise TimeoutError('observed network failure case')
        return body[start:start+length]

    monkeypatch.setattr(static_acquire,'fetch_range',fetch)
    path=tmp_path/'official.zip'
    with pytest.raises(RuntimeError,match='1 archive ranges failed'):
        static_acquire.download_archive('official-source',len(body),path)
    result=static_acquire.download_archive('official-source',len(body),path)
    assert path.read_bytes()==body
    assert calls.count(0)==calls.count(8)==1
    assert calls.count(4)==2
    assert result['reused_recorded_bytes']==8
    assert result['downloaded_bytes']==4
    rows=[json.loads(line) for line in path.with_suffix('.blocks.jsonl').read_text().splitlines()]
    assert sum(row['status']=='failed' for row in rows)==1
    assert sum(row['status']=='complete' for row in rows)==3
