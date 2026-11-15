"""Civil GTFS service clocks stay aligned on the real March26 DST date."""
from datetime import datetime
from io import BytesIO
import tarfile
from zoneinfo import ZoneInfo

from google.transit import gtfs_realtime_pb2 as pb
import pytest

from experiments.natural_gap_v2.rt_acquire import capture_metadata
from experiments.natural_gap_v2.rt_clock import civil_service_seconds
from experiments.reconstruction_v34.messages import snapshot_frames


def epoch(day, clock):
    return int(datetime.fromisoformat(day+'T'+clock).replace(tzinfo=ZoneInfo('Europe/Zurich')).timestamp())


@pytest.mark.parametrize('day', ['2023-03-23', '2023-03-26'])
def test_capture_clock_matches_gtfs_local_civil_seconds(day):
    assert civil_service_seconds(epoch(day,'07:00:03'),day) == 25203
    assert civil_service_seconds(epoch(day,'09:40:02'),day) == 34802


def feed(day):
    message = pb.FeedMessage()
    message.header.gtfs_realtime_version = '2.0'
    message.header.timestamp = epoch(day,'07:00:02')
    entity = message.entity.add()
    entity.id = 'e'
    update = entity.trip_update
    update.trip.trip_id = 't'
    update.trip.start_date = day.replace('-','')
    update.timestamp = epoch(day,'07:00:09')
    stop = update.stop_time_update.add()
    stop.stop_sequence = 1
    stop.stop_id = 'A'
    stop.departure.time = epoch(day,'08:00:00')
    stop.arrival.delay = 60
    return message


def decode(day, conversion):
    return snapshot_frames(feed(day),day,epoch(day,'00:00:00'),25203,'capture.gtfsrt',
        {'t':{(1,'departure'):(0,'A',28800),(1,'arrival'):(0,'A',28920)}},
        epoch_to_service=conversion)


def test_dst_time_delay_and_entity_receipt_share_one_civil_clock():
    day = '2023-03-26'
    frame, = decode(day,lambda value:civil_service_seconds(value,day))
    assert (frame['capture_seconds'],frame['header_seconds'],frame['update_seconds'],
            frame['receipt_seconds']) == (25203,25202,25209,25209)
    assert frame['endpoint_updates'] == [[0,'departure','FORECAST',28800],[0,'arrival','FORECAST',28980]]


def test_normal_date_preserves_all_legacy_frame_fields():
    day = '2023-03-23'
    assert decode(day,lambda value:civil_service_seconds(value,day)) == decode(day,None)


def test_metadata_prescreen_uses_civil_bins_without_reading_protobuf(tmp_path):
    day = '2023-03-26'
    archive = tmp_path/(day+'.tar.bz2')
    with tarfile.open(archive,'w:bz2') as output:
        for clock in range(25200,34801,600):
            name = f'{day}/{day}T{clock//3600:02}:{clock%3600//60:02}:03+02:00.gtfsrt'
            member = tarfile.TarInfo(name)
            payload = b'not a protobuf'
            member.size = len(payload)
            output.addfile(member,BytesIO(payload))
    result = capture_metadata({'date':day},archive,{'required_bin_starts':list(range(25200,34801,600))},
                              tmp_path/'report.json')
    assert result['capture_coverage_complete'] and result['sampled_capture_count'] == 17
    assert result['filename_civil_minus_elapsed_seconds'] == [3600]
    assert result['capture_range'] == [25203,34803]
    assert result['protobuf_values_opened'] is False
