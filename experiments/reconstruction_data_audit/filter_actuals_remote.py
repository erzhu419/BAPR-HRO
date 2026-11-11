"""Run on the research server and retain only requested official actual rows."""
import argparse
from collections import Counter
import csv
import gzip
from io import BytesIO, StringIO
import json
from pathlib import Path
import struct
from urllib.request import Request, urlopen
from zipfile import ZipFile
import zlib

ARCHIVE = 'https://archive.opentransportdata.swiss/istdaten/2023/ist-daten-2023-10.zip'


def fetch_range(url, start, length):
    header = f'bytes={start}-{start + length - 1}' if start >= 0 else f'bytes=-{length}'
    with urlopen(Request(url, headers={'Range': header}), timeout=90) as response:
        if response.status != 206:
            raise ValueError('Official archive ignored the selected byte range')
        return response.read(length), int(response.headers['Content-Range'].rsplit('/', 1)[1])


def in_window(row, start, end):
    times = [row[key] for key in ('ANKUNFTSZEIT', 'ABFAHRTSZEIT') if row[key]]
    return any(start <= int(value[-5:-3]) * 3600 + int(value[-2:]) * 60 <= end for value in times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', default='2023-10-25')
    parser.add_argument('--filter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix-bytes', type=int, help='Bounded prefix for the explicitly partial SBB calibration cohort')
    args = parser.parse_args()
    selection = json.loads(args.filter.read_text())
    stops = set(selection['stops'])
    start, end = selection['time_window_seconds']
    tail, size = fetch_range(ARCHIVE, -1, 65557)
    with ZipFile(BytesIO(tail)) as archive:
        member = archive.getinfo(f'ist-daten-2023-10/{args.date}_istdaten.csv')
        offset = member.header_offset + size - len(tail)
        header, _ = fetch_range(ARCHIVE, offset, 30)
        fields = struct.unpack('<4s5H3L2H', header)
        data_offset = offset + 30 + sum(fields[-2:])
        length = min(member.compress_size, args.prefix_bytes) if args.prefix_bytes else member.compress_size
        compressed, _ = fetch_range(ARCHIVE, data_offset, length)
    raw = zlib.decompressobj(-15).decompress(compressed)
    content = raw[:raw.rfind(b'\n') + 1].decode('utf-8-sig')
    reader = csv.DictReader(StringIO(content), delimiter=';')
    counts = Counter()
    statuses = Counter()
    scanned = selected = 0
    with gzip.open(args.output, 'wt', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, reader.fieldnames, delimiter=';')
        writer.writeheader()
        for row in reader:
            scanned += 1
            if row['BPUIC'] not in stops or not in_window(row, start, end):
                continue
            writer.writerow(row)
            selected += 1
            counts[row['BETREIBER_ID'], row['BETREIBER_ABK']] += 1
            statuses[row['AN_PROGNOSE_STATUS'], row['AB_PROGNOSE_STATUS'], row['FAELLT_AUS_TF']] += 1
    audit = {'archive': ARCHIVE, 'date': args.date, 'member': member.filename,
             'member_data_byte_range': [data_offset, data_offset + len(compressed) - 1],
             'complete_daily_member': len(compressed) == member.compress_size,
             'remote_bytes_requested': len(tail) + len(header) + len(compressed),
             'scanned_daily_rows': scanned, 'selected_rows': selected, 'local_subset_bytes': args.output.stat().st_size,
             'operator_counts': [{'operator_id': key[0], 'operator_name': key[1], 'rows': n}
                                 for key, n in counts.most_common()],
             'status_counts': [{'arrival': key[0], 'departure': key[1], 'canceled': key[2], 'rows': n}
                               for key, n in statuses.most_common()]}
    args.output.with_suffix('.audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
