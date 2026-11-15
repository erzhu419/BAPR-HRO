"""Filter one complete official actual day on the server, keeping all statuses."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
from io import StringIO
import json
from pathlib import Path
import struct
from urllib.request import Request, urlopen
import zlib


def fetch_range(url, start, length):
    with urlopen(Request(url, headers={'Range': f'bytes={start}-{start+length-1}'}),
                 timeout=90) as response:
        if response.status != 206:
            raise ValueError('Official archive ignored selected byte range')
        if not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{start+length-1}/'):
            raise ValueError('Official archive returned a different byte range')
        raw = response.read(length)
        if len(raw) != length:
            raise ValueError('Incomplete archive member')
        return raw


def fetch_member(url, start, length):
    """Read eight disjoint ranges, preserving their original compressed order."""
    boundaries = [length*i//8 for i in range(9)]
    ranges = [(start+left, right-left) for left,right in zip(boundaries, boundaries[1:])]
    with ThreadPoolExecutor(max_workers=8) as pool:
        pieces = list(pool.map(lambda item:fetch_range(url, *item), ranges))
    return b''.join(pieces)


def in_window(row, start, end):
    times = [row[key] for key in ('ANKUNFTSZEIT', 'ABFAHRTSZEIT') if row[key]]
    return any(start <= int(value[-5:-3])*3600+int(value[-2:])*60 <= end for value in times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--filter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    selection = json.loads(args.filter.read_text())
    members = [m for m in metadata['members']
               if Path(m['name']).name == f'{args.date}_istdaten.csv']
    if len(members) != 1:
        raise ValueError(f'Expected one daily CSV member, found {len(members)}')
    member = members[0]
    offset = member['absolute_local_header_offset']
    header = fetch_range(metadata['url'], offset, 30)
    fields = struct.unpack('<4s5H3L2H', header)
    data_offset = offset+30+sum(fields[-2:])
    compressed = fetch_member(metadata['url'], data_offset, member['compressed_bytes'])
    raw = zlib.decompress(compressed, -15)
    if len(raw) != member['uncompressed_bytes']:
        raise ValueError('Incomplete decompressed daily CSV')
    reader = csv.DictReader(StringIO(raw.decode('utf-8-sig')), delimiter=';')
    stops = set(selection['stops'])
    start, end = selection['time_window_seconds']
    counts, statuses = Counter(), Counter()
    scanned = selected = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
    audit = dict(archive=metadata['url'], date=args.date, member=member['name'],
                 member_data_byte_range=[data_offset, data_offset+len(compressed)-1],
                 complete_daily_member=True, remote_bytes_requested=30+len(compressed),
                 member_range_workers=8,
                 scanned_daily_rows=scanned, selected_rows=selected,
                 local_subset_bytes=args.output.stat().st_size,
                 fields=reader.fieldnames,
                 operator_counts=[dict(operator_id=k[0], operator_name=k[1], rows=n)
                                  for k,n in counts.most_common()],
                 status_counts=[dict(arrival=k[0], departure=k[1], canceled=k[2], rows=n)
                                for k,n in statuses.most_common()])
    args.output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('fields','operator_counts','status_counts')}), flush=True)


if __name__ == '__main__':
    main()
