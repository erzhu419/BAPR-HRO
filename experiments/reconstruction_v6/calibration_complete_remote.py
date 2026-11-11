"""Remote complete daily scan through bounded parallel archive byte ranges.

Copy beside the existing filter_actuals_remote.py on the research server.
Only its filtered gzip and audit are copied back to the workstation.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import codecs
import csv
import gzip
from io import BytesIO
import json
from pathlib import Path
import runpy
import struct
from zipfile import ZipFile
import zlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True)
    parser.add_argument('--filter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    old = runpy.run_path(str(Path(__file__).with_name('filter_actuals_remote.py')))
    fetch, archive_url = old['fetch_range'], old['ARCHIVE']
    tail, archive_size = fetch(archive_url, -1, 65557)
    with ZipFile(BytesIO(tail)) as archive:
        member = archive.getinfo(f'ist-daten-2023-10/{args.date}_istdaten.csv')
    offset = member.header_offset + archive_size - len(tail)
    header, _ = fetch(archive_url, offset, 30)
    fields = struct.unpack('<4s5H3L2H', header)
    start = offset + 30 + sum(fields[-2:])
    cache = args.output.parent / 'actual_member_parts' / args.date
    cache.mkdir(parents=True, exist_ok=True)
    step = 1024 * 1024
    chunks = [(i, min(step, member.compress_size - i)) for i in range(0, member.compress_size, step)]

    def part(spec):
        i, length = spec
        path = cache / f'{i}.deflate'
        if path.exists() and path.stat().st_size == length:
            return path, 0
        raw, _ = fetch(archive_url, start + i, length)
        if len(raw) != length:
            raise ValueError(f'Incomplete archive range: {i}/{length}')
        path.write_bytes(raw)
        print('RANGE', args.date, i, length, flush=True)
        return path, length

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pieces = list(pool.map(part, chunks))
    decoder = zlib.decompressobj(-15)
    text = codecs.getincrementaldecoder('utf-8-sig')()

    def lines():
        pending = ''
        for path, _ in pieces:
            pending += text.decode(decoder.decompress(path.read_bytes()))
            complete = pending.splitlines(keepends=True)
            pending = '' if not complete or complete[-1].endswith(('\n', '\r')) else complete.pop()
            yield from complete
        pending += text.decode(decoder.flush(), final=True)
        if pending:
            yield pending
        if not decoder.eof:
            raise ValueError('The complete requested member did not reach the deflate end')

    selection = json.loads(args.filter.read_text())
    stops, (lo, hi) = set(selection['stops']), selection['time_window_seconds']
    counts, statuses = Counter(), Counter()
    scanned = selected = 0
    reader = csv.DictReader(lines(), delimiter=';')
    with gzip.open(args.output, 'wt', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, reader.fieldnames, delimiter=';')
        writer.writeheader()
        for row in reader:
            scanned += 1
            if row['BPUIC'] not in stops or not old['in_window'](row, lo, hi):
                continue
            writer.writerow(row)
            selected += 1
            counts[row['BETREIBER_ID'], row['BETREIBER_ABK']] += 1
            statuses[row['AN_PROGNOSE_STATUS'], row['AB_PROGNOSE_STATUS'], row['FAELLT_AUS_TF']] += 1
    audit = {'archive': archive_url, 'date': args.date, 'member': member.filename,
        'member_data_byte_range': [start, start + member.compress_size - 1], 'complete_daily_member': True,
        'member_compressed_bytes': member.compress_size, 'member_uncompressed_bytes': member.file_size,
        'remote_bytes_requested': len(tail) + len(header) + sum(n for _, n in pieces),
        'cached_range_count': sum(n == 0 for _, n in pieces), 'scanned_daily_rows': scanned,
        'selected_rows': selected, 'local_subset_bytes': args.output.stat().st_size,
        'operator_counts': [{'operator_id': key[0], 'operator_name': key[1], 'rows': value} for key, value in counts.most_common()],
        'status_counts': [{'arrival': key[0], 'departure': key[1], 'canceled': key[2], 'rows': value}
                          for key, value in statuses.most_common()]}
    args.output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
