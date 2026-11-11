"""Directed GTFS transfer declarations, distinct from physical lower bounds."""
from dataclasses import dataclass
import csv
from pathlib import Path


@dataclass(frozen=True)
class TransferDeclaration:
    seconds: int | None
    allowed: bool | None
    source_pair: tuple[str, str] | None
    specificity: str
    kind: str
    physical_lower_seconds: int = 0


class PlatformTransfers:
    def __init__(self, stop_rows, transfer_rows):
        self.parents = {r['stop_id']: r.get('parent_station', '') for r in stop_rows}
        self.rules = {}
        for row in transfer_rows:
            key = row['from_stop_id'], row['to_stop_id']
            self.rules.setdefault(key, set()).add((row.get('transfer_type') or '0',
                int(row['min_transfer_time']) if row.get('min_transfer_time') else None))

    @classmethod
    def from_gtfs(cls, directory):
        directory = Path(directory)
        with (directory / 'stops.txt').open(encoding='utf-8-sig') as handle:
            stops = list(csv.DictReader(handle))
        with (directory / 'transfers.txt').open(encoding='utf-8-sig') as handle:
            return cls(stops, list(csv.DictReader(handle)))

    def resolve(self, arrival_platform, departure_platform):
        if arrival_platform is None or departure_platform is None:
            return TransferDeclaration(None, None, None, 'unknown_platform', 'unidentified')
        a, b = arrival_platform, departure_platform
        candidates = [(a, b, 'platform_pair')]
        pa, pb = self.parents.get(a), self.parents.get(b)
        if pb:
            candidates.append((a, pb, 'platform_station'))
        if pa:
            candidates.append((pa, b, 'station_platform'))
        if pa and pb:
            candidates.append((pa, pb, 'station_pair'))
        for left, right, specificity in candidates:
            rows = self.rules.get((left, right))
            if not rows:
                continue
            if len(rows) != 1:
                return TransferDeclaration(None, None, (left, right), specificity, 'conflicting_declarations')
            kind, seconds = next(iter(rows))
            allowed = False if kind == '3' else True
            return TransferDeclaration(seconds, allowed, (left, right), specificity,
                                       'operator_buffer_including_schedule_variance')
        return TransferDeclaration(None, None, None, 'no_declaration', 'unidentified')

    def __call__(self, arrival_platform, departure_platform):
        """Operational duration; an unidentified pair never receives120 or max."""
        rule = self.resolve(arrival_platform, departure_platform)
        return rule.seconds if rule.allowed else None
