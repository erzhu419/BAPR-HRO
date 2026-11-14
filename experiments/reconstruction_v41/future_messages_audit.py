"""Describe saved message opportunities; never evaluate a counterfactual policy."""

from collections import Counter
import gzip
import json
from pathlib import Path
from statistics import mean, median


HERE = Path(__file__).resolve().parent


def census():
    pairs, counts, gaps = [], Counter(), []
    unique_frames, affected_roots, irreversible_roots = set(), set(), set()
    for index in range(72):
        path = HERE.parent/'reconstruction_v40/roots'/f'{index:03}_cheap_frontier__combined_rolling.json.gz'
        with gzip.open(path, 'rt') as handle:
            row = json.load(handle)
        events = [event for event in row['outcome']['trace'] if event['kind'] in ('onboard', 'platform')]
        counts['roots'] += 1
        counts['decisions'] += len(events)
        for left, right in zip(events, events[1:]):
            before, now = left['clock'], right['clock']
            receipts = right.get('message_receipt', {}).get('receipts', [])
            current = [r for r in receipts if before < r['receipt_seconds'] <= now]
            early = [r for r in current if r['receipt_seconds'] < now]
            same_time = [r for r in current if r['receipt_seconds'] == now]
            irreversible = right['kind'] == 'onboard' or right['choice'] is not None
            counts['adjacent_decision_intervals'] += 1
            counts['intervals_before_irreversible_action'] += irreversible
            counts['intervals_with_new_public_receipts'] += bool(current)
            counts['intervals_with_strictly_early_receipts'] += bool(early)
            counts['early_intervals_before_irreversible_action'] += bool(early) and irreversible
            counts['receipt_occurrences'] += len(current)
            counts['strictly_early_receipt_occurrences'] += len(early)
            counts['same_clock_receipt_occurrences'] += len(same_time)
            counts['receipts_after_consumption_clock'] += sum(r['receipt_seconds'] > now for r in receipts)
            counts['intervals_with_numeric_refresh'] += any(r['changed_events'] for r in early)
            counts['intervals_with_numeric_withdrawal'] += any(r['cleared_events'] for r in early)
            counts['intervals_with_cancel_or_skip_notice'] += any(
                status[-1] in ('CANCELED', 'DELETED', 'SKIPPED') for r in early for status in r['statuses'])
            if early:
                affected_roots.add(index)
                if irreversible:
                    irreversible_roots.add(index)
            for receipt in early:
                publication = receipt['publication']
                unique_frames.add((row['day'], publication.get('archive_member'),
                                   publication.get('entity_id'), receipt['receipt_seconds']))
                gaps.append(now-receipt['receipt_seconds'])
            pairs.append(dict(root_index=index, day=row['day'],
                previous_clock=before, next_clock=now, previous_kind=left['kind'], next_kind=right['kind'],
                next_choice=right['choice'], next_action_irreversible=irreversible,
                new_receipts=len(current), strictly_early_receipts=len(early),
                numeric_refreshes=sum(len(r['changed_events']) for r in early),
                receipt_clocks=sorted({r['receipt_seconds'] for r in early})))
    counts['roots_with_strictly_early_receipts'] = len(affected_roots)
    counts['roots_with_early_receipt_before_irreversible_action'] = len(irreversible_roots)
    counts['distinct_early_date_entity_frames'] = len(unique_frames)
    return dict(counts=dict(counts), early_lead_seconds=dict(minimum=min(gaps),median=median(gaps),
        mean=mean(gaps),maximum=max(gaps)), intervals=pairs,
        interpretation='Availability before the next saved ideal event action, not a measured door deadline. Receipt counts include refreshes with unchanged forecast values and original-root-menu messages that may no longer affect the remaining route. This census establishes temporal opportunity only, not decision value.',
        scope='All 72 frozen V40 cheap_frontier__combined_rolling paths; adjacent decisions only, excluding initial accumulated history and the terminal arrival. Descriptive reuse, no new MC or control.')


if __name__ == '__main__':
    result = dict(future_publication_model=dict(defined=False, fitted=False, validated=False,
        fitted_object='V35 route/kind Gaussian error at an already received endpoint forecast; 334 training records on October 1-2.',
        validated_object='V35 current-history endpoint/onboard predictions; V36 two physical departure/arrival joint events. Neither validates future publication times, message contents/revisions or their joint history-dependent likelihood.',
        code_contract='freeze_snapshot removes future frames; V41 evaluate returns future_messages=0.',
        admission='Direction B prerequisites are not yet established. Timing opportunities alone do not admit a future-message algorithm study; this is not evidence of impossibility.'),
        timing_opportunities=census())
    path = HERE/'future_message_opportunities.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(path)
