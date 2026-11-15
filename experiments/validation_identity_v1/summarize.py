"""Retain paired score differences and verify the original query panel."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from experiments.validation_identity_v1.validate import read

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent/'reconstruction_v35'
SCORES = ('CRPS_seconds','MAE_seconds','conditional_operating_Brier')


def main():
    protocol = read(HERE/'protocol.json')
    paired, by_day = [], {}
    own_differences = []
    for day in protocol['validation_dates']:
        new = read(HERE/'validation_days'/f'{day}.json.gz')
        old = read(ORIGINAL/'validation_days'/f'{day}.json.gz')
        day_counts = {}
        for panel in ('endpoint','onboard'):
            before = {r['query_id']:r for r in old['panels'][panel]}
            after = {r['query_id']:r for r in new['panels'][panel]}
            if before.keys() != after.keys():
                raise ValueError(f'Original {day}/{panel} query panel changed')
            counts = defaultdict(Counter)
            for qid, row in after.items():
                for field in ('source_cid','cid','kind','trip_id','route_id','clock','scheduled',
                              'actual','own_observations'):
                    if row[field] != before[qid][field]:
                        raise ValueError(f'Frozen validation query changed: {qid}/{field}')
                if row['predictions']['own_only'] != before[qid]['predictions']['own_only']:
                    own_differences.append(qid)
                item = dict(query_id=qid, day=day, panel=panel, source_cid=row['source_cid'],
                    kind=row['kind'], target_has_forecast=row['target_has_forecast'],
                    statuses={name:p['status'] for name,p in row['predictions'].items()},
                    reasons={name:p.get('reason') for name,p in row['predictions'].items()
                             if p['status'] != 'ok'}, differences={})
                for baseline in ('legacy_messages','own_only'):
                    candidate, reference = row['predictions']['fitted_messages'],row['predictions'][baseline]
                    if candidate['status'] != 'ok' or reference['status'] != 'ok':
                        counts[baseline]['unsupported_pair'] += 1
                        item['differences'][baseline] = None
                        continue
                    differences = {metric:candidate[metric]-reference[metric]
                                   for metric in SCORES if metric in candidate and metric in reference}
                    item['differences'][baseline] = differences
                    for metric, value in differences.items():
                        counts[baseline][metric+(' improved' if value<0 else ' worsened' if value>0 else ' tied')] += 1
                paired.append(item)
            day_counts[panel] = {name:dict(values) for name,values in counts.items()}
        by_day[day] = day_counts
    with gzip.open(HERE/'paired_prediction_rows.json.gz','wt') as handle:
        json.dump(dict(difference='candidate minus baseline; negative scores are better; '
                      'pairwise descriptive rows do not replace the original three-arm common-support gate',
                       rows=paired),handle,separators=(',',':'),allow_nan=False)
    events = [r for r in read(HERE/'events.json.gz')['rows'] if r['split']=='validation']
    summary = read(HERE/'validation_summary.json')
    evidence = dict(validation_endpoint_queries=len(events),
        validation_actual_statuses=dict(Counter(r['actual_status'] for r in events)),
        validation_numeric_forecast_present=sum(r['forecast_seconds'] is not None for r in events),
        validation_error_eligible=sum(r['error_eligible'] for r in events),
        source_queries_and_labels_preserved=True, scored_queries_by_panel=dict(Counter(r['panel'] for r in paired)),
        all_original_scored_query_definitions_equal=True,
        own_only_predictions_equal_to_original=len(own_differences)==0,
        own_only_different_query_ids=own_differences,
        row_files='validation_days/*.json.gz', paired_differences='paired_prediction_rows.json.gz',
        by_day_score_change_counts=by_day, original_gate=summary['gate'],
        candidate_refitted=False, controls_executed=False,
        interpretation='Forecast scoring on previously developed dates. '
            'The original V35 gate and score operator are unchanged; this is not V41 policy validation or G2.')
    (HERE/'paired_prediction_summary.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in evidence.items() if k!='by_day_score_change_counts'},indent=2))


if __name__ == '__main__':
    main()
