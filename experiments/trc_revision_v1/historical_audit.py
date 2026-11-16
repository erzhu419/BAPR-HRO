"""Recover both conditional weightings from saved NG2 incident bounds."""
import argparse
from pathlib import Path

import numpy as np

from experiments.decision_gap_v1.panel import read,write
from experiments.g2_reset_v1.calibration_split import (
    PARTITIONS,_whole_date_bootstrap,extract_conditional_from_compact_summary,
)

HERE=Path(__file__).resolve().parent


def evaluate(source,output):
    summary=read(source)
    # This existing recovery verifies every incident's unresolved widths,
    # probability/indicator/Brier widths and daily event denominators.
    event=extract_conditional_from_compact_summary(summary)
    daily=[]
    for day in sorted(summary['days'],key=lambda r:r['day']):
        values=[]
        for incident in day['incidents']:
            n=len(incident['event_row_indices'])
            lo,hi=incident['bias_bounds']['00']
            reached=n-int(round(n*(hi-lo)/2))
            if reached:
                values.append(np.array([n*sum(incident['bias_bounds'][k])/2/reached
                    for k in PARTITIONS]))
        daily.append(dict(day=day['day'],nonempty_original_incidents=len(values),
            mean_residual={k:float(v) for k,v in zip(PARTITIONS,np.mean(values,axis=0))}))
    incident_inference=_whole_date_bootstrap([[d['mean_residual'][k] for k in PARTITIONS] for d in daily])
    result=dict(source=str(source),source_stage='natural_gap_v2',
        derivation='Algebraic recovery from saved original incident bounds; no new model, control, labels or sampled worlds.',
        original_acceptance=dict(verdict=summary['verdict'],primary=summary['primary']),
        population={k:event[k] for k in ('total_registered_opportunities',
            'unreached_initial_cancellation_opportunities','conditional_reached_opportunities')},
        source_diagnostics=summary.get('source_diagnostics',{}),
        weightings=dict(
            equal_incident_then_equal_date=dict(date_cluster_inference=incident_inference,days=daily,
                definition='Average reached events inside each originally registered incident; average nonempty incidents within date; average dates.'),
            equal_event_then_equal_date=dict(date_cluster_inference=event['date_cluster_inference'],
                per_day_mean_residual=event['per_day_mean_residual'],
                definition='Average all reached events within each date; average dates. This changes the original incident weighting.')),
        scope='These conditional diagnostics change the target population. Event weighting additionally changes the estimand. Neither replaces the original unresolved calibration verdict or establishes a new-policy calibration pass.')
    write(output,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=HERE.parent/'natural_gap_v2/calibration_summary.json')
    p.add_argument('--output',type=Path,default=HERE/'historical_weighting_audit.json')
    a=p.parse_args();evaluate(a.source,a.output)
