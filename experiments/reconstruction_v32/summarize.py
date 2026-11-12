"""Keep model, physical paired draws, and measured-label evidence separate."""

from collections import Counter
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from experiments.reconstruction_v32.journey import paired_statistics
from experiments.reconstruction_v32.run import HERE, PRIMARY


def summarize(directory, independent_streams=False):
    rows = []
    for path in sorted(directory.glob('*.json.gz')):
        with gzip.open(path, 'rt') as handle:
            rows.append(json.load(handle))
    if not rows:
        return None
    methods = tuple(rows[0]['actual'])
    output = {}
    for method in methods:
        pairs = [(r['actual'][PRIMARY], r['actual'][method]) for r in rows]
        known = [(a, b) for a, b in pairs if a['loss'] is not None and b['loss'] is not None]
        actual_gain = [a['loss']-b['loss'] for a, b in known]
        stats = paired_statistics(
            np.concatenate([r['sampled_losses'][PRIMARY] for r in rows]),
            np.concatenate([r['sampled_losses'][method] for r in rows]))
        if independent_streams:
            differences = [np.asarray(r['sampled_losses'][PRIMARY])-r['sampled_losses'][method]
                           for r in rows]
            se = float(np.sqrt(sum(d.var(ddof=1)/len(d) for d in differences))/len(rows))
            stats['se'] = se
            stats['ci95'] = [stats['mean_gain']-1.96*se, stats['mean_gain']+1.96*se]
            stats['uncertainty_scope'] = 'Monte Carlo error conditional on these twelve fixed roots'
        else:
            stats['uncertainty_scope'] = 'Initial cap diagnostic; reused seed numbers across roots, no aggregate inference'
        # This SE measures independent simulation error conditional on the
        # selected roots, not network/day sampling uncertainty.
        output[method] = dict(model_mean_loss=float(np.mean([r['model_values'][method] for r in rows])),
            independent_draw_mean_loss=float(np.mean([r['sampled_losses'][method] for r in rows])),
            paired_draws=stats, actual=dict(paired=len(known),
                mean_loss=float(np.mean([b['loss'] for a, b in known])) if known else None,
                mean_gain=float(np.mean(actual_gain)) if known else None,
                wins=sum(x > 0 for x in actual_gain), harms=sum(x < 0 for x in actual_gain),
                ties=sum(x == 0 for x in actual_gain),
                arrived=sum(b['status'] == 'arrived' for _, b in pairs),
                statuses=dict(Counter(b['status'] for _, b in pairs))))
    catalog = []
    for r in rows:
        model_gap = r['model_values'][PRIMARY]-r['model_values']['pair_optimal']
        attributed = sum(a['contribution'] for a in r['attribution'])
        if abs(model_gap-attributed) > 1e-6:
            raise AssertionError(('Bellman attribution differs from policy loss', r['root_id'], model_gap, attributed))
        catalog.append(dict(root_id=r['root_id'], day=r['day'], layer=r['layer'],
            model_residual=model_gap, attributed_residual=attributed,
            independent_pair=r['paired']['pair_optimal'], actual_primary=r['actual'][PRIMARY],
            actual_stopping=r['actual']['pair_optimal'], public_gap_histories=r['attribution'],
            actual_all={m: dict(loss=x['loss'], status=x['status']) for m, x in r['actual'].items()},
            cost=r['wall']))
    return dict(roots=len(rows), methods=output, catalog=catalog,
        wall={key: sum(r['wall'][key] for r in rows) for key in rows[0]['wall']},
        numerical_attribution='Each root Bellman residual sum matches its supplied-table policy gap within 1e-6.',
        scope='Development roots; exact chain for estimated G. Physical paired samples use held-out random streams, not independent dates.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-dir', type=Path, default=HERE/'roots_full')
    parser.add_argument('--output-dir', type=Path, default=HERE)
    args = parser.parse_args()
    output = dict(capped=summarize(HERE/'roots'), full_hubs=summarize(args.full_dir, independent_streams=True))
    expected = len(json.loads((HERE/'selection_full_hubs.json').read_text())['selected'])
    if output['full_hubs'] is None or output['full_hubs']['roots'] != expected:
        raise ValueError('All frozen full-hub roots are required before publishing the stage summary')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(output, indent=2)+'\n')
    full = output['full_hubs']
    if full is not None:
        (args.output_dir/'gap_catalog.json').write_text(json.dumps(full['catalog'], indent=2)+'\n')
        print(json.dumps(dict(roots=full['roots'], methods=full['methods'], wall=full['wall']), indent=2))


if __name__ == '__main__':
    main()
