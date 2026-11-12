"""Read-only summary of the declared post-readout cheap-baseline audit."""

from pathlib import Path

from experiments.reconstruction_v34.summarize import arm_summary, first_divergence, paired, read, write


HERE = Path(__file__).resolve().parent


def main():
    methods = read(HERE/'cheap_baseline_protocol.json')['methods']
    selected = read(HERE.parent/'reconstruction_v34/selection.json')['selected']
    indices = list(range(len(selected)))
    audited = [read(HERE/'cheap_roots'/f'{i:03}.json.gz') for i in indices]
    candidate = [read(HERE/'roots'/f'{i:03}.json.gz') for i in indices]
    previous = [read(HERE.parent/'reconstruction_v36/roots'/f'{i:03}.json.gz') for i in indices]
    for index, root in enumerate(selected):
        if any(row[index]['root_id'] != root['root_id'] for row in (audited, candidate, previous)):
            raise ValueError(f'Root {index} differs from the frozen cohort')
    controls = {name: [root['rows']['public_messages'][name] for root in audited] for name in methods}
    comparisons = {name: paired(dict(indices=indices, rows=candidate), rows) for name, rows in controls.items()}
    same_actions = {}
    for name, rows in controls.items():
        detail = []
        for index, row in enumerate(rows):
            old = previous[index]['rows']['public_messages']['combined_rolling']
            detail.append(dict(root_index=index, root_id=selected[index]['root_id'],
                **first_divergence(row['outcome'], old['outcome']),
                compared_trace_lengths=[len(row['outcome']['trace']), len(old['outcome']['trace'])],
                source=f'cheap_roots/{index:03}.json.gz',
                baseline_source=f'../reconstruction_v36/roots/{index:03}.json.gz'))
        same_actions[name] = dict(denominator=len(detail),
            all_actions_identical=all(row['actions_identical'] for row in detail),
            all_common_prefix_histories_equal=all(row['common_prefix_history_equal'] for row in detail),
            rows=detail)
    worlds = sum(gate['count'] for rows in controls.values() for row in rows for gate in row['gates'])
    result = dict(protocol='cheap_baseline_protocol.json', post_readout_audit=True,
        primary_results_unchanged=True, denominator=len(indices), deterministic_replays=len(indices)*len(methods),
        no_new_worlds=worlds == 0, conditional_worlds=worlds,
        gain_orientation='cheap baseline loss minus unchanged candidate loss; positive favors candidate',
        candidate=arm_summary(candidate), controls={name: arm_summary(rows) for name, rows in controls.items()},
        candidate_vs_cheap=comparisons, same_as_V36_combined=same_actions,
        timing='Candidate CPU/wall are fully charged control costs including reused evaluations; cheap costs are complete cold measured control costs. Common input/graph preparation is separate.',
        interpretation='Additional comparators declared after the primary readout. This checks the scoped baseline gap, not algorithm novelty or independent confirmation.')
    write(HERE/'cheap_baseline_summary.json', result)
    print(dict(replays=result['deterministic_replays'], conditional_worlds=worlds,
        comparisons={name: dict(positive=row['positive'], negative=row['negative'], tie=row['tie'],
                               mean_gain=row['gain_seconds']['mean']) for name, row in comparisons.items()}))


if __name__ == '__main__':
    main()
