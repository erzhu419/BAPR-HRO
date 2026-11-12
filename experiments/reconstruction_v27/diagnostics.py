"""Read saved tables/outcomes to retain proxy gaps and all DP/cheap disagreements."""

import gzip
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'reconstruction_v26'


def main():
    with gzip.open(PREVIOUS/'results.json.gz', 'rt') as handle:
        old = {(r['law'], r['root_id']): r for r in json.load(handle)['results']}
    with gzip.open(HERE/'results.json.gz', 'rt') as handle:
        results = json.load(handle)['results']
    gaps, differences = [], []
    for row in results:
        law, rid = row['law'], row['root_id']
        index = (int(rid.split('_')[2])-27600)//900
        prior = old[law, rid]
        with np.load(PREVIOUS/f'{law}_{index}_values.npz') as dp, np.load(HERE/f'{law}_{index}_values.npz') as new:
            supported = dp['supported']
            gaps.append(dict(law=law, root_id=rid, supported_states=int(supported.sum()),
                single_initial_gap=row['proxy']['expected_loss']['single_target_nominal']-prior['model']['expected_loss']['multistep_nominal'],
                single_max_supported_gap=float(np.max((new['single_target_nominal']-dp['dp'])[supported])),
                combined_max_supported_gap=float(np.max((new['combined_wait_nominal']-dp['dp'])[supported]))))
        for index, (sample, previous) in enumerate(zip(row['new_samples'], prior['samples'], strict=True)):
            cheap, dp = sample['outcomes']['single_target_nominal'], previous['outcomes']['multistep_nominal']
            if any(cheap[k] != dp[k] for k in ('action', 'tail', 'loss', 'arrived', 'status')):
                differences.append(dict(law=law, root_id=rid, world=index, single_target=cheap, dp=dp))
    (HERE/'diagnostics.json').write_text(json.dumps(dict(
        scope='Saved finite proxy tables only; no physical regret guarantee and no new worlds.',
        state_gaps=gaps, single_target_dp_disagreements=differences), indent=2)+'\n')


if __name__ == '__main__':
    main()
