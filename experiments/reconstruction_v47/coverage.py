"""Input-only coverage of the existing V46 independent mixed-platform kernel."""
from pathlib import Path

from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.run import setup
from experiments.reconstruction_v38.run import restore_node, save
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.run import normalized
from experiments.reconstruction_v46.oracle import structural, terminal, physical_path

HERE = Path(__file__).resolve().parent


def inspect():
    contract = read(HERE/'contract.json')
    manifest = read(HERE.parent/'reconstruction_v38/seed_manifest.json')
    data = read(HERE.parent/'reconstruction_v34/data.json.gz')
    fit = read(HERE.parent/'reconstruction_v34/model_fit.json')
    records = []
    for index in contract['state_indices']:
        state = manifest[index]['state']
        graph = context(data, state['day'])[0]
        model, parameters = setup(graph, fit)
        node = restore_node(state['node'])
        planner = MatchedContinuationPlanner(graph, model, parameters, state['deadline'])
        old = read(HERE.parent/'reconstruction_v45/banks'/f'{index:03}_absorbing.json.gz')
        for arm in ('alight', 'continue'):
            groups = {}
            for atom_index, atom in enumerate(old['atoms'][arm]):
                if atom['kind'] != 'platform' or atom['probability'] <= 0:
                    continue
                cid = atom['cid']
                platform = node.exits[cid]
                if normalized(planner.structure(platform)) != normalized(atom['node_structure']):
                    raise ValueError('Frozen input platform differs from saved positive atom')
                if cid not in groups:
                    original = structural(model, platform)
                    pure_terminal = all(terminal(ride) for ride in platform.rides.values())
                    groups[cid] = dict(index=index, root_index=state['root_index'], arm=arm,
                        head_exit_cid=cid, stop=platform.stop,
                        original_v46_bulk_eligible=original['eligible'],
                        original_gate_reasons=original['reasons'],
                        terminal_only_hybrid_eligible=original['eligible'] and pure_terminal,
                        pure_physical_terminal=pure_terminal,
                        candidates=[dict(cid=c, physical_terminal=terminal(ride),
                            physical_path=list(physical_path(ride)), original_path=list(ride.path))
                            for c, ride in platform.rides.items()],
                        atom_indices=[], probability=0.)
                groups[cid]['atom_indices'].append(atom_index)
                groups[cid]['probability'] += atom['probability']
            records.extend(groups.values())
    result = dict(scope='Input menus plus previously paid V45 positive prefix atoms; no policy or value evaluation',
        states=len(contract['state_indices']), groups=len(records),
        positive_platform_atoms=sum(len(r['atom_indices']) for r in records),
        original_v46_eligible_groups=sum(r['original_v46_bulk_eligible'] for r in records),
        terminal_only_hybrid_eligible_groups=sum(r['terminal_only_hybrid_eligible'] for r in records),
        original_v46_eligible_positive_atoms=sum(len(r['atom_indices']) for r in records
                                                if r['original_v46_bulk_eligible']),
        all_positive_atoms_retained=True, records=records,
        conclusion='The bulk replacement must admit the original structural(model, platform) class, including mixed one-transfer platforms. Terminal-only admission omits every first-platform fast path in this panel.')
    save(HERE/'bulk_coverage.json', result)
    print({k:v for k,v in result.items() if k not in ('records', 'conclusion')})
    return result


if __name__ == '__main__':
    inspect()
