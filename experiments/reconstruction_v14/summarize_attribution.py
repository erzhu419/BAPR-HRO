"""Make a per-root ledger without imputing any unidentified population gap."""

import csv
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / f'{name}.json').read_text())


def main():
    city = read('city_ablation_results')
    assert city['status'] == 'complete'
    roots = {tuple(r['case']): r for r in city['summary']}
    precision = {tuple(r['case']): r for r in read('confidence_results')['rows']}
    rows = []
    for root in read('menu_info')['roots']:
        case = tuple(root['case'])
        summary = root['summary']
        compare = roots[case]['comparisons']
        guarded = summary['expanded_guarded']['mean_loss']
        oracle_after_chosen = mean(r['policies']['expanded_guarded']['hindsight_same_first']
                                   for r in root['rows'])
        rows.append(dict(
            date=case[0], query=case[1], decision=case[2],
            first_visible_s=summary['first_visible']['mean_loss'],
            guarded_s=guarded,
            unrestricted_first_visible_s=summary['expanded_first_visible']['mean_loss'],
            hindsight_ab_s=summary['hindsight_ab']['mean_loss'],
            hindsight_expanded_s=summary['hindsight_expanded']['mean_loss'],
            hindsight_full_s=summary['full_hindsight']['mean_loss'],
            guarded_after_same_first_hindsight_s=oracle_after_chosen,
            guarded_suffix_difference_s=guarded-oracle_after_chosen,
            guarded_choice_difference_s=oracle_after_chosen-summary['hindsight_expanded']['mean_loss'],
            selected_r_hidden_minus_revealed_s=round(compare['r_hidden_minus_r_revealed']['mean_difference'], 8),
            sampled_suffix_hidden_minus_revealed_s=round(
                compare['suffix_labels_hidden_minus_suffix_labels_revealed']['mean_difference'], 8),
            delayed_720_completed=roots[case]['arms']['tail_720']['completed'],
            delayed_720_complete_pair_delta_s=compare['tail_720_minus_v13']['mean_difference'],
            delayed_1080_completed=roots[case]['arms']['tail_1080']['completed'],
            v13_point_gap_s=precision[case]['sample_mean_delta'],
            v13_existing_paired_upper_s=precision[case]['paired_existing_upper'],
            v13_adaptive_paired_upper_s=precision[case]['paired_predictable_eb']['upper'],
            population_causal_policy_regret_s=None, population_other_vehicle_information_s=None,
            population_hindsight_tail_s=None, population_finite_bank_bias_s=None,
        ))
    with (HERE / 'attribution.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
