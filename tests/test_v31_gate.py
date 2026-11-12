from copy import deepcopy

from experiments.reconstruction_v31.run import admission


def equal_evidence():
    first = dict(n=139, support={'supported': 139}, metrics={
        'final_cost': dict(n=129, baseline_n=129, candidate_n=129,
                           baseline={'crps': 80.}, candidate={'crps': 80.})}, binary={
        'legal_target_appears': dict(n=72, baseline_n=72, candidate_n=72,
                                     baseline=.02, candidate=.02)})
    by_day = {day: deepcopy(first) for day in ('2023-07-16', '2023-07-17')}
    all_forecasts = deepcopy(first)
    all_forecasts.update(n=151, support={'supported': 151})
    all_forecasts['metrics']['final_cost'].update(n=141, baseline_n=141, candidate_n=141)
    origin = dict(paired=48, mean_gain=0., baseline_known=48, candidate_known=48,
                  baseline_arrivals=48, candidate_arrivals=48)
    transfer = dict(paired=25, mean_gain=0., baseline_known=25, candidate_known=25,
                    baseline_arrivals=19, candidate_arrivals=19)
    policy = {kind: {method: deepcopy(row) for method in ('one_step_nominal', 'single_target_nominal')}
              for kind, row in [('origin', origin), ('first_transfer', transfer)]}
    return first, by_day, policy, all_forecasts


def test_equal_scores_and_complete_coverage_retain_development_candidate():
    result = admission(*equal_evidence())
    assert result['status'] == 'retain_development_candidate'
    assert result['failed'] == []


def test_secondary_support_daily_cost_and_label_losses_cannot_be_hidden_by_aggregate():
    for failure in ('support', 'daily_cost', 'forecast_labels', 'policy_labels'):
        first, by_day, policy, all_forecasts = equal_evidence()
        if failure == 'support':
            # First-offer support still passes; only a later V29 cell is lost.
            all_forecasts['support'] = {'supported': 150, 'unsupported': 1}
            expected = 'support'
        elif failure == 'daily_cost':
            by_day['2023-07-17']['metrics']['final_cost']['candidate']['crps'] += 1.
            expected = 'total_cost_crps_overall_and_each_date'
        elif failure == 'forecast_labels':
            all_forecasts['metrics']['final_cost']['candidate_n'] -= 1
            expected = 'forecast_labels'
        else:
            # A secondary one-step route becoming unknown also loses coverage.
            policy['first_transfer']['one_step_nominal']['candidate_known'] -= 1
            expected = 'policy_labels'
        result = admission(first, by_day, policy, all_forecasts)
        assert result['status'] == 'stop_segment_candidate'
        assert result['failed'] == [expected]
