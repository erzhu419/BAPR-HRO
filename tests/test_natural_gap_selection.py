from experiments.natural_gap_v1.select_dates import select_dates


def source(day, **changes):
    return dict(dict(day=day, actual_available=True, actual_domain_verified=True,
        rt_available=True, capture_coverage_verified=True, static_available=True), **changes)


def test_calibration_used_dates_cannot_be_reselected():
    result = select_dates([source('2023-03-10')], {'2023-03-10'}, 1)
    assert result['selected_dates'] == []
    assert result['rows'][0]['reasons'] == ['previously_developed_date']


def test_municipal_cache_and_missing_rt_remain_missing_cases():
    row = source('2023-07-18', actual_domain_verified=None,
                 rt_available=False, static_available=None)
    result = select_dates([row], set(), 1)
    assert result['considered_dates'] == 1
    assert result['selected_dates'] == []
    assert result['rows'][0]['reasons'] == ['v41_rail_actual_source_unverified',
        'missing_rt_archive', 'matching_historical_static_unverified']


def test_archive_filename_does_not_prove_predecision_coverage():
    row = source('2023-03-23', capture_coverage_verified=False)
    result = select_dates([row], set(), 1)
    assert result['rows'][0]['reasons'] == ['predecision_capture_coverage_unverified']


def test_selection_is_chronological_without_replacing_unavailable_dates():
    rows = [source('2023-03-25'), source('2023-03-23'),
            source('2023-03-24', rt_available=False)]
    partial = select_dates(rows, set(), 3)
    assert partial['eligible_dates'] == ['2023-03-23', '2023-03-25']
    assert partial['selected_dates'] == [] and not partial['dates_frozen']
    complete = select_dates(rows, set(), 2)
    assert complete['selected_dates'] == ['2023-03-23', '2023-03-25']
    assert complete['considered_dates'] == 3
    assert complete['original_72_additional_budget'] == 0
