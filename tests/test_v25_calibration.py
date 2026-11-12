import numpy as np
import pytest

from experiments.reconstruction_v25.calibration import (
    DepartureForecast, endpoint_record, summarize_endpoints, summarize_future,
)


def test_operating_interval_and_service_mass_are_calibrated_separately():
    forecast = DepartureForecast(np.array([0, 10, 20]), np.array([.2, .4, .2]), .2)
    # The arrival endpoint may be absent; a known departure is sufficient.
    metadata = dict(outcome='UNRESOLVED', departure=dict(
        status='REAL', time_seconds=10, official_journey='vehicle'))
    row = endpoint_record(forecast, metadata, 'identified')
    assert row['predicted_clock'] == 10.
    assert row['intervals']['0.5']['covered']
    summary = summarize_endpoints([row])
    assert summary['measured_departures'] == summary['scored_departures'] == 1
    assert summary['service']['brier'] == pytest.approx(.04)
    assert summary['mean_error'] == 0


def test_unseen_future_probability_keeps_before_entry_and_canceled_mass():
    forecast = DepartureForecast(np.array([0, 10, 20]), np.array([.4, .2, .3]), .1)
    # Local absence excludes time10, but not the unobserved time0 departure.
    assert forecast.future_probability(5, 15, False) == pytest.approx(.3/.8)
    assert forecast.future_probability(5, 15, True) == 0.


def test_zero_likelihood_public_absence_is_not_filled_with_zero_probability():
    forecast = DepartureForecast(np.array([10]), np.array([1.]), 0.)
    assert forecast.future_probability(5, 15, False) is None
    record = dict(future_probability=None, actual_future=False, actual_before_entry=True,
                  public_status='unseen', root_ids=['root'])
    summary = summarize_future([record])
    assert summary['unseen']['unsupported_history'] == 1
    assert summary['unseen']['brier'] is None


def test_probability_bins_count_unique_histories_and_keep_root_multiplicity_visible():
    records = [dict(future_probability=p, actual_future=y, actual_before_entry=not y,
                    public_status='unseen', root_ids=roots)
               for p, y, roots in ((.2, False, ['a', 'b']), (1., True, ['c']))]
    summary = summarize_future(records)
    assert summary['unseen']['n'] == 2
    assert summary['unseen']['root_candidate_occurrences'] == 3
    assert summary['unseen_probability_bins']['0.2-0.4']['n'] == 1
    assert summary['unseen_probability_bins']['0.8-1.0']['n'] == 1
    assert summary['unseen']['brier'] == pytest.approx(.02)
