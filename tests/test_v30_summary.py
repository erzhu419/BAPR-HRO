from experiments.reconstruction_v30.summarize import pair_summary


def test_no_unseen_target_is_board_only_not_a_missing_wait_outcome():
    board_only = dict(history_id='last_offer', target_cid=None,
        by_law={law: dict(board_id='board', target_id=None, nominal_gain=None, integrated_gain=None)
                for law in ('original20', 'route_empirical20')}, actual_gain=None)
    summary = pair_summary([board_only], {}, {})
    assert summary['records'] == summary['board_only'] == 1
    assert summary['n'] == summary['missing_actual'] == summary['paired_actual'] == 0
    assert summary['all_pair_mean_gain_identification_interval'] == [None, None]
    assert summary['actual_wait_gain']['mean'] is None
    assert all(row['cells'] == 0 for row in summary['by_law'].values())
