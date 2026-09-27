from __future__ import annotations

import numpy as np
import pandas as pd

from treasuryshield.preprocessing.core import quarterize_ytd
from treasuryshield.calibration.rolling import calibrate
from treasuryshield.regimes.filtered import _filtered_hmm
from treasuryshield.splits.temporal import Split, partition


def test_quarterization_requires_immediately_previous_quarter() -> None:
    frame = pd.DataFrame({
        "bank_id": [1, 1, 1, 1, 2, 2],
        "stata_quarter": [256, 257, 259, 260, 256, 258],
        "year": [2024, 2024, 2024, 2025, 2024, 2024],
        "quarter": [1, 2, 4, 1, 1, 3],
        "ytdincome": [2, 5, 14, 4, 10, 15],
    })
    out = quarterize_ytd(frame, ["ytdincome"])
    values = out.loc[out.bank_id.eq(1), "income"].to_numpy()
    np.testing.assert_allclose(values[[0, 1, 3]], [2, 3, 4])
    assert np.isnan(values[2])
    assert np.isnan(out.loc[out.bank_id.eq(2) & out.quarter.eq(3), "income"].iloc[0])


def test_partition_excludes_training_labels_published_after_cutoff() -> None:
    periods = pd.period_range("2016Q1", "2020Q4", freq="Q")
    frame = pd.DataFrame({
        "report_period": periods.astype(str),
        "target_quarter_end": (periods + 1).to_timestamp(how="end").normalize(),
        "target_availability_date": (periods + 1).to_timestamp(how="end").normalize() + pd.Timedelta(days=45),
        "forecast_origin": periods.to_timestamp(how="end").normalize() + pd.Timedelta(days=45),
        "mna_contaminated_radius_1": False,
    })
    train, calibration, test = partition(frame, Split("synthetic", "2019Q4", "2020Q1", "2020Q4"))
    assert train["target_quarter_end"].max() <= pd.Timestamp("2019-12-31")
    assert calibration["target_quarter_end"].max() <= pd.Timestamp("2019-12-31")
    assert test["report_period"].min() == "2020Q1"


class _ToyHMM:
    startprob_ = np.array([.6, .4])
    transmat_ = np.array([[.8, .2], [.3, .7]])

    def _compute_log_likelihood(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([-.5 * (x[:, 0] + 1) ** 2,
                                -.5 * (x[:, 0] - 1) ** 2])


def test_filtered_hmm_cannot_change_past_probabilities_when_future_appended() -> None:
    hmm = _ToyHMM()
    history = np.array([[-1.0], [.2], [1.1], [-.5]])
    prior = _filtered_hmm(hmm, history)
    extended = _filtered_hmm(hmm, np.vstack([history, [[8.0], [-7.0]]]))
    np.testing.assert_allclose(prior, extended[: len(history)], atol=1e-12)


def test_adaptive_calibration_uses_only_matured_previous_outcomes() -> None:
    calibration = pd.DataFrame({
        "report_period": ["2020Q1", "2020Q2", "2020Q3", "2020Q4"],
        "target_availability_date": pd.to_datetime(["2020-08-15", "2020-11-15",
                                                     "2021-02-15", "2021-04-15"]),
        "deposit_log_growth": [-.06, -.02, .01, .03],
    })
    test = pd.DataFrame({
        "report_period": ["2021Q1", "2021Q2", "2021Q3"],
        "forecast_origin": pd.to_datetime(["2021-05-15", "2021-08-15", "2021-11-15"]),
        "target_availability_date": pd.to_datetime(["2021-08-15", "2021-11-15",
                                                     "2022-02-15"]),
        "deposit_log_growth": [-.01, .02, -.03],
    })
    cal_pred = np.tile(np.array([-.10, -.07, -.04, -.02, 0, .02, .04]), (4, 1))
    test_pred = np.tile(np.array([-.10, -.07, -.04, -.02, 0, .02, .04]), (3, 1))
    baseline, metadata = calibrate(calibration, cal_pred, test, test_pred,
                                   kind="adaptive_global")
    changed = test.copy()
    changed.loc[0, "deposit_log_growth"] = -.20
    after, _ = calibrate(calibration, cal_pred, changed, test_pred, kind="adaptive_global")
    np.testing.assert_allclose(baseline[:2], after[:2])
    assert not np.allclose(baseline[2], after[2])
    assert metadata["rolling_origin_updates"][1]["feedback_rows"] == 0
    assert metadata["rolling_origin_updates"][2]["feedback_rows"] == 1
