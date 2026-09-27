from __future__ import annotations

import numpy as np
import pandas as pd

from treasuryshield.development.models import panel_fixed_effects


def test_within_bank_panel_model_removes_bank_intercepts_without_test_labels() -> None:
    rows = []
    for bank_id, intercept in ((1, -.1), (2, 0.0), (3, .1)):
        for quarter in range(20):
            x = quarter / 20 + bank_id * .07
            rows.append({"bank_id": bank_id, "x": x,
                         "deposit_log_growth": intercept + .2 * x})
    train = pd.DataFrame(rows)
    calibration = pd.DataFrame({"bank_id": [1, 2, 3], "x": [.2, .4, .6],
                                "deposit_log_growth": [0., 0., 0.]})
    test = pd.DataFrame({"bank_id": [1, 2, 3], "x": [.3, .5, .7],
                         "deposit_log_growth": [9., 9., 9.]})
    result = panel_fixed_effects(train, calibration, test, ["x"])
    expected = np.array([-.1 + .2 * .3, .2 * .5, .1 + .2 * .7])
    np.testing.assert_allclose(result.test[:, 4], expected, atol=5e-4)
    assert result.metadata["training_banks"] == 3
