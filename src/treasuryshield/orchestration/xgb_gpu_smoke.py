"""Fit actual XGBoost quantile models on CUDA and CPU; record device config."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xgboost as xgb


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    rng = np.random.default_rng(20260927)
    x = rng.normal(size=(40000, 20)).astype(np.float32)
    y = (0.4 * x[:, 0] - 0.25 * x[:, 1] ** 2 + rng.standard_t(5, size=len(x))).astype(np.float32)
    result: dict[str, object] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "xgboost": xgb.__version__,
        "objective": "reg:quantileerror",
        "quantile_alpha": 0.05,
        "rows": len(x),
        "features": x.shape[1],
    }
    for device in ("cpu", "cuda"):
        model = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=0.05,
            tree_method="hist", device=device, n_estimators=60,
            max_depth=4, learning_rate=0.08, n_jobs=4, random_state=20260927,
        )
        start = time.perf_counter()
        model.fit(x, y)
        elapsed = time.perf_counter() - start
        config = json.loads(model.get_booster().save_config())
        recorded_device = config["learner"]["generic_param"]["device"]
        result[f"{device}_fit_seconds"] = elapsed
        result[f"{device}_recorded_device"] = recorded_device
        result[f"{device}_predictions_head"] = model.predict(x[:3]).tolist()
        if device == "cuda" and not recorded_device.startswith("cuda"):
            raise RuntimeError(f"XGBoost silently fell back to {recorded_device}")
    out = root / "artifacts" / "xgb_gpu_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
