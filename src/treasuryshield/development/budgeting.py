"""Secondary four-quarter balance-sheet forecast and accounting reconciliation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge

from treasuryshield.development.preprocess import NumericPreprocessor
from treasuryshield.features.build import CORE_NUMERIC
from treasuryshield.orchestration.state import RunState, sha256_file, stable_hash
from treasuryshield.orchestration.warehouse import connect
from treasuryshield.splits.temporal import deterministic_sample


BALANCE = ("assets", "liab_tot", "equity", "deposits", "ln_tot", "securities", "borrowings")
NII = "partial_nii_4q_proxy"


def _panel(root: Path) -> pd.DataFrame:
    columns = ["bank_id", "stata_quarter", "report_period", "public_availability_date",
               "forecast_origin", "assets", "liab_tot", "equity", "deposits", "ln_tot",
               "securities", "borrowings", "int_inc_ln", "int_inc_sec",
               "int_exp_dep", "int_exp_ffrepo", "int_exp_othbor"]
    core = pd.read_parquet(root / "data" / "processed" / "core_panel.parquet", columns=columns)
    core = core.sort_values(["bank_id", "stata_quarter"]).reset_index(drop=True)
    # The harmonized other-borrowing expense field is unpopulated in this
    # modern window. Keep it as a coverage diagnostic, never impute it to zero
    # or call the observed-component margin complete net interest income.
    core["partial_nii_quarter"] = (core.int_inc_ln + core.int_inc_sec - core.int_exp_dep -
                                   core.int_exp_ffrepo)
    group = core.groupby("bank_id", sort=False)
    future_quarter = group.stata_quarter.shift(-4)
    contiguous = future_quarter.sub(core.stata_quarter).eq(4)
    for target in BALANCE:
        core[f"target_{target}"] = group[target].shift(-4).where(contiguous)
        core[f"previous_{target}"] = group[target].shift(1).where(
            core.stata_quarter.sub(group.stata_quarter.shift(1)).eq(1))
    future_flow = sum(group.partial_nii_quarter.shift(-lag) for lag in range(1, 5))
    core[f"target_{NII}"] = future_flow.where(contiguous)
    core[f"current_{NII}"] = core.partial_nii_quarter * 4
    core["target_available"] = group.public_availability_date.shift(-4).where(contiguous)
    features = pd.read_parquet(root / "data" / "features" / "core_features.parquet",
                               columns=["bank_id", "report_period"] + list(CORE_NUMERIC) +
                               [f"macro_{name}_pit" for name in ("UNRATE", "CFNAI", "CPIAUCSL")])
    for name in ("UNRATE", "CFNAI", "CPIAUCSL"):
        features[f"macro_{name}_revised"] = features[f"macro_{name}_pit"]
    core = core.merge(features, on=["bank_id", "report_period"], how="inner", validate="one_to_one")
    mna = pd.read_parquet(root / "data" / "features" / "targets.parquet",
                          columns=["bank_id", "report_period", "horizon_quarters",
                                   "mna_contaminated_radius_1"])
    mna = mna.loc[mna.horizon_quarters.eq(4)].drop(columns="horizon_quarters")
    core = core.merge(mna, on=["bank_id", "report_period"], how="inner", validate="one_to_one")
    core = core.loc[core.report_period.between("2001Q1", "2023Q4") &
                    core.assets.gt(0) & core.deposits.gt(0) &
                    core.mna_contaminated_radius_1.eq(False)].copy()
    return core


def run(root: Path) -> None:
    output = root / "artifacts" / "budgeting_4q_prelock.json"
    prediction_path = root / "artifacts" / "budgeting_4q_prelock.parquet"
    config = {"panel": "core", "mode": "B", "train_end": "2021Q4",
              "test_start": "2022Q1", "test_end": "2023Q4", "horizon": 4,
              "seed": 17, "sample_max": 150_000,
              "partial_nii_components": ["int_inc_ln", "int_inc_sec",
                                         "-int_exp_dep", "-int_exp_ffrepo"],
              "omitted_other_borrowing_expense": True,
              "snapshot": RunState(root).data["data_snapshot_id"],
              "source_hash": sha256_file(Path(__file__))}

    def execute() -> None:
        frame = _panel(root)
        test = frame.loc[frame.report_period.between(config["test_start"], config["test_end"])].copy()
        if test.empty:
            raise ValueError("Empty prelock budgeting test")
        first_origin = test.forecast_origin.min()
        train = frame.loc[frame.report_period.le(config["train_end"]) &
                          frame.target_available.le(first_origin)].copy()
        train = deterministic_sample(train, config["sample_max"], config["seed"])
        if train.empty:
            raise ValueError("Empty mature budgeting training labels")
        if train.target_available.gt(first_origin).any():
            raise AssertionError("Budgeting training target unavailable at test origin")
        test = test.loc[test.target_available.notna()].copy()
        rule = lambda part: pd.DataFrame({
            "rapid_rate": part.macro_DFF_change_4q_revised.gt(2).astype(float),
            "high_volatility": part.macro_VIXCLS_revised.gt(30).astype(float)}, index=part.index)
        train = pd.concat([train, rule(train)], axis=1)
        test = pd.concat([test, rule(test)], axis=1)
        input_sets = {
            "panel_ridge": list(CORE_NUMERIC),
            "boosted_bank_only": [name for name in CORE_NUMERIC if not name.startswith("macro_")],
            "boosted_regime_aware": list(CORE_NUMERIC) + ["rapid_rate", "high_volatility"],
        }
        preprocessors = {name: NumericPreprocessor(features, linear=(name == "panel_ridge")).fit(train)
                         for name, features in input_sets.items()}
        forecasts = []
        for target in list(BALANCE) + [NII]:
            actual_column = f"target_{target}"
            current_column = f"current_{target}" if target == NII else target
            valid_train = train[actual_column].notna() & train[current_column].notna()
            valid_test = test[actual_column].notna() & test[current_column].notna()
            if valid_train.sum() < 1000 or valid_test.sum() < 100:
                continue
            tr = train.loc[valid_train]
            te = test.loc[valid_test]
            y_train = ((tr[actual_column] - tr[current_column]) / tr.assets).to_numpy(dtype=float)
            current = te[current_column].to_numpy(dtype=float)
            assets = te.assets.to_numpy(dtype=float)
            actual = te[actual_column].to_numpy(dtype=float)
            forecast = te[["bank_id", "report_period"]].copy()
            forecast["target"] = target
            forecast["actual"] = actual
            forecast["current_assets"] = assets
            forecast["persistence"] = current
            previous = te[f"previous_{target}"].to_numpy(dtype=float) if target != NII else np.full(len(te), np.nan)
            trend = current + 4 * (current - previous)
            forecast["trend"] = np.where(np.isfinite(trend), trend, current)
            for name, features in input_sets.items():
                pre = preprocessors[name]
                if name == "panel_ridge":
                    model = Ridge(alpha=100.0, random_state=17)
                else:
                    model = xgb.XGBRegressor(
                        objective="reg:squarederror", tree_method="hist", device="cuda",
                        n_estimators=100, max_depth=4, learning_rate=.06,
                        subsample=.85, colsample_bytree=.85, min_child_weight=30,
                        reg_lambda=10, max_bin=128, n_jobs=8, random_state=17)
                model.fit(pre.transform(tr), y_train)
                if name != "panel_ridge" and '"device":"cuda:' not in model.get_booster().save_config():
                    raise RuntimeError("Budgeting XGBoost did not execute on CUDA")
                estimate = current + model.predict(pre.transform(te)) * assets
                forecast[name] = estimate if target == NII else np.maximum(estimate, 0)
            forecasts.append(forecast)
            print(f"Budgeting target {target}: train={len(tr)} test={len(te)}", flush=True)
        predictions = pd.concat(forecasts, ignore_index=True)
        predictions.to_parquet(prediction_path, index=False, compression="zstd")
        metrics = []
        for (target, method), _ in (((t, m), None) for t in predictions.target.unique()
                                    for m in ("persistence", "trend", *input_sets)):
            sub = predictions.loc[predictions.target.eq(target)]
            difference = (sub[method] - sub.actual).to_numpy(dtype=float)
            weights = sub.current_assets.to_numpy(dtype=float)
            metrics.append({"target": target, "model": method, "n": len(sub),
                            "mae_fraction_current_assets": float(np.mean(np.abs(difference) / weights)),
                            "rmse_fraction_current_assets": float(np.sqrt(np.mean((difference / weights) ** 2)))})
        # Project each predicted accounting triple onto A=L+E, with equal
        # dollar-scale corrections. This is an accounting check, not a capital model.
        identity = {}
        for method in ("persistence", "trend", *input_sets):
            triple = predictions.loc[predictions.target.isin(("assets", "liab_tot", "equity")),
                                       ["bank_id", "report_period", "target", "actual", method,
                                        "current_assets"]]
            wide = triple.pivot(index=["bank_id", "report_period"], columns="target",
                                values=["actual", method, "current_assets"]).dropna()
            if wide.empty:
                continue
            a, l, e = (wide[(method, name)].to_numpy(dtype=float)
                       for name in ("assets", "liab_tot", "equity"))
            observed = np.column_stack([wide[("actual", name)].to_numpy(dtype=float)
                                        for name in ("assets", "liab_tot", "equity")])
            scale = wide[("current_assets", "assets")].to_numpy(dtype=float)
            gap = a - l - e
            raw = np.column_stack([a, l, e])
            adjusted = np.column_stack([a - gap / 3, l + gap / 3, e + gap / 3])
            identity[method] = {
                "rows": len(wide),
                "raw_mean_abs_identity_gap_fraction_assets": float(np.mean(np.abs(gap) / scale)),
                "reconciled_max_abs_identity_gap_fraction_assets": float(np.max(
                    np.abs(adjusted[:, 0] - adjusted[:, 1] - adjusted[:, 2]) / scale)),
                "raw_mean_abs_error_fraction_assets": float(np.mean(np.abs(raw - observed) / scale[:, None])),
                "reconciled_mean_abs_error_fraction_assets": float(np.mean(
                    np.abs(adjusted - observed) / scale[:, None])),
            }
        payload = {"created_at": datetime.now(timezone.utc).isoformat(),
                   "data_snapshot_id": config["snapshot"], "config_hash": stable_hash(config),
                   "train_rows": len(train), "test_rows": len(test),
                   "metrics": metrics, "accounting_reconciliation": identity,
                   "nii_definition": "Four future quarterly (loan+security interest income minus deposit and fed-funds/repo interest expense); incomplete NII proxy excluding other-borrowing interest expense.",
                   "other_borrowing_interest_expense_observed_share_test": float(test.int_exp_othbor.notna().mean()),
                   "limitations": ["A separate secondary study; not an internal budget or regulatory capital forecast.",
                                   "Core source is restated and uses an assumed 45-day public release lag.",
                                   "Other-borrowing interest expense is unavailable in the modern source window and omitted; this proxy may overstate NII.",
                                   "Accounting projection may alter component accuracy; both raw and reconciled errors are reported."]}
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        con = connect(root)
        try:
            con.begin()
            con.execute("CREATE TABLE IF NOT EXISTS budgeting_results (target VARCHAR, model VARCHAR, metric VARCHAR, value DOUBLE, n BIGINT, artifact_path VARCHAR)")
            con.execute("DELETE FROM budgeting_results WHERE artifact_path='artifacts/budgeting_4q_prelock.json'")
            for item in metrics:
                for metric in ("mae_fraction_current_assets", "rmse_fraction_current_assets"):
                    con.execute("INSERT INTO budgeting_results VALUES (?, ?, ?, ?, ?, ?)",
                                [item["target"], item["model"], metric, item[metric], item["n"],
                                 "artifacts/budgeting_4q_prelock.json"])
            for model, item in identity.items():
                for metric, value in item.items():
                    if metric == "rows":
                        continue
                    con.execute("INSERT INTO budgeting_results VALUES (?, ?, ?, ?, ?, ?)",
                                ["balance_identity", model, metric, value, item["rows"],
                                 "artifacts/budgeting_4q_prelock.json"])
            con.execute("INSERT INTO budgeting_results VALUES (?, ?, ?, ?, ?, ?)",
                        [NII, "observed_component_definition",
                         "other_borrowing_interest_expense_observed_share_test",
                         payload["other_borrowing_interest_expense_observed_share_test"],
                         len(test), "artifacts/budgeting_4q_prelock.json"])
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()
        print(f"Budgeting complete: {len(predictions)} target forecasts", flush=True)

    RunState(root).run("budgeting_4q_prelock", "RUNNING_EXPERIMENTS", config,
                       ["artifacts/budgeting_4q_prelock.json",
                        "artifacts/budgeting_4q_prelock.parquet"], execute, resume=True)


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[3])
