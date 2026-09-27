"""Independent prelock subgroup and data-perturbation audit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from treasuryshield.development.models import fit_model
from treasuryshield.features.build import CORE_NUMERIC
from treasuryshield.orchestration.state import RunState, sha256_file, stable_hash
from treasuryshield.orchestration.warehouse import connect
from treasuryshield.splits.temporal import SPLITS, load_joined, partition
from treasuryshield.statistics.metrics import pinball, rearrange


MODEL = "B8_lightgbm_quantile"
SPLIT = "tightening"
RUN_ID = "robustness_core_B_1q_tightening_B8"


def _predict(model: dict, frame: pd.DataFrame) -> np.ndarray:
    x = model["preprocessor"].transform(frame)
    return rearrange(np.column_stack([estimator.predict(x) for estimator in model["models"]]))


def _stat(y: np.ndarray, pred: np.ndarray, weight: np.ndarray | None = None) -> tuple[float, float, int]:
    if weight is None:
        weight = np.ones(len(y))
    weight = np.maximum(np.nan_to_num(np.asarray(weight, dtype=float)), 0)
    return (float(np.average(pinball(y, pred[:, 1], .05), weights=weight)),
            float(np.average(y <= pred[:, 1], weights=weight)), len(y))


def run(root: Path) -> None:
    output = root / "artifacts" / "robustness_prelock.json"
    config = {"model": MODEL, "split": SPLIT, "mode": "B", "horizon": 1,
              "seed": 1701, "source_hash": sha256_file(Path(__file__)),
              "snapshot": RunState(root).data["data_snapshot_id"]}

    def execute() -> None:
        frame = load_joined(root, "core", 1, mode="B")
        frame = frame.loc[frame.report_period.le(SPLITS[SPLIT].test_end)].copy()
        train, calibration, test = partition(frame, SPLITS[SPLIT])
        result = fit_model(MODEL, train, calibration, test, list(CORE_NUMERIC), seed=17)
        baseline = rearrange(result.test)
        y = test.deposit_log_growth.to_numpy(dtype=float)
        base_loss, base_coverage, _ = _stat(y, baseline)
        rng = np.random.default_rng(config["seed"])
        variants: list[dict] = []
        feature_names = list(CORE_NUMERIC)

        def mutable_test() -> pd.DataFrame:
            modified = test.copy()
            modified[feature_names] = modified[feature_names].astype(float)
            return modified

        def score_variant(experiment_id: str, kind: str, level: str, modified: pd.DataFrame,
                          *, changed_cells: int) -> None:
            pred = _predict(result.model, modified)
            loss, coverage, n = _stat(y, pred)
            variants.append({"experiment_id": experiment_id, "kind": kind, "level": level,
                             "q05_pinball": loss, "q05_coverage": coverage, "n": n,
                             "changed_cells": changed_cells})

        for share in (.05, .10, .20):
            modified = mutable_test()
            mask = rng.random((len(modified), len(feature_names))) < share
            values = modified[feature_names].to_numpy(copy=True)
            values[mask] = np.nan
            modified.loc[:, feature_names] = values
            score_variant("E28", "random_feature_missing", f"{share:.0%}", modified,
                          changed_cells=int(mask.sum()))

        # A previous report can only be stale, never replaced by a later one.
        prior = frame.sort_values(["bank_id", "report_period"])
        previous = prior.groupby("bank_id", sort=False)[feature_names].shift(1)
        previous.index = prior.index
        modified = mutable_test()
        old = previous.loc[test.index, feature_names]
        count = int(old.notna().to_numpy().sum())
        modified.loc[:, feature_names] = old.to_numpy()
        score_variant("E29", "previous_quarter_bank_inputs", "1_quarter", modified,
                      changed_cells=count)

        scale = train[feature_names].std(numeric_only=True).reindex(feature_names).fillna(0).to_numpy()
        for level in (.05, .10):
            modified = mutable_test()
            values = modified[feature_names].to_numpy(dtype=float, copy=True)
            noise = rng.normal(0, level, size=values.shape) * scale[None, :]
            finite = np.isfinite(values)
            values[finite] += noise[finite]
            modified.loc[:, feature_names] = values
            score_variant("E30", "gaussian_measurement_noise_train_sd", f"{level:.0%}", modified,
                          changed_cells=int(finite.sum()))

        for family, columns in (
            ("macro_features_missing", [x for x in feature_names if x.startswith("macro_")]),
            ("deposit_composition_missing", [x for x in feature_names if x in {
                "brokered_domestic_ratio", "uninsured_proxy_domestic_share",
                "uninsured_time_deposit_ratio", "uninsured_domestic_share",
                "brokered_domestic_share", "transaction_domestic_share",
                "nontransaction_domestic_share", "large_time_domestic_share"}]),
        ):
            modified = mutable_test()
            modified.loc[:, columns] = np.nan
            score_variant("E28", family, "all", modified,
                          changed_cells=len(modified) * len(columns))

        subgroups: list[dict] = []
        size = pd.cut(test.assets, [-np.inf, 100_000, 1_000_000, 10_000_000, np.inf],
                      labels=["micro", "small", "medium", "large"])
        residential = test.residential_loan_share.ge(.5)
        brokered = test.brokered_domestic_ratio.ge(.1)
        liquidity = test.cash_asset_ratio.ge(train.cash_asset_ratio.median())
        uninsured = test.uninsured_proxy_domestic_share.ge(
            train.uninsured_proxy_domestic_share.median())
        top_states = train.state.value_counts().head(10).index
        state_group = test.state.where(test.state.isin(top_states), "OTHER").fillna("UNKNOWN")
        group_series = {
            "E24": ("bank_size", size.astype(str)),
            "E25": ("business_model", pd.Series(np.select(
                [brokered, residential], ["brokered_funding", "residential_lending"],
                default="other"), index=test.index)),
            "E26": ("cash_to_assets", pd.Series(np.where(liquidity, "high", "low"), index=test.index)),
            "E27": ("uninsured_proxy", pd.Series(np.where(uninsured, "high", "low"), index=test.index)),
            "E25_geo": ("state_top10_or_other", state_group),
        }
        for experiment_id, (family, labels) in group_series.items():
            for group in sorted(labels.dropna().unique()):
                selection = labels.eq(group).to_numpy()
                if selection.sum() < 20:
                    continue
                for weighting, weights in (
                    ("equal_bank", None), ("deposit_weighted", test.deposits.to_numpy()[selection]),
                    ("asset_weighted", test.assets.to_numpy()[selection]),
                ):
                    loss, coverage, n = _stat(y[selection], baseline[selection], weights)
                    subgroups.append({"experiment_id": experiment_id, "family": family,
                                      "group": group, "weighting": weighting,
                                      "q05_pinball": loss, "q05_coverage": coverage, "n": n})

        payload = {"created_at": datetime.now(timezone.utc).isoformat(),
                   "data_snapshot_id": config["snapshot"], "config_hash": stable_hash(config),
                   "train_rows": len(train), "calibration_rows": len(calibration),
                   "test_rows": len(test), "baseline_q05_pinball": base_loss,
                   "baseline_q05_coverage": base_coverage,
                   "perturbations": variants, "subgroups": subgroups,
                   "limitations": ["Uninsured grouping uses the documented Core proxy, not direct uninsured deposits.",
                                   "Perturbations change inference inputs only; model weights remain fixed."]}
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        con = connect(root)
        try:
            con.begin()
            con.execute("DELETE FROM robustness WHERE artifact_path = 'artifacts/robustness_prelock.json'")
            con.execute("DELETE FROM subgroup_metrics WHERE run_id = ?", [RUN_ID])
            for item in variants:
                for metric, value, reference in (
                    ("q05_pinball", item["q05_pinball"], base_loss),
                    ("q05_coverage", item["q05_coverage"], base_coverage),
                ):
                    con.execute("INSERT INTO robustness VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                [item["experiment_id"], item["kind"], item["level"], SPLIT,
                                 "core", 1, metric, value, reference,
                                 "artifacts/robustness_prelock.json"])
            for item in subgroups:
                for metric, value in (("pinball", item["q05_pinball"]),
                                      ("coverage", item["q05_coverage"])):
                    con.execute("INSERT INTO subgroup_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                [RUN_ID, item["family"], item["group"], metric, "Q05",
                                 item["weighting"], value, item["n"]])
            for eid in [f"E{x:02d}" for x in range(24, 31)]:
                con.execute("UPDATE experiment_registry SET status='completed', evidence_paths=?, updated_at=current_timestamp WHERE experiment_id=?",
                            ["artifacts/robustness_prelock.json", eid])
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()
        print(f"Robustness complete: baseline Q05 pinball {base_loss:.6f}, {len(variants)} perturbations")

    RunState(root).run(RUN_ID, "RUNNING_EXPERIMENTS", config,
                       ["artifacts/robustness_prelock.json"], execute, resume=True)


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[3])
