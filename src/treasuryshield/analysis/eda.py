"""Pre-lock coverage, quality, and stress-episode EDA.

Only origins through 2023Q4 are used to decide features and model design.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from treasuryshield.orchestration.state import RunState, sha256_file

EPISODES = {
    "pre_gfc": ("2001Q1", "2006Q4"),
    "gfc": ("2007Q3", "2009Q4"),
    "pandemic": ("2020Q1", "2021Q4"),
    "tightening": ("2022Q1", "2023Q4"),
}


def _stats(values: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"n": 0, "mean": None, "p01": None, "p05": None, "median": None, "p95": None, "p99": None}
    q = clean.quantile([.01, .05, .5, .95, .99])
    return {"n": int(len(clean)), "mean": float(clean.mean()), "p01": float(q.loc[.01]),
            "p05": float(q.loc[.05]), "median": float(q.loc[.5]),
            "p95": float(q.loc[.95]), "p99": float(q.loc[.99])}


def build(root: Path) -> None:
    features = pd.read_parquet(root / "data" / "features" / "core_features.parquet",
                               columns=["bank_id", "report_period", "year", "assets", "deposits",
                                        "cash_asset_ratio", "liquid_broad_asset_ratio",
                                        "loan_deposit_ratio", "capital_asset_ratio",
                                        "borrowings_asset_ratio", "brokered_domestic_ratio",
                                        "uninsured_proxy_domestic_share", "npl_loan_ratio",
                                        "profit_asset_ratio", "deposit_cost_annualized",
                                        "deposit_growth_qoq_lagged", "deposit_beta_4q"])
    features = features.loc[features["report_period"].le("2023Q4")].copy()
    targets = pd.read_parquet(root / "data" / "features" / "targets.parquet",
                              columns=["bank_id", "report_period", "horizon_quarters", "deposit_log_growth",
                                       "borrowings_change_asset_scaled", "liquid_narrow_change",
                                       "liquid_broad_change", "mna_contaminated_radius_1"])
    targets = targets.loc[targets["report_period"].le("2023Q4") & targets["horizon_quarters"].eq(1)]
    joined = features.merge(targets, on=["bank_id", "report_period"], how="left", validate="one_to_one")
    episodes = {}
    for name, (start, end) in EPISODES.items():
        x = joined.loc[joined["report_period"].between(start, end)]
        y = x.loc[x["mna_contaminated_radius_1"].eq(False)]
        episodes[name] = {
            "bank_quarters": len(x), "institutions": int(x["bank_id"].nunique()),
            "median_assets_thousands": float(x["assets"].median()),
            "total_deposits_thousands": float(x["deposits"].sum()),
            "cash_asset_ratio": _stats(x["cash_asset_ratio"]),
            "loan_deposit_ratio": _stats(x["loan_deposit_ratio"]),
            "capital_asset_ratio": _stats(x["capital_asset_ratio"]),
            "deposit_cost_annualized": _stats(x["deposit_cost_annualized"]),
            "deposit_1q_growth_clean": _stats(y["deposit_log_growth"]),
            "mna_contaminated_1q": int(x["mna_contaminated_radius_1"].fillna(False).sum()),
        }
    coverage = joined.groupby("report_period", observed=True).agg(
        institutions=("bank_id", "nunique"), assets_thousands=("assets", "sum"),
        deposits_thousands=("deposits", "sum"),
        uninsured_proxy_coverage=("uninsured_proxy_domestic_share", lambda s: s.notna().mean()),
        deposit_beta_coverage=("deposit_beta_4q", lambda s: s.notna().mean()),
        deposit_1q_median=("deposit_log_growth", "median"),
        deposit_1q_p05=("deposit_log_growth", lambda s: s.quantile(.05)),
    ).reset_index()
    coverage.to_parquet(root / "artifacts" / "eda_prelock_quarterly.parquet", index=False)
    correlations = {}
    clean = joined.loc[joined["mna_contaminated_radius_1"].eq(False)].copy()
    for name in ["cash_asset_ratio", "liquid_broad_asset_ratio", "loan_deposit_ratio",
                 "capital_asset_ratio", "borrowings_asset_ratio", "brokered_domestic_ratio",
                 "uninsured_proxy_domestic_share", "npl_loan_ratio", "profit_asset_ratio",
                 "deposit_cost_annualized", "deposit_growth_qoq_lagged", "deposit_beta_4q"]:
        z = clean[[name, "deposit_log_growth"]].replace([np.inf, -np.inf], np.nan).dropna()
        correlations[name] = {"n": len(z), "spearman": float(z.corr(method="spearman").iloc[0, 1]) if len(z) > 10 else None}
    extreme = joined.loc[joined["deposit_log_growth"].abs().gt(np.log(2))].copy()
    extreme["classification"] = np.where(extreme["mna_contaminated_radius_1"].fillna(False),
                                          "structural_event", "plausible_or_unresolved_extreme")
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Origins no later than 2023Q4; final 2024+ outcomes never inspected in EDA",
        "rows": len(joined), "banks": int(joined["bank_id"].nunique()),
        "episodes": episodes, "feature_target_spearman": correlations,
        "extreme_1q_absolute_log_growth_over_log2": int(len(extreme)),
        "extreme_classification": extreme["classification"].value_counts().to_dict(),
        "quarterly_artifact": "artifacts/eda_prelock_quarterly.parquet",
    }
    (root / "data" / "manifests" / "eda_prelock.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Pre-lock EDA: {len(joined)} bank-quarters, {len(extreme)} absolute >log(2) 1Q changes", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    inputs = [root / "data" / "features" / f"{x}.parquet" for x in ["core_features", "targets"]]
    RunState(root).run(
        "eda_prelock", "EXPLORING_DATA",
        {"input_sha256": {p.name: sha256_file(p) for p in inputs}, "max_origin": "2023Q4"},
        ["artifacts/eda_prelock_quarterly.parquet", "data/manifests/eda_prelock.json"],
        lambda: build(root), resume=True)


if __name__ == "__main__":
    main()
