"""Construct historically available own-bank, dynamic, and macro features."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from treasuryshield.orchestration.state import RunState, sha256_file

CORE_NUMERIC = [
    "log_assets", "log_deposits", "cash_asset_ratio", "liquid_broad_asset_ratio",
    "loan_deposit_ratio", "capital_asset_ratio", "borrowings_asset_ratio",
    "npl_loan_ratio", "brokered_domestic_ratio", "uninsured_proxy_domestic_share",
    "uninsured_time_deposit_ratio", "profit_asset_ratio", "loan_loss_reserve_ratio",
    "residential_loan_share", "commercial_real_estate_loan_share", "ci_loan_share",
    "deposit_cost_annualized", "deposit_growth_qoq_lagged", "deposit_growth_yoy_lagged",
    "asset_growth_qoq_lagged", "cash_ratio_change_qoq_lagged",
    "borrowings_change_asset_lagged", "deposit_growth_rolling_vol_4q",
    "deposit_growth_acceleration", "deposit_beta_4q",
    "macro_DFF_revised", "macro_DFF_change_4q_revised", "macro_DGS2_revised",
    "macro_DGS10_revised", "macro_curve_10y2y_revised", "macro_VIXCLS_revised",
    "macro_DPSACBW027NBOG_qoq_revised", "macro_UNRATE_revised", "macro_CFNAI_revised",
    "macro_CPIAUCSL_revised", "quarter",
]
ENHANCED_NUMERIC = CORE_NUMERIC + [
    "uninsured_domestic_share", "brokered_domestic_share", "transaction_domestic_share",
    "nontransaction_domestic_share", "large_time_domestic_share",
    "reciprocal_brokered_share", "other_borrowed_money_domestic_asset_ratio",
]


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator > 0)


def _lag(frame: pd.DataFrame, value: str, quarters: int) -> pd.Series:
    group = frame.groupby("bank_id", sort=False)
    previous = group[value].shift(quarters)
    previous_quarter = group["stata_quarter"].shift(quarters)
    return previous.where((frame["stata_quarter"] - previous_quarter).eq(quarters))


def _growth(current: pd.Series, previous: pd.Series) -> pd.Series:
    return np.log(current.where(current > 0) / previous.where(previous > 0))


def build(root: Path) -> None:
    core = pd.read_parquet(root / "data" / "processed" / "core_panel.parquet")
    core = core.sort_values(["bank_id", "stata_quarter"]).reset_index(drop=True)
    macro = pd.read_parquet(root / "data" / "processed" / "macro_origins.parquet")
    macro_columns = [c for c in macro if c.startswith("macro_")]
    frame = core.merge(macro[["report_period"] + macro_columns], on="report_period",
                       how="left", validate="many_to_one")
    if not frame["feature_information_date"].le(frame["forecast_origin"]).all():
        raise ValueError("Call Report feature date exceeds forecast origin")
    for col in [c for c in macro_columns if c.endswith("assumed_availability_date") or c.endswith("vintage_date")]:
        if frame[col].gt(frame["forecast_origin"]).any():
            raise ValueError(f"Macro availability leaks into future: {col}")
    frame["log_assets"] = np.log(frame["assets"].where(frame["assets"] > 0))
    frame["log_deposits"] = np.log(frame["deposits"].where(frame["deposits"] > 0))
    frame["uninsured_proxy_domestic_share"] = 1 - _ratio(frame["insured_deposits"], frame["domestic_dep"])
    frame["loan_loss_reserve_ratio"] = _ratio(frame["llres"], frame["ln_tot"])
    frame["residential_loan_share"] = _ratio(frame["ln_rre"], frame["ln_tot"])
    frame["commercial_real_estate_loan_share"] = _ratio(frame["ln_cre"], frame["ln_tot"])
    frame["ci_loan_share"] = _ratio(frame["ln_ci"], frame["ln_tot"])
    dep_prev = _lag(frame, "deposits", 1)
    dep_prev4 = _lag(frame, "deposits", 4)
    asset_prev = _lag(frame, "assets", 1)
    frame["deposit_growth_qoq_lagged"] = _growth(frame["deposits"], dep_prev)
    frame["deposit_growth_yoy_lagged"] = _growth(frame["deposits"], dep_prev4)
    frame["asset_growth_qoq_lagged"] = _growth(frame["assets"], asset_prev)
    frame["cash_ratio_change_qoq_lagged"] = frame["cash_asset_ratio"] - _lag(frame, "cash_asset_ratio", 1)
    frame["borrowings_change_asset_lagged"] = _ratio(frame["borrowings"] - _lag(frame, "borrowings", 1), asset_prev)
    avg_dep = (frame["deposits"] + dep_prev) / 2
    frame["deposit_cost_annualized"] = 4 * _ratio(frame["int_exp_dep"], avg_dep)
    group = frame.groupby("bank_id", sort=False)
    vol = group["deposit_growth_qoq_lagged"].rolling(4, min_periods=4).std().reset_index(level=0, drop=True)
    contiguous = (frame["stata_quarter"] - group["stata_quarter"].shift(4)).eq(4)
    frame["deposit_growth_rolling_vol_4q"] = vol.where(contiguous)
    frame["deposit_growth_acceleration"] = frame["deposit_growth_qoq_lagged"] - _lag(frame, "deposit_growth_qoq_lagged", 1)
    cost_delta = frame["deposit_cost_annualized"] - _lag(frame, "deposit_cost_annualized", 4)
    rate_delta = frame["macro_DFF_revised"] - frame.groupby("bank_id", sort=False)["macro_DFF_revised"].shift(4)
    rate_delta = rate_delta.where(contiguous & rate_delta.abs().ge(0.25))
    frame["deposit_beta_4q"] = cost_delta / (rate_delta / 100)
    frame["season_q1"] = frame["quarter"].eq(1).astype("int8")
    frame["season_q2"] = frame["quarter"].eq(2).astype("int8")
    frame["season_q3"] = frame["quarter"].eq(3).astype("int8")
    frame["season_q4"] = frame["quarter"].eq(4).astype("int8")
    frame["core_eligible"] = frame["assets"].gt(0) & frame["deposits"].gt(0)
    out = root / "data" / "features" / "core_features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = ["bank_id", "fdic_cert", "stata_quarter", "report_period", "quarter_end", "year", "quarter",
            "public_availability_date", "feature_information_date", "forecast_origin", "source_vintage",
            "panel_mode", "entity_type", "state", "assets", "deposits", "cash", "securities", "borrowings",
            "equity", "domestic_dep", "insured_deposits", "core_eligible", "liquid_assets_narrow",
            "liquid_assets_broad", "liquid_broad_asset_ratio", "cash_asset_ratio"]
    keep = list(dict.fromkeys(keep + CORE_NUMERIC + [c for c in frame if c.startswith("macro_")]))
    frame[keep].to_parquet(out, index=False, compression="zstd")
    enhanced = pd.read_parquet(root / "data" / "processed" / "enhanced_panel.parquet")
    enhanced = enhanced.drop(columns=["quarter_end", "public_availability_date", "panel_mode"], errors="ignore")
    merged = frame[keep].merge(enhanced, on=["bank_id", "report_period"], how="inner", validate="one_to_one")
    merged["reciprocal_brokered_share"] = _ratio(merged["reciprocal_brokered_deposits"], merged["domestic_deposits_ffiec"])
    merged["other_borrowed_money_domestic_asset_ratio"] = _ratio(merged["other_borrowed_money_domestic"], merged["assets"])
    merged["enhanced_eligible"] = merged["core_eligible"] & merged["domestic_deposits_ffiec"].gt(0)
    merged.to_parquet(root / "data" / "features" / "enhanced_features.parquet", index=False, compression="zstd")
    prelock = frame.loc[frame["year"].le(2023)]
    quality = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "core_rows": len(frame), "enhanced_join_rows": len(merged),
        "core_eligible_rows": int(frame["core_eligible"].sum()),
        "enhanced_eligible_rows": int(merged["enhanced_eligible"].sum()),
        "core_features": CORE_NUMERIC, "enhanced_features": ENHANCED_NUMERIC,
        "prelock_feature_coverage": {c: float(prelock[c].notna().mean()) for c in CORE_NUMERIC},
        "insured_deposit_definition": "NY Fed RCON2702 then RCONF049+RCONF045; proxy is 1 - insured/domestic and is distinct from FFIEC estimated uninsured",
        "point_in_time_limit": "Bank reports are restated retrieval snapshots, shifted 45 days; exact historical amendments unavailable",
    }
    (root / "data" / "manifests" / "feature_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(f"Core features {len(frame)} rows; Enhanced matched {len(merged)} rows", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    inputs = [root / "data" / "processed" / f"{x}.parquet" for x in ["core_panel", "enhanced_panel", "macro_origins"]]
    RunState(root).run(
        "features", "BUILDING_DATA",
        {"input_sha256": {p.name: sha256_file(p) for p in inputs}, "core": CORE_NUMERIC,
         "enhanced": ENHANCED_NUMERIC},
        ["data/features/core_features.parquet", "data/features/enhanced_features.parquet",
         "data/manifests/feature_quality.json"], lambda: build(root), resume=True)


if __name__ == "__main__":
    main()
