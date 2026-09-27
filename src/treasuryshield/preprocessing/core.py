"""Create a bank-level, restated Core Panel from harmonized official Call Reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

from treasuryshield.orchestration.state import RunState, sha256_file

BALANCE_COLUMNS = [
    "id_rssd", "date", "id_fdic_cert", "entity_type", "state_abbr_nm",
    "reason_term_cd", "assets", "cash", "securities", "ln_tot", "ln_tot_gross",
    "llres", "ffrepo_ass", "ffrepo_liab", "othbor_liab", "liab_tot", "equity",
    "deposits", "demand_deposits", "time_deposits", "ln_re", "ln_rre",
    "ln_cre", "ln_ci", "ln_cons", "ln_cc", "npl_tot", "brokered_dep",
    "insured_deposits", "domestic_dep", "foreign_dep", "dom_deposit_ib",
    "dom_deposit_nib", "nontransaction_dep", "transaction_dep", "htmsec_ac",
    "afssec_fv", "uninsured_time_dep", "time_dep_gt250k",
    "securities_mat_1y_less", "securities_3mo_less", "ust_sec_3mo_less",
]
INCOME_COLUMNS = [
    "id_rssd", "date", "ytdint_exp_dep", "ytdint_exp_ffrepo",
    "ytdint_exp_othbor", "ytdnet_operating_earn", "ytdllprov",
    "ytdint_inc_sec", "ytdint_inc_ln", "ytdoperating_inc_tot",
    "ytdoperating_exp_tot",
]
COMMERCIAL_ENTITY_TYPES = {"NAT", "SMB", "NMB"}
SOURCE_VINTAGE = "NYFED_JUN2026_RESTATED"
AVAILABILITY_LAG_DAYS = 45


def quarterize_ytd(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Subtract only the immediately preceding quarter in the same bank-year."""
    out = frame.sort_values(["bank_id", "stata_quarter"]).copy()
    groups = out.groupby(["bank_id", "year"], sort=False)
    prior_quarter = groups["quarter"].shift(1)
    for column in columns:
        prior = groups[column].shift(1)
        quarterly = np.where(
            out["quarter"].eq(1), out[column],
            np.where(prior_quarter.eq(out["quarter"] - 1), out[column] - prior, np.nan),
        )
        out[column.removeprefix("ytd")] = quarterly
    return out


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.where(denominator > 0) / denominator.where(denominator > 0)


def build(root: Path) -> None:
    base = root / "data" / "interim" / "newyorkfed"
    bs_path = base / "call-reports-balance-sheets-Jun2026.dta"
    inc_path = base / "call-reports-income-statements-Jun2026.dta"
    out_path = root / "data" / "processed" / "core_panel.parquet"
    dq_path = root / "data" / "manifests" / "core_quality.json"
    coverage_path = root / "data" / "manifests" / "core_coverage.csv"
    print("Reading selected New York Fed balance-sheet columns", flush=True)
    bs, _ = pyreadstat.read_dta(str(bs_path), usecols=BALANCE_COLUMNS)
    print("Reading selected New York Fed income columns", flush=True)
    inc, _ = pyreadstat.read_dta(str(inc_path), usecols=INCOME_COLUMNS)
    if bs.duplicated(["id_rssd", "date"]).any() or inc.duplicated(["id_rssd", "date"]).any():
        raise ValueError("Duplicate bank-quarter keys in source data")
    source_rows = len(bs)
    bs = bs.loc[bs["date"].ge(96) & bs["entity_type"].isin(COMMERCIAL_ENTITY_TYPES)].copy()
    inc = inc.loc[inc["date"].ge(96)].copy()
    print(f"Commercial 1984+ rows before accounting filters: {len(bs)}", flush=True)
    merged = bs.merge(inc, on=["id_rssd", "date"], how="left", validate="one_to_one")
    del bs, inc
    merged = merged.rename(columns={"id_rssd": "bank_id", "date": "stata_quarter",
                                    "id_fdic_cert": "fdic_cert", "state_abbr_nm": "state"})
    code = merged["stata_quarter"].astype("int64")
    merged["year"] = (1960 + code // 4).astype("int16")
    merged["quarter"] = (code % 4 + 1).astype("int8")
    period = pd.PeriodIndex.from_fields(year=merged["year"].to_numpy(),
                                        quarter=merged["quarter"].to_numpy(), freq="Q")
    merged["report_period"] = period.astype(str)
    merged["quarter_end"] = period.to_timestamp(how="end").normalize()
    merged["public_availability_date"] = merged["quarter_end"] + pd.Timedelta(days=AVAILABILITY_LAG_DAYS)
    merged["forecast_origin"] = merged["public_availability_date"]
    merged["feature_information_date"] = merged["public_availability_date"]
    merged["source_vintage"] = SOURCE_VINTAGE
    merged["retrieval_date"] = pd.Timestamp("2026-09-27")
    merged["panel_mode"] = "A_RESTATED"
    ytd_cols = [c for c in INCOME_COLUMNS if c.startswith("ytd")]
    merged = quarterize_ytd(merged, ytd_cols)
    merged["borrowings"] = merged[["ffrepo_liab", "othbor_liab"]].sum(axis=1, min_count=1)
    merged["liquid_assets_narrow"] = merged["cash"]
    merged["liquid_assets_broad"] = merged[["cash", "securities"]].sum(axis=1, min_count=2)
    merged["cash_asset_ratio"] = _safe_ratio(merged["cash"], merged["assets"])
    merged["liquid_broad_asset_ratio"] = _safe_ratio(merged["liquid_assets_broad"], merged["assets"])
    merged["loan_deposit_ratio"] = _safe_ratio(merged["ln_tot"], merged["deposits"])
    merged["capital_asset_ratio"] = _safe_ratio(merged["equity"], merged["assets"])
    merged["borrowings_asset_ratio"] = _safe_ratio(merged["borrowings"], merged["assets"])
    merged["npl_loan_ratio"] = _safe_ratio(merged["npl_tot"], merged["ln_tot"])
    merged["brokered_domestic_ratio"] = _safe_ratio(merged["brokered_dep"], merged["domestic_dep"])
    merged["uninsured_time_deposit_ratio"] = _safe_ratio(merged["uninsured_time_dep"], merged["deposits"])
    merged["profit_asset_ratio"] = _safe_ratio(merged["net_operating_earn"], merged["assets"])
    merged["asset_identity_gap"] = merged["assets"] - merged["liab_tot"] - merged["equity"]
    merged["asset_identity_rel_gap"] = merged["asset_identity_gap"].abs() / merged["assets"].where(merged["assets"] > 0)
    good = merged["assets"].gt(0) & merged["deposits"].gt(0)
    quality = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_rows": source_rows,
        "commercial_1984_plus_rows": len(merged),
        "supervised_eligible_rows": int(good.sum()),
        "nonpositive_assets": int(merged["assets"].le(0).sum()),
        "nonpositive_deposits": int(merged["deposits"].le(0).sum()),
        "duplicate_bank_quarter": int(merged.duplicated(["bank_id", "stata_quarter"]).sum()),
        "asset_identity_gap_gt_1pct": int(merged["asset_identity_rel_gap"].gt(0.01).sum()),
        "asset_identity_gap_nonmissing": int(merged["asset_identity_rel_gap"].notna().sum()),
        "income_join_coverage": float(merged["ytdint_exp_dep"].notna().mean()),
        "min_quarter": str(period.min()), "max_quarter": str(period.max()),
        "availability_lag_days_assumed": AVAILABILITY_LAG_DAYS,
        "source_vintage": SOURCE_VINTAGE,
        "commercial_entity_types": sorted(COMMERCIAL_ENTITY_TYPES),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False, compression="zstd")
    pd.DataFrame({"report_period": merged["report_period"], "bank_id": merged["bank_id"],
                  "assets": merged["assets"], "deposits": merged["deposits"]}).groupby("report_period").agg(
        institutions=("bank_id", "nunique"), rows=("bank_id", "size"),
        assets_thousands=("assets", "sum"), deposits_thousands=("deposits", "sum")
    ).reset_index().to_csv(coverage_path, index=False)
    dq_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(json.dumps(quality, indent=2), flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    bs_path = root / "data" / "interim" / "newyorkfed" / "call-reports-balance-sheets-Jun2026.dta"
    inc_path = root / "data" / "interim" / "newyorkfed" / "call-reports-income-statements-Jun2026.dta"
    RunState(root).run(
        "core_panel", "BUILDING_DATA",
        {"balance_sha256": sha256_file(bs_path), "income_sha256": sha256_file(inc_path),
         "balance_columns": BALANCE_COLUMNS, "income_columns": INCOME_COLUMNS,
         "commercial_entity_types": sorted(COMMERCIAL_ENTITY_TYPES),
         "availability_lag_days": AVAILABILITY_LAG_DAYS},
        ["data/processed/core_panel.parquet", "data/manifests/core_quality.json",
         "data/manifests/core_coverage.csv"],
        lambda: build(root), resume=True,
    )


if __name__ == "__main__":
    main()
