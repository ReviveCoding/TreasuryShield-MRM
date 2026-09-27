"""Combine official raw-field provenance with all declared model-derived features."""

from __future__ import annotations

import csv
from pathlib import Path

from treasuryshield.features.build import CORE_NUMERIC, ENHANCED_NUMERIC
from treasuryshield.peers.context import PEER_VALUES

EXPLICIT = {
    "log_assets": ("Natural log of positive consolidated assets", "F1 Scale", "log(assets)", "log USD thousands"),
    "log_deposits": ("Natural log of positive consolidated deposits", "F1 Scale", "log(deposits)", "log USD thousands"),
    "cash_asset_ratio": ("Cash divided by positive assets", "F2 Liquidity", "cash/assets", "ratio"),
    "liquid_broad_asset_ratio": ("Cash plus securities divided by positive assets", "F2 Liquidity", "(cash+securities)/assets", "ratio"),
    "loan_deposit_ratio": ("Total loans divided by positive deposits", "F6 Asset quality", "loans/deposits", "ratio"),
    "capital_asset_ratio": ("Equity divided by positive assets", "F5 Capital", "equity/assets", "ratio"),
    "borrowings_asset_ratio": ("Federal funds/repo and other borrowings divided by assets", "F4 Funding", "borrowings/assets", "ratio"),
    "npl_loan_ratio": ("Nonperforming loans divided by positive total loans", "F6 Asset quality", "npl_tot/ln_tot", "ratio"),
    "brokered_domestic_ratio": ("Brokered deposits divided by domestic deposits", "F3 Deposit composition", "brokered_dep/domestic_dep", "ratio"),
    "uninsured_proxy_domestic_share": ("One minus insured deposits divided by domestic deposits; historical proxy", "F3 Deposit composition", "1-insured_deposits/domestic_dep", "ratio"),
    "uninsured_time_deposit_ratio": ("Large/uninsured time-deposit field divided by consolidated deposits", "F3 Deposit composition", "uninsured_time_dep/deposits", "ratio"),
    "profit_asset_ratio": ("Single-quarter net operating earnings divided by assets", "F7 Profitability", "net_operating_earn/assets", "ratio"),
    "loan_loss_reserve_ratio": ("Loan-loss allowance divided by loans", "F6 Asset quality", "llres/ln_tot", "ratio"),
    "residential_loan_share": ("Residential real-estate loans divided by total loans", "F6 Asset quality", "ln_rre/ln_tot", "ratio"),
    "commercial_real_estate_loan_share": ("Commercial real-estate loans divided by total loans", "F6 Asset quality", "ln_cre/ln_tot", "ratio"),
    "ci_loan_share": ("Commercial and industrial loans divided by total loans", "F6 Asset quality", "ln_ci/ln_tot", "ratio"),
    "deposit_cost_annualized": ("Annualized quarterly interest expense on deposits divided by average deposits", "F8 Interest-rate sensitivity", "4*int_exp_dep/mean(deposits[t],deposits[t-1])", "ratio"),
    "deposit_growth_qoq_lagged": ("Observed prior-quarter deposit log growth at forecast origin", "F9 Dynamics", "log(deposits[t]/deposits[t-1])", "log change"),
    "deposit_growth_yoy_lagged": ("Observed four-quarter deposit log growth at forecast origin", "F9 Dynamics", "log(deposits[t]/deposits[t-4])", "log change"),
    "asset_growth_qoq_lagged": ("Observed prior-quarter asset log growth", "F9 Dynamics", "log(assets[t]/assets[t-1])", "log change"),
    "cash_ratio_change_qoq_lagged": ("Observed change in cash-to-assets over one quarter", "F9 Dynamics", "cash_asset_ratio[t]-cash_asset_ratio[t-1]", "ratio change"),
    "borrowings_change_asset_lagged": ("Observed quarter borrowing change scaled by prior assets", "F9 Dynamics", "(borrowings[t]-borrowings[t-1])/assets[t-1]", "ratio"),
    "deposit_growth_rolling_vol_4q": ("Four contiguous observed quarterly deposit-growth standard deviation", "F9 Dynamics", "std(last 4 observed QoQ log changes)", "log change"),
    "deposit_growth_acceleration": ("Change in observed quarterly deposit growth", "F9 Dynamics", "growth_qoq[t]-growth_qoq[t-1]", "log change"),
    "deposit_beta_4q": ("Four-quarter deposit-cost change over policy-rate change where rate move is at least 25bp", "F8 Interest-rate sensitivity", "delta(deposit_cost,4)/(delta(DFF,4)/100)", "ratio"),
    "uninsured_domestic_share": ("Direct FFIEC domestic uninsured-deposit estimate divided by domestic deposits", "F3 Deposit composition", "RCON5597/RCON2200", "ratio"),
    "brokered_domestic_share": ("Direct FFIEC brokered deposits divided by domestic deposits", "F3 Deposit composition", "RCON2365/RCON2200", "ratio"),
    "transaction_domestic_share": ("FFIEC transaction deposits divided by domestic deposits", "F3 Deposit composition", "RCON2215/RCON2200", "ratio"),
    "nontransaction_domestic_share": ("FFIEC nontransaction deposits divided by domestic deposits", "F3 Deposit composition", "RCON2385/RCON2200", "ratio"),
    "large_time_domestic_share": ("FFIEC time deposits greater than $250K divided by domestic deposits", "F3 Deposit composition", "RCONJ474/RCON2200", "ratio"),
    "reciprocal_brokered_share": ("FFIEC reciprocal brokered deposits divided by domestic deposits", "F3 Deposit composition", "RCONG803/RCON2200", "ratio"),
    "other_borrowed_money_domestic_asset_ratio": ("FFIEC domestic other borrowed money divided by Core assets", "F4 Funding", "RCON3190/assets", "ratio"),
}


def _derived_row(name: str, fields: list[str]) -> dict[str, str]:
    if name in EXPLICIT:
        definition, family, transformation, unit = EXPLICIT[name]
        source = "New York Fed harmonized Call Reports; FFIEC if mnemonic RCON; FRED for policy-rate denominator"
    elif name.startswith("macro_"):
        definition = "Forecast-origin macro/system observation or transformation: " + name
        family, transformation, unit = "F10 Macro/system/peer", "as-of origin with source-specific lag", "source series unit"
        source = "FRED / ALFRED / Federal Reserve H.8"
    elif name.startswith(("peer_", "own_minus_peer_")):
        definition = "Contemporaneous own-bank and peer cross-section statistic: " + name
        family, transformation, unit = "F10 Macro/system/peer", "quarter peer group median, difference, or percentile", "source ratio or percentile"
        source = "Derived from all available bank reports at the same origin"
    elif name.startswith("regime_"):
        definition = "Filtered regime probability or state: " + name
        family, transformation, unit = "F10 Macro/system/peer", "training-fold fitted forward HMM or fixed rule", "probability or state"
        source = "FRED / ALFRED macro vector"
    elif name == "quarter":
        definition, family, transformation, unit, source = (
            "Calendar quarter of report", "F9 Dynamics", "decode Stata quarter", "category", "New York Fed date")
    else:
        raise ValueError(f"Derived feature missing documented definition: {name}")
    return {
        "canonical_name": name, "economic_definition": definition, "source": source,
        "raw_fields_mnemonics": "See source columns/formulas in transformation and raw-field rows",
        "report_form": "Core harmonized Call Reports; 031/041/051 for Enhanced fields",
        "valid_from": "2017Q1 for Enhanced direct fields; source-dependent otherwise",
        "valid_to": "2026Q1 Core model source snapshot",
        "stock_or_flow": "derived", "unit": unit,
        "consolidation_scope": "Domestic denominator for RCON direct deposit shares; consolidated for Core asset/deposit ratios",
        "publication_lag": "45-day bank-report proxy; source-specific macro lag in macro registry",
        "missing_semantics": "Missing remains missing until training-partition-only imputation plus indicator",
        "transformation": transformation,
        "winsorization_policy": "Training-fold-only 0.5/99.5 percentile clipping for linear/neural models; unmodified for trees",
        "allowed_use": "predictive feature", "leakage_risk": "History and peer inputs must be available at origin; HMM is forward filtered",
        "feature_family": family,
        "notes": "Exact historical bank amendment vintage unavailable; do not infer causality",
    }


def build(root: Path) -> None:
    source_path = root / "data" / "manifests" / "feature_registry.csv"
    with source_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    identifiers = {"id_rssd": "Bank RSSD identifier", "date": "Stata quarterly index (1960Q1=0)",
                   "entity_type": "New York Fed institutional entity class"}
    for row in rows:
        if not row["economic_definition"] and row["canonical_name"] in identifiers:
            row["economic_definition"] = identifiers[row["canonical_name"]]
    names = set(CORE_NUMERIC + ENHANCED_NUMERIC + ["quarter"])
    for value in PEER_VALUES:
        names.update([f"peer_{value}_median", f"own_minus_peer_{value}", f"peer_{value}_percentile"])
    names.update(["peer_group_size", "regime_p0", "regime_p1", "regime_p2", "regime_p3", "regime_hard"])
    existing = {row["canonical_name"] for row in rows}
    rows += [_derived_row(name, fields) for name in sorted(names - existing)]
    output = root / "data" / "manifests" / "feature_registry_complete.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f"Complete feature registry: {len(rows)} source and derived entries")


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[3])
