"""Generate auditable feature and target registries from the official dictionary."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import openpyxl

from treasuryshield.preprocessing.core import BALANCE_COLUMNS, INCOME_COLUMNS

IDENTIFIERS = {"id_rssd", "date", "id_fdic_cert", "entity_type", "state_abbr_nm", "reason_term_cd"}
FAMILIES = {
    "assets": "F1 Scale", "deposits": "F1 Scale", "cash": "F2 Liquidity",
    "securities": "F2 Liquidity", "ffrepo_ass": "F2 Liquidity",
    "demand_deposits": "F3 Deposit composition", "time_deposits": "F3 Deposit composition",
    "brokered_dep": "F3 Deposit composition", "insured_deposits": "F3 Deposit composition",
    "domestic_dep": "F3 Deposit composition", "foreign_dep": "F3 Deposit composition",
    "dom_deposit_ib": "F3 Deposit composition", "dom_deposit_nib": "F3 Deposit composition",
    "nontransaction_dep": "F3 Deposit composition", "transaction_dep": "F3 Deposit composition",
    "uninsured_time_dep": "F3 Deposit composition", "time_dep_gt250k": "F3 Deposit composition",
    "ffrepo_liab": "F4 Funding", "othbor_liab": "F4 Funding",
    "liab_tot": "F4 Funding", "equity": "F5 Capital",
    "llres": "F6 Asset quality", "npl_tot": "F6 Asset quality",
}
FFIEC = [
    ("uninsured_domestic_deposits", "Estimated uninsured deposits in domestic offices", "RCON5597", "RCO", "F3 Deposit composition", "domestic"),
    ("uninsured_consolidated_deposits", "Consolidated estimated uninsured deposits where reported", "RCONL197", "RCO", "F3 Deposit composition", "consolidated; form-dependent"),
    ("brokered_domestic_deposits", "Total brokered deposits in domestic offices", "RCON2365", "RCE", "F3 Deposit composition", "domestic"),
    ("transaction_domestic_deposits", "Total transaction accounts in domestic offices", "RCON2215", "RCE", "F3 Deposit composition", "domestic"),
    ("nontransaction_domestic_deposits", "Total nontransaction accounts in domestic offices", "RCON2385", "RCE", "F3 Deposit composition", "domestic"),
    ("large_time_domestic_deposits", "Time deposits greater than $250K in domestic offices", "RCONJ474", "RCE", "F3 Deposit composition", "domestic"),
    ("reciprocal_brokered_deposits", "Reciprocal brokered deposits", "RCONG803", "RCO", "F3 Deposit composition", "domestic"),
    ("other_borrowed_money_domestic", "Other borrowed money in domestic offices", "RCON3190", "RC", "F4 Funding", "domestic"),
]


def _dictionary_maps(path: Path) -> dict[str, list[dict[str, str]]]:
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    mapping: dict[str, list[dict[str, str]]] = defaultdict(list)
    for sheet_name in ("balance sheet", "income statement"):
        current = ""
        sheet = book[sheet_name]
        for values in sheet.iter_rows(min_row=2, values_only=True):
            if values[0] is not None and str(values[0]).strip():
                current = str(values[0]).strip()
            if not current or values[2] is None or values[3] is None:
                continue
            mapping[current].append({
                "mnemonic": str(values[2]).strip(),
                "field": str(values[3]).strip(),
                "valid_period": str(values[4] or "").strip(),
                "notes": str(values[5] or "").strip(),
            })
    return mapping


def build(root: Path) -> None:
    dictionary = root / "data" / "raw" / "newyorkfed" / "historical_call_data_dictionary.xlsx"
    mappings = _dictionary_maps(dictionary)
    schema = json.loads((root / "data" / "manifests" / "ny_schema.json").read_text(encoding="utf-8"))
    labels = {}
    for file_name, details in schema.items():
        if not file_name.endswith(".dta"):
            continue
        labels.update({row["name"]: row.get("label", "") for row in details["columns"]})
    fields = [
        "canonical_name", "economic_definition", "source", "raw_fields_mnemonics",
        "report_form", "valid_from", "valid_to", "stock_or_flow", "unit",
        "consolidation_scope", "publication_lag", "missing_semantics",
        "transformation", "winsorization_policy", "allowed_use", "leakage_risk",
        "feature_family", "notes",
    ]
    rows = []
    for name in BALANCE_COLUMNS + [n for n in INCOME_COLUMNS if n not in {"id_rssd", "date"}]:
        kind = "identifier" if name in IDENTIFIERS else "flow_ytd" if name.startswith("ytd") else "stock"
        source_map = mappings.get(name, [])
        rows.append({
            "canonical_name": name,
            "economic_definition": labels.get(name, name),
            "source": "New York Fed harmonized Call Reports",
            "raw_fields_mnemonics": json.dumps(source_map, separators=(",", ":")),
            "report_form": "Historical FFIEC/FDIC forms; see source dictionary",
            "valid_from": "source-specific; see raw_fields_mnemonics",
            "valid_to": "source-specific; see raw_fields_mnemonics",
            "stock_or_flow": kind,
            "unit": "USD thousands" if kind != "identifier" else "identifier/date/category",
            "consolidation_scope": "source-dependent, generally harmonized consolidated RCFD; inspect dictionary",
            "publication_lag": "45 calendar days assumed for pseudo-point-in-time mode",
            "missing_semantics": "Never convert missing to zero; historical form nonapplicability requires field-period review",
            "transformation": "YTD to single-quarter within bank-year" if kind == "flow_ytd" else "none",
            "winsorization_policy": "Training-fold-only, if selected; raw values retained",
            "allowed_use": "identifier" if kind == "identifier" else "target and feature" if name == "deposits" else "feature",
            "leakage_risk": "Use only after assumed public availability date; no future values",
            "feature_family": FAMILIES.get(name, "F6 Asset quality" if name.startswith("ln_") else "F7 Profitability" if name.startswith("ytd") else "F2 Liquidity"),
            "notes": "Exact original amendment vintage is not reconstructed.",
        })
    for name, definition, mnemonic, schedule, family, scope in FFIEC:
        rows.append({
            "canonical_name": name, "economic_definition": definition,
            "source": "FFIEC CDR public quarterly bulk",
            "raw_fields_mnemonics": mnemonic, "report_form": "031/041/051 where reported; schedule " + schedule,
            "valid_from": "2017Q1 acquisition window; actual availability checked per file",
            "valid_to": "2026Q2 acquisition window; actual availability checked per file",
            "stock_or_flow": "stock", "unit": "USD thousands",
            "consolidation_scope": scope,
            "publication_lag": "45 calendar days assumed; last-submission timestamp retained separately",
            "missing_semantics": "Absent raw mnemonic is structural; blank value for present field is reportable missing",
            "transformation": "level and ratio with matching domestic/consolidated denominator",
            "winsorization_policy": "Training-fold-only, if selected; raw values retained",
            "allowed_use": "Enhanced Panel feature", "leakage_risk": "Use only after assumed public availability date",
            "feature_family": family, "notes": "Do not mix RCON domestic and RCFD consolidated concepts without scope label",
        })
    out = root / "data" / "manifests" / "feature_registry.csv"
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    target_rows = [
        ["T1", f"deposit_log_growth_{h}q", f"log(deposits[t+{h}]/deposits[t])", h,
         "positive deposits at both dates; exact quarter continuity; M&A window excluded", "distributional" ]
        for h in (1, 2, 4)
    ] + [
        ["T2", "borrowing_change_asset_scaled", "(borrowings[t+h]-borrowings[t])/assets[t]", "1/2/4", "positive assets; same bank", "regression"],
        ["T3", "liquid_narrow_change", "cash/assets change", "1/2/4", "positive assets; same bank", "regression"],
        ["T3", "liquid_broad_change", "(cash+securities)/assets change", "1/2/4", "positive assets; same bank", "regression"],
        ["T4", "liquidity_stress_event", "training-period-only peer-relative union of tail deposit runoff, funding increase, and buffer deterioration", "1/2/4", "threshold frozen inside training fold", "classification"],
        ["T5", "fdic_failure_external", "FDIC failure match by certificate and date", "secondary", "never infer failure from disappearance", "external_validity"],
    ]
    with (root / "data" / "manifests" / "target_registry.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["target_id", "canonical_name", "definition", "horizon_quarters", "eligibility", "task"])
        writer.writerows(target_rows)
    print(f"Registered {len(rows)} features and {len(target_rows)} targets")


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[3])
