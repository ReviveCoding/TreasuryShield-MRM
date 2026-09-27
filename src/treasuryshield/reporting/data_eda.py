"""Source-linked data quality and pre-lock EDA reports from stored artifacts."""

from __future__ import annotations

import html
import json
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from treasuryshield.orchestration.state import RunState, sha256_file

CSS = """
body{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:35px auto;padding:0 20px;color:#172536;background:#fff}
h1,h2,h3{color:#0b3552}h1{border-bottom:3px solid #166f9c;padding-bottom:10px}
table{border-collapse:collapse;width:100%;margin:14px 0 28px}th,td{border:1px solid #ccd5dc;padding:8px;text-align:left;vertical-align:top}
th{background:#e8f1f6}tr:nth-child(even){background:#f8fbfd}.note{background:#f2f7fa;border-left:4px solid #1875a5;padding:12px}
.caveat{background:#fff4e6;border-left:4px solid #ce8020;padding:12px}img{max-width:100%;height:auto}
small{color:#4b5b68}code{background:#f1f5f8;padding:2px 4px}a{color:#116494}
"""


def _page(title: str, body: str) -> str:
    return f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>{body}</body></html>"


def _table(headers: list[str], rows: list[list[object]]) -> str:
    th = "".join(f"<th>{html.escape(str(v))}</th>" for v in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def _plot(root: Path, coverage: pd.DataFrame, targets: pd.DataFrame,
          macro: pd.DataFrame) -> None:
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    years = coverage["report_period"].str[:4].astype(int) + (coverage["report_period"].str[-1].astype(int) - 1) / 4
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10})
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(years, coverage["institutions"], color="#116494", linewidth=2)
    ax.set(title="Commercial-bank reporter coverage", xlabel="Report year", ylabel="RSSD institutions")
    ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figures / "01_data_coverage_timeline.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(years, coverage["institutions"], color="#116494", linewidth=2)
    for start, end, name in [(2007.5, 2009.99, "GFC"), (2020, 2021.99, "Pandemic"),
                             (2022, 2023.99, "Tightening")]:
        ax.axvspan(start, end, alpha=.1, color="#ba563b")
        ax.text((start + end) / 2, ax.get_ylim()[1] * .95, name, ha="center", va="top", fontsize=8)
    ax.set(title="Institution count and pre-2024 stress windows", xlabel="Report year", ylabel="Institutions")
    ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figures / "02_institution_count_timeline.png"); plt.close(fig)
    clean = targets.loc[targets["report_period"].le("2023Q4") & targets["horizon_quarters"].eq(1)
                        & targets["mna_contaminated_radius_1"].eq(False), "deposit_log_growth"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(clean.clip(-.4, .4), bins=100, density=True, color="#257a9e", alpha=.85)
    ax.axvline(clean.quantile(.05), color="#b53835", label="Pre-lock Q05")
    ax.set(title="One-quarter deposit log growth, pre-2024 eligible origins",
           xlabel="Log growth (display clipped to ±0.4; tails retained in metrics)", ylabel="Density")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "03_deposit_growth_distribution.png"); plt.close(fig)
    m = macro.loc[macro["report_period"].between("2001Q1", "2023Q4")]
    q = coverage.loc[coverage["report_period"].between("2001Q1", "2023Q4")]
    m = m.merge(q[["report_period", "deposits_thousands"]], on="report_period", how="inner")
    m["deposit_qoq"] = m["deposits_thousands"].pct_change(fill_method=None) * 100
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(m))
    ax.plot(x, m["deposit_qoq"], color="#116494", linewidth=1.3, label="Bank panel aggregate deposit QoQ (%)")
    ax.axhline(0, color="#485767", linewidth=.8)
    ax2 = ax.twinx()
    ax2.plot(x, m["macro_DFF_revised"], color="#c15b3e", alpha=.7, label="DFF (%)")
    ticks = np.arange(0, len(x), 16)
    ax.set_xticks(ticks, m["report_period"].iloc[ticks].str[:4], rotation=0)
    ax.set(title="Aggregate deposits and policy rate, descriptive only", xlabel="Report year", ylabel="Deposit QoQ (%)")
    ax2.set_ylabel("Effective federal funds rate (%)")
    ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figures / "04_macro_deposit_history.png"); plt.close(fig)


def build(root: Path) -> None:
    snapshot = json.loads((root / "data" / "manifests" / "snapshot.json").read_text(encoding="utf-8"))
    core = json.loads((root / "data" / "manifests" / "core_quality.json").read_text(encoding="utf-8"))
    enhanced = json.loads((root / "data" / "manifests" / "ffiec_quality.json").read_text(encoding="utf-8"))
    features = json.loads((root / "data" / "manifests" / "feature_quality.json").read_text(encoding="utf-8"))
    macro_quality = json.loads((root / "data" / "manifests" / "macro_quality.json").read_text(encoding="utf-8"))
    entities = json.loads((root / "data" / "manifests" / "entity_quality.json").read_text(encoding="utf-8"))
    y9c = json.loads((root / "data" / "manifests" / "y9c_quality.json").read_text(encoding="utf-8"))
    eda = json.loads((root / "data" / "manifests" / "eda_prelock.json").read_text(encoding="utf-8"))
    coverage = pd.read_parquet(root / "artifacts" / "eda_prelock_quarterly.parquet")
    targets = pd.read_parquet(root / "data" / "features" / "targets.parquet",
                              columns=["report_period", "horizon_quarters", "mna_contaminated_radius_1", "deposit_log_growth"])
    macro = pd.read_parquet(root / "data" / "processed" / "macro_origins.parquet",
                            columns=["report_period", "macro_DFF_revised"])
    _plot(root, coverage, targets, macro)
    by_form = pd.read_parquet(root / "data" / "processed" / "enhanced_panel.parquet",
                              columns=["ffiec_form", "uninsured_domestic_deposits"])
    by_form = by_form.groupby("ffiec_form").agg(rows=("ffiec_form", "size"),
                                                 uninsured_coverage=("uninsured_domestic_deposits", lambda s: s.notna().mean())).reset_index()
    con = duckdb.connect(str(root / "artifacts" / "results.duckdb"))
    con.execute("DELETE FROM data_quality WHERE scope = 'snapshot_or_detail'")
    facts = [
        ("raw_files", snapshot["raw_file_count"]),
        ("raw_compressed_bytes", snapshot["total_compressed_bytes"]),
        ("raw_uncompressed_bytes", snapshot["total_uncompressed_bytes"]),
        ("core_rows", core["commercial_1984_plus_rows"]),
        ("core_supervised_eligible", core["supervised_eligible_rows"]),
        ("enhanced_raw_rows", enhanced["rows"]),
        ("enhanced_matched_rows", features["enhanced_join_rows"]),
        ("y9c_bhc_rows", y9c["rows"]),
        ("fdic_history_events", entities["history_events"]),
        ("fdic_failure_rows", entities["failure_rows"]),
        ("alfred_vintage_snapshots", macro_quality["historical_vintage_snapshots"]),
    ]
    facts += [(f"ffiec_{row.ffiec_form}_uninsured_coverage", row.uninsured_coverage)
              for row in by_form.itertuples(index=False)]
    con.executemany("INSERT INTO data_quality VALUES ('snapshot_or_detail', ?, ?, '', 'reports/02_data_lineage_and_quality.html')",
                    [(name, float(value)) for name, value in facts])
    con.close()
    rows = [
        ["Raw source snapshot", snapshot["data_snapshot_id"], f"{snapshot['raw_file_count']:,} files; {snapshot['total_compressed_bytes']/1e9:.2f} GB retained", "data/manifests/raw_manifest.csv"],
        ["New York Fed source", "1959Q4–2026Q1", f"{core['source_rows']:,} source bank-quarters", "data/manifests/ny_coverage.json"],
        ["Commercial-bank Core Panel", "1984Q1–2026Q1", f"{core['commercial_1984_plus_rows']:,} rows; {core['supervised_eligible_rows']:,} positive asset/deposit rows", "data/manifests/core_quality.json"],
        ["FFIEC Enhanced source", "2017Q1–2026Q2", f"{enhanced['rows']:,} raw bank-quarters; {features['enhanced_join_rows']:,} matched Core rows", "data/manifests/ffiec_quality.json"],
        ["FDIC institution history", "1970 onward", f"{entities['history_events']:,} events; {entities['failure_rows']:,} failure rows", "data/manifests/entity_quality.json"],
        ["ALFRED vintages", "2001Q1–2026Q2", f"{macro_quality['historical_vintage_snapshots']:,} quarter-origin snapshots", "data/manifests/macro_quality.json"],
        ["FR Y-9C extension", "2001Q1–2021Q1", f"{y9c['rows']:,} separate BHC-quarters", "data/manifests/y9c_quality.json"],
    ]
    body = f"<h1>Data lineage and quality</h1><p>Snapshot <code>{snapshot['data_snapshot_id']}</code>; generated from validated raw hashes and processed manifests.</p>"
    body += _table(["Dataset", "Coverage", "Verified scope", "Evidence"], rows)
    body += "<p class='note'>The <a href='https://www.newyorkfed.org/research/banking_research/balance-sheets-income-statements'>New York Fed</a>, <a href='https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx'>FFIEC CDR</a>, <a href='https://api.fdic.gov/banks/docs'>FDIC BankFind</a>, <a href='https://fred.stlouisfed.org/'>FRED/ALFRED</a>, and <a href='https://www.chicagofed.org/banking/financial-institution-reports/bhc-data'>Chicago Fed</a> are the official public sources. The raw manifest gives each file URL, hash, period, and vintage.</p>"
    body += "<h2>Coverage and accounting checks</h2><img src='../figures/01_data_coverage_timeline.png' alt='Institution coverage timeline'>"
    body += _table(["Check", "Observed result", "Interpretation"], [
        ["Duplicate RSSD-quarter keys", core["duplicate_bank_quarter"], "Hard check passed"],
        ["Nonpositive assets", core["nonpositive_assets"], "Excluded from supervised labels"],
        ["Nonpositive deposits", core["nonpositive_deposits"], "Excluded from supervised labels"],
        ["Asset = liabilities + equity gap >1%", f"{core['asset_identity_gap_gt_1pct']:,} / {core['asset_identity_gap_nonmissing']:,}", "Investigate; raw values retained"],
        ["Income join coverage", f"{core['income_join_coverage']:.4%}", "YTD flows converted only when prior quarter exists"],
        ["Direct FFIEC uninsured field overall", f"{enhanced['uninsured_coverage']:.1%}", "Form-dependent structural missingness; no zero fill"],
    ])
    body += "<h3>Direct uninsured-deposit coverage by form</h3>"
    body += _table(["FFIEC form", "Bank quarters", "Coverage"], [[row.ffiec_form, f"{row.rows:,}", f"{row.uninsured_coverage:.1%}"] for row in by_form.itertuples(index=False)])
    body += "<p class='caveat'>The New York Fed file is a currently restated historical snapshot, and the FFIEC bulk ZIPs are retrieval snapshots. A 45-day filing lag produces pseudo-point-in-time forecasts but does not reconstruct old bank amendments. Direct FFIEC uninsured estimates are not interchangeable with the historical insured-deposit proxy. H.8 remains an aggregate weekly series. Source and form scope are explicit in the feature registry.</p>"
    body += "<h2>Entity history and source limits</h2>"
    body += f"<p>{entities['structural_matched_1984_plus']:,} structural events from FDIC history/failure records matched a 1984+ bank ID within two quarters. Matching confidence and unmatched cases remain in <code>entity_events.parquet</code>. A disappearing Call Report row is not a failure label.</p>"
    body += "<p>The NIC current holding-company download endpoint presented a CAPTCHA in this environment. Official Chicago Fed historical FR Y-9C files through 2021Q1 were acquired and parsed separately. This leaves recent BHC external validation unavailable from that route.</p>"
    body += "<p><small>Numerical facts are stored in <code>artifacts/results.duckdb</code> tables <code>data_quality</code> and <code>data_coverage</code>, with detailed manifests under <code>data/manifests/</code>.</small></p>"
    (root / "reports" / "02_data_lineage_and_quality.html").write_text(_page("TreasuryShield data lineage and quality", body), encoding="utf-8")
    episode_rows = []
    for name, item in eda["episodes"].items():
        target = item["deposit_1q_growth_clean"]
        episode_rows.append([name, f"{item['bank_quarters']:,}", f"{item['institutions']:,}",
                             f"{target['p01']:.3f}", f"{target['p05']:.3f}",
                             f"{target['median']:.3f}", f"{item['mna_contaminated_1q']:,}"])
    eda_body = "<h1>Exploratory analysis, pre-2024</h1><p class='note'>Only origins through 2023Q4 enter these descriptive target analyses. The final 2024+ outcome period remained locked during feature and model specification.</p>"
    eda_body += "<img src='../figures/02_institution_count_timeline.png' alt='Bank count and historical stress windows'>"
    eda_body += "<h2>Distribution and stress periods</h2><img src='../figures/03_deposit_growth_distribution.png' alt='Pre-lock one-quarter deposit growth distribution'>"
    eda_body += _table(["Period", "Bank quarters", "Banks", "1Q Q01", "1Q Q05", "1Q median", "M&A affected"], episode_rows)
    eda_body += "<img src='../figures/04_macro_deposit_history.png' alt='Descriptive aggregate deposits and effective federal funds rate'>"
    eda_body += f"<p>{eda['extreme_1q_absolute_log_growth_over_log2']:,} one-quarter changes exceeded |log(2)| before 2024. {eda['extreme_classification'].get('structural_event', 0):,} overlapped the preregistered M&A window; {eda['extreme_classification'].get('plausible_or_unresolved_extreme', 0):,} require economic or case-specific review and were not deleted as errors.</p>"
    eda_body += "<p>Associations shown here are descriptive, not causal. Coverage shifts and changing bank composition limit raw period comparisons. Macro series use <a href='https://fred.stlouisfed.org/'>FRED</a> current values for this historical display; vintage-aware variants are evaluated separately.</p>"
    eda_body += "<p><small>Quarterly aggregate data are in <code>data_coverage</code>; source calculations are in <code>artifacts/eda_prelock_quarterly.parquet</code> and <code>data/manifests/eda_prelock.json</code>.</small></p>"
    (root / "reports" / "03_eda_report.html").write_text(_page("TreasuryShield pre-lock EDA", eda_body), encoding="utf-8")
    print("Wrote data-quality and pre-lock EDA reports plus four source-data figures", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    RunState(root).run(
        "data_quality_eda_reports", "REPORTING",
        {"snapshot": json.loads((root / "data" / "manifests" / "snapshot.json").read_text())["data_snapshot_id"],
         "eda_sha256": sha256_file(root / "data" / "manifests" / "eda_prelock.json")},
        ["reports/02_data_lineage_and_quality.html", "reports/03_eda_report.html",
         "figures/01_data_coverage_timeline.png", "figures/02_institution_count_timeline.png",
         "figures/03_deposit_growth_distribution.png", "figures/04_macro_deposit_history.png"],
        lambda: build(root), resume=True)


if __name__ == "__main__":
    main()
