"""Nineteen evidence-linked publication figures; fail if required data are absent."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from treasuryshield.orchestration.state import RunState, sha256_file, stable_hash


TITLES = [
    "Data coverage timeline", "Institution count timeline", "Deposit-growth distributions",
    "Macro and deposit stress history", "Forward-filtered HMM probabilities",
    "Quantile calibration", "Tail exceedance by quarter", "Crisis model comparison",
    "Paired block-bootstrap intervals", "Feature and context ablations",
    "Subgroup robustness", "Challenger disagreement and tail misses",
    "Scenario funding-gap fan", "Peak funding-need distribution",
    "Reverse-stress frontier", "Recovery cost and residual shortfall",
    "Historical monitoring states", "Feature-importance stability", "CPU and GPU benchmark",
]


def _quarter_axis(values: pd.Series) -> np.ndarray:
    return pd.PeriodIndex(values, freq="Q").to_timestamp(how="end").to_numpy()


def _fresh_fig() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(9.3, 5.4), constrained_layout=True)
    return fig, ax


def run(root: Path) -> None:
    selection_path = root / "artifacts" / "model_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    final_run = selection["final_run_id"]
    stress_run = selection["stress_run_id"]
    stress_base = root / "artifacts" / "stress" / stress_run
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(root / "artifacts" / "results.duckdb"), read_only=True)
    metrics = con.execute("SELECT * FROM metrics").df()
    bootstrap = con.execute("SELECT * FROM bootstrap_results").df()
    ablations = con.execute("SELECT * FROM ablations").df()
    subgroup = con.execute("SELECT * FROM subgroup_metrics").df()
    con.close()
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14,
                         "axes.labelsize": 11, "savefig.dpi": 210})
    manifest = []
    source_hash_cache: dict[Path, str] = {}

    def save(fig: plt.Figure, number: int, source_paths: list[Path], note: str = "") -> None:
        if any(not path.is_file() for path in source_paths):
            raise FileNotFoundError([str(p) for p in source_paths if not p.is_file()])
        fig.suptitle(f"{number:02d}. {TITLES[number - 1]}", fontsize=15, weight="bold")
        fig.text(.01, -.035, "TreasuryShield-MRM | public-data research proxy", fontsize=8, color="#57606a")
        path = out_dir / f"figure_{number:02d}.png"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        for source in source_paths:
            if source not in source_hash_cache:
                source_hash_cache[source] = sha256_file(source)
        manifest.append({"number": number, "title": TITLES[number - 1],
                         "path": str(path.relative_to(root)).replace("\\", "/"),
                         "sha256": sha256_file(path),
                         "sources": [{"path": str(p.relative_to(root)).replace("\\", "/"),
                                      "sha256": source_hash_cache[p]} for p in source_paths],
                         "note": note})

    coverage_path = root / "artifacts" / "eda_prelock_quarterly.parquet"
    coverage = pd.read_parquet(coverage_path)
    coverage = coverage.loc[coverage.report_period.ge("2001Q1")]
    x = _quarter_axis(coverage.report_period)
    fig, ax = _fresh_fig()
    ax.plot(x, coverage.uninsured_proxy_coverage, label="Core uninsured proxy observed")
    ax.plot(x, coverage.deposit_beta_coverage, label="4Q deposit beta observed")
    ax.set(ylabel="Share of bank-quarter observations", ylim=(0, 1))
    ax.legend(loc="best")
    save(fig, 1, [coverage_path], "Prelock through 2023Q4; missingness is structural in some fields.")

    fig, ax = _fresh_fig()
    ax.plot(x, coverage.institutions, color="#174a7e")
    ax.set(ylabel="Commercial banks", xlabel="Report quarter")
    save(fig, 2, [coverage_path])

    target_path = root / "data" / "features" / "targets.parquet"
    target = pd.read_parquet(target_path, columns=["report_period", "horizon_quarters",
                                                  "deposit_log_growth", "mna_contaminated_radius_1"])
    target = target.loc[target.report_period.between("2001Q1", "2023Q4") &
                        target.horizon_quarters.eq(1) & target.mna_contaminated_radius_1.eq(False)]
    sample = target.deposit_log_growth.sample(n=min(180_000, len(target)), random_state=1701)
    fig, ax = _fresh_fig()
    ax.hist(sample.clip(-.35, .35), bins=90, color="#2471a3", alpha=.85)
    ax.set(xlabel="One-quarter log deposit growth (clipped for display at ±0.35)",
           ylabel="Sampled bank-quarters")
    save(fig, 3, [target_path], "Deterministic 180,000-row or smaller sample; extreme observations retained in analysis.")

    macro_path = root / "data" / "processed" / "macro_origins.parquet"
    macro = pd.read_parquet(macro_path, columns=["report_period", "macro_DFF_revised"])
    macro = macro.loc[macro.report_period.between("2001Q1", "2023Q4")]
    merged = coverage.merge(macro, on="report_period", validate="one_to_one")
    fig, ax = _fresh_fig()
    dates = _quarter_axis(merged.report_period)
    ax.plot(dates, merged.deposit_1q_p05, color="#a64032", label="Bank deposit growth P05")
    ax.set(ylabel="One-quarter log deposit growth P05", xlabel="Report quarter")
    ax2 = ax.twinx()
    ax2.plot(dates, merged.macro_DFF_revised, color="#234e70", alpha=.6,
             label="Federal funds rate (%)")
    ax2.set_ylabel("Federal funds rate (%)")
    save(fig, 4, [coverage_path, macro_path], "Bank growth distribution and published macro origin series.")

    regime_path = root / "artifacts" / "regime_probabilities_prelock.parquet"
    regime = pd.read_parquet(regime_path)
    regime = regime.loc[regime.report_period.ge("2010Q1")]
    fig, ax = _fresh_fig()
    t = _quarter_axis(regime.report_period)
    for state in range(3):
        ax.plot(t, regime[f"regime_p{state}"], label=f"Filtered state {state}")
    ax.set(ylabel="Filtered probability", xlabel="Report quarter", ylim=(0, 1))
    ax.legend(ncol=3, fontsize=9)
    save(fig, 5, [regime_path, root / "artifacts" / "regime_probabilities_prelock.json"],
         "State numbers are descriptive and fitted only through 2021Q4.")

    calibration = metrics.loc[(metrics.run_id == final_run) & (metrics.metric == "coverage") &
                              (metrics.weighting == "equal_bank")]
    calibration = calibration.loc[calibration["quantile"].isin(
        ["Q01", "Q05", "Q10", "Q25", "Q50", "Q75", "Q90"])].copy()
    nominal = np.array([.01, .05, .10, .25, .50, .75, .90])
    observed = np.array([float(calibration.loc[calibration["quantile"].eq(f"Q{int(q*100):02d}"),
                                               "value"].iloc[0]) for q in nominal])
    fig, ax = _fresh_fig()
    ax.plot(nominal, nominal, ls="--", color="gray", label="Nominal")
    ax.plot(nominal, observed, marker="o", color="#a64032", label="Selected locked holdout")
    ax.set(xlabel="Nominal lower-tail probability", ylabel="Empirical proportion below forecast",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    save(fig, 6, [root / "artifacts" / "results.duckdb",
                  root / "artifacts" / "predictions" / f"{final_run}.parquet"])

    final_path = root / "artifacts" / "predictions" / f"{final_run}.parquet"
    final = pd.read_parquet(final_path, columns=["report_period", "actual", "q05"])
    rate = final.assign(breach=final.actual.lt(final.q05)).groupby("report_period").breach.agg(
        ["mean", "size"]).reset_index()
    fig, ax = _fresh_fig()
    ax.bar(_quarter_axis(rate.report_period), rate["mean"], width=60, color="#9d473c")
    ax.axhline(.05, color="black", ls="--", label="Nominal 5%")
    ax.set(ylabel="Fraction below forecast Q05", xlabel="Locked holdout forecast quarter",
           ylim=(0, max(.1, float(rate["mean"].max()) * 1.15)))
    ax.legend()
    save(fig, 7, [final_path])

    compare = metrics.loc[(metrics.metric == "pinball") & (metrics["quantile"] == "Q05") &
                          (metrics.weighting == "equal_bank") & (metrics.panel == "core") &
                          (metrics.horizon == 1) & metrics.split.isin(("gfc", "pandemic", "tightening")) &
                          metrics.model_id.isin(("B1_peer", "B5_linear_quantile",
                                                 "B7_xgb_quantile", "B8_lightgbm_quantile",
                                                 "RAHQ_C_tree", "RAHQ_C_nn"))]
    compare = compare.loc[compare["mode"].eq("B") & compare.run_id.str.endswith(
        tuple(("B1_peer", "B5_linear_quantile", "B7_xgb_quantile",
               "B8_lightgbm_quantile", "RAHQ_C_tree", "RAHQ_C_nn")))]
    grid = compare.pivot_table(index="model_id", columns="split", values="value", aggfunc="first")
    grid = grid.reindex(columns=["gfc", "pandemic", "tightening"]).dropna(how="all")
    fig, ax = _fresh_fig()
    grid.plot(kind="bar", ax=ax, width=.8)
    ax.set(ylabel="Equal-bank Q05 pinball loss", xlabel="Model", title="Vintage-aware prelock folds")
    ax.tick_params(axis="x", labelrotation=35)
    ax.legend(title="Episode", loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, 8, [root / "artifacts" / "results.duckdb"])

    boot = bootstrap.loc[bootstrap["quantile"].eq("Q05") &
                         bootstrap.comparison_id.str.startswith(final_run + "__vs__") &
                         bootstrap.comparison_id.str.endswith("__time_block")].copy()
    boot = boot.sort_values("estimate")
    if boot.empty:
        raise ValueError("No final paired time-block bootstrap results")
    fig, ax = plt.subplots(figsize=(9.3, 7.4), constrained_layout=True)
    labels = [row.reference.replace("_", " ") for row in boot.itertuples()]
    yloc = np.arange(len(boot))
    ax.hlines(yloc, boot.ci_low, boot.ci_high, color="#235580", lw=2)
    ax.scatter(boot.estimate, yloc, color="#235580", zorder=3)
    ax.set_yticks(yloc, labels, fontsize=8)
    ax.axvline(0, color="black", ls="--")
    ax.set(xlabel="Selected minus comparator Q05 pinball (negative favors selected)")
    save(fig, 9, [root / "artifacts" / "results.duckdb"])

    abl = ablations.loc[ablations.experiment_id.isin(
        [f"E{x:02d}" for x in range(10, 20)]) & ablations.metric.eq("q05_pinball") &
                        ablations.reference_value.notna()].copy()
    abl["relative_change"] = (abl.value - abl.reference_value) / abl.reference_value
    abl = abl.sort_values("relative_change").drop_duplicates("variant").tail(14)
    fig, ax = _fresh_fig()
    colors = ["#9b4338" if value > 0 else "#287e69" for value in abl.relative_change]
    ax.barh(abl.variant, abl.relative_change * 100, color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set(xlabel="Change in Q05 pinball loss vs reference (%)")
    save(fig, 10, [root / "artifacts" / "results.duckdb"], "Positive values indicate worse loss.")

    groups = subgroup.loc[subgroup.run_id.eq(final_run) &
                          (subgroup.metric == "pinball") & (subgroup["quantile"] == "Q05") &
                          (subgroup.weighting == "equal_bank")]
    groups = groups.sort_values(["subgroup_family", "subgroup"])
    fig, ax = _fresh_fig()
    labels = (groups.subgroup_family.str.replace("_", " ", regex=False) + ": " +
              groups.subgroup.str.replace("_", " ", regex=False))
    ax.barh(labels, groups.value, color="#4b7895")
    ax.set(xlabel="Q05 pinball loss", title="Locked holdout, Core proxy groups")
    save(fig, 11, [root / "artifacts" / "results.duckdb",
                   root / "artifacts" / "independent_final_review.json"])

    disagreement_path = root / "artifacts" / "disagreement_prelock.json"
    disagreement = json.loads(disagreement_path.read_text(encoding="utf-8"))["episodes"]
    fig, ax = _fresh_fig()
    ax.bar([e["split"] for e in disagreement],
           [e["tail_miss_lift_at_top_decile"] for e in disagreement], color="#a75f40")
    ax.axhline(1, color="black", ls="--", label="No lift")
    ax.set(ylabel="Tail-miss rate in highest disagreement decile / average")
    ax.legend()
    save(fig, 12, [disagreement_path])

    draw_path = Path(f"{stress_base}_sample_draws.parquet")
    draws = pd.read_parquet(draw_path)
    fig, ax = _fresh_fig()
    for scenario, part in draws.groupby("scenario"):
        times = [7, 30, 90]
        lows = [part[f"gap_{day}d"].quantile(.10) for day in times]
        mid = [part[f"gap_{day}d"].quantile(.50) for day in times]
        highs = [part[f"gap_{day}d"].quantile(.90) for day in times]
        line, = ax.plot(times, mid, marker="o", label=scenario)
        ax.fill_between(times, lows, highs, color=line.get_color(), alpha=.12)
    ax.axhline(0, color="#555", ls="--", lw=1)
    ax.set(xlabel="Research scenario horizon (days)",
           ylabel="Net funding gap proxy (USD thousands; positive = shortfall)")
    ax.legend()
    save(fig, 13, [draw_path], "Bands are 10th–90th draw quantiles on the sampled bank profiles.")

    fig, ax = _fresh_fig()
    series = [part.peak_funding_need_proxy.to_numpy() for _, part in draws.groupby("scenario")]
    labels = [name for name, _ in draws.groupby("scenario")]
    ax.boxplot(series, tick_labels=labels, showfliers=False)
    ax.set(ylabel="Peak funding-need proxy (USD thousands)")
    save(fig, 14, [draw_path], "Box plots exclude visual outlier markers; values remain in stored draws.")

    reverse_path = Path(f"{stress_base}_reverse.parquet")
    reverse = pd.read_parquet(reverse_path)
    reverse = reverse.loc[reverse.reachable_within_100pct_runoff]
    frontier = reverse.groupby(["haircut", "access_reduction"]).break_even_runoff_fraction.median().reset_index()
    fig, ax = _fresh_fig()
    for access, part in frontier.groupby("access_reduction"):
        ax.plot(part.haircut, part.break_even_runoff_fraction, marker="o", label=f"Access loss {access:.0%}")
    ax.set(xlabel="Securities haircut", ylabel="Median breaking deposit-runoff fraction", ylim=(0, 1))
    ax.legend()
    save(fig, 15, [reverse_path], "Unreachable combinations above 100% runoff are excluded from the plotted frontier.")

    recovery_path = Path(f"{stress_base}_recovery.parquet")
    recovery = pd.read_parquet(recovery_path)
    summary = recovery.groupby("strategy")[["action_cost", "residual_shortfall"]].mean().reset_index()
    fig, ax = _fresh_fig()
    ax.scatter(summary.action_cost, summary.residual_shortfall, s=80, color="#386f8f")
    for row in summary.itertuples():
        ax.annotate(row.strategy, (row.action_cost, row.residual_shortfall), xytext=(4, 5),
                    textcoords="offset points", fontsize=9)
    ax.set(xlabel="Mean action cost proxy (USD thousands)",
           ylabel="Mean residual shortfall (USD thousands)")
    save(fig, 16, [recovery_path], "Costs and capacities are declared scenario assumptions.")

    monitoring_path = root / "artifacts" / "monitoring_prelock.json"
    timeline = pd.DataFrame(json.loads(monitoring_path.read_text(encoding="utf-8"))["timeline"])
    states = ["GREEN", "WATCH", "REVIEW", "RECALIBRATE", "REDEVELOP"]
    policies = ["static", "annual_retrain", "quarterly_recalibration", "event_triggered", "performance_triggered"]
    matrix = timeline.pivot(index="policy", columns="report_period", values="state").reindex(policies)
    values = matrix.map(lambda value: states.index(value)).to_numpy()
    fig, ax = _fresh_fig()
    cmap = matplotlib.colors.ListedColormap(["#b6d7a8", "#ffe599", "#f9cb9c", "#e69191", "#bd6d6d"])
    image_state = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=4)
    state_key = fig.colorbar(image_state, ax=ax, ticks=np.arange(len(states)),
                             fraction=.035, pad=.025)
    state_key.ax.set_yticklabels(states)
    ax.set_yticks(np.arange(len(policies)), [name.replace("_", " ") for name in policies])
    ax.set_xticks(np.arange(0, len(matrix.columns), 4), matrix.columns[::4], rotation=45)
    ax.set(xlabel="Historical forecast quarter")
    save(fig, 17, [monitoring_path])

    importance_path = root / "artifacts" / "feature_importance_prelock.parquet"
    importance = pd.read_parquet(importance_path)
    top = importance.groupby("feature").gain_share.mean().nlargest(12).index
    matrix = importance.loc[importance.feature.isin(top)].pivot(
        index="feature", columns="split", values="gain_share").reindex(top)
    fig, ax = _fresh_fig()
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index, fontsize=8)
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns)
    fig.colorbar(im, ax=ax, label="Share of training gain")
    save(fig, 18, [importance_path, root / "artifacts" / "feature_importance_prelock.json"])

    benchmark_path = root / "artifacts" / "hardware_benchmarks.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))["matched_workloads"]
    fig, ax = _fresh_fig()
    positions = np.arange(2)
    cpu = [next(item["training_seconds"] for item in benchmark
                if item["workload"].startswith(prefix) and item["device"] == "cpu")
           for prefix in ("xgboost", "pytorch")]
    gpu = [next(item["training_seconds"] for item in benchmark
                if item["workload"].startswith(prefix) and item["device"] == "cuda")
           for prefix in ("xgboost", "pytorch")]
    ax.bar(positions - .2, cpu, width=.38, label="CPU")
    ax.bar(positions + .2, gpu, width=.38, label="RTX CUDA")
    ax.set_xticks(positions, ["XGBoost\n100K rows · 7Q · 80 trees",
                              "PyTorch MLP\n40K rows · 4 epochs"])
    ax.set(ylabel="Training time (seconds)")
    ax.legend()
    save(fig, 19, [benchmark_path], "Workloads are matched within each pair; the two pairs differ in model and size.")

    if len(manifest) != 19 or [item["number"] for item in manifest] != list(range(1, 20)):
        raise AssertionError("Required figure contract incomplete")
    output = out_dir / "figure_manifest.json"
    output.write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(),
                                  "data_snapshot_id": RunState(root).data["data_snapshot_id"],
                                  "selected_run_id": final_run, "stress_run_id": stress_run,
                                  "figures": manifest}, indent=2), encoding="utf-8")
    print("Generated 19 evidence-linked figures", flush=True)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    selected = json.loads((project_root / "artifacts" / "model_selection.json").read_text(encoding="utf-8"))
    RunState(project_root).run(
        "required_figures", "REPORTING",
        {"selection_sha256": sha256_file(project_root / "artifacts" / "model_selection.json"),
         "results_sha256": sha256_file(project_root / "artifacts" / "results.duckdb"),
         "stress_sha256": sha256_file(project_root / "artifacts" / "stress" /
                                       f"{selected['stress_run_id']}_manifest.json"),
         "source_sha256": sha256_file(Path(__file__))},
        [f"figures/figure_{number:02d}.png" for number in range(1, 20)] +
        ["figures/figure_manifest.json"], lambda: run(project_root), resume=True)
