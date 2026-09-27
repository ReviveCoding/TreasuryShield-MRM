"""Resume-safe prelock model matrix; never touches the 2024+ outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path

from treasuryshield.development.run import run
from treasuryshield.splits.temporal import load_joined


def execute(root: Path, group: str) -> None:
    if group == "mna":
        frame = load_joined(root, "core", 1, mode="B")
        for radius in (0, 2, None):
            run(root, "B8_lightgbm_quantile", "tightening", "core", 1,
                mode="B", mna_radius=radius, frame_override=frame)
        return
    jobs: dict[str, list[tuple[str, str, str, int, str, dict]]] = {
        "complete_1q": [
            (model, split, "core", 1, "A", {})
            for split in ("pandemic", "tightening")
            for model in ("B0_unconditional", "B6_quantile_forest", "B10_mlp", "B10_gru")
        ],
        "mode_b_simple": [
            (model, split, "core", 1, "B", {})
            for split in ("gfc", "pandemic", "tightening")
            for model in ("B1_peer", "B5_linear_quantile")
        ],
        "linear_panel_refresh": [
            (model, split, "core", 1, "A", {})
            for split in ("gfc", "pandemic", "tightening")
            for model in ("B3_ols", "B4_panel_fe")
        ],
        "routine_temporal": [
            (model, "routine_2018_19", "core", 1, "B", {})
            for model in ("B1_peer", "B5_linear_quantile", "B7_xgb_quantile",
                          "B8_lightgbm_quantile", "RAHQ_C_tree", "RAHQ_C_nn")
        ],
        "multi_horizon": [
            (model, split, "core", horizon, "B", {})
            for horizon in (2, 4)
            for split in ("gfc", "pandemic", "tightening")
            for model in ("B0_unconditional", "B1_peer", "B2_persistence",
                          "B5_linear_quantile", "B7_xgb_quantile", "B8_lightgbm_quantile",
                          "RAHQ_tree", "RAHQ_C_tree")
        ],
        "proposed_nn": [
            (model, split, "core", horizon, "B", {})
            for horizon in (1, 2, 4)
            for split in ("gfc", "pandemic", "tightening")
            for model in ("RAHQ_nn", "RAHQ_C_nn")
        ],
        "enhanced": [
            (model, split, "enhanced", 1, "B", {})
            for split in ("pandemic", "tightening")
            for model in ("B1_peer", "B5_linear_quantile", "B7_xgb_quantile",
                          "B8_lightgbm_quantile", "RAHQ_C_tree")
        ],
        "calibration": [
            ("RAHQ_tree", split, "core", 1, "B", {"calibration_kind": kind})
            for split in ("gfc", "pandemic", "tightening")
            for kind in ("peer", "regime")
        ] + [
            ("B8_lightgbm_quantile", split, "core", 1, "B", {"calibration_kind": "peer"})
            for split in ("gfc", "pandemic", "tightening")
        ],
        "adaptive": [
            (model, split, "core", 1, "B", {"calibration_kind": "adaptive_global"})
            for split in ("gfc", "pandemic", "tightening")
            for model in ("B8_lightgbm_quantile", "RAHQ_C_tree", "RAHQ_C_nn")
        ],
        "neural_ablation": [
            ("RAHQ_C_nn", "tightening", "core", 1, "B", {"architecture": "mlp"}),
            ("RAHQ_C_nn", "gfc", "core", 1, "B", {"tail_weighted": True}),
            ("RAHQ_C_nn", "tightening", "core", 1, "B", {"tail_weighted": True}),
        ],
        "contexts": [
            ("RAHQ_tree", "tightening", "core", 1, "B", cfg)
            for cfg in (
                {"peer_kind": "none"}, {"peer_kind": "learned"},
                {"regime_kind": "none"}, {"regime_kind": "rule"},
                {"regime_kind": "hmm_hard"},
                {"regime_kind": "hmm_smoothed_negative_control"},
                {"regime_kind": "none", "calibration_kind": "global"},
                {"states": 2}, {"states": 4},
                {"remove_families": ["peer"]}, {"remove_families": ["macro"]},
                {"remove_families": ["macro", "regime"], "regime_kind": "none"},
                {"remove_families": ["regime"]},
                {"remove_families": ["dynamics"]},
                {"remove_families": ["deposit_composition"]},
                {"remove_families": ["rate_sensitivity"]},
                {"remove_families": ["peer", "regime"]},
            )
        ] + [
            ("RAHQ_tree", "gfc", "core", 1, "B",
             {"remove_families": ["deposit_composition"]})
        ],
    }
    cached_key = None
    cached_frame = None
    for model, split, panel, horizon, mode, config in jobs[group]:
        key = (panel, horizon, mode)
        if key != cached_key:
            cached_frame = load_joined(root, panel, horizon, mode=mode)
            cached_key = key
        print(f"RUN {group}: {model} {split} {panel} {horizon}Q {mode} {config}", flush=True)
        run(root, model, split, panel, horizon, mode=mode, model_config=config,
            frame_override=cached_frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("group", nargs="+", choices=["complete_1q", "mode_b_simple",
                                                     "linear_panel_refresh", "multi_horizon",
                                                     "routine_temporal", "proposed_nn", "enhanced", "calibration",
                                                     "adaptive", "neural_ablation", "contexts", "mna"])
    for group in parser.parse_args().group:
        execute(Path(__file__).resolve().parents[3], group)


if __name__ == "__main__":
    main()
