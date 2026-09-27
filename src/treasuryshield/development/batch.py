"""Serial, resume-safe experiment batch; reuses loaded panel by horizon."""

from __future__ import annotations

import argparse
from pathlib import Path

from treasuryshield.development.run import run
from treasuryshield.splits.temporal import load_joined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["gfc", "pandemic", "tightening"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--panel", choices=["core", "enhanced"], default="core")
    parser.add_argument("--mode", choices=["A", "B"], default="A")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    for horizon in args.horizons:
        frame = load_joined(root, args.panel, horizon, mode=args.mode)
        print(f"Loaded {args.panel} {args.mode} {horizon}Q: {len(frame)} labeled origins", flush=True)
        for split in args.splits:
            if args.panel == "enhanced" and split == "gfc":
                continue
            for model in args.models:
                run(root, model, split, args.panel, horizon,
                    mode=args.mode, frame_override=frame)


if __name__ == "__main__":
    main()
