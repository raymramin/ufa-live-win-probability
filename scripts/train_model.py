#!/usr/bin/env python
"""Train smooth win model from data/training_frame.parquet or CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from live_win_prob.model import train_smooth_win_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "training_frame.parquet")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "smooth_win_model.joblib")
    parser.add_argument("--lam", type=float, default=8.0)
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Missing {args.data}. Run scripts/export_training_frame.py or demo_synthetic.py")

    if args.data.suffix == ".parquet":
        frame = pd.read_parquet(args.data)
    else:
        frame = pd.read_csv(args.data)

    model = train_smooth_win_model(frame, output_path=args.out, gam_lam=args.lam)
    print(f"Trained SmoothWinModel → {args.out}")
    print(f"Features: {model.feature_cols}")


if __name__ == "__main__":
    main()
