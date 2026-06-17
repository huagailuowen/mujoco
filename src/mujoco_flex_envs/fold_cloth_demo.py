from __future__ import annotations

import argparse
from pathlib import Path

from .fold_cloth import FoldClothConfig, FoldClothEnv, save_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MuJoCo flex-grid cloth folding scripted rollout.")
    parser.add_argument(
        "--config",
        default="configs/fold_cloth_medium.yml",
        help="Path to a fold-cloth YAML config.",
    )
    parser.add_argument(
        "--output",
        default="outputs/fold_cloth_medium",
        help="Directory for episode0.hdf5 and episode0.mp4.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FoldClothConfig.from_yaml(args.config)
    env = FoldClothEnv(config)
    try:
        rollout = env.rollout()
        h5_path, video_path = save_rollout(rollout, Path(args.output), config)
    finally:
        env.close()

    print(f"Saved HDF5: {h5_path}")
    print(f"Saved video: {video_path}")
    print(f"Frames: {rollout['rgb'].shape[0]}")


if __name__ == "__main__":
    main()

