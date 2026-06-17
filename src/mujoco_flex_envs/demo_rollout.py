from __future__ import annotations

import argparse
from pathlib import Path

from .flexible_line import FlexibleLineConfig, FlexibleLineEnv, save_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MuJoCo flexible-line scripted rollout.")
    parser.add_argument(
        "--config",
        default="configs/flexible_line_medium.yml",
        help="Path to a flexible-line YAML config.",
    )
    parser.add_argument(
        "--output",
        default="outputs/flexible_line_medium",
        help="Directory for episode0.hdf5 and episode0.mp4.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FlexibleLineConfig.from_yaml(args.config)
    env = FlexibleLineEnv(config)
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

