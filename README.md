# MuJoCo Flexible-Line Environments

This directory is a standalone MuJoCo simulation environment for flexible line, cable, and rope-like tasks.

Path:

```bash
/mnt/sda/cy/shared_project/TTTdynamics/sim_envs/MuJoCo
```

The first target is configurable flexible-line dynamics: line length, radius, density, damping, friction, bend stiffness, twist stiffness, table friction, solver settings, and camera RGB-D output. This is intended to replace hinge-driven RoboTwin cloth placeholders for deformable-line data collection.

## Environment Setup

Use mirrors for Python/conda package downloads. Do not use HTTP proxy for `uv` or `conda`.

```bash
cd /mnt/sda/cy/shared_project/TTTdynamics/sim_envs/MuJoCo

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
  uv venv .venv --python 3.11

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
  uv pip install -e .
```

Conda fallback:

```bash
cd /mnt/sda/cy/shared_project/TTTdynamics/sim_envs/MuJoCo

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  CONDARC=$PWD/conda/.condarc \
  conda env create -f conda/environment.yml
```

GitHub operations are the only place where using the usual proxy is allowed.

## Smoke Test

```bash
cd /mnt/sda/cy/shared_project/TTTdynamics/sim_envs/MuJoCo
MUJOCO_GL=egl .venv/bin/python examples/run_flexible_line_demo.py \
  --config configs/flexible_line_medium.yml \
  --output outputs/flexible_line_medium
```

Expected outputs:

```text
outputs/flexible_line_medium/episode0.hdf5
outputs/flexible_line_medium/episode0.mp4
```

The HDF5 file stores:

- `observations/rgb`: RGB frames, shape `(T, H, W, 3)`
- `observations/depth`: depth frames, shape `(T, H, W)`
- `observations/cable_qpos`: MuJoCo qpos snapshots
- `actions/gripper_pos`: scripted gripper target positions
- `metadata/*`: camera and physics metadata

## Why MuJoCo

MuJoCo is a better fit than RoboTwin for flexible line/cable dynamics because the object is simulated as a deformable or articulated physical object rather than being closed by task code. This project starts with a cable-style line and a moving gripper proxy, then can be extended to real robot arms, 12D affordance metadata, and LeRobot conversion.

