# MuJoCo Flexible-Object Environments

This directory is a standalone MuJoCo simulation environment for flexible line, cable, rope-like, and shell-cloth manipulation tasks.

Path:

```bash
/mnt/sda/cy/shared_project/TTTdynamics/sim_envs/MuJoCo
```

The environments are intended to replace hinge-driven RoboTwin placeholders when the object physics must matter. The line task uses MuJoCo cable dynamics. The cloth task uses `mujoco.elasticity.shell`, flex edge constraints, table contact, and grasp constraints welded to cloth vertices during the grasp phase.

The fold-cloth setup follows the same structure as existing MuJoCo cloth manipulation examples such as `benchmarking_cloth`: shell cloth, edge equality, high-friction table contact, corner/edge grasp constraints, and scripted quasi-static folding trajectories. The Panda mesh variant additionally follows the robot model structure from `dynamic-cloth-folding`, which used a Franka Emika Panda for cloth-folding experiments.

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

The fold-cloth environment is pinned to `mujoco==3.2.6`. This is intentional: the local `3.9.0` wheel only exposed `mujoco.elasticity.cable`, while `3.2.6` includes `mujoco.elasticity.shell`, which is needed to reproduce the existing cloth-manipulation configuration style.

Conda fallback:

```bash
cd /mnt/sda/cy/shared_project/TTTdynamics/sim_envs/MuJoCo

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  CONDARC=$PWD/conda/.condarc \
  conda env create -f conda/environment.yml
```

GitHub operations are the only place where using the usual proxy is allowed.

## Flexible-Line Smoke Test

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

## Robot Fold-Cloth Demo

```bash
cd /mnt/sda/cy/shared_project/TTTdynamics/sim_envs/MuJoCo
MUJOCO_GL=egl .venv/bin/python examples/run_fold_cloth_demo.py \
  --config configs/fold_cloth_medium.yml \
  --output outputs/fold_cloth_medium_robot
```

Expected outputs:

```text
outputs/fold_cloth_medium_robot/episode0.hdf5
outputs/fold_cloth_medium_robot/episode0.mp4
```

The medium config is 5.8 seconds at 50 FPS, so the demo should produce 290 frames. The two robot end-effectors grip the right cloth edge, lift, fold across the center line, lower, release the weld constraints, and then lift away while the cloth settles.

The default fold-cloth configs use lightweight Cartesian end-effectors. To run the robot-mesh variant, use:

```bash
cd /mnt/sda/cy/shared_project/TTTdynamics/sim_envs/MuJoCo
MUJOCO_GL=egl .venv/bin/python examples/run_fold_cloth_demo.py \
  --config configs/fold_cloth_medium_panda.yml \
  --output outputs/fold_cloth_medium_panda
```

`configs/fold_cloth_medium_panda.yml` instantiates two Franka Panda mesh arms using STL assets from `dynamic-cloth-folding`, solves position-only IK for each gripper site, and keeps the same cloth vertex weld grasp model used by the MuJoCo cloth-manipulation references. This validated demo keeps the grasp active through the rollout; release timing should be tuned separately because releasing a deformable shell from hard constraints can destabilize the scene. It is an environment-level robot model, not a trained control policy.

`configs/fold_cloth_soft.yml` intentionally uses much softer shell parameters than the medium config so the cloth collapses and wrinkles. `configs/fold_cloth_stiff.yml` uses a stable high-stiffness setting that behaves more like a resistant sheet without the startup bounce caused by over-hard shell parameters.

`configs/fold_cloth_very_stiff.yml` raises Young's modulus further while keeping the sheet thin enough to avoid the startup spring-board artifact. Use it when testing a more resistant cloth without returning to the unstable stress-test parameters.

`configs/fold_cloth_thick_stiff.yml` increases physical thickness instead. This makes the sheet behavior more visually distinct and more plate-like, while keeping Young's modulus lower to avoid the severe startup bounce of the earlier over-hard stress test.

`configs/fold_cloth_thick_stiff_light.yml` keeps the thick-stiff shell parameters but reduces total cloth mass to one tenth. It is useful for checking how much inertial weight changes the same scripted fold.

`configs/fold_cloth_coarse_stiff.yml` reduces the cloth grid from 17x13 vertices to 9x7 vertices. This cuts the quad count from 192 to 48, roughly a 4x reduction, making the sheet visibly coarser and more resistant to local bending. More aggressive 6x5 and 7x5 grids were tested but are unstable with the current shell-cloth and welded-grasp setup.

The HDF5 file stores:

- `observations/rgb`: RGB frames, shape `(T, H, W, 3)`
- `observations/depth`: depth frames, shape `(T, H, W)`
- `observations/sim_qpos` and `observations/sim_qvel`: full MuJoCo state snapshots
- `observations/robot_qpos` and `observations/robot_qvel`: six Cartesian robot joint states
- `observations/robot_ee_pos`: actual end-effector positions
- `observations/cloth_vertex_pos`: cloth vertex world positions
- `actions/robot_target_pos`: scripted robot target positions
- `actions/grasp_active`: whether the cloth weld grasp is active
- `metadata/cloth_shell_plugin`: `mujoco.elasticity.shell`

Panda mesh assets are stored under `assets/franka_panda/`. They come from the MIT-licensed `dynamic-cloth-folding` project; the upstream license is included next to the assets.

## Why MuJoCo

MuJoCo is a better fit than RoboTwin for flexible line/cable/cloth dynamics because the object is simulated as a deformable physical object rather than being closed by task code. This project now has both cable-style line dynamics and a shell-cloth folding demo with robot end-effectors, and can be extended to concrete robot URDFs, 12D affordance metadata, and LeRobot conversion.
