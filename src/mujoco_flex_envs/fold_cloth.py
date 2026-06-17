from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
import yaml


@dataclass
class FoldClothConfig:
    seed: int
    duration: float
    timestep: float
    control_dt: float
    image_width: int
    image_height: int
    fps: int
    physics: dict[str, Any]
    cloth: dict[str, Any]
    gripper: dict[str, Any]
    robot: dict[str, Any] | None = None
    trajectory: dict[str, Any] | None = None
    output: dict[str, Any] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FoldClothConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)


def _vec(values: list[float] | tuple[float, ...] | np.ndarray, precision: int = 6) -> str:
    return " ".join(f"{float(v):.{precision}g}" for v in values)


def _fraction_to_grid_index(fraction: float, count: int) -> int:
    return int(np.clip(round(float(fraction) * (count - 1)), 0, count - 1))


def _cloth_vertex_body_name(x_idx: int, y_idx: int, grid_y: int) -> str:
    return f"cloth_{x_idx * grid_y + y_idx}"


def _edge_anchor(size_x: float, size_y: float, x_side: str, y_fraction: float, z: float) -> np.ndarray:
    x = -0.5 * size_x if x_side == "left" else 0.5 * size_x
    y = -0.5 * size_y + float(y_fraction) * size_y
    return np.array([x, y, z], dtype=np.float64)


def _smoothstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _coerce_waypoints(config: FoldClothConfig, cloth_z: float) -> list[dict[str, float]]:
    trajectory = config.trajectory or {}
    waypoints = trajectory.get("waypoints")
    if waypoints:
        return [
            {
                "time": float(point["time"]),
                "x": float(point["x"]),
                "z": float(point["z"]),
                "y_offset": float(point.get("y_offset", 0.0)),
            }
            for point in waypoints
        ]

    path = config.gripper.get("path", [])
    if path:
        times = np.linspace(0.0, config.duration, len(path))
        return [
            {
                "time": float(time),
                "x": float(point["right_x"]),
                "z": float(point["z"]),
                "y_offset": 0.0,
            }
            for time, point in zip(times, path)
        ]

    size_x, _ = [float(v) for v in config.cloth["size"]]
    right_x = 0.5 * size_x
    return [
        {"time": 0.00, "x": right_x, "z": cloth_z, "y_offset": 0.0},
        {"time": 0.35, "x": right_x, "z": cloth_z + 0.015, "y_offset": 0.0},
        {"time": 1.10, "x": right_x, "z": cloth_z + 0.145, "y_offset": 0.0},
        {"time": 2.80, "x": -0.20 * size_x, "z": cloth_z + 0.155, "y_offset": 0.0},
        {"time": 3.70, "x": -0.48 * size_x, "z": cloth_z + 0.030, "y_offset": 0.0},
        {"time": 4.10, "x": -0.48 * size_x, "z": cloth_z + 0.026, "y_offset": 0.0},
        {"time": 4.70, "x": -0.48 * size_x, "z": cloth_z + 0.150, "y_offset": 0.0},
        {"time": 5.00, "x": -0.48 * size_x, "z": cloth_z + 0.150, "y_offset": 0.0},
    ]


def _interpolate_waypoints(waypoints: list[dict[str, float]], time_s: float) -> dict[str, float]:
    if time_s <= waypoints[0]["time"]:
        return waypoints[0]
    if time_s >= waypoints[-1]["time"]:
        return waypoints[-1]

    for left, right in zip(waypoints[:-1], waypoints[1:]):
        if left["time"] <= time_s <= right["time"]:
            span = max(right["time"] - left["time"], 1e-9)
            alpha = _smoothstep((time_s - left["time"]) / span)
            return {
                "time": time_s,
                "x": (1.0 - alpha) * left["x"] + alpha * right["x"],
                "z": (1.0 - alpha) * left["z"] + alpha * right["z"],
                "y_offset": (1.0 - alpha) * left.get("y_offset", 0.0)
                + alpha * right.get("y_offset", 0.0),
            }

    return waypoints[-1]


def _cartesian_robot_xml(
    name: str,
    base_pos: np.ndarray,
    rgba: str,
    grip_radius: float,
    robot_cfg: dict[str, Any],
) -> tuple[str, str]:
    kp = float(robot_cfg.get("kp", 900.0))
    damping = float(robot_cfg.get("joint_damping", 8.0))
    armature = float(robot_cfg.get("joint_armature", 0.02))
    x_range = robot_cfg.get("x_range", [-0.55, 0.12])
    y_range = robot_cfg.get("y_range", [-0.10, 0.10])
    z_range = robot_cfg.get("z_range", [-0.006, 0.26])
    body = f"""    <body name="{name}_base" pos="{_vec(base_pos)}">
      <geom name="{name}_base_marker" type="cylinder" pos="0 0 -0.025" size="0.018 0.025" rgba="0.18 0.18 0.18 1" contype="0" conaffinity="0"/>
      <body name="{name}_x_link">
        <joint name="{name}_x" type="slide" axis="1 0 0" range="{_vec(x_range)}" damping="{damping:.8f}" armature="{armature:.8f}"/>
        <inertial pos="0 0 0" mass="0.05" diaginertia="0.00002 0.00002 0.00002"/>
        <geom name="{name}_x_carriage" type="box" size="0.014 0.006 0.006" rgba="{rgba}" contype="0" conaffinity="0"/>
        <body name="{name}_y_link">
          <joint name="{name}_y" type="slide" axis="0 1 0" range="{_vec(y_range)}" damping="{damping:.8f}" armature="{armature:.8f}"/>
          <inertial pos="0 0 0" mass="0.05" diaginertia="0.00002 0.00002 0.00002"/>
          <geom name="{name}_y_carriage" type="box" size="0.006 0.014 0.006" rgba="{rgba}" contype="0" conaffinity="0"/>
          <body name="{name}_tool">
            <joint name="{name}_z" type="slide" axis="0 0 1" range="{_vec(z_range)}" damping="{damping:.8f}" armature="{armature:.8f}"/>
            <inertial pos="0 0 0" mass="0.08" diaginertia="0.00003 0.00003 0.00003"/>
            <geom name="{name}_palm" type="sphere" size="{grip_radius:.8f}" rgba="{rgba}" condim="4" friction="2.0 0.08 0.002"/>
            <geom name="{name}_finger_a" type="capsule" fromto="0 -0.018 0 0 -0.006 -0.006" size="0.0035" rgba="{rgba}" condim="4" friction="2.0 0.08 0.002"/>
            <geom name="{name}_finger_b" type="capsule" fromto="0 0.018 0 0 0.006 -0.006" size="0.0035" rgba="{rgba}" condim="4" friction="2.0 0.08 0.002"/>
            <site name="{name}_tip" pos="0 0 0" size="0.004" rgba="1 1 1 1"/>
          </body>
        </body>
      </body>
    </body>"""
    actuator = f"""    <position name="{name}_x_pos" joint="{name}_x" kp="{kp:.8f}" ctrllimited="true" ctrlrange="{_vec(x_range)}"/>
    <position name="{name}_y_pos" joint="{name}_y" kp="{kp:.8f}" ctrllimited="true" ctrlrange="{_vec(y_range)}"/>
    <position name="{name}_z_pos" joint="{name}_z" kp="{kp:.8f}" ctrllimited="true" ctrlrange="{_vec(z_range)}"/>"""
    return body, actuator


def build_mjcf(config: FoldClothConfig) -> str:
    cloth = config.cloth
    phys = config.physics
    gripper = config.gripper
    robot_cfg = config.robot or {}

    size_x, size_y = [float(v) for v in cloth["size"]]
    grid_x, grid_y = [int(v) for v in cloth["grid"]]
    if grid_x < 3 or grid_y < 3:
        raise ValueError("cloth.grid must be at least [3, 3]")

    spacing_x = size_x / (grid_x - 1)
    spacing_y = size_y / (grid_y - 1)
    cloth_z = max(float(cloth.get("radius", 0.004)) * 2.0, float(cloth.get("thickness", 0.0008)) * 6.0)

    y_fracs = robot_cfg.get("grasp_y_fraction", gripper.get("right_edge_y_fraction", [0.15, 0.85]))
    if len(y_fracs) != 2:
        raise ValueError("robot.grasp_y_fraction must contain exactly two values for the dual-gripper demo")
    anchors = [_edge_anchor(size_x, size_y, "right", frac, cloth_z) for frac in y_fracs]

    moving_rgba_values = gripper.get("moving_rgba", [0.95, 0.22, 0.10, 1.0])
    grip_radius = float(gripper.get("radius", 0.014))
    gravity = _vec(phys.get("gravity", [0.0, 0.0, -9.81]))
    table_friction = _vec(phys.get("table_friction", [1.6, 0.03, 0.0004]))
    cloth_friction = _vec(cloth.get("friction", [1.4, 0.02, 0.0003]))
    color = _vec(cloth.get("color", [0.10, 0.55, 0.85, 1.0]))
    iterations = int(phys.get("solver_iterations", 160))
    ls_iterations = int(phys.get("solver_ls_iterations", 50))
    edge_damping = float(cloth.get("edge_damping", 1.0))

    robot_bodies: list[str] = []
    actuators: list[str] = []
    equalities: list[str] = []
    for idx, anchor in enumerate(anchors):
        name = f"fold_robot_{idx}"
        rgba = _vec(moving_rgba_values[idx] if isinstance(moving_rgba_values[0], list) else moving_rgba_values)
        body, actuator = _cartesian_robot_xml(name, anchor, rgba, grip_radius, robot_cfg)
        robot_bodies.append(body)
        actuators.append(actuator)
        y_idx = _fraction_to_grid_index(float(y_fracs[idx]), grid_y)
        cloth_body = _cloth_vertex_body_name(grid_x - 1, y_idx, grid_y)
        equalities.append(
            f'    <weld name="{name}_grasp" body1="{name}_tool" body2="{cloth_body}" '
            'solref="0.003 1" solimp="0.90 0.98 0.001"/>'
        )

    return f"""<mujoco model="ttt_robot_fold_cloth">
  <compiler angle="radian"/>
  <option timestep="{config.timestep:.8f}" gravity="{gravity}" integrator="implicitfast" solver="CG" tolerance="1e-7" iterations="{iterations}" ls_iterations="{ls_iterations}">
    <flag energy="enable"/>
  </option>
  <size njmax="12000" nconmax="2500"/>
  <extension>
    <plugin plugin="mujoco.elasticity.shell"/>
  </extension>
  <default>
    <geom solref="0.004 1" solimp="0.9 0.98 0.001"/>
  </default>
  <visual>
    <headlight ambient="0.42 0.42 0.42" diffuse="0.55 0.55 0.55" specular="0.16 0.16 0.16"/>
    <quality shadowsize="2048" offsamples="4"/>
    <map znear="0.01" zfar="10"/>
  </visual>
  <asset>
    <texture name="table_checker" type="2d" builtin="checker" rgb1="0.78 0.78 0.74" rgb2="0.64 0.64 0.60" width="512" height="512"/>
    <material name="table_material" texture="table_checker" texrepeat="3 2" reflectance="0.01" shininess="0.0" specular="0.0"/>
  </asset>
  <worldbody>
    <light name="key" pos="0 -0.9 1.5" dir="0 0 -1" diffuse="0.85 0.85 0.85"/>
    <geom name="table" type="box" pos="0 0 -0.012" size="0.58 0.40 0.012" material="table_material" friction="{table_friction}"/>
    <camera name="overview" pos="0 -0.72 0.52" xyaxes="1 0 0 0 0.58 0.82" fovy="45"/>
{chr(10).join(robot_bodies)}
    <body name="cloth_parent" pos="0 0 {cloth_z:.8f}">
      <flexcomp name="cloth" type="grid" dim="2" count="{grid_x} {grid_y} 1" spacing="{spacing_x:.8f} {spacing_y:.8f} {max(spacing_x, spacing_y):.8f}" mass="{float(cloth.get("mass", 0.08)):.8f}" radius="{float(cloth.get("radius", 0.004)):.8f}" rgba="{color}">
        <edge equality="true" damping="{edge_damping:.8f}"/>
        <contact condim="4" selfcollide="none" solref="0.003" friction="{cloth_friction}"/>
        <plugin plugin="mujoco.elasticity.shell">
          <config key="poisson" value="{float(cloth.get("poisson", 0.42)):.8f}"/>
          <config key="thickness" value="{float(cloth.get("thickness", 0.0008)):.8f}"/>
          <config key="young" value="{float(cloth.get("young", 40000.0)):.8f}"/>
        </plugin>
      </flexcomp>
    </body>
  </worldbody>
  <actuator>
{chr(10).join(actuators)}
  </actuator>
  <equality>
{chr(10).join(equalities)}
  </equality>
</mujoco>
"""


class FoldClothEnv:
    def __init__(self, config: FoldClothConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(config))
        self.data = mujoco.MjData(self.model)
        self.camera_name = "overview"
        self._renderer: mujoco.Renderer | None = None
        self._robot_names = ["fold_robot_0", "fold_robot_1"]
        self._robot_base_pos = np.asarray(
            [self.model.body(f"{name}_base").pos.copy() for name in self._robot_names],
            dtype=np.float64,
        )
        self._robot_body_ids = [
            self.model.body(f"{name}_tool").id for name in self._robot_names
        ]
        self._robot_joint_ids = [
            self.model.joint(f"{name}_{axis}").id
            for name in self._robot_names
            for axis in ("x", "y", "z")
        ]
        self._robot_qpos_ids = np.asarray(
            [self.model.jnt_qposadr[joint_id] for joint_id in self._robot_joint_ids],
            dtype=np.int64,
        )
        self._robot_qvel_ids = np.asarray(
            [self.model.jnt_dofadr[joint_id] for joint_id in self._robot_joint_ids],
            dtype=np.int64,
        )
        self._robot_actuator_ids = [
            self.model.actuator(f"{name}_{axis}_pos").id
            for name in self._robot_names
            for axis in ("x", "y", "z")
        ]
        self._grasp_eq_ids = [
            self.model.eq(f"{name}_grasp").id for name in self._robot_names
        ]
        self._cloth_body_ids = np.asarray(
            [
                body_id
                for body_id in range(self.model.nbody)
                if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "").startswith("cloth_")
                and (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "")[6:].isdigit()
            ],
            dtype=np.int64,
        )
        self._initial_targets = self._robot_base_pos.copy()
        self._waypoints = _coerce_waypoints(config, float(self._robot_base_pos[0, 2]))
        self._release_time = float((config.trajectory or {}).get("release_time", config.duration + 1.0))

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _renderer_instance(self) -> mujoco.Renderer:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.config.image_height,
                width=self.config.image_width,
            )
        return self._renderer

    def _targets_for_time(self, time_s: float) -> np.ndarray:
        waypoint = _interpolate_waypoints(self._waypoints, time_s)
        targets = self._initial_targets.copy()
        targets[:, 0] = waypoint["x"]
        targets[:, 1] += waypoint.get("y_offset", 0.0)
        targets[:, 2] = waypoint["z"]
        return targets

    def _set_robot_targets(self, targets: np.ndarray) -> None:
        q_targets = (targets - self._robot_base_pos).reshape(-1)
        for actuator_id, value in zip(self._robot_actuator_ids, q_targets):
            ctrl_range = self.model.actuator_ctrlrange[actuator_id]
            self.data.ctrl[actuator_id] = float(np.clip(value, ctrl_range[0], ctrl_range[1]))

    def _set_grasp_active(self, active: bool) -> None:
        value = 1 if active else 0
        for eq_id in self._grasp_eq_ids:
            self.data.eq_active[eq_id] = value

    def reset(self) -> dict[str, np.ndarray]:
        mujoco.mj_resetData(self.model, self.data)
        self._set_grasp_active(True)
        self._set_robot_targets(self._initial_targets)
        mujoco.mj_forward(self.model, self.data)
        settle_steps = int(self.config.physics.get("reset_settle_steps", 120))
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)
        return self.observe(self._initial_targets, True)

    def step(self, time_s: float) -> dict[str, np.ndarray]:
        targets = self._targets_for_time(time_s)
        grasp_active = time_s < self._release_time
        self._set_grasp_active(grasp_active)
        self._set_robot_targets(targets)
        inner_steps = max(1, round(self.config.control_dt / self.config.timestep))
        for _ in range(inner_steps):
            mujoco.mj_step(self.model, self.data)
        return self.observe(targets, grasp_active)

    def observe(self, target_pos: np.ndarray, grasp_active: bool) -> dict[str, np.ndarray]:
        renderer = self._renderer_instance()
        renderer.update_scene(self.data, camera=self.camera_name)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(self.data, camera=self.camera_name)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()

        return {
            "rgb": rgb,
            "depth": depth,
            "sim_qpos": self.data.qpos.copy(),
            "sim_qvel": self.data.qvel.copy(),
            "robot_qpos": self.data.qpos[self._robot_qpos_ids].copy(),
            "robot_qvel": self.data.qvel[self._robot_qvel_ids].copy(),
            "robot_ee_pos": self.data.xpos[self._robot_body_ids].copy(),
            "robot_target_pos": np.asarray(target_pos, dtype=np.float64).copy(),
            "cloth_vertex_pos": self.data.xpos[self._cloth_body_ids].copy(),
            "grasp_active": np.asarray([1.0 if grasp_active else 0.0], dtype=np.float32),
        }

    def rollout(self) -> dict[str, np.ndarray]:
        self.reset()
        n_steps = int(round(self.config.duration / self.config.control_dt))

        buffers: dict[str, list[np.ndarray]] = {
            "rgb": [],
            "depth": [],
            "sim_qpos": [],
            "sim_qvel": [],
            "robot_qpos": [],
            "robot_qvel": [],
            "robot_ee_pos": [],
            "robot_target_pos": [],
            "cloth_vertex_pos": [],
            "grasp_active": [],
        }
        for step_idx in range(n_steps):
            obs = self.step(step_idx * self.config.control_dt)
            for key, value in obs.items():
                buffers[key].append(value)

        return {
            "rgb": np.asarray(buffers["rgb"], dtype=np.uint8),
            "depth": np.asarray(buffers["depth"], dtype=np.float32),
            "sim_qpos": np.asarray(buffers["sim_qpos"], dtype=np.float32),
            "sim_qvel": np.asarray(buffers["sim_qvel"], dtype=np.float32),
            "robot_qpos": np.asarray(buffers["robot_qpos"], dtype=np.float32),
            "robot_qvel": np.asarray(buffers["robot_qvel"], dtype=np.float32),
            "robot_ee_pos": np.asarray(buffers["robot_ee_pos"], dtype=np.float32),
            "robot_target_pos": np.asarray(buffers["robot_target_pos"], dtype=np.float32),
            "cloth_vertex_pos": np.asarray(buffers["cloth_vertex_pos"], dtype=np.float32),
            "grasp_active": np.asarray(buffers["grasp_active"], dtype=np.float32),
        }


def save_rollout(
    rollout: dict[str, np.ndarray],
    output_dir: str | Path,
    config: FoldClothConfig,
) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    h5_path = out / "episode0.hdf5"
    video_path = out / "episode0.mp4"

    with h5py.File(h5_path, "w") as f:
        obs = f.create_group("observations")
        obs.create_dataset("rgb", data=rollout["rgb"], compression="gzip", compression_opts=4)
        obs.create_dataset("depth", data=rollout["depth"], compression="gzip", compression_opts=4)
        obs.create_dataset("sim_qpos", data=rollout["sim_qpos"], compression="gzip", compression_opts=4)
        obs.create_dataset("sim_qvel", data=rollout["sim_qvel"], compression="gzip", compression_opts=4)
        obs.create_dataset("robot_qpos", data=rollout["robot_qpos"], compression="gzip", compression_opts=4)
        obs.create_dataset("robot_qvel", data=rollout["robot_qvel"], compression="gzip", compression_opts=4)
        obs.create_dataset("robot_ee_pos", data=rollout["robot_ee_pos"], compression="gzip", compression_opts=4)
        obs.create_dataset("cloth_vertex_pos", data=rollout["cloth_vertex_pos"], compression="gzip", compression_opts=4)
        actions = f.create_group("actions")
        actions.create_dataset("robot_target_pos", data=rollout["robot_target_pos"], compression="gzip", compression_opts=4)
        actions.create_dataset("grasp_active", data=rollout["grasp_active"], compression="gzip", compression_opts=4)
        meta = f.create_group("metadata")
        meta.attrs["task_name"] = (config.output or {}).get("task_name", "fold_cloth")
        meta.attrs["fps"] = int(config.fps)
        meta.attrs["duration"] = float(config.duration)
        meta.attrs["control_dt"] = float(config.control_dt)
        meta.attrs["timestep"] = float(config.timestep)
        meta.attrs["reset_settle_steps"] = int(config.physics.get("reset_settle_steps", 120))
        meta.attrs["cloth_size"] = np.asarray(config.cloth["size"], dtype=np.float32)
        meta.attrs["cloth_grid"] = np.asarray(config.cloth["grid"], dtype=np.int32)
        meta.attrs["cloth_mass"] = float(config.cloth["mass"])
        meta.attrs["cloth_thickness"] = float(config.cloth["thickness"])
        meta.attrs["cloth_young"] = float(config.cloth["young"])
        meta.attrs["cloth_poisson"] = float(config.cloth["poisson"])
        meta.attrs["cloth_shell_plugin"] = "mujoco.elasticity.shell"
        meta.attrs["robot_type"] = "dual_position_actuated_cartesian_end_effectors"

    imageio.mimsave(video_path, list(rollout["rgb"]), fps=config.fps)
    return h5_path, video_path
