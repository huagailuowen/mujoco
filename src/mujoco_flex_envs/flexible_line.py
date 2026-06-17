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
class FlexibleLineConfig:
    seed: int
    duration: float
    timestep: float
    control_dt: float
    image_width: int
    image_height: int
    fps: int
    physics: dict[str, Any]
    line: dict[str, Any]
    gripper: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FlexibleLineConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)


def _vec(values: list[float] | tuple[float, ...], precision: int = 6) -> str:
    return " ".join(f"{float(v):.{precision}g}" for v in values)


def _build_chain_body(
    segment_idx: int,
    segment_count: int,
    segment_len: float,
    radius: float,
    line_cfg: dict[str, Any],
    indent: int,
) -> str:
    space = " " * indent
    rgba = _vec(line_cfg.get("color", [0.08, 0.55, 0.88, 1.0]))
    density = float(line_cfg.get("density", 650.0))
    stiffness = float(line_cfg.get("joint_stiffness", 0.08))
    damping = float(line_cfg.get("damping", 0.03))
    friction = float(line_cfg.get("joint_friction", 0.015))
    bend = float(line_cfg.get("bend_stiffness", stiffness))

    lines = [
        f'{space}<body name="line_seg_{segment_idx}" pos="{segment_len:.8f} 0 0">',
        (
            f'{space}  <joint name="line_joint_{segment_idx}" type="ball" '
            f'stiffness="{stiffness + bend:.8f}" damping="{damping:.8f}" '
            f'frictionloss="{friction:.8f}" armature="0.00002"/>'
        ),
        (
            f'{space}  <geom name="line_geom_{segment_idx}" type="capsule" '
            f'fromto="0 0 0 {segment_len:.8f} 0 0" size="{radius:.8f}" '
            f'density="{density:.8f}" rgba="{rgba}" condim="4" '
            'friction="1.2 0.015 0.0002"/>'
        ),
    ]

    if segment_idx + 1 < segment_count:
        lines.append(
            _build_chain_body(
                segment_idx + 1,
                segment_count,
                segment_len,
                radius,
                line_cfg,
                indent + 2,
            )
        )

    lines.append(f"{space}</body>")
    return "\n".join(lines)


def build_mjcf(config: FlexibleLineConfig) -> str:
    line_cfg = config.line
    phys = config.physics
    gripper = config.gripper

    length = float(line_cfg["length"])
    radius = float(line_cfg["radius"])
    segments = int(line_cfg["segments"])
    if segments < 3:
        raise ValueError("line.segments must be >= 3")
    segment_len = length / segments
    z0 = radius + 0.015
    root_x = -0.5 * length

    gravity = _vec(phys.get("gravity", [0.0, 0.0, -9.81]))
    table_friction = _vec(phys.get("table_friction", [1.0, 0.005, 0.0001]))
    iterations = int(phys.get("solver_iterations", 80))
    ls_iterations = int(phys.get("solver_ls_iterations", 20))
    gripper_start = _vec(gripper["path"][0])
    gripper_radius = float(gripper.get("radius", 0.025))
    gripper_rgba = _vec(gripper.get("rgba", [0.95, 0.25, 0.12, 1.0]))

    child_chain = _build_chain_body(
        1,
        segments,
        segment_len,
        radius,
        line_cfg,
        indent=8,
    )

    return f"""<mujoco model="ttt_flexible_line">
  <compiler angle="radian"/>
  <option timestep="{config.timestep:.8f}" gravity="{gravity}" iterations="{iterations}" ls_iterations="{ls_iterations}" integrator="implicitfast"/>
  <size njmax="4000" nconmax="800"/>
  <default>
    <geom solref="0.004 1" solimp="0.9 0.98 0.001"/>
  </default>
  <visual>
    <quality shadowsize="2048" offsamples="4"/>
    <map znear="0.01" zfar="10"/>
  </visual>
  <worldbody>
    <light name="key" pos="0 -0.8 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="table" type="box" pos="0 0 -0.012" size="0.55 0.38 0.012" rgba="0.82 0.82 0.78 1" friction="{table_friction}"/>
    <camera name="overview" pos="0 -0.72 0.48" xyaxes="1 0 0 0 0.56 0.83" fovy="46"/>
    <body name="gripper_mocap" mocap="true" pos="{gripper_start}">
      <geom name="gripper_contact" type="sphere" size="{gripper_radius:.8f}" rgba="{gripper_rgba}" condim="4" friction="2.0 0.08 0.002"/>
    </body>
    <body name="line_seg_0" pos="{root_x:.8f} -0.02 {z0:.8f}">
      <freejoint name="line_root_free"/>
      <geom name="line_geom_0" type="capsule" fromto="0 0 0 {segment_len:.8f} 0 0" size="{radius:.8f}" density="{float(line_cfg.get("density", 650.0)):.8f}" rgba="{_vec(line_cfg.get("color", [0.08, 0.55, 0.88, 1.0]))}" condim="4" friction="1.2 0.015 0.0002"/>
{child_chain}
    </body>
  </worldbody>
</mujoco>
"""


class FlexibleLineEnv:
    def __init__(self, config: FlexibleLineConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(config))
        self.data = mujoco.MjData(self.model)
        self.camera_name = "overview"
        self._gripper_mocap_id = 0
        self._renderer: mujoco.Renderer | None = None

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

    def reset(self) -> dict[str, np.ndarray]:
        mujoco.mj_resetData(self.model, self.data)
        self.data.mocap_pos[self._gripper_mocap_id] = np.array(self.config.gripper["path"][0], dtype=np.float64)
        self.data.mocap_quat[self._gripper_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        return self.observe()

    def _target_at(self, fraction: float) -> np.ndarray:
        points = np.array(self.config.gripper["path"], dtype=np.float64)
        if len(points) == 1:
            return points[0]

        fraction = float(np.clip(fraction, 0.0, 1.0))
        scaled = fraction * (len(points) - 1)
        idx = min(int(np.floor(scaled)), len(points) - 2)
        local = scaled - idx
        return (1.0 - local) * points[idx] + local * points[idx + 1]

    def step(self, target_pos: np.ndarray) -> dict[str, np.ndarray]:
        self.data.mocap_pos[self._gripper_mocap_id] = target_pos
        inner_steps = max(1, round(self.config.control_dt / self.config.timestep))
        for _ in range(inner_steps):
            mujoco.mj_step(self.model, self.data)
        return self.observe()

    def observe(self) -> dict[str, np.ndarray]:
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
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "gripper_pos": self.data.mocap_pos[self._gripper_mocap_id].copy(),
        }

    def rollout(self) -> dict[str, np.ndarray]:
        self.reset()
        n_steps = int(round(self.config.duration / self.config.control_dt))

        rgb, depth, qpos, qvel, gripper_pos = [], [], [], [], []
        for step_idx in range(n_steps):
            target = self._target_at(step_idx / max(1, n_steps - 1))
            obs = self.step(target)
            rgb.append(obs["rgb"])
            depth.append(obs["depth"])
            qpos.append(obs["qpos"])
            qvel.append(obs["qvel"])
            gripper_pos.append(obs["gripper_pos"])

        return {
            "rgb": np.asarray(rgb, dtype=np.uint8),
            "depth": np.asarray(depth, dtype=np.float32),
            "qpos": np.asarray(qpos, dtype=np.float32),
            "qvel": np.asarray(qvel, dtype=np.float32),
            "gripper_pos": np.asarray(gripper_pos, dtype=np.float32),
        }


def save_rollout(
    rollout: dict[str, np.ndarray],
    output_dir: str | Path,
    config: FlexibleLineConfig,
) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    h5_path = out / "episode0.hdf5"
    video_path = out / "episode0.mp4"

    with h5py.File(h5_path, "w") as f:
        obs = f.create_group("observations")
        obs.create_dataset("rgb", data=rollout["rgb"], compression="gzip", compression_opts=4)
        obs.create_dataset("depth", data=rollout["depth"], compression="gzip", compression_opts=4)
        obs.create_dataset("cable_qpos", data=rollout["qpos"], compression="gzip", compression_opts=4)
        obs.create_dataset("cable_qvel", data=rollout["qvel"], compression="gzip", compression_opts=4)
        actions = f.create_group("actions")
        actions.create_dataset("gripper_pos", data=rollout["gripper_pos"], compression="gzip", compression_opts=4)
        meta = f.create_group("metadata")
        meta.attrs["fps"] = int(config.fps)
        meta.attrs["duration"] = float(config.duration)
        meta.attrs["control_dt"] = float(config.control_dt)
        meta.attrs["timestep"] = float(config.timestep)
        meta.attrs["line_length"] = float(config.line["length"])
        meta.attrs["line_radius"] = float(config.line["radius"])
        meta.attrs["line_segments"] = int(config.line["segments"])
        meta.attrs["line_density"] = float(config.line["density"])
        meta.attrs["line_damping"] = float(config.line["damping"])
        meta.attrs["line_joint_stiffness"] = float(config.line["joint_stiffness"])
        meta.attrs["line_joint_friction"] = float(config.line["joint_friction"])
        meta.attrs["line_bend_stiffness"] = float(config.line["bend_stiffness"])

    imageio.mimsave(video_path, list(rollout["rgb"]), fps=config.fps)
    return h5_path, video_path

