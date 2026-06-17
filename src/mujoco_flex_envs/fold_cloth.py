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
    output: dict[str, Any] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FoldClothConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)


def _vec(values: list[float] | tuple[float, ...], precision: int = 6) -> str:
    return " ".join(f"{float(v):.{precision}g}" for v in values)


def _fraction_to_grid_index(fraction: float, count: int) -> int:
    return int(np.clip(round(float(fraction) * (count - 1)), 0, count - 1))


def _cloth_vertex_body_name(x_idx: int, y_idx: int, grid_y: int) -> str:
    return f"cloth_{x_idx * grid_y + y_idx}"


def _edge_anchor(size_x: float, size_y: float, x_side: str, y_fraction: float, z: float) -> np.ndarray:
    x = -0.5 * size_x if x_side == "left" else 0.5 * size_x
    y = -0.5 * size_y + float(y_fraction) * size_y
    return np.array([x, y, z], dtype=np.float64)


def build_mjcf(config: FoldClothConfig) -> str:
    cloth = config.cloth
    phys = config.physics
    gripper = config.gripper

    size_x, size_y = [float(v) for v in cloth["size"]]
    grid_x, grid_y = [int(v) for v in cloth["grid"]]
    if grid_x < 3 or grid_y < 3:
        raise ValueError("cloth.grid must be at least [3, 3]")
    spacing_x = size_x / (grid_x - 1)
    spacing_y = size_y / (grid_y - 1)
    cloth_z = max(float(cloth.get("radius", 0.0035)) * 2.0, float(cloth.get("thickness", 0.004)) * 1.5)

    right_fracs = gripper.get("right_edge_y_fraction", [0.25, 0.75])
    left_fracs = gripper.get("left_edge_y_fraction", [0.25, 0.75])
    right_anchors = [_edge_anchor(size_x, size_y, "right", frac, cloth_z) for frac in right_fracs]
    left_anchors = [_edge_anchor(size_x, size_y, "left", frac, cloth_z) for frac in left_fracs]

    moving_rgba = _vec(gripper.get("moving_rgba", [0.95, 0.22, 0.10, 1.0]))
    holder_rgba = _vec(gripper.get("holder_rgba", [0.15, 0.15, 0.15, 1.0]))
    grip_radius = float(gripper.get("radius", 0.018))
    gravity = _vec(phys.get("gravity", [0.0, 0.0, -9.81]))
    table_friction = _vec(phys.get("table_friction", [1.4, 0.02, 0.0002]))
    cloth_friction = _vec(cloth.get("friction", [1.3, 0.02, 0.0002]))
    color = _vec(cloth.get("color", [0.10, 0.55, 0.85, 1.0]))
    iterations = int(phys.get("solver_iterations", 120))
    ls_iterations = int(phys.get("solver_ls_iterations", 30))

    mocap_bodies = []
    equalities = []

    for idx, anchor in enumerate(right_anchors):
        name = f"fold_gripper_{idx}"
        mocap_bodies.append(
            f'    <body name="{name}" mocap="true" pos="{_vec(anchor)}">\n'
            f'      <geom name="{name}_geom" type="sphere" size="{grip_radius:.8f}" rgba="{moving_rgba}" '
            'condim="4" friction="2.0 0.08 0.002"/>\n'
            "    </body>"
        )
        y_idx = _fraction_to_grid_index(right_fracs[idx], grid_y)
        cloth_body = _cloth_vertex_body_name(grid_x - 1, y_idx, grid_y)
        equalities.append(
            f'    <connect name="{name}_pin" body1="{name}" body2="{cloth_body}" '
            f'anchor="{_vec(anchor)}" solref="0.003 1"/>'
        )

    for idx, anchor in enumerate(left_anchors):
        name = f"fold_holder_{idx}"
        mocap_bodies.append(
            f'    <body name="{name}" mocap="true" pos="{_vec(anchor)}">\n'
            f'      <geom name="{name}_geom" type="sphere" size="{grip_radius:.8f}" rgba="{holder_rgba}" '
            'condim="4" friction="2.0 0.08 0.002"/>\n'
            "    </body>"
        )
        y_idx = _fraction_to_grid_index(left_fracs[idx], grid_y)
        cloth_body = _cloth_vertex_body_name(0, y_idx, grid_y)
        equalities.append(
            f'    <connect name="{name}_pin" body1="{name}" body2="{cloth_body}" '
            f'anchor="{_vec(anchor)}" solref="0.003 1"/>'
        )

    return f"""<mujoco model="ttt_fold_cloth">
  <compiler angle="radian"/>
  <option timestep="{config.timestep:.8f}" gravity="{gravity}" iterations="{iterations}" ls_iterations="{ls_iterations}" integrator="implicitfast"/>
  <size njmax="8000" nconmax="1600"/>
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
    <camera name="overview" pos="0 -0.72 0.52" xyaxes="1 0 0 0 0.58 0.82" fovy="45"/>
{chr(10).join(mocap_bodies)}
    <body name="cloth_parent" pos="0 0 {cloth_z:.8f}">
      <flexcomp name="cloth" type="grid" dim="2" count="{grid_x} {grid_y} 1" spacing="{spacing_x:.8f} {spacing_y:.8f} {max(spacing_x, spacing_y):.8f}" mass="{float(cloth.get("mass", 0.08)):.8f}" radius="{float(cloth.get("radius", 0.0035)):.8f}" rgba="{color}">
        <contact condim="4" selfcollide="none" friction="{cloth_friction}"/>
        <elasticity young="{float(cloth.get("young", 900.0)):.8f}" poisson="{float(cloth.get("poisson", 0.3)):.8f}" damping="{float(cloth.get("damping", 0.025)):.8f}" thickness="{float(cloth.get("thickness", 0.004)):.8f}"/>
      </flexcomp>
    </body>
  </worldbody>
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
        self._moving_mocap_ids = [
            self.model.body(f"fold_gripper_{idx}").mocapid[0]
            for idx in range(len(config.gripper.get("right_edge_y_fraction", [0.25, 0.75])))
        ]
        self._holder_mocap_ids = [
            self.model.body(f"fold_holder_{idx}").mocapid[0]
            for idx in range(len(config.gripper.get("left_edge_y_fraction", [0.25, 0.75])))
        ]
        self._initial_mocap_pos = self.model.body_mocapid.copy()

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

    def _edge_y_offsets(self) -> list[float]:
        _, size_y = [float(v) for v in self.config.cloth["size"]]
        return [
            -0.5 * size_y + float(frac) * size_y
            for frac in self.config.gripper.get("right_edge_y_fraction", [0.25, 0.75])
        ]

    def _path_target(self, fraction: float) -> list[np.ndarray]:
        path = self.config.gripper["path"]
        if len(path) == 1:
            point = path[0]
            return [np.array([float(point["right_x"]), y, float(point["z"])]) for y in self._edge_y_offsets()]

        fraction = float(np.clip(fraction, 0.0, 1.0))
        scaled = fraction * (len(path) - 1)
        idx = min(int(np.floor(scaled)), len(path) - 2)
        local = scaled - idx

        p0 = path[idx]
        p1 = path[idx + 1]
        x = (1.0 - local) * float(p0["right_x"]) + local * float(p1["right_x"])
        z = (1.0 - local) * float(p0["z"]) + local * float(p1["z"])
        return [np.array([x, y, z], dtype=np.float64) for y in self._edge_y_offsets()]

    def reset(self) -> dict[str, np.ndarray]:
        mujoco.mj_resetData(self.model, self.data)
        for mocap_id in range(self.model.nmocap):
            body_id = np.flatnonzero(self.model.body_mocapid == mocap_id)[0]
            self.data.mocap_pos[mocap_id] = self.model.body_pos[body_id]
            self.data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        return self.observe()

    def step(self, moving_targets: list[np.ndarray]) -> dict[str, np.ndarray]:
        for mocap_id, target in zip(self._moving_mocap_ids, moving_targets):
            self.data.mocap_pos[mocap_id] = target
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
            "moving_gripper_pos": self.data.mocap_pos[self._moving_mocap_ids].copy(),
            "holder_pos": self.data.mocap_pos[self._holder_mocap_ids].copy(),
        }

    def rollout(self) -> dict[str, np.ndarray]:
        self.reset()
        n_steps = int(round(self.config.duration / self.config.control_dt))

        rgb, depth, qpos, qvel, moving_pos, holder_pos = [], [], [], [], [], []
        for step_idx in range(n_steps):
            target = self._path_target(step_idx / max(1, n_steps - 1))
            obs = self.step(target)
            rgb.append(obs["rgb"])
            depth.append(obs["depth"])
            qpos.append(obs["qpos"])
            qvel.append(obs["qvel"])
            moving_pos.append(obs["moving_gripper_pos"])
            holder_pos.append(obs["holder_pos"])

        return {
            "rgb": np.asarray(rgb, dtype=np.uint8),
            "depth": np.asarray(depth, dtype=np.float32),
            "qpos": np.asarray(qpos, dtype=np.float32),
            "qvel": np.asarray(qvel, dtype=np.float32),
            "moving_gripper_pos": np.asarray(moving_pos, dtype=np.float32),
            "holder_pos": np.asarray(holder_pos, dtype=np.float32),
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
        obs.create_dataset("cloth_qpos", data=rollout["qpos"], compression="gzip", compression_opts=4)
        obs.create_dataset("cloth_qvel", data=rollout["qvel"], compression="gzip", compression_opts=4)
        actions = f.create_group("actions")
        actions.create_dataset("moving_gripper_pos", data=rollout["moving_gripper_pos"], compression="gzip", compression_opts=4)
        actions.create_dataset("holder_pos", data=rollout["holder_pos"], compression="gzip", compression_opts=4)
        meta = f.create_group("metadata")
        meta.attrs["task_name"] = (config.output or {}).get("task_name", "fold_cloth")
        meta.attrs["fps"] = int(config.fps)
        meta.attrs["duration"] = float(config.duration)
        meta.attrs["control_dt"] = float(config.control_dt)
        meta.attrs["timestep"] = float(config.timestep)
        meta.attrs["cloth_size"] = np.asarray(config.cloth["size"], dtype=np.float32)
        meta.attrs["cloth_grid"] = np.asarray(config.cloth["grid"], dtype=np.int32)
        meta.attrs["cloth_mass"] = float(config.cloth["mass"])
        meta.attrs["cloth_thickness"] = float(config.cloth["thickness"])
        meta.attrs["cloth_young"] = float(config.cloth["young"])
        meta.attrs["cloth_poisson"] = float(config.cloth["poisson"])
        meta.attrs["cloth_damping"] = float(config.cloth["damping"])

    imageio.mimsave(video_path, list(rollout["rgb"]), fps=config.fps)
    return h5_path, video_path

