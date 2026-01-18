# source/realman_rl/realman_rl/tasks/manager_based/manipulation/reach/mdp/commands.py

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence
from dataclasses import MISSING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_from_euler_xyz,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class UniformPoseCommand(CommandTerm):
    """
    通用姿态指令生成器 (Uniform Pose Command Generator).
    """

    cfg: UniformPoseCommandCfg

    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # 1. 初始化 Buffer
        self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
        
        # 2. 解析范围
        self.pos_ranges = torch.zeros(self.num_envs, 3, 2, device=self.device)
        self.pos_ranges[:, 0, :] = torch.tensor(self.cfg.ranges.pos_x, device=self.device)
        self.pos_ranges[:, 1, :] = torch.tensor(self.cfg.ranges.pos_y, device=self.device)
        self.pos_ranges[:, 2, :] = torch.tensor(self.cfg.ranges.pos_z, device=self.device)

        self.rot_ranges = torch.zeros(self.num_envs, 3, 2, device=self.device)
        self.rot_ranges[:, 0, :] = torch.tensor(self.cfg.ranges.roll, device=self.device)
        self.rot_ranges[:, 1, :] = torch.tensor(self.cfg.ranges.pitch, device=self.device)
        self.rot_ranges[:, 2, :] = torch.tensor(self.cfg.ranges.yaw, device=self.device)

        # [可选] 初始化 Metrics
        # self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        # 1. 采样位置
        r = torch.rand(len(env_ids), 3, device=self.device)
        pos_random = (self.pos_ranges[env_ids, :, 1] - self.pos_ranges[env_ids, :, 0]) * r + self.pos_ranges[env_ids, :, 0]

        # 2. 采样姿态
        r_rot = torch.rand(len(env_ids), 3, device=self.device)
        euler_random = (self.rot_ranges[env_ids, :, 1] - self.rot_ranges[env_ids, :, 0]) * r_rot + self.rot_ranges[env_ids, :, 0]
        quat_random = quat_from_euler_xyz(euler_random[:, 0], euler_random[:, 1], euler_random[:, 2])

        # 3. 设置 Command (World Frame)
        self.pose_command_w[env_ids, :3] = pos_random + self._env.scene.env_origins[env_ids]
        self.pose_command_w[env_ids, 3:] = quat_random

    def _update_command(self):
        pass
    
    # 【核心修复】必须实现这个方法，否则会报 TypeError: abstract method
    def _update_metrics(self):
        """Update metrics."""
        # 这里通常用来计算 command 相关的 debug 信息
        # 暂时留空即可，因为主要的 Reward 计算在 rewards.py 里
        pass

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_w

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if hasattr(self, "goal_visualizer"):
            self.goal_visualizer.visualize(
                self.pose_command_w[:, :3], 
                self.pose_command_w[:, 3:]
            )


@configclass
class UniformPoseCommandCfg(CommandTermCfg):
    """Configuration for the uniform pose command generator."""
    
    class_type: type = UniformPoseCommand

    asset_name: str = MISSING
    body_names: list[str] = MISSING

    @configclass
    class Ranges:
        pos_x: tuple[float, float] = (0.3, 0.6)
        pos_y: tuple[float, float] = (-0.3, 0.3)
        pos_z: tuple[float, float] = (0.1, 0.5)
        roll: tuple[float, float] = (0.0, 0.0)
        pitch: tuple[float, float] = (0.0, 0.0)
        yaw: tuple[float, float] = (-3.14, 3.14)

    ranges: Ranges = Ranges()

    visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/target_pose"
    )
    visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)