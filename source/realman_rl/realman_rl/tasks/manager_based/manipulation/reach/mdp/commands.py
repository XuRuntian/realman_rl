# realman_rl/tasks/manager_based/manipulation/reach/mdp/commands.py

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence
from dataclasses import MISSING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # 默认的坐标轴标记
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_from_euler_xyz,
    sample_uniform,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class UniformPoseCommand(CommandTerm):
    """
    通用姿态指令生成器 (Uniform Pose Command Generator).

    功能：
    1. 在指定的笛卡尔空间范围 (ranges) 内随机采样目标位置 (x, y, z)。
    2. 在指定的欧拉角范围 (ranges) 内随机采样目标姿态 (roll, pitch, yaw)。
    3. 支持可视化目标点。

    这个类非常适合 Reach（到达）、Push（推）、Pick（抓）等任务的目标生成。
    """

    cfg: UniformPoseCommandCfg

    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedRLEnv):
        # 初始化父类
        super().__init__(cfg, env)

        # 1. 定义指令缓冲区 (Buffer)
        # command 格式通常为: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z] (7维)
        # 或者简化为 [pos_x, pos_y, pos_z] (3维)，取决于 cfg.resampling_time_range
        # 这里我们统一存储 7 维 (Pos + Quat)，具体 Observation 怎么用由 observation term 决定
        self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
        
        # 2. 解析配置范围
        # 将配置字典转换为 Tensor 方便计算
        self.pos_ranges = torch.zeros(self.num_envs, 3, 2, device=self.device)
        self.pos_ranges[:, 0, :] = torch.tensor(self.cfg.ranges.pos_x, device=self.device)
        self.pos_ranges[:, 1, :] = torch.tensor(self.cfg.ranges.pos_y, device=self.device)
        self.pos_ranges[:, 2, :] = torch.tensor(self.cfg.ranges.pos_z, device=self.device)

        self.rot_ranges = torch.zeros(self.num_envs, 3, 2, device=self.device)
        self.rot_ranges[:, 0, :] = torch.tensor(self.cfg.ranges.roll, device=self.device)
        self.rot_ranges[:, 1, :] = torch.tensor(self.cfg.ranges.pitch, device=self.device)
        self.rot_ranges[:, 2, :] = torch.tensor(self.cfg.ranges.yaw, device=self.device)

        # 3. 初始化 Metrics (可选，用于记录调试信息)
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_rot"] = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        """
        核心逻辑：当环境 Reset 或倒计时结束时，重新生成目标。
        """
        # 1. 采样位置 (Position)
        # 在 [min, max] 范围内均匀采样
        r = torch.rand(len(env_ids), 3, device=self.device)
        pos_random = (self.pos_ranges[env_ids, :, 1] - self.pos_ranges[env_ids, :, 0]) * r + self.pos_ranges[env_ids, :, 0]

        # 2. 采样姿态 (Orientation)
        # 同样在 Euler 角度范围内采样，然后转为四元数
        r_rot = torch.rand(len(env_ids), 3, device=self.device)
        euler_random = (self.rot_ranges[env_ids, :, 1] - self.rot_ranges[env_ids, :, 0]) * r_rot + self.rot_ranges[env_ids, :, 0]
        quat_random = quat_from_euler_xyz(euler_random[:, 0], euler_random[:, 1], euler_random[:, 2])

        # 3. 处理参考系 (可选)
        # 如果你想让生成的点是相对于机器人的 (比如在机器人前方 0.5m)，需要结合 env origins
        # Isaac Lab 的 Command 通常是 Global Frame 下的
        # 这里我们假设 ranges 是相对于 env_origins 的偏移量
        self.pose_command_w[env_ids, :3] = pos_random + self._env.scene.env_origins[env_ids]
        self.pose_command_w[env_ids, 3:] = quat_random

    def _update_command(self):
        """
        每一步仿真都会调用。通常用于移动目标。
        对于静态 Reach 任务，这里不需要做任何事，只需保持 command 不变。
        """
        pass

    @property
    def command(self) -> torch.Tensor:
        """
        暴露给外部 (Observation/Reward) 使用的接口。
        返回完整的 [x, y, z, qw, qx, qy, qz]
        """
        return self.pose_command_w

    def _set_debug_vis_impl(self, debug_vis: bool):
        """
        处理可视化 (Debug Visualization)。
        """
        # 如果需要显示，且没有初始化过 visualizer
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                # 使用配置中的 marker 配置创建 visualizer
                self.goal_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """
        渲染循环的回调，用于更新 Marker 的位置。
        """
        if hasattr(self, "goal_visualizer"):
            # 将 Marker 移动到当前的 command 位置
            self.goal_visualizer.visualize(
                self.pose_command_w[:, :3], 
                self.pose_command_w[:, 3:]
            )


@configclass
class UniformPoseCommandCfg(CommandTermCfg):
    """Configuration for the uniform pose command generator."""
    
    class_type: type = UniformPoseCommand

    # 1. 定义采样范围的数据结构
    @configclass
    class Ranges:
        # 默认范围 (相对于环境原点)
        pos_x: tuple[float, float] = (0.3, 0.6)   # 机器人前方 0.3 到 0.6 米
        pos_y: tuple[float, float] = (-0.3, 0.3)  # 左右各 0.3 米
        pos_z: tuple[float, float] = (0.1, 0.5)   # 高度 0.1 到 0.5 米
        
        # 欧拉角范围 (弧度)
        roll: tuple[float, float] = (0.0, 0.0)    # 保持水平
        pitch: tuple[float, float] = (0.0, 0.0)
        yaw: tuple[float, float] = (-3.14, 3.14)  # 任意旋转

    # 将上面的结构实例化
    ranges: Ranges = Ranges()

    # 2. 可视化配置
    # 默认使用坐标轴显示目标点，prim_path 必须唯一
    visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/target_pose"
    )
    # 调整 Marker 大小
    visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    # 3. 基础配置 (继承自 CommandTermCfg)
    # resample_frequency_range: 重新采样的频率 (通常 Reach 任务不需要中途变，除非做 Tracking)
    # 对于 Episodic 任务，通常设置为无穷大或仅在 reset 时触发