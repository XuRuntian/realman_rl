# realman_rl/tasks/manager_based/manipulation/reach/mdp/rewards.py

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude, combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# =============================================================================
# 辅助函数 (与 observations.py 类似，用于获取目标)
# =============================================================================

def _get_target_pose_w(env: ManagerBasedRLEnv, command_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """从 Command Manager 获取目标的世界坐标 (位置, 姿态)。"""
    command = env.command_manager.get_command(command_name)
    target_pos = command[:, :3]
    target_rot = command[:, 3:7]
    return target_pos, target_rot

# =============================================================================
# 核心奖励函数
# =============================================================================

def end_effector_position_tracking(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    位置跟踪奖励：手离目标越近，分数越高。
    使用高斯核: reward = exp(-error^2 / std^2)
    
    参数:
        std: 灵敏度参数。std 越小，要求精度越高（误差稍大一点奖励就掉很快）。
    """
    # 1. 获取目标位置
    target_pos_w, _ = _get_target_pose_w(env, command_name)
    
    # 2. 获取末端执行器位置
    robot: Articulation = env.scene[asset_cfg.name]
    # 注意：这里的 body_names 必须在 Config 里指定为你的末端 Link (如 "r_link7")
    body_idx = robot.find_bodies(asset_cfg.body_names)[0] 
    eef_pos_w = robot.data.body_pos_w[:, body_idx, :]

    # 3. 计算距离 (欧氏距离)
    error = torch.norm(target_pos_w - eef_pos_w, dim=-1)
    
    # 4. 计算奖励 (Log空间或Exp空间均可，这里用 Exp)
    return torch.exp(-torch.square(error) / (std**2))


def end_effector_orientation_tracking(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    姿态跟踪奖励：手的朝向和目标越一致，分数越高。
    (可选：如果你只关心碰到物体，不关心抓取角度，可以不用这个奖励)
    """
    # 1. 获取目标姿态
    _, target_quat_w = _get_target_pose_w(env, command_name)
    
    # 2. 获取末端姿态
    robot: Articulation = env.scene[asset_cfg.name]
    body_idx = robot.find_bodies(asset_cfg.body_names)[0]
    eef_quat_w = robot.data.body_quat_w[:, body_idx, :]
    
    # 3. 计算四元数误差 (0 表示完全重合，1 表示反向)
    error = quat_error_magnitude(target_quat_w, eef_quat_w)
    
    # 4. 计算奖励
    return torch.exp(-torch.square(error) / (std**2))


def end_effector_distance_penalty(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    (备选方案) 直接惩罚距离。
    reward = -distance
    这种奖励比较"硬"，容易导致训练不稳定，但也是一种思路。通常不如 Exp 奖励好用。
    """
    target_pos_w, _ = _get_target_pose_w(env, command_name)
    
    robot: Articulation = env.scene[asset_cfg.name]
    body_idx = robot.find_bodies(asset_cfg.body_names)[0]
    eef_pos_w = robot.data.body_pos_w[:, body_idx, :]
    
    return -torch.norm(target_pos_w - eef_pos_w, dim=-1)