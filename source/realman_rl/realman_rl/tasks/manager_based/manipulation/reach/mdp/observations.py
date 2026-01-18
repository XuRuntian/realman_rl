# source/realman_rl/realman_rl/tasks/manager_based/manipulation/reach/mdp/observations.py

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# =============================================================================
# 辅助函数
# =============================================================================

def target_position_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """获取世界坐标系下的目标位置。"""
    command = env.command_manager.get_command(command_name)
    return command[:, :3]

def target_orientation_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """获取世界坐标系下的目标姿态(四元数)。"""
    command = env.command_manager.get_command(command_name)
    return command[:, 3:7]

# =============================================================================
# 核心观测
# =============================================================================

def eef_pos_b(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """获取末端执行器相对于机器人基座的位置。"""
    robot: Articulation = env.scene[asset_cfg.name]
    
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    
    # 【修正】使用 [0][0] 获取整数索引，避免维度变为 (N, 1, 3)
    body_idx = robot.find_bodies(asset_cfg.body_names)[0][0]
    
    eef_pos_w = robot.data.body_pos_w[:, body_idx, :]
    eef_quat_w = robot.data.body_quat_w[:, body_idx, :]

    eef_pos_b, _ = subtract_frame_transforms(
        root_pos_w, root_quat_w, eef_pos_w, eef_quat_w
    )
    
    return eef_pos_b


def eef_to_target_pos_b(
    env: ManagerBasedEnv,
    command_name: str = "ee_pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    获取 '从末端指向目标' 的向量 (Vector from EE to Target)。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # 1. 目标位置 (World)
    target_pos_w = target_position_w(env, command_name)
    
    # 2. 末端位置 (World)
    # 【修正】使用 [0][0] 获取整数索引
    # 这样取出来的 shape 是 (N, 3)，而不是 (N, 1, 3)
    body_idx = robot.find_bodies(asset_cfg.body_names)[0][0]
    
    eef_pos_w = robot.data.body_pos_w[:, body_idx, :]
    eef_quat_w = robot.data.body_quat_w[:, body_idx, :]
    
    # 3. 计算 Target 在 End-Effector 坐标系下的位置
    # 现在 eef_pos_w 是 (N, 3)，target_pos_w 也是 (N, 3)，可以正常计算了
    target_pos_in_eef, _ = subtract_frame_transforms(
        eef_pos_w, eef_quat_w, target_pos_w, torch.zeros_like(eef_quat_w)
    )
    
    return target_pos_in_eef