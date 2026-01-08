# Copyright (c) 2024-2025 Realman RL Project
# description: Custom termination functions for manipulation tasks.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ==============================================================================
#  基于任务状态的终止
# ==============================================================================

def reach_target_success(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_body_name: str = "link_tcp",
    threshold: float = 0.02
) -> torch.Tensor:
    """
    当机械臂成功摸到目标时，终止回合 (Reset)。
    (适用于 Sparse Reward 或者你希望每次成功都重置的场景)
    """
    # 1. 获取末端位置
    robot = env.scene[robot_cfg.name]
    ee_id = robot.find_bodies(ee_body_name)[0][0]
    ee_pos_w = robot.data.body_pos_w[:, ee_id]

    # 2. 获取目标位置
    target = env.scene[object_cfg.name]
    # 假设目标是刚体
    target_pos_w = target.data.root_pos_w

    # 3. 计算距离
    distance = torch.norm(target_pos_w - ee_pos_w, dim=-1)

    # 4. 如果距离小于阈值，返回 True (终止)
    return distance < threshold


# ==============================================================================
#  基于物理限制的终止 (防止机器人发疯)
# ==============================================================================

def joint_limit_violation(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    如果关节角度超过了物理极限，重置环境。
    """
    robot = env.scene[robot_cfg.name]
    
    # 获取关节位置
    joint_pos = robot.data.joint_pos
    
    # 获取关节限制 (在 URDF/Assets 配置里定义的)
    # 注意: 通常我们会留一点 buffer，或者直接用 robot.data.joint_limits
    lower_limits = robot.data.default_joint_limits[:, :, 0]
    upper_limits = robot.data.default_joint_limits[:, :, 1]
    
    # 检查是否越界
    out_of_limits = torch.any(joint_pos < lower_limits, dim=-1) | \
                    torch.any(joint_pos > upper_limits, dim=-1)
                    
    return out_of_limits