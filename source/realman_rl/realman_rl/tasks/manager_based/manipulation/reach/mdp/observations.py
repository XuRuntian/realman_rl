# realman_rl/tasks/manager_based/manipulation/reach/mdp/observations.py

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# =============================================================================
# 辅助函数：获取目标位置
# =============================================================================

def target_position_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """获取世界坐标系下的目标位置。"""
    # 从 Command Manager 获取指令
    # 我们之前的 UniformPoseCommand 返回的是 [x, y, z, qw, qx, qy, qz]
    command = env.command_manager.get_command(command_name)
    return command[:, :3]

def target_orientation_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """获取世界坐标系下的目标姿态(四元数)。"""
    command = env.command_manager.get_command(command_name)
    return command[:, 3:7]

# =============================================================================
# 核心观测：相对位置
# =============================================================================

def eef_pos_b(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    获取末端执行器(End-Effector)相对于机器人基座(Base)的位置。
    这对机械臂任务很有用，让它知道手在哪里。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # 获取基座（Root）和末端（Body）的世界坐标
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    
    # 找到末端执行器的索引 (需要在配置文件里指定 body_names=["link_name"])
    # 如果没指定 body_names，默认取所有 bodies，这通常不是我们想要的
    body_idx = robot.find_bodies(asset_cfg.body_names)[0]
    eef_pos_w = robot.data.body_pos_w[:, body_idx, :]
    eef_quat_w = robot.data.body_quat_w[:, body_idx, :] # 这一步虽然这里没用到，但保持对称性

    # 将末端位置转换到基座坐标系下
    eef_pos_b, _ = subtract_frame_transforms(
        root_pos_w, root_quat_w, eef_pos_w, eef_quat_w
    )
    
    return eef_pos_b


def target_pos_b(
    env: ManagerBasedEnv, 
    command_name: str = "ee_pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    获取目标(Target)相对于机器人基座(Base)的位置。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # 1. 获取目标世界坐标
    target_pos_w = target_position_w(env, command_name)
    target_quat_w = target_orientation_w(env, command_name)
    
    # 2. 获取机器人基座世界坐标
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    # 3. 转换坐标系: Target in Base Frame
    # 数学含义: R_base^T * (P_target - P_base)
    target_pos_in_base, _ = subtract_frame_transforms(
        root_pos_w, root_quat_w, target_pos_w, target_quat_w
    )
    
    return target_pos_in_base


def eef_to_target_pos_b(
    env: ManagerBasedEnv,
    command_name: str = "ee_pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    【最关键的观测】获取 "从末端指向目标" 的向量 (Vector from EE to Target)。
    
    这个观测对训练收敛至关重要，它直接告诉 Policy 误差方向。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # 1. 目标位置 (World)
    target_pos_w = target_position_w(env, command_name)
    
    # 2. 末端位置 (World)
    body_idx = robot.find_bodies(asset_cfg.body_names)[0]
    eef_pos_w = robot.data.body_pos_w[:, body_idx, :]
    
    # 3. 向量差 (World Frame)
    error_vec_w = target_pos_w - eef_pos_w
    
    # 4. (可选) 如果你的机器人基座是固定的，可以直接返回 error_vec_w。
    # 如果机器人基座会动（比如移动底盘），最好把它转到基座坐标系下。
    # 这里我们演示转到基座坐标系：
    root_quat_w = robot.data.root_quat_w
    
    # 使用 subtract_frame_transforms 的一个小技巧：
    # 把 error_vec 当作一个相对于原点的位置，减去原点(0,0,0)的变换，其实就是旋转向量
    # 或者更简单的：直接用 quat_rotate_inverse (Isaac Lab 有对应工具)
    # 这里为了通用性，还是用 subtract_frame_transforms
    
    # 这里的逻辑是：计算 Error 相对于 Base 的坐标
    # 但更简单的做法是：只要知道 "Target相对于EE" 的向量即可。
    # 让我们直接返回 World Frame 下的误差向量，对于固定臂这通常足够了。
    # 更好的做法通常是返回 "Target position in EE frame" (目标在手坐标系下的位置)
    
    # --- 方案 B: Target in End-Effector Frame (推荐) ---
    eef_quat_w = robot.data.body_quat_w[:, body_idx, :]
    
    target_pos_in_eef, _ = subtract_frame_transforms(
        eef_pos_w, eef_quat_w, target_pos_w, torch.zeros_like(eef_quat_w) # 姿态由于没对齐需求，暂时忽略
    )
    
    return target_pos_in_eef