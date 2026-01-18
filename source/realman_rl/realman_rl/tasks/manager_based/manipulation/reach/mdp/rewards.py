# realman_rl/tasks/manager_based/manipulation/reach/mdp/rewards.py

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def _get_target_pose_w(env: ManagerBasedRLEnv, command_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    command = env.command_manager.get_command(command_name)
    return command[:, :3], command[:, 3:7]

# =============================================================================
# 核心奖励函数
# =============================================================================

def end_effector_position_tracking(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # 1. Target: (N, 3)
    target_pos_w, _ = _get_target_pose_w(env, command_name)
    
    # 2. EEF
    robot: Articulation = env.scene[asset_cfg.name]
    # 【注意】这里 find_bodies(...)[0] 返回的是索引列表/Tensor
    body_idx = robot.find_bodies(asset_cfg.body_names)[0] 
    
    # 获取位置 -> (N, 1, 3) (因为 body_idx 是列表)
    eef_pos_w = robot.data.body_pos_w[:, body_idx, :]

    # 3. 计算误差
    #为了计算正确，我们将 target 也升维到 (N, 1, 3)
    target_pos_expanded = target_pos_w.unsqueeze(1)
    
    # 结果是 (N, 1)
    error = torch.norm(target_pos_expanded - eef_pos_w, dim=-1)
    
    # 4. 计算奖励 (N, 1)
    reward = torch.exp(-torch.square(error) / (std**2))
    
    # 【核心修复】必须降维成 (N,)，否则报错！
    return torch.sum(reward, dim=1)


def end_effector_orientation_tracking(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    _, target_quat_w = _get_target_pose_w(env, command_name)
    
    robot: Articulation = env.scene[asset_cfg.name]
    body_idx = robot.find_bodies(asset_cfg.body_names)[0]
    # (N, 1, 4)
    eef_quat_w = robot.data.body_quat_w[:, body_idx, :]
    
    # target 需要适配 eef 的维度，虽然 quat_error_magnitude 通常能广播，
    # 但为了保险，我们手动处理维度
    target_quat_expanded = target_quat_w.unsqueeze(1)

    # 结果 (N, 1)
    error = quat_error_magnitude(target_quat_expanded, eef_quat_w)
    
    reward = torch.exp(-torch.square(error) / (std**2))
    
    # 【核心修复】降维
    return torch.sum(reward, dim=1)


def end_effector_distance_penalty(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    target_pos_w, _ = _get_target_pose_w(env, command_name)
    
    robot: Articulation = env.scene[asset_cfg.name]
    body_idx = robot.find_bodies(asset_cfg.body_names)[0]
    eef_pos_w = robot.data.body_pos_w[:, body_idx, :] # (N, 1, 3)
    
    target_pos_expanded = target_pos_w.unsqueeze(1) # (N, 1, 3)
    
    dist = torch.norm(target_pos_expanded - eef_pos_w, dim=-1) # (N, 1)
    
    # 【核心修复】降维
    return -torch.sum(dist, dim=1)