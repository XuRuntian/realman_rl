# realman_rl/tasks/manager_based/manipulation/reach/mdp/terminations.py

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reach_target_success(
    env: ManagerBasedRLEnv, 
    threshold: float, 
    command_name: str = "ee_pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    成功判定：距离小于阈值。
    """
    # (N, 3)
    target_pos_w = env.command_manager.get_command(command_name)[:, :3]

    robot: Articulation = env.scene[asset_cfg.name]
    body_idx = robot.find_bodies(asset_cfg.body_names)[0]
    
    # (N, 1, 3)
    eef_pos_w = robot.data.body_pos_w[:, body_idx, :]

    # 升维 target 以匹配
    target_pos_expanded = target_pos_w.unsqueeze(1)

    # (N, 1)
    distance = torch.norm(target_pos_expanded - eef_pos_w, dim=-1)

    # (N, 1) -> boolean
    success = distance < threshold

    # 【修复】降维成 (N,)
    # 这里的 any(dim=1) 意味着：如果配置了多个末端，只要有一个到达目标就算成功
    # 如果只有一个末端，any 和 squeeze 效果一样
    return success.any(dim=1)


def object_dropped(
    env: ManagerBasedRLEnv, 
    threshold: float, 
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    # 这里的 root_pos_w 通常本身就是 (N, 3)，所以不需要复杂处理
    object_asset = env.scene[asset_cfg.name]
    
    # (N,)
    return object_asset.data.root_pos_w[:, 2] < threshold