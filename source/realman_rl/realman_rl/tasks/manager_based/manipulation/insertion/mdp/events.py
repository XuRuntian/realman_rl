# source/realman_rl/realman_rl/tasks/manager_based/manipulation/insertion/mdp/events.py

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

# 引入原版实现，并重命名为 _impl 以便内部调用
from isaaclab.envs.mdp import reset_joints_by_scale as _reset_joints_by_scale_impl
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# =============================================================================
# 安全封装 (Safe Wrappers)
# =============================================================================

def reset_joints_by_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
):
    """
    [安全版] 按照比例缩放随机重置关节。
    
    修复了 Isaac Lab 原版函数在 startup 阶段因 env_ids 为 None 而导致的 crash 问题。
    参考: randomize_joint_default_pos 的 env_ids 处理逻辑。
    """
    # 1. 获取 Asset (为了获取 device)
    # 注意：这里我们只用它来确定 device，具体逻辑交给原版函数
    asset = env.scene[asset_cfg.name]

    # 2. 处理 env_ids 为 None 的情况 (Startup 阶段的关键修复)
    if env_ids is None:
        # 使用 torch.arange 生成所有环境的 ID，确保它是一个 Tensor
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # 3. 调用原版实现，传入处理过的 env_ids
    return _reset_joints_by_scale_impl(
        env,
        env_ids,
        asset_cfg=asset_cfg,
        position_range=position_range,
        velocity_range=velocity_range
    )