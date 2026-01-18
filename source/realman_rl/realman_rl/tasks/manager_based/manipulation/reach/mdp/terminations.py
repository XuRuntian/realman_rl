# realman_rl/tasks/manager_based/manipulation/reach/mdp/terminations.py

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# =============================================================================
# 自定义终止条件
# =============================================================================

def reach_target_success(
    env: ManagerBasedRLEnv, 
    threshold: float, 
    command_name: str = "ee_pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    成功判定：当末端执行器与目标的距离小于阈值时，认为任务完成，终止环境。
    (这通常配合一个巨大的 sparse reward 使用)
    """
    # 1. 获取目标位置 (从 Command Manager)
    # 这里的切片 [:3] 对应 UniformPoseCommand 的 [x, y, z]
    target_pos_w = env.command_manager.get_command(command_name)[:, :3]

    # 2. 获取末端执行器位置
    robot: Articulation = env.scene[asset_cfg.name]
    # 确保 config 里 body_names 填了末端的名字
    body_idx = robot.find_bodies(asset_cfg.body_names)[0]
    eef_pos_w = robot.data.body_pos_w[:, body_idx, :]

    # 3. 计算距离
    distance = torch.norm(target_pos_w - eef_pos_w, dim=-1)

    # 4. 如果距离小于阈值，返回 True (终止)
    return distance < threshold


def object_dropped(
    env: ManagerBasedRLEnv, 
    threshold: float, 
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    (可选) 如果做抓取任务：物体掉地上了就重置。
    通过检测物体的高度是否低于阈值。
    """
    # 获取物体对象 (RigidObject)
    object_asset = env.scene[asset_cfg.name]
    
    # 检查 z 坐标
    return object_asset.data.root_pos_w[:, 2] < threshold