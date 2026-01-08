from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def log_dist_to_target(
    env: ManagerBasedRLEnv, 
    robot_cfg: SceneEntityCfg, 
    object_cfg: SceneEntityCfg,
    ee_body_name: str
) -> torch.Tensor:
    """
    对数距离惩罚 (Log Distance Penalty)。
    距离越近，惩罚越小（也就是奖励越大，因为通常奖励是负的距离）。
    这种形式比指数形式在早期训练时引导性更强。
    """
    # 1. 获取末端位置 (World Frame)
    robot = env.scene[robot_cfg.name]
    ee_id = robot.find_bodies(ee_body_name)[0][0]
    ee_pos_w = robot.data.body_pos_w[:, ee_id]

    # 2. 获取目标位置 (World Frame)
    target = env.scene[object_cfg.name]
    # 注意：如果 target 是刚体用 root_pos_w，如果是 marker 可能不同，这里假设是物体
    target_pos_w = target.data.root_pos_w

    # 3. 计算欧氏距离
    dist = torch.norm(target_pos_w - ee_pos_w, dim=-1)

    # 4. 返回 log 形式 (通常加上一个小的 epsilon 防止 log(0))
    # 比如: 1.0 / (1.0 + dist**2) 也是一种很好的变体
    return torch.log(dist + 1.0e-5)

def reach_target_exp(
    env: ManagerBasedRLEnv, 
    robot_cfg: SceneEntityCfg, 
    object_cfg: SceneEntityCfg,
    ee_body_name: str,
    std: float
) -> torch.Tensor:
    """
    指数形式的 Reach 奖励 (模仿你刚才发的代码风格)。
    当末端完全重合时，奖励为 1.0。
    """
    # 1. 获取末端位置
    robot = env.scene[robot_cfg.name]
    ee_id = robot.find_bodies(ee_body_name)[0][0]
    ee_pos_w = robot.data.body_pos_w[:, ee_id]

    # 2. 获取目标位置
    target = env.scene[object_cfg.name]
    target_pos_w = target.data.root_pos_w

    # 3. 计算平方误差
    error_sq = torch.sum(torch.square(target_pos_w - ee_pos_w), dim=-1)

    # 4. 指数映射 (借鉴了你发的代码)
    return torch.exp(-error_sq / std**2)
