# Copyright (c) 2024-2025 Realman RL Project
# description: Custom observation functions for manipulation tasks.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# ==============================================================================
#  相对于机器人基座 (Base Frame) 的观测
#  (这对于训练非常重要，因为它让训练结果具有"平移不变性")
# ==============================================================================

def end_effector_pos_in_robot_root_frame(
    env: ManagerBasedEnv, 
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_name: str = "link_tcp" # 默认值，实际使用时在 Config 中覆盖
) -> torch.Tensor:
    """
    获取机械臂末端(End-Effector)相对于机器人基座(Root)的位置。
    
    Args:
        robot_cfg: 机器人的配置，用于查找资产。
        ee_body_name: 机械臂末端刚体的名字 (在URDF里的 link name)。
    """
    # 1. 获取机器人的资产对象
    robot: Articulation = env.scene[robot_cfg.name]
    
    # 2. 获取末端执行器的索引 (通过名字找 ID)
    # 注意：这里假设只有一个刚体作为末端，如果有多个需要调整逻辑
    ee_id = robot.find_bodies(ee_body_name)[0][0]

    # 3. 获取数据 (均为世界坐标系 World Frame)
    # 机器人基座的位置和姿态
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    
    # 末端执行器的位置和姿态
    ee_pos_w = robot.data.body_pos_w[:, ee_id]
    ee_quat_w = robot.data.body_quat_w[:, ee_id]

    # 4. 坐标变换: World -> Robot Base
    # 公式: Pos_local = Inverse(Base_Transform) * (Pos_world - Base_pos)
    # Isaac Lab 提供了 subtract_frame_transforms 帮我们做这个数学运算
    ee_pos_b, _ = subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
    )

    return ee_pos_b


def object_pos_in_robot_root_frame(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    获取目标物体(Object)相对于机器人基座(Root)的位置。
    """
    # 1. 获取资产
    robot: Articulation = env.scene[robot_cfg.name]
    # 注意：目标可能是 RigidObject (如方块) 也可以是 Articulation
    obj = env.scene[object_cfg.name]

    # 2. 获取数据 (世界坐标系)
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    # 获取物体位置 (如果是 RigidObject，通常用 root_pos_w)
    if isinstance(obj, RigidObject) or isinstance(obj, Articulation):
        obj_pos_w = obj.data.root_pos_w
        obj_quat_w = obj.data.root_quat_w
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")

    # 3. 坐标变换
    obj_pos_b, _ = subtract_frame_transforms(
        root_pos_w, root_quat_w, obj_pos_w, obj_quat_w
    )

    return obj_pos_b


# ==============================================================================
#  向量差观测 (Vector Diff)
#  (强化学习非常喜欢这个，因为它直接告诉网络"误差"是多少)
# ==============================================================================

def end_effector_to_object_vector(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_body_name: str = "link_tcp"
) -> torch.Tensor:
    """
    计算从末端执行器指向目标物体的向量 (在机器人基座坐标系下)。
    Vector = Object_Pos - EE_Pos
    """
    # 复用上面的函数计算两个位置
    ee_pos_b = end_effector_pos_in_robot_root_frame(env, robot_cfg, ee_body_name)
    obj_pos_b = object_pos_in_robot_root_frame(env, robot_cfg, object_cfg)

    # 计算向量差
    return obj_pos_b - ee_pos_b