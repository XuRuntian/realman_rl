# realman_rl/tasks/manager_based/manipulation/reach/mdp/events.py

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp import (
    reset_joints_by_scale,       # [标准] 按照比例缩放随机重置关节
    reset_joints_by_offset,      # [标准] 按照偏移量随机重置关节
    reset_root_state_uniform,    # [标准] 如果你的基座会动，用这个
    push_by_setting_velocity,    # [标准] 给机器人一个随机扰动（推一把）
)
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# --- 自定义 Event 区域 ---
# 目前 Reach 任务通常不需要自定义 Event，
# 因为 "目标位置" (Target) 已经由 commands.py 管理了。
# 而 "机器人关节重置" 可以直接用上面的 reset_joints_by_scale。

# 如果你未来需要 "随机化物体摩擦力" 或 "随机化物体质量"，
# 也可以直接引用 isaaclab.envs.mdp 中的函数，不需要在这里重写。