# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *
# 路径: ~/realman_rl/source/realman_rl/realman_rl/__init__.py
# (这是正确的家)

import gymnasium as gym

# 1. 导入配置
# 注意这里的相对路径：现在是在包根目录，所以 .tasks 是对的
from .tasks.manager_based.realman_rl.realman_rl_env_cfg import RealmanRlEnvCfg
from .agents.rsl_rl_ppo_cfg import RealmanPPORunnerCfg

# 2. 注册环境
gym.register(
    id="Realman-Reach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RealmanRlEnvCfg,
        "rsl_rl_cfg_entry_point": RealmanPPORunnerCfg,
    },
)

print("[INFO] Realman-Reach-v0 registered successfully from package root!")