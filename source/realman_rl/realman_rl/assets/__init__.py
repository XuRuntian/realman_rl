# import gymnasium as gym
# from .tasks.manager_based.realman_rl.realman_rl_env_cfg import RealmanRlEnvCfg

# # 注册环境
# gym.register(
#     id="Realman-Reach-v0",  # 给你的任务起个名字
#     entry_point="isaaclab.envs:ManagerBasedRLEnv", # 使用标准的管理器环境
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": RealmanRlEnvCfg, # 指向你的配置类
#         "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:RealmanPPORunnerCfg", # PPO配置(见下一步)
#     },
# )

from .realman_cfg import REALMAN_RMC_CFG