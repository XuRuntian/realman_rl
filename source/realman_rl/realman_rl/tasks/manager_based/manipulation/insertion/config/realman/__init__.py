import gymnasium as gym

from . import agents, realman_reach_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="RealmanRL-Isaac-Manipulation-Insertion-Realman-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.realman_insertion_env_cfg:RealmanInsertionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RealmanManipulationInsertionPPORunnerCfg",
    },
)