# source/realman_rl/realman_rl/tasks/manager_based/manipulation/reach/config/realman/realman_reach_env_cfg.py

from isaaclab.utils import configclass

from realman_rl.assets.realman import REALMAN_ROBOT_CFG
from realman_rl.tasks.manager_based.manipulation.reach.reach_object_env_cfg import RealmanReachEnvCfg as ReachObjectEnvCfgBase


@configclass
class RealmanReachEnvCfg(ReachObjectEnvCfgBase):
    """
    RealMan 机器人的 Reach 任务配置。
    """
    def __post_init__(self):
        super().__post_init__()

        # =======================================================
        # 1. 替换机器人
        # =======================================================
        self.scene.robot = REALMAN_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # =======================================================
        # 2. 【核心修复】指定正确的末端执行器 (End Effector)
        # =======================================================
        # 根据报错信息，你的可用 link 有: 'r_link7', 'l_link7', 'r_Link_finger1' 等。
        # 这里我们选择【右臂手腕 (r_link7)】作为控制目标。
        # 如果你想控制指尖，可以改成 "r_Link_finger1"
        target_ee_body = "r_link7"

        # -------------------------------------------------------
        # (A) 修改指令生成器 (Command) - 告诉机器人要去哪里
        # -------------------------------------------------------
        self.commands.ee_pose.body_names = [target_ee_body]

        # -------------------------------------------------------
        # (B) 修改观测 (Observations) - 告诉策略“手离目标有多远”
        # -------------------------------------------------------
        # 必须深入到 params["asset_cfg"] 里去修改 body_names
        self.observations.policy.target_to_eef.params["asset_cfg"].body_names = [target_ee_body]
        self.observations.critic.target_to_eef.params["asset_cfg"].body_names = [target_ee_body]

        # -------------------------------------------------------
        # (C) 修改奖励 (Rewards) - 告诉环境“手越近分越高”
        # -------------------------------------------------------
        self.rewards.reaching_reward.params["asset_cfg"].body_names = [target_ee_body]

        # =======================================================
        # 3. 其他调整
        # =======================================================
        self.actions.joint_pos.scale = 1.0