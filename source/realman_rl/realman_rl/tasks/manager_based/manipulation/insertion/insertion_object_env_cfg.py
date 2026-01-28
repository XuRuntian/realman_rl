# realman_rl/tasks/manager_based/manipulation/insertion/insertion_object_env_cfg.py

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# 导入我们自定义的 MDP 模块
import realman_rl.tasks.manager_based.manipulation.insertion.mdp as mdp

# 导入你的机器人配置 (假设你在 assets 文件夹下定义好了)
# 如果你还没有定义 REALMAN_ROBOT_CFG，你需要先去 assets/realman.py 里定义它
# 这里暂时假设你有一个默认的变量
from realman_rl.assets.realman import REALMAN_ROBOT_CFG 

##
# Scene definition
##

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the reach scene."""

    # 1. 地面 (Plane)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # 2. 机器人 (Robot)
    # 注意：这里的 robot 变量名对应下面 asset_cfg 里的 "robot"
    robot: ArticulationCfg = REALMAN_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 3. 灯光 (Lights)
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    # 使用我们写的 UniformPoseCommand
    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_names=["r_link7"],  # 【关键】请确保这里是你的末端执行器 Link 名称
        resampling_time_range=(2.0, 4.0), # 每 2-4 秒换一次目标，或者设置为 (1e9, 1e9) 让它一集只变一次
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.6),  # 机器人前方区域
            pos_y=(-0.4, 0.4),
            pos_z=(0.1, 0.6),
            roll=(0.0, 0.0),   # 简单的 Reach 任务通常不要求特定的姿态
            pitch=(0.0, 0.0),
            yaw=(-3.14, 3.14),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # 关节位置控制
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=1.0, 
        use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 1. 关节感知 (Proprioception)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))

        # 2. 任务感知 (Task) - 使用我们写的 target_to_eef
        target_to_eef = ObsTerm(
            func=mdp.eef_to_target_pos_b, # 这个函数在 observations.py 里
            params={
                "command_name": "ee_pose", # 必须和 CommandsCfg 里的名字一致
                "asset_cfg": SceneEntityCfg("robot", body_names=["r_link7"])
            }
        )
        
        # 3. 之前的动作 (Action History)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Critic 通常可以直接复用 Policy 的观测，或者加上 Privileged Information (无噪声)
    # 这里为了简单，直接复制 Policy 配置但去掉噪声
    @configclass
    class CriticCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        target_to_eef = ObsTerm(
            func=mdp.eef_to_target_pos_b,
            params={
                "command_name": "ee_pose", 
                "asset_cfg": SceneEntityCfg("robot", body_names=["r_link7"])
            }
        )
        actions = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # 1. 启动时重置
    reset_startup = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # 2. Episode 结束时重置 (增加一点随机性)
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "position_range": (0.9, 1.1), # +/- 10% 的随机扰动
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # --- 任务奖励 (正分) ---
    reaching_reward = RewTerm(
        func=mdp.end_effector_position_tracking, # 这个函数在 rewards.py 里
        weight=1.0,
        params={
            "std": 0.25, # 精度控制
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=["r_link7"]),
        },
    )

    # --- 惩罚项 (负分) ---
    # 动作幅度惩罚 (使动作更平滑)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    
    # 关节加速度惩罚
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # 关节限位惩罚 (快撞到限位时扣分)
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # 1. 超时 (Time Out)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # # 2. 关节超限 (Safety)
    # joint_pos_limit = DoneTerm(
    #     func=mdp.joint_pos_limits,
    #     params={"asset_cfg": SceneEntityCfg("robot")}
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    # 暂时不需要课程学习
    pass


##
# Environment configuration
##

@configclass
class RealmanInsertionEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the RealMan Insertion environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 2 # 控制频率: 100Hz (如果是 200Hz 仿真)
        self.episode_length_s = 5.0 # Reach 任务通常很快，5秒足够
        
        # Simulation settings
        self.sim.dt = 0.005 # 200Hz Physics
        self.sim.render_interval = self.decimation
        
        # Physics material
        self.sim.physics_material = self.scene.terrain.physics_material
        
        # Viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)