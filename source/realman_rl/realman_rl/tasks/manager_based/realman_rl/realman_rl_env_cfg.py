import torch
import math

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp

# 导入资产和传感器
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
# 【关键】导入相机配置
from isaaclab.sensors import CameraCfg

from isaaclab.managers import (
    SceneEntityCfg,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    EventTermCfg as EventTerm,
    TerminationTermCfg as DoneTerm
)

# 导入你的机器人资产配置
from realman_rl.assets.realman_cfg import REALMAN_RMC_CFG


# =========================================================
# 1. 自定义 MDP 逻辑函数
# =========================================================

def object_pos_in_robot_frame(env, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """计算物体相对于机器人基座的位置 (Obs)"""
    robot_root_pos = env.scene[robot_cfg.name].data.root_pos_w
    object_root_pos = env.scene[object_cfg.name].data.root_pos_w
    return object_root_pos - robot_root_pos


def reward_hand_reaching_object(
    env, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, std: float = 0.25
) -> torch.Tensor:
    """靠近奖励"""
    hand_pos = env.scene[robot_cfg.name].data.body_pos_w[:, robot_cfg.body_ids[0]]
    object_pos = env.scene[object_cfg.name].data.root_pos_w
    distance = torch.norm(hand_pos - object_pos, dim=-1)
    return 1.0 / (1.0 + (distance / std).pow(2))


def reward_object_lifted(
    env, object_cfg: SceneEntityCfg, minimal_height: float = 0.65
) -> torch.Tensor:
    """提起奖励"""
    object_pos_z = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    return torch.where(object_pos_z > minimal_height, 1.0, 0.0)


# =========================================================
# 2. 配置类定义
# =========================================================

@configclass
class ActionsCfg:
    """定义动作空间"""
    arm_left = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["l_joint[1-7]"], scale=0.5, use_default_offset=True,
    )
    arm_right = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["r_joint[1-7]"], scale=0.5, use_default_offset=True,
    )
    gripper_left = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["l_Joint_finger.*"], scale=0.02, offset=0.02, use_default_offset=False, 
    )
    gripper_right = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["r_Joint_finger.*"], scale=0.02, offset=0.02, use_default_offset=False, 
    )


@configclass
class ObservationsCfg:
    """
    【关键修改】这里只保留物理状态观测，不加入相机图像。
    这样训练速度快，且不需要改 PPO 算法结构。
    """
    @configclass
    class PolicyCfg(ObsGroup):
        # 1. 关节位置和速度
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        # 2. 物体位置 (这相当于给了机器人"天眼"，适合初期训练)
        object_position = ObsTerm(
            func=object_pos_in_robot_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object")
            } 
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """随机化事件"""
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """奖励函数"""
    # 任务奖励
    reaching_left = RewTerm(
        func=reward_hand_reaching_object, weight=1.0, 
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["l_link7"]), # 请确认 URDF 名字
            "object_cfg": SceneEntityCfg("object"),
            "std": 0.25
        }
    )
    reaching_right = RewTerm(
        func=reward_hand_reaching_object, weight=1.0, 
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["r_link7"]), 
            "object_cfg": SceneEntityCfg("object"),
            "std": 0.25
        }
    )
    lifting = RewTerm(
        func=reward_object_lifted, weight=5.0, 
        params={"object_cfg": SceneEntityCfg("object"), "minimal_height": 0.65}
    )

    # 惩罚项
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel = RewTerm(func=mdp.joint_vel_l1, weight=-0.005, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropped = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={"minimum_height": 0.3, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class RealmanRlSceneCfg(InteractiveSceneCfg):
    """场景定义：包含机器人、相机、物体"""
    
    # 1. 机器人
    robot = REALMAN_RMC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # -------------------------------------------------------------
    # 2. 相机配置 (只渲染，不进入 RL 观测)
    #    分辨率设为 320x240，方便人眼看，也不会太卡
    # -------------------------------------------------------------
    
    # 左手腕相机
    camera_left_wrist = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/world_root/l_link7/camera_left_wrist", # 挂载点
        update_period=0.1, # 10Hz 更新一次，节省性能
        height=240, width=320,
        data_types=["rgb"], 
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.05), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    # 右手腕相机
    camera_right_wrist = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/world_root/r_link7/camera_right_wrist",
        update_period=0.1,
        height=240, width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.05), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    # 头部/全局相机
    camera_head = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/world_root/head_link2/camera_head", # 请确认 base_link 或 camera_link 存在
        update_period=0.1,
        height=240, width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.05), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    # 3. 环境资产
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.2, 0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.3)),
    )
    # 可抓取物体
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 1.0)),
            
            # 1. 刚体属性 (只管 运动学/重力/阻尼)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            
            # 2. 【新增】质量属性 (专门管 质量/质心)
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1), 
            
            # 3. 物理材质 (摩擦力)
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.0, 1.65), 
        ),
    )

@configclass
class RealmanRlEnvCfg(ManagerBasedRLEnvCfg):
    """最终环境入口"""
    scene: RealmanRlSceneCfg = RealmanRlSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 0.0166
        self.episode_length_s = 5.0