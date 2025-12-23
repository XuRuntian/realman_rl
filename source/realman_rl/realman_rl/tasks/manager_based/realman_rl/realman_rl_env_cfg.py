# python
import torch
import math

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp

# 导入资产和传感器
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.sensors import CameraCfg

# 导入管理器配置
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
# 1. 自定义 MDP 逻辑函数 (Strictly Typed)
# =========================================================

def object_pos_in_robot_frame(env, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    计算物体相对于机器人基座的位置。
    用于 Observation。
    """
    # 也就是: object_global_pos - robot_global_pos
    robot_root_pos = env.scene[robot_cfg.name].data.root_pos_w
    object_root_pos = env.scene[object_cfg.name].data.root_pos_w
    return object_root_pos - robot_root_pos


def reward_hand_reaching_object(
    env, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, std: float = 0.25
) -> torch.Tensor:
    """
    通用靠近奖励：计算指定末端(hand)与物体的距离。
    
    参数:
        robot_cfg: 必须指定 body_names 为具体的某个手部 link (如 l_link7)
    """
    # 获取末端执行器位置 (使用 robot_cfg 中解析出的 body_ids)
    # 注意: 这里取 body_ids[0] 是因为我们在 params 里会分别为左手和右手实例化这个函数
    hand_pos = env.scene[robot_cfg.name].data.body_pos_w[:, robot_cfg.body_ids[0]]
    object_pos = env.scene[object_cfg.name].data.root_pos_w
    
    # 计算欧式距离
    distance = torch.norm(hand_pos - object_pos, dim=-1)
    
    # 使用 tanh 核函数将距离映射到 [0, 1]
    # distance = 0 时奖励为 1，distance 越大奖励越小
    return 1.0 / (1.0 + (distance / std).pow(2))


def reward_object_lifted(
    env, object_cfg: SceneEntityCfg, minimal_height: float = 0.65
) -> torch.Tensor:
    """
    抓取成功奖励：当物体高度超过阈值时给予高分。
    """
    object_pos_z = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    
    # 二值奖励：超过高度给 1.0，否则 0.0
    # 也可以改成连续奖励： return torch.clamp(object_pos_z - minimal_height, min=0.0)
    return torch.where(object_pos_z > minimal_height, 1.0, 0.0)


# =========================================================
# 2. 配置类定义 (Configuration Classes)
# =========================================================

@configclass
class ActionsCfg:
    """定义动作空间 (Action Space)。"""
    
    # 左臂 7自由度
    arm_left = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["l_joint[1-7]"], 
        scale=0.5,
        use_default_offset=True,
    )
    # 右臂 7自由度
    arm_right = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["r_joint[1-7]"],
        scale=0.5,
        use_default_offset=True,
    )
    
    # 左夹爪
    gripper_left = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["l_joint_finger.*"], # 使用正则，确保匹配所有手指关节
        scale=0.02,        
        offset=0.02,       
        use_default_offset=False, 
    )
    
    # 右夹爪
    gripper_right = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["r_joint_finger.*"],
        scale=0.02,        
        offset=0.02,       
        use_default_offset=False, 
    )


@configclass
class ObservationsCfg:
    """定义观测空间 (Observation Space)。"""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """策略网络输入的观测组。"""
        
        # 关节位置和速度 (Proprioception)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        # 物体位置 (相对于机器人)
        object_position = ObsTerm(
            func=object_pos_in_robot_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object")
            } 
        )

        def __post_init__(self):
            self.enable_corruption = True # 开启噪声
            self.concatenate_terms = True # 将所有项拼接成一个长向量
            
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """定义仿真过程中的事件 (Randomization/Resets)。"""
    
    # 1. 物体位置随机化
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    # 2. 机器人关节随机化 (增加鲁棒性)
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
    """定义奖励函数 (Reward Function)。"""
    
    # -------------------------------------------------------
    # 任务奖励 (Task Rewards) - 权重通常较大，正值
    # -------------------------------------------------------
    
    # 左手靠近物体
    reaching_left = RewTerm(
        func=reward_hand_reaching_object, 
        weight=1.0, 
        params={
            # 【关键】这里指定 body_names 为左手末端 link
            # 请确保 URDF 中名字正确，例如 "l_link7" 或 "l_end_effector"
            "robot_cfg": SceneEntityCfg("robot", body_names=["l_link7"]), 
            "object_cfg": SceneEntityCfg("object"),
            "std": 0.25
        }
    )
    
    # 右手靠近物体
    reaching_right = RewTerm(
        func=reward_hand_reaching_object, 
        weight=1.0, 
        params={
            # 【关键】这里指定 body_names 为右手末端 link
            "robot_cfg": SceneEntityCfg("robot", body_names=["r_link7"]), 
            "object_cfg": SceneEntityCfg("object"),
            "std": 0.25
        }
    )
    
    # 物体被提起
    lifting = RewTerm(
        func=reward_object_lifted,
        weight=5.0, # 提起是最终目标，给予大权重
        params={
            "object_cfg": SceneEntityCfg("object"),
            "minimal_height": 0.65, # 桌高 0.6 + 5cm
        }
    )

    # -------------------------------------------------------
    # 惩罚/正则化 (Penalties) - 权重通常较小，负值
    # -------------------------------------------------------
    
    # 惩罚动作幅度过大 (使动作平滑)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    
    # 惩罚关节速度过快 (保护电机，减少抖动)
    joint_vel = RewTerm(func=mdp.joint_vel_l1, weight=-0.005, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    """定义终止条件 (Termination Conditions)。"""
    
    # 时间耗尽
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # 物体掉落 (失败终止)
    object_dropped = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={"minimum_height": 0.3, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class RealmanRlSceneCfg(InteractiveSceneCfg):
    """场景定义 (Scene Definition)。"""
    
    # -------------------------------------------------------
    # 1. 机器人与传感器
    # -------------------------------------------------------
    robot = REALMAN_RMC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # 左手腕相机
    camera_left_wrist = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/l_link7/camera_left_wrist",
        update_period=0.1,
        height=240, width=320, # 分辨率可以根据显存调整
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.05), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    # 右手腕相机
    camera_right_wrist = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/r_link7/camera_right_wrist",
        update_period=0.1,
        height=240, width=320,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.05), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    # 头部相机 (假设安装在 camera_link)
    camera_head = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera_link/camera_head",
        update_period=0.1,
        height=240, width=320,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.05), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    # -------------------------------------------------------
    # 2. 环境资产 (Environment Assets)
    # -------------------------------------------------------
    
    # 灯光
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    
    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    
    # 桌子
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.2, 0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, 
                disable_gravity=True,
            ),
            # 增加摩擦力，防止物体打滑
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.3), 
        ),
    )
    
    # 抓取目标 (Object)
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 1.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                mass=0.1,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.65), # 放在桌面上方一点点
        ),
    )


@configclass
class RealmanRlEnvCfg(ManagerBasedRLEnvCfg):
    """
    Realman 双臂机器人强化学习环境入口配置。
    任务：双臂靠近并抓取桌面上的立方体。
    """
    scene: RealmanRlSceneCfg = RealmanRlSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # 仿真步长设置
        self.decimation = 4    # 控制频率 = Physics / 4
        self.sim.dt = 0.0166   # 物理频率 60Hz
        self.episode_length_s = 5.0