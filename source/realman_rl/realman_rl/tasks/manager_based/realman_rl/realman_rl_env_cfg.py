from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab.sim as sim_utils

# 【修正 1】从 assets 导入 AssetBaseCfg (用于灯光和地面)
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg

# 【修正 2】从 managers 导入所有 Term 配置类
from isaaclab.managers import (
    SceneEntityCfg,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    EventTermCfg as EventTerm,
    TerminationTermCfg as DoneTerm
)

# mdp 只保留逻辑函数和 Action 配置类
import isaaclab.envs.mdp as mdp

# 导入机器人配置
from realman_rl.assets.realman_cfg import REALMAN_RMC_CFG

# --- 补充缺失函数 ---
def object_pos_in_robot_frame(env, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg):
    robot_root_pose = env.scene[robot_cfg.name].data.root_pose_w
    object_root_pose = env.scene[object_cfg.name].data.root_pose_w
    return object_root_pose[:, :3] - robot_root_pose[:, :3]

# --------------------------------------------------------

@configclass
class ActionsCfg:
    """Action specifications."""
    # Action 配置类确实是在 mdp 里的
@configclass
class ActionsCfg:
    """Action specifications."""
    # 手臂控制 (JointPositionActionCfg 默认带了 class_type，所以没报错)
    arm_left = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["l_joint[1-7]"], 
            scale=0.5,
            use_default_offset=True,
    )
    arm_right = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["r_joint[1-7]"],
            scale=0.5,
            use_default_offset=True,
    )
    
    # --- 修正点：显式指定 class_type ---
    gripper_left = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["l_Joint_finger[1-2]"],
        scale=0.01625,        # 缩放
        offset=0.01625,       # 偏移
        use_default_offset=False, # 不使用默认姿态，使用我们上面的绝对偏移
    )
    
    gripper_right = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["r_Joint_finger[1-2]"],
        scale=0.01625,        # 缩放
        offset=0.01625,       # 偏移
        use_default_offset=False, # 不使用默认姿态，使用我们上面的绝对偏移
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
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
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l1, weight=-0.01, params={"asset_cfg": SceneEntityCfg("robot")})

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class RealmanRlSceneCfg(InteractiveSceneCfg):
    # 机器人
    robot = REALMAN_RMC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # 【关键修正】使用 AssetBaseCfg，而不是 mdp.LightCfg
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    
    # 【关键修正】使用 AssetBaseCfg，而不是 mdp.AssetBaseCfg
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
    )

@configclass
class RealmanRlEnvCfg(ManagerBasedRLEnvCfg):
    scene: RealmanRlSceneCfg = RealmanRlSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (1.5, 0.0, 1.2)
        self.sim.render_interval = self.decimation