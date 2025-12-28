# source/realman_rl/realman_rl/assets/realman_cfg.py

from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

# 定义机器人的配置
REALMAN_RMC_CFG = ArticulationCfg(
    # 1. 指定 USD 文件路径
    # 【注意】请把下面的路径改成你机器人的真实 USD 路径！
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/user/realman_rl/source/realman_rl/realman_rl/assets/realman_rmc_aidal/overseas_75_b_v_description_rmg24_with_sites/realman.usd",
        
        # 1. 【新增】这里是让它站稳的关键！
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            fix_root_link=True,  # <--- 将底座焊死在空间中，不受重力掉落影响
        ),

        # 2. 原有的刚体属性保留
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    
    # 2. 定义初始状态
    init_state=ArticulationCfg.InitialStateCfg(
        # pos 参数决定了它“焊”在什么位置
        # 如果桌子在 (0.6, 0, 0)，通常机器人应该在 (0, 0, 0)
        # 你现在的 (1.0, 10.0, 1) 可能会让它离桌子非常远，建议先改回原点
        pos=(-0.5, 0, 0.3), 
        joint_pos={
            ".*_finger.*":0.0325,
            # 关节
            "r_joint1": 0.0,
            "r_joint2": 2.27,
            "r_joint3": -1.31,
            "r_joint4": -1,  
            "r_joint5": 0.0,
            "r_joint6": 0.0,
            "r_joint7": 0.0,

            "l_joint1": 0.0,
            "l_joint2": -2.27,
            "l_joint3": 1.31,
            "l_joint4": 1,
            "l_joint5": 0,
            "l_joint6": 0.0,
            "l_joint7": 0.0,

            #头
            "head_joint2": -0.419,

        },
    ),

    # 3. 定义驱动器 (Actuators)
    # 这是一个通用的驱动器配置，适用于大多数机械臂
    # 3. 定义驱动器 (Actuators)
    actuators={
        # --- 手臂电机 (大扭矩，高刚度) ---
        "arm": ImplicitActuatorCfg(
            # 使用正则排除掉夹爪关节 (假设夹爪名字里有 finger)
            # 注意：这里的正则需要根据你真实的关节名字来写！
            # 如果你的手臂关节叫 joint1-7，夹爪叫 finger1-2
            joint_names_expr=["l_joint[1-7]", "r_joint[1-7]"], 
            stiffness=400.0,
            damping=40.0,
        ),
        
        # --- 夹爪电机 (小扭矩，低刚度) ---
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[".*_finger.*"], # 匹配左右手的夹爪
            effort_limit=50.0,  # 夹爪力气小一点
            velocity_limit=2.0, # 夹爪动得慢一点
            stiffness=200.0,    # 软一点，防止夹飞物体
            damping=10.0,
        ),

    },
)
