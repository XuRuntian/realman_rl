from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlOnPolicyRunnerCfg

@configclass
class RealmanPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24       # 每个环境采样多少步 (总buffer = 4096 * 24)
    max_iterations = 1500        # 训练多少轮
    save_interval = 50           # 多少轮保存一次模型

    # 算法参数 (经验值)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,       # 熵系数，初期设大点(0.01)鼓励探索
        num_learning_epochs=5,
        num_mini_batches=4,      # 显存如果不够就改大这个数(如8)
        learning_rate=1.0e-3,    # 学习率
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # 网络结构 [隐藏层1, 隐藏层2, ...]
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )