# train_my_robot.py
import argparse
import sys
import os
from datetime import datetime

# ==============================================================================
# 1. 启动仿真 App (必须最先执行！)
# ==============================================================================
from isaaclab.app import AppLauncher

# 定义参数
parser = argparse.ArgumentParser(description="Train Realman Robot")
parser.add_argument("--task", type=str, default="Realman-Reach-v0", help="任务名称")
parser.add_argument("--num_envs", type=int, default=4096, help="环境数量")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
parser.add_argument("--max_iterations", type=int, default=1500, help="训练总轮数")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="禁用 Fabric 物理后端 (通常不用开启)")
# 添加 Isaac Lab 标准参数 (headless, video 等)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 App (这一步之后，pxr 等库才可用)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==============================================================================
# 2. 导入依赖库 (App 启动后才能导)
# ==============================================================================
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

# 导入 Isaac Lab 工具
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# 【关键】把你的源代码目录加入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(current_dir, "source", "realman_rl")
if source_path not in sys.path:
    sys.path.append(source_path)
    print(f"[INFO] Added custom source path: {source_path}")

# 【关键】导入你的包 (触发注册)
import realman_rl 
print("[INFO] Realman RL package imported successfully.")

# ==============================================================================
# 3. 主训练循环
# ==============================================================================
def main():
    # 1. 准备环境配置
    # parse_env_cfg 会自动处理 device 设置
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device="cpu" if args_cli.cpu else "cuda:0",  # <--- ✅ 修正为 device 字符串
        num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric
    )
    
    # 2. 创建环境
    print(f"[INFO] Creating environment for task: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)

    # 3. 加载 PPO 配置
    # 这会从你 __init__.py 注册的 entry_point 读取 RealmanPPORunnerCfg
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    
    # 覆盖部分参数
    agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations:
        agent_cfg.max_iterations = args_cli.max_iterations

    # 4. 设置日志目录
    log_root = os.path.join(current_dir, "logs", "rsl_rl", args_cli.task)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"[INFO] Logging to: {log_dir}")
    
    # 5. 创建 RSL-RL 运行器 (PPO)
    # agent_cfg 是一个配置对象，to_dict() 转为字典传给 runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=env.device)
    
    # 6. 开始训练
    print(f"[INFO] Starting training...")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    
    # 7. 结束
    env.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        raise e
    finally:
        # 关闭仿真 App
        simulation_app.close()