"""
This script demonstrates how to create a simple environment with random actions.
Usage:
    python test_env.py
"""

import argparse
from isaaclab.app import AppLauncher

# --- 1. 参数解析 & 启动仿真 App ---
# Isaac Lab 要求必须最先执行这一步
parser = argparse.ArgumentParser(description="Test Realman RL Environment")
# 添加标准参数 (如 --headless, --num_envs)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Omniverse (Kit)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 2. 只有在 App 启动后，才能导入 Isaac Lab 模块 ---
# --- 2. 只有在 App 启动后，才能导入 Isaac Lab 模块 ---
import torch
import sys
import os

# 【修复 1】把你的源代码目录加入 Python 搜索路径
# 这样 Python 才能找到 source 文件夹里的代码
current_dir = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(current_dir, "source", "realman_rl")
sys.path.append(source_path)

# 打印一下确认路径对了没 (调试用)
print(f"[INFO] Added to sys.path: {source_path}")

from isaaclab.envs import ManagerBasedRLEnv

# 【修复 2】根据你真实的文件夹结构和文件名导入
# 你的文件夹: realman_rl/assets/
# 你的文件名: realman_cfg.py (注意是 cfg 不是 config)
try:
    from realman_rl.tasks.manager_based.realman_rl.realman_rl_env_cfg import RealmanRlEnvCfg
    print("[SUCCESS] Successfully imported RealmanRlEnvCfg!")
except ImportError as e:
    print("\n" + "="*50)
    print(f"[ERROR] 导入依然失败: {e}")
    print(f"请确认 source/realman_rl/realman_rl/tasks/manager_based/realman_rl/realman_rl_env_cfg.py 文件存在且内容正确")
    print("="*50 + "\n")
    raise e

def main():
    """Main function to run the environment loop."""
    
    # 实例化配置
    env_cfg = RealmanRlEnvCfg()
    
    # 【Debug 技巧】强制把环境数设为 1，方便在图形界面里观察细节
    # 如果你是 4096 个环境叠在一起，画面会非常混乱
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 5.0 # 拉开一点距离

    # 1. 创建环境
    print("[INFO] Creating environment...")
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # 2. 重置环境 (获取初始观测)
    print("[INFO] Resetting environment...")
    obs, _ = env.reset()

    # 打印一下维度的信息，确认有没有写错
    print("-" * 40)
    print(f"[INFO] Number of Envs : {env.num_envs}")
    print(f"[INFO] Action Space   : {env.action_space}")
    print(f"[INFO] Observation    : {obs['policy'].shape}") # 检查观测维度
    print("-" * 40)

    # 3. 仿真循环
    print("[INFO] Starting simulation loop... Press Ctrl+C to stop.")
    sim_step = 0
    
    while simulation_app.is_running():
        # 生成随机动作 (Random Action)
        # 动作范围通常在 [-1, 1] 之间
        # 维度会自动匹配你的 ActionsCfg (arm_left + arm_right + grippers)
        actions = 2 * torch.rand(env.num_envs, env.action_space.shape[1], device=env.device) - 1
        
        # 执行一步仿真
        # 返回值: 观测, 奖励, 终止标志(Terminated), 截断标志(Truncated), 额外信息
        obs, rew, terminated, truncated, extras = env.step(actions)

        # 这里的 reset 是由 ManagerBasedRLEnv 内部自动处理的
        # 如果 terminated 为 True，它会自动重置那个特定的环境
        
        sim_step += 1
        if sim_step % 100 == 0:
            print(f"Step {sim_step}: Environment is running smoothly.")

    # 4. 关闭环境
    env.close()

if __name__ == "__main__":
    main()
    # 关闭 App
    simulation_app.close()