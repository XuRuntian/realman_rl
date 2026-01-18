# realman-rl
- 启动isaac-sim
```
conda deactivate  && source ~/IsaacLab/.venv/bin/activate
${ISAACSIM_PATH}/isaac-sim.sh
```
- 安装环境
```
cd realman_rl && uv pip install -e source/realman_rl
```
- 查看注册的环境
```
python scripts/list_envs.py
```
- 测试环境
```
python scripts/random_agent.py --task RealmanRL-Isaac-Manipulation-Reach-Realman-v0 --num_envs 1
```