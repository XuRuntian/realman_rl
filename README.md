# realman-rl
- 启动isaac-sim
```
conda deactivate && cd ~/IsaacLab && source .venv/bin/activate
${ISAACSIM_PATH}/isaac-sim.sh
```
- 测试代码
```
python test_env.py --enable_cameras

```
- 训练
```
# 加上 --enable_cameras 让仿真器开启渲染引擎
python ~/realman_rl/train_my_robot.py \
  --task Realman-Reach-v0 \
  --num_envs 4096 \
  --headless \
  --enable_cameras
```
## 操作日志
- 版本:isaac-sim 5.1
为了运行test_env.py，在导出的usd文件中将worldBody中的Articulation Root属性删除了