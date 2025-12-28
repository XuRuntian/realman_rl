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
## 操作日志
- 版本:isaac-sim 5.1
为了运行test_env.py，在导出的usd文件中将worldBody中的Articulation Root属性删除了