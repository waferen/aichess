# AutoDL 云服务器部署指南

## 服务器配置适配

### 方案1：高性能配置（32GB vGPU）

**配置：**
- GPU: vGPU 32GB 显存
- CPU: 17核 Intel Xeon Platinum 8470Q
- 内存: 60GB

**性能：⭐⭐⭐⭐⭐**
- 自动启动：5-7个数据收集进程
- 100轮训练：约8-9小时

---

### 方案2：标准配置（RTX 3090）⭐当前配置

**配置：**
- GPU: RTX 3090 (24GB 显存)
- CPU: 4核
- 内存: 16GB

**性能：⭐⭐⭐⭐**
- 自动启动：2个数据收集进程
- 100轮训练：约15-18小时
- 成本：更低

**说明：** 脚本会自动检测CPU核心数并调整并行进程数：
- ≤4核：启动2个collect进程
- 5-8核：启动3个collect进程
- >8核：启动5个collect进程

---

## 快速部署

### 1. 上传代码到AutoDL

**方法1：Git克隆（推荐）**
```bash
cd /root/autodl-fs
git clone <你的仓库地址>
cd aichess
```

**方法2：Jupyter上传**
1. 在本地压缩项目为 `aichess.zip`
2. AutoDL Jupyter界面 → Upload
3. 解压：
```bash
cd /root/autodl-fs
unzip aichess.zip
cd aichess
```

### 2. 运行配置脚本

```bash
bash autodl_setup.sh
```

这个脚本会自动：
- ✅ 检测系统配置（CPU、GPU、内存）
- ✅ 检查环境和依赖
- ✅ 安装numpy、pygame
- ✅ 创建必要目录
- ✅ 优化配置参数
- ✅ 推荐合适的并行度

### 3. 启动训练

```bash
bash autodl_start.sh
```

这个脚本会：
- 🔥 自动检测CPU核心数
- 🔥 自动调整并行进程数
- 🔥 启动数据收集进程
- 🔥 启动训练进程
- 📝 将日志保存到数据盘

### 4. 监控训练

**实时监控面板：**
```bash
bash autodl_monitor.sh
```

**手动监控命令：**
```bash
# 查看训练日志
tail -f /root/autodl-fs/logs/train.log

# 查看数据收集日志
tail -f /root/autodl-fs/logs/collect_1.log

# GPU使用情况
nvidia-smi

# 进程状态
ps aux | grep python

# 数据文件大小
ls -lh /root/autodl-fs/train_data_buffer.pkl
```

---

## 不同配置的训练时间

### 100轮训练对比

| 配置 | Collect进程 | 单次更新 | 100轮时间 |
|------|------------|----------|-----------|
| 本地单进程 | 1 | 38分钟 | ~64小时 |
| RTX 3090 (4核) | 2 | 10分钟 | ~16小时 |
| RTX 3090 (8核) | 3 | 7分钟 | ~12小时 |
| 32GB vGPU (17核) | 5-7 | 5分钟 | ~9小时 |

### 成本估算（100轮）

| GPU型号 | 价格/小时 | 100轮时间 | 成本 |
|---------|----------|-----------|------|
| RTX 3060 | ¥2 | 18小时 | ¥36 |
| RTX 3090 | ¥3.5 | 16小时 | ¥56 |
| 32GB vGPU | ¥4.5 | 9小时 | ¥40 |

---

## 优化后的配置

脚本会根据配置自动优化：

### RTX 3090 (4核CPU)
```python
'game_batch_num': 100,  # 保持100轮
'batch_size': 512,      # 保持默认
'train_update_interval': 60,  # 优化：从600秒减少到60秒
# 启动2个collect进程
```

### 高性能配置（17核CPU）
```python
'game_batch_num': 100,
'batch_size': 1024,     # 可以增加到1024
'play_out': 1600,       # 可以增加模拟次数
'train_update_interval': 60,
# 启动5-7个collect进程
```

---

## 目录结构

部署后的目录结构：

```
/root/autodl-fs/
├── aichess/                    # 项目代码
│   ├── autodl_setup.sh        # 配置脚本
│   ├── autodl_start.sh        # 启动脚本（自动检测配置）
│   ├── autodl_monitor.sh      # 监控脚本
│   ├── collect.py
│   ├── train.py
│   └── ...
├── models/                    # 模型文件
│   ├── current_policy.pkl
│   └── current_policy_batch_20.pkl
│   └── current_policy_batch_40.pkl
│   └── ...
├── logs/                      # 训练日志
│   ├── train.log
│   ├── collect_1.log
│   ├── collect_2.log
│   └── ...
└── train_data_buffer.pkl      # 训练数据
```

---

## 常用命令

### 启动训练
```bash
cd /root/autodl-fs/aichess
bash autodl_start.sh
```

### 停止训练
```bash
pkill -f "python collect.py"
pkill -f "python train.py"
```

### 重启训练
```bash
# 先停止
pkill -f "python collect.py"
pkill -f "python train.py"

# 等待几秒
sleep 5

# 重新启动
bash autodl_start.sh
```

### 查看进度
```bash
# 方法1：监控脚本
bash autodl_monitor.sh

# 方法2：手动查看
tail -f /root/autodl-fs/logs/train.log
```

---

## 故障排除

### GPU显存不足
```bash
# 减少并行进程数
# 编辑 autodl_start.sh 中的 NUM_COLLECT_PROCESSES
# 或者减少 batch_size
```

### CPU占用过高
```bash
# 减少collect进程数
# 脚本会自动检测，≤4核CPU默认启动2个进程
```

### 数据文件未生成
```bash
# 检查collect进程是否正常运行
ps aux | grep collect

# 查看collect日志
tail -f /root/autodl-fs/logs/collect_1.log
```

### 训练不更新
```bash
# 检查数据文件大小
ls -lh /root/autodl-fs/train_data_buffer.pkl

# 查看训练日志
tail -f /root/autodl-fs/logs/train.log
```

---

## 节省成本技巧

1. **使用RTX 3090** - 性价比最高
2. **及时关机** - 训练完成后立即关机
3. **晚上启动** - 睡觉时训练，早上完成
4. **监控资源** - 确保GPU满载运行

---

## 训练进度预估

### RTX 3090 (4核, 2个collect进程)

| 轮数 | 预计时间 | 说明 |
|------|----------|------|
| 20轮 | 3小时 | 💾 第1个检查点 |
| 40轮 | 6小时 | 💾 第2个检查点 |
| 60轮 | 9小时 | 💾 第3个检查点 |
| 80轮 | 12小时 | 💾 第4个检查点 |
| 100轮 | 15-18小时 | ✅ 完成 |

**建议：晚上8点启动 → 第二天上午11点-下午2点完成**

---

## 自动化特性

### 自动配置检测

脚本会自动检测：
- CPU核心数
- GPU型号和显存
- 系统内存

并根据配置自动调整：
- 并行进程数
- 批次大小
- 训练参数

### 智能启动

```bash
# 4核CPU → 自动启动2个collect进程
# 8核CPU → 自动启动3个collect进程
# 17核CPU → 自动启动5-7个collect进程
```

无需手动修改配置！

---

## 下一步

训练完成后：
1. 测试模型：`python play_with_ai.py`
2. 下载模型：从Jupyter文件浏览器下载
3. 决定是否继续训练到1000轮

---

## 更新日志

- 2026-03-08: 创建部署文档
- 支持自动配置检测
- 适配多种云服务器配置
