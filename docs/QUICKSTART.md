# 快速启动指南

## 环境要求

- Python 3.7+
- NVIDIA GPU + CUDA
- PyTorch 或 PaddlePaddle

## 安装

```bash
# PyTorch版本（推荐）
pip install torch numpy

# 或 PaddlePaddle版本
pip install paddlepaddle-gpu numpy
```

## 配置

编辑 `config.py`:

```python
'use_frame': 'pytorch',  # 选择 'pytorch' 或 'paddle'
'use_redis': False,       # False使用本地文件存储
```

## 三种运行模式

### 1. 人机对弈（测试）

```bash
# GUI界面
python UIplay.py

# 命令行版本
python play_with_ai.py
```

### 2. 训练AI

**需要开启多个终端窗口：**

```bash
# 终端1-3: 数据收集（可多开）
python collect.py

# 终端4: 训练（只开1个）
python train.py
```

### 3. 查看训练进度

训练过程会显示：
- 当前批次: `batch i: 100`
- 损失值: `loss: 2.345`
- KL散度: `kl: 0.015`

## 文件说明

```
aichess/
├── collect.py          # 运行这个收集数据
├── train.py            # 运行这个训练模型
├── UIplay.py           # 运行这个和AI下棋
├── config.py           # 配置参数
├── pytorch_net.py      # 神经网络（PyTorch）
└── game.py             # 象棋规则
```

## 常见问题

**Q: 训练需要多长时间？**
A: 建议训练24小时以上，AI水平会明显提升

**Q: 可以只用CPU吗？**
A: 不可以，MCTS需要大量神经网络推理，必须用GPU

**Q: 如何知道AI变强了？**
A: 运行 `play_with_ai.py` 测试，或观察训练loss下降

## 下一步

- 查看 [TRAINING_GUIDE.md](TRAINING_GUIDE.md) 了解详细架构
- 调整 `config.py` 中的参数优化性能
- 阅读 `pytorch_net.py` 了解网络结构

## 视频教程

- [B站教学视频](https://www.bilibili.com/video/BV183411g7GX)
- [知乎文章](https://zhuanlan.zhihu.com/p/528824058)
