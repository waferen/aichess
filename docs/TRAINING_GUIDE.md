# 中国象棋AI训练指南

本文档详细介绍如何运行和训练基于AlphaZero算法的中国象棋AI。

## 目录

- [系统架构](#系统架构)
- [环境准备](#环境准备)
- [快速开始](#快速开始)
- [训练流程](#训练流程)
- [网络结构详解](#网络结构详解)
- [配置说明](#配置说明)
- [文件说明](#文件说明)

---

## 系统架构

本项目采用AlphaZero算法，通过自我对弈不断改进AI水平。系统采用多进程架构：

```
数据收集进程 (可多个)          训练进程 (1个)
     ↓                              ↓
  自我对弈                     读取训练数据
     ↓                              ↓
  MCTS搜索                    神经网络训练
     ↓                              ↓
  生成训练样本                  保存最新模型
     ↓                              ↓
  存储到共享缓冲区  ←──────────  更新模型
```

### 核心组件

1. **神经网络**: 双头网络，输出策略概率和局面价值
2. **蒙特卡洛树搜索 (MCTS)**: 结合神经网络进行高效搜索
3. **游戏逻辑**: 完整的中国象棋规则实现
4. **数据缓冲区**: 存储自我对弈生成的训练数据

---

## 环境准备

### 硬件要求

- **GPU**: 必须使用NVIDIA显卡
- **内存**: 建议8GB以上
- **存储**: 至少5GB可用空间

### 软件依赖

#### PyTorch版本（推荐）

```bash
pip install torch numpy
```

或使用GPU版本：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### PaddlePaddle版本

```bash
pip install paddlepaddle-gpu
```

### 其他依赖

```bash
pip install numpy redis  # 如果使用Redis存储
```

---

## 快速开始

### 1. 配置框架选择

编辑 `config.py` 文件：

```python
CONFIG = {
    'use_frame': 'pytorch',  # 或 'paddle'
    'use_redis': False,      # True使用Redis，False使用本地文件
    # ... 其他配置
}
```

### 2. 人机对弈测试

运行GUI界面：

```bash
python UIplay.py
```

或命令行版本：

```bash
python play_with_ai.py
```

### 3. 开始训练

**步骤1**: 启动数据收集进程（可开多个）

```bash
python collect.py
```

**步骤2**: 启动训练进程（只开1个）

```bash
python train.py
```

---

## 训练流程

### 多进程训练架构

本项目采用生产者-消费者模式：

```
终端1: python collect.py
终端2: python collect.py  # 可选，加速数据收集
...
终端N: python train.py    # 只需要一个
```

### 数据收集 (collect.py)

1. **加载最新模型**: 每局游戏开始前从磁盘加载最新模型
2. **自我对弈**: 使用MCTS进行自我对弈
3. **数据增强**: 通过水平翻转扩充数据集（翻倍）
4. **存储数据**: 保存到共享缓冲区

**数据格式**:
- 状态: [9, 10, 9] 张量，表示当前棋盘
- 策略: 2086维向量，MCTS搜索得到的走法概率
- 胜者: 1（红胜）、2（黑胜）、-1（平局）

### 模型训练 (train.py)

1. **读取数据**: 从缓冲区读取最新训练数据
2. **采样批次**: 随机采样512个样本
3. **网络训练**: 执行5个epoch的训练
4. **KL散度控制**: 自适应调整学习率
5. **保存模型**: 每10分钟保存一次模型

### 训练监控

训练过程中会输出以下信息：

```
batch i: 100, episode_len: 156
step i 100:
kl:0.01523, lr_multiplier:1.200, loss:2.345, entropy:3.456,
explained_var_old:0.123456789, explained_var_new:0.234567890
```

- **kl**: 策略变化程度，用于自适应学习率
- **lr_multiplier**: 学习率倍数
- **loss**: 总损失
- **entropy**: 策略熵，衡量探索程度
- **explained_var**: 方差解释率，衡量价值函数质量

---

## 网络结构详解

### 整体架构

神经网络采用残差网络结构，输入棋盘状态，输出策略概率和价值评估。

```
输入: [batch, 9, 10, 9]
  ↓
卷积层 + BN + ReLU
  ↓
7个残差块
  ↓
  ├─→ 策略头 → [batch, 2086]
  │
  └─→ 价值头 → [batch, 1]
```

### 输入表示

**形状**: [9, 10, 9]

- 9个通道，每个通道是10×9的棋盘
- 通道0-6: 红方6种棋子（车、马、象、士、帅、炮、兵）
- 通道7-13: 黑方6种棋子
- 通道8: 历史信息（最近4步）

### 残差块结构

```python
class ResBlock:
    def __init__(self, num_filters=256):
        self.conv1 = Conv2d(256, 256, 3×3, padding=1)
        self.bn1 = BatchNorm2d(256)
        self.relu1 = ReLU()
        self.conv2 = Conv2d(256, 256, 3×3, padding=1)
        self.bn2 = BatchNorm2d(256)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        return self.relu2(x + y)  # 残差连接
```

### 策略头 (Policy Head)

预测走法概率分布：

```python
conv(256 → 16, 1×1) → BN → ReLU → Flatten → FC(1440 → 2086) → LogSoftmax
```

- 输出: 2086维向量（所有可能的合法走法）
- 使用交叉熵损失训练

### 价值头 (Value Head)

评估当前局面胜率：

```python
conv(256 → 8, 1×1) → BN → ReLU → Flatten → FC(720 → 256) →
ReLU → FC(256 → 1) → Tanh
```

- 输出: [-1, 1]区间的标量
- 1: 红方必胜
- -1: 黑方必胜
- 0: 均势
- 使用MSE损失训练

### 损失函数

```python
total_loss = policy_loss + value_loss

policy_loss = -mean(sum(MCTS_probs * log(action_probs)))
value_loss = MSE(predicted_value, actual_winner)
```

---

## 配置说明

### config.py 参数详解

```python
CONFIG = {
    # 对弈参数
    'kill_action': 30,        # 和棋回合数
    'play_out': 1200,         # MCTS每次移动的模拟次数
    'c_puct': 5,              # PUCT算法的探索常数

    # 训练参数
    'buffer_size': 100000,    # 经验池最大容量
    'batch_size': 512,        # 训练批次大小
    'epochs': 5,              # 每次更新的训练轮数
    'kl_targ': 0.02,          # KL散度目标值
    'game_batch_num': 3000,   # 总训练批次

    # 模型参数
    'use_frame': 'pytorch',   # 框架选择: 'pytorch' 或 'paddle'
    'paddle_model_path': 'current_policy.model',
    'pytorch_model_path': 'current_policy.pkl',
    'train_data_buffer_path': 'train_data_buffer.pkl',

    # 系统参数
    'train_update_interval': 600,  # 模型更新间隔（秒）
    'use_redis': False,            # 是否使用Redis
}
```

### 参数调优建议

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|----------|------|
| play_out | 1200 | 800-1600 | 越大越强但越慢 |
| batch_size | 512 | 256-1024 | 根据显存调整 |
| epochs | 5 | 3-10 | 训练轮数 |
| buffer_size | 100000 | 50000-200000 | 经验池大小 |

---

## 文件说明

### 核心文件

| 文件 | 功能 | 说明 |
|------|------|------|
| `collect.py` | 数据收集 | 自我对弈生成训练数据 |
| `train.py` | 模型训练 | 神经网络训练和更新 |
| `game.py` | 游戏逻辑 | 中国象棋规则实现 |
| `mcts.py` | 蒙特卡洛树 | 结合神经网络的搜索算法 |
| `mcts_pure.py` | 纯MCTS | 不使用神经网络的基线版本 |
| `pytorch_net.py` | PyTorch网络 | 神经网络定义（PyTorch版） |
| `paddle_net.py` | Paddle网络 | 神经网络定义（Paddle版） |
| `config.py` | 配置文件 | 所有超参数设置 |

### 交互文件

| 文件 | 功能 |
|------|------|
| `UIplay.py` | GUI界面人机对弈 |
| `play_with_ai.py` | 命令行人机对弈 |

### 辅助文件

| 文件 | 功能 |
|------|------|
| `my_redis.py` | Redis连接管理 |
| `zip_array.py` | 数据压缩工具 |

### 生成文件

训练过程中会生成以下文件：

- `current_policy.pkl` / `current_policy.model`: 最新模型
- `train_data_buffer.pkl`: 训练数据缓冲区
- `models/`: 历史模型保存目录

---

## 训练技巧

### 加速数据收集

1. **多开collect.py**: 根据GPU显存开2-4个进程
2. **调整play_out**: 降低到800可加速但会减弱强度
3. **使用Redis**: 多进程共享数据更高效

### 提升训练效果

1. **增加数据量**: 训练前生成10万+样本
2. **调整学习率**: 通过KL散度自动调整
3. **定期评估**: 使用pure MCTS测试模型强度

### 常见问题

**Q: 训练loss不下降？**
A: 检查数据量是否足够，尝试降低学习率

**Q: 内存不足？**
A: 减小buffer_size或batch_size

**Q: 训练很慢？**
A: 确保使用GPU，减少play_out次数

---

## 进阶使用

### 评估模型强度

```python
# 在train.py中取消注释评估代码
win_ratio = self.policy_evaluate(n_games=10)
```

### 调整网络结构

编辑 `pytorch_net.py` 中的网络参数：

```python
class Net(nn.Module):
    def __init__(self, num_channels=256,      # 特征通道数
                 num_res_blocks=7):           # 残差块数量
```

### 使用分布式训练

1. 设置 `use_redis = True`
2. 在多台机器上运行collect.py
3. 一台机器运行train.py

---

## 参考文献

本项目基于以下资源开发：

1. [AlphaZero论文](https://arxiv.org/abs/1712.01815)
2. [中国象棋cchess零](https://zhuanlan.zhihu.com/p/34433581)
3. [五子棋AlphaZero](https://github.com/junxiaosong/AlphaZero_Gomoku)
4. [边做边学深度强化学习](https://book.douban.com/subject/35357436/)

---

## 许可证

详见 [LICENSE](LICENSE) 文件
