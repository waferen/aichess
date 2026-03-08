# AIChess 代码结构详解

> 基于 AlphaZero 算法打造的中国象棋 AI 项目

## 📁 项目整体架构

```
aichess/
├── 核心训练模块
│   ├── collect.py          # 自我对弈数据收集
│   ├── train.py            # 模型训练
│   └── config.py           # 配置文件
│
├── 游戏逻辑
│   ├── game.py             # 中国象棋游戏逻辑实现
│   ├── mcts.py             # 蒙特卡洛树搜索（带神经网络）
│   └── mcts_pure.py        # 纯蒙特卡洛树搜索（无神经网络）
│
├── 神经网络
│   ├── pytorch_net.py      # PyTorch 版本的策略价值网络
│   └── paddle_net.py       # PaddlePaddle 版本的策略价值网络
│
├── 人机对弈
│   ├── play_with_ai.py     # 命令行版人机对弈
│   └── UIplay.py           # GUI 图形界面人机对弈
│
└── 辅助工具
    ├── my_redis.py         # Redis 数据存储
    ├── zip_array.py        # 数据压缩工具
    └── models/             # 模型存放目录
```

## 🔧 核心模块详解

### 1. **config.py** - 配置中心

所有训练和游戏参数的集中配置文件。

```python
# 关键配置参数：
- 'kill_action': 30          # 和棋回合数
- 'dirichlet': 0.2           # 探索噪声参数
- 'play_out': 1200           # MCTS 每次移动的模拟次数
- 'c_puct': 5                # PUCT 算法中 U 的权重
- 'buffer_size': 100000      # 经验回放池大小
- 'batch_size': 512          # 训练批次大小
- 'epochs': 5                # 每次更新的训练轮数
- 'game_batch_num': 3000     # 训练更新总次数
- 'use_frame': 'pytorch'     # 深度学习框架选择 (pytorch/paddle)
- 'use_redis': False         # 是否使用 Redis 存储数据
```

### 2. **game.py** - 象棋游戏逻辑

实现完整的中国象棋规则和游戏逻辑（约 1300 行）。

**核心特性：**
- **棋盘表示**：使用 10×9 的二维数组表示棋盘状态
- **棋子编码**：使用 7 维 one-hot 向量表示每个棋子
  - 红方：车、马、象、士、帅、炮、兵
  - 黑方：对应棋子（使用负值）
  - 空位：全零向量

**核心类：**

```python
class Board:
    """管理棋盘状态和走子规则"""
    - get_legal_moves()      # 获取所有合法走法
    - do_move()              # 执行走子
    - game_end()             # 判断游戏是否结束
    - get_state()            # 获取当前状态（用于神经网络输入）

class Game:
    """管理游戏流程"""
    - start_self_play()      # 自我对弈
    - play_vs_ai()           # 人机对弈
```

**特殊规则实现：**
- 长将、长捉检测
- 困毙判负
- 将军检测
- 三次重复判和

### 3. **mcts.py** - 蒙特卡洛树搜索

实现带神经网络指导的 MCTS 算法。

**TreeNode 类：**
```python
class TreeNode:
    """MCTS 树的节点"""
    - _parent: 父节点
    - _children: 子节点字典 {action: TreeNode}
    - _n_visits: 访问次数
    - _Q: 动作价值（平均收益）
    - _u: 置信上限
    - _P: 先验概率（来自神经网络）

    关键方法：
    - expand()        # 展开节点
    - select()        # 选择最佳子节点（PUCT 算法）
    - update()        # 更新节点统计信息
```

**MCTSPlayer 类：**
```python
class MCTSPlayer:
    """使用神经网络指导的 MCTS 玩家"""
    - get_action()        # 获取最佳动作
    - evaluate()          # 使用神经网络评估局面
    - simulate()          # MCTS 模拟
```

**PUCT 算法：**
```
U = c_puct * P * sqrt(父节点访问次数) / (1 + 当前节点访问次数)
选择得分 = Q + U
```

### 4. **pytorch_net.py** - 策略价值网络

实现双头网络，同时输出策略（走子概率）和价值（局面评分）。

**网络架构：**
```
输入: [batch, 9, 10, 9]
  ↓ 9 个历史棋盘状态（当前 + 8 个历史）

[卷积层 + 批归一化 + ReLU]
  ↓ Conv2d(9 → 256, kernel_size=3×3)

[7 个残差块]
  ↓ 每个残差块包含：
    - Conv2d(256 → 256, 3×3)
    - BatchNorm2d
    - ReLU
    - 残差连接

  分叉成两个头：

  ├─ 策略头 (Policy Head)
  │   ↓
  │ Conv2d(256 → 16, 1×1)
  │   ↓
  │ Flatten + Dense(16×10×9 → 2086)
  │   ↓
  │ LogSoftmax  # 输出每个动作的概率
  │
  └─ 价值头 (Value Head)
      ↓
      Conv2d(256 → 4, 1×1)
      ↓
      Flatten + Dense(4×10×9 → 64)
      ↓
      ReLU + Dense(64 → 1)
      ↓
      Tanh  # 输出局面评分 [-1, 1]
```

**输出说明：**
- **策略输出**：2086 维向量，表示所有可能的走子概率
- **价值输出**：标量值，-1 表示黑方必胜，1 表示红方必胜

### 5. **collect.py** - 自我对弈数据收集

通过自我对弈产生训练数据。

```python
class CollectPipeline:
    """自我对弈数据收集流程"""

    def collect(self):
        """收集自我对弈数据"""
        while True:
            # 1. 加载最新模型
            self.load_model()

            # 2. 自我对弈
            for _ in range(self.n_games):
                data = self.game.start_self_play(
                    self.mcts_player,
                    temp=self.temperature
                )
                # 3. 存储到经验池
                self.data_buffer.extend(data)

            # 4. 保存数据（文件或 Redis）
            self.save_data()

            # 5. 定期检查并加载新模型
            self.check_and_load_new_model()
```

**数据格式：**
```python
(state, mcts_probs, winner)
- state: [9, 10, 9] 棋盘状态
- mcts_probs: [2086] MCTS 搜索得到的动作概率分布
- winner: +1(红胜) / -1(黑胜) / 0(和棋)
```

### 6. **train.py** - 模型训练

使用收集的数据训练神经网络。

```python
class TrainPipeline:
    """模型训练流程"""

    def run(self):
        """主训练循环"""
        for i in range(self.game_batch_num):
            # 1. 从经验池采样数据
            training_data = self.get_training_data()

            # 2. 训练神经网络
            for epoch in range(self.epochs):
                loss, entropy, kl = self.policy_value_net.train(
                    states, mcts_probs, winners
                )

            # 3. KL 散度自适应调整学习率
            if kl > self.kl_targ * 2.5:
                self.lr_multiplier /= 2
            elif kl < self.kl_targ * 0.5:
                self.lr_multiplier *= 2

            # 4. 定期评估模型
            if i % self.check_freq == 0:
                self.evaluate_and_save_model()
```

**损失函数：**
```python
loss = value_loss + policy_loss + l2 regularization

value_loss = (预测价值 - 真实 winner)²
policy_loss = -∑(MCTS 概率 × log(预测概率))
```

### 7. **play_with_ai.py** - 命令行人机对弈

简单的命令行界面人机对弈。

```python
# 运行方式：
python play_with_ai.py

# 功能：
- 选择执红/执黑
- 输入坐标走子（如 "e2e4"）
- 显示棋盘状态
- AI 自动走子
```

### 8. **UIplay.py** - GUI 人机对弈

基于 Tkinter 的图形化人机对弈界面。

**特性：**
- 可视化棋盘
- 鼠标点击走子
- 悔棋功能
- 音效支持（bgm 目录）
- 难度调节

### 9. **mcts_pure.py** - 纯 MCTS

不使用神经网络的纯 MCTS 实现，用于：
- 早期训练阶段评估模型
- 对比实验
- 不需要 GPU 的场景

### 10. **辅助模块**

**my_redis.py**
```python
# Redis 数据存储，用于多进程数据共享
def get_redis_cli():
    """获取 Redis 连接"""
    return redis.Redis(
        host=CONFIG['redis_host'],
        port=CONFIG['redis_port'],
        db=CONFIG['redis_db']
    )
```

**zip_array.py**
```python
# 数据压缩工具，减少存储空间
def compress_data(data):
    """压缩训练数据"""
    ...

def decompress_data(compressed_data):
    """解压训练数据"""
    ...
```

## 🔄 训练流程（多进程架构）

```
┌─────────────────────────────────────────────────────┐
│                    训练流程                           │
└─────────────────────────────────────────────────────┘

[终端 1-多个] python collect.py
        ↓
    自我对弈（MCTS + 当前模型）
        ↓
    产生数据：(state, mcts_probs, winner)
        ↓
    存储到经验池（文件或 Redis）
        ↓
    定期检查并加载最新模型
         ↺ (循环)

[终端 2-单个] python train.py
        ↓
    从经验池采样数据（batch_size=512）
        ↓
    训练神经网络
        ↓
    计算损失并更新参数
        ↓
    KL 散度调整学习率
        ↓
    定期保存模型（每 100 个 step）
        ↓
         ↺ (循环)

         ↑
         │ 数据流
         │ 模型流
```

## 🎮 使用方式

### 1. 训练阶段

```bash
# 步骤 1：启动数据收集（可开启多个终端）
python collect.py

# 步骤 2：启动模型训练（只开启一个终端）
python train.py
```

**注意事项：**
- 需要英伟达 GPU（支持 CUDA）
- 推荐使用 PyTorch 版本（性能更好）
- collect.py 可以多开以加速数据收集
- train.py 只能开一个

### 2. 人机对弈

```bash
# 命令行版
python play_with_ai.py

# GUI 版（推荐）
python UIplay.py
```

### 3. 配置切换

```python
# 在 config.py 中修改：
CONFIG['use_frame'] = 'pytorch'   # 使用 PyTorch
CONFIG['use_frame'] = 'paddle'    # 使用 PaddlePaddle

CONFIG['use_redis'] = True        # 使用 Redis 存储数据
CONFIG['use_redis'] = False       # 使用文件存储数据
```

## 🌟 技术亮点

### 1. AlphaZero 算法
- **无需人类棋谱**：完全通过自我对弈学习
- **策略迭代**：模型不断自我提升
- **MCTS + 神经网络**：结合搜索和学习的优势

### 2. 双框架支持
- **PyTorch**：主流框架，社区活跃
- **PaddlePaddle**：国产框架，对中文支持好
- **无缝切换**：只需修改配置文件

### 3. 多进程训练
- **数据收集和训练分离**：提高效率
- **可扩展性强**：可根据硬件资源调整进程数
- **数据共享**：支持 Redis 分布式存储

### 4. 完整的象棋规则
- **基础规则**：所有棋子的走法
- **特殊规则**：长将、长捉、困毙
- **胜负判定**：将军、无子可动

### 5. 深度学习技术
- **残差网络**：7 层残差块提取特征
- **双头输出**：策略 + 价值
- **批归一化**：加速训练
- **KL 散度自适应**：自动调整学习率

## 📊 数据流图

```
┌──────────┐
│   Board  │  当前棋盘状态
└────┬─────┘
     │ state
     ↓
┌──────────┐
│  Neural  │  神经网络预测
│  Network │  P (先验概率) + v (价值)
└────┬─────┘
     │ P
     ↓
┌──────────┐
│   MCTS   │  蒙特卡洛树搜索
│  Player  │  改进的概率分布 π
└────┬─────┘
     │ action
     ↓
┌──────────┐
│   Board  │  新的棋盘状态
│  update  │
└──────────┘
     │ (state, π, winner)
     ↓
┌──────────┐
│  Buffer  │  经验池
└────┬─────┘
     │ 采样
     ↓
┌──────────┐
│   Train  │  训练更新网络
└──────────┘
```

## 🔑 关键概念

### MCTS（蒙特卡洛树搜索）
- **选择（Selection）**：使用 PUCT 算法选择最有希望的节点
- **展开（Expansion）**：展开叶子节点
- **模拟（Simulation）**：使用神经网络快速评估
- **回溯（Backpropagation）**：更新路径上所有节点的统计信息

### PUCT 算法
```
Q + U = Q + c_puct × P × √(N_parent) / (1 + N_child)
```
- **Q**：动作价值（平均收益）
- **U**：探索项（鼓励探索）
- **P**：先验概率（来自神经网络）
- **c_puct**：探索常数

### 温度参数
- **temp = 1**：初期，鼓励探索
- **temp → 0**：后期，选择最优动作

## 📚 相关资源

- [AlphaZero 论文](https://arxiv.org/abs/1712.01815)
- [中国象棋规则](https://zh.wikipedia.org/wiki/中国象棋)
- [PyTorch 文档](https://pytorch.org/docs/)
- [PaddlePaddle 文档](https://www.paddlepaddle.org.cn/)

## 🎯 总结

这是一个**结构清晰、功能完整**的强化学习项目，特别适合：
- 学习 AlphaZero 算法
- 理解深度强化学习实践
- 研究中国象棋 AI
- 多进程分布式训练

代码模块化良好，各个组件职责明确，便于理解和扩展！
