# AIChess - 中国象棋AI

基于AlphaZero算法的中国象棋AI项目，使用MCTS（蒙特卡洛树搜索）和深度神经网络进行训练。

## 项目特点

- **MCTS搜索**: 使用蒙特卡洛树搜索进行棋局推演
- **双神经网络**: 支持策略网络（Policy Network）和价值网络（Value Network）
- **自对弈训练**: 通过自我对弈生成训练数据
- **双框架支持**: 支持 PaddlePaddle 和 PyTorch 两种深度学习框架

## 项目结构

```
aichess/
├── config.py          # 配置文件
├── game.py            # 游戏规则实现
├── mcts.py            # MCTS搜索算法
├── mcts_pure.py       # 纯MCTS（无神经网络）
├── paddle_net.py      # PaddlePaddle网络实现
├── pytorch_net.py     # PyTorch网络实现
├── train.py           # 训练入口
├── collect.py         # 数据收集
├── UIplay.py          # 图形界面
├── play_with_ai.py    # 人机对弈
└── zip_array.py       # 数据压缩工具
```

## 环境要求

- Python 3.8+
- PaddlePaddle 或 PyTorch
- NumPy, Pickle

## 快速开始

### 训练模型

```bash
bash autodl_one_click.sh
```

### 人机对弈

```bash
python play_with_ai.py
```

### 配置说明

在 `config.py` 中可修改以下参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| play_out | 每次移动模拟次数 | 200 |
| c_puct | PUCT探索常数 | 5 |
| batch_size | 批大小 | 512 |
| game_batch_num | 训练局数 | 200 |
| use_frame | 使用的深度学习框架 | pytorch |

## 技术细节

- **棋盘表示**: 9×10 的特征平面
- **策略输出**: 2086种合法着法
- **网络结构**: 13层残差网络 + 全局特征提取
- **训练算法**: AlphaZero风格的自我对弈强化学习