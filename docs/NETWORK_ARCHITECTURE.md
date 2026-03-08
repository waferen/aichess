# 网络结构详解

## 概述

本项目使用深度残差网络实现AlphaZero算法，网络采用双头结构：
- **策略头**: 输出走法概率分布
- **价值头**: 输出局面胜率评估

## 网络架构图

```
输入层 [batch, 9, 10, 9]
        ↓
    ┌───────────────┐
    │  初始卷积块    │
    │ Conv 9→256    │
    │ 3×3, padding=1│
    │ + BN + ReLU   │
    └───────────────┘
        ↓
    ┌───────────────┐
    │               │
    │  残差块 ×7    │
    │               │
    │  ┌─────────┐  │
    │  │ Conv2d  │  │
    │  │ BN+ReLU │  │
    │  │ Conv2d  │  │
    │  │   +BN   │  │
    │  │ Add+ReLU│  │
    │  └─────────┘  │
    └───────────────┘
        ↓
    ┌───────────┴───────────┐
    ↓                       ↓
策略头                   价值头
    ↓                       ↓
┌─────────┐           ┌─────────┐
│ Conv    │           │ Conv    │
│256→16   │           │256→8    │
│1×1      │           │1×1      │
└─────────┘           └─────────┘
    ↓                       ↓
┌─────────┐           ┌─────────┐
│ BN+ReLU │           │ BN+ReLU │
└─────────┘           └─────────┘
    ↓                       ↓
┌─────────┐           ┌─────────┐
│Flatten  │           │Flatten  │
│1440     │           │720      │
└─────────┘           └─────────┘
    ↓                       ↓
┌─────────┐           ┌─────────┐
│ FC      │           │ FC      │
│1440→2086│           │720→256  │
└─────────┘           └─────────┘
    ↓                       ↓
┌─────────┐           ┌─────────┐
│LogSoftmax│          │ ReLU    │
└─────────┘           └─────────┘
    ↓                       ↓
    ↓                   ┌─────────┐
    ↓                   │ FC      │
[batch, 2086]           │256→1    │
策略概率                └─────────┘
                           ↓
                       ┌─────────┐
                       │ Tanh    │
                       └─────────┘
                           ↓
                      [batch, 1]
                      局面价值
```

## 输入表示

### 张量形状

```
[batch_size, 9, 10, 9]
    ↓        ↓  ↓  ↓
  批次    通道 高  宽
```

### 通道含义

棋盘状态使用9个通道表示，每个通道是10×9的矩阵：

```
通道 0: 红车  (位置为1，否则为0)
通道 1: 红马
通道 2: 红象
通道 3: 红士
通道 4: 红帅
通道 5: 红炮
通道 6: 红兵
通道 7: 黑车  (位置为-1，否则为0)
通道 8: 黑马
通道 9: 黑象
通道10: 黑士
通道11: 黑帅
通道12: 黑炮
通道13: 黑兵
```

### 示例

假设红方帅在(0,4)位置：

```python
# 通道4，第0行第4列
state[4][0][4] = 1

# 黑方将在(9,4)位置
state[11][9][4] = -1
```

## 残差块详解

### 结构

```python
class ResBlock(nn.Module):
    def __init__(self, num_filters=256):
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(256)

    def forward(self, x):
        # 主路径
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)

        # 残差连接
        y = x + y
        return F.relu(y)
```

### 优势

- **梯度流动**: 残差连接避免梯度消失
- **特征复用**: 保留低层特征
- **深度网络**: 支持更深的网络结构

## 策略头 (Policy Head)

### 目标

预测所有合法走法的概率分布，指导MCTS搜索。

### 网络结构

```python
# 1. 降维卷积
conv(256 → 16 channels, kernel_size=1×1)
batch_norm + relu
# 输出: [batch, 16, 10, 9]

# 2. 展平
flatten → [batch, 16×10×9] = [batch, 1440]

# 3. 全连接层
linear(1440 → 2086)
# 输出: [batch, 2086]

# 4. 激活函数
log_softmax
```

### 输出含义

输出2086维向量，每个元素对应一种可能的走法：

```
索引     走法    说明
0        0010    (0,0) → (1,0)
1        0011    (0,0) → (1,1)
...
2085     9878    (9,8) → (7,8)
```

### 训练

使用交叉熵损失：

```python
policy_loss = -mean(sum(MCTS_probs * log(action_probs)))
```

- 目标: MCTS搜索得到的走法概率分布
- 预测: 神经网络输出的概率分布
- 目标: 使两者尽可能接近

## 价值头 (Value Head)

### 目标

评估当前局面的胜率，用于指导MCTS的节点评估。

### 网络结构

```python
# 1. 降维卷积
conv(256 → 8 channels, kernel_size=1×1)
batch_norm + relu
# 输出: [batch, 8, 10, 9]

# 2. 展平
flatten → [batch, 8×10×9] = [batch, 720]

# 3. 第一层全连接
linear(720 → 256)
relu
# 输出: [batch, 256]

# 4. 第二层全连接
linear(256 → 1)
# 输出: [batch, 1]

# 5. 激活函数
tanh
```

### 输出含义

输出标量 v ∈ [-1, 1]：

```
值      含义
+1.0    红方必胜
+0.5    红方优势
 0.0    均势
-0.5    黑方优势
-1.0    黑方必胜
```

### 训练

使用均方误差损失：

```python
value_loss = MSE(predicted_value, actual_winner)
```

- 目标: 实际游戏结果（1/2/-1）
- 预测: 网络输出的价值
- 目标: 准确预测最终结果

## 损失函数

### 总损失

```python
total_loss = policy_loss + value_loss
```

### 各部分作用

| 损失项 | 公式 | 作用 |
|--------|------|------|
| policy_loss | -Σp·log(q) | 使策略接近MCTS |
| value_loss | MSE(v, z) | 准确评估局面 |

### 权衡

- 两者权重相等 (1:1)
- 策略损失主导搜索方向
- 价值损失提高评估准确性

## 训练过程

### 前向传播

```python
# 1. 输入状态
state = [batch, 9, 10, 9]

# 2. 特征提取
x = conv_block(state)  # [batch, 256, 10, 9]
for res_block in res_blocks:
    x = res_block(x)    # [batch, 256, 10, 9]

# 3. 策略头
policy = policy_head(x)  # [batch, 2086]

# 4. 价值头
value = value_head(x)    # [batch, 1]
```

### 反向传播

```python
# 1. 计算损失
loss = policy_loss + value_loss

# 2. 反向传播
loss.backward()

# 3. 更新参数
optimizer.step()
```

## 性能优化

### 半精度训练 (FP16)

```python
from torch.cuda.amp import autocast

with autocast():
    log_act_probs, value = self.policy_value_net(state)
```

- 减少显存占用
- 加速推理和训练
- 精度损失可忽略

### 数据增强

```python
# 水平翻转
state_flip = state.transpose([1, 2, 0])
for i in range(10):
    for j in range(9):
        state_flip[i][j] = state[i][8-j]
```

- 数据集翻倍
- 加速训练
- 提高泛化能力

## 超参数调优

### 网络结构参数

```python
num_channels = 256      # 特征通道数
num_res_blocks = 7      # 残差块数量
```

| 参数 | 默认值 | 调优建议 |
|------|--------|----------|
| num_channels | 256 | 128/256/512，显存充足可增大 |
| num_res_blocks | 7 | 5/7/10，更深更强但更慢 |

### 训练参数

```python
learning_rate = 1e-3    # 初始学习率
l2_const = 2e-3         # L2正则化
batch_size = 512        # 批次大小
epochs = 5              # 每次更新轮数
```

## 网络变体

### 轻量级版本

```python
num_channels = 128
num_res_blocks = 5
```

- 适合快速实验
- 显存需求减半
- 强度略有下降

### 高强度版本

```python
num_channels = 512
num_res_blocks = 10
```

- 更强棋力
- 需要更多显存
- 训练更慢

## 可视化

### 特征图

第k个残差块输出的特征图：

```python
feature_map = x[0, k, :, :]  # [10, 9]
```

### 注意力权重

策略头对各个位置的关注：

```python
attention = policy_conv_weight  # [16, 256, 1, 1]
```

## 实现细节

### PyTorch实现

位置: `pytorch_net.py`

```python
class Net(nn.Module):
    def __init__(self, num_channels=256, num_res_blocks=7):
        # ... 网络定义

    def forward(self, x):
        # ... 前向传播
        return policy, value
```

### PaddlePaddle实现

位置: `paddle_net.py`

结构相同，API略有差异。

## 调试技巧

### 检查输出

```python
policy, value = net(test_input)
print(f"Policy shape: {policy.shape}")  # [8, 2086]
print(f"Value shape: {value.shape}")    # [8, 1]
```

### 验证概率

```python
policy_probs = np.exp(policy.numpy())
print(f"Sum: {np.sum(policy_probs[0])}")  # 应该接近1.0
```

### 检查价值范围

```python
print(f"Value range: [{value.min()}, {value.max()}]")  # 应该在[-1, 1]
```

## 参考资料

- [ResNet论文](https://arxiv.org/abs/1512.03385)
- [AlphaZero论文](https://arxiv.org/abs/1712.01815)
- [深度强化学习](https://book.douban.com/subject/35357436/)
