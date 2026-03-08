# MCTS（蒙特卡洛树搜索）算法详解

## 目录
- [算法概述](#算法概述)
- [核心概念](#核心概念)
- [代码结构](#代码结构)
- [算法流程](#算法流程)
- [关键组件详解](#关键组件详解)
- [参数配置](#参数配置)

---

## 算法概述

MCTS（Monte Carlo Tree Search，蒙特卡洛树搜索）是一种用于决策过程的启发式搜索算法。在AlphaZero架构中，MCTS与神经网络结合，用于：
- 搜索最佳走法
- 生成训练数据（状态-动作概率对）
- 评估当前局面

### 核心思想
通过多次模拟（playout）构建搜索树，每次模拟包含四个步骤：**选择、扩展、模拟、回溯**（Selection, Expansion, Simulation, Backpropagation）。

---

## 核心概念

### PUCT算法
本项目使用PUCT（Predictor + UCB applied to Trees）算法，结合了：
- **先验概率（P）**：来自神经网络的策略网络输出
- **动作价值（Q）**：累积的平均奖励
- **访问次数（N）**：节点被访问的次数
- **探索常数（c_puct）**：控制探索vs利用的平衡

**PUCT公式：**
```
U = c_puct × P × √(N_parent) / (1 + N_child)
Score = Q + U
```

---

## 代码结构

```
mcts.py
├── TreeNode          # 树节点类
│   ├── expand()      # 扩展子节点
│   ├── select()      # 选择最优子节点
│   ├── update()      # 更新节点值
│   └── update_recursive()  # 递归更新
│
├── MCTS              # MCTS算法类
│   ├── _playout()    # 单次模拟
│   ├── get_move_probs()  # 获取动作概率
│   └── update_with_move()  # 更新树根
│
└── MCTSPlayer        # AI玩家类
    ├── get_action()  # 获取走法
    └── reset_player()  # 重置玩家
```

---

## 算法流程

### 1. 初始化
```python
# 创建根节点
self._root = TreeNode(None, 1.0)

# 参数设置
c_puct = 5        # 探索常数
n_playout = 1200  # 模拟次数
```

### 2. 单次模拟（_playout）

**步骤1：选择（Selection）**
```python
while not node.is_leaf():
    # 选择 Q+U 最大的子节点
    action, node = node.select(c_puct)
    state.do_move(action)
```
- 从根节点开始
- 递归选择最优子节点（Q+U最大）
- 直到到达叶节点

**步骤2：评估与扩展（Evaluation & Expansion）**

```python
# 使用神经网络评估叶节点
action_probs, leaf_value = self._policy(state)

# 如果游戏未结束，扩展节点
if not end:
    node.expand(action_probs)
```
- 神经网络输出：(action_probs, leaf_value)
  - `action_probs`: 每个合法动作的概率（2086维向量）
  - `leaf_value`: 当前局面评分 [-1, 1]

**步骤3：回溯（Backpropagation）**
```python
# 递归更新路径上所有节点
node.update_recursive(-leaf_value)
```
- 从叶节点向上更新到根节点
- 注意：交替玩家时，值需要取反

### 3. 获取动作概率（get_move_probs）

```python
# 执行 n_playout 次模拟
for n in range(self._n_playout):
    state_copy = copy.deepcopy(state)
    self._playout(state_copy)

# 根据访问次数计算最终概率
act_probs = softmax(1.0/temp * log(visits))
```

**温度参数（temp）的作用：**
- `temp → 0`：选择访问次数最多的动作（利用）
- `temp = 1`：按访问次数比例选择（探索）
- `temp > 1`：增加探索，使分布更均匀

### 4. 树重用（update_with_move）

```python
# 移动根节点到选中的子节点
if last_move in self._root._children:
    self._root = self._root._children[last_move]
    self._root._parent = None
```
- 保留已搜索的子树
- 避免重复计算
- 显著提升效率

---

## 关键组件详解

### TreeNode（树节点）

**属性：**
```python
_parent      # 父节点
_children    # 子节点字典 {action: TreeNode}
_n_visits    # 访问次数
_Q           # 动作价值（平均奖励）
_u           # 置信上限
_P           # 先验概率（来自神经网络）
```

**核心方法：**

1. **expand() - 扩展节点**
```python
def expand(self, action_priors):
    for action, prob in action_priors:
        if action not in self._children:
            self._children[action] = TreeNode(self, prob)
```
- 为每个合法动作创建子节点
- 存储神经网络输出的先验概率

2. **select() - 选择子节点**
```python
def select(self, c_puct):
    return max(self._children.items(),
               key=lambda act_node: act_node[1].get_value(c_puct))
```
- 选择 Q+U 最大的子节点
- 贪心策略

3. **update() - 更新节点**
```python
def update(self, leaf_value):
    self._n_visits += 1
    self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits
```
- 增量式更新平均值
- 避免存储所有历史值

### MCTSPlayer（AI玩家）

**自我对弈模式：**
```python
if self._is_selfplay:
    # 添加Dirichlet噪声进行探索
    move = np.random.choice(
        acts,
        p=0.75*probs + 0.25*np.random.dirichlet(0.2 * np.ones(len(probs)))
    )
```
- 75% MCTS概率 + 25% Dirichlet噪声
- 增加探索多样性
- 生成更好的训练数据

**对战模式：**
```python
else:
    # 直接选择MCTS推荐的动作
    move = np.random.choice(acts, p=probs)
```

---

## 参数配置

### config.py 中的相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `play_out` | 1200 | 每次移动的MCTS模拟次数 |
| `c_puct` | 5 | PUCT探索常数 |
| `dirichlet` | 0.2 | Dirichlet噪声参数（自我对弈） |
| `temp` | 1.0 | 温度参数（早期探索） |

### 参数调优建议

**play_out（模拟次数）**
- 训练初期：800-1200（快速生成数据）
- 训练后期：1600-2000（提高质量）
- 对战模式：2000+（最强棋力）

**c_puct（探索常数）**
- 值越大，越依赖先验概率（探索）
- 值越小，越依赖动作价值（利用）
- 推荐范围：3-5

**dirichlet（噪声强度）**
- 只在自我对弈时使用
- 推荐范围：0.1-0.3
- 国际象棋：0.3，围棋：0.03，本项目：0.2

---

## 算法优势

1. **渐进式优化**
   - 不需要完整的搜索树
   - 计算资源集中在最有希望的分支

2. **神经网络引导**
   - 先验概率缩小搜索空间
   - 价值函数评估叶节点
   - 避免随机模拟的方差

3. **树重用**
   - 保留已计算的子树
   - 显著提升效率
   - 适合连续决策

4. **自博弈训练**
   - 生成高质量训练数据
   - 无需人类棋谱
   - 不断自我超越

---

## 示例流程图

```
开始MCTS搜索
    │
    ├─→ 选择: 从根节点沿最优路径下到叶节点
    │       (选择 Q+U 最大的子节点)
    │
    ├─→ 评估: 用神经网络评估叶节点
    │       输出: (动作概率, 局面价值)
    │
    ├─→ 扩展: 如果非终局，创建子节点
    │
    └─→ 回溯: 从叶节点向上更新所有节点
              更新 Q 值和访问次数 N
                │
                └─→ 重复 n_playout 次
                     │
                     └─→ 根据访问次数计算最终概率
                          │
                          └─→ 返回动作概率分布
```

---

## 性能特点

- **时间复杂度**: O(n_playout × branching_factor × depth)
- **空间复杂度**: O(tree_size)，通常远小于完整博弈树
- **并行性**: 不同模拟可以并行执行

---

## 参考资料

- [AlphaGo Zero论文](https://www.nature.com/articles/nature24270)
- [蒙特卡洛树搜索 - 维基百科](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
- [PUCT算法详解](https://www.chessprogramming.org/UCT)
