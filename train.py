"""使用收集到数据进行训练"""

import random
from collections import defaultdict, deque

import numpy as np
import pickle
import time
from datetime import datetime

import zip_array
from config import CONFIG
from game import Game, Board
from mcts import MCTSPlayer
from mcts_pure import MCTS_Pure


def get_timestamp():
    """获取当前时间戳"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log_info(msg):
    """输出带时间戳的日志"""
    print(f"[{get_timestamp()}] {msg}")

if CONFIG['use_redis']:
    import my_redis, redis
    import zip_array

if CONFIG['use_frame'] == 'paddle':
    from paddle_net import PolicyValueNet
elif CONFIG['use_frame'] == 'pytorch':
    from pytorch_net import PolicyValueNet
else:
    print('暂不支持您选择的框架')


# 定义整个训练流程
class TrainPipeline:

    def __init__(self, init_model=None):
        log_info("=" * 70)
        log_info("初始化训练流程")
        log_info("=" * 70)

        # 训练参数
        self.board = Board()
        self.game = Game(self.board)
        self.n_playout = CONFIG['play_out']
        self.c_puct = CONFIG['c_puct']
        self.learn_rate = 1e-3
        self.lr_multiplier = 1  # 基于KL自适应的调整学习率
        self.temp = 1.0
        self.batch_size = CONFIG['batch_size']  # 训练的batch大小
        self.epochs = CONFIG['epochs']  # 每次更新的train_step数量
        self.kl_targ = CONFIG['kl_targ']  # kl散度控制
        self.check_freq = 20  # 保存模型的频率（100轮训练时每20轮保存一次）
        self.game_batch_num = CONFIG['game_batch_num']  # 训练更新的次数
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 500

        log_info(f"训练配置:")
        log_info(f"  - Batch Size: {self.batch_size}")
        log_info(f"  - Epochs: {self.epochs}")
        log_info(f"  - KL Target: {self.kl_targ}")
        log_info(f"  - Learning Rate: {self.learn_rate}")
        log_info(f"  - MCTS Playouts: {self.n_playout}")
        log_info(f"  - 框架: {CONFIG['use_frame']}")
        log_info(f"  - 最大Batches: {self.game_batch_num}")

        if CONFIG['use_redis']:
            self.redis_cli = my_redis.get_redis_cli()
            log_info("使用Redis数据存储")
        self.buffer_size = maxlen=CONFIG['buffer_size']
        self.data_buffer = deque(maxlen=self.buffer_size)

        if init_model:
            try:
                self.policy_value_net = PolicyValueNet(model_file=init_model)
                log_info(f"✓ 已加载模型: {init_model}")
            except:
                # 从零开始训练
                log_info(f"✗ 模型路径不存在，从零开始训练")
                self.policy_value_net = PolicyValueNet()
        else:
            log_info("从零开始训练")
            self.policy_value_net = PolicyValueNet()

        log_info("=" * 70)


    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2 + 1,
                                          is_shown=1)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio


    def policy_updata(self):
        """更新策略价值网络"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        # print(mini_batch[0][1],mini_batch[1][1])
        mini_batch = [zip_array.recovery_state_mcts_prob(data) for data in mini_batch]
        state_batch = [data[0] for data in mini_batch]
        state_batch = np.array(state_batch).astype('float32')

        mcts_probs_batch = [data[1] for data in mini_batch]
        mcts_probs_batch = np.array(mcts_probs_batch).astype('float32')

        winner_batch = [data[2] for data in mini_batch]
        winner_batch = np.array(winner_batch).astype('float32')

        # 旧的策略，旧的价值函数
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier
            )
            # 新的策略，新的价值函数
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1))
            if kl > self.kl_targ * 4:  # 如果KL散度很差，则提前终止
                break

        # 自适应调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        # print(old_v.flatten(),new_v.flatten())
        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))

        log_info(f"训练指标:")
        log_info(f"  ├─ KL散度: {kl:.5f}")
        log_info(f"  ├─ 学习率倍数: {self.lr_multiplier:.3f} (当前lr: {self.learn_rate * self.lr_multiplier:.2e})")
        log_info(f"  ├─ Loss: {loss:.4f}")
        log_info(f"  ├─ Entropy: {entropy:.4f}")
        log_info(f"  ├─ 解释方差(旧): {explained_var_old:.6f}")
        log_info(f"  └─ 解释方差(新): {explained_var_new:.6f}")

        return loss, entropy

    def run(self):
        """开始训练"""
        log_info("🚀 开始训练流程")
        log_info("=" * 70)
        start_time = time.time()

        try:
            for i in range(self.game_batch_num):
                iter_start_time = time.time()

                # 加载数据
                if not CONFIG['use_redis']:
                    log_info("📦 从文件加载数据...")
                    while True:
                        try:
                            with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
                                data_file = pickle.load(data_dict)
                                self.data_buffer = data_file['data_buffer']
                                self.iters = data_file['iters']
                                del data_file
                            log_info(f"✓ 已载入数据 (样本数: {len(self.data_buffer)})")
                            break
                        except:
                            log_info("⏳ 等待数据文件生成...")
                            time.sleep(30)
                else:
                    log_info("📦 从Redis加载数据...")
                    while True:
                        try:
                            l = len(self.data_buffer)
                            data = my_redis.get_list_range(self.redis_cli,'train_data_buffer', l if l == 0 else l - 1,-1)
                            self.data_buffer.extend(data)
                            self.iters = self.redis_cli.get('iters')
                            if self.redis_cli.llen('train_data_buffer') > self.buffer_size:
                                self.redis_cli.lpop('train_data_buffer',self.buffer_size/10)
                            log_info(f"✓ 已载入数据 (样本数: {len(self.data_buffer)})")
                            break
                        except:
                            log_info("⏳ 等待Redis数据...")
                            time.sleep(5)

                # 训练步骤
                log_info("-" * 70)
                log_info(f"🔄 Batch {i+1}/{self.game_batch_num} | Step: {self.iters}")
                log_info(f"📊 数据缓冲区大小: {len(self.data_buffer)} / {self.buffer_size}")

                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_updata()

                    # 保存模型
                    model_path = CONFIG['pytorch_model_path'] if CONFIG['use_frame'] == 'pytorch' else CONFIG['paddle_model_path']
                    if CONFIG['use_frame'] == 'paddle':
                        self.policy_value_net.save_model(CONFIG['paddle_model_path'])
                    elif CONFIG['use_frame'] == 'pytorch':
                        self.policy_value_net.save_model(CONFIG['pytorch_model_path'])
                    else:
                        log_info('✗ 不支持所选框架')

                    log_info(f"💾 模型已保存: {model_path}")
                else:
                    log_info(f"⚠️  数据不足，等待更多样本 (当前: {len(self.data_buffer)}, 需要: {self.batch_size})")

                iter_time = time.time() - iter_start_time
                total_time = time.time() - start_time
                log_info(f"⏱️  本轮耗时: {iter_time:.1f}s | 累计耗时: {total_time:.1f}s ({total_time/60:.1f}min)")

                # 定期保存检查点
                if (i + 1) % self.check_freq == 0:
                    checkpoint_path = f'models/current_policy_batch{i + 1}.pkl' if CONFIG['use_frame'] == 'pytorch' else f'models/current_policy_batch{i + 1}.model'
                    self.policy_value_net.save_model(checkpoint_path)
                    log_info(f"🎯 检查点已保存: {checkpoint_path}")
                    log_info(f"📈 训练进度: {i+1}/{self.game_batch_num} ({100*(i+1)/self.game_batch_num:.1f}%)")

                log_info("=" * 70)

                # 等待下一次更新
                if i < self.game_batch_num - 1:
                    wait_time = CONFIG['train_update_interval']
                    log_info(f"💤 等待 {wait_time}s 后进行下一次更新...")
                    time.sleep(wait_time)

        except KeyboardInterrupt:
            log_info("\n⏹️  训练已手动停止")
            total_time = time.time() - start_time
            log_info(f"⏱️  总训练时长: {total_time:.1f}s ({total_time/60:.1f}min)")
        except Exception as e:
            log_info(f"\n❌ 训练出错: {str(e)}")
            import traceback
            traceback.print_exc()

        log_info("🏁 训练结束")


if __name__ == '__main__':
    if CONFIG['use_frame'] == 'paddle':
        training_pipeline = TrainPipeline(init_model='current_policy.model')
        training_pipeline.run()
    elif CONFIG['use_frame'] == 'pytorch':
        training_pipeline = TrainPipeline(init_model='current_policy.pkl')
        training_pipeline.run()
    else:
        log_info('暂不支持您选择的框架')
