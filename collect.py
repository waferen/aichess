"""自我对弈收集数据"""
import random
from collections import deque
import copy
import os
import pickle
import time
from datetime import datetime
from game import Board, Game, move_action2move_id, move_id2move_action, flip_map
from mcts import MCTSPlayer
from config import CONFIG

if CONFIG['use_redis']:
    import my_redis, redis

import zip_array

if CONFIG['use_frame'] == 'paddle':
    from paddle_net import PolicyValueNet
elif CONFIG['use_frame'] == 'pytorch':
    from pytorch_net import PolicyValueNet
else:
    print('暂不支持您选择的框架')


def get_timestamp():
    """获取当前时间戳"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log_info(msg):
    """输出带时间戳的日志"""
    print(f"[{get_timestamp()}] {msg}")


# 定义整个对弈收集数据流程
class CollectPipeline:

    def __init__(self, init_model=None):
        log_info("=" * 70)
        log_info("初始化数据收集流程")
        log_info("=" * 70)

        # 象棋逻辑和棋盘
        self.board = Board()
        self.game = Game(self.board)
        # 对弈参数
        self.temp = 1  # 温度
        self.n_playout = CONFIG['play_out']  # 每次移动的模拟次数
        self.c_puct = CONFIG['c_puct']  # u的权重
        self.buffer_size = CONFIG['buffer_size']  # 经验池大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0
        self.init_model = init_model

        # 统计信息
        self.total_games = 0
        self.total_steps = 0
        self.winner_stats = {1: 0, 2: 0, -1: 0}  # 红方胜、黑方胜、和棋

        if CONFIG['use_redis']:
            self.redis_cli = my_redis.get_redis_cli()
            log_info("使用Redis数据存储")

        log_info(f"收集配置:")
        log_info(f"  - MCTS Playouts: {self.n_playout}")
        log_info(f"  - C_PUCT: {self.c_puct}")
        log_info(f"  - 温度参数: {self.temp}")
        log_info(f"  - 缓冲区大小: {self.buffer_size}")
        log_info(f"  - 框架: {CONFIG['use_frame']}")
        log_info("=" * 70)

    # 从主体加载模型
    def load_model(self):
        if CONFIG['use_frame'] == 'paddle':
            model_path = CONFIG['paddle_model_path']
        elif CONFIG['use_frame'] == 'pytorch':
            model_path = CONFIG['pytorch_model_path']
        else:
            log_info('暂不支持所选框架')
            return

        try:
            self.policy_value_net = PolicyValueNet(model_file=model_path)
            log_info(f"✓ 已加载最新模型: {model_path}")
        except:
            self.policy_value_net = PolicyValueNet()
            log_info("✗ 模型文件不存在，使用初始模型")

        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """左右对称变换，扩充数据集一倍，加速一倍训练速度"""
        extend_data = []
        # 棋盘状态shape is [9, 10, 9], 走子概率，赢家
        for state, mcts_prob, winner in play_data:
            # 原始数据
            extend_data.append(zip_array.zip_state_mcts_prob((state, mcts_prob, winner)))
            # 水平翻转后的数据
            state_flip = state.transpose([1, 2, 0])
            state = state.transpose([1, 2, 0])
            for i in range(10):
                for j in range(9):
                    state_flip[i][j] = state[i][8 - j]
            state_flip = state_flip.transpose([2, 0, 1])
            mcts_prob_flip = copy.deepcopy(mcts_prob)
            for i in range(len(mcts_prob_flip)):
                mcts_prob_flip[i] = mcts_prob[move_action2move_id[flip_map(move_id2move_action[i])]]
            extend_data.append(zip_array.zip_state_mcts_prob((state_flip, mcts_prob_flip, winner)))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """收集自我对弈的数据"""
        for i in range(n_games):
            game_start = time.time()
            self.load_model()  # 从本体处加载最新模型

            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp, is_shown=False)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)

            # 增加数据（对称变换）
            play_data = self.get_equi_data(play_data)
            data_augmented = len(play_data)

            # 更新统计
            self.total_games += 1
            self.total_steps += self.episode_len
            self.winner_stats[winner] = self.winner_stats.get(winner, 0) + 1

            game_time = time.time() - game_start

            # 显示游戏结果
            winner_text = {1: "红方胜", 2: "黑方胜", -1: "和棋"}
            log_info("-" * 70)
            log_info(f"🎮 游戏 #{self.total_games}")
            log_info(f"  ├─ 胜者: {winner_text.get(winner, '未知')}")
            log_info(f"  ├─ 步数: {self.episode_len}")
            log_info(f"  ├─ 生成样本: {data_augmented} (含对称变换)")
            log_info(f"  └─ 耗时: {game_time:.1f}s")

            # 保存数据
            if CONFIG['use_redis']:
                while True:
                    try:
                        for d in play_data:
                            self.redis_cli.rpush('train_data_buffer', pickle.dumps(d))
                        self.redis_cli.incr('iters')
                        self.iters = self.redis_cli.get('iters')
                        log_info(f"✓ 数据已保存到Redis (总迭代: {self.iters})")
                        break
                    except Exception as e:
                        log_info(f"✗ Redis存储失败: {e}")
                        time.sleep(1)
            else:
                if os.path.exists(CONFIG['train_data_buffer_path']):
                    while True:
                        try:
                            with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
                                data_file = pickle.load(data_dict)
                                self.data_buffer = deque(maxlen=self.buffer_size)
                                self.data_buffer.extend(data_file['data_buffer'])
                                self.iters = data_file['iters']
                                del data_file
                                self.iters += 1
                                self.data_buffer.extend(play_data)
                            log_info(f"✓ 已加载并合并现有数据")
                            break
                        except Exception as e:
                            log_info(f"✗ 加载数据文件失败: {e}，30秒后重试...")
                            time.sleep(30)
                else:
                    self.data_buffer.extend(play_data)
                    self.iters += 1
                    log_info(f"✓ 创建新数据文件")

                # 保存到文件
                data_dict = {'data_buffer': self.data_buffer, 'iters': self.iters}
                with open(CONFIG['train_data_buffer_path'], 'wb') as data_file:
                    pickle.dump(data_dict, data_file)

                log_info(f"💾 数据已保存: {CONFIG['train_data_buffer_path']}")
                log_info(f"📊 缓冲区状态: {len(self.data_buffer)}/{self.buffer_size} 样本")

            # 显示统计信息
            self.print_stats()

        return self.iters

    def print_stats(self):
        """打印统计信息"""
        log_info("-" * 70)
        log_info(f"📈 累计统计:")
        log_info(f"  ├─ 总游戏数: {self.total_games}")
        log_info(f"  ├─ 总步数: {self.total_steps}")
        log_info(f"  ├─ 红方胜: {self.winner_stats[1]} ({100*self.winner_stats[1]/max(self.total_games,1):.1f}%)")
        log_info(f"  ├─ 黑方胜: {self.winner_stats[2]} ({100*self.winner_stats[2]/max(self.total_games,1):.1f}%)")
        log_info(f"  ├─ 和棋: {self.winner_stats[-1]} ({100*self.winner_stats[-1]/max(self.total_games,1):.1f}%)")
        if not CONFIG['use_redis']:
            log_info(f"  └─ 数据缓冲区: {len(self.data_buffer)}/{self.buffer_size}")
        log_info("=" * 70)

    def run(self):
        """开始收集数据"""
        log_info("🚀 开始数据收集流程")
        log_info("=" * 70)
        start_time = time.time()

        try:
            while True:
                iters = self.collect_selfplay_data()
                # log_info(f'batch i: {iters}, episode_len: {self.episode_len}')
        except KeyboardInterrupt:
            total_time = time.time() - start_time
            log_info("\n⏹️  数据收集已手动停止")
            log_info(f"⏱️  总运行时长: {total_time:.1f}s ({total_time/60:.1f}min)")
            self.print_stats()
        except Exception as e:
            log_info(f"\n❌ 数据收集出错: {str(e)}")
            import traceback
            traceback.print_exc()

        log_info("🏁 数据收集结束")


if __name__ == '__main__':
    if CONFIG['use_frame'] == 'paddle':
        collecting_pipeline = CollectPipeline(init_model='current_policy.model')
        collecting_pipeline.run()
    elif CONFIG['use_frame'] == 'pytorch':
        collecting_pipeline = CollectPipeline(init_model='current_policy.pkl')
        collecting_pipeline.run()
    else:
        log_info('暂不支持您选择的框架')
