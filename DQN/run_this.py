from maze_env import Maze
from RL_brain import DQN
import time
from TKtest import MAZE

def run_maze():
    print("====Game Start====")
    step = 0
    max_episode = 500
    for episode in range(max_episode):
        state = env.reset()  # 重置智能体位置
        step_every_episode = 0
        # epsilon = episode / max_episode  # 动态变化随机值
        while True:
            if episode < 10:
                time.sleep(0.1)
            if episode > 480:
                time.sleep(0.1)
            env.render()  # 显示新位置
            action = model.choose_action(state,episode)  # 根据状态选择行为
            # 环境根据行为给出下一个状态，奖励，是否结束。
            next_state, reward, terminal = env.step(action)
            model.store_transition(state, action, reward, next_state)  # 把训练的放进经验池
            # 控制学习起始时间(先积累记忆再学习)和控制学习的频率(积累多少步经验学习一次)
            if step > 200 and step % 5 == 0:
                model.learn(2)
                print("学习一下")
            # 进入下一步
            state = next_state
            if terminal:
                print("episode=", episode, end=",")
                print("step=", step_every_episode)
                break
            step += 1
            step_every_episode += 1
    # 游戏环境结束
    print("====Game Over====")
    env.destroy()


if __name__ == "__main__":
    env = Maze()  # 环境
    model = DQN(
        n_states=env.n_states,
        n_actions=env.n_actions
    )  # 算法模型
    run_maze()
    env.mainloop()
    model.plot_cost()  # 误差曲线
