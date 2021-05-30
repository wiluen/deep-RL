from maze_env import Maze
from TKtest import MAZE
from RL_brain import QLearningTable

def update():
    for episode in range(100):
        observation=env.reset()
        totol_reward=0
        totol_step=0
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation   direct:l u r d
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_
            totol_reward+=reward
            totol_step+=1
            # break while loop when end of this episode
            if done:
                break

        print("========================")
        print(f'第{episode}回合')
        print('奖赏:',totol_reward)
        print('步数:',totol_step)

    print('game over')
    env.destroy()


if __name__=='__main__':
    env=MAZE()
    RL=QLearningTable(actions=list(range(env.n_actions)))
    # print(RL.q_table)
    env.after(100, update)

    env.mainloop()
