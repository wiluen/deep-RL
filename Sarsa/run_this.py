from maze_env import Maze
from RL_brain import SarsaTable
def update():
    for episode in range(100):
        observation=env.reset()
        # RL choose action based on observation   direct:l u r d
        action = RL.choose_action(str(observation))
        totol_reward=0
        totol_step=0
        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_),action_)

            # swap observation
            observation = observation_
            action=action_
            # break while loop when end of this episode
            totol_reward+=reward
            totol_step+=1
            if done:
                break
        print(f'第{episode}回合')
        print('奖赏:',totol_reward)
        print('步数:',totol_step)
        print("========================")


    print('game over')
    env.destroy()


if __name__=='__main__':
    env=Maze()

    RL=SarsaTable(actions=list(range(env.n_actions)))
    # [0,1,2,3]
    env.after(100, update)
    env.mainloop()
