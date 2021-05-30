import torch
import time
import multiagent.scenarios as scenarios
import numpy as np
from multiagent.environment import MultiAgentEnv
from model import DDPGAgent, MADDPG

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_trainers(env, obs_shape_n, action_shape_n):
    return MADDPG(env.n, obs_shape_n, action_shape_n, 0.7, 20000)


def make_env(scenario_name, benchmark=False):
    """
    create the environment from script
    """
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def train():
    env = make_env("simple_tag")         # 创建环境
    obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]# 总维度
    max_ob_dim = max(obs_shape_n)
    for i in range(0, len(obs_shape_n)):
        obs_shape_n[i] = max_ob_dim        # 会存在给的ob输出维度不一样的问题，在这里把维度统一
    action_shape_n = [env.action_space[i].n for i in range(env.n)]
    maddpg = get_trainers(env, obs_shape_n, action_shape_n)
    #maddpg.load_model(100)
    episode_rewards = [0.0]
    obs_n = env.reset() # [[],[],[]]
    for i in range(0, len(obs_n)):
        while len(obs_n[i]) < obs_shape_n[i]:
            obs_n[i] = np.append(obs_n[i], np.array([0.0]))
    for episode in range(0, 10000):
        for step in range(0, 500):
            action_n = [agent.act_prob(torch.from_numpy(obs.astype(np.float32)).to(device)).detach().cpu().numpy()
                        for agent, obs in zip(maddpg.agents, obs_n)]
            # 自己的obs算出action
            # 所有的action
            new_obs_n, rew_n, done_n, info_n = env.step(action_n) # 返回的是所有agent的信息
            for i in range(0, len(new_obs_n)):
                while len(new_obs_n[i]) < obs_shape_n[i]:
                    new_obs_n[i] = np.append(new_obs_n[i], np.array([0.0]))
            maddpg.add_data(obs_n, action_n, rew_n, new_obs_n, done_n)  # 放入池
            episode_rewards[-1] += np.sum(rew_n)
            #print(episode_rewards)
            obs_n = new_obs_n
            done = all(done_n)
            # train需要中心化训练
            if step % 400 == 0: # 特定步骤更新
                maddpg.update(maddpg.memory.sample(6400))
                maddpg.update_all_agents()
            if done or step == 499:
                obs_n = env.reset()
                for i in range(0, len(obs_n)):
                    while len(obs_n[i]) < obs_shape_n[i]:
                        obs_n[i] = np.append(obs_n[i], np.array([0.0]))
                episode_rewards[-1] = episode_rewards[-1]/step
                print(f'{episode}回合,{step}步,episode_reward为:{episode_rewards[-1]}')
                episode_rewards.append(0)
                break
        if episode % 100 == 0 and episode > 1:
            maddpg.save_model(episode)
            print(f'第{episode}回合，保存模型')


def play():
    env = make_env("simple_tag")
    obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
    max_ob_dim = max(obs_shape_n)
    for i in range(0, len(obs_shape_n)):
        obs_shape_n[i] = max_ob_dim
    action_shape_n = [env.action_space[i].n for i in range(env.n)]
    maddpg = get_trainers(env, obs_shape_n, action_shape_n)
    maddpg.load_model(300)
    obs_n = env.reset()
    for i in range(0, len(obs_n)):
        while len(obs_n[i]) < obs_shape_n[i]:
            obs_n[i] = np.append(obs_n[i], np.array([0.0]))
    #env.render()
    # play去中心化执行
    for episode in range(0, 10000):
        for step in range(0, 500):
            action_n = [agent.Actor(torch.from_numpy(obs.astype(np.float32)).to(device)).detach().cpu().numpy()
                        for agent, obs in zip(maddpg.agents, obs_n)]
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            #print(rew_n)
            #time.sleep(0.1)
            #env.render()
            for i in range(0, len(new_obs_n)):
                while len(new_obs_n[i]) < obs_shape_n[i]:
                    new_obs_n[i] = np.append(new_obs_n[i], np.array([0.0]))
            obs_n = new_obs_n
            done = all(done_n)
            if done or step == 499:
                print(rew_n)
                obs_n = env.reset()
                for i in range(0, len(obs_n)):
                    while len(obs_n[i]) < obs_shape_n[i]:
                        obs_n[i] = np.append(obs_n[i], np.array([0.0]))
                break


if __name__ == '__main__':
    train()
    #play()