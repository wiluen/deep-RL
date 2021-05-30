import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gym

GAMMA=0.95
LR=0.01
device=torch.device('cuda')
RENDER=False

class PGNetwork(nn.Module):
    def __init__(self,n_state,n_action):
        super(PGNetwork,self).__init__()
        self.fc1=nn.Linear(n_state,20)
        self.fc2=nn.Linear(20,n_action)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x


class Actor(object):# policy gradient策略网络
    def __init__(self,env):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.network=PGNetwork(self.state_dim,self.action_dim).to(device)
        self.optimizer=torch.optim.Adam(self.network.parameters(),lr=LR)
        self.time_step=0

    def choose_action(self,observation):
        observation=torch.FloatTensor(observation).to(device)
        network_output=self.network.forward(observation)
        with torch.no_grad():
            prob_weights=F.softmax(network_output,dim=0).cuda().data.cpu().numpy()
        action=np.random.choice(range(prob_weights.shape[0]),p=prob_weights)
        return action

    def learn(self,state,action,td_error):
        self.time_step+=1
        softmax_input=self.network.forward(torch.FloatTensor(state).to(device)).unsqueeze(0)
        action=torch.LongTensor([action]).to(device)
        neg_log_prob=F.cross_entropy(input=softmax_input,target=action,reduction='none')
        loss=neg_log_prob*td_error
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Hyper Parameters for Critic
EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch
REPLACE_TARGET_FREQ = 10

class QNetwork(nn.Module):
    def __init__(self,n_state,n_action):
        super(QNetwork,self).__init__()
        self.fc1=nn.Linear(n_state,20)
        self.fc2=nn.Linear(20,1)


    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x


class Critic(object): # Q网络
    def __init__(self,env):
        self.state_dim=env.observation_space.shape[0]
        self.action_dim=env.action_space.n
        self.network=QNetwork(self.state_dim,self.action_dim).to(device)
        self.optimizer=torch.optim.Adam(self.network.parameters(),lr=LR)
        self.loss_func=nn.MSELoss()
        self.time_step=0
        self.epsilon=EPSILON

    def train(self,state,reward,next_state):
        s=torch.FloatTensor(state).to(device)
        s_=torch.FloatTensor(next_state).to(device)
        v=self.network.forward(s)
        v_=self.network.forward(s_)
        loss=self.loss_func(reward+GAMMA*v_,v)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            td_error=reward+GAMMA*v_-v
        return td_error

# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 3000  # Episode limitation
STEP = 3000  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode

def main():
    env=gym.make(ENV_NAME)
    actor=Actor(env)
    critic=Critic(env)
    for episode in range(EPISODE):
        state=env.reset()
        for step in range(STEP):
            action=actor.choose_action(state)
            next_state,reward,done,_=env.step(action)
            td_error=critic.train(state,reward,next_state)
            actor.learn(state,action,td_error)
            state=next_state
            if done:
                break

        if episode%100==0:
            totol_reward=0
            for i in range(TEST):# 100回合测试10次看看效果
                state=env.reset()
                for j in range(STEP):
                    if RENDER :env.render()
                    action=actor.choose_action(state)
                    _,reward,done,_=env.step(action)
                    totol_reward+=reward
                    if done:
                        break
            avg_reward=totol_reward/TEST
            print('episode:',episode,'test avg_reward:',avg_reward)


if __name__=='__main__':
    main()



