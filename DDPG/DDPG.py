import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import gym
import time

LR=0.001
MEMORY_CAPACITY=2000
MINI_BATCH=32
GAMMA=0.95
ENV_NAME = 'Pendulum-v0'
TAU=0.01
EPISODE=400
STEP=200
TEST=10


class ANet(nn.Module):
    def __init__(self,n_state,n_action):
        super(ANet,self).__init__()
        self.fc1=nn.Linear(n_state,30)
        self.fc2=nn.Linear(30,n_action)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.tanh(self.fc2(x))
        return x*2


class CNet(nn.Module):
    def __init__(self,n_state,n_action):
        super(CNet,self).__init__()
        self.fc1=nn.Linear(n_state,30)
        self.fc2=nn.Linear(n_action,30)
        # self.fc3=nn.Linear(30,1)
        self.fc3=nn.Linear(60,1)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self,s,a):
        x=self.fc1(s)
        y=self.fc2(a)
        # out=F.relu(x+y)
        input=torch.cat([x,y],dim=1)
        out=F.relu(input)
        return self.fc3(out)


class DDPG(object):
    def __init__(self,env):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        print(self.action_dim)
        self.Actor_eval=ANet(self.state_dim,self.action_dim)
        self.Actor_target=ANet(self.state_dim,self.action_dim)
        self.Critic_eval=CNet(self.state_dim,self.action_dim)
        self.Critic_target=CNet(self.state_dim,self.action_dim)
        self.optimizer_Actor=torch.optim.Adam(self.Actor_eval.parameters(),lr=LR)
        self.optimizer_Critic=torch.optim.Adam(self.Critic_eval.parameters(),lr=LR)
        self.loss_td=nn.MSELoss()
        self.memory = np.zeros((MEMORY_CAPACITY, self.state_dim*2+1+self.action_dim))
        self.memory_counter=0

    def choose_action(self,s):
        s=torch.unsqueeze(torch.FloatTensor(s), 0)
        # tensor([[1.7771]], grad_fn=<MulBackward0>)
        return self.Actor_eval(s)[0].detach()

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, action, [reward], next_state))  # 横向拼接
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # soft load target network  y=y*0.99+x*0.01
        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.'+x+'.data.mul_((1-TAU))')
            eval('self.Actor_target.'+x+'.data.add_(TAU*self.Actor_eval.'+x+'.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.'+x+'.data.mul_((1-TAU))')
            eval('self.Critic_target.'+x+'.data.add_(TAU*self.Critic_eval.'+x+'.data)')

        # sample batch for train
        index=np.random.choice(MEMORY_CAPACITY,MINI_BATCH)
        memory=self.memory[index,:]
        state=torch.FloatTensor(memory[:,:self.state_dim])
        action=torch.FloatTensor(memory[:,self.state_dim:self.state_dim+self.action_dim])
        reward=torch.FloatTensor(memory[:,-self.state_dim-1:-self.state_dim])
        next_state=torch.FloatTensor(memory[:,-self.state_dim:])
        # s,a,r,s'
        # maxmize Q evaluate
        a=self.Actor_eval(state)
        q=self.Critic_eval.forward(state,a)
        loss=-torch.mean(q)
        self.optimizer_Actor.zero_grad()
        loss.backward()
        self.optimizer_Actor.step()

        a_=self.Actor_target(next_state)
        q_=self.Critic_target(next_state,a_)
        q_target=reward+GAMMA*q_
        q_v=self.Critic_eval(state,action)
        td_error=self.loss_td(q_target,q_v)
        self.optimizer_Critic.zero_grad()
        td_error.backward()
        self.optimizer_Critic.step()


def main():
    VAR=3
    totol_r=0
    env=gym.make(ENV_NAME).unwrapped
    env.seed(1)
    ddpg=DDPG(env)
    for i in range(EPISODE):
        s = env.reset()
        ep_reward = 0
        for j in range(STEP):
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, VAR), -2, 2)    # add randomness to action selection for exploration
            s_, r, done, _ = env.step(a)
            ddpg.store_transition(s, a, r / 10, s_)
            if ddpg.memory_counter > MEMORY_CAPACITY:
                VAR *= .9995    # decay the action randomness
                ddpg.learn()
            s = s_
            ep_reward += r
            if j == STEP-1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % VAR, )
                # if ep_reward > -300:RENDER = True
                break


if __name__=='__main__':
    main()

