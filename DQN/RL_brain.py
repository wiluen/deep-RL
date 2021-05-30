import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 10)
        self.fc2 = nn.Linear(10, n_actions)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class DQN:
    def __init__(self, n_states, n_actions):
        print("<DQN init>")
        # DQN有两个net:target net和eval net,具有选动作，存经历，学习三个基本功能
        self.eval_net, self.target_net = Net(n_states, n_actions), Net(n_states, n_actions)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.01)
        self.n_actions = n_actions
        self.n_states = n_states

        # 使用变量
        self.learn_step_counter = 0  # target网络学习计数
        self.memory_counter = 0  # 记忆计数
        self.memory = np.zeros((2000, n_states * 2 + 2))  # 2*2(state和next_state,每个x,y坐标确定)+2(action和reward),存储2000个记忆体
        self.cost = []  # 记录损失值

    def choose_action(self, state, episode):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # []->[[]]
        # tensor([[ 0.5000,  0.5000]])
        if np.random.uniform() < 0.9:
            action_value = self.eval_net.forward(state)
            # tensor([[ 0.0564,  0.3940, -0.0179,  0.2653]], grad_fn=<AddmmBackward>)
            action = torch.max(action_value, 1)[1].data.numpy()[0]  # max(tensor,dim=0:列dim=1:行)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))  # 横向拼接
        index = self.memory_counter % 2000  # 满了就覆盖旧的
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self,n_state):
        # print("<learn>")
        # target net 更新频率,用于预测，不会及时更新参数
        if self.learn_step_counter % 100 == 0:
            self.target_net.load_state_dict((self.eval_net.state_dict()))  # 加载模型
            print('加载targetQ')
        self.learn_step_counter += 1

        # 使用记忆库中批量数据
        sample_index = np.random.choice(2000, 32)  # 2000个中随机抽取32个作为batch_size
        memory = self.memory[sample_index, :]  # 抽取的记忆单元，并逐个提取
        state = torch.FloatTensor(memory[:, :n_state])
        action = torch.LongTensor(memory[:, n_state:n_state+1])
        reward = torch.LongTensor(memory[:, n_state+1:n_state+2])
        next_state = torch.FloatTensor(memory[:, -n_state:])
        # print('state:',state)    32个
        # state: tensor([[ 0.0000,  0.0000],
        #                [ 0.0000,  0.0000],
        #                [ 0.0000,  0.0000],
        #                [ 0.0000,  0.0000],
        #                [ 0.0000,  0.0000],
        # 计算loss,q_eval:所采取动作的预测value,q_target:所采取动作的实际value
        q_eval = self.eval_net(state).gather(1, action)
        # eval_net->(64,4)->按照action索引提取出q_value
        # gather在one-hot为输出的多分类问题中，可以把最大值坐标作为index传进去，
        # 然后提取到每一行的正确预测结果，这也是gather可能的一个作用。
        q_next = self.target_net(next_state).detach() # 切断反向传播
        q_target = reward + 0.9 * q_next.max(1)[0].unsqueeze(1) # label
        # torch.max->[values=[],indices=[]] max(1)[0]->values=[]
        # print('q_eval:',q_eval)
        # q_eval: tensor([[0.5372],
        #                 [0.5372],
        #                 [0.5372],
        #                 [0.5372],
        #                 [0.5372],
        #                 [0.5372],

        # print('q_next:',q_next)
        # q_next: tensor([[0.4845, 0.5553, 0.4481, 0.5882],
        #                 [0.4845, 0.5553, 0.4481, 0.5882],
        #                 [0.4845, 0.5553, 0.4481, 0.5882],
        #                 [0.4845, 0.5553, 0.4481, 0.5882],

        # q_target: tensor([[0.5882],
        #                   [0.5882],
        #                   [0.5882],
        #                   [0.5882],
        #                   ]   和q_eval形式相同
        loss = self.loss(q_eval, q_target)
        self.cost.append(loss.item())
        # 反向传播更新
        self.optimizer.zero_grad()  # 梯度重置
        loss.backward()  # 反向求导
        self.optimizer.step()  # 更新模型参数
        # 目标：q_eval逼近q_target
        # 忆：q预测逼近q现实

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost)), self.cost)
        plt.xlabel("step")
        plt.ylabel("cost")
        plt.show()