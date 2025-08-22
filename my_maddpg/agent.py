#创建初始网络，输出动作,更新网络，更新目标网络
import torch
import numpy as np
from network import ActorNetwork,CriticNetwork

class Agent:
    def __init__(self,shape_s_self,shape_a_self,sum_sa,i,
                          gamma, tau, alpha, beta, fc1, fc2,
                          min_action,max_action):
        self.gamma = gamma
        self.tau = tau
        self.shape_a_self = shape_a_self
        self.agent_id = i
        self.min_action = min_action
        self.max_action = max_action
        self.actor = ActorNetwork(shape_s_self,shape_a_self,alpha,fc1,fc2)
        self.actor_target = ActorNetwork(shape_s_self, shape_a_self, alpha, fc1, fc2)
        self.critic = CriticNetwork(sum_sa,beta,fc1,fc2)
        self.critic_target = CriticNetwork(sum_sa, beta, fc1, fc2)

        self.update_target_network()

    def output_action(self,state_self,action_noise):
        # 1. 将观察值转换为PyTorch张量，observation[np.newaxis, :]扩展一个维度
        state = torch.tensor(state_self[np.newaxis, :], dtype=torch.float,device=self.actor.device)
        # 2. 通过actor网络获取动作
        actions = self.actor.forward(state)
        # 3. 生成随机噪声
        noise = torch.randn(size=(self.shape_a_self,)).to(self.actor.device)
        # 4. 评估模式时关闭噪声！！！
        noise *= torch.tensor(1 - int(action_noise))
        # 5. 添加噪声并裁剪到合法范围
        action = torch.clamp(actions + noise,
                         torch.tensor(self.min_action, device=self.actor.device),
                         torch.tensor(self.max_action, device=self.actor.device))
        # print('action:', action)
        # 6. 返回numpy格式的动作
        return action.data.cpu().numpy()[0]

    def update_target_network(self):
        tau = self.tau
        src = self.actor
        dest = self.actor_target
        for param, target in zip(src.parameters(), dest.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)

        src = self.critic
        dest = self.critic_target
        for param, target in zip(src.parameters(), dest.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)

    def learn(self,replay_buffer,agents):
        if replay_buffer.cntr < replay_buffer.batch_size: return
        s_all, s_all_next, reward, terminal, agent_s, agent_s_next, agent_a = replay_buffer.sample()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        s_all = torch.tensor(np.array(s_all), dtype=torch.float, device=device)
        s_all_next = torch.tensor(np.array(s_all_next), dtype=torch.float, device=device)
        reward = torch.tensor(np.array(reward), dtype=torch.float, device=device)
        terminal = torch.tensor(np.array(terminal), device=device)

        agent_s = [torch.tensor(agent_s[idx],device=device, dtype=torch.float)
                        for idx in range(len(agents))]
        agent_s_next = [torch.tensor(agent_s_next[idx],device=device, dtype=torch.float)
                            for idx in range(len(agents))]
        agent_a = [torch.tensor(agent_a[idx], device=device, dtype=torch.float)
                   for idx in range(len(agents))]
        #计算target_Q
        with torch.no_grad():
            #隐式前向传播，得到下一步动作
            a_all_next = torch.cat([agent.actor_target(agent_s_next[i]) for i, agent in enumerate(agents)],dim=1)
            #print(a_all_next.shape) [1024,15]
            #输入下一步状态和所有动作 .squeeze进行降维操作
            Q_next = self.critic_target.forward(s_all_next, a_all_next).squeeze()
            #print(critic_value_.shape) [1024]
            #把样本中终止状态的Q归零  “布尔索引”（boolean indexing） 数组通过布尔列表选取
            Q_next[terminal[:, self.agent_id]] = 0.0
            target_Q = reward[:, self.agent_id] + self.gamma * Q_next

        a_ll = torch.cat([agent_a[idx] for idx in range(len(agents))],dim=1)
        Q = self.critic.forward(s_all, a_ll).squeeze()
        #得到 y 的函数（MSE），进行梯度下降 更新critic网络
        critic_loss = torch.nn.functional.mse_loss(target_Q, Q)
        # 梯度清零，反向传播，梯度裁剪，梯度更新
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic.optimizer.step()

        #只有agent自己进行传播
        agent_a[self.agent_id] = self.actor.forward(agent_s[self.agent_id])
        #其他人保持动作不变
        a_all_self_change = torch.cat([agent_a[i] for i in range(len(agents))], dim=1)
        #得到 y 的函数（-Q）进行梯度下降 更新actor网络
        actor_loss = -self.critic.forward(s_all, a_all_self_change).mean()
        #梯度清零，反向传播，梯度裁剪，梯度更新
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor.optimizer.step()

        #软更新目标网络
        self.update_target_network()





