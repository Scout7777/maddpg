#初始化（s,a,s_next,r） 储存经验，随机采样 越考前的样本采集次数越多
import numpy as np

class ReplayBuffer:
    def __init__(self,shape_s,shape_a,sum_s,n_agents,
                    max_size,batch_size):
        self.shape_s = shape_s
        self.shape_a = shape_a
        self.n_agents = n_agents
        self.max_size = max_size
        self.batch_size = batch_size
        self.cntr = 0

        self.s_buffer = np.zeros((max_size,sum_s))
        self.s_next_buffer = np.zeros((max_size,sum_s))
        self.reward_buffer = np.zeros((max_size,n_agents))
        self.terminal_buffer = np.zeros((max_size,n_agents),dtype=bool)

        self.agent_s_buffer = []
        self.agent_s_next_buffer = []
        self.agent_a_buffer = []
        for i in range(n_agents):
            self.agent_s_buffer.append(np.zeros((max_size,shape_s[i])))
            self.agent_s_next_buffer.append(np.zeros((max_size,shape_s[i])))
            self.agent_a_buffer.append(np.zeros((max_size,shape_a[i])))


    def store(self,list_s_all_dictionary,list_s_all_next,s_all,s_all_next,list_a_all,
                                list_reward,terminal):
        index = self.cntr % self.max_size
        self.s_buffer[index] = s_all
        self.s_next_buffer[index] = s_all_next
        self.reward_buffer[index] = list_reward
        self.terminal_buffer[index] = terminal
        for i in range(self.n_agents):
            self.agent_s_buffer[i][index] = list_s_all_dictionary[i]
            self.agent_s_next_buffer[i][index] = list_s_all_next[i]
            self.agent_a_buffer[i][index] = list_a_all[i]

        self.cntr += 1


    def sample(self):
        batch = np.random.choice(self.cntr,self.batch_size,replace=False)
        s_all = self.s_buffer[batch]
        s_all_next = self.s_next_buffer[batch]
        reward = self.reward_buffer[batch]
        terminal = self.terminal_buffer[batch]
        agent_s = []
        agent_s_next = []
        agent_a = []
        for agent_idx in range(self.n_agents):
            agent_s.append(self.agent_s_buffer[agent_idx][batch])
            agent_s_next.append(self.agent_s_next_buffer[agent_idx][batch])
            agent_a.append(self.agent_a_buffer[agent_idx][batch])
        return s_all,s_all_next,reward,terminal,agent_s,agent_s_next,agent_a



