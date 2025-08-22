#初始化创建多个agent并管理,输出所有agent动作，更新所有agent
from agent import Agent

class Agents:
    def __init__(self,shape_s,shape_a,sum_sa,n_agents,env,
                 gamma, tau, alpha, beta,fc1,fc2):
        self.agents = []
        for i in range(n_agents):
            agent_name = list(env.action_spaces.keys())[i]
            min_action = env.action_space(agent_name).low
            max_action = env.action_space(agent_name).high
            agent = Agent(shape_s[i],shape_a[i],sum_sa,i,
                          gamma, tau, alpha, beta, fc1, fc2,
                          min_action,max_action)
            self.agents.append(agent)

    def output_action_all(self,s_all_dictionary,action_noise):
        a_all_dictionary = {}
        for agent_name,agent in zip(s_all_dictionary,self.agents):
            action = agent.output_action(s_all_dictionary[agent_name], action_noise)
            a_all_dictionary[agent_name] = action
        return a_all_dictionary

    def learn(self, replay_buffer):
        for agent in self.agents:
            agent.learn(replay_buffer, self.agents)


