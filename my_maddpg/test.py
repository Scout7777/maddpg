import numpy as np
import time
from pettingzoo.mpe import simple_world_comm_v3
from pettingzoo.mpe import simple_speaker_listener_v4
from pettingzoo.mpe import simple_adversary_v3

TEST_ENV = simple_world_comm_v3

parallel_env = TEST_ENV.parallel_env(continuous_actions=True)

_, _ = parallel_env.reset()
n_agents = parallel_env.max_num_agents
#初始化智能体参数
obs_space = [] # 存储每个智能体的观测维度
n_actions = [] # 存储每个智能体的动作维度

observation_space = parallel_env.observation_space
action_space = parallel_env.action_space


for agent in parallel_env.agents:
    obs_space.append(parallel_env.observation_space(agent).shape[0])
    n_actions.append(parallel_env.action_space(agent).shape[0])


critic_input_n = sum(obs_space) + sum(n_actions)




#print('observation_space:',observation_space)
#print('action_space:',action_space)


print('n_agents:',n_agents)
print('obs_space:',obs_space,'n_actions:',n_actions)
print('critic_dims:', critic_input_n)

env = parallel_env
agent = list(env.action_spaces.keys())[0]
# parallel_env.action_spaces: ['leadadversary_0', 'adversary_0', 'adversary_1', 'adversary_2', 'agent_0','agent_1']
min_action = env.action_space(agent).low
max_action = env.action_space(agent).high
print('parallel_env.action_spaces:',list(parallel_env.action_spaces.keys()))

print('min_action:',min_action)
print('max_action:',max_action)
