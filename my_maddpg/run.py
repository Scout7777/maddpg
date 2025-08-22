#构建循环
import numpy as np
import time
from maddpg import Agents
from relayBuffer import ReplayBuffer
from pettingzoo.mpe import simple_adversary_v3

TEST_ENV = simple_adversary_v3


def evaluate(agents,env,episode1,action_noise):
    reward_list = []
    for i in range(10):
        s_all_dictionary,infos = env.reset()
        reward = 0
        terminal = [False] * env.max_num_agents
        while not any(terminal):

            a_all = agents.output_action_all(s_all_dictionary,action_noise)

            s_all_next, rewards, done, trunc, infos = env.step(a_all)
            list_trunc = list(trunc.values())
            list_reward = list(rewards.values())
            list_done = list(done.values())
            terminal = [d or t for d, t in zip(list_done, list_trunc)]
            s_all_dictionary = s_all_next
            reward += sum(list_reward)
        reward_list.append(reward)
    reward_mean = np.mean(reward_list)
    print(f'Evaluation episode {episode1} 'f' average score {reward_mean:.1f}')
    return reward_mean

def list_s_dictionary_to_vector(list_s_dictionary):
    state = np.array([])
    for s in list_s_dictionary:
        state = np.concatenate([state, s])
    return state

def plot(agents):
    for i in range(5):
        render_env = TEST_ENV.parallel_env(
            continuous_actions=True, max_cycles=50, render_mode='human')
        render_obs, _ = render_env.reset()
        render_done = [False] * render_env.max_num_agents
        while not any(render_done):
            render_actions = agents.output_action_all(render_obs, action_noise=False)
            render_obs_, render_reward, render_done, render_trunc, render_info = render_env.step(render_actions)
            render_obs = render_obs_
            list_render_trunc = list(render_trunc.values())
            list_render_done = list(render_done.values())
            render_done = [d or t for d, t in zip(list_render_done, list_render_trunc)]
            time.sleep(0.05)
        render_env.close()

def run():
    MAX_EPISODE = 10000
    episode = 0
    n_sapmles_in = 0
    eval_rewards = []
    eval_episodes = []
    env = TEST_ENV.parallel_env(continuous_actions=True, max_cycles=50)
    s_all_dictionary,info = env.reset()
    n_agents = env.max_num_agents
    shape_s = []
    shape_a = []

    for agent_name in env.agents:
        shape_s.append(env.observation_space(agent_name).shape[0])
        shape_a.append(env.action_space(agent_name).shape[0])

    sum_sa = sum(shape_s) + sum(shape_a)
    agents = Agents(shape_s,shape_a,sum_sa,n_agents,env=env,
                    gamma=0.95, tau=0.01,alpha=0.001, beta=0.0001,fc1=128,fc2=128)
    sum_s = sum(shape_s)
    replay_buffer = ReplayBuffer(shape_s,shape_a,sum_s,n_agents,
                                max_size=1000000,batch_size=1024)
    reward = evaluate(agents,env,episode,action_noise=True)
    eval_episodes.append(episode)
    eval_rewards.append(reward)
    while episode < MAX_EPISODE:
        s_all_dictionary,infos = env.reset()
        terminal = [False] * env.max_num_agents
        while not any(terminal):
            a_all = agents.output_action_all(s_all_dictionary, action_noise=False)
            s_all_next, reward, done, trunc, infos = env.step(a_all)
            list_trunc = list(trunc.values())
            list_reward = list(reward.values())
            list_done = list(done.values())
            list_a_all = list(a_all.values())
            list_s_all_dictionary = list(s_all_dictionary.values())
            list_s_all_next = list(s_all_next.values())

            s_all = list_s_dictionary_to_vector(list_s_all_dictionary)
            s_all_next_list = list_s_dictionary_to_vector(list_s_all_next)

            terminal = [d or t for d, t in zip(list_done, list_trunc)]
            #每次存储经验
            replay_buffer.store(list_s_all_dictionary,list_s_all_next,s_all,s_all_next_list,list_a_all,
                                list_reward,terminal)
            #100个样本就学习一次
            if n_sapmles_in % 100 == 0:
                agents.learn(replay_buffer)
            n_sapmles_in += 1
            s_all_dictionary = s_all_next
        if episode % 20 == 1:
            reward = evaluate(agents,env,episode,action_noise=True)
            eval_rewards.append(reward)
            eval_episodes.append(episode)
        # if episode % 200 == 0:
        #     plot(agents)
        episode += 1
    np.save('data/maddpg_rewards.npy', np.array(eval_rewards))
    np.save('data/maddpg_episodes.npy', np.array(eval_episodes))



if __name__ == ('__main__'):
    run()