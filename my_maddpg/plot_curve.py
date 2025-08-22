import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, rewards, filename, lines=None):
    maddpg_rewards = rewards
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    N = len(maddpg_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(
                maddpg_rewards[max(0, t-100):(t+1)])

    ax.plot(x, running_avg, color="C0")
    ax.set_xlabel("Training Episodes", color="C0")
    ax.set_ylabel("MADDPG Rewards", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")
    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

if __name__ == ('__main__'):
    maddpg_rewards = np.load('data/maddpg_rewards.npy')
    maddpg_episodes = np.load('data/maddpg_episodes.npy')
    plot_learning_curve(x=maddpg_episodes,
                        rewards=(maddpg_rewards),
                        filename='plot/maddpg.png')