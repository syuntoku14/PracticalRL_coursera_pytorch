import os
import sys
try:
    sys.path.remove(
        '/home/syuntoku14/catkin_ws/devel/lib/python2.7/dist-packages')
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import gym
from gym.core import ObservationWrapper
from gym.spaces import Box
import cv2
from framebuffer import FrameBuffer
from replay_buffer import ReplayBuffer


class ReplayBuffer(ReplayBuffer):
    def concat(self, exp_replay):
        self._storage += exp_replay._storage
        excess = max(len(self._storage) - self._maxsize, 0)
        print(excess)
        self._storage = self._storage[excess:]


class PreprocessAtari(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        ObservationWrapper.__init__(self, env)

        self.img_size = (84, 84)
        self.observation_space = Box(
            0.0, 1.0, (1, self.img_size[0], self.img_size[1]))

    def _observation(self, img):
        img = img[34:-16, 8:-8, :]
        img = cv2.resize(img, self.img_size)
        img = img.mean(-1, keepdims=True)  # grayscale
        img = img.astype('float32') / 255.

        return img.transpose([2, 0, 1])


def make_env():
    env = gym.make("BreakoutDeterministic-v4")
    env = PreprocessAtari(env)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
    return env


class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        # input obs, output n_actions
        # The network is based on "Mnih, 2015"
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.l1 = nn.Linear(64*7*7, 512)
        self.l2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size()[0], -1)
        x = self.l1(x)
        x = self.l2(x)
        return x


class DQNAgent:
    def __init__(self, state_shape, n_actions, epsilon=0, reuse=False):
        """A simple DQN agent"""

        self.dqn = DQN(n_actions)
        self.epsilon = epsilon

    def get_qvalues(self, states):
        """takes agent's observation, returns qvalues. """
        qvalues = self.dqn(states)
        return qvalues

    def get_qvalues_for_actions(self, qvalues, actions):
        return qvalues.gather(1,
                              actions.unsqueeze(0).transpose(0, 1)).squeeze()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = torch.tensor(
            np.random.choice(n_actions, size=batch_size))
        best_actions = qvalues.argmax(1)
        should_explore = torch.tensor(np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])).byte()
        return torch.where(should_explore, random_actions, best_actions)


def play_and_record(agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer.
    :returns: return sum of rewards over time
    """
    # Make sure that the state is only one batch state, 4x84x84
    # State at the beginning of rollout
    s = env.framebuffer
    R = 0.0

    # Play the game for n_steps as per instructions above
    for t in range(n_steps):
        qvalues = agent.get_qvalues(torch.tensor(s).unsqueeze(0))
        action = agent.sample_actions(qvalues).item()
        next_s, r, done, _ = env.step(action)
        exp_replay.add(s, action, r, next_s, done=done)
        if done == True:
            s = env.reset()
        else:
            s = next_s
    return R


def optimize(current_action_qvalues, optimizer, target_dqn,
             reward_batch, next_obs_batch, is_done_batch):
    next_qvalues_target = target_dqn.get_qvalues(next_obs_batch)

    # compute state values by taking max over next_qvalues_target for all actions
    next_state_values_target = next_qvalues_target.max(1)[0]
    next_state_values_target = torch.where(torch.tensor(is_done_batch).byte(),
                                           torch.tensor(reward_batch),
                                           torch.tensor(next_state_values_target))

    # compute Q_reference(s,a) as per formula above.
    reference_qvalues = reward_batch + gamma * next_state_values_target

    # Define loss function for sgd.
    td_loss = (current_action_qvalues - reference_qvalues) ** 2
    td_loss = torch.mean(td_loss)

    optimizer.zero_grad()
    td_loss.backward()
    optimizer.step()

    return td_loss.item()


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            s = torch.tensor(s).unsqueeze(0)
            qvalues = agent.get_qvalues(s)
            action = qvalues.argmax(
                dim=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                break

        rewards.append(reward)
    return np.mean(rewards)


def convert_to_tensor(obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch):
    obs_batch = torch.tensor(obs_batch)
    act_batch = torch.tensor(act_batch)
    reward_batch = torch.tensor(reward_batch).float()
    next_obs_batch = torch.tensor(next_obs_batch)
    is_done_batch = is_done_batch.astype(np.int)
    return obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch


def save_data(folder_path, agent, mean_reward_history, td_loss_history):
    torch.save(policy_agent.dqn.state_dict(),
               folder_path + 'atari_dqn_state_dict.pt')
    with open(folder_path + 'mean_reward_history.l', 'wb') as f:
        pickle.dump(mean_reward_history, f)
    with open(folder_path + 'td_loss_history.l', 'wb') as f:
        pickle.dump(td_loss_history, f)


def load_data(folder_path):
    state_dict = None
    mean_reward_history = []
    td_loss_history = []
    state_dict = torch.load(folder_path + 'atari_dqn_state_dict.pt')
    with open(folder_path + 'mean_reward_history.l', 'rb') as f:
        mean_reward_history = pickle.load(f)
    with open(folder_path + 'td_loss_history.l', 'rb') as f:
        td_loss_history = pickle.load(f)

    return state_dict, mean_reward_history, td_loss_history


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env()
    env.reset()
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape

    mean_rw_history = []
    td_loss_history = []

    gamma = 0.99
    policy_agent = DQNAgent(state_dim, n_actions, epsilon=0.5)
    target_agent = DQNAgent(state_dim, n_actions)

    rl_path = '/home/syuntoku14/OneDrive/ai/coursera/Practical_RL-coursera/week4_approx/speedup_dqn/data/'
    state_dict, mean_rw_history, td_loss_history = load_data(rl_path)
    policy_agent.dqn.load_state_dict(state_dict)
    policy_agent.dqn.eval()
    target_agent.dqn.load_state_dict(state_dict)
    target_agent.dqn.eval()

