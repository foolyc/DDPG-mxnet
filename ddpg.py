#!/usr/bin/env python
# --------------------------------------------------------
# Deep Deterministic Policy Gradient
# Written by Chao Yu
# --------------------------------------------------------

import gym

import os
os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'
import mxnet as mx
import numpy as np
print os.environ['MXNET_ENGINE_TYPE']
from ou_noise import OUNoise
from DDPGNet import DDPGNet
from replay_buffer import ReplayBuffer
from ou_noise import OUNoise
from config import *
# Hyper Parameters:



class DDPG:
    """docstring for DDPG"""

    def __init__(self, env):
        mx.random.seed(seed)
        np.random.seed(seed)
        self.env = env
        if flg_gpu :
            self.ctx = mx.gpu(0)
        else:
            self.ctx = mx.cpu()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.ddpgnet = DDPGNet(self.state_dim, self.action_dim)
        self.exploration_noise = OUNoise(self.action_dim)
        self.replay_buffer = ReplayBuffer(memory_size)

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.memory_size = memory_size
        self.memory_start_size = memory_start_size
        self.discount = discount
        self.max_path_length = max_path_length
        self.eval_samples = eval_samples

        self.soft_target_tau = soft_target_tau
        self.n_updates_per_sample = n_updates_per_sample
        self.include_horizon_terminal = include_horizon_terminal

        self.qfunc_loss_averages = []
        self.policy_loss_averages = []
        self.q_averages = []
        self.y_averages = []
        self.strategy_path_returns = []
        self.ddpgnet.init()
        self.train_step = 0

    def train(self):
        # print "train step",self.time_step
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.get_batch(self.batch_size)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch, [self.batch_size, self.action_dim])

        # Calculate y_batch
        next_qvals = self.ddpgnet.get_target_q(next_state_batch).asnumpy()

        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * next_qvals[i][0])
        y_batch = np.resize(y_batch, [self.batch_size, 1])

        # Update critic by minimizing the loss L
        self.ddpgnet.update_critic(state_batch, action_batch, y_batch)

        # Update actor by maxmizing Q
        self.ddpgnet.update_actor(state_batch)

        self.train_step += 1
        # update target networks
        self.ddpgnet.update_target()



    def noise_action(self, state):
        # Select action a_t according to the current policy and exploration noise
        state = np.reshape(state, (1, self.state_dim))
        action = self.ddpgnet.get_step_action(state)
        return action + self.exploration_noise.noise()

    def action(self, state):
        state = np.reshape(state, (1, self.state_dim))
        action = self.ddpgnet.get_step_action(state)
        return action

    def perceive(self, state, action, reward, next_state, done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Store transitions to replay start size then start training
        if self.replay_buffer.count() > memory_start_size:
            self.train()

            # if self.time_step % 10000 == 0:
            # self.actor_network.save_network(self.time_step)
            # self.critic_network.save_network(self.time_step)

        # Re-iniitialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()










