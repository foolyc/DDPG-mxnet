#!/usr/bin/env python
# --------------------------------------------------------
# Deep Deterministic Policy Gradient
# Written by Chao Yu
# --------------------------------------------------------

flg_gpu = True
batch_size = 64
n_epochs = 1000
epoch_length = 1000
memory_size = 1000000
memory_start_size = 10000
discount = 0.99
max_path_length = 1000
eval_samples = 10000
critic_updater = "adam"
critic_lr = 1e-3
actor_updater = "adam"
actor_lr = 1e-4
soft_target_tau = 1e-3
n_updates_per_sample = 1
include_horizon_terminal = False
seed = 12345

GAMMA = 0.99
LAYER1_SIZE = 400
LAYER2_SIZE = 300


ENV_NAME = 'InvertedPendulum-v1'
EPISODES = 1000000
TEST = 50