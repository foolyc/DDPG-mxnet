#!/usr/bin/env python
# --------------------------------------------------------
# Deep Deterministic Policy Gradient
# Written by Chao Yu
# --------------------------------------------------------

import mxnet as mx
import numpy as np
import math
from config import *
from mxnet.ndarray import square
from mxnet.ndarray import sqrt

class DDPGNet(object):
    """
    Continous Multi-Layer Perceptron Q-Value Network
    for determnistic policy training.
    """
    def __init__(self, state_dim, action_dim):
        # own code
        if flg_gpu :
            self.ctx = mx.gpu(0)
        else:
            self.ctx = mx.cpu()
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.obs = mx.symbol.Variable("obs")
        self.act = mx.symbol.Variable("act")
        self.yval = mx.symbol.Variable("yval")

    def create_actor_net(self, obs):
        actor_fc1 = mx.symbol.FullyConnected(
            data=obs,
            name="actor_fc1",
            num_hidden=LAYER1_SIZE)
        actor_relu1 = mx.symbol.Activation(
            data=actor_fc1,
            name="actor_relu1",
            act_type="relu")
        actor_fc2 = mx.symbol.FullyConnected(
            data=actor_relu1,
            name="actor_fc2",
            num_hidden=LAYER2_SIZE)
        actor_relu2 = mx.symbol.Activation(
            data=actor_fc2,
            name="actor_relu2",
            act_type="relu")
        actor_fc3 = mx.symbol.FullyConnected(
            data=actor_relu2,
            name='actor_fc3',
            num_hidden=self.action_dim)
        actor_relu2 = mx.symbol.Activation(
            data=actor_fc3,
            name="actor_relu3",
            act_type="tanh")
        return actor_relu2

    def create_critic_net(self, obs, act):
        critic_fc1 = mx.symbol.FullyConnected(
            data=obs,
            name="critic_fc1",
            num_hidden=LAYER1_SIZE)
        critic_relu1 = mx.symbol.Activation(
            data=critic_fc1,
            name="critic_relu1",
            act_type="relu")
        critic_concat = mx.symbol.Concat(critic_relu1, act, name="critic_concat")
        critic_fc2 = mx.symbol.FullyConnected(
            data=critic_concat,
            name="critic_fc2",
            num_hidden=LAYER2_SIZE)
        critic_relu2 = mx.symbol.Activation(
            data=critic_fc2,
            name="critic_relu2",
            act_type="relu")
        critic_fc3 = mx.symbol.FullyConnected(
            data=critic_relu2,
            name="critic_fc3",
            num_hidden=1)
        return critic_fc3


    def init(self):
        qval_sym_critic = self.create_critic_net(self.obs, self.act)
        act_sym = self.create_actor_net(self.obs)
        qval_sym_actor = self.create_critic_net(self.obs, act_sym)

        critic_loss = 1.0 / self.batch_size * mx.symbol.sum(mx.symbol.square(qval_sym_critic - self.yval))
        critic_loss = mx.symbol.MakeLoss(critic_loss, name="critic_loss")
        critic_out = mx.sym.Group([critic_loss, mx.sym.BlockGrad(qval_sym_critic)])
        critic_input_shapes = {
            "obs": (self.batch_size, self.state_dim),
            "act": (self.batch_size, self.action_dim),
            "yval": (self.batch_size, 1)}
        self.critic = critic_out.simple_bind(ctx=self.ctx, **critic_input_shapes)

        # for debug
        # actor_loss =  -1.0 / self.batch_size * mx.symbol.sum(qval_sym_critic)
        # self.grad = mx.symbol.MakeLoss(actor_loss).simple_bind(ctx=self.ctx, **grad_input_shape)
        grad_arg_dict = {}
        for name, arr in self.critic.arg_dict.items():
            if name not in ['yval']:
                grad_arg_dict[name] = arr
        self.grad_batch = mx.nd.empty((self.batch_size, 1), ctx=self.ctx)
        self.grad = mx.symbol.MakeLoss(-qval_sym_critic).bind(ctx=self.ctx, args=grad_arg_dict,
                                                              args_grad={'act': self.grad_batch},
                                                              shared_exec=self.critic)

        actor_input_shapes = {
            "obs": (self.batch_size, self.state_dim)}
        self.actor = act_sym.simple_bind(ctx=self.ctx, **actor_input_shapes)

        new_input_shapes = {"obs": (1, actor_input_shapes["obs"][1])}
        self.actor_one = self.actor.reshape(**new_input_shapes)


        target_out = mx.sym.Group([qval_sym_actor, act_sym])
        self.target = target_out.simple_bind(ctx=self.ctx, **actor_input_shapes)

        # define optimizer
        self.critic_updater = mx.optimizer.get_updater(mx.optimizer.create(critic_updater, learning_rate=critic_lr, wd=weight_decay))
        self.actor_updater = mx.optimizer.get_updater(mx.optimizer.create(actor_updater, learning_rate=actor_lr))

        # init params
        for name, arr in self.target.arg_dict.items():
            if 'fc1' in name:
                scale = 1/math.sqrt(self.state_dim)
                initializer = mx.initializer.Uniform(scale)
                initializer._init_weight(name, arr)
            elif 'fc2' in name:
                if 'critic' in name:
                    scale = 1/math.sqrt(LAYER1_SIZE + self.action_dim)
                    initializer = mx.initializer.Uniform(scale)
                    initializer._init_weight(name, arr)
                elif 'actor' in name:
                    scale = 1 / math.sqrt(LAYER1_SIZE)
                    initializer = mx.initializer.Uniform(scale)
                    initializer._init_weight(name, arr)
                else:
                    pass
            elif 'fc3' in name:
                scale = 3e-3
                initializer = mx.initializer.Uniform(scale)
                initializer._init_weight(name, arr)
            else:
                pass

        self.critic_state = {}
        self.actor_state = {}
        for name, arr in self.target.arg_dict.items():
            if 'actor' in name:
                arr.copyto(self.actor.arg_dict[name])
                shape = self.actor.arg_dict[name].shape
                self.actor_state[name] = (mx.nd.zeros(shape, self.ctx), mx.nd.zeros(shape, self.ctx))
                # arr.copyto(self.actor_one.arg_dict[name])
            if 'critic' in name:
                arr.copyto(self.critic.arg_dict[name])
                shape = self.critic.arg_dict[name].shape
                self.critic_state[name] = (mx.nd.zeros(shape, self.ctx), mx.nd.zeros(shape, self.ctx))
                # arr.copyto(self.grad.arg_dict[name])


    def update_critic(self, obs, act, yval):
        self.critic.arg_dict["obs"][:] = obs
        self.critic.arg_dict["act"][:] = act
        self.critic.arg_dict["yval"][:] = yval
        self.critic.forward(is_train=True)
        # print self.critic.outputs[0].asnumpy()
        self.critic.backward()
        # print "critic param :", self.critic.arg_dict['critic_fc3_bias'].asnumpy()
        # print "critic grad :", self.critic.grad_dict['critic_fc3_bias'].asnumpy()
        # update_param(self, weight, grad, state, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, wd=0)
        for i, index in enumerate(self.critic.grad_dict):
            if 'critic' in index:
                self.update_param(self.critic.arg_dict[index], self.critic.grad_dict[index], self.critic_state[index], lr=critic_lr, wd = weight_decay)
                # self.critic.arg_dict[index].copyto(self.grad.arg_dict[index])
        # print "updating ................."
        # print "critic param :", self.critic.arg_dict['critic_fc3_bias'].asnumpy()
        # print "critic grad :", self.critic.grad_dict['critic_fc3_bias'].asnumpy()
        # print "step updating over.............................."

    def update_actor(self, obs):
        self.actor.arg_dict["obs"][:] = obs
        self.actor.forward(is_train=True)
        # for test
        self.grad.arg_dict["obs"][:] = obs
        self.grad.arg_dict['act'][:] = self.actor.outputs[0]
        self.grad.forward()
        self.grad.backward()
        # print np.std(grad_batch.asnumpy()),np.mean(grad_batch.asnumpy())

        # print self.grad_batch.asnumpy()

        self.actor.backward(self.grad_batch)
        for i, index in enumerate(self.actor.arg_dict):
            if 'actor' in index:
                # print index
                # print self.actor.grad_dict[index].asnumpy()
                self.update_param(self.actor.arg_dict[index], self.actor.grad_dict[index], self.actor_state[index],
                                  lr=actor_lr)
                # self.actor.arg_dict[index].copyto(self.actor_one.arg_dict[index])
        # self.actor.forward(is_train=True)
        # print "actor loss after update", self.actor.outputs[0].asnumpy()

    def update_target(self):
        for name, arr in self.target.arg_dict.items():
            if 'actor' in name:
                self.target.arg_dict[name] = (1 - soft_target_tau)* arr + soft_target_tau * self.actor.arg_dict[name]
            elif 'critic' in name:
                self.target.arg_dict[name] = (1 - soft_target_tau)* arr + soft_target_tau * self.critic.arg_dict[name]

    def get_target_q(self, obs):
        self.target.arg_dict["obs"][:] = obs
        self.target.forward(is_train=False)
        return self.target.outputs[0]

    def get_step_action(self, obs):
        # single observation
        self.actor_one.arg_dict["obs"][:] = obs
        self.actor_one.forward(is_train=False)
        return self.actor_one.outputs[0].asnumpy()[0][0]


    def update_param(self, weight, grad, state, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, wd=0):
        mean, variance = state

        mean *= beta1
        mean += grad * (1. - beta1)

        variance *= beta2
        variance += (1 - beta2) * square(grad)

        coef1 = 1. - beta1
        coef2 = 1. - beta2
        lr *= math.sqrt(coef2) / coef1

        weight -= lr * mean / (sqrt(variance) + epsilon)
        if wd > 0.:
            weight[:] -= (lr * wd) * weight


if __name__ =="__main__":
    d=DDPGNet(4,1)
    d.init()