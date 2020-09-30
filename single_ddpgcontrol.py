# -*- coding: utf-8 -*-

# @Author: KeyangZhang
# @Email: 3238051@qq.com
# @Date: 2020-04-26 19:24:48
# @LastEditTime: 2020-05-03 15:03:25
# @LastEditors: Keyangzhang

from single_intersection_env import CTMEnv
import tensorflow as tf
import numpy as np
import time
from collections import namedtuple
import csv
import os

from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.networks import network
from tf_agents.networks import encoding_network
from tf_agents.utils import common as common_utils
from tf_agents.networks import utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils
from tf_agents.policies import random_tf_policy
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory


class ActorNetwork(network.Network):

    def __init__(self,
                 observation_spec,  
                 action_spec,
                 activation_fn=tf.keras.activations.tanh,
                 name='ActorNetwork'):

        super(ActorNetwork, self).__init__(input_tensor_spec=observation_spec,
                                           state_spec=(),
                                           name=name)

        self._action_spec = action_spec

        initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)

        # self._bn_layer = tf.keras.layers.BatchNormalization(axis=-1)

        self._dense1 = tf.keras.layers.Dense(units=10,
                                             kernel_initializer=initializer, 
                                             activation=activation_fn)
        self._dense2 = tf.keras.layers.Dense(units=10,
                                             kernel_initializer=initializer, 
                                             activation=activation_fn)
        self._dense3 = tf.keras.layers.Dense(units=10,
                                             kernel_initializer=initializer, 
                                             activation=activation_fn)
        self._action_projection_layer = tf.keras.layers.Dense(action_spec.shape.num_elements(),
                                                              kernel_initializer=initializer,
                                                              activation=tf.keras.activations.sigmoid)

    def call(self, observations, step_type=(), network_state=()):

        # observations = self._bn_layer(observations)
        tempt = self._dense1(observations)
        tempt = self._dense2(tempt)
        tempt = self._dense3(tempt)
        actions = self._action_projection_layer(tempt)
        actions = common_utils.scale_to_spec(actions, self._action_spec)

        return actions, network_state


class CriticNetwork(network.Network):

    def __init__(self,
                 observation_spec,
                 action_spec,
                 q_value_spec,
                 activation_fn=tf.keras.activations.tanh,
                 name='CriticNetwork'):
        super(CriticNetwork, self).__init__(
            input_tensor_spec=(observation_spec, action_spec), state_spec=(), name=name)

        self._q_value_spec = q_value_spec

        initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
        
        self._preprocess_observation_layer = tf.keras.layers.Dense(4, activation=activation_fn)
        self._preprocess_action_layer = tf.keras.layers.Dense(5, activation=activation_fn)
        self._combine_layer = tf.keras.layers.Concatenate(axis=-1)

        self._dense1 = tf.keras.layers.Dense(units=10,
                                             kernel_initializer=initializer, 
                                             activation=activation_fn)
        self._dense2 = tf.keras.layers.Dense(units=10,
                                             kernel_initializer=initializer, 
                                             activation=activation_fn)
        
        self._q_value_projection_layer = tf.keras.layers.Dense(
            1, activation=tf.keras.activations.sigmoid)

    def call(self, inputs, step_type=(), network_state=()):
        observations, actions = inputs
        observations = self._preprocess_observation_layer(observations)
        actions = self._preprocess_action_layer(actions)
        
        tempt = self._combine_layer([observations,actions])
        tempt = self._dense1(tempt)
        tempt = self._dense2(tempt)

        q_values = self._q_value_projection_layer(tempt)
        q_values = common_utils.scale_to_spec(q_values, self._q_value_spec)


        return q_values, network_state
    


class ExpertPolicy(object):
    
    def __init__(self):
        self.ActionStep = namedtuple('action_step',['action','info'])

    def action(self,timesteps):

        x  = tf.constant([[2.97,2.03,2.97,2.03,6]],dtype='float32')
        action_step = self.ActionStep(x,())
        
        return action_step



def compute_avg_return(environment, policy, num_episodes=1):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            print('obs:',time_step.observation.numpy()[0],'action:',action_step.action.numpy()[0],'reward:',time_step.reward.numpy()[0])
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    print('-------------------------------')
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)




num_iterations = 1
replay_buffer_max_length = 100
batch_size = 2
collect_steps_per_iteration = 90
random_collect_steps_per_iteration = 90
collect_steps_inital = 300
collect_steps_expert = 0
log_interval = 1
eval_interval = 5
lrchange_interval = 25
num_eval_episodes = 1
learning_rate=0.005

q_value_spec = tensor_spec.BoundedTensorSpec(shape=(),
                                             dtype=np.float32,
                                             minimum=-120,
                                             maximum=0,
                                             name='q_value')


train_start_time = time.time()

train_py_env = CTMEnv()
eval_py_env = CTMEnv()

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)


actor_network = ActorNetwork(train_env.observation_spec(),train_env.action_spec())
critic_network = CriticNetwork(train_env.observation_spec(),train_env.action_spec(),q_value_spec)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

agent = ddpg_agent.DdpgAgent(train_env.time_step_spec(),
                             train_env.action_spec(),
                             actor_network,
                             critic_network,
                             actor_optimizer=optimizer,
                             critic_optimizer=optimizer,
                             gamma=0.99,
                             ou_stddev=5.0,
                             ou_damping=0.9)

agent.initialize()



eval_policy = agent.policy
collect_policy = agent.collect_policy
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())
expert_policy = ExpertPolicy()

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)


collect_data(train_env, expert_policy, replay_buffer, steps=collect_steps_expert)
collect_data(train_env, random_policy, replay_buffer, steps=collect_steps_inital)


dataset = replay_buffer.as_dataset(num_parallel_calls=3,
                                   sample_batch_size=batch_size,
                                   num_steps=2).prefetch(3)

iterator = iter(dataset)


e_s_time = time.time()  
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
print('initial return:',avg_return)
returns = [avg_return]
train_losses = [None]
iteration_dur_times = [time.time()-e_s_time]


for _ in range(num_iterations):
    
    iteration_start_time = time.time()
    # Collect a few steps using collect_policy and save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
        collect_step(train_env, agent.collect_policy, replay_buffer)

    for _ in range(random_collect_steps_per_iteration):
        collect_step(train_env, random_policy, replay_buffer)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))
        train_losses.append(train_loss)

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
    
    # if step % lrchange_interval == 0:
    #     optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate*(1-step/num_iterations))
    #     agent._actor_optimizer = optimizer
    #     agent._critic_optimizer = optimizer
    
    iteration_end_time = time.time()
    iteration_dur_times.append(iteration_end_time-iteration_start_time)

print(time.time()-train_start_time)


# save model and result
date = time.localtime()
addinfo = '{0}-{1}-{2}-{3}-{4}'.format(date.tm_mon,date.tm_mday,date.tm_hour,date.tm_min,date.tm_year)

checkpoint=tf.train.Checkpoint(actor_network=actor_network,critic_network=critic_network)
checkpoint.save('./result/single-agent/networksave/actorcritc-{0}.ckpt'.format(addinfo))

if not os.path.exists('./result/single-agent'):
    os.makedirs('./result/single-agent')

with open('./result/single-agent/reward_{0}.csv'.format(addinfo),'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step','avg_reward'])
    for i,avg_reward in enumerate(returns):
        writer.writerow([i,avg_reward])

with open('./result/single-agent/trainloss_{0}.csv'.format(addinfo),'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step','train_loss'])
    for i,loss in enumerate(train_losses):
        writer.writerow([i,loss])
