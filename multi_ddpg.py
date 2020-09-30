# -*- coding: utf-8 -*-

# @Author: KeyangZhang
# @Email: 3238051@qq.com
# @Date: 2020-05-03 10:13:25
# @LastEditTime: 2020-05-03 19:22:47
# @LastEditors: Keyangzhang

from multi_intersection_com import CTMEnv,ActorNetwork,CriticNetwork
import tensorflow as tf
import numpy as np
import time
import os
import csv

from tf_agents.environments import tf_py_environment
from tf_agents.agents.ddpg import ddpg_agent

from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from tf_agents.trajectories.trajectory import Trajectory,policy_step
from tf_agents.trajectories.time_step import TimeStep,time_step_spec
from tf_agents.specs.tensor_spec import TensorSpec,BoundedTensorSpec

def random_collect_step(environment,policies,replay_buffers):
    
    aggregate_time_step = environment.current_time_step()
    is_first_step = aggregate_time_step.reward.shape == (1,) # first time env reward is [0], while others are [[r1,r2,r3,r4]]
    aggregate_action_step = {} # action in form of [[e1,e2,e3,e4]]
    squeezed_action_step = {} # action in form of [e1,e2,e3,e4]
    if is_first_step:
        for i,name in enumerate(AGENT_NAMES):
            # create time_step satisfies the spec of single random policy
            time_step = TimeStep(aggregate_time_step.step_type[0],aggregate_time_step.reward[0],aggregate_time_step.discount[0],aggregate_time_step.observation[name][0])

            
            # random policy recieve observation(time_step) and return action(action_step)
            action_step = policies[i].action(time_step)
            squeezed_action_step[name]=action_step
            
            # create action_step(policy_step) satisfies the spec of aggregate environment
            action = tf.convert_to_tensor([action_step.action],dtype='float32')
            action_step = policy_step.PolicyStep(action,action_step.state,action_step.info)
            aggregate_action_step[name]=action_step
    else:
        for i,name in enumerate(AGENT_NAMES):
            # create time_step satisfies the spec of single random policy
            time_step = TimeStep(aggregate_time_step.step_type[0][0],aggregate_time_step.reward[0][i],aggregate_time_step.discount[0],aggregate_time_step.observation[name][0])

            # random policy recieve observation(time_step) and return action(action_step)
            action_step = policies[i].action(time_step)
            squeezed_action_step[name]=action_step

            # create action_step(policy_step) satisfies the spec of aggregate environment
            action = tf.convert_to_tensor([action_step.action],dtype='float32')
            action_step = policy_step.PolicyStep(action,action_step.state,action_step.info)
            aggregate_action_step[name]=action_step

    # let enviroment take one step forward.
    aggregate_next_time_step = environment.step(aggregate_action_step)

    if is_first_step:
        for i, name in enumerate(AGENT_NAMES):
            action_step = aggregate_action_step[name]
            observation = tf.convert_to_tensor(aggregate_time_step.observation[name].numpy(),dtype='float32')
            tra = Trajectory(aggregate_time_step.step_type[0],
                       observation,
                       action_step.action,
                       action_step.info,
                       aggregate_next_time_step.step_type[0][0], 
                       aggregate_time_step.reward[0],
                       aggregate_time_step.discount[0])
            replay_buffers[i].add_batch(tra)
            
    else:
        for i, name in enumerate(AGENT_NAMES):
            action_step = aggregate_action_step[name]
            observation = tf.convert_to_tensor(aggregate_time_step.observation[name].numpy(),dtype='float32')
            if aggregate_next_time_step.step_type.shape == (1,):
                tra = Trajectory(aggregate_time_step.step_type[0][0],
                        observation,
                        action_step.action,
                        action_step.info,
                        aggregate_next_time_step.step_type[0],
                        aggregate_time_step.reward[0][i],
                        aggregate_time_step.discount[0])
            else:
                tra = Trajectory(aggregate_time_step.step_type[0][0],
                        observation,
                        action_step.action,
                        action_step.info,
                        aggregate_next_time_step.step_type[0][0],
                        aggregate_time_step.reward[0][i],
                        aggregate_time_step.discount[0]) 
            replay_buffers[i].add_batch(tra)


def collect_step(environment,policies,replay_buffers):

    aggregate_time_step = environment.current_time_step()
    is_first_step = aggregate_time_step.reward.shape == (1,) # first time env reward is [0], while others are [[r1,r2,r3,r4]]
    aggregate_action_step = {} # action in form of [[e1,e2,e3,e4]]

    if is_first_step:
        for i,name in enumerate(AGENT_NAMES):
            # abstract observation and construct time step for each agent separately
            # for the first step, env output step type is in shape [num]
            observation = tf.convert_to_tensor(aggregate_time_step.observation[name].numpy(),dtype='float32')
            time_step = TimeStep(aggregate_time_step.step_type[0],aggregate_time_step.reward[0],aggregate_time_step.discount[0],observation)

            # agent policy receive time_step and output single action
            action_step = policies[i].action(time_step)
            
            # add single action to joint action
            action = tf.convert_to_tensor(action_step.action,dtype='float32')
            action_step = policy_step.PolicyStep(action,action_step.state,action_step.info)
            aggregate_action_step[name]=action_step
    else:
        for i,name in enumerate(AGENT_NAMES):
            # abstract observation and construct time step for each agent separately
            # for the step other than the first step, env output step type is in shape [[num,num,num,num]]
            observation = tf.convert_to_tensor(aggregate_time_step.observation[name].numpy(),dtype='float32')
            time_step = TimeStep(aggregate_time_step.step_type[0][0],aggregate_time_step.reward[0][i],aggregate_time_step.discount[0],observation)
            
            # agent policy receive time_step and output single action
            action_step = policies[i].action(time_step)

            # add single action to joint action
            action = tf.convert_to_tensor(action_step.action,dtype='float32')
            action_step = policy_step.PolicyStep(action,action_step.state,action_step.info)
            aggregate_action_step[name]=action_step

    # let enviroment take one step forward.
    aggregate_next_time_step = environment.step(aggregate_action_step)

    if is_first_step:
        for i, name in enumerate(AGENT_NAMES):
            action_step = aggregate_action_step[name]
            observation = tf.convert_to_tensor(aggregate_time_step.observation[name].numpy(),dtype='float32')
            tra = Trajectory(aggregate_time_step.step_type[0],
                       observation,
                       action_step.action,
                       action_step.info,
                       aggregate_next_time_step.step_type[0][0], 
                       aggregate_time_step.reward[0],
                       aggregate_time_step.discount[0])
            replay_buffers[i].add_batch(tra)
            
    else:
        for i, name in enumerate(AGENT_NAMES):
            action_step = aggregate_action_step[name]
            observation = tf.convert_to_tensor(aggregate_time_step.observation[name].numpy(),dtype='float32')
            if aggregate_next_time_step.step_type.shape == (1,):
                tra = Trajectory(aggregate_time_step.step_type[0][0],
                        observation,
                        action_step.action,
                        action_step.info,
                        aggregate_next_time_step.step_type[0],
                        aggregate_time_step.reward[0][i],
                        aggregate_time_step.discount[0])
            else:
                tra = Trajectory(aggregate_time_step.step_type[0][0],
                        observation,
                        action_step.action,
                        action_step.info,
                        aggregate_next_time_step.step_type[0][0],
                        aggregate_time_step.reward[0][i],
                        aggregate_time_step.discount[0]) 
            replay_buffers[i].add_batch(tra)

def collect_data(enviroment,policies,replay_buffers,steps):
    for _ in range(steps):
        collect_step(enviroment,policies,replay_buffers)

def random_collect_data(enviroment,random_policies,replay_buffers,steps):
    for _ in range(steps):
        random_collect_step(enviroment,random_policies,replay_buffers)


def compute_avg_return(environment, policies, num_episodes=1):

    total_return = [0 for _ in AGENT_NAMES]
    
    for _ in range(num_episodes):

        aggregate_time_step = environment.reset()
        episode_return = [0 for _ in AGENT_NAMES]
        
        while not aggregate_time_step.is_last().numpy().all():
            
            is_first_step = aggregate_time_step.reward.shape == (1,)
            aggregate_action = {}
            for i, name in enumerate(AGENT_NAMES):
                if is_first_step:
                    observation = tf.convert_to_tensor(aggregate_time_step.observation[name].numpy(),dtype='float32')
                    time_step = TimeStep(aggregate_time_step.step_type[0],aggregate_time_step.reward[0],aggregate_time_step.discount[0],observation)
                    
                else:
                    observation = tf.convert_to_tensor(aggregate_time_step.observation[name].numpy(),dtype='float32')
                    time_step = TimeStep(aggregate_time_step.step_type[0][0],aggregate_time_step.reward[0][i],aggregate_time_step.discount[0],observation)
                action_step = policies[i].action(time_step)
                aggregate_action[name]=action_step
            
            for i in range(len(AGENT_NAMES)):
                if is_first_step:
                    episode_return[i]+=0
                else:
                    episode_return[i]+=aggregate_time_step.reward[0][i]
            
            aggregate_time_step = environment.step(aggregate_action)

        for i in range(len(AGENT_NAMES)):
            total_return[i]+=episode_return[i]

    avg_return = [0 for _ in range(len(AGENT_NAMES))]
    for i in range(len(AGENT_NAMES)):
        avg_return[i] = total_return[i] / num_episodes
    
    avg_return = [r.numpy() for r in avg_return]
    
    return avg_return


# hyper-parameters
LEARNING_RATE = 0.005
REPLAY_BUFFER_MAX_LENGTH=100
BATCH_SIZE = 256
NUM_EVAL_EPISODES = 1
AGENT_NAMES = ['inter0','inter1','inter2','coordinate']
NUM_ITERATIONS = 5
RANDOM_COLLECT_STEPS_INITIAL = 60
AGENTS_COLLECT_STEPS_INITIAL = 60
AGENTS_COLLECT_STEPS_PER_ITERATION = 0
RANDOM_COLLECT_STEPS_PER_ITERATION = 0
LOG_INTERVAL = 1
EVAL_INTERVAL = 5



# create some specifications for single agent and coordinate agent
FIXED_STEP_TYPE= tf.convert_to_tensor(0,dtype='int32')

obs_spec4independent_agent=BoundedTensorSpec(shape=(8,), dtype=tf.float32, minimum=0, maximum=3.4028235e+38, name='observation')
act_spec4independent_agent=BoundedTensorSpec(shape=(4,), dtype=tf.float32, minimum=0, maximum=10, name='action')
q_spec4independent_agent = BoundedTensorSpec(shape=(),dtype=tf.float32,minimum=-120,maximum=0,name='q_value')
ts_spec4independent_agent = time_step_spec(obs_spec4independent_agent)

obs_spec4coordinate_agent=BoundedTensorSpec(shape=(12,), dtype=tf.float32, minimum=0, maximum=3.4028235e+38, name='observation')
act_spec4coordinate_agent=BoundedTensorSpec(shape=(3,), dtype=tf.float32, minimum=40, maximum=100, name='action')
q_spec4coordinate_agent = BoundedTensorSpec(shape=(),dtype=tf.float32,minimum=-360,maximum=0,name='q_value')
ts_spec4coordintae_agent = time_step_spec(obs_spec4coordinate_agent)


tf.compat.v1.enable_v2_behavior()
tf.keras.backend.set_floatx('float32')


train_py_env = CTMEnv()
eval_py_env = CTMEnv()
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eavl_env = tf_py_environment.TFPyEnvironment(eval_py_env)


# create agent system
# first, create three indenpent single intersection agents
# then, create one coordinate agent
agent_system = []
replay_buffers = []
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE)
for i in range(3):
    actor_network = ActorNetwork(obs_spec4independent_agent,act_spec4independent_agent)
    critic_network = CriticNetwork(obs_spec4independent_agent,act_spec4independent_agent,q_spec4independent_agent)
    agent = ddpg_agent.DdpgAgent(ts_spec4independent_agent, act_spec4independent_agent, actor_network, critic_network,
                                 actor_optimizer=optimizer,
                                 critic_optimizer=optimizer,
                                 gamma=1,
                                 ou_stddev=5.0,
                                 ou_damping=0.9)
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                                   batch_size=train_env.batch_size,
                                                                   max_length=REPLAY_BUFFER_MAX_LENGTH)

    agent_system.append(agent)
    replay_buffers.append(replay_buffer)
#  create coordinate agent
actor_network = ActorNetwork(obs_spec4coordinate_agent, act_spec4coordinate_agent)
critic_network = CriticNetwork(obs_spec4coordinate_agent, act_spec4coordinate_agent, q_spec4coordinate_agent)
agent = ddpg_agent.DdpgAgent(ts_spec4coordintae_agent, act_spec4coordinate_agent, actor_network, critic_network,
                                        actor_optimizer=optimizer,
                                        critic_optimizer=optimizer,
                                        gamma=1,
                                        ou_stddev=5.0,
                                        ou_damping=0.9)
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                                batch_size=train_env.batch_size,
                                                                max_length=REPLAY_BUFFER_MAX_LENGTH)
agent_system.append(agent)
replay_buffers.append(replay_buffer)

for agent in agent_system:
    agent.initialize()

# create policies for each agent
collect_policies = [agent.collect_policy for agent in agent_system]
eval_policies = [agent.policy for agent in agent_system]
random_policies = []
for _ in AGENT_NAMES[:3]:
    random_policy = random_tf_policy.RandomTFPolicy(ts_spec4independent_agent,act_spec4independent_agent)
    random_policies.append(random_policy)
random_policy = random_tf_policy.RandomTFPolicy(ts_spec4coordintae_agent,act_spec4coordinate_agent)
random_policies.append(random_policy)


# before interactive learning, let random policy get some experience(iteractive data)

random_collect_data(train_env,random_policies,replay_buffers,RANDOM_COLLECT_STEPS_INITIAL)
collect_data(train_env,collect_policies,replay_buffers,AGENTS_COLLECT_STEPS_INITIAL)




# Agent needs 2 rows of trajectory to get observation and next observation
datasets = [rb.as_dataset(num_parallel_calls=3,sample_batch_size=BATCH_SIZE,num_steps=2).prefetch(3) for rb in replay_buffers]
iterators = [iter(dataset) for dataset in datasets]



e_s_time = time.time()
train_start_time = e_s_time
avg_return = compute_avg_return(train_env, eval_policies, NUM_EVAL_EPISODES)
print('initial return:',avg_return)
returns = [avg_return]
train_losses = [[None,None,None,None]]
iteration_dur_times = [time.time()-e_s_time]

for step in range(1,NUM_ITERATIONS+1):
    
    iteration_start_time = time.time()
    
    # Collect a few steps using collect_policy and save to the replay buffer.
    collect_data(train_env, collect_policies, replay_buffers,AGENTS_COLLECT_STEPS_PER_ITERATION)
    random_collect_data(train_env, random_policies, replay_buffers,RANDOM_COLLECT_STEPS_PER_ITERATION)

    # Sample a batch of data from the buffer and update the agent's network.
    train_loss = []
    for iterator,agent in zip(iterators,agent_system):
        experience, unused_info = next(iterator)
        train_loss.append(agent.train(experience).loss.numpy())
    
    if step % LOG_INTERVAL == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))
        train_losses.append(train_loss)

    if step % EVAL_INTERVAL == 0:
        avg_return = compute_avg_return(train_env, eval_policies, NUM_EVAL_EPISODES)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
    
    iteration_end_time = time.time()
    iteration_dur_times.append(iteration_end_time-iteration_start_time)



date = time.localtime()
addinfo = '{0}-{1}-{2}-{3}-{4}'.format(date.tm_mon,date.tm_mday,date.tm_hour,date.tm_min,date.tm_year)

if not os.path.exists('./result/multi-agent'):
    os.makedirs('./result/multi-agent')

with open('./result/multi-agent/reward_{0}.csv'.format(addinfo),'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step','avg_reward0','avg_reward1','avg_reward2','avg_reward3','avg_reward_coordinate'])
    for i,avg_reward in enumerate(returns):
        res = [i]
        res.extend(avg_reward)
        writer.writerow(res)

with open('./result/multi-agent/trainloss_{0}.csv'.format(addinfo),'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step','train_loss0','train_loss1','train_loss2','train_loss_coordintae'])
    for i,loss in enumerate(train_losses):
        res = [i]
        res.extend(loss)
        writer.writerow(res)

print(time.time()-train_start_time)
