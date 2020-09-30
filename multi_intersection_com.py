# -*- coding: utf-8 -*-

# @Author: KeyangZhang
# @Email: 3238051@qq.com
# @Date: 2020-04-29 09:44:42
# @LastEditTime: 2020-05-03 19:17:05
# @LastEditors: Keyangzhang

from ctmnet.networkcreation import create_intersection
from ctmnet.simulation import Simulator
import numpy as np
import time
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import network
from tf_agents.utils import common as common_utils

INTSEC_ENTRANCES_NAMES = {'N': ('N_entrance_L', 'N_entrance_T', 'N_entrance_R'),
                          'S': ('S_entrance_L', 'S_entrance_T', 'S_entrance_R'),
                          'W': ('W_entrance_L', 'W_entrance_T', 'W_entrance_R'),
                          'E': ('E_entrance_L', 'E_entrance_T', 'E_entrance_R')}

INTSEC_EXITS_NAMES = {'N': 'N_exit',
                      'S': 'S_exit',
                      'W': 'W_exit',
                      'E': 'E_exit'}

INTSEC0_ENTRANCES_PARAS = {'W': {'lane_number': (1, 2, 1),
                                 'lane_length': (0.075, 0.075, 0.075),
                                 'cell_length': (0.01, 0.01, 0.01),
                                 'free_speed': (36, 36, 36)},
                           'E': {'lane_number': (1, 2, 1),
                                 'lane_length': (0.075, 0.075, 0.075),
                                 'cell_length': (0.01, 0.01, 0.01),
                                 'free_speed': (36, 36, 36)},
                           'N': {'lane_number': (1, 1, 1),
                                 'lane_length': (0.075, 0.075, 0.075),
                                 'cell_length': (0.01, 0.01, 0.01),
                                 'free_speed': (36, 36, 36)},
                           'S': {'lane_number': (1, 1, 1),
                                 'lane_length': (0.075, 0.075, 0.075),
                                 'cell_length': (0.01, 0.01, 0.01),
                                 'free_speed': (36, 36, 36)}}

INTSEC0_EXITS_PARAS = {'W': {'lane_number': 4, 'lane_length': 0.05, 'cell_length': 0.025, 'free_speed': 50},
                       'E': {'lane_number': 4, 'lane_length': 0.05, 'cell_length': 0.025, 'free_speed': 50},
                       'N': {'lane_number': 3, 'lane_length': 0.05, 'cell_length': 0.025, 'free_speed': 50},
                       'S': {'lane_number': 3, 'lane_length': 0.05, 'cell_length': 0.025, 'free_speed': 50}}


INTSEC1_ENTRANCES_PARAS = {'W': {'lane_number': (1, 2, 1),
                                 'lane_length': (0.075, 0.075, 0.075),
                                 'cell_length': (0.01, 0.01, 0.01),
                                 'free_speed': (36, 36, 36)},
                           'E': {'lane_number': (1, 2, 1),
                                 'lane_length': (0.075, 0.075, 0.075),
                                 'cell_length': (0.01, 0.01, 0.01),
                                 'free_speed': (36, 36, 36)},
                           'N': {'lane_number': (1, 1, 1),
                                 'lane_length': (0.075, 0.075, 0.075),
                                 'cell_length': (0.01, 0.01, 0.01),
                                 'free_speed': (36, 36, 36)},
                           'S': {'lane_number': (1, 1, 1),
                                 'lane_length': (0.075, 0.075, 0.075),
                                 'cell_length': (0.01, 0.01, 0.01),
                                 'free_speed': (36, 36, 36)}}

INTSEC1_EXITS_PARAS = {'W': {'lane_number': 4, 'lane_length': 0.05, 'cell_length': 0.025, 'free_speed': 50},
                       'E': {'lane_number': 4, 'lane_length': 0.05, 'cell_length': 0.025, 'free_speed': 50},
                       'N': {'lane_number': 3, 'lane_length': 0.05, 'cell_length': 0.025, 'free_speed': 50},
                       'S': {'lane_number': 3, 'lane_length': 0.05, 'cell_length': 0.025, 'free_speed': 50}}


INTSEC2_ENTRANCES_PARAS = {'W': {'lane_number': (1, 2, 1),
                                 'lane_length': (0.075, 0.075, 0.075),
                                 'cell_length': (0.01, 0.01, 0.01),
                                 'free_speed': (36, 36, 36)},
                           'E': {'lane_number': (1, 2, 1),
                                 'lane_length': (0.075, 0.075, 0.075),
                                 'cell_length': (0.01, 0.01, 0.01),
                                 'free_speed': (36, 36, 36)},
                           'N': {'lane_number': (1, 1, 1),
                                 'lane_length': (0.075, 0.075, 0.075),
                                 'cell_length': (0.01, 0.01, 0.01),
                                 'free_speed': (36, 36, 36)},
                           'S': {'lane_number': (1, 1, 1),
                                 'lane_length': (0.075, 0.075, 0.075),
                                 'cell_length': (0.01, 0.01, 0.01),
                                 'free_speed': (36, 36, 36)}}

INTSEC2_EXITS_PARAS = {'W': {'lane_number': 4, 'lane_length': 0.05, 'cell_length': 0.025, 'free_speed': 50},
                       'E': {'lane_number': 4, 'lane_length': 0.05, 'cell_length': 0.025, 'free_speed': 50},
                       'N': {'lane_number': 3, 'lane_length': 0.05, 'cell_length': 0.025, 'free_speed': 50},
                       'S': {'lane_number': 3, 'lane_length': 0.05, 'cell_length': 0.025, 'free_speed': 50}}


INTSEC3_ENTRANCES_PARAS = {'W': {'lane_number': (1, 2, 1),
                                 'lane_length': (0.075, 0.075, 0.075),
                                 'cell_length': (0.01, 0.01, 0.01),
                                 'free_speed': (36, 36, 36)},
                           'E': {'lane_number': (1, 2, 1),
                                 'lane_length': (0.075, 0.075, 0.075),
                                 'cell_length': (0.01, 0.01, 0.01),
                                 'free_speed': (36, 36, 36)},
                           'N': {'lane_number': (1, 1, 1),
                                 'lane_length': (0.075, 0.075, 0.075),
                                 'cell_length': (0.01, 0.01, 0.01),
                                 'free_speed': (36, 36, 36)},
                           'S': {'lane_number': (1, 1, 1),
                                 'lane_length': (0.075, 0.075, 0.075),
                                 'cell_length': (0.01, 0.01, 0.01),
                                 'free_speed': (36, 36, 36)}}

INTSEC3_EXITS_PARAS = {'W': {'lane_number': 4, 'lane_length': 0.05, 'cell_length': 0.025, 'free_speed': 50},
                       'E': {'lane_number': 4, 'lane_length': 0.05, 'cell_length': 0.025, 'free_speed': 50},
                       'N': {'lane_number': 3, 'lane_length': 0.05, 'cell_length': 0.025, 'free_speed': 50},
                       'S': {'lane_number': 3, 'lane_length': 0.05, 'cell_length': 0.025, 'free_speed': 50}}


ARTERIAL_ROADS = {'artery_1': {'lane_number': 4, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60},
                  'artery_2': {'lane_number': 4, 'lane_length': 1.5, 'cell_length': 0.3, 'free_speed': 60},
                  'artery_3': {'lane_number': 4, 'lane_length': 1.5, 'cell_length': 0.3, 'free_speed': 60},
                  'artery_4': {'lane_number': 4, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60},
                  'artery_5': {'lane_number': 4, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60},
                  'artery_6': {'lane_number': 4, 'lane_length': 1.5, 'cell_length': 0.3, 'free_speed': 60},
                  'artery_7': {'lane_number': 4, 'lane_length': 1.5, 'cell_length': 0.3, 'free_speed': 60},
                  'artery_8': {'lane_number': 4, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60}}

LOCAL_ROADS = {'local_1': {'lane_number': 3, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60},
               'local_2': {'lane_number': 3, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60},
               'local_3': {'lane_number': 3, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60},
               'local_4': {'lane_number': 3, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60},
               'local_5': {'lane_number': 3, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60},
               'local_6': {'lane_number': 3, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60},
               'local_7': {'lane_number': 3, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60},
               'local_8': {'lane_number': 3, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60},
               'local_9': {'lane_number': 3, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60},
               'local_10': {'lane_number': 3, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60},
               'local_11': {'lane_number': 3, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60},
               'local_12': {'lane_number': 3, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60}}

CONNECTIONS = [['artery_1', ('0_W_entrance_L', '0_W_entrance_T', '0_W_entrance_R'), (1, 2, 1)],
               ['0_E_exit', 'artery_2', None],
               ['artery_2', ('1_W_entrance_L', '1_W_entrance_T',
                             '1_W_entrance_R'), (0.5, 2, 0.5)],
               ['1_E_exit', 'artery_3', None],
               ['artery_3', ('2_W_entrance_L', '2_W_entrance_T',
                             '2_W_entrance_R'), (0.5, 2, 0.5)],
               ['2_E_exit', 'artery_4', None],
               ['artery_5', ('2_E_entrance_L', '2_E_entrance_T',
                             '2_E_entrance_R'), (1, 2, 1)],
               ['2_W_exit', 'artery_6', None],
               ['artery_6', ('1_E_entrance_L', '1_E_entrance_T',
                             '1_E_entrance_R'), (0.5, 2, 0.5)],
               ['1_W_exit', 'artery_7', None],
               ['artery_7', ('0_E_entrance_L', '0_E_entrance_T',
                             '0_E_entrance_R'), (0.5, 2, 0.5)],
               ['0_W_exit', 'artery_8', None],
               ['local_1', ('0_N_entrance_L', '0_N_entrance_T',
                            '0_N_entrance_R'), (1, 1, 1)],
               ['0_N_exit', 'local_2', None],
               ['local_3', ('1_N_entrance_L', '1_N_entrance_T',
                            '1_N_entrance_R'), (1, 1, 1)],
               ['1_N_exit', 'local_4', None],
               ['local_5', ('2_N_entrance_L', '2_N_entrance_T',
                            '2_N_entrance_R'), (1, 1, 1)],
               ['2_N_exit', 'local_6', None],
               ['local_7', ('2_S_entrance_L', '2_S_entrance_T',
                            '2_S_entrance_R'), (1, 1, 1)],
               ['2_S_exit', 'local_8', None],
               ['local_9', ('1_S_entrance_L', '1_S_entrance_T',
                            '1_S_entrance_R'), (1, 1, 1)],
               ['1_S_exit', 'local_10', None],
               ['local_11', ('0_S_entrance_L', '0_S_entrance_T',
                             '0_S_entrance_R'), (1, 1, 1)],
               ['0_S_exit', 'local_12', None]]

PHASE_PARAS = {'EW_TR': {'start': 0, 'end': 20},
               'EW_L': {'start': 20, 'end': 30},
               'NS_TR': {'start': 30, 'end': 50},
               'NS_L': {'start': 50, 'end': 60}}

LANE_GROUPS_PARAS = {'EW_TR': ('W_T', 'W_R', 'E_T', 'E_R'),
                    'EW_L': ('W_L', 'E_L'),
                    'NS_TR': ('N_T', 'N_R', 'S_T', 'S_R'),
                    'NS_L': ('S_L', 'S_L')}

ARRIVALS_PARAS = {'0_W': {'section_id': 'artery_1', 'volume': 1500,'distribution':'poisson'},
                  '0_N': {'section_id': 'local_1', 'volume': 800,'distribution':'poisson'},
                  '0_S': {'section_id': 'local_11', 'volume': 800,'distribution':'poisson'},
                  '1_N': {'section_id': 'local_3', 'volume': 800,'distribution':'poisson'},
                  '1_S': {'section_id': 'local_9', 'volume': 800,'distribution':'poisson'},
                  '2_N': {'section_id': 'local_5', 'volume': 800,'distribution':'poisson'},
                  '2_S': {'section_id': 'local_7', 'volume': 800,'distribution':'poisson'},
                  '2_E': {'section_id': 'artery_5', 'volume': 1500,'distribution':'poisson'}}

DEPARTURES_PARAS = {'0_W': {'section_id': 'artery_8', 'capacity': 1412*4},
                  '0_N': {'section_id': 'local_2', 'capacity': 1412*3},
                  '0_S': {'section_id': 'local_12', 'capacity': 1412*3},
                  '1_N': {'section_id': 'local_4', 'capacity': 1412*3},
                  '1_S': {'section_id': 'local_10', 'capacity': 1412*3},
                  '2_N': {'section_id': 'local_6', 'capacity': 1412*3},
                  '2_S': {'section_id': 'local_8', 'capacity': 1412*3},
                  '2_E': {'section_id': 'artery_4', 'capacity': 1412*3}}

INTSECTION_NUMS = [0,1,2]

class SimModel(object):
    """"three intersections CTM simulation Model"""

    def __init__(self):
      
        sim = create_intersection(0, INTSEC0_ENTRANCES_PARAS, INTSEC0_EXITS_PARAS,
                  control_cycle=60, phases=PHASE_PARAS,
                  lane_groups=LANE_GROUPS_PARAS)
        sim = create_intersection(1, INTSEC1_ENTRANCES_PARAS, INTSEC1_EXITS_PARAS,
                              control_cycle=60, phases=PHASE_PARAS,
                              lane_groups=LANE_GROUPS_PARAS,simulator=sim)
        sim = create_intersection(2, INTSEC2_ENTRANCES_PARAS, INTSEC2_EXITS_PARAS,
                              control_cycle=60, phases=PHASE_PARAS,
                              lane_groups=LANE_GROUPS_PARAS,simulator=sim)
        for sec_id, argdict in ARTERIAL_ROADS.items():
            sim.create('section',sec_id,argdict)

        for sec_id, argdict in LOCAL_ROADS.items():
            sim.create('section',sec_id,argdict)

        for up,down,p in CONNECTIONS:
            sim.connect(up,down,p)

        for arr_id, argdict in ARRIVALS_PARAS.items():
            sim.create('arrival',arr_id,argdict)

        for dep_id, argdict in DEPARTURES_PARAS.items():
            sim.create('departure',dep_id,argdict)
      
        # 仿真预热600s
        sim.run(600)
        
        self._simulator = sim

        # 相位对应action中元素的编号
        self._phase_code = {0:'EW_TR',1:'EW_L',2:'NS_TR',3:'NS_L'}

        self._intersection_codes = INTSECTION_NUMS

        self._position_name = ['inter0','inter1','inter2','coordinate']

      
    def take_action(self,action):
        """take action and return next state and reward
        
        args:
            action: sequence, in the shape of [[e1,e2,e3,e4],[e1,e2,e3,e4],[e1,e2,e3,e4],[e1,e2,e3]]
        
        return:
            (state,reward): state in the shape of ((8,)-list,(8,)-list,(8,)-list,(12,)-list)
        """

        # 提取action
        coordinate_paras = action['coordinate']
        cycle = int(coordinate_paras[0])
        offsets = coordinate_paras[1:]
        ratios = [action['inter0'],action['inter1'],action['inter2']]

        # 根据action设置周期相位差
        for code in self._intersection_codes:
            sc = self._simulator.get('signalcontroller',code)
            sc.cycle = cycle
            if code != 0:
                sc.offset = int(offsets[code-1])
        
        # 根据action设置各相位时长
        for code in self._intersection_codes:
            sc = self._simulator.get('signalcontroller',code)
            ratio = ratios[code]
            ratio_sum = sum(ratio)
            scaled_ratio = [r/ratio_sum for r in ratio]
            green_time = [int(r*cycle) for r in scaled_ratio[:3]]
            green_time.append(cycle-sum(green_time))
      
            start = 0
            end = 0
            for i,gt in enumerate(green_time):
                end = start+gt
                ph_id = self._phase_code[i]
                sc.set_phase(ph_id,start,end)
                start = end
        
        # 运行一个周期并输出状态和反馈
        sum_delay = [0 for code in self._intersection_codes]
        avg_queue = [[0,0,0,0] for code in self._intersection_codes]
        avg_inflow = [[0,0,0,0] for code in self._intersection_codes]
        for _ in range(cycle):
            self._simulator.run_single_step()
            
            for code in self._intersection_codes:
                sc = self._simulator.get('signalcontroller',code)
                delay = sc.get_delay(method='average')
                queue = sc.get_queue()
                inflow = sc.get_inflow()
            
                sum_delay[code]+= sum(delay.values())/cycle

                for i in range(4):
                    ph_id = self._phase_code[i]
                    avg_queue[code][i]+=queue[ph_id]/cycle
                    avg_inflow[code][i]+=inflow[ph_id]/cycle
        
        state_value = []
        for queue,flow in zip(avg_queue,avg_inflow):
            t = []
            for q in queue:
                t.append(q)
            for f in flow:
                t.append(f)
            state_value.append(np.array(t))
        
        t=[]
        for i in range(3):
            t.extend(avg_queue[i])
        
        state_value.append(np.array(t))
        
        state = {}
        for i,k in enumerate(self._position_name):
            state[k]=state_value[i]
            
        
        reward = [-d for d in sum_delay]
        reward.append(-sum(sum_delay))
        reward = np.array(reward)
        
        # reward = -sum(sum_delay)
        
        return state, reward
    
    def get_state(self):

        cycle = self._simulator.get('signalcontroller',0).cycle
        # 运行一个周期并输出状态
        avg_queue = [[0,0,0,0] for code in self._intersection_codes]
        avg_inflow = [[0,0,0,0] for code in self._intersection_codes]
        
        for _ in range(cycle):
            self._simulator.run_single_step()
            
            for code in self._intersection_codes:
                sc = self._simulator.get('signalcontroller',code)
                queue = sc.get_queue()
                inflow = sc.get_inflow()

                for i in range(4):
                    ph_id = self._phase_code[i]
                    avg_queue[code][i]+=queue[ph_id]/cycle
                    avg_inflow[code][i]+=inflow[ph_id]/cycle
        
        state_value = []
        for queue,flow in zip(avg_queue,avg_inflow):
            t = []
            for q in queue:
                t.append(q)
            for f in flow:
                t.append(f)
            state_value.append(np.array(t))
        
        t=[]
        for i in range(3):
            t.extend(avg_queue[i])
        
        state_value.append(np.array(t))
        
        state = {}
        for i,k in enumerate(self._position_name):
            state[k]=state_value[i]        

        return state

    def reset(self):
        
        self._simulator.clear()
        
        for code in self._intersection_codes:
            sc = self._simulator.get('signalcontroller',code)
            sc.cycle = 60
            for ph_id, ph_para in PHASE_PARAS.items():
                sc.set_phase(ph_id,ph_para['start'],ph_para['end'])

class CTMEnv(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = {'inter0':array_spec.BoundedArraySpec(shape=(4,), dtype=np.float32, minimum=1, maximum=10, name='action_0'),
                             'inter1':array_spec.BoundedArraySpec(shape=(4,), dtype=np.float32, minimum=1, maximum=10, name='action_1'),
                             'inter2':array_spec.BoundedArraySpec(shape=(4,), dtype=np.float32, minimum=1, maximum=10, name='action_2'),
                             'coordinate':array_spec.BoundedArraySpec(shape=(3,), dtype=np.float32, minimum=40, maximum=80, name='action_3')}
        self._observation_spec = {'inter0':array_spec.BoundedArraySpec(shape=(8,), dtype=np.float32, minimum=0, name='observation_0'),
                                  'inter1':array_spec.BoundedArraySpec(shape=(8,), dtype=np.float32, minimum=0, name='observation_1'),
                                  'inter2':array_spec.BoundedArraySpec(shape=(8,), dtype=np.float32, minimum=0, name='observation_2'),
                                  'coordinate':array_spec.BoundedArraySpec(shape=(12,), dtype=np.float32, minimum=0, name='observation_3')}
        
        self._simmodel = SimModel()
        self._state = self._simmodel.get_state()
        self._episode_ended = False
        self._total_step = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._simmodel.reset()
        self._state = self._simmodel.get_state()
        self._episode_ended = False
        self._total_step = 0
        return ts.restart(self._state)

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        if self._total_step >= 30:
            self._episode_ended = True
            self._state, reward = self._simmodel.take_action(action)
            self._total_step+=1
        else :
            self._state, reward = self._simmodel.take_action(action)
            self._total_step+=1

        

        if self._episode_ended:
            reward = tf.convert_to_tensor(reward,dtype='float32')
            discount = tf.convert_to_tensor(0,dtype='float32')
            return ts.TimeStep(np.asarray([2,2,2,2], dtype=np.int32),reward=reward,discount=discount,observation=self._state)
            # return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward, discount=1.0)


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
        
        self._preprocess_observation_layer = tf.keras.layers.Dense(4, activation=activation_fn,autocast=False)
        self._preprocess_action_layer = tf.keras.layers.Dense(5, activation=activation_fn,autocast=False)
        self._combine_layer = tf.keras.layers.Concatenate(axis=-1)

        self._dense1 = tf.keras.layers.Dense(units=10,
                                             kernel_initializer=initializer, 
                                             activation=activation_fn,autocast=False)
        self._dense2 = tf.keras.layers.Dense(units=10,
                                             kernel_initializer=initializer, 
                                             activation=activation_fn,autocast=False)
        
        self._q_value_projection_layer = tf.keras.layers.Dense(
            1, activation=tf.keras.activations.sigmoid,autocast=False)

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