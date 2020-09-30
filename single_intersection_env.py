# -*- coding: utf-8 -*-

# @Author: KeyangZhang
# @Email: 3238051@qq.com
# @Date: 2020-04-13 09:56:55
# @LastEditTime: 2020-04-30 09:26:57
# @LastEditors: Keyangzhang

from ctmnet.networkcreation import create_intersection
from ctmnet.simulation import Simulator

import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

INTSEC0_ENTRANCES_NAMES = {'N': ('0_N_entrance_L', '0_N_entrance_T', '0_N_entrance_R'),
                           'S': ('0_S_entrance_L', '0_S_entrance_T', '0_S_entrance_R'),
                           'W': ('0_W_entrance_L', '0_W_entrance_T', '0_W_entrance_R'),
                           'E': ('0_E_entrance_L', '0_E_entrance_T', '0_E_entrance_R')}

INTSEC0_EXITS_NAMES = {'N': '0_N_exit',
                       'S': '0_S_exit',
                       'W': '0_W_exit',
                       'E': '0_E_exit'}

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

MAINROADS_PARAS = {'W_up': {'lane_number': 4, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60},
                   'W_down': {'lane_number': 4, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60},
                   'E_up': {'lane_number': 4, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60},
                   'E_down': {'lane_number': 4, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60},
                   'N_up': {'lane_number': 3, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60},
                   'N_down': {'lane_number': 3, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60},
                   'S_up': {'lane_number': 3, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60},
                   'S_down': {'lane_number': 3, 'lane_length': 1, 'cell_length': 0.2, 'free_speed': 60}}

CONNECTION_MAP = {'N_up': ('0_N_entrance_L', '0_N_entrance_T', '0_N_entrance_R'),
                  'S_up': ('0_S_entrance_L', '0_S_entrance_T', '0_S_entrance_R'),
                  'W_up': ('0_W_entrance_L', '0_W_entrance_T', '0_W_entrance_R'),
                  'E_up': ('0_E_entrance_L', '0_E_entrance_T', '0_E_entrance_R'),
                  '0_N_exit':'N_down',
                  '0_S_exit':'S_down',
                  '0_W_exit':'W_down',
                  '0_E_exit':'E_down'}

FLOWS_RATIOS = ((1,1,1),(1,1,1),(1,3,1),(1,3,1),None,None,None,None)

SIGNALCONTROLLER_PARAS = {0:{'cycle':60,'offset':0}}

PHSAES_PARAS = {'NS_TR': {'signalcontroller_id': 0, 'start': 0, 'end': 20},
                'NS_L': {'signalcontroller_id': 0, 'start': 20, 'end': 30},
                'EW_TR': {'signalcontroller_id': 0, 'start': 30, 'end': 50},
                'EW_L': {'signalcontroller_id': 0, 'start': 50, 'end': 60}}

LAMPS_PARAS = {'N_L': {'signalcontroller_id': 0, 'phase_id': 'NS_L', 'section_id': '0_N_entrance_L'},
               'N_T': {'signalcontroller_id': 0, 'phase_id': 'NS_TR', 'section_id': '0_N_entrance_T'},
               'N_R': {'signalcontroller_id': 0, 'phase_id': 'NS_TR', 'section_id': '0_N_entrance_R'},
               'S_L': {'signalcontroller_id': 0, 'phase_id': 'NS_L', 'section_id': '0_S_entrance_L'},
               'S_T': {'signalcontroller_id': 0, 'phase_id': 'NS_TR', 'section_id': '0_S_entrance_T'},
               'S_R': {'signalcontroller_id': 0, 'phase_id': 'NS_TR', 'section_id': '0_S_entrance_R'},
               'W_L': {'signalcontroller_id': 0, 'phase_id': 'EW_L', 'section_id': '0_W_entrance_L'},
               'W_T': {'signalcontroller_id': 0, 'phase_id': 'EW_TR', 'section_id': '0_W_entrance_T'},
               'W_R': {'signalcontroller_id': 0, 'phase_id': 'EW_TR', 'section_id': '0_W_entrance_R'},
               'E_L': {'signalcontroller_id': 0, 'phase_id': 'EW_L', 'section_id': '0_E_entrance_L'},
               'E_T': {'signalcontroller_id': 0, 'phase_id': 'EW_TR', 'section_id': '0_E_entrance_T'},
               'E_R': {'signalcontroller_id': 0, 'phase_id': 'EW_TR', 'section_id': '0_E_entrance_R'}}

ARRIVALS_PARAS = {'N_up': {'section_id': 'N_up', 'volume': 800,'distribution':'poisson'},
                  'S_up': {'section_id': 'S_up', 'volume': 800,'distribution':'poisson'},
                  'W_up': {'section_id': 'W_up', 'volume': 1500,'distribution':'poisson'},
                  'E_up': {'section_id': 'E_up', 'volume': 1500,'distribution':'poisson'}}

DEPARTURES_PARAS = {'N_down': {'section_id': 'N_down', 'capacity': 5000},
                  'S_down': {'section_id': 'S_down', 'capacity': 5000},
                  'W_down': {'section_id': 'W_down', 'capacity': 5000},
                  'E_down': {'section_id': 'E_down', 'capacity': 5000}}



class SimModel(object):
    """"a singal intersection CTM simulation Model"""
    
    def __init__(self):
        sim = create_intersection(0,INTSEC0_ENTRANCES_PARAS,INTSEC0_EXITS_PARAS)

        # create sections
        for sec_id, sec_paras in MAINROADS_PARAS.items():
            sim.create('section',sec_id,sec_paras)

        # connect sections
        for (upstream, downstream),ratio in zip(CONNECTION_MAP.items(),FLOWS_RATIOS) :
            sim.connect(upstream,downstream,ratio)

        # set signal control
        sc0 = sim.create('signalcontroller',0,{'cycle':60,'offset':0})

        for ph_id, ph_paras in PHSAES_PARAS.items():
            sim.create('phase',ph_id,ph_paras)

        for lamp_id, lamp_paras in LAMPS_PARAS.items():
            sim.create('lamp',lamp_id,lamp_paras)

        # set demand
        for arr_id, arr_paras in ARRIVALS_PARAS.items():
            sim.create('arrival',arr_id,arr_paras)

        for dep_id, dep_paras in DEPARTURES_PARAS.items():
            sim.create('departure',dep_id,dep_paras)
        
        # 仿真预热10分钟
        sim.run(600)
        
        self.simulator = sim
        self.signalcontroller = sc0
        self.phase_code = {0:'EW_TR',1:'EW_L',2:'NS_TR',3:'NS_L'}
        self.total_reward = 0
        self.total_delay = 0
        self.total_simtime = 0
    
    def take_action(self, action):
        """
        
        args:
            action: sequence, shape is (5,),the first four elemnts represent green raio for each phase. 
                    the last element is new cycle
        return
        (state, reward): state is sequence of shape-(4,0), each element is max queue, inflow for each pahse
                         reward is a scaler, negative value of average delay.
        """
        ratios = [action[i] for i in range(4)]
        ratio_sum = sum(ratios)
        green_ratios = [r/ratio_sum for r in ratios]
        
        new_cycle = int(action[-1])
        
        new_cycle = new_cycle*4+40
        
        green_times = [int(green_ratios[i]*new_cycle+0.5) for i in range(3)]
        sum_first3 = sum(green_times)
        green_times.append(new_cycle-sum_first3)

        self.signalcontroller.cycle = new_cycle

        start = 0
        end = 0
        for i,gt in enumerate(green_times):
            end = start+gt
            ph_id = self.phase_code[i]
            self.signalcontroller.set_phase(ph_id,start,end)
            start = end
        
        # 计算周期内每秒的平均延误作为reward
        # 计算周期内各相位每秒平均最大排队长度、各相位平均流入作为state
        sum_delay = 0
        sum_queue = {value:0 for value in self.phase_code.values()}
        sum_inflow = {value:0 for value in self.phase_code.values()}

        for _ in range(new_cycle):
            self.simulator.run_single_step()
            
            delays = self.signalcontroller.get_delay(method='average')
            sum_delay+= sum(delays.values())

            queues = self.signalcontroller.get_queue()
            for ph_id in queues:
                sum_queue[ph_id]+=queues[ph_id]
            
            inflows = self.signalcontroller.get_inflow()
            for ph_id in inflows:
                sum_inflow[ph_id]+= inflows[ph_id]

        reward = -sum_delay/new_cycle
        
        self.total_reward+=reward
        self.total_delay+=sum_delay
        self.total_simtime+=new_cycle

        state = []
        for i in range(4):
            ph_id = self.phase_code[i]
            state.append(sum_queue[ph_id]/new_cycle)
            state.append(10*sum_inflow[ph_id]/new_cycle)
        
        return (state,reward)
    
    def reset(self):
        
        self.simulator.clear()
        
        # initial control parameters
        self.signalcontroller.cycle = 60
        for ph_id, ph_paras in PHSAES_PARAS.items():
            self.signalcontroller.set_phase(ph_id,ph_paras['start'],ph_paras['end'])
        
        self.simulator.run(600)
    
    def get_state(self):

        queues = self.signalcontroller.get_queue()
        inflows = self.signalcontroller.get_inflow()
        state = []
        for i in range(4):
            ph_id = self.phase_code[i]
            state.append(queues[ph_id])
            state.append(10*inflows[ph_id])
        
        return state


        

class CTMEnv(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(5,), dtype=np.float32, minimum=[1,1,1,1,0], maximum=[10,10,10,10,10], name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(8,), dtype=np.float32, minimum=0, name='observation')
        
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
        self._total_step =0
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        if self._total_step >= 30:
            self._episode_ended = True
            self._state, reward = self._simmodel.take_action(action)
        else :
            self._state, reward = self._simmodel.take_action(action)

        self._total_step+=1

        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype=np.float32), reward)
        else:
            return ts.transition(
                np.array(self._state, dtype=np.float32), reward, discount=1.0)

