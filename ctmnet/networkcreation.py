# -*- coding: utf-8 -*-

# @Author: KeyangZhang
# @Email: 3238051@qq.com
# @Date: 2019-12-26 15:55:54
# @LastEditTime: 2020-04-29 18:10:01
# @LastEditors: Keyangzhang

from .simulation import Simulator
import xml.etree.ElementTree as ET


def create_intersection(intersection_num, entrances, exits, control_cycle=None, phases=None, lane_groups=None, simulator=None):
    """"create an intersection and add signal control
    
    args:
    entrances: dict, provide the parameters of entrances section in a form of 
        {direction:{para_name:(para_value for Left, para_value for through,para_value for right),...},...}
        if it does not have some branch or entance, one should use None to fill in the corresponding position. 
        it may look like {'N':None,'S':{para_name:(None,para_value for through,None),...},...}
    exits: dict, provide the parameters of exits section in a form of
        {direction:{para_name:para_value,...},...} e.g. {'N':{'lane_length':1,'cell_length':0.2},'S':{...},...}
    control_cycle: int
    phases: dict, provide the parameters of phases in a form of 
        {phase_id:{'start':value,'end':value},...}
    lane_group: dict, {phase_id:(section_position,...),...} e.g. {0:{'N_T','N_R','S_T','S_R'}}

    """

    if simulator is None:
        simulator = Simulator()
        
    for direct in entrances:
        paras_dict =  entrances[direct]
        if paras_dict is not None:
            for i, turn in enumerate(['L','T','R']):
                # LTR分别对应左转直行右转
                sec_paras = {}
                for para_name in paras_dict:
                    sec_paras[para_name] = paras_dict[para_name][i]
                if None in sec_paras.values():
                    break
                else:
                    sec_id = '{0}_{1}_entrance_{2}'.format(intersection_num,direct,turn)
                    sec_paras['name']=sec_id
                    simulator.create('section',sec_id,sec_paras)
    
    for direct in exits :
        sec_paras = exits[direct]
        if sec_paras is not None :
            sec_id = '{0}_{1}_exit'.format(intersection_num,direct)
            sec_paras['name']=sec_id
            simulator.create('section',sec_id,sec_paras)
    

    connect_map = {'N_exit': ('W_entrance_L', 'S_entrance_T', 'E_entrance_R'),
                   'W_exit': ('S_entrance_L', 'E_entrance_T', 'N_entrance_R'),
                   'S_exit': ('E_entrance_L', 'N_entrance_T', 'W_entrance_R'),
                   'E_exit': ('N_entrance_L', 'W_entrance_T', 'S_entrance_R')
                   }
    new_connect_map = {}
    
    for key, value in connect_map.items():
        new_v = ['{0}_{1}'.format(intersection_num,element) for element in value]
        new_k = '{0}_{1}'.format(intersection_num,key)
        new_connect_map[new_k] = new_v
    
    for downstream_id, upstream_ids in new_connect_map.items():
        if simulator.get('section',downstream_id):
            connected_upstream_ids = []
            for upstream_id in upstream_ids:
                if simulator.get('section',upstream_id):
                    connected_upstream_ids.append(upstream_id)
            simulator.connect(connected_upstream_ids,downstream_id)
    

    # add signal control
    if control_cycle is not None:
        simulator.create('signalcontroller',intersection_num,{'cycle':control_cycle})
    
    if phases is not None:
        for ph_id,ph_para in phases.items():
            ph_para['signalcontroller_id']=intersection_num
            simulator.create('phase',ph_id,ph_para)
    
    if lane_groups is not None:
        for ph_id,group in lane_groups.items():
            for lamp_id in group:
                argdict = {}
                direct,turn = lamp_id.split('_')
                argdict['signalcontroller_id']=intersection_num
                argdict['section_id'] = '{0}_{1}_entrance_{2}'.format(intersection_num,direct,turn)
                argdict['phase_id']=ph_id
                simulator.create('lamp',lamp_id,argdict)
    
    return simulator




def create_from_xml(netfile=None,sigfile=None,simulator=None):

    if simulator is None:
        simulator = Simulator()

    # create section and connection
    if netfile is not None:
        tree = ET.parse(netfile)
        root = tree.getroot()
        paras = ['lane_length','lane_number',
            'cell_length','free_speed',
            'wave_speed','jam_density']
        for sec in root.iter('section'):
            sec_id = sec.attrib['id']
            argdict = {para:eval(sec.find(para).text) for para in paras}
            simulator.create('section',sec_id, argdict)

        for cnc in root.iter('connector'):
            downstream = eval(cnc.find('downstream').text)
            upstream = eval(cnc.find('upstream').text)
            if isinstance(upstream,list):
                downstream = str(downstream)
                priority = eval(cnc.find('priority').text)

                # 将来把priority标签里面的存储格式改为[num1,num2,...],下面这句就可以删掉
                priority = list(map(eval,priority))

                simulator.connect(upstream,downstream,priority)
                     
            elif isinstance(downstream,list):
                upstream = str(upstream)
                priority = eval(cnc.find('priority').text)

                # 将来把priority标签里面的存储格式改为[num1,num2,...],下面这句就可以删掉
                priority = list(map(eval,priority))

                simulator.connect(upstream,downstream,priority)
            
            else:
                upstream = str(upstream)
                downstream = str(downstream)
                simulator.connect(upstream,downstream)

    # config signal control
    if sigfile is not None:
        tree = ET.parse(sigfile)
        root = tree.getroot()

        # create signal controllers
        paras = ['cycle','offset']
        for sc in root.iter('signalcontroller'):
            sc_id = sc.attrib['id']
            argdict = {para:eval(sc.find(para).text) for para in paras}
            simulator.create('signalController',sc_id,argdict)
        
        # create phase
        paras = ['green_start','green_end']
        for ph in root.iter('phase'):
            ph_id = ph.attrib['id']
            argdict = {para:eval(ph.find(para).text) for para in paras}
            argdict['signalcontroller_id']=ph.find('signalcontroller_id').text
            simulator.create('phase',ph_id,argdict)
        
        # create lamp
        paras = ['signalcontroller_id','phase_id','lamp_id']
        for lamp in root.iter('lamp'):
            lamp_id = lamp.attrib['id']
            argdict = {para:ph.find(para).text for para in paras}
            simulator.create('lamp',lamp_id,argdict)

    return simulator
