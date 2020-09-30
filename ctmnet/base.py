# -*- coding: utf-8 -*-

# @Author: KeyangZhang
# @Date: 2019-12-04 17:28:17
# @LastEditTime: 2020-04-29 18:10:26
# @LastEditors: Keyangzhang

import warnings

def _mid(a,b,c):
    """return the middle value of three numbers"""
    if a<=b<=c or c<=b<=a:
        return b
    elif b<=a<=c or c<=a<=b:
        return a
    elif a<=c<=b or b<=c<=a:
        return c

class _Cell(object):
    """Cell
    
    cell is the basic element of section 
    
    """

    def __init__(self, demand=0, supply=0):
        super(_Cell, self).__init__()
        self.volume = 0
        self.demand = 0
        self.supply = 0

class Section(object):
    """road section

    Section is the basic element of the network and it can be signle-lane or multi-lanes
    It can be an artery, a link road, an entrance or an export road of an intersection
    Once the road is initialized the arritubes cannot be change
    

    Atrributes:
        lane_length: float, the length of the section, the unit is km
        lane_number: int, the number of lanes this section contains, default: 1
        cell_length: float, the length of all the cells in this section, default: 0.05km
        free_speed: float, the free speed of vehicles in this section, default: 60km/h
        wave_speed: float, the wave_speed of this section, default：8km/h
                    for its definition, please refer to fundamental diagram thoery of traffic flow
        jam_density: float, the jam_density of this section, default: 200veh/(km*lane) 
                    for its definition, please refer to traffic flow theory

    """

    def __init__(self, lane_length, lane_number=1,
                 cell_length=0.05, free_speed=60, wave_speed=8,
                 jam_density=200, name='empty'):
        super(Section, self).__init__()

        # constant parameters
        self.lane_length = lane_length
        self.lane_number = lane_number
        self.cell_length = cell_length
        self.cells_number = int(lane_length/cell_length+0.5)  # the number of cell this section contains
        if self.cells_number<=1:
            raise ValueError('the number of cells should not be smaller than 2')
        self.free_speed = free_speed
        self.wave_speed = wave_speed
        self.name = name

        #   the actual jam density of this secthin
        self.jam_density = jam_density * lane_number

        #   the capacity of each cell
        #   the upper limit value of flow between two direct connected cells
        self.capacity = self.jam_density / \
            (1 / self.free_speed + 1 / self.wave_speed)
        
        #   the initial time interval of each simulation step
        self.interval = self.cell_length * 3600 / self.free_speed
        if self.interval < 1:
            raise ValueError('You should reset cell length and free speed to make the initial simulation interval be more than 1 second')
        
        #   the maximum number of vehicles that flow form cell i-1 to cell i during sim interval
        self.max_volume = self.capacity * self.interval / 3600

        # dynamic parameters
        self.cells = [_Cell() for i in range(self.cells_number)]

        self.flows = [0 for i in range(self.cells_number-1)]
        
        #   demand of this section, equals to the demand of last cell
        self.demand = 0
        #   supply of this section, equals to the supply of first cell
        self.supply = min(self.max_volume,
                          self.wave_speed / self.free_speed *(self.cell_length * self.jam_density -0)
                          ) / self.interval
        #   inflow of this section per simsecond
        self.inflow = 0
        #   outflow of this section per simsecond
        self.outflow = 0

    def __getitem__(self, position):
        return self.cells[position]

    def __str__(self):
        return str([round(cell.volume,2) for cell in self])

    def calculate_demand(self):
        """calculate demand of each cell and this section"""
        
        for cell in self.cells:
            cell.demand = min(cell.volume, self.max_volume) /self.interval
        self.demand = self.cells[-1].demand

    def calculate_supply(self):
        """calculate supply of each cell and this section"""
        
        for cell in self.cells:
            cell.supply = min(self.max_volume,
                              self.wave_speed / self.free_speed *
                              (self.cell_length * self.jam_density -
                               cell.volume)) /self.interval
        self.supply = self.cells[0].supply

    def calculate_flow(self):
        """calculate flows between each two cells"""
        
        for i in range(0, self.cells_number-1):
            self.flows[i] = min(self.cells[i].demand, self.cells[i+1].supply)

    def update_volume(self):
        """update the volume of each cell"""

        # for the first cell
        self.cells[0].volume = self.cells[0].volume + \
            self.inflow - self.flows[0]
        # for the intermediate cells
        for i in range(1, self.cells_number-1):
            self.cells[i].volume = self.cells[i].volume + \
                self.flows[i-1]-self.flows[i]
        # for the last cells
        self.cells[-1].volume = self.cells[-1].volume + \
            self.flows[-1] - self.outflow
    
    def last_sim_step_volume(self):
        """计算上一个仿真步长结束时的元胞内车辆数，同时也是当前仿真步长开始时的元胞内车辆数"""

        vols = [0 for i in range(self.cells_number)]

        vols[0] = self.cells[0].volume + self.flows[0] - self.inflow

        for i in range(1,self.cells_number-1):
            vols[i] = self.cells[i].volume + self.flows[i] - self.flows[i-1]
        
        vols[-1] = self.cells[-1].volume + self.outflow - self.flows[-1]

        return vols


    
    def velocity(self,level='cell'):
        """get the operation velocity(km/h) of each cell/section at just ended simluation step"""

        #  每个section中总是储存t+1时刻的volume，t到t+1的flow，即一个仿真步长（step）过程中的流量和仿真步长结束时的元胞中车辆数
        #  但计算速度需要用到仿真步长开始时的元胞密度，因此要对应时刻的元胞中车辆数vol_t = Vol_t+1 + outflow_t - inflow_t     
        vels = []
        vols = self.last_sim_step_volume()
        
        if level=='cell':
            # 计算第一个元胞
            vol = vols[0]
            outflow = self.flows[0]
            if vol == 0 :
                vels.append(0)
            else :
                vel = outflow*3600/(vol/self.cell_length)
                vels.append(round(vel,2))
            
            # 计算中间元胞
            for i in range(1,self.cells_number-1):
                vol = vols[i]
                outflow = self.flows[i]
                if vol == 0 :
                    vels.append(0)
                else:
                    vel = outflow*3600/(vol/self.cell_length)
                    vels.append(round(vel,2))

            # 计算最后一个元胞
            vol = vols[-1]
            outflow = self.outflow
            if vol==0:
                vels.append(0)
            else:
                vel = outflow*3600/(vol/self.cell_length)
                vels.append(round(vel,2))
            
            return vels
        
        elif level=='section':            
            # 先计算每一个元胞的再按照volume计算加权平均
            
            # 计算第一个元胞
            vol = vols[0]
            outflow = self.flows[0]
            if vol == 0 :
                vels.append(0)
            else :
                vel = outflow*3600/(vol/self.cell_length)
                vels.append(round(vel,2))
            
            # 计算中间元胞
            for i in range(1,self.cells_number-1):
                vol = vols[i]
                outflow = self.flows[i]
                if vol == 0 :
                    vels.append(0)
                else:
                    vel = outflow*3600/(vol/self.cell_length)
                    vels.append(round(vel,2))

            # 计算最后一个元胞
            vol = vols[-1]
            outflow = self.outflow
            if vol==0:
                vels.append(0)
            else:
                vel = outflow*3600/(vol/self.cell_length)
                vels.append(round(vel,2))           

            
            # 将速度按照volume加权平均
            weighted_vels = [vel*vol for vel, vol in zip(vels,vols)]
            sum_vol = sum(vols)
            if sum_vol == 0:
                avg_vel = 0
            else:
                avg_vel = round(sum(weighted_vels)/sum_vol,2)
            
            return avg_vel


        else :
            raise ValueError('no such level for collecting data')
    
    def queue(self,level='section', speed_threshold=8):
        """get number of vehicles in queue for current section at just ended simulation step     
        """

        queues = [0 for i in range(self.cells_number)]
        vels = self.velocity(level='cell')
        vols = self.last_sim_step_volume()
        
        for i, (vel, vol) in enumerate(zip(vels, vols)):
            if vel <= speed_threshold and vol != 0:
                queues[i] = vol
        
        if level == 'cell':
            return queues
        elif level == 'section':
            return sum(queues)
        else:
            raise ValueError('no such a level for data collection:{0}'.format(level))
    
    def delay(self,level='section'):
        """get delay of cells/section in the just eded simulation step"""

        delays = [0 for i in range(self.cells_number)]
        vels = self.velocity(level='cell')
        vols = self.last_sim_step_volume()

        for i, (vel,vol) in enumerate(zip(vels,vols)):
            # 每一个仿真步长的时间是1s
            # 计算一秒内实际走过的路程，并计算此路程在自由流状态下的行驶时间
            # 此时间和1s的差值乘以元胞内的车辆数，即为元胞的延误
            delays[i] = (1 - 1*vel/self.free_speed)*vol
        
        if level == 'cell':
            return delays
        elif level == 'section':
            return sum(delays)
        else:
            raise ValueError('no such a level for data collection:{0}'.format(level))

    def clear(self):
        
        for cell in self.cells:
            cell.volume = 0
        
        


class SignalController(object):
    """SignalController
    
    this class resembles the signal controller at the intersection, 
    which controls the signal light status of each entrance

    Attributes:
        cycle: the cycle of the signal plan applied by this signal controller, the unit is second
               for its definition, please refer to the traffic signal control theory
        offset: the offset of the signal plan applied by this signal controller, the unit is second
                for its definition, please refer to the traffic signal control theory

    """
    
    def __init__(self, cycle, offset=0):
        super(SignalController, self).__init__()
        self.lamps = {}  # {lamp_id:(sec_id,pahse_id),...}
        self.phases = {}  # {phase_id:phase,...}
        self.lane_groups = {} # {phase_id:[sec1,sec2,...],...}
        self.cycle = cycle
        self.offset = offset
    
    def create_phase(self,phase_id,start,end):
        if start>end :
            raise ValueError('start should be smaller than end')
        
        elif start>=self.cycle or end>self.cycle :
            text = """start or end exceeds the range of cycle and has taken the remainder of cycle. 
                      inital green start:{0}, now:{1}; inital green end:{2}, now:{3},cycle:{4}""".format(start,start % self.cycle,end,end % self.cycle,self.cycle)
            warnings.warn(text)
            start = start % self.cycle
            end  = end % self.cycle
            phase = Phase(start,end)
            self.phases[phase_id]= phase
            self.lane_groups[phase_id] = [] # initial a lane group
            return phase
        
        else :
            phase = Phase(start,end)
            self.phases[phase_id]=phase
            self.lane_groups[phase_id] = [] # initial a lane group
            return phase

    def set_phase(self, phase_id, start, end):
        ph = self.phases[phase_id]
        
        if start>end :
            raise ValueError('start should be smaller than end [start:{0},end:{1}]'.format(start,end))
        
        if start>self.cycle or end>self.cycle:
            text = """start or end exceeds the range of cycle and has taken the remainder of cycle. 
                      inital green start:{0}, now:{1}; inital green end:{2}, now:{3},cycle:{4}""".format(start,start % self.cycle,end,end % self.cycle,self.cycle)
            warnings.warn(text)
            start = start % self.cycle
            end  = end % self.cycle
        
        ph.start = start 
        ph.end = end

    def create_lamp(self,lamp_id,section_id, section, phase_id):
        """add a lamp on a section"""
        
        self.lamps[lamp_id]=(section_id,phase_id)

        self.lane_groups[phase_id].append(section)
        
        return (section_id,phase_id)


    def update_signal(self,current_time):
        """update the status of light controled by this signal controller"""
        time = (current_time+self.offset)%self.cycle
        
        for ph_id,group in self.lane_groups.items():
            
            ph = self.phases[ph_id]
            
            if not (ph.start<=time<ph.end):
                # when the light is red, the section cannot generate demand
                for sec in group:
                    sec.demand=0
    
    def get_queue(self):
        """获取不同相位下最大排队长度"""
        result = {}
        for ph_id, group in self.lane_groups.items():
            ques = [sec.queue(level='section') for sec in group]
            result[ph_id] = max(ques)
        return result
    
    def get_delay(self, method='average'):
        """获取各相位的延误"""
        result = {}
        if method == 'sum':
            for ph_id, group in self.lane_groups.items():
                delays = [sec.delay(level='section') for sec in group]
                result[ph_id] = sum(delays)
        
        elif method == 'average':
            # 统计单车延误
            for ph_id, group in self.lane_groups.items():
                average_delay = 0
                for sec in group:
                    vol = sum(sec.last_sim_step_volume())
                    if vol !=0 :
                        average_delay+=sec.delay(level='section')/vol 
                result[ph_id] = average_delay
        
        return result

    def get_inflow(self):
        """获取各相位的流入车辆数
        return：
        result：dict,{ph_id:inflow}
        """
        result= {}
        for ph_id, group in self.lane_groups.items():
            inflows = [sec.inflow for sec in group]
            result[ph_id] = sum(inflows)
        return result

class Phase(object):
    """Phase

    A phase is a specific movement that has a unique signal indication
    different intersections can have the same phase combination

    Attributes:
        start: the start time of green status of a lamp in the cycle, the unit is second
        end: the end time of green status of a lamp in the cycle, the unit is second

    """
    def __init__(self, start, end):
        super(Phase, self).__init__()
        self.start = start
        self.end = end
    

class _Connector(object):
    """Connector

    the connector connects the upstream sections and downstream sections
    and it does not have volume and is used merely for calcuate flows between sections

    Attributes:
        upstream: list/tuple or Section, the upstream section/sections
        downstream: list/tuple or Section, the downstream section/sections
        connect_type: string, there are following three types of connection
                      confluent: the upstream has mutilple sections (<=3) while the downstream only has one
                      split: the upstream has one section while the upstream has severals (<=3)
                      straight: the upstream and downstream both only have one section
        

    """

    def __init__(self, upstream, downstream,priority=None):
        super(_Connector, self).__init__()
        self.upstream = upstream
        self.downstream = downstream
        
        if isinstance(upstream,(tuple,list)) and isinstance(downstream,Section) :
            self.upnum = len(upstream)
            self.downnum = 1
            self.connect_type='confluent'
            self._func=self._confluent
            
            if priority is not None:
                s = sum(priority)
                self.priority = [p/s for p in priority]  # standardize the priority
            else:
                self.priority = [1/self.upnum for i in range(self.upnum)]
        
        elif isinstance(upstream,Section) and isinstance(downstream,(tuple,list)):
            self.upnum=1
            self.downnum=len(downstream)
            self.connect_type = 'split'
            self._func=self._split
            
            if priority is not None:
                s = sum(priority)
                self.priority = [p/s for p in priority]  # standardize the priority
            else:
                self.priority = [1/self.downnum for i in range(self.downnum)]
        
        elif isinstance(upstream,Section) and isinstance(downstream,Section) :
            self.upnum=1
            self.downnum=1
            self.connect_type = 'straight'
            self._func = self._straight
        
        else:
            raise ValueError('cannot connect these two objects')


    def _straight(self):
        """the calculate method for staright connection"""
        
        flow = min(self.upstream.demand, self.downstream.supply)
        self.upstream.outflow = flow
        self.downstream.inflow = flow
    
    def _split(self):
        """the calculate method for split connection"""
        
        temp = [self.upstream.demand]
        for item, p in zip(self.downstream, self.priority):
            temp.append(item.supply/p)
        
        flow = min(temp) # total flow
        
        self.upstream.outflow = flow
        
        for item, p in zip(self.downstream, self.priority):
            item.inflow = p * flow

    def _confluent(self):
        """the calculate method for confluent connection"""
        
        # self.upnum == 2 means that this is a conflunce section
        if self.upnum == 2 :
            u1, u2 = self.upstream
            p1, p2 = self.priority
            supply = self.downstream.supply
            if (u1.demand+u2.demand) <= supply:
                u1.outflow = u1.demand
                u2.outflow = u2.demand
                self.downstream.inflow = (u1.demand+u2.demand)
            else:
                flow1 = _mid(u1.demand, supply-u2.demand, p1*supply)
                flow2 = _mid(u2.demand, supply-u1.demand, p2*supply)
                u1.outflow = flow1
                u2.outflow = flow2
                self.downstream.inflow = (flow1+flow2)
        
        # self.upnum == 3 means that this connector is in the intersection
        elif self.upnum == 3:
            
            # use "rec" to record entrance that is not in red (demand is not 0)
            rec = [] 
            for i,item in enumerate(self.upstream):
                if item.demand == 0:
                    item.outflow=0
                else:
                    rec.append(i)
            
            if len(rec) == 2 :
                u1 = self.upstream[rec[0]]
                p1 = self.priority[rec[0]]
                u2 = self.upstream[rec[1]]
                p2 = self.priority[rec[1]]
                
                # adjust p1 and p2 to satisfy the condition sum(priorities)=1
                s = p1 + p2
                p1 = p1/s
                p2 = p2/s
                
                supply = self.downstream.supply
                if (u1.demand+u2.demand) <= supply:
                    u1.outflow = u1.demand
                    u2.outflow = u2.demand
                    self.downstream.inflow = (u1.demand+u2.demand)
                else:
                    flow1 = _mid(u1.demand, supply-u2.demand, p1*supply)
                    flow2 = _mid(u2.demand, supply-u1.demand, p2*supply)
                    u1.outflow = flow1
                    u2.outflow = flow2
                    self.downstream.inflow = (flow1+flow2)
            elif len(rec) == 1:
                u = self.upstream[rec[0]]
                flow = min(u.demand, self.downstream.supply)
                u.outflow = flow
                self.downstream.inflow = flow
            else:
                # that means len(rec)==3 
                sumdemand = sum([item.demand for item in self.upstream])
                if sumdemand <= self.downstream.supply:
                    for item in self.upstream:
                        item.outflow = item.demand
                    self.downstream.inflow = sumdemand
                else:
                    flows = [p*self.downstream.supply for p in self.priority]
                    for item, flow in zip(self.upstream, flows):
                        item.outflow = flow
                    self.downstream.inflow = self.downstream.supply

        else:
            raise ValueError('there is a confluentor having more than three branches.')
    
    def calculate_flow(self):
        """calculate the inflows and outflows for the upstream and downstream sections"""
        self._func()

