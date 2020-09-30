# -*- coding: utf-8 -*-

# @Author: KeyangZhang
# @Email: 3238051@qq.com
# @Date: 2019-12-26 10:42:53
# @LastEditTime: 2020-04-30 15:06:39
# @LastEditors: Keyangzhang

from .base import Section,SignalController,Phase
from .base import _Connector
from random import uniform

def _poisson(lam):
    u = uniform(0,1)
    p = 2.718281828459045**(-lam)
    f = p
    k = 0
    if u < f:
        return k
    else:   
        while f < u:
            p = p*lam/(k+1)
            f += p
            k += 1
        return k-1

def _constant(x):
    return x

class _ArrivalController(object):
    """upstream arrival
    
    it is analogue to the demand, which is measured by traffic volume (pcu/hour).
    
    """
    def __init__(self):
        self.arrivals = {}  # {arrival_id:[section,volume,distribution_function],...}


    def create_arrival(self,arrival_id, section, volume, distribution='poisson'):

        
        if distribution == 'poisson':
            self.arrivals[arrival_id]=[section,volume,_poisson]

        elif distribution == 'constant':
            self.arrivals[arrival_id]=[section,volume,_constant]
        
        else:
            raise ValueError('No such type of distribution')

    def update(self):
        for sec, vol, func in self.arrivals.values():
            vol = vol/3600 # change the scale from pcu/h to pcu/s
            demand = func(vol)
            flow = min(demand, sec.supply)
            sec.inflow = flow

class _DepartureController(object):
    """downstream departure

    it defines the capacity or congestion level of downstream of certain sections 
    
    """
    def __init__(self):

        self.departures = {}  # {departure_id:[section,capcity,distribution_function],...}
    
    def create_departure(self,departure_id, section, capacity, distribution='constant'): 
        
        if distribution == 'constant':
            self.departures[departure_id]=[section,capacity,_constant]
        else:
            raise ValueError('No such type of distribution')
    
    def set_departure(self,departure_id,new_capacity=None,new_distribution=None):
        if new_capacity is not None:
            self.departures[departure_id][1] = new_capacity
        if new_distribution is not None:
            self.departures[departure_id][2] = new_distribution 

    
    def update(self):
        
        for sec, cap, func in self.departures.values():
            cap = cap/3600 # change scale from pcu/h to pcu/s
            supply = func(cap)
            flow = min(supply, sec.demand)
            sec.outflow = flow

class Simulator(object):
    """Simulator
    
    the simulator is designed for creating, connecting and storing elements of the network
    it provides basic functions to execute simulation

    Attributes:
        sections: dict, contains created sections and their ids
        signalcontrollers: dict, contains created signal controllers and their ids
        phases: dict, contains designed phases and their ids
        connectors: list, contains the created connectors

    """
    
    def __init__(self):
        super(Simulator, self).__init__()
        self.sections = {}
        self.signalcontrollers = {}
        self.connectors = []
        self.currenttime = 0
        self.arrivalcontroller = _ArrivalController()
        self.departurecontroller = _DepartureController()

    def create(self,object_class,object_id,kargs):
        """ create network elements

        it can create the network elements including Section, SignalController, 
        Phase, lamp, demand(arrival) and donwstream capacity(departure) that the road at border has

        Args:
            object_class: str, it should be one of the Section, SignalController or Phase
            kargs: dict, contains the parameters and value for the object you want to create
                   the keys should be consistent with the parameters of the inital function
                   e.g. for Section, it may be {'lane_length':5,'lane_number':3,'cell_length':0.5}     

        """
        if object_class=='section':
            obj = Section(**kargs)
            self.sections[object_id]=obj
            return obj
        
        elif object_class=='signalcontroller':
            obj = SignalController(**kargs)
            self.signalcontrollers[object_id]=obj
            return obj
        
        elif object_class=='phase':
            sc = self.signalcontrollers[kargs['signalcontroller_id']]
            start, end = kargs['start'],kargs['end']
            obj = sc.create_phase(object_id,start,end)
            return obj
        
        elif object_class=='lamp':
            sc = self.signalcontrollers[kargs['signalcontroller_id']]
            sec_id,ph_id = kargs['section_id'], kargs['phase_id']
            sec = self.sections[sec_id]
            obj = sc.create_lamp(object_id,sec_id,sec,ph_id)
            return obj

        elif object_class=='arrival':
            sec_id, vol = kargs['section_id'],kargs['volume']
            distri = kargs.get('distribution',None)
            if distri is not None:
                sec = self.sections[sec_id]
                self.arrivalcontroller.create_arrival(object_id,sec,vol,distri)
            else:
                sec = self.sections[sec_id]
                self.arrivalcontroller.create_arrival(object_id,sec,vol)   

        elif object_class=='departure':
            sec_id, cap = kargs['section_id'],kargs['capacity']
            distri = kargs.get('distribution',None)
            if distri is not None:
                sec = self.sections[sec_id]
                self.departurecontroller.create_departure(object_id,sec,cap,distri)
            else:
                sec = self.sections[sec_id]
                self.departurecontroller.create_departure(object_id,sec,cap) 

        else:
            raise ValueError('No such object: '+object_class)
    
    def get(self,object_class, object_id):

        if object_class == 'section':
            return self.sections.get(object_id,None)
        
        elif object_class == 'signalcontroller':
            return self.signalcontrollers.get(object_id,None)
        
        else:
            raise ValueError('no such object type: {0}'.format(object_class))

    def connect(self,id_upstream,id_downstream,priority=None):
        """ connect sections

        it is used to connect the created sections 
        
        Args:
            id_upstream: id or tuple/list, the id or the tuple/list of ids of the upstream sections
            id_downstream: id or tuple/list, the id or the tuple/list of ids of the downstream sections
            priority: float, the relative weight of the flows  
        """
        if isinstance(id_upstream,(list,tuple)):
            upstream = [self.sections[i] for i in id_upstream]
            downstream = self.sections[id_downstream]
            cnct = _Connector(upstream,downstream,priority)
            self.connectors.append(cnct)
   
        elif isinstance(id_downstream,(list,tuple)):
            upstream = self.sections[id_upstream]
            downstream = [self.sections[i] for i in id_downstream]
            cnct = _Connector(upstream,downstream,priority)
            self.connectors.append(cnct)
        
        else:
            upstream = self.sections[id_upstream]
            downstream = self.sections[id_downstream]
            cnct = _Connector(upstream,downstream)
            self.connectors.append(cnct)


    def run(self,sumtime):
        """ run simulation for sumtime

        Args:
            sumtime: integer, total time of simulation

        """
        t = 0
        while t<=sumtime:
            self.run_single_step()
            t+=1         

    def run_single_step(self):
        """execute a single step of simulation
        
        """
        # calculate demand and supply
        for sec in self.sections.values():
            sec.calculate_demand()
            sec.calculate_supply()

        # update signal status
        for sig in self.signalcontrollers.values():
            sig.update_signal(self.currenttime)

        # set the demand and downstream capacity
        self.arrivalcontroller.update()
        self.departurecontroller.update()

        # calculate flow
        for sec in self.sections.values():
            sec.calculate_flow()
        for cnct in self.connectors:
            cnct.calculate_flow()

        # update volume
        for sec in self.sections.values():
            sec.update_volume()

        self.currenttime+=1

    def clear(self):
        for sec in self.sections.values():
            sec.clear()
        
        self.currenttime = 0
    
    def get_statistic(self, statistic,level='cell'):

        if statistic == 'volume':
            result = self.__cal_volume(level)
            return result
        
        if statistic == 'density':
            result = self.__cal_density(level)
            return result
        
        if statistic == 'occupancy':
            result = self.__cal_occupancy(level)
            return result

    def __cal_volume(self, level='cell'):
        """get the volume (pcu) of each cell/section at current sim-second

        Args:
            level:str,  it can be cell or section.
        
        Returns:
            volumes: dict, the key is section id and the value is list of volumes of each cell.
                     if level='section', the value is total volume of this section.
            
        """
        volumes = {}
        if level=='cell':
            for i,sec in self.sections.items():
                volumes[i] = [round(c.volume,1) for c in sec]
        elif level=='section':
            for i,sec in self.sections.items():
                volumes[i] = round(sum([c.volume for c in sec]))
        else :
            raise ValueError('no such level for collecting data')
        return volumes
    
    def __cal_density(self, level='cell'):
        """get the average single-lane density(pcu/km) of each cell/section at current sim-second

        Args:
            level:str,  it can be cell or section.
    
        Returns:
            densities: dict, the key is section id and the value is list of densities of each cell.
                       if level='section', the value is the average signle-lane density of this section
            
        """
        densities = {}
        if level=='cell':
            for i,sec in self.sections.items():
                densities[i] = [round(c.volume/(sec.cell_length*sec.lane_number),3) for c in sec]
        elif level=='section':
            for i,sec in self.sections.items():
                volume = sum([c.volume for c in sec])
                densities[i] = round(volume/(sec.lane_length*sec.lane_number),3)
        else :
            raise ValueError('no such level for collecting data')
        
        return densities
        
    
    def __cal_occupancy(self,level='cell'):
        """get the occupancy of each cell/section at current sim-second

        Args:
            level:str,  it can be cell or section.
       
        Returns:
            occupancies: dict, the key is section id and the value is list of occupancies of each cell.
                         if level='section', the value is occupancy of this section.
            
        """
        occupancies = {}
        if level=='cell':
            for i,sec in self.sections.items():
                occupancies[i] = [round(c.volume/sec.cell_length*sec.jam_density,3) for c in sec]
        elif level=='section':
            for i,sec in self.sections.items():
                volume = sum([c.volume for c in sec])
                occupancy = volume/(sec.lane_length*sec.jam_density)
                occupancies[i]=round(occupancy,3)
        else :
            raise ValueError('no such level for collecting data')
        
        return occupancies