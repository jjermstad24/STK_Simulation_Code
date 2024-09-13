import random
import decimal
import numpy as np
import pandas as pd
import datetime
from faker import Faker
from shapely.geometry import Polygon, Point
import plotly.graph_objects as go
import plotly.express as px
import os, sys
import plotly.offline
from dataclasses import dataclass, field
from typing import Dict, List
from alive_progress import alive_bar
from deap import base
from deap import creator
from deap import tools
from IPython.display import clear_output
import scipy.interpolate as interpolate
import time
import pointpats
import re

def Random_Decimal(t):
    lower,upper = t
    return float(decimal.Decimal(random.randrange(lower*10000,upper*10000))/10000)

def Print_Spacing(len=100):
    for i in range(len):
        print("-",end="")
    print()

def Sun_Synchronous_Orbit(revolutions):
    mu = 398600.44
    J2 = 1.08262668*10**-3
    Re = 6378
    RAAN_Rate_SS = 1.99096871*10**-7
    Tsid = 86164
    period = np.pi*2*Tsid/revolutions/(2*np.pi-RAAN_Rate_SS*Tsid)
    altitude = (mu*(period/2/np.pi)**2)**(1/3)-Re
    inclination = np.degrees(np.arccos(-2*RAAN_Rate_SS*((Re+altitude)**7/mu)**0.5/(3*J2*Re**2)))
    return altitude,inclination

def time_convert(date):
    fmt = "%d %b %Y %H:%M:%S.%f"
    try:
        t = datetime.datetime.strptime(date[:-3], fmt)
    except:
        t = datetime.datetime.strptime(date, fmt)
    return pd.Timestamp(year=t.year, month=t.month, day=t.day, hour = t.hour, minute = t.minute ,second=t.second, microsecond=t.microsecond)

def Create_Poly(filename):
    df = pd.read_csv(filename)
    l = []
    for i in range(len(df)):
        l.append((df['Lat'][i],df['Lon'][i]))
    return Polygon(l)

def polygon_random_points (poly, num_points,targets_filename):
    points = pointpats.random.poisson(poly, size=num_points)
    return pd.DataFrame({'Lat':points[:,1],'Lon':points[:,0]}).dropna().to_csv(targets_filename,index=False)

def plot_targets_and_polygon(poly,filename):
    df = pd.read_csv(filename)
    fig = go.Figure(go.Scattermapbox(
        mode = "markers",
        lon = df['Lon'],
        lat = df['Lat'],
        marker = {'size': 10}))

    fig.add_trace(go.Scattermapbox(
        mode = "lines",
        lon = np.array(poly.exterior.coords.xy)[0],
        lat = np.array(poly.exterior.coords.xy)[1],
        marker = {'size': 10}))

    fig.update_layout(
        margin ={'l':0,'t':0,'b':0,'r':0},
        mapbox = {
            'center': {'lon': 0, 'lat': 0},
            'style': "open-street-map",
            'center': {'lon': 0, 'lat': 0},
            'zoom': 0})
    return fig

def Pointing_File_Generator(filename,period):
    f = open(filename,"w")
    f.write("stk.v.12.1.1\nBegin\tAttitude\nNumberofAttitudePoints\t162\nSequence\t323\nRepeatPattern\n")
    for i in range(162):
        f.write(f'{period/162*(i+1)} {(i%9+1)*10-5} {(i//9+1)*10-5}\n')
    f.write('End Attitude')
    f.close()
class Optimizer:
    def __init__(self,stk_object,n_pop,n_gen,n_sats):
        self.stk_object = stk_object
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.n_sats = n_sats
        creator.create("FitnessMax", base.Fitness, weights=(7.0,-1.0,-2.0))
        creator.create("Satellite", list, fitness=creator.FitnessMax)
        self.lower = [575,0,0,1,1]
        self.upper = [630,180,180,self.n_sats,self.n_sats]

        # Registering variables to the satellite
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_alt", random.randint, self.lower[0], self.upper[0])
        self.toolbox.register("attr_inc", random.randint, self.lower[1], self.upper[1])
        self.toolbox.register("attr_aop", random.randint, self.lower[2], self.upper[2])
        self.toolbox.register("attr_num_planes", random.randint, self.lower[3], self.upper[3])
        self.toolbox.register("attr_sat_per_plane", random.randint, self.lower[4], self.upper[4])

        # Registering satellite to the model
        self.toolbox.register("satellite", tools.initCycle, creator.Satellite,
                        (self.toolbox.attr_alt,
                        self.toolbox.attr_inc,
                        self.toolbox.attr_aop,
                        self.toolbox.attr_num_planes,
                        self.toolbox.attr_sat_per_plane), n=1)

        # Registering tools for the algorithm
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.satellite)
        self.toolbox.register("evaluate", self.cost_function)
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded,eta=0.75,low=[x for x in self.lower],up=[x for x in self.upper])
        self.toolbox.register("mutate", tools.mutUniformInt, low=[x for x in self.lower],up=[x for x in self.upper], indpb=0.3)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)

    def run(self):
        self.fits = []
        CXPB = 0.7;MUTPB=0.3
        clear_output(wait=False)
        g = 0
        # Creating a population to evolve
        pop = self.toolbox.population(n=self.n_pop)
        print("-- Generation %i --" % g)
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        hof = tools.HallOfFame(5)
        hof.update(pop)

        percent = {"Gen":[],"avg":[],"std":[],"min":[],"max":[]}
        time = {"Gen":[],"avg":[],"std":[],"min":[],"max":[]}
        sats = {"Gen":[],"avg":[],"std":[],"min":[],"max":[]}

        self.fits.append([ind.fitness.values for ind in pop])
        record = self.stats.compile(pop)
        print(pd.DataFrame(record))
        for idx,df in enumerate([percent,time,sats]):
            df['Gen'].append(g)
            df['avg'].append(record['avg'][idx])
            df['std'].append(record['std'][idx])
            df['min'].append(record['min'][idx])
            df['max'].append(record['max'][idx])

        # Begin the evolution
        while g < self.n_gen:
            # clear_output(wait=True)
            g = g + 1
            print("-- Generation %i --" % g)
            # A new generation
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            pop[:] = offspring
            hof.update(pop)

            self.fits.append([ind.fitness.values for ind in pop])
            record = self.stats.compile(pop)
            print(pd.DataFrame(record))
            for idx,df in enumerate([percent,time,sats]):
                df['Gen'].append(g)
                df['avg'].append(record['avg'][idx])
                df['std'].append(record['std'][idx])
                df['min'].append(record['min'][idx])
                df['max'].append(record['max'][idx])
        print("------------------------------------------------------------------------")
        return hof,percent,time,sats
    
    def cost_function(self,Individual=[0,0,0,0,0],write=True):
        Alt = Individual[0]
        Inc = Individual[1]
        Aop = Individual[2]
        num_sats = int(Individual[3])
        num_planes = int(Individual[4])
        if num_planes <= num_sats:
            if write:
                file = open("../../Input_Files/Satellites_File.txt","w")
                file.write("Per,Apo,Inc,AoP,Asc,Loc,Tar\n")
                sats = num_sats*[1]
                planes = np.array_split(sats,num_planes)
                Asc = 0
                for plane in planes:
                    Loc = 0
                    for sat in plane:
                        file.write(f"{Alt},{Alt},{Inc},{Aop},{round(Asc,4)},{round(Loc,4)},{1}\n")
                        if len(plane)>1: Loc += 360/(len(plane)-1)
                    if len(planes)>1:Asc += 180/(len(planes)-1)
                file.close()
            satellites_filename = 'Input_Files/Satellites_File.txt'
            self.stk_object.Satellite_Loader(satellites_filename)
            print("------------------------------------------------------------------------")
            self.stk_object.Compute_AzEl()
            percentages = [100*np.count_nonzero(self.stk_object.target_bins[idx])/324 for idx in range(len(self.stk_object.targets))]
            times = [self.stk_object.target_times[idx]/86400 for idx in range(len(self.stk_object.targets))]
            return np.average(percentages),np.max(times),len(self.stk_object.satellites)
        else:
            return 0,self.stk_object.root.CurrentScenario.StopTime/86400,12
        
class Interval:
    def __init__(self,access_point,target_number,satellite_number,Interpolate=True):
        self.start = access_point[0,0]
        self.stop = access_point[-1,0]
        self.target_number = target_number
        self.satellite_number = satellite_number
        if Interpolate:
            times = np.arange(self.start,self.stop,2.5)
            if max(access_point[:,2])>=60 and len(access_point[:,0])>3:
                az_t = interpolate.interpn(points=[access_point[:,0]],values=np.array([np.unwrap(access_point[:,1],period=360)]).T,xi=times,method='pchip')[:,0]%360
                el_t = interpolate.interp1d(x=access_point[:,0],y=[access_point[:,2]],kind='cubic')(times)[0]
            else:
                ans = interpolate.interp1d(x=access_point[:,0],y=[np.unwrap(access_point[:,1],period=360),access_point[:,2]],kind='quadratic')(times).T
                az_t = ans[:,0]%360;el_t = ans[:,1]
            
            self.bins = np.unique([int(az//10)*9+int(el//10) for az,el in zip(az_t,el_t)])
        else:
            self.bins = np.unique([int(az//10)*9+int(el//10) for az,el in zip(access_point[:,1],access_point[:,2])])

def Interpolate(time,az,el):
    times = np.arange(time[0],time[-1],2.5)
    if max(el)>=60 and len(time)>3:
        az_t = interpolate.interpn(points=[time],values=np.array([np.unwrap(az,period=360)]).T,xi=times,method='pchip')[:,0]%360
        el_t = interpolate.interp1d(x=time,y=[el],kind='cubic')(times)[0]
    else:
        ans = interpolate.interp1d(x=time,y=[np.unwrap(az,period=360),el],kind='quadratic')(times).T
        az_t = ans[:,0]%360;el_t = ans[:,1]
    return times,az_t,el_t