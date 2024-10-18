import random
import numpy as np
import pandas as pd
import datetime
from shapely.geometry import Polygon, Point
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

def get_ind(n_sats):
    df = pd.read_csv(f"../../Input_Files/Satellites_File.txt")
    return [df["Per"].values[0],df["Inc"].values[0],max(df["Asc"]),len(np.unique(df["Loc"]))]

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


class Optimizer2:
    def __init__(self, stk_object, n_pop, n_gen, weights=(7.0,-1.0,-5.0)):
        self.stk_object = stk_object
        self.n_pop = n_pop
        self.n_gen = n_gen
        creator.create("FitnessMax", base.Fitness, weights=weights)
        creator.create("Satellite", list, fitness=creator.FitnessMax)
        self.lower = [575, 87, 30, 0, 3, 1]
        self.upper = [630, 95, 150, 30, 12, 12]

        # self.norm_array = np.array([100,self.stk_object.duration.total_seconds(),12,12])
        self.norm_array = np.array([100,12,12])

        # Registering variables to the satellite
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_alt", random.randint, self.lower[0], self.upper[0])
        self.toolbox.register("attr_inc", random.randint, self.lower[1], self.upper[1])
        self.toolbox.register("attr_initial_raan", random.randint, self.lower[2], self.upper[2])
        self.toolbox.register("attr_delta_raan", random.randint, self.lower[3], self.upper[3])
        self.toolbox.register("attr_num_sats", random.randint, self.lower[4], self.upper[4])
        self.toolbox.register("attr_num_planes", random.randint, self.lower[5], self.upper[5])

        def satellite_init():
            # Normalized variables between 0 and 1
            alt_norm = random.random()  # Normalized altitude
            inc_norm = random.random()  # Normalized inclination
            raan_norm = random.random()  # Normalized RAAN
            delta_raan_norm = random.random()  # Normalized Delta RAAN
            num_sats_norm = random.random()  # Normalized number of satellites
            num_planes_norm = random.random()  # Normalized number of planes
            
            # Denormalize to original ranges
            alt = int(alt_norm * (self.upper[0] - self.lower[0]) + self.lower[0])
            inc = int(inc_norm * (self.upper[1] - self.lower[1]) + self.lower[1])
            initial_raan = int(raan_norm * (self.upper[2] - self.lower[2]) + self.lower[2])
            delta_raan = int(delta_raan_norm * (self.upper[3] - self.lower[3]) + self.lower[3])
            num_sats = int(num_sats_norm * (self.upper[4] - self.lower[4]) + self.lower[4])
            num_planes = min(int(num_planes_norm * (self.upper[5] - self.lower[5]) + self.lower[5]), num_sats)


            return creator.Satellite([alt, inc, initial_raan, delta_raan, num_sats, num_planes])

        self.toolbox.register("satellite", satellite_init)

        # Registering tools for the algorithm

        def mutate_individual(mutant):
            # Apply the registered mutation operator directly
            tools.mutUniformInt(mutant, low=self.lower, up=self.upper, indpb=0.8)
            
            # Enforce the constraint after mutation
            if mutant[4] < mutant[5]:  # if n_sats < n_planes
                mutant[5] = mutant[4]  # Set n_planes = n_sats

            return mutant
        
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.satellite)
        self.toolbox.register("evaluate", self.cost_function)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=int(self.n_pop // 2))
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)


    def run(self,read=False,enable_print=False):
        self.enable_print = enable_print
        self.fits = []
        CXPB = 0.7;MUTPB=0.3
        self.g = 0
        # Creating a population to evolve
        pop = self.toolbox.population(n=self.n_pop)
        i = np.random.randint(0,self.n_pop)
        if read:
            for idx in range(4):
                pop[i][idx] = get_ind(4)[idx]

        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        hof = tools.HallOfFame(5)
        hof.update(pop)

        self.fits.append([ind.fitness.values for ind in pop])
        record = self.stats.compile(pop)

        clear_output(wait=False)
        print("-- Generation %i --" % self.g)
        print(pd.DataFrame(record))

        with open(f"../../Pop_Over_Gen/pop_gen.csv","w") as file:
            file.write("Gen,Pop,Alt,Inc,Initial_Raan,Delta_Raan,Num_Sats,Num_Planes,Avg_Percentage,Num_Sats,Num_Planes,\n")
            for i in range(self.n_pop):
                file.write(f"{self.g},{i},")
                for idx in range(6):
                    file.write(f"{pop[i][idx]},")
                for fit in pop[i].fitness.getValues()*self.norm_array:
                    file.write(f"{fit},")
                file.write("\n")
        # Begin the evolution
        while self.g < self.n_gen:
            # clear_output(wait=True)
            self.g = self.g + 1

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

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            pop[:] = offspring
            hof.update(pop)

            self.fits.append([ind.fitness.values for ind in pop])
            record = self.stats.compile(pop)

            clear_output(wait=False)
            print("-- Generation %i --" % self.g)
            print(pd.DataFrame(record))
            
            with open(f"../../Pop_Over_Gen/pop_gen.csv","a") as file:
                for i in range(self.n_pop):
                    file.write(f"{self.g},{i},")
                    for idx in range(6):
                        file.write(f"{pop[i][idx]},")
                    for fit in pop[i].fitness.getValues()*self.norm_array:
                        file.write(f"{fit},")
                    file.write("\n")
        return hof
    
    def cost_function(self,Individual=[0,0,0,0,0,0],write=True):
        n_planes = Individual[5]
        n_sats = Individual[4]

        gen = self.g

        if write:
            self.Load_Individual(Individual)
        self.stk_object.Satellite_Loader(f'../../Input_Files/Satellites_File.txt')
        self.stk_object.Compute_AzEl(self.enable_print)
        percentages = np.array([np.count_nonzero(self.stk_object.target_bins[idx])/324 for idx in range(len(self.stk_object.targets))])
        times = np.array([self.stk_object.target_times[idx]/self.norm_array[1] for idx in range(len(self.stk_object.targets))])
        
        percentage = np.average(percentages)

        penalty = 0
        if n_planes > n_sats:
            penalty = 100 * (n_planes - n_sats) 


        if percentage < 1:
            if gen >= 1:
                percentage = percentage - (gen)/500
        else:
            percentage = 1

        fitness = tuple([round(percentage,4), (n_sats + n_planes)/24 + penalty])

        return tuple(fitness)
    
    def Load_Individual(self,Individual=[0,0,0,0,0,0]):
        Alt = Individual[0]
        Inc = Individual[1]
        initial_raan = Individual[2]
        delta_raan = Individual[3]
        n_planes = int(Individual[5])
        n_sats = int(Individual[4])
        
        if n_planes > n_sats:
            n_planes = n_sats

        Asc = initial_raan
        file = open(f"../../Input_Files/Satellites_File.txt","w")
        file.write("Per,Apo,Inc,Asc,Loc\n")
        sats = n_sats*[1]
        planes = np.array_split(sats,n_planes)
        i=1
        for plane in planes:
            Loc = 0
            for sat in plane:
                file.write(f"{Alt},{Alt},{Inc},{round(Asc%360,4)},{round(Loc,4)}\n")
                if len(plane)>1: Loc += 360/len(plane)
            if len(planes)>1:Asc -= i*((-1)**(i))*delta_raan
            i+=1
        file.close()

        
def Interpolate(time,az,el):
    times = np.arange(time[0],time[-1],2.5)
    if max(el)>=60 and len(time)>3:
        az_t = interpolate.interpn(points=[time],values=np.array([np.unwrap(az,period=360)]).T,xi=times,method='pchip')[:,0]%360
        el_t = interpolate.interp1d(x=time,y=[el],kind='cubic')(times)[0]
    else:
        ans = interpolate.interp1d(x=time,y=[np.unwrap(az,period=360),el],kind='quadratic')(times).T
        az_t = ans[:,0]%360;el_t = ans[:,1]
    return times,az_t,el_t

def check_manueverability(previous_times,
                          previous_crossrange,
                          previous_alongrange,
                          new_time,
                          new_crossrange,
                          new_along_range,
                          slew_rate,
                          cone_angle):

    d_theta_2 = (new_crossrange**2+new_along_range**2)**0.5-cone_angle
    if d_theta_2 < 0:
        d_theta_2 = 0

    if len(previous_times)>0:
        d_theta_1 = (previous_crossrange**2+previous_alongrange**2)**0.5-cone_angle
        d_theta_1[d_theta_1<0] = 0

        d_time = np.abs(new_time-previous_times)

        return np.divide(d_theta_1+d_theta_2,d_time,out=1.1*slew_rate*np.ones_like(d_time),where=d_time!=0)<=slew_rate
    elif (slew_rate==0 and d_theta_2==0) or (slew_rate>0):
        return [[True]]
    else:
        return [[False]]

def get_best_available_access(satellite_specific_plan,bin_access_points,slew_rate,cone_angle,time_threshold):
    if len(bin_access_points)>0:
        for idx in range(len(bin_access_points)):
            previous_sat_accesses = satellite_specific_plan[int(bin_access_points[idx,3])]
            window = [i for i, t in enumerate(previous_sat_accesses["Time"]) if abs(t - bin_access_points[idx,0]) <= time_threshold]
            feasible = check_manueverability(np.array(previous_sat_accesses["Time"])[window],
                                             np.array(previous_sat_accesses["Cross Range"])[window],
                                             np.array(previous_sat_accesses["Along Range"])[window],
                                             bin_access_points[idx,0],
                                             bin_access_points[idx,1],
                                             bin_access_points[idx,2],
                                             slew_rate,
                                             cone_angle)
            if np.all(feasible):
                return bin_access_points[idx]
    return False

def Generate_Performance_Curve(cost_curve_dicts, curve_type='Optimization', xaxis='Number of Targets', yaxis='Avg_time'):
    
    fig = go.Figure()
    for n_sats, n_sats_df in cost_curve_dicts[curve_type].items():
        fig.add_trace(go.Scatter(
            x=n_sats_df[xaxis],
            y=n_sats_df[yaxis],
            mode='lines+markers',
            name=str(n_sats)
        ))

    fig.add_hline(30, line_dash='dash', line_color='red')
    fig.add_annotation(x=100, y=32, text='30 Day Constraint', font=dict(color='red', size=15), showarrow=False)
    fig.update_layout(
        title=f'{curve_type} {yaxis} vs. {xaxis}',
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        legend_title='Number of Satellites',
        template='plotly',
        height=600,
        width=1000
    )
    fig.show()

def send_message_to_discord(message, channel_id = 1203813613903675502):

    bot_token='MTI5MjE4NjkxNDE5MDkxNzcyMg.GtL1SI.T3aRSzcSCGAFcTDPwZFhqd4xcjiIx0NnAn4ays'

    if len(bot_token) > 10:
        import discord
        import nest_asyncio
        import asyncio
        nest_asyncio.apply()
        intents = discord.Intents.default()
        intents.message_content = True
        bot = discord.Client(intents=intents)
        async def send_message_and_exit():
            channel = bot.get_channel(channel_id)
            if channel is not None:
                await channel.send(message)
            else:
                print("Channel not found.")
            await bot.close()
        @bot.event
        async def on_ready():
            await send_message_and_exit()
            await bot.close()
        bot.run(bot_token)
