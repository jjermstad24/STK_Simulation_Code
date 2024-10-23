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


class Optimizer:
    def __init__(self, stk_object, n_pop, n_gen, run_num,weights=(.5,.25,.25)):
        self.run_num = run_num
        self.weights = weights
        self.stk_object = stk_object
        self.n_pop = n_pop
        self.n_gen = n_gen
        creator.create("FitnessMax", base.Fitness, weights=weights)
        creator.create("Satellite", list, fitness=creator.FitnessMax)
        self.lower = [575, 80, 30, 0, 3, 1]
        self.upper = [630, 100, 150, 30, 12, 12]

        self.norm_array = np.array([100,self.stk_object.duration.total_seconds()/86400,24])
        
        
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

        if self.run_num == 0:
            w = 'w'
        else:
            w = 'a'
        with open(f"../../Pop_Over_Gen/pop_gen.csv",f"{w}") as file:
            if self.run_num == 0:
                file.write(f"Run_Num,Per_Weight,Time_Weight,Cost_Weight,Gen,Pop,Alt,Inc,Initial_Raan,Delta_Raan,Num_Sats,Num_Planes,Avg_Percentage,Avg_Time,Cost,\n")
            
            for i in range(self.n_pop):
                file.write(f'{self.run_num},')
                for weight in self.weights:
                    file.write(f"{weight},")
                file.write(f'{self.g},{i},')
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
                    file.write(f'{self.run_num},')
                    for weight in self.weights:
                        file.write(f"{weight},")
                    file.write(f'{self.g},{i},')
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
        self.stk_object.Results_Runner()
        self.stk_object.Create_Data_Comparison_df(Unplanned=False)
        percentages = self.stk_object.data_comparison['Planned (%)']
        times = self.stk_object.data_comparison['Planned (Time)']

        percentage = np.average(percentages)
        time = np.average(times)

        penalty = 0
        if n_planes > n_sats:
            penalty = 10 * (n_planes - n_sats) 

        # if percentage < 1:
        #     # if gen >= 1:
        #     #     percentage = percentage - (gen)/500
        #     time = self.stk_object.durtation.total_seconds()

        objective = tuple(np.array([percentage, time, (n_sats + n_planes) + penalty])/self.norm_array)

        return objective
    
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
                file.write(f"{Alt},{Alt},{Inc},{round(Asc%180,4)},{round(Loc,4)}\n")
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
    d_theta_2 = np.hypot(new_crossrange, new_along_range) - cone_angle
    d_theta_2 = max(d_theta_2, 0)  # Clamp to 0 if negative

    if len(previous_times) > 0:
        # Compute d_theta_1 for the previous points
        d_theta_1 = np.hypot(previous_crossrange, previous_alongrange) - cone_angle
        np.maximum(d_theta_1, 0, out=d_theta_1)  # Clamp values in-place

        # Calculate time differences
        d_time = np.abs(new_time - previous_times)

        # Return maneuverability condition, ensuring no division by zero
        ratio = np.divide(d_theta_1 + d_theta_2, d_time,
                          out=np.full_like(d_time, 10),
                          where=d_time != 0)
        
        return ratio <= slew_rate

    # Simplified handling for edge cases when there are no previous times
    return [[slew_rate > 0 or (slew_rate == 0 and d_theta_2 == 0)]]

def get_best_available_access(satellite_specific_plan,bin_access_points,slew_rate,cone_angle,time_threshold=60):
    if len(bin_access_points)>0:
        for point in bin_access_points:
            previous_sat_accesses = satellite_specific_plan[int(point[3])]
            feasible = check_manueverability(np.array(previous_sat_accesses["Time"]),
                                             np.array(previous_sat_accesses["Cross Range"]),
                                             np.array(previous_sat_accesses["Along Range"]),
                                             point[0],
                                             point[1],
                                             point[2],
                                             slew_rate,
                                             cone_angle)
            
            if np.all(feasible):
                return point
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

    bot_token='MTI5MjE4NjkxNDE5MDkxNzcyMg.GQpP9E.3h92p2D3WQolvR5sZ9CPZVWGwqlY6gpBSA4gx8'

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

def create_pareto(df,objective1='Cost',obj1_type=-1, objective2='Avg_Percentage',obj2_type=1, plot=True,plot_title='Pareto Frontier'):

    if obj1_type < 0:
        df = df.sort_values(by=objective1,ascending=True,ignore_index = True)
    else:
        df = df.sort_values(by=objective1,ascending=False,ignore_index = True)

    objective1_index = df.columns.tolist().index(objective1)
    objective2_index = df.columns.tolist().index(objective2)

    pareto_frontier = []
    pareto_frontier = [[df[idx][0] for idx in df.columns.tolist()]]

    for index, row in df[1:].iterrows():
        point = [row[idx] for idx in df.columns.tolist()]
        if obj2_type > 0:
            if point[objective2_index] > pareto_frontier[-1][objective2_index]:
                if point[objective1_index] == pareto_frontier[-1][objective1_index]:
                    pareto_frontier.pop(-1)
                pareto_frontier.append(point)
        else:
            if point[objective2_index] < pareto_frontier[-1][objective2_index]:
                if point[objective1_index] == pareto_frontier[-1][objective1_index]:
                    pareto_frontier.pop(-1)
                pareto_frontier.append(point)
    pareto_frontier = pd.DataFrame(pareto_frontier, columns=df.columns.tolist())

    if plot:
        fig = make_subplots()

        scatter = go.Scatter(x=df[objective1],y=df[objective2],hovertext=df.apply(lambda row: '<br>'.join([f'{col}: {row[col]}' for col in df.columns]), axis=1),
                            hoverinfo='text',mode='markers',name='Data Points',marker=dict(size=8))
    
        pareto_line = go.Scatter(x=pareto_frontier[objective1],y=pareto_frontier[objective2],hovertext=pareto_frontier.apply(lambda row: '<br>'.join([f'{col}: {row[col]}' for col in pareto_frontier.columns]), axis=1),
            hoverinfo='text',mode='lines+markers',name='Pareto Frontier',line=dict(color='green'),marker=dict(size=8))
        
        fig.add_trace(scatter)
        fig.add_trace(pareto_line)
        fig.update_layout(title=f'{plot_title}',xaxis_title=f'{objective1}',yaxis_title=f'{objective2}',legend=dict(x=1, y=1.25),template='plotly_white')
        fig.show()

    return pareto_frontier

def pop_over_gen_animator(pop_over_gen_df,plot_title='Pop_Over_Gen', objective1='Cost', obj1_type=1,objective2='Avg_Percentage', obj2_type=1,constraint=True, 
                                  xticks=[0,5,10,15,20], yticks=[0,25,50,75,100], add_pareto=False):
    
    colors = ['blue', 'orange', 'black', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'olive', 'yellow']
    
    figs = []
        
    fig_animation = make_subplots()
    
    frames = []

    gens = pop_over_gen_df['Gen'].unique()

    # Loop through each generation and plot scatter points and optionally the Pareto frontier
    for gen in gens:
        df = pop_over_gen_df[pop_over_gen_df['Gen'] == gen]
        
        # Scatter plot for each generation
        scatter = go.Scatter(x=df[objective1],y=df[objective2],hovertext=df.apply(lambda row: '<br>'.join([f'{col}: {row[col]}' for col in df.columns]), axis=1),hoverinfo='text',mode='markers',
            name=f'{gen}',marker=dict(color=colors[gen % len(colors)], size=8))
        
        if gen == 0:
            fig_animation.add_trace(scatter)

        if add_pareto:
            pareto_frontier = create_pareto(df,objective1=objective1, obj1_type=obj1_type,objective2=objective2,obj2_type=obj2_type,plot=False)

            pareto_trace = go.Scatter(x=pareto_frontier[objective1],y=pareto_frontier[objective2],hovertext=pareto_frontier.apply(lambda row: '<br>'.join([f'{col}: {row[col]}' for col in pareto_frontier.columns]), axis=1),
                hoverinfo='text',mode='lines+markers',name=f'Pareto Frontier {gen}',line=dict(color='green'),marker=dict(size=8))
            
            if gen == 0:
                fig_animation.add_trace(pareto_trace)

            frames.append(go.Frame(data=[scatter, pareto_trace], name=str(gen)))
        else:
            frames.append(go.Frame(data=[scatter], name=str(gen)))

    if constraint:
        if 'Time' in objective1:
            fig_animation.add_vline(30, line_dash='dash', line_color='red')
            fig_animation.add_annotation(x=30.5, y=50, text='30 Day Constraint', font=dict(color='red', size=15), textangle=90, showarrow=False)
        else:
            fig_animation.add_annotation(x=15, y=105, text='30 Day Constraint', font=dict(color='red', size=15), showarrow=False)

    fig_animation.update_layout(
        title=f'{plot_title}',
        xaxis_title=objective1,
        yaxis_title=objective2,
        xaxis=dict(tickmode='array', tickvals=xticks, range=[xticks[0], xticks[-1]]),
        yaxis=dict(tickmode='array', tickvals=yticks, range=[yticks[0]-10, yticks[-1]+10]),
        legend=dict(title='Gen', title_font=dict(size=15), font=dict(size=12, color='black')),
        font=dict(size=15), 
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
                'prefix': 'Generation: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [[str(gen)], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 300}}],
                    'label': str(gen),
                    'method': 'animate'
                } for gen in gens
            ]
        }]
    )
    fig_animation.frames = frames
    fig_animation.show()

class MultiObjectiveOptimizer:
    def __init__(self, stk_object, n_pop, n_gen):
        self.stk_object = stk_object
        self.n_pop = n_pop
        self.n_gen = n_gen
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
        creator.create("Satellite", list, fitness=creator.FitnessMulti)
        self.lower = [575, 80, 30, 0, 3, 1]
        self.upper = [630, 100, 150, 30, 12, 12]

        self.norm_array = np.array([100,self.stk_object.duration.total_seconds()/86400,24])
        
        
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
        self.toolbox.register("select", tools.selNSGA2)
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
            file.write(f"Gen,Pop,Alt,Inc,Initial_Raan,Delta_Raan,Num_Sats,Num_Planes,Avg_Percentage,Avg_Time,Cost,\n")
            for i in range(self.n_pop):
                file.write(f'{self.g},{i},')
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
                    file.write(f'{self.g},{i},')
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
        self.stk_object.Results_Runner()
        self.stk_object.Create_Data_Comparison_df(Unplanned=False)
        percentages = self.stk_object.data_comparison['Planned (%)']
        times = self.stk_object.data_comparison['Planned (Time)']

        percentage = np.average(percentages)
        time = np.average(times)

        penalty = 0
        if n_planes > n_sats:
            penalty = 10 * (n_planes - n_sats) 

        # if percentage < 1:
        #     # if gen >= 1:
        #     #     percentage = percentage - (gen)/500
        #     time = self.stk_object.durtation.total_seconds()

        objective = tuple(np.array([percentage, time, (n_sats + n_planes) + penalty])/self.norm_array)

        return objective
    
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
                file.write(f"{Alt},{Alt},{Inc},{round(Asc%180,4)},{round(Loc,4)}\n")
                if len(plane)>1: Loc += 360/len(plane)
            if len(planes)>1:Asc -= i*((-1)**(i))*delta_raan
            i+=1
        file.close()

