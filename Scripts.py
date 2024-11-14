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
import json
import gc
import time

def time_convert(date):
    fmt = "%d %b %Y %H:%M:%S.%f"
    try:
        t = datetime.datetime.strptime(date[:-3], fmt)
    except:
        t = datetime.datetime.strptime(date, fmt)
    return pd.Timestamp(year=t.year, month=t.month, day=t.day, hour = t.hour, minute = t.minute ,second=t.second, microsecond=t.microsecond)

def get_ind(n_planes):
    df = pd.read_csv(f"../../Output_Files/pareto.csv")
    df = df[df['Num_Planes'] == n_planes]
    if len(df) > 0:
        df = df.sort_values(by='Avg_Time')
        df = df.reset_index(drop=True)
        df = df[df.columns[:6]]
        return df.iloc[0].to_list()
    else:
        return 0

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
                          previous_dtheta,
                          new_time,
                          new_dtheta,
                          slew_rate):

    if len(previous_times) > 0:

        # Calculate time differences
        d_time = np.abs(new_time - previous_times)

        # Return maneuverability condition, ensuring no division by zero
        ratio = np.divide(previous_dtheta + new_dtheta, d_time,
                          out=np.full_like(d_time, 10),
                          where=d_time != 0)
        
        return ratio <= slew_rate

    # Simplified handling for edge cases when there are no previous times
    return [[slew_rate > 0 or (slew_rate == 0 and new_dtheta == 0)]]

def get_best_available_access(satellite_specific_plan,bin_access_points,slew_rate):
    if len(bin_access_points)>0:
        for point in bin_access_points:
            previous_sat_accesses = satellite_specific_plan[int(point[2])]
            feasible = check_manueverability(np.array(previous_sat_accesses["Time"]),
                                             np.array(previous_sat_accesses["dTheta"]),
                                             point[0],
                                             point[1],
                                             slew_rate)
            
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

def send_message_to_discord(message, channel_id = 1203813613903675502,bot_token=32):
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

def create_pareto(df,objective1='Cost',obj1_type=-1, objective2='Avg_Percentage',obj2_type=1, plot=True,plot_title='Pareto Frontier',xlabel='',ylabel=''):

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
                            hoverinfo='text',mode='markers',name='Dominated Designs',marker=dict(size=8))
    
        pareto_line = go.Scatter(x=pareto_frontier[objective1],y=pareto_frontier[objective2],hovertext=pareto_frontier.apply(lambda row: '<br>'.join([f'{col}: {row[col]}' for col in pareto_frontier.columns]), axis=1),
            hoverinfo='text',mode='lines+markers',name='Pareto Frontier',line=dict(color='green'),marker=dict(size=8))
        
        fig.add_trace(scatter)
        fig.add_trace(pareto_line)

        if len(xlabel) == 0:
            xlabel = objective1
        if len(ylabel) == 0:
            ylabel = objective2

        fig.update_layout(title=f'{plot_title}',xaxis_title=f'{xlabel}',yaxis_title=f'{ylabel}',legend=dict(x=1, y=1.25),template='plotly_white')
        fig.show()

    return pareto_frontier
    
def Load_Individual(Individual=[0,0,0,0,0,0]):
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
    
def Update_Pareto_Performance(stk_object,design_idx,tar_list=[15,34]):
    pareto_designs = pd.read_csv("../../Output_Files/pareto.csv")

    with open('../../Output_Files/pareto_performance.json', "r") as json_file:
        design_evaluations = json.load(json_file)

    new_df = {}

    execute = len(pareto_designs)*[False]
    execute[design_idx] = True

    for idx,design in pareto_designs.iterrows():
        ind = design.tolist()[:6]
        
        if not(f'{ind}' in design_evaluations.keys()):
            new_df[f'{ind}'] = {"Cost":design.iloc[8]}
        else:
            new_df[f'{ind}'] = design_evaluations[f'{ind}']

        for tar_num in tar_list:
            if execute[idx]:
                new_df[f'{ind}'][f'{tar_num} Targets'] = {}
                stk_object.Target_Loader(f"../../Input_Files/Target_Packages/Targets_{tar_num}.txt")
                
                t1 = time.time()
                Load_Individual(ind)
                stk_object.Satellite_Loader("../../Input_Files/Satellites_File.txt")
                
                stk_object.Generate_Pre_Planning_Data()
                stk_object.Plan(enable_print=True)

                t2 = time.time()

                if np.average([np.count_nonzero(stk_object.target_bins[tar_num])/324*100 for tar_num in range(len(stk_object.targets))]) == 100:
                    stk_object.hundred = True
                else:
                    stk_object.hundred = False
                stk_object.Create_Data_Comparison_df()
                df = stk_object.data_comparison
                new_df[f'{ind}'][f'{tar_num} Targets']['Computation_Time'] = round(t2-t1,2)
                for key in ['Unplanned (%)', 'Unplanned (Time)', 'Planned (%)', 'Planned (Time)']:
                    new_df[f'{ind}'][f'{tar_num} Targets'][key] = df[key].to_list()

    with open('../../Output_Files/pareto_performance.json', "w") as json_file:
        json.dump(new_df,json_file,indent=4)
                


def json_to_html(json_data, output_file="json_viewer.html"):
    # HTML template with JavaScript and CSS for collapsible keys
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>JSON Viewer</title>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            ul {{ list-style-type: none; }}
            .collapsible {{ cursor: pointer; }}
            .nested {{ display: none; }}
            .active {{ display: block; }}
        </style>
    </head>
    <body>
        <h2>JSON Viewer</h2>
        <ul id="jsonContainer"></ul>
        <script>
            // JSON data as a JavaScript variable
            const jsonData = {json.dumps(json_data, indent=4)};

            function createTreeView(obj, container) {{
                for (let key in obj) {{
                    if (obj.hasOwnProperty(key)) {{
                        const li = document.createElement("li");
                        if (typeof obj[key] === 'object' && obj[key] !== null) {{
                            li.innerHTML = '<span class="collapsible">➕ ' + key + '</span>';
                            const nestedUl = document.createElement("ul");
                            nestedUl.classList.add("nested");
                            createTreeView(obj[key], nestedUl);
                            li.appendChild(nestedUl);
                        }} else {{
                            li.textContent = key + ": " + obj[key];
                        }}
                        container.appendChild(li);
                    }}
                }}
            }}

            // Toggle display for collapsible items
            document.addEventListener("click", function(e) {{
                if (e.target.classList.contains("collapsible")) {{
                    e.target.classList.toggle("active");
                    const content = e.target.nextElementSibling;
                    if (content) {{
                        content.classList.toggle("active");
                        e.target.textContent = e.target.textContent.includes("➕") 
                            ? e.target.textContent.replace("➕", "➖") 
                            : e.target.textContent.replace("➖", "➕");
                    }}
                }}
            }});

            // Initialize the JSON viewer
            const jsonContainer = document.getElementById("jsonContainer");
            createTreeView(jsonData, jsonContainer);
        </script>
    </body>
    </html>
    """

    # Write the HTML content to an output file with utf-8 encoding
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(html_content)

def Generate_Performance_Curve():
    with open('../../Output_Files/pareto_performance.json', "r") as json_file:
        pareto_performance_dict = json.load(json_file)
    fig = go.Figure()
    for Individual, design_data in pareto_performance_dict.items():
        num_targets = [int(targets.split(' ')[0]) for targets in list(design_data.keys())[1:]]
        times = [np.average(design_data[f'{targets}']['Planned (Time)']) for targets in list(design_data.keys())[1:]]
        percentages = [np.average(design_data[f'{targets}']['Planned (%)']) for targets in list(design_data.keys())[1:]]

        results_df = pd.DataFrame({'Tar_Num': num_targets, 'Times': times, 'Percentages': percentages})
        results_df = results_df[results_df['Percentages'] == 100]

        cost = int(design_data['Cost']/1e6)
        design = pd.DataFrame([Individual[1:-1].split(',')], columns=['Alt','Inc', 'Initial_Raan','Delta_Raan','Num_Sats', 'Num_Planes'])
        fig.add_trace(go.Scatter(
            x=results_df['Tar_Num'],
            y=results_df['Times'],
            hovertext=design.apply(lambda row: '<br>'.join([f'{col}: {row[col]}' for col in design.columns]), axis=1),
            hoverinfo='text',
            mode='lines+markers',
            name=str(cost)))
        
    fig.add_hline(30, line_dash='dash', line_color='red')
    fig.update_layout(
        title='Average Time vs. Number of Targets',
        xaxis_title='Number of Targets',
        yaxis_title='Average Time',
        legend_title='Cost [M$]',
        template='plotly',
        height=600,
        width=1000
    )
    fig.show()
