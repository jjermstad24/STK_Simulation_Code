import random
import decimal
import numpy as np
import pandas as pd
import datetime
from faker import Faker
from shapely.geometry import Polygon, Point
import plotly.graph_objects as go
import plotly.express as px

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

def polygon_random_points (poly, num_points):
    min_x, min_y, max_x, max_y = poly.bounds
    df = {'Lat':[],'Lon':[],'Alt':[]}
    count = 0
    while count != num_points:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            df['Lat'].append(random_point.y)
            df['Lon'].append(random_point.x)
            df['Alt'].append(0)
            count += 1
    return pd.DataFrame(df)

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