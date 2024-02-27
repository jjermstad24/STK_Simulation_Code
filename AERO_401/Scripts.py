import random
import decimal
import numpy as np
import pandas as pd
import datetime

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