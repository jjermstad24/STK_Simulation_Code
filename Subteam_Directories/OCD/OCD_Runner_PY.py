import sys
sys.path.append("../../")
from STK_Sim import *

Filename = 'AERO_402_Further_Assessment'

stk_object = STK_Simulation(False,Filename)

n_pop = int(input('n_pop = '))
n_gen = int(input('n_gen = '))
n_sats = int(input('n_sats = '))

comp_dt = float(input('Compuation dt [secs] = '))
comp_duration = int(input('Computation duration [days] = '))

if input('Create New Targets? (True/False) ') == 'True':
    poly = Create_Poly("../../Input_Files/Targets_File.txt")
    polygon_random_points(poly,int(input('n_targets = ')),)

stk_object.Target_Loader("../../Input_Files/Targets_File.txt")

if input('New Optimization? (True/False) ') == 'True':
    stk_object.dt = float(input('Optimization dt [secs] = '))
    duration = datetime.timedelta(days=int(input('Optimization duration [days] = ')), seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)

    stk_object.root.UnitPreferences.SetCurrentUnit("DateFormat", "UTCG")
    start_time = time_convert(stk_object.root.CurrentScenario.StartTime)
    stop_time=(start_time+duration).strftime("%d %b %Y %H:%M:%S.%f")
    stk_object.root.CurrentScenario.StopTime=stop_time
    stk_object.root.UnitPreferences.SetCurrentUnit("DateFormat", "EpSec")
    
    opt = Optimizer(stk_object,n_pop,n_gen,n_sats,weights=(7,-8,-1,-2))
    stk_object.Interpolate=True
    starting_ind = get_ind()                                # Gets individual associating with ../../Input_Files/Satellite_File.txt
    hof,percent,std,time,sats = opt.run(starting_ind)       # Run optimization putting individual into the initial population
    opt.cost_function(hof[0])  


stk_object.Satellite_Loader("../../Input_Files/Satellites_File.txt")

stk_object.dt = comp_dt
duration = datetime.timedelta(days=comp_duration, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
stk_object.root.UnitPreferences.SetCurrentUnit("DateFormat", "UTCG")
start_time = time_convert(stk_object.root.CurrentScenario.StartTime)
stop_time=(start_time+duration).strftime("%d %b %Y %H:%M:%S.%f")
stk_object.root.CurrentScenario.StopTime=stop_time
stk_object.root.UnitPreferences.SetCurrentUnit("DateFormat", "EpSec")

stk_object.Generate_Planning_Data()

stk_object.Plan_Data(slew_rate=1)

data_comparison = {}
data_comparison["Unplanned (%)"] = [len(np.unique(stk_object.Planning_Data[stk_object.Planning_Data['Target'].values==tar_num]['Bin Number'].values))/324*100 for tar_num in range(len(stk_object.targets))]
data_comparison["Planned (%)"] = [len(np.unique(stk_object.Planned_Data[stk_object.Planned_Data['Target'].values==tar_num]['Bin Number'].values))/324*100 for tar_num in range(len(stk_object.targets))]
data_comparison["Unplanned (Time)"] = [max(stk_object.Planning_Data[stk_object.Planning_Data['Target'].values==tar_num]['Time'].values/86400) for tar_num in range(len(stk_object.targets))]
data_comparison["Planned (Time)"] = [max(stk_object.Planned_Data[stk_object.Planned_Data['Target'].values==tar_num]['Time'].values/86400) for tar_num in range(len(stk_object.targets))]

data_comparison = pd.DataFrame(data_comparison)
print(data_comparison)
print(data_comparison.describe())