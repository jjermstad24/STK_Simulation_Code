import sys
sys.path.append("../../")
from STK_Sim import *

Filename = 'AERO_402_Further_Assessment'

stk_object = STK_Simulation(False,Filename)

n_sats = int(input('n_sats = '))

comp_dt = float(input('Compuation dt [secs] = '))
comp_duration = int(input('Computation duration [days] = '))

n_targets= int(input('Number of Targets? (15, 34, 62, 82, 109, 186, 494, 1131) '))
opt_bool = bool(input('New Optimization? (True/False) ') == 'True')

if opt_bool:
    n_pop = int(input('n_pop = '))
    n_gen = int(input('n_gen = '))
    opt_dt = float(input('Optimization dt [secs] = '))
    opt_duration = int(input('Optimization duration [days] = '))

    initial_entry = bool(input('Start with initial constellation? (True/False) ') == 'True')

stk_object.Target_Loader(f"../../Input_Files/Targets_{n_targets}.txt")

if opt_bool:

    stk_object.dt = opt_dt
    duration = datetime.timedelta(days=opt_duration, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)

    stk_object.root.UnitPreferences.SetCurrentUnit("DateFormat", "UTCG")
    start_time = time_convert(stk_object.root.CurrentScenario.StartTime)
    stop_time=(start_time+duration).strftime("%d %b %Y %H:%M:%S.%f")
    stk_object.root.CurrentScenario.StopTime=stop_time
    stk_object.root.UnitPreferences.SetCurrentUnit("DateFormat", "EpSec")
    
    opt = Optimizer(stk_object,n_pop,n_gen,n_sats,weights=(7,-5,-1))
    stk_object.Interpolate=True
    

    hof,percent,std,time = opt.run(read=True)

    print("Hall of Fame:")
    for entry in hof:
        print(entry,"->",entry.fitness)

    # Get best individual from optimization and load them into ../../Input_Files/Satellite_File.txt

    best_stats = opt.cost_function(hof[0])

else:
    stk_object.Satellite_Loader("../../Input_Files/Satellites_File.txt")


stk_object.dt = comp_dt
duration = datetime.timedelta(days=comp_duration, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
stk_object.root.UnitPreferences.SetCurrentUnit("DateFormat", "UTCG")
start_time = time_convert(stk_object.root.CurrentScenario.StartTime)
stop_time=(start_time+duration).strftime("%d %b %Y %H:%M:%S.%f")
stk_object.root.CurrentScenario.StopTime=stop_time
stk_object.root.UnitPreferences.SetCurrentUnit("DateFormat", "EpSec")

stk_object.Generate_Holding_Data()

stk_object.Plan(slew_rate=1,cone_angle=20)

data_comparison = {}
data_comparison["Unplanned (%)"] = [len(np.unique(stk_object.Holding_Data[stk_object.Holding_Data['Target'].values==tar_num]['Bin Number'].values))/324*100 for tar_num in range(len(stk_object.targets))]
data_comparison["Planned (%)"] = [len(np.unique(stk_object.Planned_Data[stk_object.Planned_Data['Target'].values==tar_num]['Bin Number'].values))/324*100 for tar_num in range(len(stk_object.targets))]
data_comparison["Planned (Time)"] = [max(stk_object.Planned_Data[stk_object.Planned_Data['Target'].values==tar_num]['Time'].values)/86400 for tar_num in range(len(stk_object.targets))]

data_comparison = pd.DataFrame(data_comparison)
print(data_comparison)
print(data_comparison.describe())


dfs_to_excel(r"H:/Shared drives/AERO 401 Project  L3Harris Team 1/Subteam Designs/OCD/Superstars.xlsx", str(n_sats), df1=pd.read_csv("../../Input_Files/Satellites_File.txt"), df2=data_comparison)