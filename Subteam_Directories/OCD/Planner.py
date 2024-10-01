initial_conditions = [int(input('n_sats: ')),
                      int(input('n_targets (15, 34, 65, 82, 109, 186, 494, 1131): ')),
                      float(input('dt: ')),
                      int(input('duration: '))]

n_sats, n_targets, dt, duration = initial_conditions

import sys
sys.path.append("../../")
from STK_Sim import *

initial_conditions = pd.DataFrame(initial_conditions)

Filename = 'AERO_402_Further_Assessment'

stk_object = STK_Simulation(False,Filename)

stk_object.Satellite_Loader(f"../../Input_Files/Constellations/{n_sats}.txt")
stk_object.Target_Loader(f"../../Input_Files/Target_Packages/Targets_{n_targets}.txt") 

stk_object.dt = dt
stk_object.Set_Duration(duration)
stk_object.Generate_Holding_Data()
stk_object.Plan(slew_rate=1,cone_angle=20)

stk_object.Create_Data_Comparison_df(Planned=True, Unplanned=True)

print(stk_object.data_comparison)
print(pd.DataFrame(stk_object.data_comparison.describe()))

dfs_to_excel(r"H:/Shared drives/AERO 401 Project  L3Harris Team 1/Subteam Designs/OCD/Superstars.xlsx", f'{n_sats} - {n_targets} - {duration}', df1=pd.read_csv(f"../../Input_Files/Constellations/{n_sats}.txt"), df2=stk_object.data_comparison, df3=initial_conditions)