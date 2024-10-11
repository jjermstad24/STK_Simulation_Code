import sys
sys.path.append("../../")
from STK_Sim import *

Filename = 'AERO_402_Further_Assessment'

print("Booting STK")
stk_object = STK_Simulation(False,Filename)
print("Loaded STK")

n_pop = 25
n_gen = 10
n_sats = 3

opt = Optimizer(stk_object,n_pop,n_gen,n_sats)

stk_object.set_sim_time(days=2)
stk_object.Target_Loader("../../Input_Files/Target_Packages/Targets_15.txt")

pd.read_csv(f"../../Input_Files/Constellations/{n_sats}.txt").to_csv(f"../../Input_Files/Satellites_File_{n_sats}.txt",index=False)

print("Beginning Optimization")
with open(f"../../Pop_Over_Gen/{n_sats}.csv","w") as f:
    hof = opt.run(read=True,enable_print=True,file=f)

opt.Load_Individual(hof[0])

stk_object.set_sim_time(days=30)
stk_object.dt = 60

for n_targets in [15,34,65,82,109,186,494]:
    stk_object.Target_Loader(f"../../Input_Files/Target_Packages/Targets_{n_targets}.txt")
    stk_object.Generate_Pre_Planning_Data()
    data_comparison = {}
    data_comparison["Unplanned (%)"] = [np.count_nonzero(stk_object.target_bins[tar_num])/324*100 for tar_num in range(len(stk_object.targets))]
    data_comparison["Unplanned (Time)"] = [np.max(stk_object.target_times[tar_num])/86400 for tar_num in range(len(stk_object.targets))]
    pd.DataFrame(data_comparison).to_csv(f"../../Optimization_Results/{n_sats}_{n_targets}.csv",index=False)
    data_comparison = pd.DataFrame(data_comparison)
    send_message_to_discord(message='',channel_id=1203813613903675502)