import sys
sys.path.append("../../")
from STK_Sim import *

Filename = 'AERO_402_Further_Assessment'

print("Booting STK")
stk_object = STK_Simulation(False,Filename)
print("Loaded STK")

n_pop = 25
n_gen = 5
n_sats = 6

opt = Optimizer(stk_object,n_pop,n_gen,n_sats)

stk_object.set_sim_time(days=5)
stk_object.Target_Loader("../../Input_Files/Target_Packages/Targets_34.txt")

pd.read_csv(f"../../Input_Files/Constellations/{n_sats}.txt").to_csv(f"../../Input_Files/Satellites_File_{n_sats}.txt",index=False)

print("Beginning Optimization")
with open(f"../../Pop_Over_Gen/{n_sats}.csv","w") as f:
    hof = opt.run(read=True,enable_print=True,file=f)

send_message_to_discord(message=f'{n_sats} Optimization Complete',channel_id=channel_id,bot_token=bot_token)

opt.Load_Individual(hof[0])

stk_object.set_sim_time(days=45)
stk_object.dt = 60

discord_message = [[f'Targets', 'Avg Unplanned %', 'Max Unplanned Time']]

for n_targets in [15,34,65,82,109,186,494]:
    stk_object.Target_Loader(f"../../Input_Files/Target_Packages/Targets_{n_targets}.txt")
    stk_object.Results_Runner(Plan=False)
    stk_object.Create_Data_Comparison_df(Unplanned=True, Planned=False)

    data_comparison = pd.DataFrame(stk_object.data_comparison)
    data_comparison.to_csv(f"../../Optimization_Results/{n_sats}_{n_targets}.csv",index=False)
    discord_message.append([data_comparison["Unplanned (%)"].count(), data_comparison["Unplanned (%)"].mean(), data_comparison["Unplanned (Time)"].max()])


discord_message = pd.DataFrame(discord_message, columns=discord_message[0]).drop(0)
send_message_to_discord(message=f'***{n_sats}***\n\n{discord_message}',channel_id=channel_id,bot_token=bot_token)