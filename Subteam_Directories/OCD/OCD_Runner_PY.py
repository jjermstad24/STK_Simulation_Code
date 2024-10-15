optimize = input('Run new optimization? (T/F): ')
if optimize == 'T':
    n_pop = int(input('n_pop: '))
    n_gen = int(input('n_gen: '))
    
n_sats = int(input('n_sats: '))
duration = int(input('duration (days): '))


import sys
sys.path.append("../../")
from STK_Sim import *


print("Booting STK")
stk_object = STK_Simulation(False)
print("Loaded STK")

stk_object.set_sim_time(days=duration)

if optimize == 'T':
    opt = Optimizer(stk_object,n_pop,n_gen,n_sats)
    stk_object.Target_Loader("../../Input_Files/Target_Packages/Targets_15.txt")
    percentage = 0
    while percentage != 100:

        pd.read_csv(f"../../Input_Files/Constellations/{n_sats}.txt").to_csv(f"../../Input_Files/Satellites_File_{n_sats}.txt",index=False)
        print("Beginning Optimization")
        with open(f"../../Pop_Over_Gen/{n_sats}.csv","w") as f:
            hof = opt.run(read=True,enable_print=True,file=f)

        send_message_to_discord(message=f'{n_sats} Optimization Complete')
        opt.Load_Individual(hof[0])

        df = pd.read_csv(f"../../Pop_Over_Gen/{n_sats}.csv")
        percentage = (df[(df['Alt'] == hof[0][0]) & (df['Inc'] == hof[0][1]) & (df['Initial_Raan'] == hof[0][2]) & (df['Delta_Raan'] == hof[0][3]) & (df['Num_Planes'] == hof[0][4])].iloc[0]['Avg_Percentage'])

        pd.read_csv(f"../../Input_Files/Satellites_File_{n_sats}.txt").to_csv(f"../../Input_Files/Constellations/{n_sats}.txt",index=False)
    

stk_object.Satellite_Loader(f"../../Input_Files/Constellations/{n_sats}.txt")


discord_message = [[f'Targets', 'Avg Unplanned %', 'Max Unplanned Time']]

for n_targets in [15,34,65,82,109,186,494,1131]:
    stk_object.Target_Loader(f"../../Input_Files/Target_Packages/Targets_{n_targets}.txt")
    stk_object.Results_Runner(Plan=False)
    stk_object.Create_Data_Comparison_df(Unplanned=True, Planned=False)

    data_comparison = pd.DataFrame(stk_object.data_comparison)
    data_comparison.to_csv(f"../../Optimization_Results/{n_sats}_{n_targets}.csv",index=False)
    discord_message.append([data_comparison["Unplanned (%)"].count(), data_comparison["Unplanned (%)"].mean(), data_comparison["Unplanned (Time)"].max()])


discord_message = pd.DataFrame(discord_message, columns=discord_message[0]).drop(0)
send_message_to_discord(message=f'***{n_sats}***\n\n{discord_message}')