# Loading targets into stk from file.

import sys
sys.path.append("../../")
from STK_Sim import *
from Optimizers import *

Filename = 'AERO_402_Further_Assessment'

stk_object = STK_Simulation(False,Filename)
stk_object.set_sim_time(days=30)
stk_object.Target_Loader("../../Input_Files/Target_Packages/Targets_65.txt")

cost_df = pd.read_csv('../../Input_Files/Cost.csv').set_index('System')
historical_df = pd.read_csv('../../Pop_Over_Gen/historical.csv')

num_planes = 3

bot_token = 2

try:
    opt = MultiObjectiveOptimizer(stk_object,n_pop=25,n_gen=10,historical_df=historical_df,cost_df=cost_df)
    opt.lower[5] = num_planes
    opt.upper[5] = num_planes
    opt.lower[4] = num_planes
    print("Beginning Optimization")
    opt.run(read=True,enable_print=True)
    send_message_to_discord('Optimization Done', bot_token=bot_token)
except Exception as e:
    error_type = type(e).__name__  # Get the error type
    send_message_to_discord(f"Optimization Failed with error: {error_type}",bot_token=bot_token)
    print(f"Optimization Failed with error: {error_type}")  # For debugging