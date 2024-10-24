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

try:
    opt = MultiObjectiveOptimizer(stk_object,n_pop=10,n_gen=10,historical_df=historical_df,cost_df=cost_df)
    print("Beginning Optimization")
    opt.run(read=False,enable_print=False)
    send_message_to_discord('Optimization Done')
except Exception as e:
    error_type = type(e).__name__  # Get the error type
    send_message_to_discord(f"Optimization Failed with error: {error_type}")
    print(f"Optimization Failed with error: {error_type}")  # For debugging