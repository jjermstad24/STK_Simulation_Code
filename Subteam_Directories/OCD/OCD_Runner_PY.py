# Loading targets into stk from file.

import sys
sys.path.append("../../")
from STK_Sim import *

Filename = 'AERO_402_Further_Assessment'

stk_object = STK_Simulation(False,Filename)
stk_object.set_sim_time(days=0, hours=1)
stk_object.Target_Loader("../../Input_Files/Target_Packages/Targets_15.txt")

opt = MultiObjectiveOptimizer(stk_object,n_pop=10,n_gen=10)
print("Beginning Optimization")
opt.run(read=False,enable_print=False)

try:
    opt = MultiObjectiveOptimizer(stk_object,n_pop=100,n_gen=10)
    print("Beginning Optimization")
    opt.run(read=False,enable_print=True)
    send_message_to_discord('Optimization Done')
except Exception as e:
    error_type = type(e).__name__  # Get the error type
    send_message_to_discord(f"Optimization Failed with error: {error_type}")
    print(f"Optimization Failed with error: {error_type}")  # For debugging