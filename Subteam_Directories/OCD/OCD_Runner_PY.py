# Loading targets into stk from file.

import sys
sys.path.append("../../")
from STK_Sim import *

Filename = 'AERO_402_Further_Assessment'

stk_object = STK_Simulation(False,Filename)
stk_object.set_sim_time(days=3)
stk_object.Target_Loader("../../Input_Files/Target_Packages/Targets_15.txt")



# for run_num,weight in enumerate(weights[0:3]):
#     per_weight = .5
#     time_weight = weight
#     cost_weight = 1-time_weight-per_weight

per_weight = .5
time_weight = .25
cost_weight = 1-time_weight-per_weight
opt = Optimizer(stk_object,n_pop=2,n_gen=1,run_num=0,weights=(per_weight,-time_weight,-cost_weight))
print("Beginning Optimization")
opt.run(read=False,enable_print=True)

# send_message_to_discord('Optimization Done')

