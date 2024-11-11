# Loading targets into stk from file.

import sys
sys.path.append("../../")
from STK_Sim import *
from Optimizers import *

Filename = 'AERO_402_Further_Assessment'

stk_object = STK_Simulation(False,Filename)
stk_object.set_sim_time(days=1)

for idx in [0,1,2]:
    Update_Pareto_Performance(stk_object,idx,tar_list=[15,34,65,82,109,186])