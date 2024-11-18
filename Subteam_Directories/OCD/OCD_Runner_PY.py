# Loading targets into stk from file.

import sys
sys.path.append("../../")
from STK_Sim import *
from Optimizers import *

Filename = 'AERO_402_Further_Assessment'

stk_object = STK_Simulation(False,Filename)

stk_object.opt = False

stk_object.dt = 5

stk_object.set_sim_time(days=30)

idx = 0

for tar_num in [105,182,481]:
    result = Update_Pareto_Performance(stk_object,idx,tar_num)