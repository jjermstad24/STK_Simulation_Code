optimize = input('Run new optimization? (T/F): ')
if optimize == 'T':
    n_pop = int(input('n_pop: '))
    n_gen = int(input('n_gen: '))

duration = int(input('duration (days): '))

import sys
sys.path.append("../../")
from STK_Sim import *

print("Booting STK")
stk_object = STK_Simulation(False)
print("Loaded STK")

stk_object.set_sim_time(days=duration)

if optimize == 'T':
    opt = Optimizer2(stk_object,n_pop,n_gen)
    stk_object.Target_Loader("../../Input_Files/Target_Packages/Targets_65.txt")
    print("Beginning Optimization")
    with open(f"../../Pop_Over_Gen/pop_gen.csv","w") as f:
        hof = opt.run(read=False,enable_print=True)

    send_message_to_discord(message=f'Optimization Complete')
    opt.Load_Individual(hof[0])