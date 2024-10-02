print('\nPlease input the desired initial conditions for optimization: \n')
initial_conditions = [
                    int(input('n_sats_min: ')),
                    int(input('n_sats_max: ')),
                    int(input('n_pop: ')),
                    int(input('n_gen: ')),
                    float(input('opt_dt: ')),
                    int(input('opt_duration: ')),
                    float(input('comp_dt: ')),
                    int(input('comp_duration: '))]

n_sats_min, n_sats_max, n_pop, n_gen, opt_dt, opt_duration, comp_dt, comp_duration = initial_conditions
n_targets = 34

with open('Optimizer_Results.txt', 'w') as file:
    local_vars = list(locals().copy().items())
    file.write('Initial Conditions:\n\n')
    for var_name, value in local_vars[10:-1]:
        file.write(f"{var_name}: {value}\n")

import sys
sys.path.append("../../")
from STK_Sim import *


with alive_bar(3,force_tty=True,bar='classic',title='- Setting Up STK_Scenario...',length=3) as bar:
    Filename = 'AERO_402_Further_Assessment'
    bar()
    stk_object = STK_Simulation(False,Filename)
    bar()
    stk_object.Target_Loader(f"../../Input_Files/Target_Packages/Targets_{n_targets}.txt")
    stk_object.dt = opt_dt
    stk_object.Set_Duration(opt_duration)
    bar()

for n_sats in range(n_sats_min,n_sats_max+1):
    os.system('cls')
    print(f'{n_sats} sats:\n')

    lower_bounds = [580,40,0,0,0,1] # alt, inc, AOP, initial_raan, delta_raan, n_planes
    upper_bounds = [630,150,0,180,90,n_sats]

    opt = Optimizer(stk_object,n_pop,n_gen,n_sats,weights=(7,-5, -1),lower_bounds=lower_bounds, upper_bounds=upper_bounds)
    stk_object.Interpolate=True
    res = opt.run(read=False)
    
    stk_object.dt = comp_dt
    stk_object.Set_Duration(comp_duration)
    best_stats = opt.cost_function(res[0][0])

    with open('Optimizer_Results.txt', 'a') as file:
        file.write(f'\n{n_sats} sats: \n{best_stats}')

    shutil.copy("../../Input_Files/Satellites_File.txt", f"../../Input_Files/Constellations/{n_sats}.txt")

