from IPython.display import clear_output

from deap import base
from deap import creator
from deap import tools

from STK_Sim import *

Filename = 'AERO_401_Initial_Assessment'

stk_object = STK_Simulation(False,Filename)

# Setting scenario time variables
stk_object.root.UnitPreferences.SetCurrentUnit("DateFormat", "UTCG")
start_time = time_convert(stk_object.root.CurrentScenario.StartTime)
dt = datetime.timedelta(days=5, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
stop_time=(start_time+dt).strftime("%d %b %Y %H:%M:%S.%f")
stk_object.root.CurrentScenario.StopTime=stop_time
stk_object.root.UnitPreferences.SetCurrentUnit("DateFormat", "EpSec")

# All of these variables substantially change calculation time
dt = 60
n_targets = 15
n_sats = 12
n_pop = 10
n_gen = 10

# Generating Targets

# Generating a polygon to bound the lat/lon coordinates, you can create your polygon, in the same format as Targets_Polygon.txt.
poly = Create_Poly('Input_Files/Targets_Polygon.txt')

# Writing random points within the polygon to a target file.
targets_filename = 'Input_Files/Targets_File.txt'
polygon_random_points(poly,n_targets).to_csv(targets_filename,index=False)

# Loading targets into stk from file.
targets_filename = 'Input_Files/Targets_File.txt'
stk_object.Target_Loader(targets_filename)

# Running Optimization

# Creating DEAP optimization model (positive weights to maximize, negative weights to minimize)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Creating satellite for the model
creator.create("Satellite", list, fitness=creator.FitnessMax)

# DEAP Cost Function
def satellite_performance(Individual):
    file = open("Input_Files/Satellites_File.txt","w")
    file.write("Per,Apo,Inc,AoP,Asc,Loc,Tar,Aft\n")
    nvars = 5
    for n in range(len(Individual)//nvars):
        Alt = Individual[nvars*n]
        Inc = Individual[nvars*n+1]
        AoP = Individual[nvars*n+2]
        Asc = Individual[nvars*n+3]
        Loc = Individual[nvars*n+4]
        Tar = 1
        file.write(f"{Alt},{Alt},{Inc},{AoP},{Asc},{Loc},{Tar}\n")
    file.close()
    satellites_filename = 'Input_Files/Satellites_File.txt'
    stk_object.Satellite_Loader(satellites_filename)
    stk_object.Compute_AzEl(dt)
    num_total_angles = 0
    for t in range(len(stk_object.targets)):
        num_total_angles += len(np.where(stk_object.Azimuth_vs_Elevation[f"Target{t+1}"]>0)[0])
    return 100*num_total_angles/324/len(stk_object.targets)

# Lower and Upper Bounds of Variables
lower = [500,0,0,0,0]
upper = [615,180,180,180,180]

# Registering variables to the satellite
toolbox = base.Toolbox()
toolbox.register("attr_alt", random.uniform, lower[0], upper[0])
toolbox.register("attr_inc", random.uniform, lower[1], upper[1])
toolbox.register("attr_aop", random.uniform, lower[2], upper[2])
toolbox.register("attr_asc", random.uniform, lower[3], upper[3])
toolbox.register("attr_loc", random.uniform, lower[4], upper[4])

# Registering satellite to the model
toolbox.register("satellite", tools.initCycle, creator.Satellite,
                 (toolbox.attr_alt,
                  toolbox.attr_inc,
                  toolbox.attr_aop,
                  toolbox.attr_asc,
                  toolbox.attr_loc), n=n_sats)

# Registering tools for the algorithm
toolbox.register("population", tools.initRepeat, list, toolbox.satellite)
toolbox.register("evaluate", satellite_performance)
# toolbox.register("mate", tools.cxBlend, alpha=0.1)
toolbox.register("mate", tools.cxSimulatedBinaryBounded,eta=0.25,low=n_sats*lower,up=n_sats*upper)
# toolbox.register("mutate", gp.mutEphemeral,mode="one")
toolbox.register("mutate", tools.mutPolynomialBounded,eta=0.25,low=n_sats*lower,up=n_sats*upper,indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)

clear_output(wait=True)

g = 0

# Creating a population to evolve
pop = toolbox.population(n=n_pop)
print("-- Generation %i --" % g)
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values += (fit,)
CXPB, MUTPB = 0.6, 0.3
fits = [ind.fitness.values[0] for ind in pop]
hof = tools.HallOfFame(5)

# Begin the evolution
while max(fits) < 100 and g < n_gen:
    clear_output(wait=True)
    g = g + 1
    print("-- Generation %i --" % g)
    # A new generation
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))
    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values += (fit,)
    pop[:] = offspring
    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    hof.update(pop)

# Picking the best run to find specifics

Individual = hof[0]
file = open("Input_Files/Satellites_File.txt","w")
file.write("Per,Apo,Inc,AoP,Asc,Loc,Tar\n")
nvars = 5
for n in range(len(Individual)//nvars):
    Alt = Individual[nvars*n]
    Inc = Individual[nvars*n+1]
    AoP = Individual[nvars*n+2]
    Asc = Individual[nvars*n+3]
    Loc = Individual[nvars*n+4]
    Tar = 1
    file.write(f"{Alt},{Alt},{Inc},{AoP},{Asc},{Loc},{Tar}\n")

file.close()

labels = "Alt,Inc,AoP,Asc,Loc".split(",")
df = {}
for s in range(n_sats):
    df[f'Satellite{s+1}'] = hof[0][5*s:5*(s+1)]
df = pd.DataFrame(df)
df.index = "Alt,Inc,AoP,Asc,Loc".split(",")
print(df)
print("\nfitness=",hof[0].fitness.values[0])

stk_object.Satellite_Loader("Input_Files/Satellites_File.txt")

stk_object.Compute_AzEl(dt)
stk_object.Compute_Time_Sorted_Data()

# Visualization of Targets Azimuth and Elevation data (includes all satellites).

figs1 = []
figs2 = []
buttons1 = []
buttons2 = []

fig3 = go.Figure()

for t in range(len(stk_object.targets)):
    fig1 = px.imshow(stk_object.Azimuth_vs_Elevation[f"Target{t+1}"].transpose(), text_auto=True,
                labels=dict(x="Azimuth",y="Elevation", color="Total Frames Captured"),)
    figs1.append(fig1.data[0])
    for s in range(len(stk_object.sensors)):
        df = stk_object.AzEl_data[f'Target{t+1}->Satellite{s+1}']
        if type(df) != int:
            df['Azimuth'] = df['Azimuth'].astype(float)
            df['Elevation'] = df['Elevation'].astype(float)
            fig2 = go.Scatter(y=df['Elevation'],x=df['Azimuth'],name=f'{t+1}->{s+1}',mode='lines')
            figs2.append(fig2)
        else:
            figs2.append(go.Scatter(x=[],y=[]))

    if type(stk_object.time_sorted_data[f'Target{t+1}']) != int:
        df = pd.DataFrame(stk_object.time_sorted_data[f'Target{t+1}'])
        fig3.add_trace(go.Scatter(x=df['Time'],y=df['Percent Imaged'],name=f'Target{t+1}'))

    z1 = np.zeros(len(stk_object.targets),dtype=bool);z1[t]=1
    z2 = np.zeros(len(stk_object.targets)*len(stk_object.sensors),dtype=bool)
    z2[len(stk_object.sensors)*t] = 1
    for s in range(len(stk_object.sensors)):
        z2[len(stk_object.sensors)*t+s]=1
    buttons1.append(dict(label=f"Target{t+1}",
                        method="restyle",
                        args=[{"visible": z1},]))
    buttons2.append(dict(label=f"Target{t+1}",
                        method="restyle",
                        args=[{"visible": z2},]))

fig1=go.Figure(figs1)
fig1.update_layout(xaxis_title="Azimuth", yaxis_title="Elevation",
    updatemenus=[
        dict(
            active=0,
            buttons=buttons1,
              x=1.1,
              xanchor="left",
              y=1.1,
              yanchor="top")])

fig2=go.Figure(figs2)
fig2.update_layout(xaxis_title="Azimuth", yaxis_title="Elevation",
    updatemenus=[
        dict(
            active=0,
            buttons=buttons2,
              x=1.1,
              xanchor="left",
              y=1.1,
              yanchor="top")])

fig3.update_layout(xaxis_title="Time", yaxis_title="% Imaged")

# fig1.show()
# fig2.show()
# fig3.show()

f1 = PlotlyViewer(fig1)
f2 = PlotlyViewer(fig2)
f3 = PlotlyViewer(fig3)

stk_object.root.CloseScenario()

STKDesktop.ReleaseAll()