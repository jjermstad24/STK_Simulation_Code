# Loading targets into stk from file.

import sys
sys.path.append("../../")
from STK_Sim import *

Filename = 'AERO_402_Further_Assessment'

stk_object = STK_Simulation(False,Filename)


stk_object.set_sim_time(days=15)
stk_object.Target_Loader("../../Input_Files/Target_Packages/Targets_65.txt")

class Optimizer:
    def __init__(self, stk_object, n_pop, n_gen, run_num,weights=(7.0,2.0,-4.0)):
        self.run_num = run_num
        self.weights = weights
        self.stk_object = stk_object
        self.n_pop = n_pop
        self.n_gen = n_gen
        creator.create("FitnessMax", base.Fitness, weights=weights)
        creator.create("Satellite", list, fitness=creator.FitnessMax)
        self.lower = [575, 80, 30, 0, 3, 1]
        self.upper = [630, 100, 150, 30, 12, 12]

        
        
        if self.stk_object.duration.total_seconds() < 86400:
            self.per_unit_time = 3600
            self.time_unit = 'hours'
        else:
            self.per_unit_time = 86400
            self.time_unit = 'days'

        self.norm_array = np.array([100,self.stk_object.duration.total_seconds()/self.per_unit_time,24])
        
        
        # Registering variables to the satellite
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_alt", random.randint, self.lower[0], self.upper[0])
        self.toolbox.register("attr_inc", random.randint, self.lower[1], self.upper[1])
        self.toolbox.register("attr_initial_raan", random.randint, self.lower[2], self.upper[2])
        self.toolbox.register("attr_delta_raan", random.randint, self.lower[3], self.upper[3])
        self.toolbox.register("attr_num_sats", random.randint, self.lower[4], self.upper[4])
        self.toolbox.register("attr_num_planes", random.randint, self.lower[5], self.upper[5])

        def satellite_init():
            # Normalized variables between 0 and 1
            alt_norm = random.random()  # Normalized altitude
            inc_norm = random.random()  # Normalized inclination
            raan_norm = random.random()  # Normalized RAAN
            delta_raan_norm = random.random()  # Normalized Delta RAAN
            num_sats_norm = random.random()  # Normalized number of satellites
            num_planes_norm = random.random()  # Normalized number of planes
            
            # Denormalize to original ranges
            alt = int(alt_norm * (self.upper[0] - self.lower[0]) + self.lower[0])
            inc = int(inc_norm * (self.upper[1] - self.lower[1]) + self.lower[1])
            initial_raan = int(raan_norm * (self.upper[2] - self.lower[2]) + self.lower[2])
            delta_raan = int(delta_raan_norm * (self.upper[3] - self.lower[3]) + self.lower[3])
            num_sats = int(num_sats_norm * (self.upper[4] - self.lower[4]) + self.lower[4])
            num_planes = min(int(num_planes_norm * (self.upper[5] - self.lower[5]) + self.lower[5]), num_sats)


            return creator.Satellite([alt, inc, initial_raan, delta_raan, num_sats, num_planes])

        self.toolbox.register("satellite", satellite_init)

        # Registering tools for the algorithm

        def mutate_individual(mutant):
            # Apply the registered mutation operator directly
            tools.mutUniformInt(mutant, low=self.lower, up=self.upper, indpb=0.8)
            
            # Enforce the constraint after mutation
            if mutant[4] < mutant[5]:  # if n_sats < n_planes
                mutant[5] = mutant[4]  # Set n_planes = n_sats

            return mutant
        
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.satellite)
        self.toolbox.register("evaluate", self.cost_function)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=int(self.n_pop // 2))
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)


    def run(self,read=False,enable_print=False):
        self.enable_print = enable_print
        self.fits = []
        CXPB = 0.7;MUTPB=0.3
        self.g = 0
        # Creating a population to evolve
        pop = self.toolbox.population(n=self.n_pop)
        i = np.random.randint(0,self.n_pop)
        if read:
            for idx in range(4):
                pop[i][idx] = get_ind(4)[idx]

        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        hof = tools.HallOfFame(5)
        hof.update(pop)

        self.fits.append([ind.fitness.values for ind in pop])
        record = self.stats.compile(pop)

        clear_output(wait=False)
        print("-- Generation %i --" % self.g)
        print(pd.DataFrame(record))

        if self.run_num == 0:
            w = 'w'
        else:
            w = 'a'
        with open(f"../../Pop_Over_Gen/pop_gen.csv",f"{w}") as file:
            if self.run_num == 0:
                file.write(f"Run_Num,Per_Weight,Time_Weight,Cost_Weight,Gen,Pop,Alt,Inc,Initial_Raan,Delta_Raan,Num_Sats,Num_Planes,Avg_Percentage,Avg_Time [{self.time_unit}],Cost,\n")
            
            for i in range(self.n_pop):
                file.write(f'{self.run_num},')
                for weight in self.weights:
                    file.write(f"{weight},")
                file.write(f'{self.g},{i},')
                for idx in range(6):
                    file.write(f"{pop[i][idx]},")
                for fit in pop[i].fitness.getValues()*self.norm_array:
                    file.write(f"{fit},")
                file.write("\n")
        # Begin the evolution
        while self.g < self.n_gen:
            # clear_output(wait=True)
            self.g = self.g + 1

            # A new generation
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            pop[:] = offspring
            hof.update(pop)

            self.fits.append([ind.fitness.values for ind in pop])
            record = self.stats.compile(pop)

            clear_output(wait=False)
            print("-- Generation %i --" % self.g)
            print(pd.DataFrame(record))
            
            with open(f"../../Pop_Over_Gen/pop_gen.csv","a") as file:
                
                for i in range(self.n_pop):
                    file.write(f'{self.run_num},')
                    for weight in self.weights:
                        file.write(f"{weight},")
                    file.write(f'{self.g},{i},')
                    for idx in range(6):
                        file.write(f"{pop[i][idx]},")
                    for fit in pop[i].fitness.getValues()*self.norm_array:
                        file.write(f"{fit},")
                    file.write("\n")
        return hof
    
    def cost_function(self,Individual=[0,0,0,0,0,0],write=True):
        n_planes = Individual[5]
        n_sats = Individual[4]

        gen = self.g

        if write:
            self.Load_Individual(Individual)
        self.stk_object.Satellite_Loader(f'../../Input_Files/Satellites_File.txt')
        self.stk_object.Compute_AzEl(self.enable_print)
        percentages = np.array([np.count_nonzero(self.stk_object.target_bins[idx])/324 for idx in range(len(self.stk_object.targets))])
        times = np.array([self.stk_object.target_times[idx]/self.per_unit_time for idx in range(len(self.stk_object.targets))])
        
        percentage = np.average(percentages)
        avg_tme = np.average(times)

        penalty = 0
        if n_planes > n_sats:
            penalty = 10 * (n_planes - n_sats) 

        if percentage < 1:
            # if gen >= 1:
            #     percentage = percentage - (gen)/500
            avg_tme = self.norm_array[1]

        fitness = [percentage, avg_tme/self.norm_array[1], (n_sats + n_planes)/24 + penalty]

        return tuple(fitness)
    
    def Load_Individual(self,Individual=[0,0,0,0,0,0]):
        Alt = Individual[0]
        Inc = Individual[1]
        initial_raan = Individual[2]
        delta_raan = Individual[3]
        n_planes = int(Individual[5])
        n_sats = int(Individual[4])
        
        if n_planes > n_sats:
            n_planes = n_sats

        Asc = initial_raan
        file = open(f"../../Input_Files/Satellites_File.txt","w")
        file.write("Per,Apo,Inc,Asc,Loc\n")
        sats = n_sats*[1]
        planes = np.array_split(sats,n_planes)
        i=1
        for plane in planes:
            Loc = 0
            for sat in plane:
                file.write(f"{Alt},{Alt},{Inc},{round(Asc%180,4)},{round(Loc,4)}\n")
                if len(plane)>1: Loc += 360/len(plane)
            if len(planes)>1:Asc -= i*((-1)**(i))*delta_raan
            i+=1
        file.close()


weights = np.linspace(.01,.49,15)
for run_num,weight in enumerate(weights[0:3]):
    per_weight = .5
    time_weight = weight
    cost_weight = 1-time_weight-per_weight
    opt = Optimizer(stk_object,n_pop=50,n_gen=5,run_num=run_num,weights=(per_weight,-time_weight,-cost_weight))
    print("Beginning Optimization")
    opt.run(read=False,enable_print=True)
    # hof = opt.run(read=False,enable_print=True)
    # pd.DataFrame([hof[0]], columns = 'Alt,Inc,Init_RAAN,Delta_RAAN,N_sats,N_planes'.split(','))

send_message_to_discord('Optimization Done')
