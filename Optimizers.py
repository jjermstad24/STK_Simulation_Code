from Scripts import *

class Optimizer:
    def __init__(self, stk_object, n_pop, n_gen, run_num,weights=(.5,.25,.25)):
        self.run_num = run_num
        self.weights = weights
        self.stk_object = stk_object
        self.n_pop = n_pop
        self.n_gen = n_gen
        creator.create("FitnessMax", base.Fitness, weights=weights)
        creator.create("Satellite", list, fitness=creator.FitnessMax)
        self.lower = [575, 80, 30, 0, 3, 1]
        self.upper = [630, 100, 150, 30, 12, 12]

        self.norm_array = np.array([100,self.stk_object.duration.total_seconds()/86400,24])
        
        
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
                file.write(f"Run_Num,Per_Weight,Time_Weight,Cost_Weight,Gen,Pop,Alt,Inc,Initial_Raan,Delta_Raan,Num_Sats,Num_Planes,Avg_Percentage,Avg_Time,Cost,\n")
            
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
            Load_Individual(Individual)
        self.stk_object.Satellite_Loader(f'../../Input_Files/Satellites_File.txt')
        self.stk_object.Results_Runner()
        self.stk_object.Create_Data_Comparison_df(Unplanned=False)
        percentages = self.stk_object.data_comparison['Planned (%)']
        times = self.stk_object.data_comparison['Planned (Time)']

        percentage = np.average(percentages)
        time = np.average(times)

        penalty = 0
        if n_planes > n_sats:
            penalty = 10 * (n_planes - n_sats) 

        # if percentage < 1:
        #     # if gen >= 1:
        #     #     percentage = percentage - (gen)/500
        #     time = self.stk_object.durtation.total_seconds()

        objective = tuple(np.array([percentage, time, (n_sats + n_planes) + penalty])/self.norm_array)

        return objective

class MultiObjectiveOptimizer:
    def __init__(self, stk_object, n_pop, n_gen,historical_df,cost_df):
        self.cost_df = cost_df
        self.historical_df = historical_df
        self.stk_object = stk_object
        self.n_pop = n_pop
        self.n_gen = n_gen
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
        creator.create("Satellite", list, fitness=creator.FitnessMulti)
        self.lower = [575, 80, 30, 0, 3, 1]
        self.upper = [630, 100, 150, 30, 12, 12]
        self.norm_array = np.array([100, self.stk_object.duration.total_seconds() / 86400, self.prog_cost_function([0,0,0,0,12,12])])

        # Registering variables to the satellite
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_alt", random.randint, self.lower[0], self.upper[0])
        self.toolbox.register("attr_inc", random.randint, self.lower[1], self.upper[1])
        self.toolbox.register("attr_initial_raan", random.randint, self.lower[2], self.upper[2])
        self.toolbox.register("attr_delta_raan", random.randint, self.lower[3], self.upper[3])
        self.toolbox.register("attr_num_sats", random.randint, self.lower[4], self.upper[4])
        self.toolbox.register("attr_num_planes", random.randint, self.lower[5], self.upper[5])

        # Satellite initialization function
        def satellite_init():
            # Generate normalized variables between 0 and 1
            alt_norm = random.random()  
            inc_norm = random.random()  
            raan_norm = random.random()  
            delta_raan_norm = random.random()  
            num_sats_norm = random.random()  
            num_planes_norm = random.random()  

            # Denormalize to original ranges
            alt = int(alt_norm * (self.upper[0] - self.lower[0]) + self.lower[0])
            inc = int(inc_norm * (self.upper[1] - self.lower[1]) + self.lower[1])
            initial_raan = int(raan_norm * (self.upper[2] - self.lower[2]) + self.lower[2])
            delta_raan = int(delta_raan_norm * (self.upper[3] - self.lower[3]) + self.lower[3])
            num_sats = int(num_sats_norm * (self.upper[4] - self.lower[4]) + self.lower[4])
            num_planes = min(int(num_planes_norm * (self.upper[5] - self.lower[5]) + self.lower[5]), num_sats)

            return creator.Satellite([alt, inc, initial_raan, delta_raan, num_sats, num_planes])

        self.toolbox.register("satellite", satellite_init)

        # Use generator to save memory
        self.toolbox.register("population", tools.initRepeat, iter, self.toolbox.satellite)
        
        # Custom mutation
        def mutate_individual(mutant):
            tools.mutUniformInt(mutant, low=self.lower, up=self.upper, indpb=0.8)
            if mutant[4] < mutant[5]:  # Ensure num_planes <= num_sats
                mutant[5] = mutant[4]
            return mutant

        # Register evaluate as a method of the class
        self.toolbox.register("evaluate", self.lazy_evaluate)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", mutate_individual)
        self.toolbox.register("select", tools.selNSGA2)

        self.stats = None  # Disable statistics to reduce memory usage

    def lazy_evaluate(self, individual):
        if not individual.fitness.valid:
            individual.fitness.values = self.objective_function(individual)
        return individual.fitness.values

    def run(self, read=False, enable_print=False):
        self.enable_print = enable_print
        self.fits = []
        CXPB = 0.7; MUTPB = 0.3

        with alive_bar(self.n_gen+1, force_tty=True, bar='classic', title='- Optimizing', length=10,disable=enable_print) as bar:
            self.g = 0
            print("-- Generation %i --" % self.g)   
            # Creating a population to evolve
            pop = list(self.toolbox.population(n=self.n_pop))  # Ensure this is a list

            i = np.random.randint(0, self.n_pop)
            if read:
                for idx in range(4):
                    pop[i][idx] = get_ind(4)[idx]

            fitnesses = list(map(lambda ind: self.toolbox.evaluate(ind), pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            hof = tools.HallOfFame(1)
            hof.update(pop)

            self.fits.append([ind.fitness.values for ind in pop])
            record = self.stats.compile(pop) if self.stats else None


            if record is not None:
                print(pd.DataFrame(record))

            self.write_population_to_csv(pop,'w',output_file= f"../../Pop_Over_Gen/pop_gen.csv")
            self.write_population_to_csv(pop,'a',output_file= f"../../Pop_Over_Gen/historical.csv")

            bar()
            clear_output()
            gc.collect()
            # Begin the evolution
            while self.g < self.n_gen:
                self.g += 1
                print("-- Generation %i --" % self.g)

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
                fitnesses = list(map(lambda ind: self.toolbox.evaluate(ind), invalid_ind))

                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                    
                pop[:] = offspring
                hof.update(pop)

                self.fits.append([ind.fitness.values for ind in pop])
                record = self.stats.compile(pop) if self.stats else None


                if record is not None:
                    print(pd.DataFrame(record))
                
                self.write_population_to_csv(pop,'a',output_file= f"../../Pop_Over_Gen/pop_gen.csv")
                self.write_population_to_csv(pop,'a',output_file= f"../../Pop_Over_Gen/historical.csv")

                bar()
                clear_output()

            gc.collect()
            return hof

    def objective_function(self, Individual=[0, 0, 0, 0, 0, 0], write=True):
        n_planes = Individual[5]
        n_sats = Individual[4]

        old_run = self.historical_df[(self.historical_df[self.historical_df.columns.tolist()[0:6]] == Individual).all(axis=1)]

        if old_run.empty:
            if write:
                Load_Individual(Individual)
            self.stk_object.Satellite_Loader(f'../../Input_Files/Satellites_File.txt')
            self.stk_object.Results_Runner(enable_print=self.enable_print)
            self.stk_object.Create_Data_Comparison_df(Unplanned=False)
            percentage = np.average(self.stk_object.data_comparison['Planned (%)'])
            times = self.stk_object.data_comparison['Planned (Time)']
            cost = self.prog_cost_function(Individual)

            if times.isnull().any() or percentage != 1:
                time = self.stk_object.duration.total_seconds() / 86400
            else:
                time = np.average(times)

            penalty = 0
            if n_planes > n_sats:
                penalty = 500000000
            cost = cost + penalty

        else:
            print('OLD RUN USED!!!')
            percentage = old_run['Avg_Percentage']
            time = old_run['Avg_Time']
            cost = old_run['Cost']

        return tuple(np.array([percentage,time,cost])/self.norm_array)
    
    def prog_cost_function(self, Individual):
        n_planes = Individual[5]
        n_sats = Individual[4]
        cost_df = self.cost_df
        per_sat = (cost_df['Per Satellite']*n_sats).sum()
        total_cost = cost_df['Set Cost'].sum() + per_sat
        return total_cost
    
    def write_population_to_csv(self, pop, write_type, output_file):
        if output_file == f"../../Pop_Over_Gen/pop_gen.csv":
            with open(output_file, write_type) as file:
                if self.g == 0:
                    file.write("Gen,Pop,Alt,Inc,Initial_Raan,Delta_Raan,Num_Sats,Num_Planes,Avg_Percentage,Avg_Time,Cost,\n")
                for i in range(self.n_pop):
                    file.write(f'{self.g},{i},')
                    for idx in range(6):
                        file.write(f"{pop[i][idx]},")
                    for fit in pop[i].fitness.getValues() * self.norm_array:
                        file.write(f"{fit},")
                    file.write("\n")
        elif output_file == f"../../Pop_Over_Gen/historical.csv":
            with open(output_file, write_type) as file:
                for i in range(self.n_pop):
                    for idx in range(6):
                        file.write(f"{pop[i][idx]},")
                    for fit in pop[i].fitness.getValues() * self.norm_array:
                        file.write(f"{fit},")
                    file.write("\n")

