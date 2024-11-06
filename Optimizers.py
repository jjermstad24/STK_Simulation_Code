from Scripts import *

class MultiObjectiveOptimizer:
    def __init__(self, stk_object, n_pop, n_gen,historical_df,cost_df):
        self.cost_df = cost_df
        self.historical_df = historical_df
        self.stk_object = stk_object
        self.n_pop = n_pop
        self.n_gen = n_gen
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Satellite", list, fitness=creator.FitnessMulti)
        self.lower = [575, 80, 30, 0, 3, 3]
        self.upper = [630, 100, 150, 30, 12, 12]
        self.norm_array = np.array([self.stk_object.duration.total_seconds() / 86400, self.prog_cost_function([0,0,0,0,12,12])])

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

        # Register evaluate as a method of the class
        self.toolbox.register("evaluate", self.lazy_evaluate)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", tools.mutUniformInt, low=self.lower, up=self.upper, indpb=0.8)
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

            if read:
                i = np.random.randint(0,self.n_pop)
                if get_ind(self.upper[5]) != 0:
                    for idx in range(len(self.upper)):
                        pop[i][idx] = get_ind(self.upper[5])[idx]
                


            fitnesses = list(map(lambda ind: self.toolbox.evaluate(ind), pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            hof = tools.HallOfFame(1)
            hof.update(pop)

            self.fits.append([ind.fitness.values for ind in pop])
            record = self.stats.compile(pop) if self.stats else None


            if record is not None:
                print(pd.DataFrame(record))

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

                bar()
                clear_output()

            gc.collect()
            return hof

    def objective_function(self, Individual=[0,0,0,0,0,0], write=True):
        n_planes = Individual[5]
        n_sats = Individual[4]

        old_run = self.historical_df[(self.historical_df[self.historical_df.columns.tolist()[0:6]] == Individual).all(axis=1)]

        if old_run.empty:
            if write:
                Load_Individual(Individual)
            self.stk_object.Satellite_Loader(f'../../Input_Files/Satellites_File.txt')
            self.stk_object.Results_Runner(enable_print=self.enable_print,opt=True)
            self.stk_object.Create_Data_Comparison_df(Unplanned=False)
            percentage = np.average(self.stk_object.data_comparison['Planned (%)'])
            times = self.stk_object.data_comparison['Planned (Time)']
            cost = self.prog_cost_function(Individual)

            if times.isnull().any() or percentage < 100:
                percentage = 0
                time = self.stk_object.duration.total_seconds() / 86400
                penalty = 3000000000
            else:
                time = np.average(times)

            penalty = 0
            
            if n_planes > n_sats:
                penalty = 3000000000
            cost = cost + penalty

        else:
            print('OLD RUN USED!!!')
            percentage = old_run['Avg_Percentage'].iloc[0]
            time = old_run['Avg_Time'].iloc[0]
            cost = old_run['Cost'].iloc[0]

        objectives = [percentage,time,cost]

        self.write_population_to_csv(Individual,objectives=objectives,write_type='a')

        return tuple(np.array([time,cost])/self.norm_array)
    
    def prog_cost_function(self, Individual):
        cost_df = self.cost_df  

        cost_per_launch = 12000000

        n_planes = Individual[5]
        n_sats = Individual[4]

        operations_cost = 132565233
        first_sat_cost = cost_df['First_Sat_Cost'].sum()
        additional_sats_cost = (cost_df['Add_Sat_Cost']*(n_sats-1)).sum()
        launch_costs = cost_per_launch*n_planes
        
        total_cost = operations_cost +  first_sat_cost + additional_sats_cost + launch_costs
        return total_cost
    
    def write_population_to_csv(self, Individual, objectives, write_type):
       output_file = f"../../Output_Files/historical.csv"
       with open(output_file, write_type) as file:
            for dv in Individual:
                file.write(f"{dv},")
            for fit in objectives:
                file.write(f"{fit}")
                if fit != objectives[-1]:
                    file.write(',')
            file.write("\n")

