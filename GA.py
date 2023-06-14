import numpy as np
import random

class EconomicLoadDispatchGA:
    def __init__(self, num_generators, min_power, max_power, demand, line_loss, lower_ramp, upper_ramp):
        self.num_generators = num_generators
        self.min_power = min_power
        self.max_power = max_power
        self.demand = demand
        self.line_loss = line_loss
        self.lower_ramp = lower_ramp
        self.upper_ramp = upper_ramp
        self.best_parents = []  # Store the best parent of each generation

    def initialize_population(self, population_size):
        population = []
        for _ in range(population_size):
            chromosome = [random.uniform(self.min_power[i], self.max_power[i]) for i in range(self.num_generators)]
            population.append(chromosome)
        return population

    def calculate_fitness(self, chromosome):
        total_cost = 0.0
        for i in range(self.num_generators):
            # Equation 2: Fi(Pi) = ai*Pi^2 + bi*Pi + ci + eisin(fi(Pmin - Pi))
            cost = a[i] * chromosome[i]**2 + b[i] * chromosome[i] + c[i] + e[i] * np.sin(f[i] * (self.min_power[i] - chromosome[i]))
            total_cost += cost
        return total_cost

    def check_constraints(self, chromosome):
        total_power = sum(chromosome)
        # Equation 4: Pi - Pd - Ploss = 0
        if abs(total_power - self.demand - self.line_loss) > 1e-6:
            return False

        for i in range(self.num_generators):
            # Equation 5: Pmin <= Pi <= Pmax
            if chromosome[i] < self.min_power[i] or chromosome[i] > self.max_power[i]:
                return False

            if i > 0:
                # Equation 6: max(Pimin, Pit-1 - DRi) <= Pi <= min(Pimax, Pit-1 + URi)
                if chromosome[i] - chromosome[i-1] > self.upper_ramp[i] or chromosome[i-1] - chromosome[i] > self.lower_ramp[i]:
                    return False
        return True

    def select_parents(self, population, fitness_values, num_parents):
        parents = []
        fitness_sum = sum(fitness_values)
        probabilities = [fitness / fitness_sum for fitness in fitness_values]
        for _ in range(num_parents):
            selected = np.random.choice(range(len(population)), p=probabilities)
            parents.append(population[selected])
        return parents

    def crossover(self, parents, num_offsprings):
        offsprings = []
        for _ in range(num_offsprings):
            parent1, parent2 = random.sample(parents, 2)
            offspring = []
            for i in range(self.num_generators):
                if random.random() < 0.5:
                    offspring.append(parent1[i])
                else:
                    offspring.append(parent2[i])
            offsprings.append(offspring)
        return offsprings

    def mutate(self, chromosome, mutation_rate):
        mutated_chromosome = []
        for gene in chromosome:
            if random.random() < mutation_rate:
                mutated_gene = random.uniform(self.min_power[i], self.max_power[i])
                mutated_chromosome.append(mutated_gene)
            else:
                mutated_chromosome.append(gene)
        return mutated_chromosome

    def run(self, population_size, num_generations, mutation_rate):
        population = self.initialize_population(population_size)

        for generation in range(num_generations):
            fitness_values = [self.calculate_fitness(chromosome) for chromosome in population]

            best_fitness = min(fitness_values)
            best_index = fitness_values.index(best_fitness)
            best_solution = population[best_index]

            print(f"Generation {generation+1}: Best Fitness = {best_fitness}, Best Solution = {best_solution}")

            if self.check_constraints(best_solution):
                break

            self.best_parents.append(best_solution)

            parents = self.select_parents(population, fitness_values, population_size // 2)
            offsprings = self.crossover(parents, population_size // 2)
            population = parents + offsprings
            for i in range(population_size):
                population[i] = self.mutate(population[i], mutation_rate)

        return best_solution
