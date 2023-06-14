import numpy as np
import random
import matplotlib.pyplot as plt
from economic_load_dispatch_ga import EconomicLoadDispatchGA

# Define problem-specific parameters
num_generators = 3
min_power = [50, 30, 40]
max_power = [200, 150, 180]
demand = 400
line_loss = 10
lower_ramp = [10, 5, 8]
upper_ramp = [12, 6, 10]

# Define fuel cost coefficients
a = [0.01, 0.03, 0.02]
b = [0.1, 0.3, 0.2]
c = [10, 20, 15]
e = [0.001, 0.002, 0.003]
f = [0.01, 0.02, 0.03]

# Create an instance of the EconomicLoadDispatchGA class
eld_ga = EconomicLoadDispatchGA(num_generators, min_power, max_power, demand, line_loss, lower_ramp, upper_ramp)

# Run the genetic algorithm and get the best solution
best_solution = eld_ga.run(population_size=100, num_generations=50, mutation_rate=0.01)

# Get the best parent of each generation
best_parents = eld_ga.best_parents

# Plotting the best parent for each generation
generation_numbers = range(1, len(best_parents) + 1)
best_fitness_values = [eld_ga.calculate_fitness(parent) for parent in best_parents]

plt.plot(generation_numbers, best_fitness_values)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Best Fitness Value over Generations')
plt.show()
