import numpy as np
from scipy.spatial import distance
from ctsp_algorithms import RN, NN, AG, AGP1
from main import Game

def compute_visited_iter(array):
    cols = array.shape[1]

    visited_iter = []
    visited_iter.append(array[:,0])

    for i in range(1, cols):
        visited_iter.append(np.unique(np.concatenate([visited_iter[-1], array[:,i]])))

    return visited_iter

def distance(city1, city2):
    x_distance = abs(city1[0] - city2[0])
    y_distance = abs(city1[1] - city2[1])
    return np.sqrt(x_distance**2 + y_distance**2)

def cost(tour, tsp_instance, visited_iter, bonus):
    cost = 0    

    for i in range(0, len(tour)-1):
        cost += distance(tsp_instance[tour[i]], tsp_instance[tour[i+1]])
    
        # Determine what bonus
        if tour[i] in visited_iter[i]:
            if tour[i] in visited_iter[i-1]:
                cost += bonus[2]
            else:
                cost += bonus[1]
        else:
            cost += bonus[0]
    return cost

def perturb(tour):
    # choose two cities at random
    i, j = np.random.choice(len(tour), 2, replace=False)
    new_tour = np.copy(tour)
    # swap them
    new_tour[i], new_tour[j] =  new_tour[j], new_tour[i]
    return new_tour

def crossover(tour_a, tour_b):
    cuttoff = np.random.randint(0, len(tour_a))
    ans = []
    for i in range(0, cuttoff):
        ans.append(tour_a[i])
    i = 0
    while len(ans) < len(tour_a):
        if tour_b[i] not in ans:
            ans.append(tour_b[i])
        i += 1
    return np.array(ans)

def initialise_population(number_of_cities, population_size):
    pop = []
    for _ in range(population_size):
        pop.append(np.random.permutation(number_of_cities))
    return pop

def initialise_population(number_of_cities, population_size):
    pop = []
    for _ in range(population_size):
        pop.append(np.random.permutation(number_of_cities))
    return pop

def genetic_algorithm(tsp_instance, population_size,number_of_generations, mutation_prob, crossover_prob, visited_iter, bonus):
    number_of_cities = len(tsp_instance)
    pop = initialise_population(number_of_cities, population_size)
    #compute fitness
    cost_values = np.array([cost(x, tsp_instance, visited_iter, bonus) for x in pop])

    # Sort the and rank the cost values
    fitness = np.argsort(np.argsort(-cost_values))
    # A larger number from cost_value will have lower probability
    fitness_probability = fitness/sum(fitness)
    
    for _ in range(number_of_generations):
        # create a new population
        new_pop = []
        fittest = pop[np.argmax(fitness)]
        new_pop.append(np.copy(fittest)) #always take fittest
        for _ in range(population_size-1):
            parent_index = np.random.choice(range(population_size), p = fitness_probability)
            parent = pop[parent_index]
            # sometimes mutate
            if np.random.rand() < mutation_prob:
                mutant = perturb(parent.copy())
                new_pop.append(mutant)
            # sometimes crossover 
            elif np.random.rand() < crossover_prob:
                another_parent_index = np.random.choice(range(population_size), p = fitness_probability)
                another_parent = pop[another_parent_index]
                new_pop.append(crossover(parent, another_parent))
            # most times just copy the parent
            else:
                new_pop.append(parent.copy())
        pop = new_pop
        #compute fitness
        cost_values = np.array([cost(x, tsp_instance, visited_iter, bonus) for x in pop])
        # fitness = 1.0/cost_values
        # fitness_probability = fitness/(np.sum(fitness))

        # Sort the and rank the cost values
        fitness = np.argsort(np.argsort(-cost_values))
        # A larger number from cost_value will have lower probability
        fitness_probability = fitness/sum(fitness)
    best = pop[np.argmin(cost_values)]
    best_cost = cost(best, tsp_instance, visited_iter, bonus)
    return best, best_cost

number_of_cities = 20
grid_size = 100
game = Game(number_of_cities, 2, grid_size)
score_card = game.simulate([NN, AGP1])

visited_iter = compute_visited_iter(game.path)
sol, cost_value = genetic_algorithm(game.cities, 50, 2000, 0.1, 0.1, visited_iter, game.bonus)