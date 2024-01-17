import numpy as np
import random

#Genetic Algorithm Class for the travelling Salesman Problem(TSP)
class GeneticAlgorithm:
  def __init__(self, cities, weights, start_city, population_size, num_generations):

    self.cities = cities                            # list of indices of G cities
    self.distances = weights                        # list of lists representing matrix of distances between cities
    self.start_city = start_city                    # index of start city or node
    self.population_size = population_size          # population size
    self.chromosome_length = len(self.cities) + 1   # chromosome length is the number of cities G + 1
    self.num_generations = num_generations          # number of generations or epochs

    self.population = []                            # a list of lists for population of individuals
    for _ in range(self.population_size):
      individual = [0] * self.chromosome_length     
      self.population.append(individual)

    self.population_fitness = []                    # fitness list which contains the fitness for each individual in the population
    for _ in range(self.population_size):
      self.population_fitness.append(0)
    
    self.population_rank = []                       # rank list which contains the rank calculated in rank selection for each individual in the population 
    for _ in range(self.population_size):
      self.population_rank.append(0)

  # Initialize Population P
  def initialize_population(self):
    cities_except_first = self.cities
    cities_except_first.remove(self.start_city)

    for i in range(self.population_size):
        # Shuffle the list
        shuffled_list = cities_except_first[:]  # Create a copy of the original list
        random.shuffle(shuffled_list)

        chromosome = shuffled_list
        chromosome.insert(0, self.start_city)   # append start city at the beginning
        chromosome.append(self.start_city)      # append start city at the end
        self.population[i] =  chromosome

  # Evaluate fitness for all chromosomes in P
  def evaluate_fitness(self):
    for i in range(self.population_size):
        self.population_fitness[i] = 100 / self.fitness(self.population[i])

  # fitness function will return the sum of distances between every two consecutive cities in the chromosome  
  def fitness(self, chromosome):
    fitness = 0
    for i in range(len(chromosome) - 1):
        city_A = chromosome[i]
        city_B = chromosome[i+1]
        fitness += self.distances[city_A][city_B]
    return fitness
  

  # rank selection function will return the indices of selected chromosomes
  def rank_selection(self):

    # compute the rank for each chromosome based on it's fitness by sorting individuals by increasing fitness
    population_rank = [i[0] for i in sorted(enumerate(self.population_fitness), key=lambda x: x[1])]
    
    
    rank_probabilities = [element / sum(population_rank) for element in population_rank]
    #print("rank weights: ", rank_probabilities)

    # select 2 chromosomes from population based on rank selection with Probability of selection proportional to rank
    selected_chromosomes = np.random.choice(len(self.population), 2, replace=False, p=rank_probabilities)
    #print("selected chromosomes :", selected_chromosomes)

    return selected_chromosomes
  

  # tournament selection function will return the indices of selected chromosomes
  def tournament_selection(self):
    
    population_rank = [i[0] for i in sorted(enumerate(self.population_fitness), key=lambda x: x[1])]

    # Select at random 3 individuals to get candidates for first parent
    candidates_firstparent = np.random.choice(len(self.population), 3, replace=False)
    
    # Select at random 3 individuals to get candidates for second parent
    candidates_secondparent = np.random.choice(len(self.population), 3, replace=False)

    index_firstparent = -1
    fitness_firstparent = 0
    for i in range(len(self.population_fitness)):
      for j in (candidates_firstparent):  
        if i == j:
          if self.population_fitness[i] > fitness_firstparent:
            fitness_firstparent = self.population_fitness[i]
            index_firstparent = i

    index_secondparent = -1
    fitness_secondparent = 0
    for i in range(len(self.population_fitness)):
      for j in (candidates_secondparent):  
        if i == j:
          if self.population_fitness[i] > fitness_secondparent:
            fitness_secondparent = self.population_fitness[i]
            index_secondparent = i

    return [index_firstparent, index_secondparent]

