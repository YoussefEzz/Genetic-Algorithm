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


  # one-point crossover function 
  def onepoint_crossover(self, index_firstparent, index_secondparent):
    firstparent = self.population[index_firstparent]
    secondparent = self.population[index_secondparent]

    crossover_point = random.randint(1, self.chromosome_length - 2)

    firstoffspring = firstparent[0 : crossover_point] + secondparent[crossover_point : self.chromosome_length]
    secondoffspring = secondparent[0 : crossover_point] + firstparent[crossover_point : self.chromosome_length]
    return [firstoffspring, secondoffspring]

# partially-mapped crossover(PMX) function 
  def partially_mapped_crossover(self, index_firstparent, index_secondparent):
    parent1 = self.population[index_firstparent]
    parent2 = self.population[index_secondparent]

    child1 = [None] * self.chromosome_length
    child2 = [None] * self.chromosome_length

    # Select two random crossover points
    crossover_point1 = random.randint(0, self.chromosome_length - 1)
    crossover_point2 = random.randint(0, self.chromosome_length - 1)
    if crossover_point1 > crossover_point2:
        crossover_point1, crossover_point2 = crossover_point2, crossover_point1

    # Copy the selected portion from parents to children
    child1[crossover_point1:crossover_point2+1] = parent1[crossover_point1:crossover_point2+1]
    child2[crossover_point1:crossover_point2+1] = parent2[crossover_point1:crossover_point2+1]

    # Map the elements between crossover points
    for i in range(crossover_point1, crossover_point2+1):
        if parent2[i] not in child1:
            index = parent2.index(parent1[i])
            while child1[index] is not None:
                index = parent2.index(parent1[index])
            child1[index] = parent2[i]

        if parent1[i] not in child2:
            index = parent1.index(parent2[i])
            while child2[index] is not None:
                index = parent1.index(parent2[index])
            child2[index] = parent1[i]

    # Copy the remaining elements
    for i in range(self.chromosome_length):
        if child1[i] is None:
            child1[i] = parent2[i]
        if child2[i] is None:
            child2[i] = parent1[i]

    return child1, child2


    return 


