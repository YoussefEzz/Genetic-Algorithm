import numpy as np
import random

#Genetic Algorithm Class for the travelling Salesman Problem(TSP)
class GeneticAlgorithm:
  def __init__(self, cities, weights, start_city, population_size):

    self.cities = cities                    # list of indices of G cities
    self.distances = weights                # list of lists representing matrix of distances between cities
    self.start_city = start_city            # index of start city or node
    self.population_size = population_size  # population size
    self.chromosome_length = len(self.cities) + 1

    self.population = []                    # a list of lists for population of individuals
    for _ in range(self.population_size):
      individual = [0] * self.chromosome_length    # Create a sublist with zeros
      self.population.append(individual)

    self.population_fitness = []                       # fitness list which contains the fitness for each individual in the chromosome 
    for _ in range(self.population_size):
      self.population_fitness.append(0)
    
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
        self.population_fitness[i] = self.fitness(self.population[i])

  # fitness function will return the sum of distances between every two consecutive cities in the chromosome  
  def fitness(self, chromosome):
    fitness = 0
    for i in range(len(chromosome) - 1):
        city_A = chromosome[i]
        city_B = chromosome[i+1]
        fitness += self.distances[city_A][city_B]
    return fitness