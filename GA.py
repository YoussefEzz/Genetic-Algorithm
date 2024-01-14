import numpy as np
import random

#Genetic Algorithm Class for the travelling Salesman Problem(TSP)
class GeneticAlgorithm:
  def __init__(self, cities, weights, start_city, population_size):

    self.cities = cities                    # list of indices of G cities
    self.weights = weights                  # list of lists representing matrix of distances between cities
    self.start_city = start_city            # index of start city or node
    self.population_size = population_size  # population size

    self.population = []                    # an array of arrays for population of individuals
    

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
        self.population.append( chromosome)
    