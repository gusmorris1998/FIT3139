from ctsp_algorithms import RN, NN, AG, AGP1
from scipy.spatial import distance
import numpy as np


class Game:
    def __init__(self, number_of_cities, number_of_players, grid_size):
        self.game = np.array([[]] * number_of_players)
        self.path = np.array([[]] * number_of_players, dtype=int)
        self.visited = {}
        self.not_visited = []
        self.number_of_players = number_of_players
        self.number_of_cities = number_of_cities
        self.grid_size = grid_size
        self.cities = self.generate_tsp_instance(number_of_cities, grid_size)
        self.bonus = (-100, -50, 0)

    def generate_tsp_instance(self, number_of_cities, grid_size):
        ans = []
        while len(ans) < number_of_cities:
            (x, y) = np.random.randint(0, grid_size, size=2)
            if (x, y) not in ans:
                ans.append((x, y))

        return ans

    def simulate(self, strategies):

        # Initialize arrays
        self.not_visited = [[i for i in range(self.number_of_cities)] for _ in range(self.number_of_players)]
        self.game = np.array([[]] * self.number_of_players)
        self.path = np.array([[]] * self.number_of_players, dtype=int)

        # Random choice of initial positions
        initial_cities = np.random.choice(self.not_visited[0], self.number_of_players)
        initial_scores = np.zeros([self.number_of_players, 1])
        self.path = np.concatenate([self.game, initial_cities.reshape([self.number_of_players, 1])], 1)
        self.game = np.concatenate([self.game, initial_scores], 1)

        for j in range(self.number_of_players):
            self.not_visited[j].pop(initial_cities[j])

        # Update global cities visited.
        self.visited = set(initial_cities)

        # Number of cities dictates number of steps
        for i in range(self.number_of_cities - 1):
            self.step(strategies, i)

        return self.compute_final_scores()

    def step(self, stratergies, i):
        choices = []

        # Iterate through the players -- Note: Have number of players as class var; make decision on this
        for j in range(len(stratergies)):
            # Get the choice from the statergies, define in different files
            choice = stratergies[j].get_choice(self.game, self.path, self.visited, self.not_visited, self.cities, j)
            choices.append(choice)

        # Compute scores from the choices
        scores = self.compute_scores(choices)

        self.path = np.concatenate([self.path, np.array(choices, dtype=int).reshape(self.number_of_players, 1)], 1)
        self.game = np.concatenate([self.game, np.array(scores).reshape(self.number_of_players, 1)], 1)

    def compute_scores(self, choices):
        # Initializes round scores as 0 for each
        score = [0]*len(choices)
        # Previous cities
        for j in range(len(choices)):
            # Location of agent j from previous step
            previous_location = self.cities[int(self.path[j, -1])]
            new_location = self.cities[choices[j]]

            # Compute distance between previous and new location of agent
            dist = distance.cdist([previous_location], [new_location])[0,0]

            # Conditional dictates whether agent gets a bonus from being first visitor of city
            if choices[j] not in self.visited:
                # Index list of agents who also chose the choice of unvisited city
                index_list = np.where(np.array(choices) == choices[j])[0]
                # Length 1 => only single first visitor => larger bonus
                if len(index_list) == 1: score[j] += self.bonus[0]
                else: 
                    # Multiple agents as first visitor at this game step => intermediate bonus
                    for indice in index_list: 
                        score[indice] += self.bonus[1]

                # Add city to global visited
                self.visited.add(choices[j])

            score[j] += dist
        
        return score
    
    def compute_final_scores(self):
        return np.sum(self.game, axis=1)
        
    

number_of_cities = 20
grid_size = 100

game = Game(number_of_cities, 2, grid_size)
a = game.simulate([NN, AGP1])
a