import numpy as np

def get_choice(game, path, visited, not_visited, cities, j):
    index = np.random.randint(0, len(not_visited[j]))

    return not_visited[j].pop(index)