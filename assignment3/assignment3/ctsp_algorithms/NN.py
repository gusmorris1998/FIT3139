import numpy as np
from scipy.spatial import distance

def get_choice(game, path, visited, not_visited, cities, j):
    previous_location = cities[int(path[j, -1])]
    valid_locations = np.take(np.array(cities,dtype="f,f"), not_visited[j])
    index = distance.cdist([previous_location], valid_locations.tolist())[0].argmin()

    return not_visited[j].pop(index)
    
