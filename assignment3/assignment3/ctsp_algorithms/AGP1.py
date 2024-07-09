import numpy as np
from ctsp_algorithms import AG
from scipy.spatial import distance

def get_choice(game, path, visited, not_visited, cities, j):

    rival_next_moves = np.array([], dtype=int)
    # Computes the next estimated move based on assumption that rival agent play NN stratergy
    for i in range(len(not_visited)):
        if i != j:
            rival_previous_location = cities[int(path[i, -1])]
            rival_valid_locations = np.take(np.array(cities,dtype="f,f"), not_visited[i])
            if len(rival_valid_locations) > 0:
                rival_index = distance.cdist([rival_previous_location], rival_valid_locations.tolist())[0].argmin()
                rival_next_moves = np.append(rival_next_moves, rival_index)

    rival_next_moves = np.unique(rival_next_moves)
            
    previous_location = cities[int(path[j, -1])]
    # Valid locations differ in that it takes only non-visited options compared to NN
    valid_locations = np.take(np.array(cities,dtype="f,f"), np.setdiff1d(not_visited[j], np.unique(np.concatenate([rival_next_moves, list(visited)]))))

    # Does not exist a valid location => play AG
    if len(valid_locations) == 0:
        return AG.get_choice(game, path, visited, not_visited, cities, j)

    index = distance.cdist([previous_location], valid_locations.tolist())[0].argmin()

    return not_visited[j].pop(index)