import pickle

def load_data(file_name):
    """
    Load data from a pickle file.

    Parameters:
    file_name (str): The name of the file to load.

    Returns:
    The data loaded from the file.
    """
    with open(file_name, "rb") as f:
        return pickle.load(f)

def transform_data(data):
    """
    Transpose a DataFrame and convert it to a flat list.

    Parameters:
    data (DataFrame): The DataFrame to transform.

    Returns:
    A flat list containing the values of the DataFrame.
    """
    data = data.transpose()
    data = data.values.tolist()
    return [item for sublist in data for item in sublist]

def create_dict(pairs):
    """
    Create a dictionary from a list of pairs.

    Each pair is added to the dictionary twice, once with the first element as the key and the second as the value, and once with the second element as the key and the first as the value.

    Parameters:
    pairs (list): The list of pairs.

    Returns:
    dict: The created dictionary.
    """
    pair_dict = {}
    for pair in pairs:
        pair_dict[pair[0]] = pair[1]
        pair_dict[pair[1]] = pair[0]
    return pair_dict

def calculate_D_and_T(l, dist_matrix, dur_matrix, collection_time):
    """
    Calculate the total distance and time for a given route.

    The function iterates over the route and for each pair of consecutive locations, it adds the distance and time between them to the total. It also adds the collection time at each location to the total time.

    Parameters:
    l (list): The route.
    dist_matrix (list of list of int): The distance matrix.
    dur_matrix (list of list of int): The duration matrix.
    collection_time (list of int): The collection times.

    Returns:
    tuple: The total distance and time for the route.
    """
    distance = 0
    time = 0
    for i in range(len(l)):
        if i != len(l) - 1:
            distance += dist_matrix[l[i]][l[i + 1]]
            time += dur_matrix[l[i]][l[i + 1]]
        time += collection_time[l[i]]
    return distance, time


def has_duplicates(lst):
    """
    Check if a list has duplicates.

    The function creates a set from the list, which removes any duplicates. If the length of the set is different from the length of the list, the list has duplicates.

    Parameters:
    lst (list): The list to check.

    Returns:
    bool: True if the list has duplicates, False otherwise.
    """
    return len(lst) != len(set(lst))


def is_permutation(l, init_solu):
    """
    Check if a list is a permutation.

    A list is a permutation if it starts and ends with the same elements as the initial solution.

    Parameters:
    l (list): The list to check.
    init_solu (list): The initial solution.

    Returns:
    bool: True if the list is a permutation, False otherwise.
    """
    return l[0] == 0 and l[-1] == init_solu[-1]

def calculate_variance(population):
    """
    Calculate the variance of fitness values within the population.

    Parameters:
    population (list): The current population whose fitness variance is to be calculated.

    Returns:
    float: The variance of the fitness values within the population.
    """

    mean = sum(sol[0][0] for sol in population) / len(population)
    return sum((sol[0][0] - mean) ** 2 for sol in population) / len(population)