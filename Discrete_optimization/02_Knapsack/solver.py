#!/usr/bin/python
# -*- coding: utf-8 -*-

# AUTHOR: Juyoung Wang
# MAIL: jyw1189@gmail.com

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

# Let's import some possible packages.
import cvxpy as cp
import numpy as np
import copy

# First, we solve the model using classical DP approach.
def DP_approach(firstLine, item_count, capacity, items):
    
    # Generating numpy zero DP matrix.
    big_number = item_count*max([i.weight for i in items]) + 1
    DP_matrix = np.zeros((capacity + 1, item_count + 1)).T
    DP_matrix = np.asarray([np.asarray([0 if i == 0 else -big_number for j in range(capacity + 1)]) for i in range(item_count + 1)]).T

    # Fill DP matrix.
    for i in range(capacity + 1):
        for j in range(item_count):
            if items[j].weight <= i:
                DP_matrix[i][j+1] = np.max([DP_matrix[i][j], items[j].value + DP_matrix[i - items[j].weight][j]])
            else:
                DP_matrix[i][j+1] = DP_matrix[i][j]
    
    # Get knapsack solution value
    solution_value = DP_matrix[-1][-1]
    solution_vector = np.zeros(item_count)
    i, j = capacity, item_count
    
    # Get knapsack solution vectors
    for r in range(item_count):
        if (DP_matrix[i][j-r] != DP_matrix[i][j-r-1]):
            i -= items[j-r-1].weight
            solution_vector[j-r-1] = 1

    return int(round(solution_value)), [int(round(i)) for i in solution_vector]


# Since DP seems to be not appropriate for large size problems, we will solve the problem using rough heuristic algorithm. We will put all items in our knapsack and we will be taking items with high X (variable criteria) until we get a feasible solution. Once we reach the feasible solution, we will stand at the one previous step and we will be seeking for an element to be wiped, which will give us the best feasible value.
def int_ap(x):
    return int(round(x))

def int_ap_table(X):
    Result = []
    Result = []
    for i in X:
        j = i
        for position in range(3):
            j[position] = int_ap(j[position])
        Result.append(j)
    return Result


def Rough_heuristic(firstLine, item_count, capacity, items):
    
    # We start our problem with fully filled (overweight) knapsack.
    solution_vector = [1 for i in range(item_count)]
    
    # By modifying this line, we can adjust our greedy selection criteria.
    item_vector_original = np.array([[i.weight, i.index, i.value] for i in items])
    item_vector = np.array([[i.weight, i.index, i.value, (i.value/i.weight), (i.weight/i.value)] for i in items])
    item_vector = int_ap_table(item_vector[np.argsort(item_vector[:,0])].tolist()[::-1])
    total_weight, total_value = sum([i.weight for i in items]), sum([i.value for i in items])
    heuristic_table = []
    passing_indicator = False
    
    for i in range(len(item_vector)):
        if passing_indicator == False and total_weight - item_vector[i][0] > capacity:
            #previous_vector, previous_weight, previous_value = solution_vector, total_weight, total_value
            solution_vector[item_vector[i][1]] = 0
            total_weight -= item_vector[i][0]
            total_value -= item_vector[i][2]
        else:
            passing_indicator = True
            heuristic_table.append([total_value - item_vector[i][2], total_weight - item_vector[i][0], item_vector[i][1]])
            
    # Once we get the solution, we throw away some item which allows us to have our knapsack with the highest total value. 
    solution_table = [i for i in heuristic_table if i[1] <= capacity]
    max_sub = max(solution_table, key=lambda x: x[1])
    total_value, total_weight, max_index = max_sub[0], max_sub[1], max_sub[2]
    solution_vector[max_index] = 0
    #print(capacity, current_weight)
    
    # In case we remain with an empty knapsack, we implemented a simple greedy algorithm.
    w_n = 0
    v_n = 0
    if solution_vector == [0 for i in range(item_count)]:
        for i in item_vector:
            if w_n + i[0] <= capacity:
                w_n += i[0]
                v_n += i[2]
                solution_vector[i[1]] = 1
            else:
                total_value = v_n
                break

    return int(round(total_value)), [int(round(i)) for i in solution_vector]


# Although the previous polynomial time approximation gives us a reasonable solution, we would like to pour more time into solving the problem, sacrifying less optimality. To achieve this, we implement Rough_DP. This algorithm consist in taking only n_1 number of items with high X (variable criteria) and n_2 number of itmes with low weight to perform dynamic programming.
def Rough_DP(firstLine, item_count, capacity, items):
    
    # Let's define n_1 and n_2:
    n_1 = 60
    n_2 = 10
    
    # Now, we generate our DP matrix.
    big_number = item_count*max([i.weight for i in items]) + 1
    item_info_org = np.array([[i.value, i.weight, i.index, i.value/i.weight] for i in items])
    item_info = [tuple(i) for i in int_ap_table(item_info_org[np.argsort(item_info_org[:,3])].tolist()[::-1])]
    item_info_weight = [tuple(i) for i in int_ap_table(item_info_org[np.argsort(item_info_org[:,1])].tolist()[::-1])]
    item_info = [list(i) for i in list(set(item_info_weight[-n_2:] + item_info[:n_1]))]
    DP_matrix = np.zeros((capacity + 1, len(item_info))).T
    DP_matrix = np.asarray([np.asarray([0 if i == 0 else -big_number for j in range(capacity + 1)]) for i in range(item_count + 1)]).T
    
    # Fill DP matrix.
    for i in range(capacity + 1):
        for j in range(len(item_info)):
            if item_info[j][1] <= i:
                DP_matrix[i][j+1] = np.max([DP_matrix[i][j], item_info[j][0] + DP_matrix[i - item_info[j][1]][j]])
            else:
                DP_matrix[i][j+1] = DP_matrix[i][j]
    
    # Get knapsack solution value
    solution_vector = np.zeros(len(item_info))
    i, j = capacity, len(item_info)
    
    # Get knapsack solution vectors
    for r in range(len(item_info)):
        if (DP_matrix[i][j-r] != DP_matrix[i][j-r-1]):
            i -= item_info[j-r-1][1]
            solution_vector[j-r-1] = 1
    
    final_solution_vector = [0 for i in range(item_count)]
    final_value = 0
    for i in range(len(item_info)):
        if solution_vector[i] == 1:
            final_solution_vector[item_info[i][2]] = 1
            final_value += item_info[i][0]
            
    return int(round(final_value)), [int(round(i)) for i in final_solution_vector]


# Now, let's program branch and bound algorithm, using linear relaxations. First, let's compute linear relaxation:
def KP_Linear_relaxation(capacity, item_info):
    item_information = [i for i in copy.deepcopy(item_info) if i[1] <= capacity]
    cost_performance_informations = [i[1]/i[0] for i in item_information]
    item_informations = copy.deepcopy(item_information)
    for i in range(len(item_information)):
        item_informations[i] = item_information[i].append(cost_performance_informations[i])
        print(item_information[i])
        print(cost_performance_informations[i])
    return item_informations


# Here, we will compute BB algorithm:
def BB(firstLine, item_count, capacity, items):
    Node = namedtuple("Node", ['Max_Opt', 'Left_weight', 'Associated_solution'])
    # NOT DONE YET...
    return Node


# Here, we tried to implement a projected gradient method on Euclidean box, having lagrangian relaxation of our problem as the objective function. Unfortunately, this did not work well (or we have to improve it more...).
def LR_function(capacity, value_vec, weight_vec, sol_vec, lda):
    return np.dot(value_vec, sol_vec) + lda*(capacity - np.dot(weight_vec, sol_vec))


def Box_projector(x):
    x = np.where(x <= 0, 0, x)
    x = np.where(x >= 1, 1, x)
    return x
    

def Gradient_projection_approach(firstLine, item_count, capacity, items, lda):
    value_vector = np.asarray([i.value for i in items])
    weight_vector = np.asarray([i.weight for i in items])
    solution_vector = np.ones(item_count)
    L_constant = 2*item_count
    
    # For the sake of convenience, we fix our n_iteration = 1000.
    for iteration in range(1000):
        solution_vector =  Box_projector(solution_vector + (1/L_constant)*(value_vector - lda*weight_vector))
        
    return int(np.dot(value_vector, np.rint(solution_vector))), [int(round(i)) for i in np.rint(solution_vector)]

# This function calls required dataset and is used to solve the problem.
def solve_it(input_data):
    
    # This code requires the direction of input_data file.
    # Since we are working in the compressed file directory (where
    ## we consider data folder as first subfolder), our input
    ## will be of the form 'data/ks_4_0'.
    
    # Here we read our input files.
    lines = input_data.split('\n')
    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])
    items = []
    
    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
    
    # SINCE DP TAKES TOO MUCH TIME, WE USE PREVIOUSLY SAVED RESULTS.
    if item_count <= 200:
        output = DP_approach(firstLine, item_count, capacity, items)
    elif item_count <= 1000:
        output = Rough_DP(firstLine, item_count, capacity, items)
    else:
        output = Rough_heuristic(firstLine, item_count, capacity, items)
    
    # Prepare the solution in the specified output format
    output_data = str(output[0]) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, output[1]))

    return output_data

# Automatic input
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

