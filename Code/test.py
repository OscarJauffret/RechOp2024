#import numpy as np
#import time
#import random
#
## Create a large list and a large NumPy array
#size = 10_000_000
#python_list = [random.randint(0, 1000) for _ in range(size)]
#numpy_array = np.array(python_list)
#index = random.randint(0, size - 1)
#
## Timing sum over list
#start_time = time.time()
#python_list.sort(reverse=True)
##sum_list = sum(python_list)
#time_list = time.time() - start_time
#
## Timing sum over NumPy array
#start_time = time.time()
#numpy_array.sort()
#f = np.flip(numpy_array)
##sum_array = np.sum(numpy_array)
#time_array = time.time() - start_time
#
#print(f"list took {time_list:.5f} seconds.")
#print(f"NumPy array took {time_array:.5f} seconds.")
#if time_array > 0:
#    print(f"NumPy array was {time_list / time_array:.1f} times faster.")
#
import math
import numpy as np

def exponential_base(variance, a=0.01, b=1, c=1, d=0.0002):
    base = a * np.log(b + variance) + c + d * (np.log(b + variance))**3
    return base

def exponential_base2(variance, a=0.01, b=1):
    base = a * (b + variance)**0.5
    return base




n = 233  # Taille de la liste
base = 0.4  # Base de l'exponentielle, moins de 1 pour des poids décroissants

weights1 = [math.exp(-base * i) for i in range(n)]

base = 0.8
weights2 = [math.exp(-base * i) for i in range(n)]

base = 1.18
weights3 = [math.exp(-base * i) for i in range(n)]

print(weights1)
print(weights2)
print(weights3)

variances = [1, 100, 10000, 200000, 1000000, 100000000]
bases = [exponential_base(var) for var in variances]
print(bases)
for base in bases:
    print([math.exp(-base * i) for i in range(n)])

bases = [exponential_base2(var) for var in variances]
print(bases)
for base in bases:
    print([math.exp(-base * i) for i in range(n)])



#choose k (the tournament size) individuals from the population at random
#choose the best individual from the tournament with probability p
#choose the second best individual with probability p*(1-p)
#choose the third best individual with probability p*((1-p)^2)
#and so on