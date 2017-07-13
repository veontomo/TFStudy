import random
import math
def sigma(lst):
    """Standard deviation: sqrt(E[X^2] - E[X]^2)"""
    size = len(lst)
    avg = sum(lst) / size
    return math.sqrt(sum([x * x for x in lst]) / size - avg * avg)


def corr(lst1, lst2):
    """Correlation btw two variables. The lists of the variable values must be
    of equal length."""
    size = len(lst1)
    if size != len(lst2):
        return Exception('Both list must be of the same length.')
    avg1 = sum(lst1) / size    
    avg2 = sum(lst2) / size
    print("avg1", avg1, "avg2", avg2)
    lst0 = []
    for x, y in zip(lst1, lst2):
        lst0.append((x - avg1) * (y - avg2))
    return sum(lst0) / size / sigma(lst1) / sigma(lst2)

Y = [1 - 2*random.random() for i in range(0, 1000)]
X = [y*y*y for y in Y]
print(corr(X, Y))