import numpy as np
import matplotlib.pyplot as plt

with open("train.csv", encoding="ascii") as file:
    list = [next(file) for x in range(0, 10)]

title = [v.strip() for v in list[0].split(",")]
data = [[int(i) for i in v.strip().split(",")] for v in list[1:]]

X = np.array(data)[:, 1:]
Y = np.array(data)[:, 0]

Ysoft = [[1 if (i == y) else 0 for i in range(0, 10)] for y in Y]
print("There are", len(X[0]), "features")
print("There ", len(X), "items in the data set")

print(Ysoft)
plt.imshow(np.reshape(X[1], [28, 28]), cmap='gray')
plt.show()