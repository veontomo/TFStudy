#import os
#import tensorflow as tf
with open("train.csv", encoding="ascii") as file:
    list = file.readlines()
features = list[0].strip().split(",")
feature_values = [[] for i in range(0, len(features)) ]
for i in range(1, len(list)):
    line=list[i].strip().split(",")
    for k in range(0, len(line)-1):
        if not(line[k] in feature_values[k]):
            feature_values[k].append(line[k])


print(feature_values)
print(len(list))