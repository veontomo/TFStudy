import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random
import math
import operator

# Read the data into a list
with open("train.csv", encoding="ascii") as file:
    list = file.readlines()

title = [v.strip() for v in list[0].split(",")]
values = [v.strip().split(",") for v in list[1:]]


def castToInt(v, d):
    """Converts the first arg into an integer. In case of failure, the second argument is to be returned"""
    try:
        return int(v)
    except Exception as e:
        return d


intFieldNamesAsString = "Id, MSSubClass, LotFrontage, LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd, \
    MasVnrArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, 1stFlrSF, 2ndFlrSF, LowQualFinSF, GrLivArea, \
    BsmtFullBath, BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, Fireplaces, \
    GarageYrBlt, GarageCars, GarageArea, WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, \
    PoolArea, MiscVal, MoSold, YrSold, SalePrice"
intFieldNames = re.split(",\s*", intFieldNamesAsString)
intFieldPositions = [title.index(k) for k in intFieldNames]

strFieldNamesAsString = "MSZoning, Street, Alley, LotShape, LandContour, \
    Utilities, LotConfig, LandSlope, Neighborhood, Condition1, Condition2, \
    BldgType, HouseStyle, RoofStyle, RoofMatl, Exterior1st, Exterior2nd, \
    MasVnrType, ExterQual, ExterCond, Foundation, BsmtQual, BsmtCond, \
    BsmtExposure, BsmtFinType1, BsmtFinType2, Heating, HeatingQC, CentralAir, \
    Electrical, Functional, FireplaceQu, GarageType, GarageFinish, GarageQual, \
    GarageCond, PavedDrive, PoolQC, Fence, MiscFeature, SaleType, KitchenQual, SaleCondition"
strFieldNames = re.split(",\s*", strFieldNamesAsString)
strFieldPositions = [title.index(k) for k in strFieldNames]


def applyTo(arr, pos, mapper):
    """ Return a new array from arr by applying mapper to elements of arr whose position indicies are in pos.
        arr: original array
        pos: indexes of the elements of arr to be transformed by the mapper
        mapper: a transformation function
    """
    return [mapper(arr[i]) if (i in pos) else arr[i] for i in range(0, len(arr))]


valuesCast = [applyTo(line, intFieldPositions, lambda x: castToInt(x, -1)) for line in values]


def index(featureNames, data):
    """Return a dictionary of all future values.
    featureNames - list of feature names.
    data - list of feature values.
    If featureNames is a list of L elements, data is a two-dim list, each element of which is a list of L elements.
    """
    output = {name: [] for name in featureNames}
    for line in data:
        for i in range(0, len(line)):
            featureName = featureNames[i]
            featureValue = line[i]
            if not (featureValue in output[featureName]):
                output[featureName].append(featureValue)
                output[featureName].sort()
    return output


def digitalize(featureNames, featureValues, featureIndex, positions):
    result = []
    for i in range(0, len(featureValues)):
        line = []
        for k in range(0, len(featureValues[i])):
            featureName = featureNames[k]
            featureValue = featureValues[i][k]
            if k in positions:
                line.append(featureIndex[featureName].index(featureValue))
            else:
                line.append(featureValue)
        result.append(line)
    return result


def scale(x, a, b, A, B):
    """Return value to which x is mapped provided the interval (a, b) is lenearly mapped to (A, B)"""
    return ((B - A) * x + (A * b - B * a)) / (b - a)


def normalize(featureNames, featureValues, featureIndex, positions):
    result = []
    for i in range(0, len(featureValues)):
        line = []
        for k in range(0, len(featureValues[i])):
            featureName = featureNames[k]
            featureValue = featureValues[i][k]
            if k in positions:
                b1 = min(featureIndex[featureName])
                b2 = max(featureIndex[featureName])
                norm = scale(featureValue, b1, b2, 0, 1)
                line.append(norm)
            else:
                line.append(featureValue)
        result.append(line)
    return result


dataIndex = index(title, valuesCast)

digitalized = digitalize(title, valuesCast, dataIndex, strFieldPositions)

dataIndex2 = index(title, digitalized)

dataNorm = normalize(title, digitalized, dataIndex2, intFieldPositions)

labelColIndex = title.index("SalePrice")
dataColIndexes = [i for i in range(0, len(title)) if not (title[i] in ["Id", "GarageYrBlt", "SalePrice"])]

dataX = np.array(dataNorm)[:, dataColIndexes]
dataY = np.array(dataNorm)[:, [labelColIndex]]


def deviation(lst):
    size = len(lst)
    avg = sum(lst) / size
    return math.sqrt(sum([x * x for x in lst]) / size - avg * avg)


def corr(lst1, lst2):
    size = len(lst1)
    if size != len(lst2):
        return Exception('Both list must be of the same length.')
    avg1 = sum(lst1) / size
    avg2 = sum(lst2) / size
    lst0 = []
    for x, y in zip(lst1, lst2):
        lst0.append((x - avg1) * (y - avg2))
    return sum(lst0) / size / deviation(lst1) / deviation(lst2)

corrList = {title[i]: corr(dataX[:, i], dataY[:, 0]) for i in range(0, len(dataX[0, :]))}
frequency_sorted = sorted(corrList.items(), key=operator.itemgetter(1))

threshold = 0.0
for key in frequency_sorted:
    print(key[0], key[1])

exit()
# generate fake random data
# dataX = np.array([[random.random() for i in range(0, 3)] for k in range(0, 15)])
# dataY = dataX[:, -1]

F = len(dataX[0])
T = int(0.8 * len(dataX))
E = 500

print("number of features", F)
print("number of training data", T)
print("number of epochs", E)

X_train = dataX[:T]
Y_train = dataY[:T]

X_cv = dataX[T:]
Y_cv = dataY[T:]

CV = len(X_cv)
print("number of cross validation", CV)

# frequency = {}
# dataSize = len(values)
# for key in dataIndex:
#     l = len(dataIndex[key])
#     p = int(l/dataSize * 10000) / 100
#     frequency[key] = p
#
# frequency_sorted = sorted(frequency.items(), key=operator.itemgetter(1))
# threshold = 0.0
#
# for key in frequency_sorted:
#     if key[0] == 'Id':
#         continue
#     if key[1] > threshold:
#         print(key[0], key[1], dataIndex[key[0]], '\n')
#
# for k, v in zip(title, values[5]):
#     print(k, v)

model = Sequential()
model.add(Dense(1, input_dim=F, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='mean_absolute_percentage_error',
              metrics=['mae'])

history = model.fit(X_train, Y_train, nb_epoch=E, batch_size=32, verbose=0)
predictions = model.predict(X_cv)

for layer in model.layers:
    print(layer.get_weights())

for p, a in zip(predictions, Y_cv):
    print(p, a, (p - a) / a)
# visualize the training progress

COLOR_LOSS = 'blue'
COLOR_ACC = 'green'
# plt.plot(range(1, E + 1), history.history['mae'], 'k', color=COLOR_ACC)
plt.plot(range(1, E + 1), history.history['loss'], 'k', color=COLOR_LOSS)
plt.xlabel('Epoch')
plt.title('Training progress')

line1 = mlines.Line2D([], [], color=COLOR_LOSS, label='loss')
# line2 = mlines.Line2D([], [], color=COLOR_ACC, label='accuracy')
plt.legend(handles=[line1])

plt.show()
