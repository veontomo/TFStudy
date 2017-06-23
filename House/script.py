import csv
import operator
import re

# Read the data into a list
with open("test.csv", encoding="ascii") as file:
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
    PoolArea, MiscVal, MoSold, YrSold"
intFieldNames = re.split(",\s*", intFieldNamesAsString)
intFieldPositions = [title.index(k) for k in intFieldNames]


strFieldNamesAsString = "MSZoning, Street, Alley, LotShape, LandContour, \
    Utilities, LotConfig, LandSlope, Neighborhood, Condition1, Condition2, \
    BldgType, HouseStyle, RoofStyle, RoofMatl, Exterior1st, Exterior2nd, \
    MasVnrType, ExterQual, ExterCond, Foundation, BsmtQual, BsmtCond, \
    BsmtExposure, BsmtFinType1, BsmtFinType2, Heating, HeatingQC, CentralAir, \
    Electrical, Functional, FireplaceQu, GarageType, GarageFinish, GarageQual, \
    GarageCond, PavedDrive, PoolQC, Fence, MiscFeature, SaleType, SaleCondition"
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
            if not(featureValue in output[featureName]):
                output[featureName].append(featureValue)
                output[featureName].sort()
    return output

def digitalize(featureNames, featureValues, featureIndex, indexedPositions):
    result = []
    for i in range(0, len(featureValues)):
        line = []
        for k in range(0, len(featureValues[i])):
            featureName = featureNames[k]
            featureValue = featureValues[i][k]
            if k in indexedPositions:
                line.append(featureIndex[featureName].index(featureValue))
            else:
                line.append(featureValue)
        result.append(line)
    return result


dataIndex = index(title, valuesCast)
digitalized = digitalize(title, valuesCast, dataIndex, strFieldPositions )
print(digitalized)

exit()
frequency = {}
dataSize = len(values)
for key in dataIndex:
    l = len(dataIndex[key])
    p = int(l/dataSize * 10000) / 100
    frequency[key] = p

frequency_sorted = sorted(frequency.items(), key=operator.itemgetter(1))
threshold = 0.0

for key in frequency_sorted:
    if key[0] == 'Id':
        continue
    if key[1] > threshold:
        print(key[0], key[1], dataIndex[key[0]], '\n')

for k, v in zip(title, values[5]):
    print(k, v)