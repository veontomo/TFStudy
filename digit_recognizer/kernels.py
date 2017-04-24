# https://www.kaggle.com/chunfuyeh/digit-recognizer/neural-network-with-tensorflow
# Train data: https://www.kaggle.com/c/digit-recognizer/download/train.csv
# Test  data: https://www.kaggle.com/c/digit-recognizer/download/test.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from os.path import isfile, isdir
import os
# matplotlib inline

dirName = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")

# load in train & test csv
train = pd.read_csv(dirName + '/train.csv')
test = pd.read_csv(dirName + '/test.csv')



# check training data
trainLabelCounts = train['label'].value_counts(sort = False)

print(trainLabelCounts)

# check training data - plot
def getImage(data, *args):
    '''
    Get the image by specified number (Randomly)
    parameters:
        data: dataframe
        number: int, the label of the number to show
    output: 1-D numpy array
    '''
    if args:
        number = args[0]
        specifiedData = data[data['label'] == number].values
    else:
        specifiedData = data.values
    randomNumber = np.random.choice(len(specifiedData)-1, 1)
    return specifiedData[randomNumber,:]
    
def plotNumber(imageData, imageSize):
    '''
    parameters:
        data: label & 1-D array of pixels
    '''
    # show the image of the data
    if imageData.shape[1] == np.prod(imageSize):
        image = imageData[0,:].reshape(imageSize)
    elif imageData.shape[1] > np.prod(imageSize):
        label = imageData[0,0]
        image = imageData[0,1:].reshape(imageSize)
        plt.title('number: {}'.format(label))
    plt.imshow(image)

imageSize = (28, 28)
chosenNumber = 1
plotNumber(getImage(train, chosenNumber), imageSize)



# cast to numpy array
trainData = train.values[:,1:]
trainLabel = train.values[:,0]
testData = test.values

def preprocessing(data):
    '''
    min-max scaling for every image
    data: numpy array
    output: scaled numpy array
    '''
    minV = 0
    maxV = 255
    data = (data - minV) / (maxV-minV)
    return data
def one_hot_encoding(data, numberOfClass):
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(range(numberOfClass))
    return lb.transform(data)




processedTrainData = preprocessing(trainData)
processedTestData = preprocessing(testData)
one_hot_trainLabel = one_hot_encoding(trainLabel, 10)



# save in pickle
fileName = 'mnist.p'
if not isfile(fileName):
    pickle.dump((processedTrainData, trainLabel, one_hot_trainLabel, processedTestData), open(fileName, 'wb'))

# load pickle file
fileName = 'mnist.p'
trainData, trainLabel, one_hot_trainLabel, testData = pickle.load(open(fileName, mode = 'rb'))

def getInputTensor(features, numberOfClass):
    '''
    Create tf.placeholder for input & label
    '''
    inputT = tf.placeholder(dtype = tf.float32, shape = (None, features), name = 'input')
    labelT = tf.placeholder(dtype = tf.float32, shape = (None, numberOfClass), name = 'label')
    keep_prob = tf.placeholder(dtype = tf.float32)
    
    return inputT, labelT, keep_prob

def hiddenLayer(inputT, numberOfNodes):
    '''
    Create hidden layer
    '''
    inputSize = inputT.get_shape().as_list()[1]
    # create weights & biases
    weights = tf.Variable(tf.truncated_normal((inputSize, numberOfNodes)), dtype = tf.float32)
    biases = tf.zeros((numberOfNodes), dtype = tf.float32)
    # output of hidden nodes
    hiddenNodes = tf.add(tf.matmul(inputT, weights), biases)
    hiddenOutput = tf.nn.sigmoid(hiddenNodes)
    
    return hiddenOutput

def outputLayer(hiddenOutput, numberOfClass):
    '''
    Create output layer
    hiddenOutput: output from hidden layer
    numOfClass: number of classes (0~9)
    '''
    inputSize = hiddenOutput.get_shape().as_list()[1]
    # create weights & biases
    weights = tf.Variable(tf.truncated_normal((inputSize, numberOfClass)), dtype = tf.float32)
    biases = tf.zeros((numberOfClass), dtype = tf.float32)
    # output
    output = tf.add(tf.matmul(hiddenOutput, weights), biases)
    
    return output

def build_nn(inputT, numberOfNodes, numberOfClass, keep_prob):
    '''
    build fully connected neural network
    '''
    # fully_connected layers
    fc1 = hiddenLayer(inputT, numberOfNodes)
    output = outputLayer(fc1, numberOfClass)
    
    return output

numberOfNodes = 256
batchSize = 128
numberOfEpoch = 200
learningRate = 0.01
keep_prob_rate = 1.0

# Build Neural Network graph
numberOfClass = 10
imageSize = (28, 28)
features = np.prod(imageSize)
graph = tf.Graph()
tf.reset_default_graph()
with graph.as_default():
    
    # get inputs
    inputT, labelT, keep_prob = getInputTensor(features, numberOfClass)
    
    # build fully-conneted neural network
    logits = build_nn(inputT, numberOfNodes, numberOfClass, keep_prob)
    
    # softmax with cross entropy
    probability = tf.nn.softmax(logits, name = 'probability')
    
    # Cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labelT))
    
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = learningRate).minimize(cost)
    
    # accuracy
    correctPrediction = tf.equal(tf.argmax(probability, 1),tf.argmax(labelT, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

from sklearn.model_selection import train_test_split

def printResult(epoch, numberOfEpoch, trainLoss, validationLoss, validationAccuracy):
    print("Epoch: {}/{}".format(epoch+1, numberOfEpoch),
         '\tTraining Loss: {:.3f}'.format(trainLoss),
         '\tValidation Loss: {:.3f}'.format(validationLoss),
         '\tAccuracy: {:.2f}%'.format(validationAccuracy*100))

save_dir = './save'
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(numberOfEpoch):
        # training data & validation data
        train_x, val_x, train_y, val_y = train_test_split(trainData, one_hot_trainLabel, test_size = 0.2)   
        # training loss
        for i in range(0, len(train_x), batchSize):
            trainLoss, _, _ = sess.run([cost, probability, optimizer], feed_dict = {
                inputT: train_x[i: i+batchSize],
                labelT: train_y[i: i+batchSize],
                keep_prob: keep_prob_rate   
            })
            
        # validation loss
        valAcc, valLoss = sess.run([accuracy, cost], feed_dict ={
            inputT: val_x,
            labelT: val_y,
            keep_prob: 1.0
        })
        
        
        # print out
        printResult(epoch, numberOfEpoch, trainLoss, valLoss, valAcc)
    # save
    saver = tf.train.Saver()
    saver.save(sess, save_dir)

