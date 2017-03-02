# Count how many times a binary input contains 1
# 
# See
# 
# http://peterroelants.github.io/posts/rnn_implementation_part01/
# 
import os
from keras.models import Sequential
from keras.layers import Dense, GRU, Activation, LSTM, TimeDistributed
#from keras.optimizers import SGD
#import random
#import math 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# from keras.utils.visualize_util import plot

dirName = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")
os.system('cls')


