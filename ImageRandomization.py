import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
def randomNoise(dataSet, m, n, noise, angleRange):
    i = 0
    for sample in dataSet:
        index = dataSet.T[:, i, None]
        index = index.reshape((28, 28))
        index = rotate(index, angle=random.randrange(-angleRange, angleRange), reshape=False)
        index = index.flatten().T
        dataSet[i] = index
        i += 1
    noiseArray = (np.random.rand(m, n)-.5)/(1/noise)
    dataSet = dataSet + noiseArray
    dataSet = np.clip(dataSet, 0, 1)
    return dataSet

# def testPredictionTrain(index):
#     currentImage = Xtrain.T[:, index, None]
#     currentImage = currentImage.reshape((28, 28)) * 255
#     #currentImage = rotate(currentImage, angle=30, reshape=False)
#     plt.gray()
#     plt.imshow(currentImage, interpolation="nearest")
#     plt.show()
#
# data = pd.read_csv("train.csv")
# data = np.array(data)
# m, n = data.shape
# data_train = data
# data_train = data_train.T
# ytrain = data_train[0]
# Xtrain = data_train[1:n]
# Xtrain = Xtrain / 255.0
# Xtrain = Xtrain.T
# m, n = Xtrain.shape
# Xtrain = randomNoise(Xtrain, m, n, 1)
# testPredictionTrain(3)
