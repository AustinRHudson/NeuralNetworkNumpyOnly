import random
import numpy as np
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

