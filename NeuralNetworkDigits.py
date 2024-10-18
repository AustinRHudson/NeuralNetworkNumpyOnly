import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class layer:
    def __init__(self, numInputs, numNeurons):
        self.weights = np.random.rand(numInputs, numNeurons) - 0.5
        self.biases = np.random.randn(1, numNeurons)
    def forward(self, inputs):
        #print(inputs.shape, self.weights.shape, self.biases.shape)
        self.outputs = inputs.dot(self.weights) + self.biases

class ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(inputs, 0)

class softmax:
    def forward(self, x):
        z = x - np.max(x)
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        softmax = numerator / denominator
        self.outputs = softmax

def ReLUDerivative(x):
    return x > 0
def oneHotEncode(Y):
    oneHotY = np.zeros((Y.size, 10))
    oneHotY[np.arange(Y.size), Y] = 1
    oneHotY = oneHotY.T
    return oneHotY

def forwardProp(dense1, dense2, activation1, activation2, inputs):
    dense1.forward(inputs)
    activation1.forward(dense1.outputs)
    dense2.forward(activation1.outputs)
    activation2.forward(dense2.outputs)

def backProp(dense1, dense2, activation1, activation2, inputs, targets):
    m = targets.size
    oneHotY = oneHotEncode(targets)
    dZ2 = activation2.outputs.T - oneHotY
    dW2 = 1/m * dZ2.dot(activation1.outputs)
    db2 = 1/m * np.sum(dZ2)
    dZ1 = dense2.weights.dot(dZ2) * ReLUDerivative(dense1.outputs).T
    dW1 = 1/m * dZ1.dot(inputs)
    db1 = 1/m * np.sum(dZ1)
    #print(dW1.shape, db1.shape, dW2.shape, db2.shape)
    return dW1, db1, dW2, db2

def updateParameters(dense1, dense2, dW1, db1, dW2, db2, alpha):
    dense1.weights = dense1.weights - alpha * dW1.T
    dense1.biases = dense1.biases - alpha * db1
    dense2.weights = dense2.weights - alpha * dW2.T
    dense2.biases = dense2.biases - alpha * db2

def get_predictions(A2):
    return np.argmax(A2[0], 0)

def makePrediction(layer1, layer2, activation1, activation2, input):
    Xpredict = np.zeros((1, 784))
    Xpredict[0] = input
    forwardProp(layer1, layer2, activation1, activation2, input)
    return get_predictions(activation2.outputs)


def testPrediction(index):
    currentImage = Xtrain.T[:, index, None]
    currentImage = currentImage.reshape((28, 28)) * 255
    print(makePrediction(layer1, layer2, activation1, activation2, Xtrain[index]), ytrain[index])
    plt.title(makePrediction(layer1, layer2, activation1, activation2, Xtrain[index]))
    plt.gray()
    plt.imshow(currentImage, interpolation="nearest")
    plt.show()

data = pd.read_csv("train.csv")
data = np.array(data)
m, n = data.shape

data_train = data[1000:m].T
ytrain = data_train[0]
Xtrain = data_train[1:n]
Xtrain = Xtrain / 255.0
Xtrain = Xtrain.T

# Xtest = np.zeros((1, 784))
# Xtest[0] = Xtrain[1]

layer1 = layer(784, 10)
activation1 = ReLU()
layer2 = layer(10, 10)
activation2 = softmax()

iterations = 40000
numCorrect = 0
total = 0
sets = 1

#np.random.shuffle(Xtrain)

#plt.ion()

for shuffle in range(sets):
    for i in range(iterations):
        Xtest = np.zeros((1, 784))
        Xtest[0] = Xtrain[i]
        forwardProp(layer1, layer2, activation1, activation2, Xtest)
        dW1, db1, dW2, db2 = backProp(layer1, layer2, activation1, activation2, Xtest, ytrain[i])
        updateParameters(layer1, layer2, dW1, db1, dW2, db2, 0.008)
        #print(get_predictions(activation2.outputs) ,ytrain[i])
        if(get_predictions(activation2.outputs) == ytrain[i]):
            numCorrect += 1
        total += 1
    #np.random.shuffle(Xtrain.T)

#testNumber = input("What prediction?")
# while(not(testNumber == -1)):
#testPrediction(int(testNumber))
#     time.sleep(1)
#     testNumber = input("What prediction?")

#print(numCorrect/total)