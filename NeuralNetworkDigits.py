import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from ImageRandomization import randomNoise

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

class guiIndex:
    def __init__(self, index):
        self.index = index
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


def testPredictionTest(index):
    currentImage = XTesting.T[:, index, None]
    currentImage = currentImage.reshape((28, 28)) * 255
    #print(makePrediction(layer1, layer2, activation1, activation2, Xtrain[index]), ytrain[index])
    plt.title("Network: " + str(makePrediction(layer1, layer2, activation1, activation2, XTesting[index])))
    plt.gray()
    plt.imshow(currentImage, interpolation="nearest")
    plt.show()
    pass

def testPredictionTrain(index):
    currentImage = Xtrain.T[:, index, None]
    currentImage = currentImage.reshape((28, 28)) * 255
    plt.title(
        "Network: " + str(makePrediction(layer1, layer2, activation1, activation2, Xtrain[index])) + " | Label: " + str(ytrain[index]))
    plt.gray()
    plt.imshow(currentImage, interpolation="nearest")
    plt.show()

def openLabelGUI(index):
    currentImage = Xtrain.T[:, index, None]
    currentImage = currentImage.reshape((28, 28)) * 255
    plot(currentImage, index, makePrediction(layer1, layer2, activation1, activation2, XTesting[index]))

def openNextLabel(index):
    nextImage = XTesting.T[:, index, None]
    nextImage = nextImage.reshape((28, 28)) * 255
    return nextImage, makePrediction(layer1, layer2, activation1, activation2, XTesting[index])

def nextLabel(nextIndex, label, numLabel, direction):
    print(nextIndex.index)
    if(direction < 0 and not(nextIndex.index == 0)):
        nextIndex.index = nextIndex.index - 1
    elif(direction > 0 and not(nextIndex.index == 27999)):
        nextIndex.index = nextIndex.index + 1
    else:
        return
    nextLabelImageArray, prediction = openNextLabel(nextIndex.index)
    nextLabelImage = Image.fromarray(nextLabelImageArray)
    resizedNextLabelImage = nextLabelImage.resize((448, 448), resample=Image.NEAREST)
    photo = ImageTk.PhotoImage(resizedNextLabelImage)
    label.configure(image=photo)
    label.image = photo
    numLabel.configure(text=str(prediction))
def plot(testImage, index, prediction):
    GUIIndex = guiIndex(index)
    window = tk.Tk()
    window.title("Neural Network Testing")
    window.geometry("800x800")
    image = Image.fromarray(testImage)
    resizedImage = image.resize((448, 448), resample=Image.NEAREST)
    photo = ImageTk.PhotoImage(resizedImage)
    label = tk.Label(window, image=photo)
    label.pack()
    numberLabel = tk.Label(window, text=str(prediction), font=("Arial", 25))
    numberLabel.pack()
    backButton = tk.Button(window, text="Previous label", command= lambda: nextLabel(GUIIndex, label, numberLabel, -1))
    backButton.pack()
    forwardButton = tk.Button(window, text="Next label", command= lambda: nextLabel(GUIIndex, label, numberLabel, 1))
    forwardButton.pack()
    window.mainloop()
def shuffleData(data, n):
    data_train = data
    np.random.shuffle(data_train)
    data_train = data_train.T
    ytrain = data_train[0]
    Xtrain = data_train[1:n]
    Xtrain = Xtrain / 255.0
    Xtrain = Xtrain.T
    m, n = Xtrain.shape
    Xtrain = randomNoise(Xtrain, m, n, .25, 45)
    return Xtrain, ytrain

testingData = pd.read_csv("test.csv")
testingData = np.array(testingData)
m, n = testingData.shape
data_test = testingData.T
#ytesting = data_test[0]
XTesting = data_test
XTesting = XTesting/255.0
XTesting = XTesting.T

data = pd.read_csv("train.csv")
data = np.array(data)
m, n = data.shape
Xtrain, ytrain = shuffleData(data, n)



layer1 = layer(784,100)
activation1 = ReLU()
layer2 = layer(100, 10)
activation2 = softmax()

iterations = 40000
numCorrect = 0
total = 0
sets = 1

for shuffle in range(sets):
    for i in range(iterations):
        Xtest = np.zeros((1, 784))
        Xtest[0] = Xtrain[i]
        forwardProp(layer1, layer2, activation1, activation2, Xtest)
        dW1, db1, dW2, db2 = backProp(layer1, layer2, activation1, activation2, Xtest, ytrain[i])
        updateParameters(layer1, layer2, dW1, db1, dW2, db2, 0.01)
        #print(get_predictions(activation2.outputs) ,ytrain[i])
        if(get_predictions(activation2.outputs) == ytrain[i]):
            numCorrect += 1
        total += 1
    Xtrain, ytrain = shuffleData(data, n)
    print("Epochs completed: " + str(((shuffle+1)/sets) * 100) + "%")
print(numCorrect/total)

# df = pd.DataFrame(layer1.weights)
# df.to_csv("weights.csv", header=False, index=False)
# df = pd.read_csv("weights.csv")
# print(layer1.weights.shape, df)

openLabelGUI(int(100))

#testing = True
# while(testing):
#     testNumber = input("What prediction?")
#     if(int(testNumber) != -1):
#         #testPredictionTest(int(testNumber))
#         openLabelGUI(int(testNumber))
#     else:
#         testing = False

img = Image.open("testDigit.png")
img = img.convert('L')
imgArray = np.array(img)
imgArray = imgArray/255
currentImage = imgArray
currentImage = currentImage.reshape((28, 28)) * 255
Xpredict = np.zeros((1, 784))
Xpredict[0] = imgArray.flatten()
print(makePrediction(layer1, layer2, activation1, activation2, Xpredict))
print(activation2.outputs)
plt.gray()
plt.imshow(currentImage, interpolation="nearest")
#plt.show()
