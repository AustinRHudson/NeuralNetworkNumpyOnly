import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from ImageRandomization import randomNoise


#The "layer" class is a class to make a new layer object allowing for a more object oriented neural network with the possibiltiy of making a network as big as you want.
#The layer class takes in the number of inputs from the previous layer/samples and the number of neurons for weights and biases that you wish the layer to have.
#For all the classes, "forward" is the usual method call to use that layer/activation to do calculations.
class layer:
    def __init__(self, numInputs, numNeurons):
        #Initializing random weights and biases according to number of inputs and numNeurons.
        self.weights = np.random.rand(numInputs, numNeurons) - 0.5
        self.biases = np.random.randn(1, numNeurons)
    def forward(self, inputs):
        #Dot product of the inputs and weights while adding the biases to get appropriate output.
        self.outputs = inputs.dot(self.weights) + self.biases

#The "ReLU" class is a class used to make an object that will perform calculations on the previous layers output using a "Rectified Linear" activation function and storing it for later.
class ReLU:
    def forward(self, inputs):
        #Turns all negative inputs from previous layer into 0.
        self.outputs = np.maximum(inputs, 0)

#The "softmax" class is a class used to make an object that will perform calculations on the previous layers output using a "Softmax" activation function and storing it for later.
class softmax:
    def forward(self, x):
        #Subtracting by the max helps ensure no overflow errors which would destroy every subsequent calculation.
        z = x - np.max(x)
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        #Takes the exponential input of each row over the sum of the exponential input of each row, or (exponential(input[i])/sum(exponential(input))).
        softmax = numerator / denominator
        self.outputs = softmax

#A class designed to hold the index of the current sample the gui is looking at because tkinter doesn't allow returns on method calls on button press.
class guiIndex:
    def __init__(self, index):
        #Intialized index
        self.index = index

#Method uses during back propagation when taking the derivative of activation function to find first layer weights/biases.
def ReLUDerivative(x):
    #Derivative of ReLU function is 1 if X>0 and 0 if X<=0. The boolean is converted automatically into 1 and 0 which is perfect.
    return x > 0

#Method used to categorize the correct output and subtract from what the network outputted to determine loss.
def oneHotEncode(Y):
    #Makes an array of zeroes that has the dimensions of (sampleAmount, output amount).
    oneHotY = np.zeros((Y.size, 10))
    #Makes the correct sample of that column a 1 and the rest stay zero.
    oneHotY[np.arange(Y.size), Y] = 1
    #Transpose to be usable in back propagation.
    oneHotY = oneHotY.T
    return oneHotY

#This method just simply uses the forward method of each layer and activation and feeds them into eachother.
def forwardProp(dense1, dense2, activation1, activation2, inputs):
    dense1.forward(inputs)
    activation1.forward(dense1.outputs)
    dense2.forward(activation1.outputs)
    activation2.forward(dense2.outputs)

#Calculates loss and how much each parameter contributed to that loss by taking the derivative of those parameters with respect to the loss function.
def backProp(dense1, dense2, activation1, activation2, inputs, targets):
    m = targets.size
    oneHotY = oneHotEncode(targets)
    dZ2 = activation2.outputs.T - oneHotY
    dW2 = 1/m * dZ2.dot(activation1.outputs)
    db2 = 1/m * np.sum(dZ2)
    dZ1 = dense2.weights.dot(dZ2) * ReLUDerivative(dense1.outputs).T
    dW1 = 1/m * dZ1.dot(inputs)
    db1 = 1/m * np.sum(dZ1)
    return dW1, db1, dW2, db2

#Calculates the new weights and biases of each layer by taking the derivative value from back propagation and multiplying by alpha and subtracting
# the old weights and biases from the new ones. The weights and biases of each respective layer are set to their new values. The alpha value is a
# learning rate value that is designed to lower the amount the weights and biases are changed by to prevent overshooting
def updateParameters(dense1, dense2, dW1, db1, dW2, db2, alpha):
    dense1.weights = dense1.weights - alpha * dW1.T
    dense1.biases = dense1.biases - alpha * db1
    dense2.weights = dense2.weights - alpha * dW2.T
    dense2.biases = dense2.biases - alpha * db2

#Returns the high probablity digit that the network predicted from the last activation's output (Which is this case is a softmax activation).
def get_predictions(A2):
    return np.argmax(A2[0], 0)

#Does forward prop of chose sample and calls get_predictions to get the digit the network thinks it is.
def makePrediction(layer1, layer2, activation1, activation2, input):
    Xpredict = np.zeros((1, 784))
    Xpredict[0] = input
    forwardProp(layer1, layer2, activation1, activation2, input)
    return get_predictions(activation2.outputs)

#Tests the prediction of the sample given in the test data by resizing the array into 28x28 pixel array and returning color values to what they were.
#Shows the picture of the sample given along with its label and what the network predicted.
def testPredictionTest(index):
    currentImage = XTesting.T[:, index, None]
    currentImage = currentImage.reshape((28, 28)) * 255
    plt.title("Network: " + str(makePrediction(layer1, layer2, activation1, activation2, XTesting[index])))
    plt.gray()
    plt.imshow(currentImage, interpolation="nearest")
    plt.show()
    pass

#Tests the prediction of the sample given in the train data by resizing the array into 28x28 pixel array and returning color values to what they were.
#Shows the picture of the sample given along with its label and what the network predicted.
def testPredictionTrain(index):
    currentImage = Xtrain.T[:, index, None]
    currentImage = currentImage.reshape((28, 28)) * 255
    plt.title(
        "Network: " + str(makePrediction(layer1, layer2, activation1, activation2, Xtrain[index])) + " | Label: " + str(ytrain[index]))
    plt.gray()
    plt.imshow(currentImage, interpolation="nearest")
    plt.show()

#Takes the sample given and opens it in a tkinter application by calling plot.
def openLabelGUI(index):
    currentImage = Xtrain.T[:, index, None]
    currentImage = currentImage.reshape((28, 28)) * 255
    plot(currentImage, index, makePrediction(layer1, layer2, activation1, activation2, XTesting[index]))

#Takes the array of the next label and returns it along with the prediction of network.
def openNextLabel(index):
    nextImage = XTesting.T[:, index, None]
    nextImage = nextImage.reshape((28, 28)) * 255
    return nextImage, makePrediction(layer1, layer2, activation1, activation2, XTesting[index])

#Makes sure the index isn't at the beginning or end of sample list and grabs the array of the next/previous label and displays it and the network prediction.
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

#The tkinter application that is running to scan through test labels. The application contains the label picture,
# two buttons for going forward and back, and a label of the network's prediction.
def plot(testImage, index, prediction):
    GUIIndex = guiIndex(index)
    window = tk.Tk()
    window.title("Neural Network Testing")
    window.geometry("600x600")
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

#Shuffles the data's order and applies random noise to the data to make the network better at recognizing images outside the datasets.
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
sets = 10

for shuffle in range(sets):
    for i in range(iterations):
        Xtest = np.zeros((1, 784))
        Xtest[0] = Xtrain[i]
        forwardProp(layer1, layer2, activation1, activation2, Xtest)
        dW1, db1, dW2, db2 = backProp(layer1, layer2, activation1, activation2, Xtest, ytrain[i])
        updateParameters(layer1, layer2, dW1, db1, dW2, db2, 0.01)
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
