Neural Network From Scratch (NumPy Only)

A full-stack handwritten digit recognition application built from the ground up using Python, NumPy, Flask, and JavaScript. The neural network, including forward propagation, backpropagation, and gradient descent, was implemented entirely from scratch without relying on machine learning frameworks such as TensorFlow, PyTorch, or Keras.

The project includes an interactive web application where users can draw handwritten digits directly in their browser and receive real-time predictions from the trained neural network.

Why build a neural network from scratch?

Modern machine learning libraries abstract away much of the underlying mathematics. This project was created to develop a deeper understanding of how neural networks operate internally by implementing every stage of training and inference manually.

###############################################

Features:
Neural network implemented entirely with NumPy
Manual implementation of forward propagation
Manual backpropagation using the chain rule
Gradient descent optimization
Configurable network architecture
Model serialization for saving and loading trained weights
Trained on the MNIST handwritten digit dataset
Flask REST API for model inference
Interactive browser-based drawing canvas
Real-time handwritten digit recognition
Publicly deployed web application on Render

Live Application: neuralnetworknumpyonly.onrender.com

<p align="center">
  <img src="/demo.gif" width="700">
</p>

###############################################

Technologies:

Python
Flask
NumPy
HTML5
CSS3
JavaScript
Render

###############################################

Project Architecture
                Browser
      (HTML, CSS, JavaScript)
               │
               │ POST /predict
               ▼
         Flask REST API
               │
      Image Preprocessing
               │
      Resize → 28×28
      Convert to Grayscale
      Normalize Pixel Values
               │
               ▼
     NumPy Neural Network
               │
               ▼
        Predicted Digit
               │
               ▼
      JSON Response → Browser

###############################################
      
How It Works
1. Training

The network is trained using the MNIST handwritten digit dataset.

During training the model performs:

Forward propagation
Loss calculation
Backpropagation
Gradient descent weight updates

The learned parameters are then serialized so the model can be loaded without retraining.

2. Inference

When a user draws on the web interface:

The browser captures the drawing using an HTML5 canvas.
The image is sent to the Flask backend through a REST API.
The backend preprocesses the image into the 28×28 grayscale format expected by the model.
The trained neural network performs inference.
The predicted digit is returned as JSON and displayed in real time.

###############################################
