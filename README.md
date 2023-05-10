# MiniTensorFlow
MiniTensorFlow is a Python library that aims to replicate some of the functionality of Google's TensorFlow library for deep learning. Currently, it includes custom-coded classes for Dense and Sequential layers with all necessary methods implemented.

## Getting Started
To use MiniTensorFlow, you will need to have the following dependencies installed:

* numpy
* matplotlib
* h5py
* scipy
* PIL (from Image)

Once you have these dependencies installed, you can run the **'Catvsnoncat.ipynb'** file to train an ANN model to identify cat and non-cat pictures. This file imports the necessary classes from the **'MiniTensorFlow.py'** file and uses them to train the model.

## Usage

Here is an example of how to use MiniTensorFlow to create a simple neural network:

```python
from MiniTensorFlow import Sequential, Dense

# Create a Sequential model
model = Sequential([
  Dense(64, input_shape=12288, activation='relu')       # Add a Dense layer with 64 units and a ReLU activation function also a input_layer having 12288 units is created
  Dense(1, activation='sigmoid')                        # Add another Dense layer with 1 unit and a sigmoid activation function
])

# Train the model on your data
model.fit(X_train, y_train, epochs=10, learning_rate=0.0075)

# Evaluate the model on your test data
loss, accuracy = model.evaluate(X_test, y_test)

# Get predictions for test data
prediction = model.predict(X_test)
```
In this example, we create a Sequential model with two Dense layers with ReLU and sigmoid activation functions, respectively. We then compile the model with binary cross-entropy loss and the Adam optimizer, and train it on our data. Finally, we evaluate the model on our test data and compute the loss and accuracy.

## Contributions

We welcome contributions to MiniTensorFlow! If you would like to add new features or improve existing ones, please feel free to submit a pull request. Some possible areas for expansion include:

* Implementing model.compile() method
* Dropout layers
* Pooling layers
* Optimizers (e.g., Adam, SGD)
* Regularizers (e.g., L1, L2)

We hope that MiniTensorFlow can serve as a useful tool for learning about the mathematics behind neural networks and gaining confidence in OOP coding skills.
