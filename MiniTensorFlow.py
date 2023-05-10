import numpy as np



class Dense:
    
    # Layers and its properties initialization.
    def __init__(self, units, activation, input_shape=0):
        self.units = units
        self.input_shape = input_shape
        self.activation = activation


        
class Sequential:
    
    
    def __init__(self, layers):
        """
        Initializes the parameters of the neural network and defines the dimensions of different layers.

        Arguments:
        layers - List of Dense layers 
        """
        # Hyperparameters of neural network
        self.layers = layers
        self.layer_dims = []  # layer_dims as list -- [iput_shape,...units for hidden layers..., output layer units] -- len = L
        self.activation = []  # activation for layer as list -- [...activation for hidden layers..., output layer activation] -- Len = L-1
        self.costs = []       # cost as list -- len = epochs
        
        # Parameters of neural network -- Dict with key: 0 - input layer | 1,2...L-1 - Other layers
        self.weights = {}     # Wieghts for each layer, keys:1,2...L-1
        self.bias = {}        # bias for each layer, keys:1,2...L-1
        self.Z = {}           # Z for each layer, keys:1,2...L-1
        self.A = {}           # A for each layer, keys:0,1,2...L-1
        self.dZ = {}          # grad of Z for each layer, keys:1,2...L-1
        self.dw = {}          # grad of weights for each layer, keys:1,2...L-1
        self.db = {}          # grad of bias for each layer, keys:1,2...L-1
        
        for layer in layers:
            
            # instantiate Dense to extract units, input_shape if any, and activation functions for each layer
            if layer.input_shape != 0: 
                self.layer_dims.append(layer.input_shape) 
                self.layer_dims.append(layer.units)
                self.activation.append(layer.activation)
            else:
                self.layer_dims.append(layer.units)
                self.activation.append(layer.activation)
    
    
    def fit(self, X, y, epochs, learning_rate, verbose=0):
        """
        Implements a artificial neural network algorithm.

        Arguments:
        X -- input data, of shape (input_shape, number of examples)
        Y -- true "label" vector, of shape (1, number of examples)
        learning_rate -- learning rate of the gradient descent update rule
        epochs -- number of iterations of the optimization loop
        verbose -- if 0, silent training else if 1, it prints cost after every 100 iterations
        """
        L = len(self.layer_dims) # number of layers in the neural network
        self.initialize_parameters()
        
        for i in range(epochs):
            self.forward_prop(X)
            cost = self.compute_cost(self.A[L - 1], y, self.activation[L - 2])
            self.costs.append(cost)
            self.backward_prop(y)
            self.update_parameters(learning_rate)
            
            if verbose and i % 100 == 0 or i == epochs - 1:
                print("Epochs {}/{} - Loss: {}".format(i, epochs, np.squeeze(cost)))
            
              
    def forward_prop(self, X):
        """
        Implement forward propagation for the neural network

        Arguments:
        X -- data, numpy array of shape (input_size, number of examples)
        """
        L = len(self.layer_dims) # number of layers in the neural network
        self.A[0] = X
        
        for l in range(1,L):
            self.Z[l] = self.weights[l].dot(self.A[l - 1]) + self.bias[l]
            
            if self.activation[l - 1] == "sigmoid":
                self.A[l] = self.sigmoid(self.Z[l])
            elif self.activation[l - 1] == "relu":
                self.A[l] = self.relu(self.Z[l])
            elif self.activation[l - 1] == "linear":
                self.A[l] = self.Z[l]
    
    
    def backward_prop(self,y):
        """
        Implement backward propagation for the neural network

        Arguments:
        y -- true "label" vector, numpy array of shape (input_size, number of examples)
        """
        L = len(self.layer_dims) # number of layers in the neural network
        m = self.A[L - 1].shape[1]
        self.dZ[L - 1] = self.A[L - 1] - y
        self.dw[L - 1] = (1./m) * np.dot(self.dZ[L - 1], self.A[L - 2].T)
        self.db[L - 1] = (1./m) * np.sum(self.dZ[L - 1], axis = 1, keepdims = True)
        
        for l in reversed(range(1,L - 1)):
            
            if self.activation[l - 1] == "sigmoid":
                self.dZ[l] = np.dot(self.weights[l + 1].T, self.dZ[l + 1]) * self.sigmoid_grad(self.Z[l]) 
                self.dw[l] = (1./m) * np.dot(self.dZ[l], self.A[l - 1].T)
                self.db[l] = (1./m) * np.sum(self.dZ[l], axis = 1, keepdims = True)
            elif self.activation[l - 1] == "relu":  
                self.dZ[l] = np.dot(self.weights[l + 1].T, self.dZ[l + 1]) * self.relu_grad(self.Z[l]) 
                self.dw[l] = (1./m) * np.dot(self.dZ[l], self.A[l - 1].T)
                self.db[l] = (1./m) * np.sum(self.dZ[l], axis = 1, keepdims = True)

            
    def evaluate(self, X_test, y_test):
        """
        This function is used to evaluate the predictions of the trained neural network.

        Arguments:
        X_test -- data set of examples you would like to label
        y_test -- labels of the test data

        Returns:
        [cost, accuracy] -- Cost and accuracy of the predictions for the given dataset X_test
        """
        m = X_test.shape[1]
        L = len(self.layer_dims) # number of layers in the neural network
        pred = np.zeros((1, m))
        self.forward_prop(X_test)
        pred[self.A[L - 1] > 0.5] = 1
        pred[self.A[L - 1] <= 0.5] = 0
        cost = self.compute_cost(self.A[L - 1], y_test, self.activation[L - 2])
        accuracy = np.sum((pred == y_test) / m)
        print("Loss: {} - Accuracy: {}".format(cost, accuracy))
        return [cost, accuracy]
    
    
    def predict(self, X_test):
        """
        This function is used to predict the results of the trained neural network.

        Arguments:
        X_test -- data set of examples you would like to label

        Returns:
        Pred -- Predictions for the given dataset X_test
        """
        m = X_test.shape[1]
        L = len(self.layer_dims) # number of layers in the neural network
        pred = np.zeros((1, m))
        self.forward_prop(X_test)
        pred[self.A[L - 1] > 0.5] = 1
        pred[self.A[L - 1] <= 0.5] = 0
        return pred
        
                
    def initialize_parameters(self):
        np.random.seed(1)
        L = len(self.layer_dims) # number of layers in the neural network
        
        for l in range(1,L):
            self.weights[l] = np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) / np.sqrt(self.layer_dims[l - 1])
            self.bias[l] = np.zeros((self.layer_dims[l],1))
    
    
    def update_parameters(self,learning_rate):
        """
        Update parameters using gradient descent
        """
        L = len(self.layer_dims) # number of layers in the neural network
        
        for l in range(1,L):
            self.weights[l] = self.weights[l] - learning_rate * self.dw[l]
            self.bias[l] = self.bias[l] - learning_rate * self.db[l]

            
    @staticmethod
    def sigmoid(Z):
        """
        Implements the sigmoid activation in numpy

        Arguments:
        Z -- numpy array of any shape
        """
        g = 1 / (1 + np.exp(-Z))
        return (g)
    
    
    @staticmethod
    def relu(Z):
        """
        Implements the relu activation in numpy

        Arguments:
        Z -- numpy array of any shape
        """
        g = np.maximum(0,Z)
        return (g)
      
        
    @staticmethod
    def compute_cost(A, y, activation):
        """
        Implement the cost function

        Arguments:
        A -- probability vector corresponding to your label predictions, shape (1, number of examples)
        y -- true "label" vector, shape (1, number of examples)

        Returns:
        cost 
        """
        m = y.shape[1]
        
        if activation == 'sigmoid':
            cost = (1./m) * (-np.dot(y, np.log(A).T) - np.dot(1 - y, np.log(1 - A).T))
        elif activation == 'linear':
            cost = (1./(2*m)) * np.sum((A - y)**2)
            
        cost = np.squeeze(cost)
        return cost
    
    
    @staticmethod
    def relu_grad(Z):
        """
        Implements the gradient of relu activation in numpy

        Arguments:
        Z -- numpy array of any shape
        """
        dg = np.array(Z, copy=True)  
        dg[Z>0] = 1
        dg[Z<=0] = 0
        return dg
    
    
    @staticmethod
    def sigmoid_grad(Z):
        """
        Implements the gradient of sigmoid activation in numpy

        Arguments:
        Z -- numpy array of any shape
        """
        
        return np.exp(-Z) / (np.exp(-Z) + 1)**2
