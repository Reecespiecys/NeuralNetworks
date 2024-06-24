"""
Developer: Reece Gilbert
Date Completed: 6/23/2024
Title: Trainable Neural Network
"""

import numpy as np
import random

random.seed(0)  # Setting a random seed for reproducibility


# Activation Functions
def ReLU(value):
    """
    Rectified Linear Unit activation function.
    Args:
    - value (float or numpy array): Input value or array.
    Returns:
    - float or numpy array: Output after applying ReLU.
    """
    return np.maximum(0, value)


def ReLU_derivative(value):
    """
    Derivative of the Rectified Linear Unit activation function.
    Args:
    - value (float or numpy array): Input value or array.
    Returns:
    - float or numpy array: Output derivative after applying ReLU.
    """
    return np.where(value > 0, 1, 0)


def tanh_derivative(value):
    """
    Derivative of the hyperbolic tangent activation function.
    Args:
    - value (float or numpy array): Input value or array.
    Returns:
    - float or numpy array: Output derivative after applying tanh.
    """
    return 1 - np.tanh(value) ** 2


# Loss Functions
def mean_squared_error(predictions, targets):
    """
    Mean Squared Error (MSE) loss function.
    Args:
    - predictions (numpy array): Predicted values.
    - targets (numpy array): Target values.
    Returns:
    - float: Mean squared error between predictions and targets.
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    return np.mean((predictions - targets) ** 2)


def mean_squared_error_derivative(predictions, targets):
    """
    Derivative of the Mean Squared Error (MSE) loss function.
    Args:
    - predictions (numpy array): Predicted values.
    - targets (numpy array): Target values.
    Returns:
    - numpy array: Derivative of MSE with respect to predictions.
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    return 2 * (predictions - targets) / targets.size


# Perceptron Class
class Perceptron:
    def __init__(self, num_weights, layer_type):
        """
        Initializes a Perceptron with random weights and bias.
        Args:
        - num_weights (int): Number of input weights.
        - layer_type (str): Type of layer ('Hidden' or 'Output').
        """
        self.num_weights = num_weights
        self.weights = [random.uniform(-1, 1) for _ in range(num_weights)]
        self.bias = random.uniform(-1, 1)
        self.current_value = 0.0
        self.layer_type = layer_type
        self.inputs = None

    def set_num_weights(self, num):
        """
        Sets the number of weights (inputs) for the perceptron.
        Args:
        - num (int): Number of input weights.
        """
        self.num_weights = num

    def get_num_weights(self):
        """
        Returns the number of weights (inputs) for the perceptron.
        Returns:
        - int: Number of input weights.
        """
        return self.num_weights

    def set_weights(self, weights):
        """
        Sets the weights of the perceptron.
        Args:
        - weights (list): List of weights.
        """
        self.weights = weights

    def get_weights(self):
        """
        Returns the weights of the perceptron.
        Returns:
        - list: List of weights.
        """
        return self.weights

    def set_bias(self, bias):
        """
        Sets the bias of the perceptron.
        Args:
        - bias (float): Bias value.
        """
        self.bias = bias

    def get_bias(self):
        """
        Returns the bias of the perceptron.
        Returns:
        - float: Bias value.
        """
        return self.bias

    def set_current_value(self, value):
        """
        Sets the current value of the perceptron.
        Args:
        - value (float): Current value.
        """
        self.current_value = value

    def get_current_value(self):
        """
        Returns the current value of the perceptron.
        Returns:
        - float: Current value.
        """
        return self.current_value

    def set_layer_type(self, layer_type):
        """
        Sets the layer type of the perceptron ('Hidden' or 'Output').
        Args:
        - layer_type (str): Type of layer.
        """
        self.layer_type = layer_type

    def get_layer_type(self):
        """
        Returns the layer type of the perceptron.
        Returns:
        - str: Type of layer ('Hidden' or 'Output').
        """
        return self.layer_type

    def update_weights(self, error, learning_rate):
        """
        Updates the weights of the perceptron based on backpropagated error.
        Args:
        - error (float): Error for weight adjustment.
        - learning_rate (float): Learning rate for gradient descent.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * error * self.inputs[i]
        self.bias -= learning_rate * error

    def predict(self, inputs):
        """
        Performs forward pass through the perceptron.
        Args:
        - inputs (list): List of input values.
        Returns:
        - float: Output value after applying activation function.
        """
        if len(inputs) != self.num_weights:
            raise ValueError("Number of inputs must match number of weights")
        self.inputs = inputs
        self.current_value = np.dot(self.weights, inputs) + self.bias

        if self.layer_type == 'Hidden':
            return ReLU(self.current_value)
        elif self.layer_type == 'Output':
            return np.tanh(self.current_value)
        else:
            raise ValueError("Not a recognized layer type")


# Network Class
class Network:
    def __init__(self, inputs, num_hidden, num_output, learning_rate):
        """
        Initializes a Neural Network with specified parameters.
        Args:
        - inputs (list): List of input values.
        - num_hidden (list): List of integers indicating number of neurons in hidden layers.
        - num_output (list): List of integers indicating number of neurons in output layer.
        - learning_rate (float): Learning rate for gradient descent.
        """
        self.inputs = inputs
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.hidden_layers = []
        self.output_layer = []
        self.error = 0.0
        self.learning_rate = learning_rate

    def set_inputs(self, inputs):
        """
        Sets the input values for the network.
        Args:
        - inputs (list): List of input values.
        """
        self.inputs = inputs

    def get_inputs(self):
        """
        Returns the input values of the network.
        Returns:
        - list: List of input values.
        """
        return self.inputs

    def set_learning_rate(self, rate):
        """
        Sets the learning rate for gradient descent.
        Args:
        - rate (float): Learning rate value.
        """
        self.learning_rate = rate

    def get_learning_rate(self):
        """
        Returns the learning rate of the network.
        Returns:
        - float: Learning rate value.
        """
        return self.learning_rate

    def get_loss(self):
        """
        Returns the current error (loss) of the network.
        Returns:
        - float: Current error (loss) value.
        """
        return self.error

    def generate_layers(self):
        """
        Initializes hidden and output layers of the network with perceptrons.
        """
        for i in range(len(self.num_hidden)):
            layer = []
            for _ in range(self.num_hidden[i]):
                if i == 0:
                    layer.append(Perceptron(len(self.inputs), 'Hidden'))
                else:
                    layer.append(Perceptron(self.num_hidden[i - 1], 'Hidden'))
            self.hidden_layers.append(layer)

        for i in range(self.num_output[0]):
            self.output_layer.append(Perceptron(self.num_hidden[-1], 'Output'))

    def feed_forward(self):
        """
        Performs forward propagation through the network.
        Returns:
        - list: List of output values from the output layer perceptrons.
        """
        for i in range(len(self.hidden_layers)):
            for perceptron in self.hidden_layers[i]:
                if i == 0:
                    perceptron.set_current_value(perceptron.predict(self.inputs))
                else:
                    inputs = [prev_perceptron.get_current_value() for prev_perceptron in self.hidden_layers[i - 1]]
                    perceptron.set_current_value(perceptron.predict(inputs))

        outputs = [
            perceptron.predict([prev_perceptron.get_current_value() for prev_perceptron in self.hidden_layers[-1]])
            for perceptron in self.output_layer]
        return outputs

    def backward_propagation(self, targets):
        """
        Performs backpropagation to update weights based on error gradients.
        Args:
        - targets (list): List of target values for the output layer.
        """
        network_outputs = self.feed_forward()
        output_errors = mean_squared_error_derivative(network_outputs, targets)

        # Update output layer weights
        for i, perceptron in enumerate(self.output_layer):
            error = output_errors[i] * tanh_derivative(perceptron.get_current_value())
            perceptron.update_weights(error, self.learning_rate)

        # Update hidden layers weights
        for i in reversed(range(len(self.hidden_layers))):
            next_layer = self.output_layer if i == len(self.hidden_layers) - 1 else self.hidden_layers[i + 1]
            for j, perceptron in enumerate(self.hidden_layers[i]):
                if i == len(self.hidden_layers) - 1:
                    error = sum(
                        next_perceptron.weights[j] * output_errors[k] for k, next_perceptron in enumerate(next_layer))
                else:
                    error = sum(next_perceptron.weights[j] * ReLU_derivative(next_perceptron.get_current_value()) for
                                next_perceptron in next_layer)
                perceptron.update_weights(error, self.learning_rate)

    def train_network(self, targets):
        """
        Trains the network by performing forward and backward propagation.
        Args:
        - targets (list): List of target values for the output layer.
        """
        network_outputs = self.feed_forward()
        self.error = mean_squared_error(network_outputs, targets)
        self.backward_propagation(targets)
        print(f"Outputs: {network_outputs}, Targets: {targets}, Error: {self.error}")


# Simple example case showing target matching with random inputs
if __name__ == '__main__':
    targets = [-1.0, 0.0]
    running = True
    gen_count = 0
    gen_limit = 100

    # Initialize network with random inputs and specified parameters
    network = Network(
        inputs=[random.uniform(-1, 1) for _ in range(100)],
        num_hidden=[25],
        num_output=[len(targets)],
        learning_rate=0.1
    )
    network.generate_layers()

    while running:
        network.train_network(targets)
        network.learning_rate *= 0.9999

        # Check if error is below a very small threshold
        if network.error < 1e-15:
            gen_count += 1
            if gen_count >= gen_limit:
                running = False
        else:
            gen_count = 0
