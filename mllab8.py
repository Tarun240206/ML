#1
import numpy as np

def summation_unit(inputs, weights, bias=0):
    
    if len(inputs) != len(weights):
        raise ValueError("Inputs and weights must have same length")
    
    total = 0
    for i in range(len(inputs)):
        total += inputs[i] * weights[i]
    
    total += bias
    return total

import math

def summation(x, w, bias):
    return np.dot(x, w) + bias


def step(x):
    return 1 if x >= 0 else 0



def bipolar_step(x):
    return 1 if x >= 0 else -1



def sigmoid(x):
    return 1 / (1 + math.exp(-x))



def tanh(x):
    return math.tanh(x)



def relu(x):
    return max(0, x)



def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x

def error_calculation(target, output):
   
    return target - output


def mean_squared_error(targets, outputs):
    
    if len(targets) != len(outputs):
        raise ValueError("Targets and outputs must have same length")
    
    error = 0
    for t, o in zip(targets, outputs):
        error += (t - o) ** 2
    
    return error / len(targets)

inputs = [1, 2, 3]
weights = [0.5, -1, 2]
bias = 1

net_input = summation_unit(inputs, weights, bias)
print("Net Input:", net_input)


output = sigmoid(net_input)
print("Sigmoid Output:", output)


target = 1
error = error_calculation(target, output)
print("Error:", error)


targets = [1, 0, 1]
outputs = [0.8, 0.2, 0.9]
print("MSE:", mean_squared_error(targets, outputs))

#2
def train_perceptron_AND():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1])

    w = np.array([0.2, -0.75])
    b = 10
    lr = 0.05

    epochs = 0
    errors = []

    while epochs < 1000:
        total_error = 0
        for i in range(len(X)):
            y_pred = step(summation(X[i], w, b))
            e = error(y[i], y_pred)
            w += lr * e * X[i]
            b += lr * e
            total_error += e**2

        errors.append(total_error)
        epochs += 1

        if total_error <= 0.002:
            break

    return w, b, epochs, errors

#3
def train_with_activation(act_func):
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1])

    w = np.array([0.2, -0.75])
    b = 10
    lr = 0.05

    epochs = 0

    while epochs < 1000:
        total_error = 0
        for i in range(len(X)):
            y_pred = act_func(summation(X[i], w, b))
            e = y[i] - y_pred
            w += lr * e * X[i]
            b += lr * e
            total_error += e**2

        epochs += 1
        if total_error <= 0.002:
            break

    return epochs

#4
def test_learning_rates():
    rates = [i/10 for i in range(1,11)]
    results = []

    for lr in rates:
        X = np.array([[0,0],[0,1],[1,0],[1,1]])
        y = np.array([0,0,0,1])
        w = np.array([0.2, -0.75])
        b = 10

        epochs = 0

        while epochs < 1000:
            total_error = 0
            for i in range(len(X)):
                y_pred = step(summation(X[i], w, b))
                e = y[i] - y_pred
                w += lr * e * X[i]
                b += lr * e
                total_error += e**2

            epochs += 1
            if total_error <= 0.002:
                break

        results.append((lr, epochs))

    return results

#5
def train_XOR():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])

    w = np.random.rand(2)
    b = np.random.rand()
    lr = 0.05

    for epoch in range(1000):
        total_error = 0
        for i in range(len(X)):
            y_pred = step(summation(X[i], w, b))
            e = y[i] - y_pred
            w += lr * e * X[i]
            b += lr * e
            total_error += e**2

    return w, b

#6
def train_customer_data():
    X = np.array([
        [20,6,2],[16,3,6],[27,6,2],[19,1,2],[24,4,2],
        [22,1,5],[15,4,2],[18,4,2],[21,1,4],[16,2,4]
    ])
    y = np.array([1,1,1,0,1,0,1,1,0,0])

    w = np.random.rand(3)
    b = np.random.rand()
    lr = 0.01

    for epoch in range(1000):
        for i in range(len(X)):
            y_pred = sigmoid(summation(X[i], w, b))
            e = y[i] - y_pred
            w += lr * e * X[i]
            b += lr * e

    return w, b

#7
def pseudo_inverse_solution(X, y):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    weights = np.linalg.pinv(X_bias).dot(y)
    return weights

#8
def backprop_AND():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[0],[0],[1]])

    np.random.seed(0)
    W1 = np.random.rand(2,2)
    W2 = np.random.rand(2,1)
    lr = 0.05

    for epoch in range(1000):
        
        h = sigmoid(np.dot(X, W1))
        out = sigmoid(np.dot(h, W2))

        
        error = y - out

       
        d_out = error * out * (1 - out)
        d_h = d_out.dot(W2.T) * h * (1 - h)

        
        W2 += h.T.dot(d_out) * lr
        W1 += X.T.dot(d_h) * lr

    return W1, W2

#9
def backprop_XOR():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])

    np.random.seed(1)
    W1 = np.random.rand(2,2)
    W2 = np.random.rand(2,1)
    lr = 0.05

    for epoch in range(10000):
        h = sigmoid(np.dot(X, W1))
        out = sigmoid(np.dot(h, W2))

        error = y - out

        d_out = error * out * (1 - out)
        d_h = d_out.dot(W2.T) * h * (1 - h)

        W2 += h.T.dot(d_out) * lr
        W1 += X.T.dot(d_h) * lr

    return W1, W2

#10
def two_output_AND():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[1,0],[1,0],[1,0],[0,1]])

    W = np.random.rand(2,2)
    b = np.random.rand(2)
    lr = 0.05

    for epoch in range(1000):
        for i in range(len(X)):
            out = sigmoid(np.dot(X[i], W) + b)
            e = y[i] - out
            W += lr * np.outer(X[i], e)
            b += lr * e

    return W, b

#11
from sklearn.neural_network import MLPClassifier

def mlp_logic():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_and = np.array([0,0,0,1])
    y_xor = np.array([0,1,1,0])

    clf_and = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
    clf_and.fit(X, y_and)

    clf_xor = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
    clf_xor.fit(X, y_xor)

    return clf_and, clf_xor

#12
def mlp_on_dataset(X, y):
    model = MLPClassifier(hidden_layer_sizes=(5,5), max_iter=1000)
    model.fit(X, y)
    return model
