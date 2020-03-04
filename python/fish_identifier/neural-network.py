import pdb
import sys
import math
import time
import argparse
import traceback
import idx2numpy
import numpy as np
from pathlib import Path


def init_params():
    parser = argparse.ArgumentParser()

    # hyperparameters setting
    parser.add_argument('--alpha', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--decay', type=float, default=0.00,
                        help='learning rate decay')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train')
    parser.add_argument('--n_x', type=int, default=784,
                        help='number of inputs')
    parser.add_argument('--n_h', type=int, default=400,
                        help='number of hidden units')
    parser.add_argument('--n_o', type=int, default=10,
                        help='number of output units')
    parser.add_argument('--beta', type=float, default=0.98,
                        help='parameter for momentum')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='input batch size')
    parser.add_argument('--batches', type=int, default=5,
                        help='batch iterations')
    return parser.parse_args()


def init_data():
    #Get file paths
    rootDir = Path(sys.path[0]).parent
    train_data = np.load('train_imgs.npy')
    labels_data = np.load('labels.npy')

    return train_data, labels_data


def init_weights(arch):
    weights = {
        "W1" : np.random.randn(arch[0][0], arch[0][1]) * np.sqrt(1 / arch[0][0]),
        "b1" : np.random.randn(1, arch[0][1]) * np.sqrt(1 / arch[0][0]),
        "W2" : np.random.randn(arch[1][0], arch[1][1]) * np.sqrt(1 / arch[1][0]),
        "b2" : np.random.randn(1, arch[1][1]) * np.sqrt(1 / arch[1][1])
        }
    
    return weights


def init_velocities(arch):
    velocities = {
        "W1" : np.zeros((arch[0][0], arch[0][1])),
        "b1" : np.zeros((1, arch[0][1])),
        "W2" : np.zeros((arch[1][0], arch[1][1])),
        "b2" : np.zeros((1, arch[1][1]))
        }

    return velocities

    
def train():
    #Get opts, data, weights, velocities
    train_data, labels_data = init_data()
    arch = ((opts.n_x, opts.n_h), (opts.n_h, opts.n_o))
    weights = init_weights(arch)
    velocities = init_velocities(arch)
    
    #Train for n epochs
    for j in range(opts.epochs + 1):

        #Get a random mini batch and shuffle up the original data set. Update alpha
        permutation = np.random.permutation(opts.batch_size * opts.batches)
        X_epoch = train_input[permutation]
        y_epoch = train_target[permutation]
        opts.alpha *= (1 / (1 + opts.decay * j))

        for k in range(opts.batches):
            #Move through the data set according to the batch size
            begin = k * opts.batch_size
            end = begin + opts.batch_size

            X = X_epoch[begin:end]
            y = y_epoch[begin:end]
            
            # Feed forward
            outputs = feed_forward(X, weights)
            
            # Backpropagate, get error as well
            output_error, deltas = back_propagation(weights, outputs, X, y)
            

            #Using velocities for momentum in SGD
            velocities['W2'] = opts.beta * velocities['W2'] + (1 - opts.beta) * deltas['dW2']
            velocities['b2'] = opts.beta * velocities['b2'] + (1 - opts.beta) * deltas['db2']
            velocities['W1'] = opts.beta * velocities['W1'] + (1 - opts.beta) * deltas['dW1']
            velocities['b1'] = opts.beta * velocities['b1'] + (1 - opts.beta) * deltas['db1']
    
            #Update weights
            weights['W2'] = weights['W2'] - opts.alpha * velocities['W2']
            weights['b2'] = weights['b2'] - opts.alpha * velocities['b2']
            weights['W1'] = weights['W1'] - opts.alpha * velocities['W1']
            weights['b1'] = weights['b1'] - opts.alpha * velocities['b1']
            
        # From time to time, reporting the results
        if (j % 5) == 0:
            train_error = np.mean(np.abs(output_error))
            print('Epoch {:5}'.format(j), end=' - ')
            print('error: {:0.4f}'.format(train_error), end= ' - ')

            train_accuracy = accuracy(target=y, predictions=(get_predictions(outputs, y)))
            test_preds = predict(test_input, test_target, weights)
            test_accuracy = accuracy(target=test_target, predictions=test_preds)

            print('acc: train {:0.3f}'.format(train_accuracy), end= ' | ')
            print('test {:0.3f}'.format(test_accuracy), end= ' | ')
            print('alpha {:0.6f}'.format(opts.alpha))
            

    
def feed_forward(inputs, weights):
    #Empty return dict
    outputs = {}
    
    #Dot product of input value and weight
    z1 = np.dot(inputs, weights['W1']) + weights['b1']
    
    #Input is now equal to activation of output
    a1 = sigmoid(z1)

    #Dot product of hidden layer out and weight
    z2 = np.dot(a1, weights['W2']) + weights['b2']

    #Run through softmax
    a2 = softmax(z2)

    outs = {"Z1": z1, "A1": a1, "Z2": z2, "A2": a2}
    return outs

    
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
    return z * (1 - z)

def relu(z):
	return np.absolute(z)
    
def softmax(z):
    t = np.exp(z)
    a = np.exp(z) / np.sum(t, axis=1).reshape(-1,1)
    return a
        


def back_propagation(weights, outputs, train_input, train_target):
    deltas = {}

    #Calculate the error for the output layer
    output_error = calculate_error(train_target, outputs['A2'])

    #Calculate the error derivative for softmax
    error_gradient = error_derivative(train_target, outputs['A2'])

    #Output delta (gradient) is error derivative * hidden layer outs (average for batch)
    out_delta = np.dot(outputs['A1'].T, error_gradient) / error_gradient.shape[0]

    #Append the delta
    deltas['dW2'] = out_delta

    #Append the bias
    deltas['db2'] = np.sum(error_gradient, axis=0, keepdims=True) / error_gradient.shape[0]

    #Get error for the hidden layer output(previous layer error * weights)
    hidden_out_error = np.dot(error_gradient, weights['W2'].T)

    #Hidden layer error is output error * outputs * sigmoid prime
    hidden_error = hidden_out_error * outputs['A1'] * sigmoid_prime(outputs['A1'])

    #Delta is input * error
    hidden_delta = np.matmul(train_input.T, hidden_error)

    #Append the delta
    deltas['dW1'] = hidden_delta

    #Append the bias
    deltas['db1'] = np.sum(hidden_error, axis=0, keepdims=True) / error_gradient.shape[0]

    #Return
    return output_error, deltas


    
def calculate_error(target, output):
    #Get the shape of the output
    rows, cols = output.shape

    #Reshape from from just a # to all 0's
    reshaped_target = np.zeros((rows, 10))

    #Change index of correct predictions to a 1
    reshaped_target[np.arange(reshaped_target.shape[0]), target]=1

    #Add up the error
    ce = -np.sum(reshaped_target * np.log(output + 1e-8))

    #Round and return
    return round(ce, 2)



def error_derivative(target, output):
    
    rows, cols = output.shape
    reshaped_target = np.zeros((rows, 10))
    reshaped_target[np.arange(reshaped_target.shape[0]), target]=1
    return output - reshaped_target


    
def accuracy(target, predictions):
    #See the total sum of 1's (True's where predictions matched target)
    correct_preds = np.sum(predictions.astype(int))

    #Return correct / total
    return correct_preds / len(target)



def predict(inputs, target, weights):
    #Feed forward test inputs
    outputs = feed_forward(inputs, weights)

    #Get the predictions in a usable format
    preds = get_predictions(outputs, target=target).astype(int)

    #Return preds
    return preds



def get_predictions(outputs, target):
    #For each row, get the predictions (where the 1 is)
    predicts = np.argmax(outputs['A2'], axis=1)

    #Return where predictions match target
    return predicts == target


start_time = time.time()
opts = init_params()
train()
print("--- %s seconds ---" % (time.time() - start_time))
