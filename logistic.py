""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid


def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities of being second class. This is the output of the classifier.
    """
    # TODO: Finish this function
 
    Z = np.dot(data, weights[0:-1,:]) + weights[-1,:]
    y = (1/(1 + np.exp(-Z))) 

    return y


def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function

    log_pred = np.log2(1-y)
    log_targets = np.log2(targets)
    log_pred[log_pred==-np.inf]=0
    log_targets[log_targets==-np.inf]=0  

    ce = -(np.dot((1-targets).T, log_pred) + np.dot(log_targets.T, (y)))[0,0]

    #Source: http://stackoverflow.com/questions/1623849/fastest-way-to-zero-out-low-values-in-array
    greater_05_idx = y > 0.5
    less_05_idx = y < 0.5
    y[greater_05_idx] = 0
    y[less_05_idx] = 1
   
    num_wrong = 0
    for i in range(0, np.shape(y)[0]):
        if y[i] != targets[i]:
            num_wrong += 1

    frac_correct = 1 - (float(num_wrong))/(np.shape(y)[0])

    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
        y:       N x 1 vector of probabilities.
    """

    y = logistic_predict(weights, data)

    if hyperparameters['weight_regularization'] is True:
        f, df = logistic_pen(weights, data, targets, hyperparameters)

    else:

        Z = np.dot(data, weights[:-1]) + weights[-1]
        f = (np.dot(targets.T, Z) + np.sum(np.log(1 + np.exp(-Z))))[0,0]
        p_c1 = (np.exp(-Z))/(1+np.exp(-Z))
        df = (np.dot((targets - p_c1).T, data)).T
        df0 = np.sum(targets - (np.exp(-Z)/(1+np.exp(-Z)) ))
        df = np.vstack((df,df0))

    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
    """

    w_decay = hyperparameters['weight_decay']

    Z = np.dot(data, weights[:-1]) + weights[-1]
    p_c1 = (np.exp(-Z)) / (1+np.exp(-Z))
    f = ((np.dot(targets.T, Z) + np.sum(np.log(1 + np.exp(-Z)))) - (np.dot(weights.T,weights)) / 2*w_decay)[0,0]

    df = (np.dot((targets - p_c1).T, data)).T - w_decay*(weights[:-1])
    df0 = np.sum(targets - (np.exp(-Z)/(1+np.exp(-Z)) ))
    df = np.vstack((df,df0))


    return f, df
