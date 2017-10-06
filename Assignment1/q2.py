# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.misc import logsumexp
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions-y_test)**2).mean()
    return losses
 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    test_datum = test_datum.reshape(test_datum.shape[1],test_datum.shape[0])
    exp_component = np.true_divide(-l2(test_datum, x_train), 2*tau**2)
    numerator = np.exp(exp_component)
    denominator = np.exp(logsumexp(exp_component))
    a_i = np.true_divide(numerator, denominator).reshape(x_train.shape[0])
    A = np.diag(a_i)
    X_transpose = np.transpose(x_train)
    X_transpose_A_X = np.matmul(np.matmul(X_transpose, A), x_train)
    identity_matrix = np.identity(x_train.shape[1])
    X_transpose_A_X_plus_lam_I = X_transpose_A_X + lam * identity_matrix
    inverse_of_X_transpose_A_X_plus_lam_I = np.linalg.solve(X_transpose_A_X_plus_lam_I, identity_matrix)
    w_star = np.matmul(np.matmul(np.matmul(inverse_of_X_transpose_A_X_plus_lam_I, X_transpose), A), y_train)
    y_hat = np.matmul(test_datum, w_star)
    return np.squeeze(y_hat, axis=(1,))
    ## TODO




def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    ## TODO
    split_idx = np.array_split(idx, k)
    losses = np.zeros((k, taus.shape[0]))
    for i in range(k):
        test_set_index = np.sort(split_idx[i])
        training_set_index = np.sort(np.concatenate(tuple(split_idx[:i]+split_idx[i+1:])))
        x_test = x[test_set_index]
        y_test = y[test_set_index]
        x_train = x[training_set_index]
        y_train = y[training_set_index]
        losses[i,:] = run_on_fold(x_test, y_test, x_train, y_train, taus)
    return np.average(losses, axis=0)
    ## TODO


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    y = y.reshape(y.shape[0], 1)
    taus = np.logspace(1.0,3,200)
    losses = run_k_fold(x,y,taus,k=5)
    plt.plot(losses)
    plt.savefig('part2.png')
    print("min loss = {}".format(losses.min()))

