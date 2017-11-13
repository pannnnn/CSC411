'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
import math

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(10):
        means[i] = np.mean(train_data[train_labels==float(i)], axis=0)
    return means

def compute_sigma_mles(train_data, train_labels, means):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    for i in range(10):
        difference = train_data[train_labels==float(i)] - means[i]
        difference_original  = difference.reshape(-1,64,1)
        differnce_transpose = difference.reshape(-1,1,64)
        covariances[i] = np.mean(np.multiply(difference_original, differnce_transpose), axis=0)
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    log_cov = []
    for i in range(10):
        cov_diag = np.reshape(np.diag(covariances[i]), (8,8))
        log_cov.append(np.log(cov_diag))
    all_concat = np.concatenate(log_cov, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()
    # ...

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    n = digits.shape[0]
    d = digits.shape[1]
    result = np.zeros((n, 10))
    first_term = -0.5*d*np.log(2*math.pi)
    for i in range(n):
        for j in range(10):
            new_covariance = covariances[j] + 0.01 * np.identity(64)
            inverse = np.linalg.inv(new_covariance)
            second_term = -0.5*np.log(np.linalg.det(new_covariance))
            third_term = -0.5*np.dot(np.dot((digits[i]-means[j]), inverse), (digits[i]-means[j].T))
            result[i,j] = first_term + second_term + third_term
    return result

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    probability_of_x_given_class = generative_likelihood(digits, means, covariances)
    first_term = probability_of_x_given_class
    second_term = np.log(1/10)
    third_term = -np.log(1/10 * np.sum(np.exp(probability_of_x_given_class), axis=1))
    result =  first_term + second_term + third_term.reshape(-1,1)
    return result

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    # Compute as described above and return
    n = labels.shape[0]
    result = np.zeros(n)
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    for i in range(n):
        result[i] = cond_likelihood[i, int(labels[i])]
    return np.mean(result)

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    result = np.argmax(cond_likelihood, axis=1)
    return result

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels, means)
    plot_cov_diagonal(covariances)
    # Evaluation
    print("Average Conditional Likelihood for training set is: " + str(avg_conditional_likelihood(train_data, train_labels, means, covariances)))
    print("Average Conditional Likelihood for test set is: " + str(avg_conditional_likelihood(test_data, test_labels, means, covariances)))
    classified_labels_train_data = classify_data(train_data, means, covariances)
    classified_labels_test_data = classify_data(test_data, means, covariances)
    tmp = (train_labels - classified_labels_train_data)
    accuracy_train_data = tmp[tmp==0].shape[0]/tmp.shape[0]
    tmp = (test_labels - classified_labels_test_data)
    accuracy_test_data = tmp[tmp==0].shape[0]/tmp.shape[0]
    print("Accuracy for training set is: " + str(accuracy_train_data))
    print("Accuracy for test set is: " + str(accuracy_test_data))

if __name__ == '__main__':
    main()