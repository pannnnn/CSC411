'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    # eta = np.zeros((10, 64))
    N_k = np.zeros(10)
    for i in range(10):
        N_k[i] = train_labels[train_labels == float(i)].shape[0]
    N_kj = np.zeros((10,64))
    for i in range(train_data.shape[0]):
        for j in range(64):
            N_kj[int(train_labels[i]), j] += train_data[i,j]
    eta = (np.divide((N_kj + 1).T, (N_k + 2))).T
    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    img = []
    for i in range(10):
        # ...
        img_i = class_images[i]
        img.append(np.reshape(img_i, (8,8)))
    all_concat = np.concatenate(img, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    for i in range(10):
        for j in range(64):
            generated_data[i,j] = np.random.binomial(1, eta[i,j])
    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    n = bin_digits.shape[0]
    result = np.zeros((n, 10))
    for i in range(n):
        for j in range(10):
            first_term = bin_digits[i] * np.log(eta[j])
            second_term = (1 - bin_digits[i]) * np.log(1-eta[j])
            result[i, j] = np.sum(first_term + second_term)
    return result

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    probability_of_x_given_class = generative_likelihood(bin_digits,eta)
    first_term = probability_of_x_given_class
    second_term = np.log(1/10)
    third_term = -np.log(1/10 * np.sum(np.exp(probability_of_x_given_class), axis=1))
    result =  first_term + second_term + third_term.reshape(-1,1)
    return result

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''

    # Compute as described above and return
    n = labels.shape[0]
    result = np.zeros(n)
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    for i in range(n):
        result[i] = cond_likelihood[i, int(labels[i])]
    return np.mean(result)

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class
    result = np.argmax(cond_likelihood, axis=1)
    return result

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)
    
    generate_new_data(eta)

    print("Average Conditional Likelihood for training set is: " + str(avg_conditional_likelihood(train_data, train_labels, eta)))
    print("Average Conditional Likelihood for test set is: " + str(avg_conditional_likelihood(test_data, test_labels, eta)))
    classified_labels_train_data = classify_data(train_data, eta)
    classified_labels_test_data = classify_data(test_data, eta)
    tmp = (train_labels - classified_labels_train_data)
    accuracy_train_data = tmp[tmp==0].shape[0]/tmp.shape[0]
    tmp = (test_labels - classified_labels_test_data)
    accuracy_test_data = tmp[tmp==0].shape[0]/tmp.shape[0]
    print("Accuracy for training set is: " + str(accuracy_train_data))
    print("Accuracy for test set is: " + str(accuracy_test_data))

if __name__ == '__main__':
    main()
