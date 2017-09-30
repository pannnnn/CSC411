from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X,y,features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        plt.plot(X[:,i], y, 'ro', ms=1)
    
    plt.tight_layout()
    plt.show()


def fit_regression(X,Y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    identity_matrix = np.identity(X.shape[1])
    X_transpose = np.transpose(X)
    inverse_of_X_X_transpose = np.linalg.solve(np.dot(X_transpose, X), identity_matrix)
    return np.dot(np.dot(inverse_of_X_X_transpose, X_transpose), Y)

def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    
    # Visualize the features
    visualize(X, y, features)

    #TODO: Split data into train and test
    data_count = X.shape[0]
    indices = np.arange(data_count)
    indices_80per = np.sort(np.random.choice(data_count, int(data_count*0.8), replace=False))
    indices_20per = np.sort(np.delete(indices, indices_80per))
    training_set_x = X[indices_80per]
    training_set_x = np.insert(training_set_x, 0, 1, 1)
    training_set_y = y[indices_80per]
    test_set_x = X[indices_20per]
    test_set_x = np.insert(test_set_x, 0, 1, 1)
    test_set_y = y[indices_20per]

    # Fit regression model
    w = fit_regression(training_set_x, training_set_y)

    # Compute fitted values, MSE, etc.
    expected_y =  np.dot(test_set_x, w)
    MSE  = np.sum(np.square(expected_y - test_set_y))/test_set_x.shape[0]


if __name__ == "__main__":
    main()

