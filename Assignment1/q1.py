from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

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
    raise NotImplementedError()

def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    
    # Visualize the features
    visualize(X, y, features)

    #TODO: Split data into train and test
    # Set X to be 1-dimension
    data_count = X.shape[0]
    indices = np.arange(data_count)
    indices_80per = np.sort(np.random.choice(data_count, int(data_count*0.8), replace=False))
    indices_20per = np.sort(np.delete(indices, indices_80per))
    training_set_x = X[indices_80per]
    training_set_y = y[indices_80per]
    test_set_x = X[indices_20per]
    test_set_y = y[indices_20per]
    
    # Fit regression model
    w = fit_regression(X, y)

    # Compute fitted values, MSE, etc.


if __name__ == "__main__":
    main()

