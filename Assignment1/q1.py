from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        plt.xlabel(features[i])
        plt.plot(X[:,i], y, 'ro', ms=1)
    
    plt.tight_layout()
    plt.savefig('part1_features.png')
    # plt.show()


def fit_regression(X,Y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    identity_matrix = np.identity(X.shape[1])
    X_transpose = np.transpose(X)
    inverse_of_X_X_transpose = np.linalg.solve(np.matmul(X_transpose, X), identity_matrix)
    return np.matmul(np.matmul(inverse_of_X_X_transpose, X_transpose), Y)

def plot_table(features, w):
    features= features.reshape(13,1)
    weights = w[1:].astype(str).reshape(13,1)
    data = np.concatenate((features, weights), axis=1)
    collabel = ["Feature", "Weights"]
    plt.table(cellText=data,colLabels=collabel,loc='center')
    plt.axis('off')
    plt.savefig('part1_table.png', bbox_inches='tight')


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
    expected_y =  np.matmul(test_set_x, w)
    # MSE
    MSE  = np.mean(np.square(expected_y - test_set_y))
    print("Mean Square Error is: " + str(MSE))
    # MAE
    MAE = np.mean(np.absolute(expected_y - test_set_y))
    print("Mean Absolute Error is: " + str(MAE))
    # Mean Absolute Percentage Error
    MAPE = 100 * np.mean(np.absolute(np.true_divide(expected_y - test_set_y, test_set_y)))
    print("Mean Absolute Percentage Error: " + str(MAPE))

    # plot the table
    # plot_table(features, w)   


if __name__ == "__main__":
    main()

