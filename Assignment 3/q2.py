import numpy as np 

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch  

class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum

        lr - learning rate
        beta - momentum hyperparameter
    '''

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta
        self.vel = 0.0

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        self.vel = -(self.lr) * grad +  self.beta * self.vel
        result = params + self.vel
        return result


class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count + 1)
        
    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        # Add bias
        X = np.concatenate((np.ones(X.shape[0]).reshape(-1,1), X), axis=1)
        # w = np.append(np.random.normal(0.0, 0.1), self.w)
        result = 1 - np.multiply(np.matmul(X, self.w), y)
        result[result<0] = 0
        return result

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        # make X of shape (n, m+1)
        hinge_loss = self.hinge_loss(X, y)
        y[np.argwhere(hinge_loss==0)] = 0
        X = np.concatenate((np.ones(X.shape[0]).reshape(-1,1), X), axis=1)
        # w = np.append(0, self.w)
        regularized_w = np.copy(self.w)
        regularized_w[0] = 0
        gradient = regularized_w - (self.c / X.shape[0]) * np.sum(np.multiply(y.reshape(-1,1), X), axis=0)
        return gradient

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        X = np.concatenate((np.ones(X.shape[0]).reshape(-1,1), X), axis=1)
        result = np.dot(X, self.w)
        result[result>=0] = 1
        result[result<0] = -1
        return result

    def compute_loss(self, X, Y):
        regularization_term = 0.5 * np.dot(self.w, self.w)
        loss = regularization_term + (self.c / X.shape[0]) * np.sum(self.hinge_loss(X,Y))
        return loss

def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets

def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]

    for _ in range(steps):
        # Optimize and update the history
        w = optimizer.update_params(w, func_grad(w))
        w_history.append(w)
    return w_history

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.

    SVM weights can be updated using the attribute 'w'. i.e. 'svm.w = updated_weights'
    '''
    svm = SVM(penalty, train_data.shape[1])
    sampler = BatchSampler(train_data, train_targets, batchsize)
    
    for _ in range(iters):
        sample_data, sample_targets = sampler.get_batch()
        svm.w = optimizer.update_params(svm.w, svm.grad(sample_data, sample_targets))

    return svm
    
def train_svm_and_test_on_MNIST(train_data, train_targets, test_data, test_targets, c, m, iters, beta):
    optimizer_svm = GDOptimizer(0.05, beta)
    trained_svm = optimize_svm(train_data, train_targets, c, optimizer_svm, m, iters)

    training_loss = trained_svm.hinge_loss(train_data, train_targets)
    # training_loss = trained_svm.compute_loss(train_data, train_targets)
    print("The training loss: " + str(np.mean(training_loss)))

    test_loss = trained_svm.hinge_loss(test_data, test_targets)
    # test_loss = trained_svm.compute_loss(test_data, test_targets)
    print("The test loss: " + str(np.mean(test_loss)))

    training_set_prediction = trained_svm.classify(train_data)
    train_difference = train_targets - training_set_prediction
    training_set_accuray_rate = len(train_difference[train_difference == 0])/train_difference.shape[0]
    print("The classification accuracy on the training set: " + str(training_set_accuray_rate))

    test_set_prediction = trained_svm.classify(test_data)
    test_difference = test_targets - test_set_prediction
    test_set_accuray_rate = len(test_difference[test_difference == 0])/test_difference.shape[0]
    print("The classification accuracy on the test set: " + str(test_set_accuray_rate))

    plt.imshow(trained_svm.w[1:].reshape(-1,28), cmap='gray')
    if beta == 0:
        plt.savefig('2_3_w_beta_0.png')
    else:
        plt.savefig('2_3_w_beta_dot_1.png')
    plt.cla()
    plt.clf()



if __name__ == '__main__':
    # 2.1 SGD With Momentum
    optimizer_beta_0 = GDOptimizer(1.0)
    result_beta_0 = optimize_test_function(optimizer_beta_0)
    optimizer_beta_dot_9 = GDOptimizer(1.0, 0.9)
    result_beta_dot_9 = optimize_test_function(optimizer_beta_dot_9)
    plt.xlabel("itertaions")
    plt.ylabel("value")
    plt.plot(result_beta_0, 'g', result_beta_dot_9, 'r')
    plt.savefig('2_1.png')
    plt.cla()
    plt.clf()

    # 2.2 Training SVM
    train_data, train_targets, test_data, test_targets = load_data()
    c = 1.0
    iters = 500
    m = 100

    print("Beta = 0:")
    train_svm_and_test_on_MNIST(train_data, train_targets, test_data, test_targets, c, m, iters, 0)
    # You should comment beta=0 case in order to have 92% accuracy on beta=1
    # better not run two train_svm_and_test_on_MNIST in a row
    print("Beta = 0.1:")
    train_svm_and_test_on_MNIST(train_data, train_targets, test_data, test_targets, c, m, iters, 0.1)
