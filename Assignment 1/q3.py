import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

BATCHES = 50
K = 500
J = 6

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


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)

#TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    return np.average((-2) * np.transpose(X) * (y - np.matmul(X, w)), axis=1)

def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)

    # Example usage
    X_b, y_b = batch_sampler.get_batch()
    batch_grad = lin_reg_gradient(X_b, y_b, w)
    true_grad = np.average((-2) * np.transpose(X) * (y - np.matmul(X, w)), axis=1)
    square_distance = np.sum((batch_grad - true_grad)**2)
    cos_similarity = cosine_similarity(batch_grad, true_grad)
    print("Square Distance Metric is: " + str(square_distance))
    print("Cosine Similarity is: " + str(cos_similarity))

    m = list(range(1,401))
    sample_variance = [0] * 400
    gradients = [0] * K
    for i in range(400):
        for j in range(K):
            X_b, y_b = batch_sampler.get_batch(m[i])
            batch_grad = lin_reg_gradient(X_b, y_b, w)
            gradients[j] = batch_grad[J]
        sample_variance[i] = np.var(gradients)
    m = np.asarray(np.log(m))
    sample_variance = np.asarray(np.log(sample_variance))
    plt.xlabel("log m")
    plt.ylabel("log sigma")
    plt.plot(m, sample_variance)
    plt.savefig('part3.png')


if __name__ == '__main__':
    main()
