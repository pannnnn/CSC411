'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        distance_test_and_training = self.l2_distance(test_point)
        k_smallest_index = distance_test_and_training.argsort()[:k]
        # k_nearest_neighbor = self.train_labels[k_smallest_index]
        # digit = float(np.argmax(np.bincount(k_nearest_neighbor.astype(int))))

        k_nearest_neighbor_distance = distance_test_and_training[k_smallest_index]
        k_nearest_neighbor_label = self.train_labels[k_smallest_index]
        occurence_count = np.bincount(k_nearest_neighbor_label.astype(int))
        same_number_occurence_labels = np.argwhere(occurence_count == np.amax(occurence_count))
        # all_labels = np.unique(k_nearest_neighbor_label)
        mean_list = []
        for i in same_number_occurence_labels:
            # x = np.where(k_nearest_neighbor_label == i)
            # y = k_nearest_neighbor_distance[x]
            # z = np.mean(y)
            mean_list.append(np.mean(k_nearest_neighbor_distance[np.where(k_nearest_neighbor_label == float(i))]))
        digit = float(same_number_occurence_labels[mean_list.index(min(mean_list))])

        # print(distance_test_and_training[k_smallest_index])
        return digit

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    kf = KFold(n_splits=10)
    accuracy_rate_per_k = []
    for k in k_range:
        accuracy_rate_across_folds = []
        # Loop over folds
        for train_index, validation_index in kf.split(train_data):
            accuracy_result_from_validation_set = []
            knn = KNearestNeighbor(train_data[train_index], train_labels[train_index])
            for i in validation_index:
                validation_point = train_data[i]
                validation_point_label = train_labels[i]
                digit = knn.query_knn(validation_point, k)
                accuracy_result_from_validation_set.append(1 if digit==validation_point_label else 0)
            accuracy_rate_across_folds.append(sum(accuracy_result_from_validation_set)/validation_index.shape[0])
        avg_accuracy_rate = sum(accuracy_rate_across_folds)/len(accuracy_rate_across_folds)
        print("Runing: k = " + str(k) +", the average accuracy rate is: " + str(avg_accuracy_rate))
        accuracy_rate_per_k.append(avg_accuracy_rate)
    optimal_k = np.argmax(accuracy_rate_per_k) + 1
    return optimal_k, accuracy_rate_per_k[optimal_k - 1]
        # Evaluate k-NN
        # ...

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    count = 0
    for i in range(eval_data.shape[0]):
        predicted_label = knn.query_knn(eval_data[i], k)
        if predicted_label == eval_labels[i]:
            count += 1
    return count/eval_data.shape[0]


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # # Example usage:
    accuracy_for_train_data = []
    accuracy_for_test_data = []
    for k in [1,15]:
        accuracy_for_train_data.append(classification_accuracy(knn, k, train_data, train_labels))
        accuracy_for_test_data.append(classification_accuracy(knn, k, test_data, test_labels))

    print("For K = 1," + 
    "\nThe train classification accuracy is: " + str(accuracy_for_train_data[0]) + 
    "\nThe test classification accuracy is: " + str(accuracy_for_test_data[0]) + "\n")

    print("For K = 15," + 
    "\nThe train classification accuracy is: " + str(accuracy_for_train_data[1]) + 
    "\nThe test classification accuracy is: " + str(accuracy_for_test_data[1]) + "\n")

    optimal_k, accuracy_rate = cross_validation(train_data, train_labels)

    accuracy_for_train_data.append(classification_accuracy(knn, optimal_k, train_data, train_labels))
    accuracy_for_test_data.append(classification_accuracy(knn, optimal_k, test_data, test_labels))

    print("\nThe optimal K is: " + str(optimal_k) + 
    "\nThe train classification accuracy is: " + str(accuracy_for_train_data[2]) + 
    "\nThe average accuracy across folds is: " + str(accuracy_rate) + 
    "\nThe test classification accuracy is: " + str(accuracy_for_test_data[2]))

if __name__ == '__main__':
    main()