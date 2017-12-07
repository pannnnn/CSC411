'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB

# The models I pick up for the final result
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# This one takes too much time to train even though yielding the best result
from sklearn.neural_network import MLPClassifier

# The following is what we've tested but does not have decent test accuracy rate 
# Ordered from highest ranking to lowest ranking
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor

def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def train_model_bow(model_name, bow_train, train_labels, bow_test, test_labels):
    if(model_name == "mnb"):
        model = MultinomialNB(alpha=0.01)
        model_full_name = "MultinomialNB"
    elif(model_name == "lr"):
        model = LogisticRegression(C=4, solver='lbfgs')
        model_full_name = "Logistic Regression"
    elif(model_name == "lsvc"):
        model = LinearSVC(C=0.3)
        model_full_name = "LinearSVC"
    elif(model_name == "mlpc"):
        model = MLPClassifier()
        model_full_name = "MLPClassifier"
    elif(model_name == "knn"):
        model = KNeighborsRegressor(n_neighbors=1)
        model_full_name = "KNeighborsRegressor"
    elif(model_name == "rfc"):
        model = RandomForestClassifier()
        model_full_name = "RandomForestClassifier"

    model.fit(bow_train, train_labels)

    #evaluate the model
    train_pred = model.predict(bow_train)
    print('{} train accuracy = {}'.format(model_full_name, (train_pred == train_labels).mean()))
    test_pred = model.predict(bow_test)
    print('{} test accuracy = {}'.format(model_full_name, (test_pred == test_labels).mean()))

    return model

def train_model_tf_idf(model_name, tf_idf_train, train_labels, tf_idf_test, test_labels):
    if(model_name == "mnb"):
        model = MultinomialNB(alpha=0.01)
        model_full_name = "MultinomialNB"
    elif(model_name == "lr"):
        model = LogisticRegression(C=4, solver='lbfgs')
        model_full_name = "Logistic Regression"
    elif(model_name == "lsvc"):
        model = LinearSVC(C=0.3)
        model_full_name = "LinearSVC"
    elif(model_name == "mlpc"):
        model = MLPClassifier()
        model_full_name = "MLPClassifier"
    elif(model_name == "knn"):
        model = KNeighborsRegressor(n_neighbors=1)
        model_full_name = "KNeighborsRegressor"
    elif(model_name == "rfc"):
        model = RandomForestClassifier()
        model_full_name = "RandomForestClassifier"

    model.fit(tf_idf_train, train_labels)

    #evaluate the MultinomialNB model
    train_pred = model.predict(tf_idf_train)
    print('{} train accuracy = {}'.format(model_full_name, (train_pred == train_labels).mean()))
    test_pred = model.predict(tf_idf_test)
    print('{} test accuracy = {}'.format(model_full_name, (test_pred == test_labels).mean()))

    # since it's 20 newsgroups
    # calculate the confusion matrix and return the two most confused index
    k = 20
    confusion_matrix = np.zeros((k,k))
    for i in range(len(test_labels)):
        confusion_matrix[int(test_pred[i]), test_labels[i]] +=1
    # plt.imshow(confusion_matrix.reshape(k,-1))
    # plt.savefig('1_confusion_matrix.png')
    # plt.cla()
    # plt.clf()
    # print(confusion_matrix.astype(int))
    np.fill_diagonal(confusion_matrix, 0.)
    confused_index = np.argmax(confusion_matrix)
    confused_index = [confused_index//k, confused_index%k]

    return model, confused_index
    
if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names_bow = bow_features(train_data, test_data)
    train_tf_idf, test_tf_idf, feature_names_tf_idf = tf_idf_features(train_data, test_data)

    bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)

    # bow implementation
    # mnb_model = train_model_bow("mnb", train_bow, train_data.target, test_bow, test_data.target)
    # lr_model = train_model_bow("lr", train_bow, train_data.target, test_bow, test_data.target)
    # lsvc_model = train_model_bow("lsvc", train_bow, train_data.target, test_bow, test_data.target)
    # mlpc = train_model_bow("mlpc", train_bow, train_data.target, test_bow, test_data.target)
    # knn = train_model_bow("knn", train_bow, train_data.target, test_bow, test_data.target)
    # rfc = train_model_bow("rfc", train_bow, train_data.target, test_bow, test_data.target)

    # tf-idf implementation
    mnb_model, confused_index = train_model_tf_idf("mnb", train_tf_idf, train_data.target, test_tf_idf, test_data.target)
    print("The two classes that classifier was most confused about: " + str(confused_index))
    lr_model, confused_index = train_model_tf_idf("lr", train_tf_idf, train_data.target, test_tf_idf, test_data.target)
    print("The two classes that classifier was most confused about: " + str(confused_index))
    lsvc_model, confused_index = train_model_tf_idf("lsvc", train_tf_idf, train_data.target, test_tf_idf, test_data.target)
    print("The two classes that classifier was most confused about: " + str(confused_index))

    # too slow to train even though gives the best result
    # mlpc = train_model_tf_idf("mlpc", train_tf_idf, train_data.target, test_tf_idf, test_data.target)
    # print(confused_index)

    # unqualified compared to the baseline
    # knn, confused_index = train_model_tf_idf("knn", train_tf_idf, train_data.target, test_tf_idf, test_data.target)
    # print(confused_index)
    # rfc, confused_index = train_model_tf_idf("rfc", train_tf_idf, train_data.target, test_tf_idf, test_data.target)
    # print(confused_index)