"""
GoodTecher Machine Learning Coding Tutorial
http://www.goodtecher.com

Machine Learning Coding Tutorial 5. Build Our Own Classifier

The program demonstrate how to build our own classifier
"""
import random
from scipy.spatial import distance

def euc(a, b):
    """
    Helper funciton to get distance between two points
    """
    return distance.euclidean(a, b)


# class of our own Kneighbors classifier
class ScrappyKNN():
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def naive_predict(self, x_test):
        predictions = []
        for row in x_test:
            label = random.choice(self.y_train)
            predictions.append(label)
        return predictions

    def closet_predict(self, x_test):
        predictions = []
        for row in x_test:
            label = self.closet(row)
            predictions.append(label)
        return predictions

    def closet(self, row):
        best_dist = euc(row, self.x_train[0])
        best_index = 0
        for i in range(1, len(self.x_train)):
            dist = euc(row, self.x_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]


# import Iris dataset
from sklearn import datasets

iris = datasets.load_iris()

# x is data, y is true label
x = iris.data
y = iris.target

# split half data as test data, half data as training data
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

# use data to train Decision Tree classifier
from sklearn import tree

tree_classifier = tree.DecisionTreeClassifier()
tree_classifier.fit(x_train, y_train)

# use data to train kNeighbors classifier
from sklearn.neighbors import KNeighborsClassifier

kNeighbors_classifier = KNeighborsClassifier()
kNeighbors_classifier.fit(x_train, y_train)

# use data to train our own kNeighbors classifier
my_classifier = ScrappyKNN()
my_classifier.fit(x_train, y_train)
my_classifier.fit(x_train, y_train)

# predict
tree_predictions = tree_classifier.predict(x_test)
kNeighbors_predictions = kNeighbors_classifier.predict(x_test)
my_classifier_naive_predictions = my_classifier.naive_predict(x_test)
my_classifier_closet_predictions = my_classifier.closet_predict(x_test)

# compare true labels with prediction values to get accuracy score
from sklearn.metrics import accuracy_score

print("tree classifier accuracy: ", accuracy_score(y_test, tree_predictions))
print("k neighbors classifier accuracy: ", accuracy_score(y_test, kNeighbors_predictions))
print("naive classifier accuracy: ", accuracy_score(y_test, my_classifier_naive_predictions))
print("closet classifier accuracy: ", accuracy_score(y_test, my_classifier_closet_predictions))
