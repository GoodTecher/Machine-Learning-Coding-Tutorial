"""
GoodTecher Machine Learning Coding Tutorial
http://www.goodtecher.com

Machine Learning Coding Tutorial 4. Testing Accuracy

The program demonstrate how to calculate machine learning prediction accuracy score
"""

# import Iris dataset
from sklearn import datasets
iris = datasets.load_iris()

# x is data, y is true label
x = iris.data
y = iris.target

# split half data as test data, half data as training data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)

# use data to train Decision Tree classifier
from sklearn import tree
tree_classifier = tree.DecisionTreeClassifier()
tree_classifier.fit(x_train, y_train)

# use data to train kNeighbors classifier
from sklearn.neighbors import KNeighborsClassifier
kNeighbors_classifier = KNeighborsClassifier()
kNeighbors_classifier.fit(x_train, y_train)

# predict
tree_predictions = tree_classifier.predict(x_test)
kNeighbors_predictions = kNeighbors_classifier.predict(x_test)

# compare true labels with prediction values to get accuracy score
from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, tree_predictions))
print (accuracy_score(y_test, kNeighbors_predictions))
