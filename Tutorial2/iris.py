"""
GoodTecher Machine Learning Coding Tutorial
http://www.goodtecher.com

"Iris" Machine Learning Program

The program takes a measurements (the length and width of the pedal and sepal) 
of a flower as input 
and predicts whether it is setosa, versicolor or virginica 
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus

# load Iris dataset
iris = load_iris()

# picks some data from Iris dataset as test data
# and rest data would be training data
test_idx = [0, 10, 50, 100]

# training data
train_data = np.delete(iris.data, test_idx, axis = 0)
train_target = np.delete(iris.target, test_idx)

# testing data
test_data = iris.data[test_idx]
test_target = iris.target[test_idx]

# train classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# display and compare test target and predict target
print ("Test target: ")
print (test_target)
print ("clf.predict: ")
print (clf.predict(test_data))

# output Decision Tree procedure to a PDF file
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")