"""
GoodTecher Machine Learning Coding Tutorial
http://www.goodtecher.com

a simple Machine Learning Program to classify a pieces of fruit 

The program takes a description of the fruit as input and predicts whether it's an apple or orange as output 
based on features like its weight and surface smoothness
"""

# Import Decision Tree from python `scikit-learn` library
# Decision Trees are a non-parametric supervised learning method used for classification and regression. 
from sklearn import tree

# Training Data
# sample fruit features: weight, surface texture (0: bumpy, 1: smooth)
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# sample fruit labels: {0: orange, 1: apple}
labels = [0, 0, 1, 1]

# Build a Decision Tree Classifier
clf = tree.DecisionTreeClassifier()
# Use training data to train classifier, in other words the classifier is learning by finding patterns in training data
clf = clf.fit(features, labels)

# output the prediction for a fruit
print (clf.predict([[160, 0]]))