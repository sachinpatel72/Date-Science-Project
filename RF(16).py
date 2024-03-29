# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:46:48 2020

@author: Asus
"""

#------------------------------Random Forest--------------------------------
# Random Forest Classification

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Purchase_History.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy',max_depth = 3, min_samples_leaf=5)
classifier.fit(X_train, y_train)

#To see no. of decision trees created
len(classifier.estimators_)

#To see the decision trees created
classifier.estimators_

#To access a particular decision tree, we can use inexing
classifier.estimators_[1]

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
#Accuracy = 94%

# Random Forest visualization

#Since RF is quite big & clumpsy to draw due to large no. of DT, its not possible to 
#visualiza an entire RF on a small system like our laptop.
#Hence, we visualize individual DTs from this RF.

# Decision Tree visualization-----------------
from sklearn import tree
#Lets create a blank chart of desired size using matplotlib library and place our Decision tree there.
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
cn=['0','1']
tree.plot_tree(classifier.estimators_[1],class_names=cn,filled = True)

#if you want save figure, use savefig method in returned figure object.
fig.savefig('imagename.png')