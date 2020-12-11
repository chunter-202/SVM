#####################################################
#   Taylor Chase Hunter                             #
#   COSC 425    Project 4                           #
#                                                   #
#   This code is an implmentation of an SVM and     #
#   various kernel methods using SciKit learn.      #
#   This code in particular uses the Ion data file  #
#####################################################

import numpy as np
import pandas as pd
import pylab as pl
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

#   Import training data for setting 1  #
cs = pd.read_csv(r"C:\Users\sepro\Documents\Cosc Files\425\Project_3\Lib\ionosphere.data")

#   load into dataframe  #
data = pd.DataFrame(cs)

scaler = MinMaxScaler(feature_range=(-1, 1))

#   Seperate features from target   #
X = data.iloc[:, 0:34]
X = scaler.fit_transform(X)
Y = data.iloc[:,-1]

#   Split the data  #
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=4)
y_train_converted = y_train.values.ravel()

#   This was used for testing acuracy of indivudal methods  #
k_one = svm.SVC(kernel='rbf', C=1200, gamma=0.00225).fit(x_train, y_train_converted)
k_one_predict = k_one.predict(x_test)
accuracy = accuracy_score(y_test, k_one_predict)


#   This was used for the grid searches and modified as needed  #
tuned_parameters = [

    {
        'kernel' : ['rbf'],
        'gamma' : [.0025],
        'C' : [1200]
    }
]

#####   Taken directly from svm.py  #####
scores = ['precision', 'recall']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(x_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))
    print()

#   This was were I applied K-cross validation  #
scores1 = cross_val_score(k_one, X, Y, cv=5)
print("Accuracy on set")
print(accuracy)
print("Cross validate of the best")
print(scores1)
print(sum(scores1) / len(scores1))

