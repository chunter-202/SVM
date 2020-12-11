#####################################################
#   Taylor Chase Hunter                             #
#                                                   #
#   This code is an implmentation of an SVM and     #
#   various kernel methods using SciKit learn.      #
#   This code in particular uses the SAT data file  #
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

#   Read in data    #
train_csv = pd.read_csv(r"C:\Users\sepro\Documents\Cosc Files\425\Project_3\Lib\sat.trn", delimiter= ' ')
test_csv = pd.read_csv(r"C:\Users\sepro\Documents\Cosc Files\425\Project_3\Lib\sat.tst", delimiter= ' ')
data = pd.DataFrame(train_csv)
test_Data = pd.DataFrame(test_csv)

#   Set our scaler  #
scaler = MinMaxScaler(feature_range=(-1, 1))

#   Divide up the data  #
x_train = data.iloc[:, 0:35]
x_train = scaler.fit_transform(x_train)
y_train = data["target"]
x_test = test_Data.iloc[:, 0:35]
x_test = scaler.fit_transform(x_test)
y_test = test_Data["target"]
y_train_converted = y_train.values.ravel()


#   These were used in conjunction with the fine grid search in order to oberseve accuracies    #
poly_svc_one = svm.SVC(kernel='poly', C=12, degree=2).fit(x_train, y_train_converted)
poly_svc_two = svm.SVC(kernel='poly', C=13, degree=2).fit(x_train, y_train_converted)
poly_svc_three = svm.SVC(kernel='poly', C=14, degree=2).fit(x_train, y_train_converted)
poly_svc_four = svm.SVC(kernel='poly', C=15, degree=2).fit(x_train, y_train_converted)
predicted_poly_one = poly_svc_one.predict(x_test)
predicted_poly_two = poly_svc_two.predict(x_test)
predicted_poly_three = poly_svc_three.predict(x_test)
predicted_poly_four = poly_svc_four.predict(x_test)
print("SVM + Poly (D=2, C=12)\t-> " + str(accuracy_score(y_test, predicted_poly_one)))
print("SVM + Poly (D=2, C=13)\t-> " + str(accuracy_score(y_test, predicted_poly_two)))
print("SVM + Poly (D=2, C=14)\t-> " + str(accuracy_score(y_test, predicted_poly_three)))
print("SVM + Poly (D=2, C=15)\t-> " + str(accuracy_score(y_test, predicted_poly_four)))

#   This was used for the grid searches and modified as needed  #
tuned_parameters = [
    {
        'kernel': ['poly'],
        'degree': [2],
        'C': [13]
    },
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

