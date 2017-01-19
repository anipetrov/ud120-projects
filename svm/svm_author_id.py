#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

##Optimizing dataset size
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

##Import svc and accuracy libraries from sklearn
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

clf = SVC(kernel = "rbf", C=10000.0)

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"

print accuracy_score(pred, labels_test)

##Who were particular emails sent from, Sarah(0) or Chris(1)?
ex = [10,26,50]
for i in ex:
  print i, ": ", pred[i]

##How many of the test emails were sent by Chris(1) according to our prediction?
chris = 0
for i in pred:
  if i == 1:
    chris += 1

print chris


