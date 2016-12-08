from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from math import log
import random
import numpy as np
from sklearn import metrics
class EasyEnsemble:
    def __init__(self,T=4,rounds=10):

        self.clf = []
        self.T = T
        self.n_sub_classifier = rounds
        for i in range(T):
            self.clf.append(AdaBoostClassifier(n_estimators=rounds,algorithm="SAMME"))
    def fit(self,X,Y):
        if Y.sum()*2>Y.shape[0]:
            minority =0
        else:
            minority =1
        X_minority = X[Y==minority]
        X_majority = X[Y!=minority]
        total_majority = X_majority.shape[0]
        need_sample = X_minority.shape[0]
        for i in xrange(self.T):
            sampled_x = X_majority[random.sample(range(total_majority),need_sample)]
            train_X = np.vstack((sampled_x,X_minority))
            train_Y = np.zeros(2*need_sample)
            train_Y[:need_sample]=1
            self.clf[i].fit(train_X,train_Y)
    def get_params(self):
        return {"T":self.T,"rounds":self.n_sub_classifier}
    def predict(self, X):
        Y = np.zeros(X.shape[0])
        print self.clf[0].estimators_
        for i in xrange(self.T):
            Y += self.clf[i].decision_function(X)
        print Y
        Y[Y>0]=1
        Y[Y<0]=0
        return Y
    def predict_proba(self,X):
        Y = np.zeros(X.shape[0])
        print self.clf[0].estimators_
        for i in xrange(self.T):
            Y += self.clf[i].decision_function(X)
            # print self.clf[i].decision_function(X)
        return Y
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings(action="ignore")
    from sklearn.datasets import load_digits
    data = load_digits()
    RX = data.data
    RY = data.target
    X = RX[RY<=1]
    Y = RY[RY<=1]
    print Y
    # clf = AdaBoostClassifier(algorithm="SAMME",n_estimators=100)
    # clf.fit(X,Y)
    # print clf.predict(X)
    # for i in xrange(100):
    #     preY = clf.estimators_[i].predict(X)
    #     preY[preY==0] =-1
    # print Y
    clf=EasyEnsemble(T=4,rounds=5)
    clf.fit(X,Y)
    pre_Y = clf.predict(X)
    pre_prob = clf.predict_proba(X)
    accuracy = np.ones(Y.shape[0])
    print Y.shape[0],np.sum(accuracy[Y==pre_Y])
    print Y
    print pre_prob
    print pre_Y