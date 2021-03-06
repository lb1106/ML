from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from math import log
import random
import numpy as np

class RUSBoost:
    def __init__(self, n_classifier, rate=0.35,min_loss=0.0000000001):
        """

        :param n_classifier:
        :param rate: proportion of minority instances to total instances
        """
        self.clf = []
        self.n_classifier = n_classifier
        self.alphas =[]
        self.min_loss = min_loss
        for i in range(n_classifier):
            self.clf.append(DecisionTreeClassifier())
            self.alphas.append(0)
        self.rate = rate
    def fit(self,X,Y):
        global_weight = np.ones(X.shape[0])
        global_weight /= X.shape[0]
        if Y.sum()*2>Y.shape[0]:
            minority = 0
        else:
            minority = 1
        all_indexs = np.arange(X.shape[0],dtype=np.int32)
        minority_indexs = all_indexs[Y==minority]
        X_minority = X[minority_indexs]
        Y_minority = Y[minority_indexs]
        majority_indexs = all_indexs[Y!=minority]
        need_sample = int (len(minority_indexs)/self.rate - len(minority_indexs))
        for i in xrange(self.n_classifier):
            #print majority_indexs
            #print need_sample
            sampled_index = random.sample(majority_indexs,need_sample)
            Y_majority = Y[sampled_index]
            X_majority = X[sampled_index]
            trainX = np.vstack((X_minority,X_majority))
            print Y_majority.shape,Y_minority.shape
            trainY  = np.hstack((Y_minority,Y_majority))
            train_weights = np.hstack((global_weight[minority_indexs],global_weight[sampled_index]))
            train_weights /= train_weights.sum()
            #DecisionTreeClassifier().fit(trainX,trainY,sample_weight=train_weights)
            self.clf[i].fit(trainX,trainY,sample_weight=train_weights)
            loss =self.min_loss
            probs = self.clf[i].predict_proba(X)
            #print probs
            for m in xrange(X.shape[0]):
                cur_loss = global_weight[m] *(1-probs[m][Y[m]] + probs[m][1-Y[m]])
                #print "cur_loss",cur_loss
                loss += cur_loss
            cur_alpha = loss / (1 - loss)
            print "step %d loss" %i,loss,"alpha",cur_alpha
            self.alphas[i] = cur_alpha
            for m in xrange(X.shape[0]):
                global_weight[m] = global_weight[m] * cur_alpha**( 0.5 * ( 1 + probs[m][Y[m]] - probs[m][1-Y[m]]))
            global_weight /= global_weight.sum()
    def get_params(self):
        return {"T":self.n_classifier,"rate":self.rate}
    def predict(self,X):
        predicts = np.ones(X.shape[0])
        total_probs = np.zeros((X.shape[0],2))
        for i in xrange(self.n_classifier):
            probs = self.clf[i].predict_proba(X)
            cur_alpha = self.alphas[i]
            #print cur_alpha
            total_probs += probs*np.log(1/cur_alpha)
        predicts[total_probs[:,0]>total_probs[:,1]] = 0
        return predicts
    def predict_proba(self,X):
        total_probs = np.zeros((X.shape[0], 2))
        for i in xrange(self.n_classifier):
            probs = self.clf[i].predict_proba(X)
            cur_alpha = self.alphas[i]
            total_probs += probs * np.log(1 / cur_alpha)
        return total_probs[:,1]
if __name__ == '__main__':
    from sklearn.datasets import load_digits

    data = load_digits()
    RX = data.data
    RY = data.target
    X = RX[RY <= 1]
    Y = RY[RY <= 1]
    #Y[5:20] =0
    #Y[:5] =1
    #print Y,Y.shape,Y.sum()
    #exit()
    clf = RUSBoost(n_classifier=10,rate=0.9)
    clf.fit(X[:300],Y[:300])
    print clf.get_params()
    accuracy = np.ones(60)
    pre_Y = clf.predict(X[300:])
    print Y[300:].shape, np.sum(accuracy[Y[300:] == pre_Y])