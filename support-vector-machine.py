# Andres Gandara
# CSE 494: Intro to Machine Learning
# HW 3

import numpy as np
import os
import pdb
import pandas as pd
import seaborn as sn
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from matplotlib import pylab as pl
from pylab import *

datasets_dir = './data/'


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def mnist(noTrSamples=1000, noTsSamples=100, digit_range=[0, 10], noTrPerClass=100, noTsPerClass=10):
    data_dir = os.path.join(datasets_dir, 'mnist/')
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trData = loaded[16:].reshape((60000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trLabels = loaded[8:].reshape((60000)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsData = loaded[16:].reshape((10000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsLabels = loaded[8:].reshape((10000)).astype(float)

    trData = trData/255.
    tsData = tsData/255.

    tsX = np.zeros((noTsSamples, 28*28))
    trX = np.zeros((noTrSamples, 28*28))
    tsY = np.zeros(noTsSamples)
    trY = np.zeros(noTrSamples)

    count = 0
    for ll in range(digit_range[0], digit_range[1]):
        # Train data
        idl = np.where(trLabels == ll)
        idl = idl[0][: noTrPerClass]
        idx = list(range(count*noTrPerClass, (count+1)*noTrPerClass))
        trX[idx, :] = trData[idl, :]
        trY[idx] = trLabels[idl]
        # Test data
        idl = np.where(tsLabels == ll)
        idl = idl[0][: noTsPerClass]
        idx = list(range(count*noTsPerClass, (count+1)*noTsPerClass))
        tsX[idx, :] = tsData[idl, :]
        tsY[idx] = tsLabels[idl]
        count += 1
    
    # np.random.seed(1)
    # test_idx = np.random.permutation(tsX.shape[0])
    # tsX = tsX[test_idx,:]
    # tsY = tsY[test_idx]
    
    return trX, trY, tsX, tsY


def loadmnist():
    trX, trY, tsX, tsY = mnist(noTrSamples=10000,
                               noTsSamples=1000, digit_range=[0, 10],
                               noTrPerClass=500, noTsPerClass=50)
    return trX, trY, tsX, tsY


def extractfeat(X):
    feat1 = X.mean(axis=1)
    n = X.shape[0]
    X = X.reshape(-1, 28, 28)
    feat2 = X.max(axis=1).var(axis=1)
    feat1 = feat1.reshape(-1, 1)
    feat2 = feat2.reshape(-1, 1)
    Y = np.concatenate((feat1, feat2), axis=1)
    return Y


def loadmnistfeat():
    trX, trY, tsX, tsY = loadmnist()
    trX = extractfeat(trX)
    tsX = extractfeat(tsX)
    return trX, trY, tsX, tsY

trX, trY, tsX, tsY = loadmnistfeat()

# 1 iterations
classifier = SVC(kernel = 'linear', random_state = 0, max_iter = 1)
classifier.fit(trX, trY)
y_pred = classifier.predict(tsX)
cm = confusion_matrix(tsY, y_pred)
print(cm)
#fig = pl.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(cm)
#pl.title('Confusion matrix of the classifier - 1 iteration')
#fig.colorbar(cax)
#ax.set_xticklabels([''])
#ax.set_yticklabels([''])
#pl.xlabel('Predicted')
#pl.ylabel('True')
#pl.show()

# 2 iterations
classifier = SVC(kernel = 'linear', random_state = 0, max_iter = 2)
classifier.fit(trX, trY)
y_pred = classifier.predict(tsX)
cm = confusion_matrix(tsY, y_pred)
print(cm)
#fig = pl.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(cm)
#pl.title('Confusion matrix of the classifier - 2 iterations')
#fig.colorbar(cax)
#ax.set_xticklabels([''])
#ax.set_yticklabels([''])
#pl.xlabel('Predicted')
#pl.ylabel('True')
#pl.show()

# 3 iterations
classifier = SVC(kernel = 'linear', random_state = 0, max_iter = 3)
classifier.fit(trX, trY)
y_pred = classifier.predict(tsX)
cm = confusion_matrix(tsY, y_pred)
print(cm)
#fig = pl.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(cm)
#pl.title('Confusion matrix of the classifier - 3 iterations')
#fig.colorbar(cax)
#ax.set_xticklabels([''])
#ax.set_yticklabels([''])
#pl.xlabel('Predicted')
#pl.ylabel('True')
#pl.show()

# 5 iterations
classifier = SVC(kernel = 'linear', random_state = 0, max_iter = 5)
classifier.fit(trX, trY)
y_pred = classifier.predict(tsX)
cm = confusion_matrix(tsY, y_pred)
print(cm)
#fig = pl.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(cm)
#pl.title('Confusion matrix of the classifier - 5 iterations')
#fig.colorbar(cax)
#ax.set_xticklabels([''])
#ax.set_yticklabels([''])
#pl.xlabel('Predicted')
#pl.ylabel('True')
#pl.show()

# Maximum iterations
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(trX, trY)
y_pred = classifier.predict(tsX)
cm = confusion_matrix(tsY, y_pred)
print(cm)

fig = pl.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
pl.title('Confusion matrix of the classifier - max iterations')
fig.colorbar(cax)
ax.set_xticklabels([''])
ax.set_yticklabels([''])
pl.xlabel('Predicted')
pl.ylabel('True')
pl.show()



