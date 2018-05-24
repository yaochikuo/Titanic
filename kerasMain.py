# -*- coding: utf-8 -*-
"""
Created on Mon May 21 19:25:13 2018

@author: huan
"""
from feature import *
from deepX import *

train,test = getData()
train,test=fillMissing(train,test)
train,test=oneHotEn(train,test)
X_train,y_train,X_test,y_test=prepareXY(train,test)
showInfo(train,test)
deepNN(X_train,y_train,X_test,y_test)
