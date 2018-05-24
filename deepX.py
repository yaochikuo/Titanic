# -*- coding: utf-8 -*-
"""
Created on Mon May 21 20:24:03 2018

@author: huan
"""
from keras.models import Sequential


def deepNN(X_train,y_train,X_test,y_test):
    model=Sequential()
    model.add(units=40,input_dim=6,kernel_initializer='uniform',activation='relu')
    model.add(units=30,kernel_initializer='uniform',activation='relu')
    model.add(units=1,kernel_initializer='uniform',activation='sigmoid')
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accracy'])
    
    
    

    