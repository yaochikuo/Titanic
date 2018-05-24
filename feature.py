# -*- coding: utf-8 -*-
"""
Created on Mon May 21 19:19:12 2018

@author: huan
"""
import pandas as pd 

def getData():
    train=pd.read_csv("train.csv")
    test=pd.read_csv("test.csv")
    return train,test
def fillMissing(train,test):
    train.Age=train.Age.fillna(train.Age.mean())
    test.Age=test.Age.fillna(test.Age.mean())
    return train,test
def oneHotEn(train,test):
    train=pd.get_dummies(train,columns=test[['Sex','Pclass']])
    test=pd.get_dummies(test,columns=test[['Sex','Pclass']])
    return train,test
def showInfo(train,test):
    print(train.info())   
    print(test.info())  
def prepareXY(train,test):    
    X_train=train[['Age','Pclass_1','Pclass_2','Pclass_3','Sex_female','Sex_male']]
    y_train=train['Survived']
    X_test=test[['Age','Pclass_1','Pclass_2','Pclass_3','Sex_female','Sex_male']]
    test['Survived']=0
    y_test=test['Survived']
    return X_train,y_train,X_test,y_test
if __name__ == "__main__":
    train,test = getData()
    train,test=fillMissing(train,test)
    showInfo(train,test)