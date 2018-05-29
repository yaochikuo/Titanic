import pandas as pd
import pandas as pd
from sklearn import linear_model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
import matplotlib.pyplot as plt


def data_pre_process():
    """step1 get training data"""
    df=pd.read_csv("d:/Desktop/python code/Titanic_keras/train.csv")
    df=df[['Sex','Pclass','Age','Survived']]
    #fill missing age data by agerage age
    df.Age=df.Age.fillna(df.Age.mean())
    #print(df.info())
    #print(df.describe())
    #one hot encoding
    df=pd.get_dummies(df,columns=df[['Sex','Pclass']])
    x_train=df[['Sex_female','Sex_male','Pclass_1','Pclass_2','Pclass_3','Age']]
    y_train=df.Survived
    
    """step2 get testing data"""
    df=pd.read_csv("d:/Desktop/python code/Titanic_keras/test.csv")
    #use Sex, Pclass, Age as training feature
    df=df[['Sex','Pclass','Age']]
    #fill missing age data by agerage age
    df.Age=df.Age.fillna(df.Age.mean())
    #one hot encoding
    df=pd.get_dummies(df,columns=df[['Sex','Pclass']])
    x_test=df[['Sex_female','Sex_male','Pclass_1','Pclass_2','Pclass_3','Age']]
    
    """step3 get real answer"""
    df=pd.read_csv("d:/Desktop/python code/Titanic_keras/gender_submission.csv")
    y_test=df.Survived
    return x_train,y_train,x_test,y_test


def deepNN(X_train,y_train,X_test,y_test):
    model=Sequential()
    model.add(Dense(units=100,input_dim=6,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(units=100,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    train_history=model.fit(x=X_train,y=y_train,validation_split=0.1,epochs=30,batch_size=100,verbose=2)
    #plt.plot(train_history.history[x_train]) 
    print('training scores=',model.evaluate(x=X_train,y=y_train)[1])
    testing_score=model.evaluate(x=X_test,y=y_test)[1]
    print('testing scores=',testing_score)
    return testing_score