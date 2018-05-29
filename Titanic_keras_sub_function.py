import pandas as pd
import pandas as pd
from sklearn import linear_model
import numpy as np
np.random.seed(1337) # for reproducibility
from keras.models import Sequential
from keras.layers import Dense,Dropout
import matplotlib.pyplot as plt


def data_pre_process():
    """step1 get training data"""
    df=pd.read_csv("d:/Desktop/python code/Titanic_keras/train.csv")
    df.Embarked=df.Embarked.fillna('C')
    #print(df.info())

    df=df[['Sex','Pclass','Age','Survived']]
    #fill missing age data by agerage age
    df.Age=df.Age.fillna(df.Age.mean())
    #print(df.describe())
    #one hot encoding
    df=pd.get_dummies(df,columns=df[['Sex','Pclass']])
    x_train=df[['Sex_female','Sex_male','Pclass_1','Pclass_2','Pclass_3','Age']]
    y_train=df.Survived
    
    """step2 get testing data"""
    df=pd.read_csv("d:/Desktop/python code/Titanic_keras/test.csv")
    #use Sex, Pclass, Age as training feature
    df['Survived']=0
    df=df[['Sex','Pclass','Age','Survived']]
    #fill missing age data by agerage age
    df.Age=df.Age.fillna(df.Age.mean())
    #one hot encoding
    df=pd.get_dummies(df,columns=df[['Sex','Pclass']])
    x_test=df[['Sex_female','Sex_male','Pclass_1','Pclass_2','Pclass_3','Age']]
    y_test=df.Survived
    
    return x_train,y_train,x_test,y_test


def deepNN(X_train,y_train,X_test,y_test):
    model=Sequential()
    model.add(Dense(units=100,input_dim=6,kernel_initializer='uniform',activation='relu'))
    model.add(Dropout(0.0))
    model.add(Dense(units=100,kernel_initializer='uniform',activation='relu'))
    model.add(Dropout(0.0))
    model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    train_history=model.fit(x=X_train,y=y_train,validation_split=0.1,epochs=500,batch_size=81,verbose=2)
    plt.plot(train_history.history['acc'],label='Training Accuracy') 
    plt.plot(train_history.history['val_acc'],label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0.7,0.9)
    plt.legend()
    plt.show()
    print('training scores=',model.evaluate(x=X_train,y=y_train)[1])
    result=model.predict(X_test)
    return result