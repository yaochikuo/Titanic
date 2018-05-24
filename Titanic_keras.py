import pandas as pd
from sklearn import linear_model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
import matplotlib.pyplot as plt 
from utils import *  
#       step 1 get training data  
df=pd.read_csv("d:/Desktop/python code/Titanic-keras/train.csv")
#       use Sex, Pclass, Age as training feature
df=df[['Sex','Pclass','Age','Survived']]
#print(df.info())
#       
df.Age=df.Age.fillna(df.Age.mean())
df.info()
print(df.Age)




#train.Age=train.Age.fillna(train.Age.mean())
df=pd.get_dummies(df,columns=df[['Sex','Pclass']])
df



#df.Sex[df.Sex=="male"]=1
#df.Sex[df.Sex=="female"]=0

x_train=df[['Sex_female','Sex_male','Pclass_1','Pclass_2','Pclass_3','Age']]



y_train=df.Survived

print(y_train)

clf=linear_model.LogisticRegression()
#clf.fit(np.array(x_train),np.array(y_train))
clf.fit(x_train,y_train)
print("scores=",clf.score(x_train,y_train))

df=pd.read_csv("d:/Desktop/python code/Titanic-keras/test.csv")
df
df=df[['Sex','Pclass','Age']]
df
df.info()
df.Age=df.Age.fillna(df.Age.mean())
df.info()

df=pd.get_dummies(df,columns=df[['Sex','Pclass']])
df
#df.Sex[df.Sex=="male"]=1
#df.Sex[df.Sex=="female"]=0
x_test=df[['Sex_female','Sex_male','Pclass_1','Pclass_2','Pclass_3','Age']]





df=pd.read_csv("d:/Desktop/python code/Titanic-keras/gender_submission.csv")
y_test=df.Survived

print("scores=",clf.score(x_test,y_test))

result=clf.predict(x_test)




#def deepNN(X_train,y_train,X_test,y_test):
model=Sequential()
model.add(Dense(units=40,input_dim=6,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units=30,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=x_train,y=y_train,validation_split=0.1,epochs=300,batch_size=100,verbose=2)

#x_train=np.array(x_train)

#plt.plot(train_history.history[x_train]) 

print(train_history)

print('scores=',model.evaluate(x=x_test,y=y_test)[1])

show_train_history(train_history,'acc','val_acc')


#deepNN(x_train,y_train,x_test,y_test)



#print('result=',result)
#for i in result:
#    print(i)
    


