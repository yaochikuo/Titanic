# -*- coding: utf-8 -*-
import pandas as pd 
from pandas import DataFrame as df
import numpy as np
from sklearn.linear_model import LogisticRegression 

# step 1 get data
data=pd.read_csv("train.csv")
data.info()
print(data.describe())
test=pd.read_csv("test.csv")



# step 2 observe data vs survived
import matplotlib as plt
import seaborn as sns
def observData():
    color={0:'Die',
           1:'Alive'}
    #sns.countplot(data.Survived,hue=data.Survived.map(color)) 
#sns.countplot(data['Survived'])
#sns.countplot(data,hue=data.Pclass)
observData()
#=========step 2  Fill in Data
data.Age=data.Age.fillna(data.Age.mean())
test.Age=test.Age.fillna(test.Age.mean())

#data.Embarked=data.Embarked.fillna('S')
#sns.countplot(data.Embarked,hue=data.Survived.map(color)) 

#sns.distplot(data.Age)

#============ step 3 Convert category variables
data=pd.get_dummies(data,columns=data[['Sex','Embarked']])
test=pd.get_dummies(test,columns=test[['Sex']])
test.info()
print("=====Above check dummy variables=============")
X_train=data[['Age','Pclass','Sex_female','Sex_male']]
y_train=data['Survived']
#print("X_train=",X_train)
#print("y_train=",y_train) 
X_test=test[['Age','Pclass','Sex_female','Sex_male']]



#============ step 4 Classifier



from sklearn.ensemble import RandomForestClassifier as RFC
model=RFC()
print(model)
model.fit(X_train,y_train)
y_test=model.predict(X_test)

print("\n\n=====Training score=====",model.score(X_train,y_train))
#print("\n\n=====Predict score=====",model.score(X_test,y_test))


test['Survived']=y_test
submit=test[['PassengerId','Survived']]
print(submit)
submit.to_csv("submit.csv",mode='w',index=False)

