import numpy as np
np.random.seed(1337) # for reproducibility
from Titanic_keras_sub_function import data_pre_process, deepNN

x_train,y_train,x_test,y_test=data_pre_process()

y_test=deepNN(x_train,y_train,x_test,y_test)

for i in range(len(y_test)):
    print("Passenger ID=%d Survived=%d" %(i+892,int(np.round(y_test[i]))))


