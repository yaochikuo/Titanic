
from Titanic_keras_sub_function import data_pre_process, deepNN


x_train,y_train,x_test,y_test=data_pre_process()

score_list=[]
for i in range(100):
    score=deepNN(x_train,y_train,x_test,y_test)
    score_list.append(score)
    print("i=%d , score=%.2f" %(i,score))

print("score_list",score_list)
print("average_correct_rate=",sum(score_list)/len(score_list))

