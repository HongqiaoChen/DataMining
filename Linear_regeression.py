import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels import api as sms
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


diabetes = pd.read_excel('D:/test/8/diabetes.xlsx')
prediction = diabetes.columns[2:-1]
X_train,X_test,Y_train,Y_test = train_test_split(diabetes[prediction],diabetes['Y'],test_size=0.2,random_state=1234)

def Linear_regeression(x_train,x_test,y_train,y_test):
    x_train_c = sms.add_constant(x_train)
    x_test_c = sms.add_constant(x_test)
    print(x_test_c)
    linear = sms.formula.OLS(y_train, x_train_c).fit()
    print('线性回归参数如下:')
    print(linear.params)
    linear_predict = linear.predict(x_test_c)
    RMSE = np.sqrt(mean_squared_error(y_test,linear_predict))
    print('均方根误差检验的结果RMSE=%.8f'%RMSE)
    return linear

linear = Linear_regeression(X_train,X_test,Y_train,Y_test)

#test = pd.DataFrame({'const':[1],'BMI':[32.1],'BP':[101],'S1':[157],'S2':[93.2],'S3':[38],'S4':[4],'S5':[4.8598],'S6':[87]})

test_c = sms.add_constant(X_test)
linear_predict = linear.predict(test_c)
linear_predict = list(linear_predict)
y_test = list(Y_test)
linear_predict = np.array(linear_predict)
y_test = np.array(y_test)
print(len(linear_predict))
print(len(y_test))
result = np.vstack((linear_predict,y_test))
result = np.transpose(result)
print(result)