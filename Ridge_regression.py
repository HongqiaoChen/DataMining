import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

diabetes = pd.read_excel('D:/test/8/diabetes.xlsx')
predictors = diabetes.columns[2:-1]
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(diabetes[predictors],diabetes['Y'],test_size=0.2,random_state=1234)

def Ridge_regression(x_train,x_test,y_train,y_test):
    Lambdas = np.logspace(-5,2,200)
    ridge_cv = RidgeCV(alphas=Lambdas,normalize=True,scoring='neg_mean_squared_error',cv=10)
    ridge_cv.fit(x_train,y_train)
    ridge_best_Lambda = ridge_cv.alpha_
    ridge = Ridge(alpha= ridge_best_Lambda,normalize=True)
    ridge.fit(x_train,y_train)
    ridge_best_intercept = ridge.intercept_
    ridge_best_coef = ridge.coef_
    print('基于最佳lambda=%.8f,训练参数如下：'%ridge_best_Lambda)
    print(pd.Series(index= ['intercpet']+x_train.columns.tolist(),data=[ridge_best_intercept]+ridge_best_coef.tolist()))

    ridge_predict = ridge.predict(x_test)
    RMSE = np.sqrt(mean_squared_error(y_test,ridge_predict))
    print('均方根误差检验的结果RMSE=%.8f'%RMSE)
    return ridge


ridge = Ridge_regression(X_train,X_test,Y_train,Y_test)
#test = pd.DataFrame({'BMI':[32.1],'BP':[101],'S1':[157],'S2':[93.2],'S3':[38],'S4':[4],'S5':[4.8598],'S6':[87]})
test = X_test.iloc[0:10]
print(test)
ridge_predict = ridge.predict(test)
print(ridge_predict)
