import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import Lasso,LassoCV
from sklearn.metrics import mean_squared_error

diabetes = pd.read_excel('D:/test/8/diabetes.xlsx')
preditcion = diabetes.columns[2:-1]
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(diabetes[preditcion],diabetes['Y'],test_size=0.2,random_state=1234)


def LASSO_regression(x_train,x_test,y_train,y_test):
    lambdas = np.logspace(-5,2,200)
    lasso_cv = LassoCV(alphas=lambdas ,normalize=True,cv=10,max_iter=10000)
    lasso_cv.fit(x_train,y_train)
    lasso_best_lambda = lasso_cv.alpha_
    lasso = Lasso(alpha=lasso_best_lambda,normalize=True,max_iter=10000)
    lasso.fit(x_train,y_train)
    print('基于最佳lambda=%.9f,训练参数如下：'%lasso_best_lambda)
    print(pd.Series(index= ['intercpet']+x_train.columns.tolist(),data= [lasso.intercept_]+lasso.coef_.tolist()))

    lasso_predict = lasso.predict(x_test)
    RMSE = np.sqrt(mean_squared_error(y_test,lasso_predict))
    print('均方根误差检验的结果RMSE=%.8f'%RMSE)
    return lasso

lasso = LASSO_regression(X_train,X_test,Y_train,Y_test)

#test = pd.DataFrame({'BMI':[32.1],'BP':[101],'S1':[157],'S2':[93.2],'S3':[38],'S4':[4],'S5':[4.8598],'S6':[87]})
test = X_test.iloc[0:4]
lasso_predict = lasso.predict(test)
print(lasso_predict)