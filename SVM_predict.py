import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import numpy as np
from  sklearn import model_selection
data = pd.read_table('D:/test/13/forestfires.csv',sep=',')
data.drop('day',axis=1,inplace=True)
y = np.log1p(data.area)
data.month = pd.factorize(data.month)[0]
prediction = data.columns[:-1]
X_train,X_test,Y_train,Y_test = train_test_split(data[prediction],y,test_size=0.2,random_state=1234)

#使用网格搜索法，选取SVM回归中的最佳C、epsilon、gamma值
epsilon = np.arange(0.1,1.5,0.2)
C = np.arange(100,1000,200)
gamma = np.arange(0.001,0.01,0.002)
parameters = {'epsilon':epsilon,'C':C,'gamma':gamma}
svr = model_selection.GridSearchCV(estimator=svm.SVR(),param_grid=parameters,scoring='neg_mean_squared_error',cv=5,verbose=1,n_jobs=2)
svr.fit(X_train,Y_train)
print(svr.best_params_)
svr_predict = svr.predict(X_test)
RSEM = np.sqrt(metrics.mean_squared_error(Y_test,svr_predict))
print(RSEM)


