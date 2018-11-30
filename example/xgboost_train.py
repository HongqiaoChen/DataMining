import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
import xgboost

data = pd.read_csv('D:/test/sss.csv',sep=',')
prediction = data.columns[2:]
X = data[prediction]
Y = data['Survived']
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.2,random_state=1234)

max_depth = np.arange(1,10)
learning_rate = np.arange(0.01,0.1,0.01)
n_estimators = np.arange(25,50,5)

params = {'max_depth':max_depth,'learning_rate':learning_rate,'n_estimators':n_estimators}
base_model = GridSearchCV(estimator=xgboost.XGBClassifier(),param_grid=params,scoring='roc_auc',cv=5,n_jobs=4,verbose=1)
base_model.fit(X_train,Y_train)
best_max_depth = base_model.best_params_['max_depth']
best_learin_rate = base_model.best_params_['learning_rate']
best_n_estimators = base_model.best_params_['n_estimators']
print(best_max_depth)
print(best_learin_rate)
print(best_n_estimators)

XGboost = xgboost.XGBClassifier(max_depth=best_max_depth,learning_rate=best_learin_rate,n_estimators=best_n_estimators)
XGboost.fit(X_train,Y_train)
XGboost_predict = XGboost.predict(X_test)
cm = pd.crosstab(XGboost_predict,Y_test)
print('混淆矩阵如下：')
print(cm)
print('基于混淆矩阵的方法检验的结果如下：')
print(metrics.classification_report(Y_test,XGboost_predict))
Y_score = XGboost.predict_proba(X_test)[:,1]
fpr, tpr,threshold =metrics.roc_curve(Y_test,Y_score)
roc_auc_xgboost = metrics.auc(fpr,tpr)
print('AUC=%f'%roc_auc_xgboost)
