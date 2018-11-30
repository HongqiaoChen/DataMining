import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn import metrics
import numpy as np

data = pd.read_csv('D:/test/sss.csv',sep=',')
prediction = data.columns[2:]
X = data[prediction]
Y = data['Survived']
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.2,random_state=1234)


max_depth = np.arange(1,10)
n_estimators = np.arange(50,200,50)
params = {'n_estimators':n_estimators,'max_depth':max_depth}
base_model = GridSearchCV(estimator=ensemble.RandomForestClassifier(),param_grid=params,scoring='roc_auc',cv=5,n_jobs=4,verbose=1)
base_model.fit(X_train,Y_train)
best_max_depth = base_model.best_params_['max_depth']
best_n_estimators = base_model.best_params_['n_estimators']

rf = ensemble.RandomForestClassifier(n_estimators=best_n_estimators,max_depth=best_max_depth,random_state=1234)
rf.fit(X_train,Y_train)
rf_predict = rf.predict(X_test)
cm = pd.crosstab(rf_predict, Y_test)
print('混淆矩阵如下：')
print(cm)
print('基于混淆矩阵的方法检验的结果如下：')
print(metrics.classification_report(Y_test, rf_predict))
Y_score = rf.predict_proba(X_test)[:, 1]
print(Y_score)
fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_score)
roc_auc_rf = metrics.auc(fpr, tpr)
print('AUC=%f' % roc_auc_rf)