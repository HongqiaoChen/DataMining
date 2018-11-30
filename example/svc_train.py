import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
from sklearn import svm

data = pd.read_csv('D:/test/sss.csv',sep=',')
prediction = data.columns[2:]
X = data[prediction]
Y = data['Survived']
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.2,random_state=1234)

C = np.array([0.05,0.1,0.5,1,2,4])
kernel = np.array(['rbf','linear','poly','sigmoid'])
param = {'C':C,'kernel':kernel}
base_model = GridSearchCV(estimator=svm.SVC(),param_grid=param,scoring='roc_auc',cv=5,n_jobs=4,verbose=1)
base_model.fit(X_train,Y_train)
best_C = base_model.best_params_['C']
best_kernel = base_model.best_params_['kernel']
print(best_C)
print(best_kernel)

svc = svm.SVC(C=best_C,kernel=best_kernel)
svc.fit(X_train,Y_train)
svc_predict = svc.predict(X_test)
cm = pd.crosstab(svc_predict,Y_test)
print('混淆矩阵如下：')
print(cm)
print('基于混淆矩阵的方法检验的结果如下：')
print(metrics.classification_report(Y_test,svc_predict))