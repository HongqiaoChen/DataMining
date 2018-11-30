import pandas as pd
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


data = pd.read_excel('D:/test/14/default of credit card clients.xls')
print(data.head())
X = data.drop({'ID','Y'},axis=1)
Y = data['Y']
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.25,random_state=1234)

AdaBoost1 = ensemble.AdaBoostClassifier()
AdaBoost1.fit(X_train,Y_train)

AdaBoost1_predict = AdaBoost1.predict(X_test)
cm = pd.crosstab(AdaBoost1_predict,Y_test)
print('混淆矩阵如下：')
print(cm)
print('基于混淆矩阵的方法检验的结果如下：')
print(metrics.classification_report(Y_test,AdaBoost1_predict))

Y_score = AdaBoost1.predict_proba(X_test)[:,1]
fpr,tpr,threshold = metrics.roc_curve(Y_test,Y_score)
roc_auc = metrics.auc(fpr,tpr)
print(roc_auc)


#模型结果不理想的条件下，改进模型
#训练树的深度
importance = pd.Series(AdaBoost1.feature_importances_,index=X.columns)

prediction = list(importance[importance>0.02].index)
X_train_now = X_train[prediction]

max_depth = [3,4,5,6]
paramsl = {'base_estimator__max_depth':max_depth}
base_model = GridSearchCV(estimator=ensemble.AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),param_grid=paramsl,scoring='roc_auc',cv=5,n_jobs=4,verbose=1)
base_model.fit(X_train_now,Y_train)
best_depth = base_model.best_params_
print(best_depth)

#训练Ada算法的参数
learning_rate = [0.01,0.05,0.1,0.2]
n_estimators = [100,300,500]
max_depth = [3,4,5,6]
params = {'learning_rate':learning_rate,'n_estimators':n_estimators,'max_depth':max_depth}
adaboost = GridSearchCV(estimator = ensemble.GradientBoostingClassifier(),
                         param_grid= params, scoring = 'roc_auc', cv = 5, n_jobs = 4, verbose = 1)
adaboost.fit(X_train[prediction],Y_train)
best_learning_rate = adaboost.best_params_['learning_rate']
best_n_estimators = adaboost.best_params_['n_estimators']

AdaBoost2 = ensemble.AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=best_depth),n_estimators=best_n_estimators,learning_rate=best_learning_rate)
AdaBoost2.fit(X_train_now,Y_train)
AdaBoost2_predict = AdaBoost2.fit(X_test[prediction])
cm = pd.crosstab(AdaBoost2_predict,Y_test)
print('混淆矩阵如下：')
print(cm)
print('基于混淆矩阵的方法检验的结果如下：')
print(metrics.classification_report(Y_test,AdaBoost2_predict))

Y_score = AdaBoost2.predict_proba(X_test)[:,1]
fpr,tpr,threshold = metrics.roc_curve(Y_test,Y_score)
roc_auc = metrics.auc(fpr,tpr)
print(roc_auc)