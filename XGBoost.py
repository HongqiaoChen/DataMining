import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import model_selection
import xgboost
import numpy as np
from sklearn import metrics

data = pd.read_table('D:/test/14/creditcard.csv',sep=',')
X = data.drop({'Time','Class'},axis=1)
Y = data.Class
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.3,random_state=1234)

# 检测到违约占比太低，数据出现严重的偏移，不适合用本数据直接建模，而应该使用SMOTE算法过度抽样
counts = data.Class.value_counts()
print('违约占比%f'%(counts[1]/(counts[0]+counts[1])))
over_sample = SMOTE(random_state=1234)
over_sample_X,over_sample_Y = over_sample.fit_sample(X_train,Y_train)

xgboost = xgboost.XGBClassifier()
xgboost.fit(over_sample_X,over_sample_Y)

xgboost_predict = xgboost.predict(np.array(X_test))
cm = pd.crosstab(xgboost_predict,Y_test)
print('混淆矩阵如下：')
print(cm)
print('基于混淆矩阵的方法检验的结果如下：')
print(metrics.classification_report(Y_test,xgboost_predict))

Y_score = xgboost.predict_proba(np.array(X_test))[:,1]
fpr, tpr,threshold =metrics.roc_curve(Y_test,Y_score)
roc_auc = metrics.auc(fpr,tpr)
print('AUC=%f'%roc_auc)
