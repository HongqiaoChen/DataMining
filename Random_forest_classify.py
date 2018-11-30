import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import metrics

data = pd.read_table('D:/test/ssss.csv',sep=',')
predticion = data.columns[2:]
X_train,X_test,Y_train,Y_test = train_test_split(data[predticion],data['Survived'],test_size=0.25,random_state=1234)

#训练模型
RF = ensemble.RandomForestClassifier(n_estimators=200,random_state=1234)
RF.fit(X_train,Y_train)

#模型的检验
#准确度
RF_predtic=RF.predict(X_test)
Accuracy = metrics.accuracy_score(Y_test,RF_predtic)
print('准确率为Accuracy=%f'%Accuracy)
#AUC
Y_score=RF.predict_proba(X_test)[:,1]
fpr,tpr,threshould = metrics.roc_curve(Y_test,Y_score)
roc_auc = metrics.auc(fpr,tpr)
print('AUC=%f'%roc_auc)
#显示参数的重要性
print(pd.Series(RF.feature_importances_,index=X_train.columns))

#使用模型去预测（实现01分类)
test = X_test.iloc[0:4]
RF_predtic = RF.predict(test)
print(RF_predtic)
print(Y_test.iloc[0:4])