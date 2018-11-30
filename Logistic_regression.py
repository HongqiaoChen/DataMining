import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
import matplotlib.pyplot as plt


sport = pd.read_table('D:/test/9/Run or Walk.csv',sep=',')
prediction = sport.columns[4:]
X_train,X_test,Y_train,Y_test = train_test_split(sport[prediction],sport['activity'],test_size=0.25,random_state=1234)


logistic = LogisticRegression()
logistic.fit(X_train,Y_train)
Intercept = logistic.intercept_[0]
Parameter = logistic.coef_[0]
Odds = np.exp(Parameter)
print('截距的值为：Intercept=%.9f'%Intercept)
print('参数为：')
print(Parameter)
print('在以上条件下，得到的各系数的优势比如下')
print(pd.Series(index=X_train.columns.tolist(),data=Odds.tolist()))

#采用混淆矩阵的方法，使用X_test检验模型
logistic_predict = logistic.predict(X_test)
print(pd.Series(logistic_predict).value_counts())
cm = metrics.confusion_matrix(Y_test,logistic_predict,labels= [0,1])
Accuracy = metrics.scorer.recall_score(Y_test,logistic_predict)
Sensitivity = metrics.scorer.recall_score(Y_test,logistic_predict)
Specificity = metrics.scorer.recall_score(Y_test,logistic_predict,pos_label=0)
print('基于混淆矩阵的方法检验的结果如下：')
print('模型准确率为：%f'%Accuracy)
print('正例覆盖率：%f'%Sensitivity)
print('负例覆盖率：%f'%Specificity)

#采用AUC，对模型进行评估
Y_score = logistic.predict_proba(X_test)[:,1]
fpr,tpr,threshold = metrics.roc_curve(Y_test, Y_score)
roc_auc = metrics.auc(fpr,tpr)
plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')
plt.plot(fpr, tpr, color='black', lw = 1)
plt.plot([0,1],[0,1], color = 'red', linestyle = '--')
plt.text(0.5,0.3,'ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.show()

#用模型去计算单个的值
#test = pd.DataFrame({'acceleration_x':[0.265],'acceleration_y':[-0.7814],'acceleration_z':[-0.0076],'gyro_x':[-0.059],'gyro_y':[0.0325],'gyro_z':[-2.9296]})
#用模型去预测一组值
# test = X_test.iloc[0:4]
# print(test)
# logistic_predict = logistic.predict(test)
# print(logistic_predict)