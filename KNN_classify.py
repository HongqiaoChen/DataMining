import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import numpy as np
from sklearn import metrics

data = pd.read_excel('D:/test/11/Knowledge.xlsx')
prediction = data.columns[:-1]
X_train,X_test,Y_train,Y_test = train_test_split(data[prediction],data['UNS'],test_size=0.25,random_state=1234)

a = int(np.ceil(np.log2(data.shape[0])))
K = list(range(1,a))
K = np.array(K)

accuracy = []
for k in K:
    KNN_cv = model_selection.cross_val_score(neighbors.KNeighborsClassifier(n_neighbors=k,weights='distance'),X_train,Y_train,cv=10,scoring='accuracy')
    accuracy.append(KNN_cv.mean())
arg_max = np.array(accuracy).argmax()
K_best = arg_max+1

KNN = neighbors.KNeighborsClassifier(n_neighbors=K_best,weights='distance')
KNN.fit(X_train,Y_train)
#混淆矩阵
KNN_predict = KNN.predict(X_test)
cm = pd.crosstab(KNN_predict,Y_test)
Accuracy = metrics.accuracy_score(Y_test,KNN_predict)
print('混淆矩阵如下：')
print(cm)
print('基于混淆矩阵的方法检验的结果如下：')


test = X_test.iloc[0:4]
KNN_predict = KNN.predict(test)
print(KNN_predict)
print(Y_test.iloc[0:4])

