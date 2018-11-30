import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
from sklearn import metrics
import numpy as np

data = pd.read_excel('D:/test/11/CCPP.xlsx')
prediction = data.columns[:-1]
# 原数据中量纲不统一，需要标准化
X = minmax_scale(data[prediction])
X_train,X_test,y_train,y_test = train_test_split(X,data['PE'],test_size=0.25,random_state=1234)

K = np.arange(1,int(np.ceil(np.log2(data.shape[0]))))
mse = []
for k in K:
    KNN_cv = cross_val_score(neighbors.KNeighborsRegressor(n_neighbors=k,weights='distance'),X_train,y_train,cv=10,scoring='neg_mean_squared_error')
    mse.append(KNN_cv.mean())
arg_max = np.array(mse).argmax()
K_best = arg_max+1

KNN = neighbors.KNeighborsRegressor(n_neighbors=K_best,weights='distance')
KNN.fit(X_train,y_train)

KNN_predict = KNN.predict(X_test)
RMSE = np.sqrt(metrics.mean_squared_error(y_test,KNN_predict))
print('均方根误差检验的结果RMSE=%.8f'%RMSE)

# 对比
result = pd.DataFrame({'Real':y_test,'Predict':KNN_predict},columns={'Real','Predict'})
print(result.head(10))

#使用模型预测
test = X_test[0:4]
KNN_predict = KNN.predict(test)
print(KNN_predict)
print(y_test.iloc[0:4])