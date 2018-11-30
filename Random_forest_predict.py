import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import numpy as np
from sklearn.metrics import mean_squared_error

data = pd.read_excel('D:/test/10/NHANES.xlsx')
prediction = data.columns[:-1]
X_train,X_test,Y_train,Y_test = train_test_split(data[prediction],data['CKD_epi_eGFR'],test_size=0.2,random_state=1234)

# 训练模型
RF = ensemble.RandomForestRegressor(n_estimators=200,random_state=1234)
RF.fit(X_train,Y_train)

# 模型的检验
# 采用均方根误差RMSE
RF_predict = RF.predict(X_test)
RMSE = np.sqrt(mean_squared_error(Y_test,RF_predict))
print('均方根误差检验的结果RMSE=%.8f'%RMSE)
# 显示参数的重要性
print(pd.Series(RF.feature_importances_,index=X_train.columns))

result = pd.DataFrame({'Real':Y_test,'Predict':RF_predict},columns={'Real','Predict'})
print(result.head(10))
#使用模型预测
# test = X_test.iloc[0:4]
# RF_predtic = RF.predict(test)
# print(RF_predtic)
# print(Y_test.iloc[0:4])