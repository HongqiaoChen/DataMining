import pandas as pd
from sklearn import model_selection
from sklearn import svm
from sklearn import metrics

data = pd.read_excel('D:/test/11/Knowledge.xlsx')
prediction = data.columns[:-1]
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(data[prediction],data['UNS'],test_size=0.25,random_state=1234)

def SVM_Nonlinearity(X_train,X_test,Y_train,Y_test):
    kernel=['rbf','linear','poly','sigmoid']
    C=[0.1,0.5,1,2,5]
    parameters = {'kernel':kernel,'C':C}
    SVC = model_selection.GridSearchCV(estimator = svm.SVC(gamma='auto'),param_grid =parameters,scoring='accuracy',cv=10,verbose =1,)
    # 模型在训练数据集上的拟合
    SVC.fit(X_train,Y_train)
    # 返回交叉验证后的最佳参数值
    print(SVC.best_params_)
    # 模型在测试集上的预测
    SVC_predict = SVC.predict(X_test)
    # 模型的预测准确率
    accuracy = metrics.accuracy_score(Y_test,SVC_predict)
    print(accuracy)
    return SVC

def SVM_linearity(X_train,X_test,Y_train,Y_test):
    C=[0.1,0.5,1,2,5]
    parameters = {'C':C}
    SVC = model_selection.GridSearchCV(estimator = svm.LinearSVC(),param_grid =parameters,scoring='accuracy',cv=10,verbose =1,)
    # 模型在训练数据集上的拟合
    SVC.fit(X_train,Y_train)
    # 返回交叉验证后的最佳参数值
    print(SVC.best_params_)
    # 模型在测试集上的预测
    SVC_predict = SVC.predict(X_test)
    # 模型的预测准确率
    accuracy = metrics.accuracy_score(Y_test,SVC_predict)
    print(accuracy)
    return SVC

SVM_Nonlinearity(X_train,X_test,Y_train,Y_test)

SVM_linearity(X_train,X_test,Y_train,Y_test)