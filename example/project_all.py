import pandas as pd
from sklearn import model_selection
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
from sklearn import neighbors
from sklearn import svm
from sklearn import naive_bayes
import xgboost

def logistic(X_train,X_test,Y_train,Y_test):
    logistic = LogisticRegression()
    logistic.fit(X_train,Y_train)
    logistic_predict = logistic.predict(X_test)
    cm = pd.crosstab(logistic_predict,Y_test)
    print('混淆矩阵如下：')
    print(cm)
    print('基于混淆矩阵的方法检验的结果如下：')
    print(metrics.classification_report(Y_test,logistic_predict))
    Y_score = logistic.predict_proba(X_test)[:,1]
    fpr,tpr,threshold = metrics.roc_curve(Y_test, Y_score)
    roc_auc_logistic = metrics.auc(fpr,tpr)
    print('AUC=%f'%roc_auc_logistic)
    return roc_auc_logistic,logistic
def randomforest(X_train,X_test,Y_train,Y_test):
    rf = ensemble.RandomForestClassifier(random_state=100)
    rf.fit(X_train,Y_train)
    rf_predict = rf.predict(X_test)
    cm = pd.crosstab(rf_predict,Y_test)
    print('混淆矩阵如下：')
    print(cm)
    print('基于混淆矩阵的方法检验的结果如下：')
    print(metrics.classification_report(Y_test,rf_predict))
    Y_score = rf.predict_proba(X_test)[:,1]
    fpr,tpr,threshold = metrics.roc_curve(Y_test, Y_score)
    roc_auc_rf = metrics.auc(fpr,tpr)
    print('AUC=%f'%roc_auc_rf)
    return roc_auc_rf,rf
def adaboost(X_train,X_test,Y_train,Y_test):
    adaboost = ensemble.AdaBoostClassifier()
    adaboost.fit(X_train,Y_train)
    adaboost_predict = adaboost.predict(X_test)
    cm = pd.crosstab(adaboost_predict,Y_test)
    print('混淆矩阵如下：')
    print(cm)
    print('基于混淆矩阵的方法检验的结果如下：')
    print(metrics.classification_report(Y_test,adaboost_predict))
    Y_score = adaboost.predict_proba(X_test)[:,1]
    fpr,tpr,threshold = metrics.roc_curve(Y_test,Y_score)
    roc_auc_adaboost = metrics.auc(fpr,tpr)
    print(roc_auc_adaboost)
    return roc_auc_adaboost,adaboost
def XGboost(X_train,X_test,Y_train,Y_test):
    XGboost = xgboost.XGBClassifier()
    XGboost.fit(X_train,Y_train)
    XGboost_predict = XGboost.predict(X_test)
    cm = pd.crosstab(XGboost_predict,Y_test)
    print('混淆矩阵如下：')
    print(cm)
    print('基于混淆矩阵的方法检验的结果如下：')
    print(metrics.classification_report(Y_test,XGboost_predict))
    Y_score = XGboost.predict_proba(X_test)[:,1]
    fpr, tpr,threshold =metrics.roc_curve(Y_test,Y_score)
    roc_auc_xgboost = metrics.auc(fpr,tpr)
    print('AUC=%f'%roc_auc_xgboost)
    return roc_auc_xgboost,XGboost
def knn(X_train,X_test,Y_train,Y_test):
    knn = neighbors.KNeighborsClassifier()
    knn.fit(X_train,Y_train)
    knn_predict = knn.predict(X_test)
    cm = pd.crosstab(knn_predict,Y_test)
    print('混淆矩阵如下：')
    print(cm)
    print('基于混淆矩阵的方法检验的结果如下：')
    print(metrics.classification_report(Y_test,knn_predict))
    Y_score = knn.predict_proba(X_test)[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_score)
    roc_auc_knn = metrics.auc(fpr, tpr)
    print('AUC=%f' % roc_auc_knn)
    return roc_auc_knn,knn
def svc_nonlinearity(X_train,X_test,Y_train,Y_test):
    svc = svm.SVC(probability=True)
    svc.fit(X_train,Y_train)
    svc_predict = svc.predict(X_test)
    cm = pd.crosstab(svc_predict, Y_test)
    print('混淆矩阵如下：')
    print(cm)
    print('基于混淆矩阵的方法检验的结果如下：')
    print(metrics.classification_report(Y_test, svc_predict))
    Y_score = svc.predict_proba(X_test)[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(Y_test.map({0:0,1:1}), Y_score)
    roc_auc_svc = metrics.auc(fpr, tpr)
    print('AUC=%f' % roc_auc_svc)
    return roc_auc_svc,svc
def gnb(X_train,X_test,Y_train,Y_test):
    gnb = naive_bayes.GaussianNB()
    gnb.fit(X_train,Y_train)
    gnb_predict = gnb.predict(X_test)
    cm = pd.crosstab(gnb_predict,Y_test)
    print('混淆矩阵如下：')
    print(cm)
    print('基于混淆矩阵的方法检验的结果如下：')
    print(metrics.classification_report(Y_test,gnb_predict))
    Y_score = gnb.predict_proba(X_test)[:,1]
    fpr, tpr,threshold =metrics.roc_curve(Y_test,Y_score)
    roc_auc_gnb = metrics.auc(fpr,tpr)
    print('AUC=%f'%roc_auc_gnb)
    return roc_auc_gnb,gnb
def mnb(X_train,X_test,Y_train,Y_test):
    mnb = naive_bayes.MultinomialNB()
    mnb.fit(X_train,Y_train)
    mnb_predict = mnb.predict(X_test)
    cm = pd.crosstab(mnb_predict,Y_test)
    print('混淆矩阵如下：')
    print(cm)
    print('基于混淆矩阵的方法检验的结果如下：')
    print(metrics.classification_report(Y_test,mnb_predict))
    Y_score = mnb.predict_proba(X_test)[:,1]
    fpr, tpr,threshold =metrics.roc_curve(Y_test,Y_score)
    roc_auc_mnb = metrics.auc(fpr,tpr)
    print('AUC=%f'%roc_auc_mnb)
    return roc_auc_mnb,mnb
def bnb(X_train,X_test,Y_train,Y_test):
    bnb = naive_bayes.BernoulliNB()
    bnb.fit(X_train,Y_train)
    bnb_predict = bnb.predict(X_test)
    cm = pd.crosstab(bnb_predict,Y_test)
    print('混淆矩阵如下：')
    print(cm)
    print('基于混淆矩阵的方法检验的结果如下：')
    print(metrics.classification_report(Y_test,bnb_predict))
    Y_score = bnb.predict_proba(X_test)[:,1]
    fpr, tpr,threshold =metrics.roc_curve(Y_test,Y_score)
    roc_auc_bnb = metrics.auc(fpr,tpr)
    print('AUC=%f'%roc_auc_bnb)
    return roc_auc_bnb,bnb
# 数据1
data = pd.read_csv('D:/test/sss.csv',sep=',')
prediction = data.columns[2:]
X = data[prediction]
Y = data['Survived']
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.2,random_state=1234)

#数据2
# data = pd.read_table('D:/test/9/123.csv',sep=',')
# prediction = data.columns[4:]
# X_train,X_test,Y_train,Y_test = model_selection.train_test_split(data[prediction],data['activity'],test_size=0.25,random_state=1234)


auc_logistic,logistic = logistic(X_train,X_test,Y_train,Y_test)
auc_rf,rf = randomforest(X_train,X_test,Y_train,Y_test)
auc_adaboost,adaboost = adaboost(X_train,X_test,Y_train,Y_test)
auc_XGboost,XGboost = XGboost(X_train,X_test,Y_train,Y_test)
auc_knn,knn = knn(X_train,X_test,Y_train,Y_test)
auc_svc,svc = svc_nonlinearity(X_train,X_test,Y_train,Y_test)
auc_gnb,gnb = gnb(X_train,X_test,Y_train,Y_test)
#auc_mnb,mnb = mnb(X_train,X_test,Y_train,Y_test)
auc_bnb,bnb = bnb(X_train,X_test,Y_train,Y_test)
dict1 ={}
dict1['auc_logistic'] = auc_logistic
dict1['auc_rf'] = auc_rf
dict1['auc_adaboost'] = auc_adaboost
dict1['auc_XGboost'] = auc_XGboost
dict1['auc_knn'] = auc_knn
dict1['auc_svc'] = auc_svc
dict1['auc_gnb'] = auc_gnb
#dict1['auc_mnb'] = auc_mnb
dict1['auc_bnb'] = auc_bnb

AUC = pd.Series(dict1)
AUC = AUC.sort_values()
print(AUC)