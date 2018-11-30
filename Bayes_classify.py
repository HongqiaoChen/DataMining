import pandas as pd
from sklearn import model_selection
from sklearn import metrics
from sklearn import naive_bayes
import jieba
from sklearn.feature_extraction.text import CountVectorizer

#当因变量全为连续的数值型时采用高斯贝叶斯分类
data = pd.read_excel('D:/test/12/Skin_Segment.xlsx')
print(data.head())
prediction = data.columns[:-1]
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(data[prediction],data['y'],test_size=0.2,random_state=1234)

gnb = naive_bayes.GaussianNB()
gnb.fit(X_train,Y_train)
gnb_predict = gnb.predict(X_test)
cm = pd.crosstab(gnb_predict,Y_test)
print('混淆矩阵如下：')
print(cm)
print('基于混淆矩阵的方法检验的结果如下：')
print(metrics.classification_report(Y_test,gnb_predict))

Y_score = gnb.predict_proba(X_test)[:,1]
fpr, tpr,threshold =metrics.roc_curve(Y_test.map({1:0,2:1}),Y_score)
roc_auc = metrics.auc(fpr,tpr)
print('AUC=%f'%roc_auc)

result = pd.DataFrame({'Real':Y_test,'Predict':gnb_predict},columns={'Real','Predict'})
print(result.head(10))

# 如果数据集中的自变量全为离散型的，使用多项式贝叶斯分类器
# data = pd.read_table('D:/test/12/mushrooms.csv',sep=',')
# 离散字符型转化为离散数值型
# prediction = data.columns[1:]
# for column in prediction:
#     data[column] = pd.factorize(data[column])[0]
# prediction = data.columns[1:]
# X_train,X_test,Y_train,Y_test = model_selection.train_test_split(data[prediction],data['type'],test_size=0.25,random_state=1234)
#
# mnb = naive_bayes.MultinomialNB()
# mnb.fit(X_train,Y_train)
#
# mnb_predict = mnb.predict(X_test)
# cm = pd.crosstab(mnb_predict,Y_test)
# print('混淆矩阵如下：')
# print(cm)
# print('基于混淆矩阵的方法检验的结果如下：')
# print(metrics.classification_report(Y_test,mnb_predict))
#
# Y_score = mnb.predict_proba(X_test)[:,1]
# fpr, tpr,threshold =metrics.roc_curve(Y_test.map({'edible':0,'poisonous':1}),Y_score)
# roc_auc = metrics.auc(fpr,tpr)
# print('AUC=%f'%roc_auc)
#
# result = pd.DataFrame({'Real':Y_test,'Predict':mnb_predict},columns={'Real','Predict'})
# print(result.head(10))

# 当数据集中的自变量X均为0-1 二元值时 ，优先选用伯努利贝叶斯分类器
# data = pd.read_excel('D:/test/12/Contents.xlsx')
# data.drop(columns={'NickName','Date'},axis=1,inplace=True)
#
# # 切词
# data.Content = data.Content.str.replace('[0-9a-zA-z]','')
# jieba.load_userdict('D:/test/12/all_words.txt')
# with open('D:/test/12/mystopwords.txt', encoding='UTF-8') as words:
#     stop_words = [i.strip() for i in words.readlines()]
# def cut_word(sentence):
#     words = [i for i in jieba.lcut(sentence) if i not in stop_words]
#     # 切完的词用空格隔开
#     result = ' '.join(words)
#     return result
# words = data.Content.apply(cut_word)
# print(words[:5])
#
# # 删去稀疏度高于99%的词，再将剩下的词构建是否出现矩阵（在本条评论中出现过1 未出现过0）
# counts = CountVectorizer(min_df=0.01)
# dtm_counts = counts.fit_transform(words).toarray()
# columns = counts.get_feature_names()
# X = pd.DataFrame(dtm_counts,columns=columns)
# Y = data['Type']
#
# X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.25,random_state=1234)
#
# bnb = naive_bayes.BernoulliNB()
# bnb.fit(X_train,Y_train)
#
# bnb_predict = bnb.predict(X_test)
# cm = pd.crosstab(bnb_predict,Y_test)
# print('混淆矩阵如下：')
# print(cm)
# print('基于混淆矩阵的方法检验的结果如下：')
# print(metrics.classification_report(Y_test,bnb_predict))
#
# Y_score = bnb.predict_proba(X_test)[:,1]
# fpr, tpr,threshold =metrics.roc_curve(Y_test.map({'Negative':0,'Positive':1}),Y_score)
# roc_auc = metrics.auc(fpr,tpr)
# print('AUC=%f'%roc_auc)