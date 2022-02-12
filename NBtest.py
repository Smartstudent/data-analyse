from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import pandas as pd
import jieba
import re
import os

#读取测试集
news = pd.read_csv("test.csv", encoding = 'gb18030')
'''
#去除没有标签的样本
index = news['channelName'].notnull()
news = news[index]
print(news)
'''
#去标点
re_obj = re.compile(r"['~`!#$%^&*()_+-=|\';:/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'：【】《》‘’“”\s]+")
def clear(text):
    return re_obj.sub("", text)
news['title'] = news['title'].apply(clear)
print(news['title'])
#分词
def cut_word(text):
    return jieba.lcut(text)
news['title'] = news['title'].apply(cut_word)
print(news['title'])
#去停用词
def get_stopword():
    s = set()
    #with open('stopword.txt', encoding = 'utf-8') as f:
    with open('stopword.txt', encoding = 'utf-8') as f:
        for line in f:
            s.add(line.strip())
    return s
def remove_stopword(words):
    return [word for word in words if word not in stopword]
stopword = get_stopword()
news['title'] = news['title'].apply(remove_stopword)
print(news['title'])
#转化为空格分隔的字符串
def join(text_list):
    return " ".join(text_list)
news['title'] = news['title'].apply(join)
print(news['title'])
#标签映射
dic = {'财经' : 0, '房产' : 1, '教育' : 2, '科技' : 3, '军事' : 4, '汽车' : 5, '体育' : 6, '游戏' : 7, '娱乐' : 8, '养生健康' : 9, '历史' : 10, '搞笑' : 11, '旅游' : 12, '母婴' : 13}
news['channelName'] = news['channelName'].map(dic)
print(news['channelName'].value_counts())

x_test = news['title']
y_test = news['channelName']
print(x_test)
print(y_test)

vectorizer = joblib.load('vectorizer-version3.pkl')
model = joblib.load('model-version3')
prediction = model.predict(vectorizer.transform(x_test))
print(prediction)
print('accuracy_score:', accuracy_score(y_test, model.predict(vectorizer.transform(x_test))))
print('recall_score:', recall_score(y_test, model.predict(vectorizer.transform(x_test)), average = 'macro'))
