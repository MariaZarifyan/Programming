# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:30:16 2017

@author: Masha
"""
import re 
import pandas as pd
import string 
import numpy as np
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import * 
from nltk.stem.snowball import SnowballStemmer 
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.learning_curve import learning_curve 
from sklearn.linear_model import LogisticRegression

# https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

'''Открываем файл. Пока что у нас только две колонки: label (ham vs. spam) и message'''
f = pandas.read_csv('SMSSpamCollection', sep='\t',
                           names=["label", "message"])

'''Посчитаем соотношение спама к нормальным сообщениям:'''
ham = 0
spam = 0
ham_arr = []
spam_arr = []
for line in f['label']:
    if "ham" in line:
        ham += 1
    if 'spam' in line:
        spam += 1
#print('ham: ', ham, 'spam: ', spam)

'''Результат: ham:  4825 spam:  747. Как мы видим, выборка не сбалансирована. Спама почти в 7 раз меньше. '''
'''Посчитаем длину каждого сообщения и добавим еще одну колонку -length - с длинами'''
f['length'] = f['message'].map(lambda text: len(text))



tokens = [word_tokenize(i) for i in f]
def convert(text):
    tokens = word_tokenize(text)
    s = [str(token) for token in tokens]
    return s 
'''Токенизация с удалением знаков препинания:'''
def tokenize(text):
    text = text.lower()
    text = re.sub("[,;':.?!]", '', text)
    return word_tokenize(text)  

'''Знаки препинания считаются токенами:'''
def p_tokenize(text):
    text = text.lower()
    return wordpunct_tokenize(text) 

'''Стемминг. Используем SnowballStemmer '''
def ball_stemming(text):
    stemmer = SnowballStemmer("english") 
    tokens = word_tokenize(text)
    stemmas = [stemmer.stem(token) for token in tokens]
    return stemmas 

'''Лемматизация. Используем WordNetLemmatizer'''
def lemming(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas

'''Очистка от стоп-слов'''
def stop_words(text):
    stopWords = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if not w in stopWords] 
    return filtered 

#f.message = f.message.apply(tokenize)
#f.message.head().apply(convert)
#f.message = f.message.apply(convert)
'''Создаем матрицу терм-документ,  в параметр analyzer по очереди подставляем все наши функции и смотрим что получается'''
#bow = CountVectorizer(analyzer=tokenize)  #precision = 0.9917
#bow = CountVectorizer(analyzer=p_tokenize)  #precision = 0.9928
#bow = CountVectorizer(analyzer=ball_stemming) #precision = 0.9917
#bow = CountVectorizer(analyzer=stop_words) #precision = 0.9940
#bow = CountVectorizer(analyzer=lemming) #precision = 0.9903
'''TfidfVectorizer'''
#bow = TfidfVectorizer(analyzer=tokenize) #precision = 0.9694
#bow = TfidfVectorizer(analyzer=p_tokenize) #precision = 0.9766
#bow = TfidfVectorizer(analyzer=ball_stemming) #precision = 0.9666
#bow = TfidfVectorizer(analyzer=stop_words) #precision = 0.9766
bow = TfidfVectorizer(analyzer=lemming) #precision = 0.9698
bow.fit_transform(f['message']) 
bowed_messages = bow.transform(f['message'])
'''Подаем матрицу на вход байесовской модели:'''
naive_model = MultinomialNB()
naive_model.fit(bowed_messages, f['label'])
pred = naive_model.predict(bowed_messages) 
print(accuracy_score(f['label'], pred))
print(classification_report(f['label'], pred))


'''Разбиваем выборку на test set  и  train set '''
msg_train, msg_test, label_train, label_test = train_test_split(f['message'], f['label'], test_size=0.2)
print(len(msg_train), len(msg_test))
cv_results = cross_val_score(naive_model, bowed_messages, f['label'], cv=10, scoring='accuracy')
print(cv_results.mean(), cv_results.std())

pipeline = Pipeline([('bow', CountVectorizer(analyzer=tokenize)),('classifier', MultinomialNB())])

cv_results = cross_val_score(pipeline,msg_train,label_train, cv=10,scoring='accuracy')
print(cv_results.mean(), cv_results.std())

'''Learning_curve'''
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
#plot_learning_curve(pipeline, "accuracy vs. training set size", msg_train, label_train, cv=5)

#dummy = DummyClassifier()
#dummy.fit(bowed_messages, f['label'])
# naive_model.predict()