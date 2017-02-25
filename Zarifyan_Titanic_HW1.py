# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 22:27:18 2017

@author: Masha
"""
import codecs
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, f1_score
import seaborn as sns
from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics, grid_search
import sklearn.ensemble as ske
from itertools import product
from sklearn.svm import SVC
import pydotplus, io
from IPython.core.display import Image 
sns.set(style="whitegrid", color_codes=True)
np.random.seed(sum(map(ord, "categorical")))

df = pd.read_csv('titanic.csv', index_col='PassengerId')
df['Age'].fillna((df['Age'].mean()), inplace=True) # В колонке Age очень много NaN-ов. Заменим их на среднее арифметическое возраста
  
'''Графики я закомментировала, т.к. они все вместе очень долго грузятся'''
#print(sns.barplot(x="Sex", y="Survived",  data=df))
'''
Как мы видим, вероятность выжить у женщин выше, чем у мужчин практически в 2.5 раза
Возможно, это связано с тем. что женщин спасали в первую очередь, 
первыми сажали в шлюпки
'''
#print(sns.barplot(x="Pclass", y="Survived",  data=df))
'''
Анализируя выживаемость в зависимости от соц.-экономического класса,
мы видим, что наибольшая доля выживших относится к 1 классу
'''
#print(sns.barplot(x="Pclass", y="Fare",  data=df))
'''
Стоимость билета в зависимости от соц.-эконом. класса: очевидно, что чем выше 
социально-экономический класс, тем состоятельнее человек, и тем дороже билет 
он может себе позволить: так, самые дорогие билеты имеют представители
1 класса, их средняя цена - 83 фунта у представителей 2 и 3 класса цена билетов 
сильно ниже и составляет примерно 20 и 17 фунтов соответственно
'''
#print(sns.barplot(x="Sex", y="Survived", hue="Pclass", data=df ))
'''
Из данного графика также видно, что у женщин из всех трех классов вероятность
выжить больше, чем у мужчин всех трёх классов. При этом, если посмотреть
на график повнимательнее, мы видим, что у мужчин в плане выживаемости сильно 
выделяется 1 класс (вероятность выжить - 38 %), в то время как у 2 и 3 класса
вероятность выжить в 2 раза ниже - примерно 18 и 17 % соответственно. У женщин, напротив,
сильно выделяются 1 и 2 классы (вероятности выжить - ~ 96 и 98 %), а у 3 класса
ниже аж в 2 раза - всего 50 %.
'''
#print(sns.barplot(x="SibSp", y="Survived",data=df))
'''
Сперва кажется, что чем больше семья, тем выше вероятность выжить, т.к. всю семью вместе спасают.
Однако, глядя на график видим, что наибольшая выживаемость у тех, кто был с 1-2 родственниками. 
Добавим этот параметр тоже.
'''
x_labels = ['Pclass', 'Fare', 'Age', 'Sex', 'SibSp']
X, y = df[x_labels], df['Survived']

pd.options.mode.chained_assignment = None  # default='warn'
'''
Меняем категориальные male и female на 0 и 1 соответственно. Первый способ (длинный):
'''
#print(X['Sex'].unique())
#X['Sex'] = X['Sex'].map({'female': 0, 'male':1}).astype(int)
# хорошо использовать pandas.get_dummies
#print(X['Sex'].unique())
#print(X)
'''Второй, более короткий и удобный с помощью get_dummies:'''
X['Sex']= pd.get_dummies(X['Sex'])
#print(X)
 
"Разделяем выборку на тренировочную и тестовую:"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
'''Строим дерево решений'''
clf = tree.DecisionTreeClassifier(min_samples_split=5, max_depth=10)
#print(clf.fit(np.array(X_train), np.array(y_train)))
#importances = pd.Series(clf.feature_importances_, index=x_labels)
#print(importances) 
print(clf.fit (X_train, y_train))
print(clf.score (X_test, y_test)) #модель предсказывает выживаемость 81% тестовой выборки
'''
dot_data = io.StringIO() 
tree.export_graphviz(clf, out_file=dot_data, 
feature_names=['Pclass', 'Fare', 'Age', 'Sex', 'SibSp']) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_png('titanic.png') 

Image(filename='titanic.png')
'''
'''Лес'''
data = RandomForestClassifier(n_estimators = 100)
data.fit(X_train, y_train)
y_pred = data.predict(X_test)
print(classification_report(y_test, y_pred))
scores = []
for t in range(1,100):
    rfc = RandomForestClassifier(n_estimators=t)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    scores.append(f1_score(y_test, y_pred))
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)

plt.plot(scores)
plt.xlabel('n_estimators')
plt.ylabel('score')
plt.show()