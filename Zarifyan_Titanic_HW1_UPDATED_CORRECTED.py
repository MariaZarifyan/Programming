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
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score
import seaborn as sns
from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics, grid_search
import sklearn.ensemble as ske
from itertools import product
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
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
 # Зададим случайные параметры модели
clf = tree.DecisionTreeClassifier(min_samples_split=5, max_depth=10)
clf.fit(np.array(X_train), np.array(y_train))
#попробуем предсказать значение целевго признака "Survived"  по входным данным
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test) 
#Проверим, как работает наша модель. Для этого подсчитаем, сколько % составляют ошибки
error_train = np.mean(y_train != y_train_pred)
error_test  = np.mean(y_test  != y_test_pred)
#print(error_train, error_test) # 6.9 %  ошибок на тренировочной выборке и 19.4 -  на тестовой
#print(classification_report(y_test, y_test_pred))
#print(metrics.confusion_matrix(y_test, y_test_pred))
#print(np.mean(cross_val_score(clf, X_train, y_train, cv=5))) 

'''Массивы с параметрами для DecisionTreeClassifier-а'''
max_depth_arr = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] 
max_depth_i = [i for i in max_depth_arr] 
min_samples_split_arr = range(2,16,1)
min_samples_split_i = [i for i in min_samples_split_arr]  

'Переберём параметры с помощью GridSearchCV'''
parameters = {'min_samples_split' : min_samples_split_i, 'max_depth' : max_depth_i}
gs = grid_search.GridSearchCV(clf, parameters)
gs.fit(np.array(X_train), np.array(y_train))
#print('Best result is ',gs.best_score_)
#print(gs.best_params_) # Полученный результат - {'max_depth': 7, 'min_samples_split': 12} 
'''Таким образом, лучшее дерево - clf = tree.DecisionTreeClassifier(min_samples_split=12, max_depth=7)
Но у меня ни одним из способом  не устанавливается graphviz, чтобы его построить...
'''
new_clf = tree.DecisionTreeClassifier(min_samples_split=12, max_depth=7) 
new_clf.fit(X_train, y_train)
error_train = np.mean(y_train != new_clf.predict(X_train))
error_test  = np.mean(y_test  != new_clf.predict(X_test))
#print(error_train, error_test) # Теперь 1 %  ошибок на тренировочной выборке и 18.6 -  на тестовой

      

''' Посчитаем метрики и построим графики '''
# Метрики к параметру максимальная глубина дерева
d = {}
for i in max_depth_arr:
    statistics = []
    tree = DecisionTreeClassifier(max_depth = i)
    tree.fit(np.array(X_train), np.array(y_train))
    y_pred = tree.predict(X_test)
    statistics.append(f1_score(y_test, y_pred))
    statistics.append(precision_score(y_test, y_pred))
    statistics.append(accuracy_score(y_test, y_pred))
    statistics.append(recall_score(y_test, y_pred))
    d[i] = statistics

result = pd.DataFrame(d, index = ['f1_score', 'precision_score', 'accuracy_score', 'recall_score'], columns = d.keys())
#print(result) # получаем матрицу с метриками к каждому параметру

result[0:1].plot.bar()
 
d1 = {} 
for i in min_samples_split_arr:
    statistics = []
    tree = DecisionTreeClassifier(min_samples_split = i)
    tree.fit(np.array(X_train), np.array(y_train))
    y_pred = tree.predict(X_test)
    statistics.append(f1_score(y_test, y_pred))
    statistics.append(precision_score(y_test, y_pred))
    statistics.append(accuracy_score(y_test, y_pred))
    statistics.append(recall_score(y_test, y_pred))
    d1[i] = statistics
     
result1 = pd.DataFrame(d1, index = ['f1_score', 'precision_score', 'accuracy_score', 'recall_score'], columns = d1.keys())
#print(result1)
result1[0:1].plot.bar() # каждый столбец - это один параметр из списка
# Как мы видим, в min_samples_split_arr практически все параметры имеют в среднем одинаковые метрики (между 0.7 и 0.8)
# Но, если посмотреть на график метрик максимальной глубины дерева, расположенный выше, мы видим, что метрикиmax_depth=2  сильно меньше метрик остальных параметров
'''Random---Forest---Classifier'''
rf = RandomForestClassifier(n_estimators=100, random_state=11)
rf.fit(X_train, y_train)

error_train = np.mean(y_train != rf.predict(X_train))
error_test  = np.mean(y_test  != rf.predict(X_test))

#print(error_train, error_test) # 1.7 %  ошибок на тренировочной выборке и 17.5-  на тестовой

n_estimators_arr = range(100, 200, 20)
n_estimators_i = [i for i in n_estimators_arr] 
random_state_arr = range(10, 30, 1)
random_state_i = [i for i in random_state_arr] 

parameters = {'n_estimators': n_estimators_i, 'random_state': random_state_i}  
gs = grid_search.GridSearchCV(rf, parameters)
gs.fit(X_train, y_train)
#print('Best result is ',gs.best_score_)
#print(gs.best_params_)  

#Best result is  0.8218298555377207
#{'n_estimators': 140, 'random_state': 28} NB: Очень-очень долго работает, несмотря на то, что тут всего 2 параметра..

  
 
 

#Вывод: сравнивая работу моделей дерева решений DecisionTreeClassifier и леса, мы видим, что
#алгоритм случайного леса работает чуть лучше

# Посмотрим, как ранжированы признаки по мере значимости, выживет человек или нет

 


feature_names = X.columns
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
#print("Feature importances:")
#for f, idx in enumerate(indices):
   #print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, feature_names[idx], importances[idx]))
'''
Рейтинг наших признаков: как мы видим, самый значимый из них  - цена билета
в то время как наличие родственников не имело практически никакой связи с выживаемостью.
Таким образом, самыми значимыми признаками в данном случае можно назвать цену билета, пол и возраст:
Feature importances:
 1. feature 'Fare ' (0.3145)
 2. feature 'Sex  ' (0.2715)
 3. feature 'Age  ' (0.2542)
 4. feature 'Pclass' (0.1083)
 5. feature 'SibSp' (0.0514)
 
'''
 

