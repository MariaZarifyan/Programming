# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:01:15 2017

@author: Masha
"""
import operator
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk import word_tokenize, wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
sns.set(style="whitegrid", color_codes=True)
np.random.seed(sum(map(ord, "categorical")))

def tokenize(text):
    text = text.lower()
    text = re.sub("[,;':.?!]", '', text)
    return word_tokenize(text)  

def lemming(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas

'''Датасет с репликами героев South Park. Выберем 7-9 сезоны:'''
f = pd.read_csv("DataSet.csv") #11366 строк
f = f.dropna() 

'''Возьмем фразы четырех главных героев'''
Kenny = (f.Character == "Kenny")
Cartman = (f.Character == "Cartman")
Kyle = (f.Character == "Kyle")
Stan = (f.Character == "Stan")

K  = f[Kenny]
C = f[Cartman]
Ky = f[Kyle]
S = f[Stan]

'''
print(len(K), len(C), len(Ky), len(S))
Output: 116 1627 1207 1390
Как мы видим, у Кенни сильно меньше фраз, чем у остальных героев, что делает нашу выборку 
сильно несбалансированной.
Попробуем взять не Кенни, а кого-нибудь другого: 
'''
Randy = (f.Character == "Randy")
R = f[Randy]
'''
print(len(R))
Output: 479 реплик. Так чуть лучше. Возьмем вместо Кенни Randy. 
'''
characters = f[Cartman | Kyle | Stan | Randy]
characters = characters.dropna()
characters = characters.reset_index(drop=True)

'''Делим выборку на трнировочную и на тестовую: '''
X_train, X_test, y_train, y_test = train_test_split(characters['Line'], characters['Character'], test_size=0.2)

countvec = CountVectorizer(tokenizer = word_tokenize, stop_words = "english")
bowed_fit_train = countvec.fit_transform(X_train)
bowed_train = countvec.transform(X_train)
bowed_test = countvec.transform(X_test)  

'''В качестве Baseline классификатора используем DummyClassifier:  '''
dummy = DummyClassifier()
dummy.fit(bowed_train, y_train)
pred = dummy.predict(bowed_test)

'''
#print(classification_report(y_test, predicted)) 

             precision    recall  f1-score   support

    Cartman       0.34      0.34      0.34       322
       Kyle       0.24      0.25      0.24       232
      Randy       0.05      0.04      0.04       118
       Stan       0.29      0.29      0.29       269

avg / total       0.26      0.26      0.26       941
Результаты совсем плохие. Попробуем другие классификаторы
Наивный Байес:
'''
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(bowed_train, y_train) 
predicted = clf.predict(bowed_test)
'''
print(classification_report(y_test, predicted)) 
error_test  = np.mean(y_test  != predicted)
print(error_test)
 precision    recall  f1-score   support

    Cartman       0.49      0.69      0.58       340
       Kyle       0.48      0.33      0.39       233
      Randy       0.75      0.15      0.25        99
       Stan       0.42      0.45      0.44       269

avg / total       0.50      0.48      0.46       941

0.524973432519
Результат значительно лучше, чем у Dummy Classifier-a, но всё равно процент ошибки высок.
RandomForest:
'''
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier()
forest.fit(bowed_train, y_train) 
predicted = forest.predict(bowed_test)
'''
print(classification_report(y_test, predicted)) 
error_test  = np.mean(y_test  != predicted)
print(error_test)
 precision    recall  f1-score   support

    Cartman       0.51      0.58      0.54       337
       Kyle       0.39      0.37      0.38       226
      Randy       0.53      0.25      0.34       106
       Stan       0.37      0.40      0.39       272

avg / total       0.44      0.44      0.44       941

0.558979808714
Результат чуть хуже.
Дерево:
'''
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(bowed_train, y_train) 
predicted = tree.predict(bowed_test)
'''
print(classification_report(y_test, predicted)) 
error_test  = np.mean(y_test  != predicted)
print(error_test)
             precision    recall  f1-score   support

    Cartman       0.46      0.48      0.47       328
       Kyle       0.34      0.32      0.33       250
      Randy       0.29      0.25      0.27        89
       Stan       0.35      0.38      0.37       274

avg / total       0.38      0.38      0.38       941

0.616365568544
У дерева результат ещё хуже. 
Логистическая регрессия:
'''
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(bowed_train, y_train) 
predicted = log.predict(bowed_test)
'''
print(classification_report(y_test, predicted)) 
error_test  = np.mean(y_test  != predicted)
print(error_test)

             precision    recall  f1-score   support

    Cartman       0.61      0.62      0.61       342
       Kyle       0.45      0.34      0.39       245
      Randy       0.64      0.30      0.41        92
       Stan       0.42      0.58      0.49       262

avg / total       0.52      0.51      0.50       941

0.493092454835
А логистическая регрессия работает лучше.
Сравнив результаты, полученные с помощью Байесовского классификатора, леса, дерева
и логистической регрессии, лучший результат показывает логистическая регрессия.
#Зададим параметр class_weight = 'balanced', т.к. у нас у одного из персонажей - Randy - фраз меньше, чем у других.  
'''
log1 = LogisticRegression(class_weight = 'balanced')
log1.fit(bowed_train, y_train) 
predicted = log1.predict(bowed_test)

print(classification_report(y_test, predicted)) 
error_test  = np.mean(y_test  != predicted)
print(error_test)

#cm = confusion_matrix(y_test, predicted)
#rint(cm)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, predicted)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes= ['Randy', 'Cartman', 'Kyle', 'Stan'],
                      title='South Park Logistic Regession')

# Plot normalized confusion matrix

plt.show()
print(log1.coef_)
print(log1.intercept_)
print(log1.classes_)
'''
[[ 0.25775081 -0.15800968  0.07197206 ...,  0.13502535 -0.05061749
  -0.10127796]
 [-0.18822322  0.04241628 -0.02909244 ..., -0.02685396  0.43593319
   0.2498792 ]
 [ 0.17302471 -0.10100682 -0.00211829 ..., -0.10086209 -0.00255736
  -0.02697291]
 [-0.48266141  0.17984736 -0.06810925 ..., -0.08589501 -0.24048126
  -0.15307864]]
 
---ВЫВОД--- 4 массива с коэффициентами соответствуют четырём выбранным нами персонажам. Как мы видим,
 одни из коэффициентов положительные, другие отрицательные. Каждый элемент массива (каждый коэффициент) соответствует слову
 из реплики персонажа. Положительный коэффициент означает, что данный признак
 повышает вероятность того, что конкретная реплика принадлежит конкретному персонажу,
отрицательный - что признак понижает вероятность. 
Если коэффициент регрессии очень большой, то это означает, что признак сильно влияет на вероятность результата,
а если коэффициент стремится к нулю, то этот признак практически не влияет на результат. В нашем же случае коэффициенты малы.

[-1.47270515 -0.75824546 -1.82046728 -0.52373872]

['Cartman' 'Kyle' 'Randy' 'Stan']
'''
'''
     precision    recall  f1-score   support

    Cartman       0.63      0.57      0.60       335
       Kyle       0.43      0.48      0.45       223
      Randy       0.54      0.45      0.49        99
       Stan       0.49      0.52      0.51       284

avg / total       0.53      0.52      0.52       941

0.47821466525
C параметром class_weight = 'balanced' результат стал чуть лучше.
'''

tf = TfidfVectorizer(tokenizer=word_tokenize , stop_words="english")
tr_tf = tf.fit_transform(X_train)
tst_tf = tf.transform(X_test)
LogR = LogisticRegression(class_weight = 'balanced')
LogR.fit(tr_tf, y_train)
'''
predicted = LogR.predict(tst_tf)
print(classification_report(y_test, predicted))  
             precision    recall  f1-score   support

    Cartman       0.56      0.64      0.59       320
       Kyle       0.43      0.44      0.43       216
      Randy       0.47      0.43      0.45       111
       Stan       0.49      0.43      0.46       294

avg / total       0.50      0.50      0.50       941

Если использовать не CountVectorizer, a TfidfVectoriser,  то результат чуть хуже, но не сильно.
'''
'''
Теперь попробуем узнать, зависит ли длина фразы от героя, который её произнёс. 
На первый взгляд, кажется, что одни герои разговорчивы, другие отвечают более односложно.
Добавим колонку length  и посчитаем длину реплики.
''' 
characters['length'] = characters['Line'].map(lambda text: len(text)) 

#print(sns.barplot(x="Character", y="length",  data=characters))
'''
По графику видно, что самые длинные реплики у Картмана (длина около 73) и у Рэнди (длина около 67)
У Кайла и Стэна покороче - (примерно 52 и 51 соответственно)
Теперь посмотрим, есть ли у каждого из героев свои уникальные слова: для начала соберем 
весь текст для каждого героя, затем токенизируем его, очистим от знаков препинания и 
уберем стоп-слова. Для начала посмотрим, вдруг у кого-нибудь из героев есть уникальные слова,
которые он повторяет довольно часто. Составим частотные списки: 
'''
Randys_words = ''
Kyles_words = ''
Cartmans_words = ''
Stans_words = ''

for i in R['Line']:
    Randys_words += i 
for i in Ky['Line']:
    Kyles_words += i 
for i in C['Line']:
   Cartmans_words += i 
for i in S['Line']:
    Stans_words += i 

def freq(words):
    words = tokenize(words)
    d = {}
    stopWords = stopwords.words('english') 
    for i in words:
     if i not in stopWords:
      if i in d:
       d[i]+=1
      else:
        d[i] = 1

    for znach in (sorted(val for key,val in d.items())):
       for key,val in d.items():
            if val==znach:
              if val > 50:
                 print(key,val)
'''
#freq_Stan = freq(Stans_words)  
#freq_Kyle = freq(Kyles_words) 
#freq_Randy = freq(Randys_words) 
freq_Cartman = freq(Cartmans_words) 
В итоге, ничего интересного тут не нашлось. Из частотных списков
видно, что персонажи всё время употребляют слово dude,  а совсем уникальных слов, которые могли бы быть
фишкой каждого персонажа, нет.  
Тогда добавим  в качестве фичи только длину фразы.
И посмотрим, как это повлияет на класификацию:
'''

#characters['length'].fillna((characters['length'].mean()), inplace=True)
'''
X, y  = characters['length'], characters['Character']
X = X[:, None]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
'''
'''
pred = rf.predict(X_test)
print(classification_report(y_test, pred))
             precision    recall  f1-score   support

    Cartman       0.38      0.44      0.41       330
       Kyle       0.27      0.28      0.28       228
      Randy       0.17      0.04      0.07        98
       Stan       0.32      0.33      0.32       285

avg / total       0.31      0.33      0.31       941
'''
'''Байес'''
#b = MultinomialNB()
#b.fit(X_train, y_train)
'''
predicted = b.predict(X_test)
print(classification_report(y_test, predicted)) 
          precision    recall  f1-score   support

    Cartman       0.35      1.00      0.52       330
       Kyle       0.00      0.00      0.00       228
      Randy       0.00      0.00      0.00        98
       Stan       0.00      0.00      0.00       285

avg / total       0.12      0.35      0.18       941
'''
#logR = LogisticRegression(class_weight = 'balanced')
#logR.fit(X_train, y_train) 
'''
predicted = logR.predict(X_test)
print(classification_report(y_test, predicted)) 
 precision    recall  f1-score   support

    Cartman       0.40      0.68      0.51       330
       Kyle       0.00      0.00      0.00       228
      Randy       0.00      0.00      0.00        98
       Stan       0.34      0.45      0.39       285

avg / total       0.24      0.38      0.30       941
Во всех случах результаты значительно хуже,чем у векторизации. Но этого и следовало ожидать - 
разница между длиной предложений у каждого из героев совсем не большая, и классифицировать по ней
невозможно. Так что это была плохая идея.


    Итак, можно сделать вывод, что лучшие результаты показывает логистическая регрессия с векторизованнными
    данными с помощью CountVectorizer. 
    Посмотрим на них еще раз. (Визуализировать будем выше, в строке 193)
'''



