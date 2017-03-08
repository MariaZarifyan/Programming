# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:30:16 2017

@author: Masha
"""
import re 
import pandas 
import numpy as np
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
#from nltk.stem import * 
from nltk.stem.snowball import SnowballStemmer 
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.learning_curve import learning_curve 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import ShuffleSplit

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

'''Возможно, я что-то делаю не так: Dummy классификатор, который ниже, не работает, выдает ошибку -  не может переконвертировать строку в число'''
X_train, X_test, y_train, y_test = train_test_split(f['message'], f['label'], test_size=0.2) 
clf = DummyClassifier(strategy='most_frequent')
#clf.fit(X_train, y_train) 
#print(clf.score(X_train, y_train)) 

'''Поэтому мы просто представим, что каждому из новых сообщений будет присваиваться свойсто ham: '''
dummy_classifier = ['ham' for i in range(len(f))]
#print(accuracy_score(dummy_classifier, f.label))
'''
 Accuracy_score =  0.8659~0.866 довольно высок, однако если использовать такой подход при реальной
 классификации спама и не-спама, толку будет мало: такой подход не определит спам качественно, и 
 большинство спамовых сообщений попадут в желаемые, что может навредить компьютеру пользователя.
 '''
tokens = [word_tokenize(i) for i in f]

#def convert(text):
    #tokens = word_tokenize(text)
    #s = [str(token) for token in tokens]
    #return s 
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
spam_arr = f[f['label']=='spam']
#print([spam_arr])
ham_arr = f[f['label']=='ham']
#print([ham_arr])
'''Нормализуем выборку. Сделаем соотношение спама к не-спаму примерно 8:9.''' 
''' Всего 747 сообщений спама. Возьмем 373  - примерно половину из них.'''
train = pandas.concat([spam_arr[:373], ham_arr[:420]], ignore_index = True)
train.groupby('label').describe()

test = pandas.concat([spam_arr[373:], ham_arr[420:]], ignore_index = True)

msg_train = list(train['message'])
label_train = np.array(train['label'])
msg_test = list(test['message'])
label_test = np.array(test['label'])

'''Создаем матрицу терм-документ,  в параметр analyzer по очереди подставляем все наши функции и смотрим что получается:'''
bow = CountVectorizer(analyzer=tokenize)    
bow.fit_transform(msg_train)
bowed_train = bow.transform(msg_train)
bowed_test = bow.transform(msg_test)
'''Подаем матрицу на вход байесовской модели:'''
naive_model = MultinomialNB()
naive_model.fit(bowed_train, label_train)
pred = naive_model.predict(bowed_test)
#print(classification_report(label_test, pred))
#print(confusion_matrix(label_test, pred))
'''
Результаты, если токенизировать без знаков препинания: 
  precision    recall  f1-score   support

        ham       1.00      0.97      0.98      4405
       spam       0.71      0.94      0.81       374

avg / total       0.97      0.96      0.97      4779

[[4258  147]
 [  21  353]]    
'''
'''Токенизируем, учитывая знаки препинания '''
bow_p = CountVectorizer(analyzer=p_tokenize)   
 
bow_p.fit_transform(msg_train)
bowed_train = bow_p.transform(msg_train)
bowed_test = bow_p.transform(msg_test)

naive_model_p = MultinomialNB()
naive_model_p.fit(bowed_train, label_train)
pred = naive_model_p.predict(bowed_test)
#print(classification_report(label_test, pred))
#print(confusion_matrix(label_test, pred))
'''
Как мы видим, при токенизации со знаками препинания (считая знак за отдельный токен),
результаты чуть улучшились
  precision    recall  f1-score   support

        ham       1.00      0.97      0.99      4405
       spam       0.76      0.95      0.85       374

avg / total       0.98      0.97      0.97      4779

[[4293  112]
 [  17  357]]
'''

'''Стемминг (SnowBallStemmer)'''
bow_s = CountVectorizer(analyzer=ball_stemming)    
bow_s.fit_transform(msg_train)
bowed_train = bow_s.transform(msg_train)
bowed_test = bow_s.transform(msg_test)

naive_model_s = MultinomialNB()
naive_model_s.fit(bowed_train, label_train)
pred = naive_model_s.predict(bowed_test)
#print(classification_report(label_test, pred))
#print(confusion_matrix(label_test, pred))
'''
Стемминг работает хуже, и хуже распознает ham.
precision    recall  f1-score   support

        ham       1.00      0.97      0.98      4405
       spam       0.73      0.94      0.82       374

avg / total       0.97      0.97      0.97      4779

[[4276  129]
 [  21  353]]
'''
'''Лемматизация (WordnetLemmatizer)'''
bow_l = CountVectorizer(analyzer=lemming)    
bow_l.fit_transform(msg_train)
bowed_train = bow_l.transform(msg_train)
bowed_test = bow_l.transform(msg_test)

naive_model_l = MultinomialNB()
naive_model_l.fit(bowed_train, label_train)
pred = naive_model_l.predict(bowed_test)
#print(classification_report(label_test, pred))
#print(confusion_matrix(label_test, pred))
'''
При лемматизации в спам ошибочно попадает ещё больше нормальных сообщений:
precision    recall  f1-score   support

        ham       1.00      0.96      0.98      4405
       spam       0.65      0.94      0.77       374

avg / total       0.97      0.96      0.96      4779

[[4218  187]
 [  21  353]]
'''
'''Удаление стоп-слов'''
bow_w = CountVectorizer(analyzer=stop_words)    
bow_w.fit_transform(msg_train)
bowed_train = bow_w.transform(msg_train)
bowed_test = bow_w.transform(msg_test)

naive_model_w = MultinomialNB()
naive_model_w.fit(bowed_train, label_train)
pred = naive_model_w.predict(bowed_test)
#print(classification_report(label_test, pred))
#print(confusion_matrix(label_test, pred))
'''
Результаты сильно ухудшились. Опять очень много хороших писем уходит в спам.
 precision    recall  f1-score   support

        ham       0.99      0.95      0.97      4405
       spam       0.59      0.94      0.73       374

avg / total       0.96      0.94      0.95      4779

[[4164  241]
 [  23  351]]
'''
''' min vs. max document frequency'''
bow_min = CountVectorizer(min_df = 0.3)    
bowed_train = bow_min.fit_transform(msg_train)
bowed_test = bow_min.transform(msg_test)

naive_model_min = MultinomialNB()
naive_model_min.fit(bowed_train, label_train)
pred = naive_model_min.predict(bowed_test)
#print(classification_report(label_test, pred))
#print(confusion_matrix(label_test, pred))
'''
min_df
 precision    recall  f1-score   support

        ham       0.95      0.83      0.89      4405
       spam       0.19      0.47      0.27       374

avg / total       0.89      0.80      0.84      4779

[[3655  750]
 [ 197  177]]
'''
bow_max = CountVectorizer(max_df = 3)    
bowed_train = bow_max.fit_transform(msg_train)
bowed_test = bow_max.transform(msg_test)

naive_model_max = MultinomialNB()
naive_model_max.fit(bowed_train, label_train)
pred = naive_model_max.predict(bowed_test)
#print(classification_report(label_test, pred))
#print(confusion_matrix(label_test, pred))
'''
 precision    recall  f1-score   support

        ham       0.98      0.93      0.95      4405
       spam       0.49      0.79      0.60       374

avg / total       0.94      0.92      0.93      4779

[[4092  313]
 [  77  297]]

min_df и max_df (особенно min) очень плохо распознают класс сообщений, и снова 
большинство ham  классифицируют как спам
'''
'''TfIdfVectorizer'''
Tf = TfidfVectorizer()
train_tf = Tf.fit_transform(msg_train)
test_tf = Tf.transform(msg_test)

naive_model_tf = MultinomialNB()
naive_model_tf.fit(train_tf, label_train)
pred = naive_model_tf.predict(test_tf)
#print(classification_report(label_test, pred))
#print(confusion_matrix(label_test, pred))
'''
Сравнив все полученные результаты, мы видим, что TfIdfVectorizer показывает 
наилучший результат: всего 23 % ham  уходит в спам.
Если же выбирать между токенизацей с удалением знаков препинания,
токенизацией с сохранением знаков препинания, стеммингом и лематизацией,
то наилучший результат покажет токенизация с сохранением знаков препинания
(знак препинания считается за отдельный токен) 
precision    recall  f1-score   support

        ham       0.99      0.98      0.99      4405
       spam       0.77      0.93      0.84       374

avg / total       0.98      0.97      0.97      4779

[[4301  104]
 [  25  349]]
'''
'''Tree'''
clf = DecisionTreeClassifier()
'''TfIdfVectorizer:'''
clf.fit(train_tf, label_train)
pred = clf.predict(test_tf)
#print(classification_report(label_test, pred))
#print(confusion_matrix(label_test, pred))
'''
  precision    recall  f1-score   support

        ham       0.98      0.93      0.95      4405
       spam       0.48      0.82      0.61       374

avg / total       0.94      0.92      0.93      4779

[[4081  324]
 [  69  305]]
'''
clf.fit(bowed_train, label_train)
pred = clf.predict(bowed_test)
#print(classification_report(label_test, pred))
#print(confusion_matrix(label_test, pred))
'''
  precision    recall  f1-score   support

        ham       0.96      0.94      0.95      4405
       spam       0.46      0.59      0.52       374

avg / total       0.92      0.91      0.92      4779

[[4147  258]
 [ 154  220]]

Дерево определяет неправильно почти половину спама.
'''
'''Forest'''
forest = RandomForestClassifier()
forest.fit(train_tf, label_train)
pred = forest.predict(test_tf)
#print(classification_report(label_test, pred))
#print(confusion_matrix(label_test, pred))
'''
А лес справляется с классификацией в разы лучше дерева, классифицируя и спам и ham  с гораздо большей точностью: 
   precision    recall  f1-score   support

        ham       0.98      0.99      0.99      4405
       spam       0.88      0.82      0.85       374

avg / total       0.98      0.98      0.98      4779

[[4365   40]
 [  69  305]]
'''
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
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

title = "Learning Curve"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = DecisionTreeClassifier()
plot_learning_curve(estimator, title, train_tf.toarray(), label_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

plt.show()

