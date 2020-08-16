import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

### READING THE DATASET

dataset = pd.read_csv('C:/Users/mowab/Downloads/news/news.csv')

print(dataset.shape)
print(dataset.describe())
print(dataset.head())


#### IMPORT ALL RELEVANT MODULES

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix



#### DEFINING THE INDEPNDENT AND DEPENDENT VARIABLE
x = dataset.iloc[:,2]
y = dataset.iloc[:, -1]

print(x)
print(y)



#### SPLITING THE DATSET INTO TRAIN AND TEST
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
print(x_test)
print(x_train)



tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)


pass_agg_classifier = PassiveAggressiveClassifier(max_iter=70)
pass_agg_classifier.fit(tfidf_train, y_train)

y_pred = pass_agg_classifier.predict(tfidf_test)



#### CALCULATING THE ACCURACY SCORE

score = accuracy_score(y_pred, y_test)
acc_score = round(score*100 , 2)

print(score)
print(acc_score)
print("Accuracy is: {} %".format(acc_score))


####DESCRIBING THE CONFUSION MATRIX

con_mat = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
print(con_mat)


