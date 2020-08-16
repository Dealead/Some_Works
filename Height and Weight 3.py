from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from pandas import read_html
import sklearn
import html5lib
import matplotlib.pyplot as plt


url = "http://socr.ucla.edu/docs/resources/SOCR_Data/SOCR_Data_Dinov_020108_HeightsWeights.html"
dataset = pd.read_html(url)[0]

x = np.array(dataset.iloc[1:, 1])
y = np.array(dataset.iloc[1:, -1])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=5)
classifier.fit(x_train, np.ravel(y_train))
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)

score = accuracy_score(y_test, y_pred)
print(score)

