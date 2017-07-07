
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd



# In[6]:

from numpy import *
from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv('C:\\tias\\titanic_1\\titanic.csv', index_col='PassengerId')
main_data_frame = pd.DataFrame(data=data, columns=['Pclass', 'Fare', 'Age', 'Sex', 'Survived'])

main_data_frame = main_data_frame[["Pclass", "Fare", "Age", "Sex", "Survived"]].dropna()   #очищаем от Nano(в)

output = main_data_frame[["Pclass", "Fare", "Age", "Sex"]].replace("female",0).replace("male",1)

clf = DecisionTreeClassifier(random_state=241)
Y = main_data_frame['Survived']
X = output


print(X.columns)
clf.fit(X, Y)
print(clf.feature_importances_)

