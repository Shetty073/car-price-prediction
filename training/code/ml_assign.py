# -*- coding: utf-8 -*-
"""ml_assign.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yyIeHkpY1LVWvX55g9jiSI6DKmrBPWzz
"""

from google.colab import drive
drive.mount('/content/drive')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets, linear_model, preprocessing, model_selection, linear_model, metrics
import joblib
import pandas as pd

df = pd.read_csv('/content/drive/My Drive/car_data.csv')
df.head()

numerical_feature = [feature for feature in df.columns if df[feature].dtypes!="O"]
numerical_feature

for feature in numerical_feature:
   sns.scatterplot(x = df[feature], y = df['Selling_Price'])
   plt.show()

df = df.drop(['Car_Name', 'Fuel_Type', 'Seller_Type', 'Transmission'], axis=1)
df

scaler = preprocessing.MinMaxScaler()
scaler.fit(df)
dataset=pd.DataFrame(scaler.transform(df),columns=df.columns)
dataset.head()

X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=42)
X_test

lr = linear_model.LinearRegression()
lr.fit(X_train,y_train)

y_predLR = lr.predict(X_test)
metrics.r2_score(y_test, y_predLR)

joblib.dump(lr, '/content/drive/My Drive/car_price_lr.pkl')