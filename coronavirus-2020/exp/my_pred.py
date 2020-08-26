"""
Experiment summary
------------------
Compares Results of a Decision Tree Regressor
and Linear Regression for predicting cases
of COVID-19 in the US.

Much of this is based on:
https://medium.com/@randerson112358/predict-stock-prices-using-machine-learning-python-f554b7167b36
"""

import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import json

# ------------ HYPERPARAMETERS -------------
BASE_PATH = 'COVID-19/csse_covid_19_data/'
# ------------------------------------------

confirmed = os.path.join(
    BASE_PATH, 
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_US.csv')
confirmed = data.load_csv_data(confirmed)
tmpFeatures = []

for val in np.unique(confirmed["Country_Region"]):
    df = data.filter_by_attribute(
        confirmed, "Country_Region", val)
    cases, _ = data.get_cases_chronologically(df)
    tmpFeatures.append(cases)

tmpFeatures = np.concatenate(tmpFeatures, axis=0)
features = np.sum(tmpFeatures, axis=0)

newCases = np.zeros(features.shape, dtype=np.int32)
i = len(features) - 1
while i > 0:
    newCases[i] = features[i] - features[i - 1]
    i -= 1

newCases[0] = features[0]

dates = np.arange(len(newCases))

daysGiven = 7
daysPred = 3
totDays = daysGiven + daysPred

newFeatures = np.zeros((len(newCases) - totDays, daysGiven))
newLabels = np.zeros((len(newFeatures), daysPred))

for i in range(len(newFeatures)):
    newFeatures[i, :] = newCases[i:i+daysGiven]
    newLabels[i] = newCases[i+daysGiven:i+totDays]

x_train, x_test, y_train, y_test = train_test_split(newFeatures, newLabels, test_size=0.4)

tree = DecisionTreeRegressor().fit(x_train, y_train)

preds = tree.predict(x_test)

tmpDates = range(totDays)

diff = 0
acc = 0

for i in range(len(preds)):
    real = np.zeros(totDays)
    fake = np.zeros(totDays)
    real[0:daysGiven] = x_test[i]
    real[daysGiven:] = y_test[i]
    fake[0:daysGiven] = x_test[i]
    fake[daysGiven:] = preds[i]

    realSum = np.sum(y_test[i])
    fakeSum = np.sum(preds[i])

    thisDiff = fakeSum - realSum
    diff += thisDiff

    if realSum != 0:
        thisAcc = np.abs(thisDiff) / realSum

        line1, = plt.plot(tmpDates, fake)
        line1.set_label('preds')
        line2, = plt.plot(tmpDates, real)
        line2.set_label('actual')
        plt.title('Decision Tree Regressor')
        plt.legend()
        plt.show()
        plt.clf()            
        acc += thisAcc

diff /= len(preds)
acc /= len(preds)

print("\nDECISION TREE REGRESSION")
print("\naverage difference = ", diff)
print("average error = ", acc)

tree = LinearRegression().fit(x_train, y_train)

preds = tree.predict(x_test)

tmpDates = range(totDays)

diff = 0
acc = 0

for i in range(len(preds)):
    real = np.zeros(totDays)
    fake = np.zeros(totDays)
    real[0:daysGiven] = x_test[i]
    real[daysGiven:] = y_test[i]
    fake[0:daysGiven] = x_test[i]
    fake[daysGiven:] = preds[i]

    realSum = np.sum(y_test[i])
    fakeSum = np.sum(preds[i])

    thisDiff = fakeSum - realSum
    diff += thisDiff

    if realSum != 0:
        thisAcc = np.abs(thisDiff) / realSum

        line1, = plt.plot(tmpDates, fake)
        line1.set_label('preds')
        line2, = plt.plot(tmpDates, real)
        line2.set_label('actual')
        plt.title('Linear Regression')
        plt.legend()
        plt.show()
        plt.clf()            
        acc += thisAcc

diff /= len(preds)
acc /= len(preds)

print("\nLINEAR REGRESSION")
print("\naverage difference = ", diff)
print("average error = ", acc)
