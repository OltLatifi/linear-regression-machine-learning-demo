import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle
from model import Model

data = pd.read_csv("student-mat.csv", sep=";")

desired_factors = ['G1', 'G2', 'G3',
                   'studytime', 'failures', 'paid', 'absences']
data = data[desired_factors]
data.paid = data.paid.eq('yes').mul(1)

PREDICT = 'G3'

X = np.array(data.drop([PREDICT], axis=1))
Y = np.array(data[PREDICT])

best_accuracy = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.2)
    model = Model(x_train, x_test, y_train, y_test)
    linear, accuracy = model.train()

    print(_, " Accuracy: ", accuracy)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        model.save_model()
