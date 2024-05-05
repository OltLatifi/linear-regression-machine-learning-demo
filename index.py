import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn import linear_model

data = pd.read_csv("student-mat.csv", sep=";")

desired_factors = ['G1', 'G2', 'G3',
                   'studytime', 'failures', 'paid', 'absences']
data = data[desired_factors]
data.paid = data.paid.eq('yes').mul(1)

PREDICT = 'G3'

X = np.array(data.drop([PREDICT], axis=1))
Y = np.array(data[PREDICT])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, Y, test_size=0.8)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

accuracy = linear.score(x_test, y_test)

print("Accuracy: ", accuracy)
print("Coeficient: ", linear.coef_)

predictions = linear.predict(x_test)

for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])
