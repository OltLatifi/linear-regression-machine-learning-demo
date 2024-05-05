from sklearn import linear_model
from pandas import DataFrame, Series
import pickle


class Model:
    def __init__(self, x_train: DataFrame, x_test: DataFrame, y_train: Series, y_test: Series) -> None:
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self) -> tuple[linear_model.LinearRegression, float]:
        linear = linear_model.LinearRegression()
        linear.fit(self.x_train, self.y_train)
        accuracy = linear.score(self.x_test, self.y_test)
        return (linear, accuracy)

    def save_model(self) -> None:
        linear = self.train()
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear[0], f)

    def load_model(self) -> linear_model.LinearRegression:
        pickle_in = open("studentmodel.pickle", "rb")
        linear = pickle.load(pickle_in)
        return linear
