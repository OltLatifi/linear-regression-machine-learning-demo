import pickle
import numpy as np
import typer


pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)


def predict(g1: int, g2: int, studytime: int, fails: int, paid: int, absences: int):
    to_predict = [g1, g2, studytime, fails, paid, absences]
    to_predict_np = np.array(to_predict).reshape(1, -1)
    print("The predicted grade is: ", linear.predict(to_predict_np))


if __name__ == "__main__":
    typer.run(predict)
