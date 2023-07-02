# This is a sample Python script.
import numpy as np
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import matplotlib.pyplot as plt

from drawer import Drawer
from linear_model_optimizer import LinearModelOptimizer
from linear_model import LinearModel
from normalizer import Normalizer


def printDataFrameInfo(dataframe):
    print("Columns:")
    print(df.columns)
    print("Types:")
    print(df.dtypes)
    print("------")


def makeTrainSet(data_frame):
    size = data_frame.shape[0]
    train_set_size = 0.9 * size
    feature = data_frame.loc[:train_set_size, ["sqft_living", "bathrooms"]].values
    prediction = data_frame.loc[:train_set_size, ["price"]].values

    return feature, prediction, train_set_size


def onWeightUpdated(iteration, optimizer):
    if iteration % 100 == 0:
        drawer.drawFeatures(x[:, 0], x[:, 1], y)

        x1 = x[:, 0].min()
        x2 = x[:, 0].max()
        y1 = x[:, 1].min()
        y2 = x[:, 1].max()

        px = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        nx = x_normalizer.normalize(px)
        ny = linear_model.evaluate(nx)
        zz = y_normalizer.denormalize(ny)

        drawer.drawSurface(x1, x2, y1, y2, zz)
        drawer.drawErrors(optimizer.errors)
        drawer.flush()

        print("Error on iteration: ", iteration, optimizer.errors[-1])



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_csv('home_data.csv')

    printDataFrameInfo(df)

    x, y, size = makeTrainSet(df)

    x_normalizer = Normalizer(x)
    y_normalizer = Normalizer(y)

    nx = x_normalizer.normalize(x)
    ny = y_normalizer.normalize(y)

    drawer = Drawer()

    linear_model = LinearModel(2)

    optimizer = LinearModelOptimizer(linear_model, onWeightUpdated)
    optimizer.optimize(nx, ny, 10000)

    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
