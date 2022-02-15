import pandas as pd
import numpy as np

# Helper functions for this:
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# demonstration of calculating metrics for a neural network model using sklearn
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score

# stuff:
import tensorflow as tf
from tensorflow import keras
from keras import Input
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.layers import Bidirectional
from keras.layers import TimeDistributed


def CalcMAPE(y_true: np.ndarray, y_pred: np.ndarray, axis: int) -> np.ndarray:
    """
    Calculates the mean absolute precentage error.

    Arguments:
        y_true (ndarray)    : n dim array of true y values
        y_pred (ndarray)    : n dim array of predicted y values
        axis (int)          : axis to average values

    Return:
        MAPE as a ndarray
    """
    abs_error = np.abs((y_true - y_pred) / y_true)
    mape = 100.0 * np.mean(abs_error, axis=axis)
    return mape


def CalcMAE(y_true: np.ndarray, y_pred: np.ndarray, axis: int) -> np.ndarray:
    """
    Calculates the mean absolute error.

    Arguments:
        y_true (ndarray)    : n dim array of true y values
        y_pred (ndarray)    : n dim array of predicted y values
        axis (int)          : axis to average values

    Return:
        MAE as a ndarray
    """
    abs_error = np.abs(y_true - y_pred)
    mae = np.mean(abs_error, axis=axis)
    return mae


class DataEval:
    """
    Evaluates a model on test data using various metrics
    """

    def __init__(self):
        # TODO: add model by default?
        # TODO: add type hinting for keras objects
        # TODO: remove unneeded variables (maybe not all need to be tracked)
        self.model_name = "Unknown Model"
        self.model = None
        self.history = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None

    def EvaluateModel(self, model_name, model, history, x_test, y_test):
        """
        Evaluates a model in the following ways:
            1: makes a prediction on the test data
            2: plots the history metrics from training
            3: plots the error of the predictions
            4: plots N random predictions vs. actual values
        """
        # [0] update vairables
        self.model_name = model_name
        self.model = model
        self.history = history
        self.x_test = x_test
        self.y_test = y_test

        # [1] make prediction on dataset
        self.y_pred = model.predict(self.x_test)

        # [2] plot history
        self._PlotHistory()

        # [3] plot distribution
        self._PlotDistribution()

        # [4] plot error graphs
        self._PlotError()

        # [5] plot N random plots
        self.PlotRandomPlots(num_plot=5)

    def _PlotHistory(self):
        """
        Plots the fit history metrics and loss
        """
        # This function will plot the model fit process
        print(self.history.history.keys())
        metric_list = {}

        # format metrics for plotting
        for key in self.history.history.keys():
            # check if key is
            sub_metric = key
            if "val_" in key:
                raw_metric = key.replace("val_", "")
            else:
                raw_metric = key

            if raw_metric in metric_list:
                metric_list[raw_metric].append(sub_metric)
            else:
                metric_list[raw_metric] = [raw_metric]

        for key in metric_list:
            plt.figure(figsize=(20, 10))
            plt.title(f"Model: {self.model_name} | Plot of {key}")
            for m in metric_list[key]:
                plt.plot(self.history.history[m])
            plt.ylabel(f"{key}")
            plt.xlabel("epoch")
            plt.legend(metric_list[key], loc="upper left")
            plt.show()

    def _PlotDistribution(self):
        """
        Plot all model Y and Y_pred to show distribution of data
        """
        # TODO: make better checks

        # [1] Y Distribution
        plt.figure(figsize=(20, 10))
        plt.title(f"Model: {self.model_name} | Y distribution")
        for observation in range(self.y_test.shape[0]):
            plt.plot(self.y_test[observation])
        plt.ylabel("kW")
        plt.xlabel("Hour")
        plt.show()

        # [2] Y_Pred Distribution
        plt.figure(figsize=(20, 10))
        plt.title(f"Model: {self.model_name} | Y_pred distribution")
        for observation in range(self.y_pred.shape[0]):
            plt.plot(self.y_pred[observation])
        plt.ylabel("kW")
        plt.xlabel("Hour")
        plt.show()

    def _PlotError(self):
        """
        Plots error plots (of several different types)
        """
        # TODO: make better checks

        plt.figure(figsize=(20, 10))
        plt.title(
            f"Model: {self.model_name} | Prediction mean_absolute_percentage_error"
        )
        y_mape_0 = CalcMAPE(self.y_test, self.y_pred, axis=0)
        y_mape_1 = CalcMAPE(self.y_test, self.y_pred, axis=1)
        plt.plot(y_mape_0)
        plt.plot(y_mape_1)
        plt.ylabel("Percent Error")
        plt.xlabel("Observation/Hour")
        plt.legend(["mape_by_hour", "mape_by_observation"], loc="upper left")
        plt.show()

        plt.figure(figsize=(20, 10))
        plt.title(f"Model: {self.model_name} | Prediction mean_absolute_error")
        y_mae_0 = CalcMAE(self.y_test, self.y_pred, axis=0)
        y_mae_1 = CalcMAE(self.y_test, self.y_pred, axis=1)
        plt.plot(y_mae_0)
        plt.plot(y_mae_1)
        plt.ylabel("kW")
        plt.xlabel("Observation/Hour")
        plt.legend(["mae_by_hour", "mae_by_observation"], loc="upper left")
        plt.show()

    def PlotRandomPlots(self, num_plot: int):
        """
        Randomly plot num_plot plots for current model and test data. If no model is loaded,
        no pots will be made

        Arguments:
            num_plot (int)  : number of random plots to make
        """
        # TODO: fix error checking
        assert self.model

        for _i in range(num_plot):
            plt.figure(figsize=(20, 10))
            x = np.random.default_rng().integers(0, self.y_pred.shape[0])
            plt.title(f"Model: {self.model_name} | Prediction, Observation #{x}")
            plt.plot(self.y_pred[x])
            plt.plot(self.y_test[x])
            plt.ylabel("kW")
            plt.xlabel("hour")
            plt.legend(["y prediction", "y actual"], loc="upper left")
            plt.show()
