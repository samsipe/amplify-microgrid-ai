import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence
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

class IModel:
    """
    Interface file for any model
    """

    def __init__(self, model_name="Unknown"):
        """
        Create an empty model
        """
        self.model = None
        self.model_name = model_name
        self.history = None

    def DisplayModel(self):
        """
        Display the model summary
        """
        if self.model:
            self.model.summary()
            tf.keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=False)

    def CreateModel(self):
        """
        Create the model. Insert model architecture here.
        """

    def TrainModel(self, x_train, y_train, x_val, y_val):
        """
        Train the model using x_train, y_train, x_val, y_val data.
        Fits the model and returns the fit history

        Arguments:
            x_train (ndarray/tensor)    : x training data
            y_train (ndarray/tensor)    : y training data
            x_val   (ndarray/tensor)    : x validation data
            y_val   (ndarray/tensor)    : y validation data

        Return:
            fit history
        """

    def Predict(self, x_test):
        """
        Runs and returns a prediction using the trained model and x_test input.

        Arguments:
            x_test (ndarray/tensor)     : x test data

        Return
            prediction output of model
        """
        if self.model:
            return self.model.predict(x_test)

### Test Models Below ###

class SimpleLSTM_1(IModel):
    """
    Simple LSTM Model.
    """

    def __init__(self, norm_layer, n_layer=1, n_series_len=48, n_series_ft=6, n_series_out=1, activation_f='tanh'):
        """ Initialize model """
        IModel.__init__(self, model_name="SimpleLSTM_1")
        self.n_layer = n_layer
        self.n_series_len = n_series_len
        self.n_series_ft = n_series_ft
        self.n_series_out = n_series_out
        self.activation_f = activation_f
        self.norm_layer = norm_layer

        # model callbacks
        self.reduce_lr = keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)

        self.SetHyperParam()
        self.CreateModel()

    def SetHyperParam(self, epoch=30, batch_size=10):
        """
        Set hyperparameters for training

        Arguments:
            epoch (int)         : training epoch
            batch_size (int)    : training batch size
        """
        self.epoch = epoch
        self.batch_size = batch_size

    def CreateModel(self):
        """
        Create the model
        """
        # norm_input = Input(shape=(self.n_series_len, self.n_series_ft))
        # seq_input = self.norm_layer(norm_input)
        model = Sequential()
        model.add(self.norm_layer)  # , input_shape=(self.n_series_len, self.n_series_ft))
        model.add(
            LSTM(self.n_layer, activation=self.activation_f, input_shape=(self.n_series_len, self.n_series_ft), return_sequences=True)
        )
        model.add(TimeDistributed(Dense(self.n_series_out)))
        self.model = model
        # norm_input = Input(shape=(self.n_series_len, self.n_series_ft))
        # lstm_input = self.norm_layer(norm_input)
        # lstm_layer = LSTM(self.n_layer, activation=self.activation_f, return_sequences=True)(lstm_input)
        # out = TimeDistributed(Dense(self.n_series_out))(lstm_layer)
        # self.model = Model(inputs=lstm_input, outputs=out)
        return self.model

    def TrainModel(self, x_train, y_train, x_val, y_val):
        """
        Train the model
        """
        if self.model is None:
            return self.history

        # Compile model
        self.model.compile(
            tf.optimizers.Adam(learning_rate=0.001), loss=keras.losses.Huber(), metrics=[keras.metrics.RootMeanSquaredError()]
        )

        # Fit Model
        self.history = self.model.fit(
            x=x_train,
            y=y_train,
            epochs=self.epoch,
            batch_size=self.batch_size,
            validation_data=(x_val, y_val),
            shuffle=False,
            callbacks=[self.reduce_lr],
        )

        return self.history
