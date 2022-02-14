import os

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
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import Normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.layers import Bidirectional
from keras.layers import TimeDistributed

from datetime import datetime


class IModel:
    """
    Interface file for any model
    """

    imodel_id = 0

    def __init__(
        self,
        model_name="unknown",
        model_id=None,
        log_dir=os.path.join('../', 'logs'),
        model_dir=os.path.join('../', 'models'),
        norm_layer=Normalization(),
        lr_factor=0.9,
        lr_patience=3,
        es_patience=10,
        l_rate=0.0005,
        min_l_rate=0.0001,
        dropout=0.25,
        batch=1,
        epoch=50,
    ):
        """
        Create an empty model
        """
        self.model = None
        self.model_name = model_name
        self.history = None

        # weight save name
        if model_id:
            self.model_id = model_id
        else:
            self.model_id = IModel.imodel_id
            IModel.imodel_id += 1

        # current time for saving models
        dt_string = datetime.now().strftime("%d-%m-%Y%_H:%M:%S")  # [dd/mm/YY H:M:S]

        # files
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.model_weights_file_name = self.model_name + '_weights_' + dt_string + '.hdf5'
        self.model_weights_file_path = os.path.abspath(os.path.join(model_dir, self.model_weights_file_name))

        # model param
        self.norm_layer = norm_layer

        # hyper parameters
        self.set_hyper_param(
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            es_patience=es_patience,
            l_rate=l_rate,
            min_l_rate=min_l_rate,
            dropout=dropout,
            batch=batch,
            epoch=epoch,
        )

        # callabacks
        self.tensor_board_cb: tf.keras.callback.Callback = None
        self.model_checkpoint_cb: tf.keras.callback.Callback = None
        self.reduce_lr_on_plateau_cb: tf.keras.callbacks.Callback = None
        self.early_stopping_cb: tf.keras.callbacks.Callback = None

        # metrics
        self.metrics = [keras.metrics.RootMeanSquaredError(), keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]

    def set_hyper_param(
        self,
        lr_factor: float = 0.9,
        lr_patience: int = 3,
        es_patience: int = 10,
        l_rate: float = 0.0005,
        min_l_rate: float = 0.0001,
        dropout: float = 0.25,
        batch: int = 1,
        epoch: int = 50,
    ):
        """
        Set hyper parameters

        Arguments:
            lr_factor (int)     : learning rate factor
            lr_patience (int)   : learning rate patience
            es_patience (int)   : early stopping patience
            l_rate (float)      : learning rate
            min_l_rate (float)  : minimum learning rate
            dropout (float)     : dropout
            batch (int)         : training batch size
            epoch (int)         : training epoch number
        """
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.es_patience = es_patience
        self.l_rate = l_rate
        self.min_l_rate = min_l_rate
        self.dropout = dropout
        self.batch = batch
        self.epoch = epoch

    def setup_callbacks(self):
        """
        Setup callbacks
        """
        self.tensor_board_cb = tf.keras.callbacks.TensorBoard(self.log_dir)
        self.model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            self.model_weights_file_path, monitor="val_loss", mode='min', save_best_only=True, save_weights_only=True, verbose=1
        )
        self.reduce_lr_on_plateau_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', mode='min', factor=self.lr_factor, patience=self.lr_patience, min_lr=self.min_l_rate, verbose=0
        )
        self.early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode='min', patience=self.es_patience, verbose=1)

    def display_model(self):
        """
        Display the model summary
        """
        if self.model:
            self.model.summary()
            keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=False)

    def create_model(self):
        """
        Create the model. Insert model architecture here.
        """

    def train_model(self, x_train, y_train, x_val, y_val):
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

    def load_weights(self, model_weights_path: str = None):
        """
        load pretrained weights into the model

        Arguments:
            model_weights_path (str)    : file path for model weights
        """
        if not (model_weights_path is None or self.model is None):
            try:
                self.model.load_weights(model_weights_path)
            except:
                print("Error: Cannot load model weights!")

    def predict(self, x_test):
        """
        Runs and returns a prediction using the trained model and x_test input.ß

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

        self.set_hyper_param()
        self.create_model()

    def set_hyper_param(self, epoch=30, batch_size=10):
        """
        Set hyperparameters for training

        Arguments:
            epoch (int)         : training epoch
            batch_size (int)    : training batch size
        """
        self.epoch = epoch
        self.batch_size = batch_size

    def create_model(self):
        """
        Create the model
        """
        # norm_input = Input(shape=(self.n_series_len, self.n_series_ft))
        # seq_input = self.norm_layer(norm_input)
        model = Sequential()
        model.add(Input(shape=(self.n_series_len, self.n_series_ft)))
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

    def train_model(self, x_train, y_train, x_val, y_val):
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


class MultiLayerLSTM(IModel):
    """
    MultiLayer LSTM Model.
    """

    def __init__(self, norm_layer, n_layer=1, n_series_len=48, n_series_ft=6, n_series_out=1, activation_f='tanh', dropout_rate=0.25):
        """ Initialize model """
        IModel.__init__(self, model_name="MultiLayerLSTM")
        self.n_layer = n_layer
        self.n_series_len = n_series_len
        self.n_series_ft = n_series_ft
        self.n_series_out = n_series_out
        self.activation_f = activation_f
        self.norm_layer = norm_layer
        self.dropout_rate = dropout_rate

        # hyper param
        FACTOR = 0.1
        PATIENCE = 5

        # model callbacks
        self.model_cp = tf.keras.callbacks.ModelCheckpoint(
            "../models/multi_layer_lstm_weights.hdf5",
            monitor="val_loss",
            mode='min',
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
        self.reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            mode='min',
            factor=FACTOR,
            patience=PATIENCE,
            min_lr=1e-8,
            verbose=1,
        )
        self.early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode='min', patience=(3 * PATIENCE), verbose=1)
        self.tensorboard_logs = tf.keras.callbacks.TensorBoard('../logs/', histogram_freq=1)
        self.callbacks = [self.model_cp, self.reduce_lr, self.early_stop, self.tensorboard_logs]

        self.set_hyper_param()
        self.create_model()

    def set_hyper_param(self, epoch=30, batch_size=10):
        """
        Set hyperparameters for training

        Arguments:
            epoch (int)         : training epoch
            batch_size (int)    : training batch size
        """
        self.epoch = epoch
        self.batch_size = batch_size

    def create_model(self):
        """
        Create the model
        """
        # norm_input = Input(shape=(self.n_series_len, self.n_series_ft))
        # seq_input = self.norm_layer(norm_input)
        model = Sequential()
        model.add(Input(shape=(self.n_series_len, self.n_series_ft)))
        model.add(self.norm_layer)  # , input_shape=(self.n_series_len, self.n_series_ft))
        model.add(Dropout(rate=self.dropout_rate))
        model.add(Dense(self.n_series_ft, activation='relu'))
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

    def train_model(self, x_train, y_train, x_val, y_val):
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
            callbacks=self.callbacks,
        )

        return self.history

    def predict(self, x_test):
        """
        Runs and returns a prediction using the trained model and x_test input.

        Arguments:
            x_test (ndarray/tensor)     : x test data

        Return
            prediction output of model
        """
        if self.model:
            self.model.load_weights("../models/multi_layer_lstm_weights.hdf5")
            return self.model.predict(x_test, verbose=1, batch_size=1, callbacks=self.callbacks)


class YeetLSTMv1(IModel):
    """
    MultiLayer LSTM Model.
    """

    def __init__(
        self,
        norm_layer: Normalization = Normalization(mean=0.0, variance=1.0),
        n_series_len=48,
        n_series_ft=6,
        n_series_out=1,
        n_lstm=None,
        activation_fn='relu',
        lr_factor=0.9,
        lr_patience=None,
        es_patience=None,
        l_rate=0.0005,
        min_l_rate=0.0001,
        dropout=0.25,
        batch=1,
        epoch=50,
    ):
        """ Initialize model """
        IModel.__init__(
            self,
            model_name="yeet_lstm_v1",
            norm_layer=norm_layer,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            es_patience=es_patience,
            l_rate=l_rate,
            min_l_rate=min_l_rate,
            dropout=dropout,
            batch=batch,
            epoch=epoch,
        )

        # CONSTANTS
        lstm_c = 10
        patience_c = 10

        # model param
        self.n_series_len = n_series_len
        self.n_series_ft = n_series_ft
        self.n_series_out = n_series_out
        self.activation_fn = activation_fn
        self.n_lstm_layers = n_lstm if n_lstm else self.n_series_len * lstm_c

        # hyper param
        # *NOTE: IModel has factor, patience, l_rate, dropout, batch, epoch
        self.es_patience = self.es_patience if self.es_patience else patience_c * self.batch

        # callbacks
        self.setup_callbacks()

        self.callbacks = [self.tensor_board_cb, self.model_checkpoint_cb, self.early_stopping_cb, self.reduce_lr_on_plateau_cb]

        self.create_model()
        self.display_model()

    def create_model(self):
        """
        Create the model
        """
        norm_inputs = Input(shape=(self.n_series_len, self.n_series_ft))
        nn_inputs = self.norm_layer(norm_inputs)
        nn_layer = Dense(self.n_lstm_layers)
        encoder_inputs = nn_layer(nn_inputs)
        encoder_l1 = LSTM(self.n_lstm_layers, return_state=True, dropout=self.dropout)
        encoder_outputs1 = encoder_l1(encoder_inputs)

        encoder_states1 = encoder_outputs1[1:]

        decoder_inputs = RepeatVector(self.n_series_len)(encoder_outputs1[0])

        decoder_l1 = LSTM(self.n_lstm_layers, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
        decoder_outputs1 = TimeDistributed(Dense(self.n_series_out, activation=self.activation_fn))(decoder_l1)

        self.model = Model(norm_inputs, decoder_outputs1)
        return self.model

    def train_model(self, x_train, y_train, x_val, y_val):
        """
        Train the model
        """
        if self.model is None:
            return self.history

        # Compile model
        self.model.compile(tf.optimizers.Adam(learning_rate=self.l_rate), loss=keras.losses.MeanSquaredError(), metrics=self.metrics)

        # Fit Model
        self.history = self.model.fit(
            x=x_train,
            y=y_train,
            epochs=self.epoch,
            batch_size=self.batch,
            validation_data=(x_val, y_val),
            shuffle=False,
            callbacks=self.callbacks,
        )

        return self.history

    def predict(self, x_test):
        """
        Runs and returns a prediction using the trained model and x_test input.

        Arguments:
            x_test (ndarray/tensor)     : x test data

        Return
            prediction output of model
        """
        if self.model:
            self.model.load_weights(self.model_weights_file_path)
            return self.model.predict(x_test, verbose=1, batch_size=1, callbacks=self.callbacks)


class YeetLSTMv2(IModel):
    """
    MultiLayer LSTM Model.
    """

    def __init__(
        self,
        norm_layer: Normalization = Normalization(mean=0.0, variance=1.0),
        n_series_len=48,
        n_series_ft=6,
        n_series_out=1,
        n_lstm=None,
        activation_fn='relu',
        lr_factor=0.8,
        lr_patience=2,
        es_patience=25,
        l_rate=0.0005,
        min_l_rate=1e-8,
        dropout=0.20,
        batch=1,
        epoch=100,
        kernel_regularizer='l2',
        model_weights_path: str = None,
    ):
        """ Initialize model """
        IModel.__init__(
            self,
            model_name="yeet_lstm_v2",
            norm_layer=norm_layer,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            es_patience=es_patience,
            l_rate=l_rate,
            min_l_rate=min_l_rate,
            dropout=dropout,
            batch=batch,
            epoch=epoch,
        )

        # CONSTANTS
        lstm_c = 10
        patience_c = 10

        # model param
        self.n_series_len = n_series_len
        self.n_series_ft = n_series_ft
        self.n_series_out = n_series_out
        self.activation_fn = activation_fn
        self.n_lstm_layers = n_lstm if n_lstm else self.n_series_len * lstm_c
        self.kernel_regularizer = kernel_regularizer

        # hyper param
        # *NOTE: IModel has factor, patience, l_rate, dropout, batch, epoch
        self.es_patience = self.es_patience if self.es_patience else patience_c * self.batch

        # callbacks
        self.setup_callbacks()
        self.lr_scheduler_cb = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
        self.callbacks = [self.tensor_board_cb, self.model_checkpoint_cb, self.early_stopping_cb, self.lr_scheduler_cb]

        self.create_model()
        self.display_model()

        self.load_weights(model_weights_path)

    def create_model(self):
        """
        Create the model
        """
        norm_inputs = Input(shape=(self.n_series_len, self.n_series_ft))
        norm_layer = self.norm_layer(norm_inputs)
        lstm_layer = LSTM(self.n_lstm_layers, return_state=True, dropout=self.dropout)
        lstm_outputs = lstm_layer(norm_layer)
        td_outputs = TimeDistributed(Dense(self.n_series_out, activation=self.activation_fn, kernel_regularizer=self.kernel_regularizer))(
            lstm_outputs
        )

        self.model = Model(norm_inputs, td_outputs)
        return self.model

    def train_model(self, x_train, y_train, x_val, y_val):
        """
        Train the model
        """
        if self.model is None:
            return self.history

        # Compile model
        self.model.compile(tf.optimizers.Adam(learning_rate=self.l_rate), loss=keras.losses.MeanSquaredError(), metrics=self.metrics)

        # Fit Model
        self.history = self.model.fit(
            x=x_train,
            y=y_train,
            epochs=self.epoch,
            batch_size=self.batch,
            validation_data=(x_val, y_val),
            shuffle=True,
            callbacks=self.callbacks,
        )

        return self.history

    def predict(self, x_test):
        """
        Runs and returns a prediction using the trained model and x_test input.

        Arguments:
            x_test (ndarray/tensor)     : x test data

        Return
            prediction output of model
        """
        if self.model:
            self.model.load_weights(self.model_weights_file_path)
            return self.model.predict(x_test, verbose=1, batch_size=1, callbacks=self.callbacks)

    def scheduler(self, epoch):
        """
        Scheduler function.

        Arguments:
            epoch (int)     : Epoch
        """
        return self.l_rate / (epoch % 10 + 1)
