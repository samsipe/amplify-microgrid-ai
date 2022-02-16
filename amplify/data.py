import json
import os
import sys
import warnings
from glob import glob

import numpy as np
import pandas as pd
import requests
from amplify.models import YeetLSTMv2
from clearml import Dataset, Model
from pysolar.radiation import get_radiation_direct
from pysolar.solar import get_altitude, get_azimuth
from tensorflow.keras.layers import Normalization


# TODO: replace prints with logging
class DataGenerator:
    """
    Generates data for training using weather and power data.
    """

    def __init__(
        self,
        weather_features: list = ["temp", "clouds_all"],
        building_features: list = ["True Power (kW)"],
        building_data_dir: str = None,
        weather_data_dir: str = None,
        cyclical_features: list = ["azimuth", "day_of_week"],
    ):
        """
        Initializes the data generator class.

        Arguments:
            weather_features (list)  : weather features to keep on import (optional)
            building_features (list) : building features to keep on import (optional)
            building_data_dir (str)  : location of local .csv of weather data (optional)
            weather_data_dir (str)   : location of local .csv of  building data (optional)
            cyclical_features (list) : columns to convert to sin/cos waveform (optional)
        """

        self.weather_features = weather_features
        self.building_features = building_features
        self.building_data_dir = building_data_dir
        self.weather_data_dir = weather_data_dir
        self.cyclical_features = cyclical_features

        # Data
        self.building_data: pd.DataFrame = None  # Building data
        self.weather_data: pd.DataFrame = None  # Weather data

        # Check for specified/local building and weather directories. If not specified, use ClearML
        if not self.building_data_dir:
            try:
                self.building_data_dir = glob(
                    Dataset.get(
                        dataset_project="amplify", dataset_name="building_data"
                    ).get_local_copy()
                    + "/**"
                )[0]
            except:
                print(
                    "No directory was specified and building data could not be retrieved from ClearML"
                )
                sys.exit(1)

        if not self.weather_data_dir:
            try:
                self.weather_data_dir = glob(
                    Dataset.get(
                        dataset_project="amplify", dataset_name="weather_data"
                    ).get_local_copy()
                    + "/**"
                )[0]
            except:
                print(
                    "No directory was specified and weather data could not be retrieved from ClearML"
                )
                sys.exit(1)

    def load_data(self):
        """
        Loads data for a specified building/weather data paths.

        Return:
            output_df (dataframe)    : weather, solar, and building data in one dataframe
        """

        # Now actually load the data. 1st check that directories are set
        if (
            (self.weather_data_dir is not None)
            & (self.building_data_dir is not None)
            & (self.weather_features is not None)
            & (self.building_features is not None)
        ):
            # With data directories set, download building data
            self.building_data = self._load_building_data(
                building_features=self.building_features
            )
            self.weather_data = self._load_weather_data()

            # return the retrieved/cleaned/merged data
            return self._additional_features()

    def _load_building_data(self, building_features: list = ["True Power (kW)"]):
        """
        Load and format building data from file.

        Return:
            building_data (dataframe)
        """

        self.building_data = pd.read_csv(
            self.building_data_dir, header=None, low_memory=False
        )

        # Forward fill the header name for each PowerScout
        self.building_data.iloc[0] = self.building_data.T[0].fillna(method="ffill")

        # Rename the 'nan' block
        self.building_data.loc[0, 0] = "Timestamp"

        # Create the multi-index
        self.building_data.columns = [
            list(self.building_data.iloc[0]),
            list(self.building_data.iloc[1]),
        ]

        # Drop the first two rows because they're just the column names, and any column with only nulls
        self.building_data = self.building_data[2:]

        # Convert timestamp column to datetime format
        self.building_data.Timestamp = pd.to_datetime(
            self.building_data.Timestamp.Timestamp,
            infer_datetime_format=True,
        )

        # Set Timestamp column as index, set columns to type 'float', rename index
        self.building_data = (
            self.building_data.set_index([("Timestamp", "Timestamp")])
            .replace("-", np.nan)
            .astype(float)
        )
        self.building_data.index.rename("Timestamp", inplace=True)

        # Set building_data to Eastern timezone and then convert to UTC
        self.building_data = self.building_data.tz_localize(
            "America/New_York", ambiguous=True
        ).tz_convert("UTC")

        # deduplicate index
        self.building_data = self.building_data.drop_duplicates(keep="last")

        # Drop any column or row with all nulls
        self.building_data = self.building_data.dropna(how="all", axis=1).dropna(
            how="all", axis=0
        )

        # remove noise (zeros) from the building_data
        self.building_data = self.building_data.replace(0, np.nan).fillna(
            method="ffill"
        )

        # Slice to the two power systems we're monitoring and rename columns
        # TODO: make the power systems vairable at some point
        self.building_data = self.building_data[
            ["PowerScout DPS126", "PowerScout DPS121"]
        ].rename(columns={"PowerScout DPS126": "solar", "PowerScout DPS121": "usage"})

        # Create our separate y-columns: solar pwr generated, buildling pwr used
        idx = pd.IndexSlice

        # Create DF with only Energy - keep just the last value (meter readings)
        self.building_data = self.building_data.loc[
            idx[:], idx[:, self.building_features]
        ]

        print("Info: Successfully loaded Building data!")

        return self.building_data

    def _load_weather_data(self):
        """
        Load, format, and calculate weather data from file.

        Return:
            weather_data (dataframe)
        """
        warnings.filterwarnings("ignore")

        self.weather_data = pd.read_csv(
            self.weather_data_dir, header=0, low_memory=False
        )

        #### pull lat/lon from weather data for irradiance
        self.building_lat = self.weather_data.lat.iloc[0]
        self.building_lon = self.weather_data.lon.iloc[0]

        ## Clean up datetime, drop 2nd datetime column
        # Convert from POSIX to ISO UTC time
        self.weather_data.dt = pd.to_datetime(
            self.weather_data.dt, unit="s", utc=True, infer_datetime_format=True
        )

        # Set date as index
        self.weather_data = self.weather_data.set_index("dt")

        # Drop 2nd datetime column that's not needed
        self.weather_data = self.weather_data[self.weather_features]

        # Deduplicate index
        self.weather_data = self.weather_data.drop_duplicates()

        # Create columns to capture data to be added after merge
        self.weather_data[["azimuth", "irradiance", "day_of_week"]] = np.nan

        # Create columns for capturing cyclical feature conversions
        for feature in self.cyclical_features:
            self.feat_sin, self.feat_cos = feature + "_sin", feature + "_cos"
            self.weather_data[[self.feat_sin, self.feat_cos]] = np.nan

        # Fill nulls in irradiance and azimuth with 0
        self.weather_data = self.weather_data.replace(np.nan, 0)

        print("Info: Successfully loaded Weather data!")
        return self.weather_data

    def _merge_building_weather(self):
        """
        With loaded weather data and building data,
        match indexes and adjust for daylight savings.

        Return:
            merged_data (dataframe)    : weather and building data in one dataframe
        """

        self.building_data = self._load_building_data()
        self.weather_data = self._load_weather_data()

        # Merge Building Solar Generation Y data to merged_data
        self.merged_data = (
            self.weather_data.merge(
                self.building_data.solar,
                "outer",
                left_index=True,
                right_index=True,
            )
            .fillna(method="ffill")
            .dropna()
        )
        for feature in self.building_features:
            self.merged_data.rename(
                columns={feature: str(feature) + " solar"}, inplace=True
            )

        # Merge Building Usage Y data to merged_data
        self.merged_data = self.merged_data.merge(
            self.building_data.usage,
            "outer",
            left_index=True,
            right_index=True,
        ).fillna(method="ffill")

        for feature in self.building_features:
            self.merged_data.rename(
                columns={feature: str(feature) + " usage"}, inplace=True
            )

        print("Info: Successfully merged Building and Weather data!")
        return self.merged_data, self.building_lat, self.building_lon

    def _additional_features(self):
        """
        Adds irradiance and azimuth features to the merged weather and building dataset

        Return:
            weather, solar, and building data in one dataframe
        """
        (
            self.output_df,
            self.building_lat,
            self.building_lon,
        ) = self._merge_building_weather()

        print("Info: Calculating Azimuth and Irradiance data, this may take awhile...")
        # Add Solar Irradiance
        self.date_list = list(self.output_df.index)
        for date in self.date_list:
            self.pydate = date.to_pydatetime()
            self.date = date.tz_localize(None)

            # Calculate Solar Azimuth
            self.output_df.loc[self.date, "azimuth"] = get_azimuth(
                self.building_lat, self.building_lon, self.pydate
            )

            # Calculate Solar Altitude
            self.altitude_deg = get_altitude(
                self.building_lat, self.building_lon, self.pydate
            )

            # Calculate Solar Irradiance
            self.output_df.loc[self.date, "irradiance"] = get_radiation_direct(
                self.pydate, self.altitude_deg
            )

        self.output_df = self.output_df.replace(np.nan, 0)

        print("Info: Successfully added Azimuth and Irradiance data!")
        print("Info: Converting Cyclical Features.")

        # Add day of week to Weather Data (+1 is for correct spread in next step)
        self.output_df.day_of_week = self.output_df.index.strftime("%w").astype(int) + 1

        # Add cyclical functions for columns identified in 'cyclical_features' argument
        for feature in self.cyclical_features:
            # Set column names to capture new values
            self.feat_sin, self.feat_cos = feature + "_sin", feature + "_cos"

            # Calculate spread of values in column
            self.feat_spread = (
                int(self.output_df[feature].max()) - int(self.output_df[feature].min())
            ) + 1

            # Convert to sinusoidal and cosinal values
            self.output_df[self.feat_sin] = np.sin(
                2 * np.pi * self.output_df[feature].astype(float) / self.feat_spread
            )
            self.output_df[self.feat_cos] = np.cos(
                2 * np.pi * self.output_df[feature].astype(float) / self.feat_spread
            )

            # Clean up dataframe by dropping original feature column
            self.output_df.drop([feature], axis=1, inplace=True)

        print("Info: Successfully converted cyclical features! Data is ready!")

        return self.output_df.round(2)


class DataSplit:
    """
    Splits data into series and then randomly splits those series
    into training, validation, and training sets.

    Return:
        split_data (tuple): a tuple of tuples containing ((train_x, train_y), (val_x, val_y), (test_x, test_y))
    """

    def __init__(
        self,
        dataframe,
        sequence: bool = True,
        train_pct: float = 0.8,
        val_pct: float = 0.1,
        test_pct: float = 0.1,
        series_length: int = 48,
        stride: int = 3,
    ):
        """
        Initializes the data splitting class.

        Arguments:
            dataframe           : a well formatted dataframe with features and two dependent variables as columns
            train_pct (float)   : amount of the dataset used in training (optional)
            val_pct (float)     : amount of the dataset used in validation (optional)
            test_pct (float)    : amount of the dataset used in testing(optional)
            shuffle (bool)      : wheather or not to randomly shuffle the series before splitting (optional)
            series_length (int) : how many observations are in a series (optional)
            stride (int)        : how many observations to stride over when building series (optional)
        """

        self.dataframe = dataframe
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.series_length = series_length
        self.stride = stride
        self.sequence = sequence

        assert (self.train_pct + self.test_pct + self.val_pct) == 1

    def _make_series(self, input_df, seq):
        """
        Create 3D array of slices based on series_length and stride

        Argument:
            input_df (df)      : a well formatted dataframe with features and two dependent variables as columns
            seq (bool)         : a boolean KWARG passed from the class instantiation, default = True

        Return:
            a 3D numpy array with each slice containing a series length of hours worth of data
        """

        ### TODO add random sample to stride between 1 and 5
        self.data = input_df.copy()
        self.start_index = self.series_length
        self.end_index = len(self.data)
        self.seq = seq

        self.output = []

        ### Check for sequence bool KWARG
        if self.seq:
            # Iterate through dataframe at stride length, creating slice of series_length
            for i in range(
                self.start_index,
                self.end_index,
                self.stride,
            ):
                self.indices = range(i - self.series_length, i, 1)
                self.output.append(self.data.iloc[self.indices])

        else:
            self.output = self.data

        # Return as array
        return np.array(self.output)

    def _train_val_test_split(self, input_df):
        """
        Splits dataset into train, val, test sets with no overlap

        Argument:
            input_df           : a well formatted dataframe with features and two dependent variables as columns

        Return:
            a tuple of dataframes: (train, val, test) of length determined by train_pct, val_pct, and test_pct
        """
        self.dataframe = input_df.copy()

        # Set indices to split on default or specified percents
        self.train_ind = int(self.dataframe.shape[0] * self.train_pct)
        self.test_ind = -int(self.dataframe.shape[0] * self.test_pct)

        # Split dataset into 3 parts using indices
        self.train_split = self.dataframe.iloc[: self.train_ind]
        self.val_split = self.dataframe.iloc[self.train_ind : self.test_ind]
        self.test_split = self.dataframe.iloc[self.test_ind :]

        return self.train_split, self.val_split, self.test_split

    def _xy_splits(self, input_df, seq):
        """
        Separates the sets of x matrixes and y column vectors

        Argument:
            input_df           : 3D numpy array with two trailing y columns

        Return:
            a tuple of numpy arrays: (x, y)
        """

        self.dataset = input_df.copy()
        self.seq = seq

        ## Remove last columns to make y vectors for the dataset

        if self.seq:
            return (
                self.dataset[:, :, :-2].astype("float32"),
                self.dataset[:, :, -2:].astype("float32"),
            )
        else:
            return (
                self.dataset[:, :-2].astype("float32"),
                self.dataset[:, -2:].astype("float32"),
            )

    def split_data(self):
        """
        Splits dataset into a tuple of tuples - (x_vals, y_vals)

        Return:
            A tuple of tuples - (train_x, train_y), (val_x, val_y), (test_x, test_y), (norm_layer)
        """
        # Run the dataframe through the splitter to create train, val, and test DFs
        self.pre_split = self._train_val_test_split(self.dataframe)

        # Drop Y's and convert to float32 to prepare the training data for normalization
        self.training_pre_split = self.pre_split[0].iloc[:, :-2].astype("float32")

        # Normalize training data
        self.norm_layer = Normalization(axis=-1)
        self.norm_layer.adapt(self.training_pre_split)

        # Convert each df to a sequence of length "series_length"
        self.seq = self.sequence
        self.output_list = ["train", "val", "test"]
        for i, df in enumerate(self.pre_split):
            self.output_list[i] = self._make_series(df, self.seq)

        ## split x and y columns from train, val, and test sequenced datasets
        self.train_split = self._xy_splits(self.output_list[0], self.seq)
        self.val_split = self._xy_splits(self.output_list[1], self.seq)
        self.test_split = self._xy_splits(self.output_list[2], self.seq)

        # Return a tuple of tuples
        print(
            "Info: Successfully split data into (train_x, train_y), (val_x, val_y), (test_x, test_y), (norm_layer)!"
        )
        return (
            self.train_split,
            self.val_split,
            self.test_split,
            self.norm_layer,
        )

        # train_split[0] -> features
        # train_split[1] -> solar
        # train_split[2] -> usage


class PredictData:
    """
                                        1) Takes in weather prediction data (API connection)
                                        2) Cleans raw API -> JSON data
                                        3) Adds day_of_week, irradiance, and azimuth
                                        4) Outputs clean features of weather prediction + pysolar for 48hrs
                                        5) a) Split datetime index to a separate df,
                                           b) Run model.predict(forecast),
                                           c) Combine datetime df with predict output
                                        8) ???
                                        9) Profit
    """

    def __init__(
        self,
        model,
        lat: float = 39.9649,
        lon: float = -75.1396,
        num_cars: int = 1,
        hrs_to_charge: int = 3,
        kw_to_charge: int = 7,
        features: list = ["dt", "temp", "clouds"],
        cyclical_features: list = ["azimuth", "day_of_week"],
        ow_api_key: str = os.environ.get("OW_API_KEY"),
    ):
        """
        Initializes the PredictData class.

        Arguments:
            model (array)            : model factors to apply during prediction (required)
            lat (float)              : latitude of location for weather forecast (optional)
            lon (float)              : longitude of location for weather forecast (optional)
            num_cars (int)           : how many cars will be charged (optional)
            hrs_to_charge (int)      : how many hours each car needs to be charged for (optional)
            kw_to_charge (int)       : how many kW each car will draw per hour (optional)
            features (list)          : list of weather features to pull from API (optional)
            cyclical_features (list) : columns to convert to sin/cos waveform (optional)
            ow_api_key (str)         : string key for Open Weather API (optional)
        """
        self.lat, self.lon = lat, lon
        self.features = features
        self.cyclical_features = cyclical_features
        self.ow_api_key = ow_api_key
        self.model = model
        self.num_cars = num_cars
        self.hrs_to_charge = hrs_to_charge
        self.kw_to_charge = kw_to_charge

    def forecast(self):
        """
        This is the main callable function. It runs the other functions
        in order, passing the results of one to the next, to return properly
        prepared weather forecast data.

        All arguments come from the Class instantiation.

        Returns:
            pred_out (dataframe)    : y-prediction dataframe indexed to datetime
        """

        # Run method for API call to retrive data and convert JSON -> dataframe
        self.raw_weather = self._get_forecast(
            ow_api_key=self.ow_api_key,
            lat=self.lat,
            lon=self.lon,
            features=self.features,
        )

        # Run method to clean weather data retrieved from API call
        self.clean_weather = self._forecast_clean(
            raw_forecast=self.raw_weather, cyclical_features=self.cyclical_features
        )

        # Run method to add azimuth, irradiance, and day of week
        self.all_features = self._additional_features(
            clean_weather=self.clean_weather,
            cyclical_features=self.cyclical_features,
            lat=self.lat,
            lon=self.lon,
        )

        # Run prediction
        self.pred_out = self._run_predict(model=self.model, features=self.all_features)

        print("Info: Usage and Generation predictions complete!")

        self.charting_df = self._charging_calcs(
            preds_df=self.pred_out,
            num_cars=self.num_cars,
            hrs_to_charge=self.hrs_to_charge,
            kw_to_charge=self.kw_to_charge,
        )

        print("Info: Costing predictions complete!")
        return self.charting_df

    def _get_forecast(self, ow_api_key: str, lat: float, lon: float, features: str):
        """
        Retrieves 48 hrs of hourly weather forecast based on latitude, longitude, and features

        Arguments:
            ow_api_key (str)         : a string of the Open Weather API key (required)
            lat (float)              : latitude of location for weather forecast (optional)
            lon (float)              : longitude of location for weather forecast (optional)
            features (list)          : list of weather features to pull from API (optional)

        Returns:
            weather_data (dataframe) : a dataframe of raw weather data from the API pull
        """
        self.lat, self.lon = lat, lon
        self.ow_api_key = ow_api_key
        self.features = features

        # Set URL for API
        self.url = (
            "https://api.openweathermap.org/data/2.5/onecall?lat="
            + str(self.lat)
            + "&lon="
            + str(self.lon)
            + "&units=metric&exclude=current,minutely,daily,alerts&appid="
            + self.ow_api_key
        )

        # Pull data into JSON format
        self.data = requests.get(self.url).json()

        # Convert data into dataframe, pulling just the 3 columns we need
        self.weather_data = pd.json_normalize(self.data["hourly"])[self.features]

        print("Info: Successfully retrieved forecast data")
        return self.weather_data

    def _forecast_clean(self, raw_forecast, cyclical_features: list):
        """
        Cleans weather forecast and creates columns to collect solar features and day of week

        Arguments:
            raw_forecast (dataframe) : a dataframe of weather data (required)
            cyclical_features (list) : list of columns to convert to sin/cos waveform (optional)

        Returns:
            weather_data (dataframe) : a dataframe of clean weather data w/null columns for cyclical features
        """
        self.weather_data = raw_forecast.copy()
        self.cyclical_features = cyclical_features

        # Convert from POSIX to ISO UTC time
        self.weather_data.dt = pd.to_datetime(
            self.weather_data.dt, unit="s", utc=True, infer_datetime_format=True
        )

        # Set date as index
        self.weather_data.set_index("dt", inplace=True)

        # Create columns to capture data to be added after merge
        self.weather_data[["azimuth", "irradiance", "day_of_week"]] = np.nan

        # Create columns for capturing cyclical feature conversions
        for feature in self.cyclical_features:
            self.feat_sin, self.feat_cos = feature + "_sin", feature + "_cos"
            self.weather_data[[self.feat_sin, self.feat_cos]] = np.nan

        # Fill nulls in irradiance and azimuth with 0
        self.weather_data.replace(np.nan, 0, inplace=True)

        print("Info: Successfully cleaned forecast data!")
        return self.weather_data

    def _additional_features(
        self, clean_weather, cyclical_features: list, lat: float, lon: float
    ):
        """
        1) Cleans weather forecast
        2) Adds solar data
        3) Converts cyclical features to sin/cos

        Arguments:
            raw_forecast (dataframe) : a dataframe of weather data (required)
            cyclical_features (list) : list of columns to convert to sin/cos waveform (optional)
            lat (float)              : latitude of location for weather forecast (optional)
            lon (float)              : longitude of location for weather forecast (optional)

        Returns:
            output_df (dataframe)    : a dataframe of clean weather, day of week, and solar data
        """
        self.output_df = clean_weather.copy()
        self.lat = lat
        self.lon = lon
        self.cyclical_features = cyclical_features

        # Create date list from index for calculating pysolar data
        self.date_list = list(self.output_df.index)

        for date in self.date_list:
            # Set date datatypes to match for pysolar calls
            self.pydate = date.to_pydatetime()
            self.date = date

            # Calculate Solar Azimuth
            self.output_df.loc[self.date, "azimuth"] = get_azimuth(
                self.lat, self.lon, self.pydate
            )

            # Calculate Solar Altitude
            self.altitude_deg = get_altitude(self.lat, self.lon, self.pydate)

            # Calculate Solar Irradiance
            self.output_df.loc[self.date, "irradiance"] = get_radiation_direct(
                self.pydate, self.altitude_deg
            )

        # Replace NaNs with 0
        self.output_df.replace(np.nan, 0, inplace=True)

        print("Info: Successfully added forecast Azimuth and Irradiance data!")
        print("Info: Converting Cyclical Forecast Features.")

        # Add day of week to Weather Data (+1 is for correct spread in next step)
        self.output_df.day_of_week = self.output_df.index.strftime("%w").astype(int) + 1

        # Add cyclical functions for columns identified in 'cyclical_features' argument
        for feature in self.cyclical_features:
            # Set column names to capture new values
            self.feat_sin, self.feat_cos = feature + "_sin", feature + "_cos"

            # Calculate spread of values in column
            self.feat_spread = (
                int(self.output_df[feature].max()) - int(self.output_df[feature].min())
            ) + 1

            # Convert to sinusoidal and cosinal values
            self.output_df[self.feat_sin] = np.sin(
                2 * np.pi * self.output_df[feature].astype(float) / self.feat_spread
            )
            self.output_df[self.feat_cos] = np.cos(
                2 * np.pi * self.output_df[feature].astype(float) / self.feat_spread
            )

            # Clean up dataframe by dropping original feature column
            self.output_df.drop([feature], axis=1, inplace=True)

        print("Info: Successfully converted cyclical features! Data is ready!")
        return self.output_df.round(2)

    def _run_predict(self, model, features):
        """
        1) Extracts datetime index from weather features.
        2) Runs model.predict on weather forecast.
        3) Marries output predicted y-values with datetime indexing

        Arguments:
            model (object)           : an object created from a model.fit method (required)
            features (dataframe)     : dataframe of forecasted weather features (required)

        Returns:
            pred_df (dataframe)      : a dataframe of y-prediction data indexed to datetime
        """
        self.prep_feats = features.copy()
        self.model = model

        # Reset index to allow saving datetime index
        self.prep_feats.reset_index(inplace=True)

        # Create new df with datetime column
        self.preds_df = pd.DataFrame(self.prep_feats.dt)

        # Drop dt off features to isolate to prediction features
        self.prep_feats.drop("dt", axis=1, inplace=True)

        # Convert to Numpy array of shape (1, 48, 7)
        self.pred_array = np.reshape(np.array(self.prep_feats), (1, 48, 7))

        # Perform prediction on forecast array
        self.y_preds = self.model.predict(self.pred_array)

        # Merge prediction output array into df.
        self.preds_df = self.preds_df.merge(
            pd.DataFrame(self.y_preds[0]), "inner", left_index=True, right_index=True
        )

        # Rename columns
        self.preds_df.rename(
            columns={0: "Predicted Solar", 1: "Predicted Usage"}, inplace=True
        )

        # Set index on column dt
        self.preds_df.set_index("dt", inplace=True)

        return self.preds_df

    def _charging_calcs(self, preds_df, num_cars, hrs_to_charge, kw_to_charge):
        """
        1) Calculates net power.
        2) Assigns and calculates billing rates/charges.
        3) Adds and calculates new usage and billing with charging
        4) Identifies lowest cost charging windows based on hrs_to_charge

        Arguments:
            preds_df (object)        : an object created from a model.fit method (required)
            num_cars (int)           : num of cars to be charged (optional)
            hrs_to_charge (int)      : how many hours each car will be charging
            kw_to_charge (int)       : how many kWh are used for charging

        Returns:
            pred_df (dataframe)      : a dataframe of usage/solar, billing, and charging windows
        """
        # Set variables for charging cost predictions
        self.df = preds_df.copy()
        self.num_cars = num_cars
        self.hrs_to_charge = hrs_to_charge
        self.kw_to_charge = kw_to_charge

        # Convert index time to Eastern TZ
        self.df.index = self.df.index.tz_convert("US/Eastern")

        # Create column of power/usage difference
        self.df["Predicted Net"] = (
            self.df["Predicted Usage"] - self.df["Predicted Solar"]
        )

        # Create day of week and hour columns for applying rates
        self.df["dow"] = self.df.index.strftime("%w").astype(int)
        self.df["hr"] = self.df.index.strftime("%H").astype(int)

        ## Set rates
        # Set off=peak price
        self.df["Rate"] = 0.04801

        # Set super-off-peak
        self.df.loc[(self.df.hr >= 0) & (self.df.hr < 6), "Rate"] = 0.02824

        # Set Peak rates
        self.df.loc[
            ((self.df.hr >= 14) & (self.df.hr < 18))
            & ((self.df.dow > 0) & (self.df.dow < 6)),
            "Rate",
        ] = 0.14402

        # Drop unneeded columns
        self.df.drop(["dow", "hr"], axis=1, inplace=True)

        # Compute estimated cost for the hour
        self.df["Predicted Net Cost"] = self.df["Predicted Net"] * self.df.Rate

        # Compute average cast of energy for the 48hr period
        self.df["Average Predicted Cost"] = self.df["Predicted Net Cost"].sum() / len(
            self.df
        )

        # Calculate power usage including variables
        self.df["Predicted Usage While Charging"] = self.df["Predicted Net"] + (
            self.kw_to_charge * self.num_cars
        )

        # Calculate charging cost given variables
        self.df["Predicted Cost While Charging"] = (
            self.df["Predicted Usage While Charging"] * self.df["Predicted Net Cost"]
        )

        # Roll through the number of hours needed for charging
        self.df["window"] = (
            self.df["Predicted Cost While Charging"].rolling(self.hrs_to_charge).sum()
        )

        # Select optimum hours for charging

        # Create empty df to collect best periods of charging
        self.charge = pd.DataFrame()

        # Loop through date stamps for best hours to charge
        for date in pd.DataFrame(self.df.window.nsmallest(5)).index:
            # Append times to the collection df based on window selection
            self.charge = self.charge.append(
                self.df.loc[date + pd.DateOffset(hours=-self.hrs_to_charge) : date]
            )

        # Set charging go/no-go column to 1
        self.charge["Optimal Charging"] = 1

        # Merge go/no-go column back to main costing df
        self.df = self.df.merge(
            self.charge["Optimal Charging"], "left", left_index=True, right_index=True
        )

        # Replace nulls in Go/No-Go with 0
        self.df["Optimal Charging"].fillna(0, inplace=True)

        # Convert Go/No-Go to integer
        self.df["Optimal Charging"] = self.df["Optimal Charging"].astype(int)

        return self.df