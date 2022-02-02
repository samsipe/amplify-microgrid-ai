import sys
import warnings
from glob import glob

import numpy as np
import pandas as pd
from clearml import Dataset
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
    ):
        """
        Initializes the data generator class.

        Arguments:
            weather_features (list)  : weather features to keep on import (optional)
            building_features (list) : building features to keep on import (optional)
            building_data_dir (str)  : location of local .csv of weather data (optional)
            weather_data_dir (str)   : location of local .csv of  building data (optional)
        """

        self.weather_features = weather_features
        self.building_features = building_features
        self.building_data_dir = building_data_dir
        self.weather_data_dir = weather_data_dir

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
            pysolar_features (dataframe)    : weather, solar, and building data in one dataframe
        """

        # Now actually load the data. 1st check that directories are set
        if (
            (self.weather_data_dir is not None)
            & (self.building_data_dir is not None)
            & (self.weather_features is not None)
            & (self.building_features is not None)
        ):
            # With data directories set, return the retrieved/cleaned/merged data
            # return self._merge_building_weather()
            return self._pysolar_features()

    def _load_building_data(self):
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

        # deduplicate index
        self.weather_data = self.weather_data.drop_duplicates()

        self.weather_data[["azimuth", "irradiance", "day_of_week"]] = np.nan

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

        # Add day of week to Weather Data
        self.merged_data.day_of_week = self.merged_data.index.strftime("%w")

        print("Successfully merged Building and Weather data!")
        return self.merged_data, self.building_lat, self.building_lon

    def _pysolar_features(self):
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

        print("Successfully added Azimuth and Irradiance data!")
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
            train_pct (float) : amount of the dataset used in training (optional)
            val_pct (float)   : amount of the dataset used in validation (optional)
            test_pct (float)  : amount of the dataset used in testing(optional)
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

        assert (self.train_pct + self.test_pct + self.val_pct) == 1

    def _make_series(self, input_df):
        """
        Create 3D array of slices based on series_length and stride

        Argument:
            input_df           : a well formatted dataframe with features and two dependent variables as columns

        Return:
            a 3D numpy array with each slice containing a series length of hours worth of data
        """

        ### TODO add random sample to stride between 1 and 5
        self.data = input_df.copy()
        self.start_index = self.series_length
        self.end_index = len(self.data)

        self.output = []

        # Iterate through dataframe at stride length, creating slice of series_length
        for i in range(
            self.start_index,
            self.end_index,
            self.stride,
        ):
            self.indices = range(i - self.series_length, i, 1)
            self.output.append(self.data.iloc[self.indices])

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

    def _xy_splits(self, input_df):
        """
        Separates the sets of x matrixes and y column vectors

        Argument:
            input_df           : 3D numpy array with two trailing y columns

        Return:
            a tuple of numpy arrays: (x, y)
        """

        self.dataset = input_df.copy()

        ## Remove last columns to make y vectors for the dataset

        return (
            self.dataset[:, :, :-2].astype("float32"),
            self.dataset[:, :, -2:].astype("float32"),
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
        self.output_list = ["train", "val", "test"]
        for i, df in enumerate(self.pre_split):
            self.output_list[i] = self._make_series(df)

        ## split x and y columns from train, val, and test sequenced datasets
        self.train_split = self._xy_splits(self.output_list[0])
        self.val_split = self._xy_splits(self.output_list[1])
        self.test_split = self._xy_splits(self.output_list[2])

        # Return a tuple of tuples
        print(
            "Successfully split data into (train_x, train_y), (val_x, val_y), (test_x, test_y), (norm_layer)!"
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
