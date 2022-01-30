import sys
import warnings
from glob import glob

import numpy as np
import pandas as pd
from clearml import Dataset
from pysolar.radiation import get_radiation_direct
from pysolar.solar import get_altitude, get_azimuth


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
            merged_data (dataframe)    : weather and building data in one dataframe
        """

        # Now actually load the data. 1st check that directories are set
        if (
            (self.weather_data_dir is not None)
            & (self.building_data_dir is not None)
            & (self.weather_features is not None)
            & (self.building_features is not None)
        ):
            # With data directories set, return the retrieved/cleaned/merged data
            return self._daylight_savings()

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

        print("Info: Successfully loaded building data!")

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

        # Add Solar Irradiance
        self.weather_data["azimuth"] = np.nan
        self.weather_data["irradiance"] = np.nan
        self.date_list = list(self.weather_data.index)
        for date in self.date_list:
            self.pydate = date.to_pydatetime()
            # Calculate Solar Azimuth
            self.weather_data.loc[date, "azimuth"] = get_azimuth(
                self.building_lat, self.building_lon, self.pydate
            )
            # Calculate Solar Altitude
            self.altitude_deg = get_altitude(
                self.building_lat, self.building_lon, self.pydate
            )
            # Calculate Solar Irradiance
            self.weather_data.loc[date, "irradiance"] = get_radiation_direct(
                self.pydate, self.altitude_deg
            )

        # Fill nulls in irradiance with 0
        self.weather_data = self.weather_data.replace(np.nan, 0)

        print("Info: Successfully loaded weather data!")
        return self.weather_data

    def _daylight_savings(self):
        """
        With loaded weather data and building data,
        match indexes and adjust for daylight savings.

        Return:
            merged_data (dataframe)    : weather and building data in one dataframe
        """

        self.building_data = self._load_building_data()
        self.weather_data = self._load_weather_data()

        # Add day of week to Weather Data
        self.weather_data["day_of_week"] = self.weather_data.index.strftime("%w")

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

        print("Successfully merged Building and Weather Data")
        return self.merged_data

class DataSplit:
    """
    Splits data into series and then randomly splits those series
    into training, validation, and training sets.
    """

    def __init__(
        self,
        dataframe,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        shuffle: bool = True,
        series_length: int = 48,
        stride: int = 3,
    ):
        """
        Initializes the data splitting class.

        Arguments:
            dataframe           : a well formatted dataframe with features and two dependent variables as columns
            train_split (float) : amount of the dataset used in training (optional)
            val_split (float)   : amount of the dataset used in validation (optional)
            test_split (float)  : amount of the dataset used in testing(optional)
            shuffle (bool)      : wheather or not to randomly shuffle the series before splitting (optional)
            series_length (int) : how many observations are in a series (optional)
            stride (int)        : how many observations to stride over when building series (optional)
        """

        self.dataframe = dataframe
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.shuffle = shuffle
        self.series_length = series_length
        self.stride = stride

    def _make_series(self, dataframe):
        """
        Create 3D array of slices based on series_length and stride

        Arguments:
            dataframe           : a well formatted dataframe with features and two dependent variables as columns
        """

        ### TODO add random sample to stride between 1 and 5
        self.data = dataframe
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

    def _train_val_test_split(self):
        """
        Randomly splits series into training, validation, and training sets.
        """

        # Create 3D array of time slices using make_array function
        self.data_array = self._make_series(self.dataframe)

        # Verify splits combine to equal 1
        assert (self.train_split + self.test_split + self.val_split) == 1

        if self.shuffle:
            # Specify seed to always have the same split distribution between runs
            self.rng = np.random.default_rng()
            # Shuffle the index numbers
            self.rng.shuffle(self.data_array, axis=0)

        # Split the shuffled index numbers into the 3 bins - train, val, test
        self.indices_or_sections = [
            int(self.train_split * self.data_array.shape[0]),
            int((1 - self.test_split) * self.data_array.shape[0]),
        ]

        # Perform the split based on shuffled index using Numpy split
        self.train_ds, self.val_ds, self.test_ds = np.split(
            self.data_array, self.indices_or_sections
        )

        return self.train_ds, self.val_ds, self.test_ds

    def xy_splits(self, dataset):
        """
        Separates the sets of x matrixes and y column vectors

        Arguments:
            datset           : 3D numpy array with two trailing y columns
        """

        self.dataset = dataset

        ## Remove last columns to make y vectors for the dataset

        return (
            self.dataset[:, :, :-2].astype("float32"),
            self.dataset[:, :, -2:].astype("float32"),
        )

    def split_data(self):
        """
        Splits dataset into a tuple of tuples
        """

        # Run train_val_split_function
        self.train_ds, self.val_ds, self.test_ds = self._train_val_test_split()

        ## split train, val, and test sets
        self.train_split = self.xy_splits(self.train_ds)
        self.val_split = self.xy_splits(self.val_ds)
        self.test_split = self.xy_splits(self.test_ds)

        # Return a tuple of tuples
        return (
            self.train_split,
            self.val_split,
            self.test_split,
        )

        # train_split[0] -> features
        # train_split[1] -> solar
        # train_split[2] -> usage
