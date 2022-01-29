import sys

import pandas as pd
import numpy as np
from glob import glob
import warnings

from pysolar.solar import get_altitude
from pysolar.radiation import get_radiation_direct
from clearml import Dataset

# TODO: replace prints with logging
class DataGenerator:
    """
    Generates data for training using weather and power data.
    """

    def __init__(
        self,
        weather_features: list = None,
        building_features: list = None,
        building_data_dir: str = None,
        weather_data_dir: str = None,
    ):
        """
        Initialize the data generator.

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

        # Set weather columns to keep
        self.weather_data_keep_columns = (
            self.weather_features
            if self.weather_features
            else ["temp", "pressure", "humidity", "clouds_all"]
        )

        # Set building columns to keep
        self.building_data_keep_columns = (
            self.building_features
            if self.building_features
            else [
                "True Power (kW)",
                "Total Energy (kWh)",
                "Reactive Energy (kVARh)",
                "Reactive Power (kVAR)",
                "Apparent Power (kVA)",
                "Apparent Energy (kVAh)",
                "dPF",
                "aPF",
                "Current (A)",
            ]
        )

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
            & (self.weather_data_keep_columns is not None)
            & (self.building_data_keep_columns is not None)
        ):
            # With data directories set, return the retrieved/cleaned/merged data
            return self._daylight_savings()

    def _load_building_data(self):
        """
        Load and format building data from file.

        Return:
            building_data (dataframe)
        """

        try:
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
            ].rename(
                columns={"PowerScout DPS126": "solar", "PowerScout DPS121": "usage"}
            )

            # Create our separate y-columns: solar pwr generated, buildling pwr used
            idx = pd.IndexSlice

            # Create DF with only Energy - keep just the last value (meter readings)
            self.building_data = self.building_data.loc[
                idx[:], idx[:, self.building_data_keep_columns]
            ]

            print("Info: Successfully loaded building data!")

        except:
            print("Error: Cannot load building data!")
            sys.exit(1)

        return self.building_data

    def _load_weather_data(self):
        """
        Load, format, and calculate weather data from file.

        Return:
            weather_data (dataframe)
        """
        warnings.filterwarnings("ignore")

        try:
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
            self.weather_data = self.weather_data[self.weather_data_keep_columns]

            # deduplicate index
            self.weather_data = self.weather_data.drop_duplicates()

            # Add Solar Irradiance
            self.weather_data["irradiance"] = np.nan
            self.date_list = list(self.weather_data.index)
            for date in self.date_list:
                self.altitude_deg = get_altitude(
                    self.building_lat, self.building_lon, date.to_pydatetime()
                )
                self.weather_data.loc[
                    date.to_pydatetime(), "irradiance"
                ] = get_radiation_direct(date.to_pydatetime(), self.altitude_deg)

            # Fill nulls in irradiance with 0
            self.weather_data = self.weather_data.replace(np.nan, 0)

            print("Info: Successfully loaded weather data!")

        except:
            print("Error: Cannot load weather data")
            sys.exit(1)

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
        self.merged_data = self.weather_data.merge(
            self.building_data.solar["True Power (kW)"],
            "outer",
            left_index=True,
            right_index=True,
        )
        self.merged_data = self.merged_data.fillna(method="ffill").dropna()
        self.merged_data.rename(columns={"True Power (kW)": "solar"}, inplace=True)

        # Merge Building Usage Y data to merged_data
        self.merged_data = self.merged_data.merge(
            self.building_data.usage["True Power (kW)"],
            "outer",
            left_index=True,
            right_index=True,
        ).fillna(method="ffill")
        self.merged_data.rename(columns={"True Power (kW)": "usage"}, inplace=True)

        return self.merged_data