
import pandas as pd
import numpy as np

import datetime as dt

from glob import glob

from pysolar.solar import get_altitude, radiation
from clearml import Dataset

# TODO: replace prints with logging
class DataGenerator:
    """
    Generates data for training using weather and power data.
    """

    def __init__(self,
                 use_local_data: bool=False,

                 weather_features: list=None,
                 building_features: list=None,

                 building_lat: float=39.9649,
                 building_lon: float=-75.1396,

                 building_data_dir: str=None,
                 weather_data_dir: str=None,
                ):
        """
        Initialize data generator.

        Arguments:
            use_local_data (bool)   : whether or not to use local data or ClearML server
        """
        self.use_local_data = use_local_data
        self.weather_features = weather_features
        self.building_features = building_features
        self.building_lat = building_lat
        self.building_lon = building_lon

        self.building_data_dir = building_data_dir
        self.weather_data_dir = weather_data_dir

        # Data
        self.start_date = None
        self.end_date = None
        self.building_data: pd.DataFrame = None  # Building data
        self.weather_data: pd.DataFrame = None  # Weather data

        # Check for specified/local building and weather directories. If not specified, use ClearML
        if not self.building_data_dir:
            try:
                self.building_data_dir = glob(
                    Dataset.get(
                        dataset_project="amplify",
                        dataset_name="building_data"
                    ).get_local_copy()
                    + "/**"
                )[0]
            except:
                print('No directory was specified and building data could not be retrieved from ClearML')

        if not self.weather_data_dir:
            try:
                self.weather_data_dir = glob(
                    Dataset.get(
                        dataset_project="amplify",
                        dataset_name="weather_data"
                    ).get_local_copy()
                    + "/**"
                )[0]
            except:
                print('No directory was specified and weather data could not be retrieved from ClearML')

        # Set weather columns to keep
        self.weather_data_keep_columns = (
            self.weather_features
            if self.weather_features
            else ["temp",
                  "pressure",
                  "humidity",
                  "clouds_all"
                 ]
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
        Loads data for a specified building/weather data paths. If no paths are specified, uses
        default paths.

        *NOTE: if providing a path, will only use local data for paths not provided

        Arguments:
            building_data_path (str)    : building data file path (optional)
            weather_data_path (str)     : weather data file path (optional)

        Return:
            (load_success, building_data, weather_data)

            load_success (bool)         : whether or not loading data was successful
            building_data (dataframe)   : building data
            weather_data (dataframe)    : weather data
        """

        # Now actually load the data. 1st check that directories are set
        if ((self.weather_data_dir is not None)
            & (self.building_data_dir is not None)
            & (self.weather_data_keep_columns is not None)
            & (self.building_data_keep_columns is not None)
           ):
            # With data directories set, return the retrieved/cleaned/merged data
            return self._daylight_savings()

    def _load_building_data(self):
        """
        Load and format building data from file

        Return:
            cleaned building data
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
            self.building_data = self.building_data.tz_localize("America/New_York",
                                                                ambiguous=True
                                                               ).tz_convert("UTC")

            # deduplicate index
            self.building_data = self.building_data.drop_duplicates(keep="last")

            # Drop any column or row with all nulls
            self.building_data = self.building_data.dropna(how="all", axis=1
                                                          ).dropna(how="all", axis=0
                                                                  )

            # remove noise (zeros) from the building_data
            self.building_data = self.building_data.replace(0, np.nan).fillna(method="ffill")

            # Slice to the two power systems we're monitoring and rename columns
            # TODO: make the power systems vairable at some point
            self.building_data = self.building_data[["PowerScout DPS126",
                                                     "PowerScout DPS121"]
                                                   ].rename(columns={"PowerScout DPS126": "solar",
                                                                     "PowerScout DPS121": "usage"
                                                                    }
                                                           )

            # Create our separate y-columns: solar pwr generated, buildling pwr used
            idx = pd.IndexSlice

            # Create DF with only Energy - keep just the last value (meter readings)
            self.building_data = self.building_data.loc[
                idx[:],
                idx[:, self.building_data_keep_columns]
            ]

            print("Info: Successfully loaded building data!")

        except:
            print("Error: Cannot load building data!")

        return self.building_data

    def _load_weather_data(self):
        """
        Load and format weather data from file

        Return:
            cleaned weather data
        """

        try:
            self.weather_data = pd.read_csv(
                self.weather_data_dir, header=0, low_memory=False
            )

            #### pull lat/lon from weather data for irradiance
            #self.building_lat
            #self.building_lon

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
                self.altitude_deg = get_altitude(self.building_lat,
                                                 self.building_lon,
                                                 date.to_pydatetime()
                                                )
                self.weather_data.loc[date.to_pydatetime(),
                                      "irradiance"
                ] = radiation.get_radiation_direct(date.to_pydatetime(),
                                                   self.altitude_deg)

            # Fill nulls in irradiance with 0
            self.weather_data = self.weather_data.replace(np.nan, 0)

            print("Info: Successfully loaded weather data!")

        except:
            print("Error: Cannot load weather data")

        return self.weather_data

    def _daylight_savings(self):
        """
        With loaded weather data and building data,
        match indexes and adjust for daylight savings.

        *NOTE: Forces Building and Weather data loading functions
        to run

        Return: merged clean building and weather data.
        """
        self.building_data = self._load_building_data()
        self.weather_data = self._load_weather_data()

            # Merge Building Solar Generation Y data to merged_data
        self.merged_data = self.weather_data.merge(
                self.building_data.solar['True Power (kW)'],
                'outer',
                left_index=True,
                right_index=True
        )
        self.merged_data = self.merged_data.fillna(method='ffill'
            ).dropna()
        self.merged_data.rename(columns={'True Power (kW)' : 'solar'}, inplace=True)

            # Merge Building Usage Y data to merged_data
        self.merged_data = self.merged_data.merge(
                self.building_data.usage['True Power (kW)'],
                'outer',
                left_index=True,
                right_index=True
            ).fillna(method='ffill')
        self.merged_data.rename(columns={'True Power (kW)' : 'usage'}, inplace=True)

        return self.merged_data
