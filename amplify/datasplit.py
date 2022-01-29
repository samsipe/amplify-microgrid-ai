import numpy as np


class DataSplit:
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

        self.dataframe = dataframe
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.shuffle = shuffle
        self.series_length = series_length
        self.stride = stride

    def _make_series(self, dataframe):
        # Create 3D array of slices based on series_length and stride

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
        # Create 3D array of time slices using make_array function
        self.data_array = self._make_series(self.dataframe)

        # Verify splits combine to equal 1
        assert (self.train_split + self.test_split + self.val_split) == 1

        if self.shuffle:
            # Specify seed to always have the same split distribution between runs
            self.rng = np.random.default_rng(seed=42)
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
        self.dataset = dataset

        ## Remove last columns to make y vectors for the dataset
        self.x_ds, self.y_solar, self.y_usage = (
            self.dataset[:, :, 0:-2],
            self.dataset[:, :, -2],
            self.dataset[:, :, -1],
        )

        return self.x_ds, self.y_solar, self.y_usage

    def split_data(self):
        # Run train_val_split_function
        self.train_ds, self.val_ds, self.test_ds = self._train_val_test_split()

        ## split train, val, and test sets
        self.train_split = self.xy_splits(self.train_ds)
        self.val_split = self.xy_splits(self.val_ds)
        self.test_split = self.xy_splits(self.test_ds)

        # Return a tuple of tuples
        return (self.train_split, self.val_split, self.test_split)

        # train_split[0] -> features
        # train_split[1] -> solar
        # train_split[2] -> usage