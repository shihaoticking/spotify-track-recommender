import os

import datasets
from pyspark.sql import DataFrame, SparkSession


class DataLoader:
    DATASET_SOURCE = 'maharshipandya/spotify-tracks-dataset'

    def __init__(self, local_folder: str = 'data/raw'):
        self.local_folder = local_folder
        self.dataset: datasets.Dataset = None

    def download_dataset(self):
        """
        Downloads the dataset specified by DATASET_SOURCE and saves it to disk.
        
        Args:
            None
        
        Returns:
            None
        """
        ds = datasets.load_dataset(self.DATASET_SOURCE, split='train')
        ds = ds.remove_columns('Unnamed: 0')
        ds.save_to_disk(self.local_folder)

    def load_dataset(self) -> datasets.Dataset:
        """
        Loads a dataset from disk if it exists, otherwise downloads it first.

        Returns:
            datasets.Dataset: The loaded dataset.
        """
        if not os.path.exists(self.local_folder) or not os.listdir(self.local_folder):
            self.download_dataset()

        self.dataset = datasets.load_from_disk(self.local_folder)

        return self.dataset

    def load_pyspark_df(self, spark: SparkSession) -> DataFrame:
        """
        Loads a PySpark DataFrame from the dataset.

        Args:
            spark (SparkSession): The SparkSession to use for creating the DataFrame.

        Returns:
            DataFrame: A PySpark DataFrame representation of the dataset.
        """
        if self.dataset is None:
            self.load_dataset()

        pandas_df = self.dataset.to_pandas()

        return spark.createDataFrame(pandas_df)
