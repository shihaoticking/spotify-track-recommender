from argparse import ArgumentParser

import pyspark.sql.functions as f

from src.utils.spark_utils import initial_spark
from src.utils.data_loader import DataLoader

APP_NAME = 'genre_popularity_generation'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gcs_output_folder', required=True)

    args = parser.parse_args()

    spark = initial_spark(APP_NAME)
    data_loader = DataLoader()

    df = data_loader.load_pyspark_df(spark)

    genre_popularity = df.groupBy('track_genre').agg(f.mean('popularity').alias('genre_popularity'))
    genre_popularity.show(10)

    genre_popularity.write.parquet(args.gcs_output_folder)
