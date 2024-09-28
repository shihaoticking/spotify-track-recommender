from argparse import ArgumentParser

import pyspark.sql.functions as f

from src.utils.data_loader import DataLoader
from src.utils.spark_utils import initial_spark

APP_NAME = 'artist_popularity_generation'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gcs_output_folder', required=True)

    args = parser.parse_args()

    spark = initial_spark(APP_NAME)
    data_loader = DataLoader()

    df = data_loader.load_pyspark_df(spark)

    df = df.dropDuplicates(subset=['track_id', 'artists'])

    artist_popularity = df.groupBy('artists').agg(f.mean('popularity').alias('artist_popularity'))
    artist_popularity.show(10)

    artist_popularity.write.parquet(args.gcs_output_folder)
