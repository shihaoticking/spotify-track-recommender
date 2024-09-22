from argparse import ArgumentParser

from src.utils.spark_utils import initial_spark
from src.utils.data_loader import DataLoader

APP_NAME = 'enrich_dataset_aggregation'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--artist_popularity_gcs_input_folder', required=True)
    parser.add_argument('--genre_popularity_gcs_input_folder', required=True)
    parser.add_argument('--gcs_output_folder', required=True)

    args = parser.parse_args()

    spark = initial_spark(APP_NAME)
    data_loader = DataLoader()

    df = data_loader.load_pyspark_df(spark)

    # load genre popularity
    genre_popularity = spark.read.parquet(args.genre_popularity_gcs_input_folder)

    # load artist popularity
    artist_popularity = spark.read.parquet(args.artist_popularity_gcs_input_folder)

    # join genre popularity and artist popularity
    df = df.join(genre_popularity, on=['track_genre'], how='left')
    df = df.join(artist_popularity, on=['artists'], how='left')

    df.show(10)

    df.write.parquet(args.gcs_output_folder)
