from argparse import ArgumentParser

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col, when

from src.utils.data_loader import DataLoader
from src.utils.spark_utils import initial_spark

APP_NAME = 'embedding_generation'

DENSE_FEATURES = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'acousticness', 
                  'instrumentalness', 'liveness', 'speechiness']
CATEGORICAL_FEATURES = ['explicit', 'key', 'mode', 'time_signature']


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gcs_output_folder', required=True)

    args = parser.parse_args()

    spark = initial_spark(APP_NAME)
    data_loader = DataLoader()

    df = data_loader.load_pyspark_df(spark)

    # Convert boolean 'explicit' column to numeric
    df = df.withColumn('explicit', when(col('explicit'), 1).otherwise(0))

    # One-hot encode categorical features
    encoder = OneHotEncoder(inputCols=CATEGORICAL_FEATURES, 
                        outputCols=[f'{col}_encoded' for col in CATEGORICAL_FEATURES])

    # Assemble all features into a single vector
    all_features = DENSE_FEATURES + [f'{col}_encoded' for col in CATEGORICAL_FEATURES]
    assembler = VectorAssembler(inputCols=all_features, outputCol='features_assembled')

    # Scale the assembled features
    scaler = StandardScaler(inputCol='features_assembled', outputCol='features_scaled',
                            withStd=True, withMean=True)

    # Create and fit the pipeline
    pipeline = Pipeline(stages=[encoder, assembler, scaler])
    model = pipeline.fit(df)
    df_scaled = model.transform(df)

    # Convert vector to array for HNSW index
    df_scaled = df_scaled.withColumn('embedding', vector_to_array('features_scaled'))

    df_final = df_scaled.select('track_id', 'embedding')
    
    df_final.write.parquet(args.gcs_output_folder)
