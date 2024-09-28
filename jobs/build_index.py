import json
import os
from argparse import ArgumentParser

import hnswlib
import numpy as np

from src.utils.gcs_utils import upload_blob
from src.utils.spark_utils import initial_spark

APP_NAME = 'build_index'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gcs_input_folder', required=True)
    parser.add_argument('--gcs_output_bucket', required=True)
    parser.add_argument('--gcs_output_blob', required=True)

    args = parser.parse_args()

    spark = initial_spark(APP_NAME)

    df = spark.read.parquet(args.gcs_input_folder)

    track_id_to_embedding_map = df.select('track_id', 'embedding').rdd.map(
        lambda row: (row['track_id'], np.array(row['embedding'], dtype=np.float32))).collectAsMap()

    # Separate track_ids and embeddings, now that they are aligned
    track_ids, embeddings = zip(*track_id_to_embedding_map.items())

    # Initialize the HNSW index for cosine similarity
    dim = len(embeddings[0])  # Get dimension from the first embedding
    index = hnswlib.Index(space='cosine', dim=dim)

    # Number of elements to insert
    num_elements = len(embeddings)

    # Initialize the index
    index.init_index(num_elements)

    # Add items (embeddings) to the index
    # We use the order of the embeddings and track_ids to make sure they align
    index.add_items(embeddings, ids=list(range(num_elements)))

    # Save index to local disk
    index_path = '/tmp/hnsw_index.bin'
    index.save_index(index_path)

    # Upload the index file to Google Cloud Storage (GCS)
    index_output_file_name = os.path.join(args.gcs_output_blob, 'hnsw_index.bin')
    upload_blob(args.gcs_output_bucket, index_path, index_output_file_name)

    # Save track_ids to local disk
    track_ids_path = '/tmp/track_ids.json'
    with open(track_ids_path, 'w') as f:
        json.dump(track_ids, f)

    # Upload the track_ids to Google Cloud Storage (GCS)
    track_ids_file_name = os.path.join(args.gcs_output_blob, 'track_ids.json')
    upload_blob(args.gcs_output_bucket, track_ids_path, track_ids_file_name)
