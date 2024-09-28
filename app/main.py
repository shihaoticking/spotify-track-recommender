import json
import os
from contextlib import asynccontextmanager
from typing import List

import gcsfs
import hnswlib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, RootModel

# Load data from GCS
TRACK_IDS_FILE = os.environ.get('TRACK_IDS_FILE')
INDEX_FILE = os.environ.get('INDEX_FILE')
DATASET_FOLDER = os.environ.get('DATASET_FOLDER')

# Objects to be served
TRACK_IDS: List[str] = None
HNSW_INDEX: hnswlib.Index = None
TRACK_ID_MAP: dict = None


class TrackResponse(BaseModel):
    track_id: str
    artists: str
    album_name: str
    track_name: str


class RecommendationResponse(RootModel[List[TrackResponse]]):
    pass


def load_track_ids() -> List[str]:
    fs = gcsfs.GCSFileSystem()
    with fs.open(TRACK_IDS_FILE, 'r') as f:
        return json.load(f)


def load_dataset() -> pd.DataFrame:
    fs = gcsfs.GCSFileSystem()
    blobs = fs.glob(DATASET_FOLDER)
    parquet_files = [f'gs://{blob}' for blob in blobs]

    dfs = []
    for parquet_file in parquet_files:
        dfs.append(pd.read_parquet(parquet_file))

    return pd.concat(dfs, ignore_index=True)


def convert_dataset_to_track_id_map(df: pd.DataFrame) -> dict:
    columns = ['track_name', 'artists', 'album_name', 'track_genres', 'embedding',
               'popularity', 'artist_popularity', 'genre_popularity']
    return df.set_index('track_id')[columns].to_dict(orient='index')


def get_embedding_dim(df: pd.DataFrame) -> int:
    return df['embedding'].iloc[0].shape[0]


def load_hnsw_index(dim: int) -> hnswlib.Index:
    fs = gcsfs.GCSFileSystem()
    local_index_file = '/tmp/hnsw_index.bin'
    fs.download(INDEX_FILE, local_index_file)

    index = hnswlib.Index(space='cosine', dim=dim)
    index.load_index(local_index_file)

    return index


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load data from GCS
    global TRACK_IDS
    global TRACK_ID_MAP
    global HNSW_INDEX

    TRACK_IDS = load_track_ids()
    dataset = load_dataset()
    TRACK_ID_MAP = convert_dataset_to_track_id_map(dataset)
    dim = get_embedding_dim(dataset)
    HNSW_INDEX = load_hnsw_index(dim)

    yield

    TRACK_IDS = None
    TRACK_ID_MAP = None
    HNSW_INDEX = None


app = FastAPI(lifespan=lifespan)


@app.get('/healthcheck')
def healthcheck():
    """
    Returns a healthcheck response.
    """
    return {'status': 'ok'}


@app.get('/recommendations/{track_id}', response_model=RecommendationResponse)
def get_recommendations(track_id: str, k: int = Query(default=10, ge=1, le=100)):
    """
    Given a track_id, returns the top_k nearest items based on the HNSW index.
    """

    if track_id not in TRACK_ID_MAP:
        raise HTTPException(status_code=404, detail='Track not found')
    
    track_info = TRACK_ID_MAP[track_id]
    embedding = track_info['embedding']

    # reshape embedding to (1, dim)
    embedding = embedding.reshape(1, -1)

    # Query the HNSW index
    labels, _ = HNSW_INDEX.knn_query(embedding, k=101)

    # Convert labels to track_ids and exclude query track_id itself
    recommendations = [TRACK_IDS[label] for label in labels[0] if TRACK_IDS[label] != track_id]

    # Apply hybrid match mode
    recommendations = sorted(recommendations, key=lambda rec: (
        TRACK_ID_MAP[rec]['popularity'],
        TRACK_ID_MAP[rec]['artist_popularity'],
        TRACK_ID_MAP[rec]['genre_popularity']
    ), reverse=True)

    # Return recommendations with schema
    return [
        {
            'track_id': rec,
            'artists': TRACK_ID_MAP[rec]['artists'],
            'album_name': TRACK_ID_MAP[rec]['album_name'],
            'track_name': TRACK_ID_MAP[rec]['track_name'],
        }
        for rec in recommendations[:k]
    ]
