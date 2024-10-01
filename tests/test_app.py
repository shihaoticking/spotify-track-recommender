from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import TrackResponse, app

client = TestClient(app)


@pytest.fixture
def mock_data():
    return {
        'TRACK_IDS': ['track1', 'track2', 'track3', 'track4', 'track5'],
        'TRACK_ID_MAP': {
            'track1': {'embedding': np.array([0.1, 0.2, 0.3]), 'artists': 'Artist1', 'album_name': 'Album1', 'track_name': 'Track1', 'popularity': 80, 'artist_popularity': 75, 'genre_popularity': 70},
            'track2': {'embedding': np.array([0.2, 0.3, 0.4]), 'artists': 'Artist2', 'album_name': 'Album2', 'track_name': 'Track2', 'popularity': 70, 'artist_popularity': 65, 'genre_popularity': 60},
            'track3': {'embedding': np.array([0.3, 0.4, 0.5]), 'artists': 'Artist3', 'album_name': 'Album3', 'track_name': 'Track3', 'popularity': 90, 'artist_popularity': 85, 'genre_popularity': 80},
            'track4': {'embedding': np.array([0.4, 0.5, 0.6]), 'artists': 'Artist4', 'album_name': 'Album4', 'track_name': 'Track4', 'popularity': 60, 'artist_popularity': 55, 'genre_popularity': 50},
            'track5': {'embedding': np.array([0.5, 0.6, 0.7]), 'artists': 'Artist5', 'album_name': 'Album5', 'track_name': 'Track5', 'popularity': 85, 'artist_popularity': 80, 'genre_popularity': 75},
        },
        'HNSW_INDEX': MagicMock()
    }


@pytest.fixture
def mock_environment(mock_data):
    with patch('app.main.TRACK_IDS', mock_data['TRACK_IDS']), \
         patch('app.main.TRACK_ID_MAP', mock_data['TRACK_ID_MAP']), \
         patch('app.main.HNSW_INDEX', mock_data['HNSW_INDEX']):
        yield


def test_get_recommendations_success(mock_environment, mock_data):
    mock_data['HNSW_INDEX'].knn_query.return_value = (np.array([[1, 2, 3, 4]]), np.array([[0.1, 0.2, 0.3, 0.4]]))
    
    response = client.get('/recommendations/track1?k=3')
    
    assert response.status_code == 200
    recommendations = response.json()
    assert len(recommendations) == 3
    assert all(isinstance(rec, dict) for rec in recommendations)
    assert all(set(rec.keys()) == {'track_id', 'artists', 'album_name', 'track_name'} for rec in recommendations)


def test_get_recommendations_track_not_found(mock_environment):
    response = client.get('/recommendations/nonexistent_track')
    
    assert response.status_code == 404
    assert response.json()['detail'] == 'Track not found'


def test_get_recommendations_invalid_k(mock_environment):
    response = client.get('/recommendations/track1?k=0')
    assert response.status_code == 422

    response = client.get('/recommendations/track1?k=101')
    assert response.status_code == 422


def test_get_recommendations_default_k(mock_environment, mock_data):
    mock_data['HNSW_INDEX'].knn_query.return_value = (np.array([[1, 2, 3, 4, 0]]), np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]))
    
    response = client.get('/recommendations/track1')
    
    assert response.status_code == 200
    recommendations = response.json()
    assert len(recommendations) == 4  # Default k is 10, but we only have 4 other tracks


def test_get_recommendations_exclude_query_track(mock_environment, mock_data):
    mock_data['HNSW_INDEX'].knn_query.return_value = (np.array([[0, 1, 2, 3, 4]]), np.array([[0, 0.1, 0.2, 0.3, 0.4]]))
    
    response = client.get('/recommendations/track1')
    
    assert response.status_code == 200
    recommendations = response.json()
    assert 'track1' not in [rec['track_id'] for rec in recommendations]


def test_get_recommendations_hybrid_match_mode(mock_environment, mock_data):
    mock_data['HNSW_INDEX'].knn_query.return_value = (np.array([[1, 2, 3, 4]]), np.array([[0.1, 0.2, 0.3, 0.4]]))
    
    response = client.get('/recommendations/track1')
    
    assert response.status_code == 200
    recommendations = response.json()
    # Check if recommendations are sorted by popularity
    popularities = [mock_data['TRACK_ID_MAP'][rec['track_id']]['popularity'] for rec in recommendations]
    assert popularities == sorted(popularities, reverse=True)


def test_get_recommendations_response_model(mock_environment, mock_data):
    mock_data['HNSW_INDEX'].knn_query.return_value = (np.array([[1, 2, 3, 4]]), np.array([[0.1, 0.2, 0.3, 0.4]]))
    
    response = client.get('/recommendations/track1')
    
    assert response.status_code == 200
    recommendations = response.json()
    for rec in recommendations:
        TrackResponse(**rec)  # This will raise a validation error if the response doesn't match the model
