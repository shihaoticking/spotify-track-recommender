# spotify-track-recommender

This repository demonstrates a data pipeline built with PySpark, job orchestration with Apache Airflow, and a FastAPI application for recommending Spotify tracks.

## Overview

The pipeline ingests Spotify track data, processes it using PySpark, and generates embeddings for each track. The embeddings are then used to build a recommendation model, which is served through a FastAPI application.

## Components

1. **Data Ingestion**: Spotify track data is ingested from a source dataset and processed using PySpark.
1. **Data Processing**: The ingested data is processed using PySpark to generate embeddings for each track.
1. **HNSW Index**: An HNSW (Hierarchical Navigable Small World) index is built on top of the track embeddings to enable efficient nearest neighbor search.

1. **Job Orchestration**: Apache Airflow is used to orchestrate the data processing jobs, ensuring that the pipeline runs smoothly and efficiently.
1. **Recommendation Model**: The processed data is used to build a recommendation model, which is served through a FastAPI application.
1. **FastAPI Application**: The FastAPI application provides a RESTful API for recommending Spotify tracks based on user input.

## Repository Structure

- `jobs`: Contains the PySpark jobs for data processing and embedding generation.
- `dags`: Contains the Apache Airflow DAGs for job orchestration.
- `app`: Contains the FastAPI application code for the recommendation model.
