# spotify-track-recommender

## Overview

This project demonstrates a robust, end-to-end machine learning pipeline for recommending Spotify tracks. It showcases proficiency in big data processing, machine learning, and modern software engineering practices.

### Key technologies

- **Data Processing**: Apache Spark (PySpark)
- **Workflow Orchestration**: Apache Airflow
- **Machine Learning**: Custom embedding generation, HNSW (Hierarchical Navigable Small World) indexing
- **API Development**: FastAPI
- **Containerization & Deployment**: Docker, Google Cloud Run
- **CI/CD**: GitHub Actions

## Architecture

1. **Data Ingestion**: Spotify track data is ingested and processed using PySpark for scalability.
1. **Feature Engineering**: Custom track embeddings are generated using PySpark, leveraging distributed computing for efficiency.
1. **Indexing**: An HNSW index is built on top of the track embeddings, enabling efficient approximate nearest neighbor search.
1. **Workflow Orchestration**: Apache Airflow manages the data processing and model training pipeline, ensuring reproducibility and scalability.
1. **Recommendation Model**: A machine learning model utilizes the processed data and HNSW index to generate personalized track recommendations.
1. **API Layer**: A FastAPI application serves the recommendation model, providing a RESTful interface for client applications.

## Repository Structure

```
spotify-track-recommender/
├── jobs/                 # PySpark jobs for data processing and embedding generation
├── dags/                 # Apache Airflow DAGs for workflow orchestration
├── app/                  # FastAPI application serving the recommendation model
├── tests/                # Unit test
├── .github/workflows/    # CI/CD pipeline configurations
└── Dockerfile            # Docker configuration for containerization
```

## CI/CD Pipeline

This project utilizes GitHub Actions for continuous integration and deployment:

1. Build, Test, and Push Image Workflow:
    - Triggered on push events
    - Builds Docker image
    - Runs automated tests using pytest
    - Pushes image to Google Cloud Artifact Registry (main branch only)

2. Deploy to Google Cloud Run Workflow:
    - Manually triggered
    - Deploys the latest image to Google Cloud Run
