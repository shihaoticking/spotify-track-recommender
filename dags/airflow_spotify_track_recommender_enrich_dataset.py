from airflow import DAG
from airflow.contrib.operators.dataproc_operator import (
    DataprocClusterCreateOperator,
    DataprocSubmitPySparkJobOperator,
    DataprocClusterDeleteOperator
)
from airflow.models import Variable
from airflow.utils.dates import days_ago

# Define default arguments
default_args = {
    'owner': 'howard_wang',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1
}

# Get variables
PROJECT = Variable.get("project_id")
MAINFILE_BUCKET = Variable.get("mainfile_bucket")
SPOTIFY_RECOMMENDER_BUCKET = Variable.get("spotify_recommender_bucket")

# Define DAG
dag = DAG(
    'spotify_track_recommender_enrich_dataset',
    default_args=default_args,
    description='DAG for Spotify Track Dataset Enrichment',
    schedule_interval=None,
)

# Create Dataproc cluster
create_cluster = DataprocClusterCreateOperator(
    task_id='create_dataproc_cluster',
    project_id=PROJECT,
    cluster_name='spotify-recommender-cluster-{{ ds_nodash }}',
    num_workers=0,
    region='us-west1',
    zone='us-west1-a',
    master_machine_type='n2-standard-4',
    master_disk_type='pd-ssd',
    master_disk_size=100,
    image_version='2.2-debian12',
    properties={
        'dataproc:pip.packages': 'datasets==3.0.0'
    },
    dag=dag
)

# Define PySpark job submissions
job_args = [
    {
        'task_id': 'artist_popularity_generation',
        'main': f'gs://{MAINFILE_BUCKET}/spotify-track-recommender/artist_popularity_generation.py',
        'arguments': [f'--gcs_output_folder=gs://{SPOTIFY_RECOMMENDER_BUCKET}/artist_popularity_generation/']
    },
    {
        'task_id': 'genre_popularity_generation',
        'main': f'gs://{MAINFILE_BUCKET}/spotify-track-recommender/genre_popularity_generation.py',
        'arguments': [f'--gcs_output_folder=gs://{SPOTIFY_RECOMMENDER_BUCKET}/genre_popularity_generation/']
    },
    {
        'task_id': 'enrich_dataset_aggregation',
        'main': f'gs://{MAINFILE_BUCKET}/spotify-track-recommender/jobs/enrich_dataset_aggregation.py',
        'arguments': [
            f'--artist_popularity_gcs_input_folder=gs://{SPOTIFY_RECOMMENDER_BUCKET}/artist_popularity_generation/',
            f'--genre_popularity_gcs_input_folder=gs://{SPOTIFY_RECOMMENDER_BUCKET}/genre_popularity_generation/',
            f'--gcs_output_folder=gs://{SPOTIFY_RECOMMENDER_BUCKET}/enrich_dataset/'
        ]
    }
]

pyspark_jobs = []
for job in job_args:
    pyspark_job = DataprocSubmitPySparkJobOperator(
        task_id=job['task_id'],
        main=job['main'],
        arguments=job['arguments'],
        pyfiles=[f'gs://{MAINFILE_BUCKET}/spotify-track-recommender/src.zip'],
        region='us-west1',
        cluster_name='spotify-recommender-cluster-{{ ds_nodash }}',
        project_id=PROJECT,
        dag=dag
    )
    pyspark_jobs.append(pyspark_job)

# Delete Dataproc cluster
delete_cluster = DataprocClusterDeleteOperator(
    task_id='delete_dataproc_cluster',
    project_id=PROJECT,
    cluster_name='spotify-recommender-cluster-{{ ds_nodash }}',
    region='us-west1',
    trigger_rule='all_done',
    dag=dag
)

# Set up task dependencies
create_cluster >> pyspark_jobs >> delete_cluster
