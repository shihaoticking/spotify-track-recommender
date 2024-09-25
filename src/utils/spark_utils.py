from pyspark.sql import SparkSession


def initial_spark(app_name: str) -> SparkSession:
    """
    Create a SparkSession with specified app name and common config.
    """
    return (
        SparkSession.builder
        .master('local[*]')
        .appName(app_name)
        .getOrCreate()
    )
