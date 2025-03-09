"""
Configuration module for easyinference.

Loads configuration from environment variables with fallback to .env file.
"""

import os
from typing import List

# Google Cloud Platform Configuration
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-project-id")
GCP_PROJECT_NUM = os.getenv("GCP_PROJECT_NUM", "123456789012")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")
VERTEX_BUCKET = os.getenv("VERTEX_BUCKET", "your-gcs-bucket")

# SQL Configuration
TABLE_NAME = os.getenv("TABLE_NAME", "your-table")
SQL_DATABASE_NAME = os.getenv("SQL_DATABASE_NAME", "your-database")
SQL_USER = os.getenv("SQL_USER", "db-user")
SQL_PASSWORD = os.getenv("SQL_PASSWORD", "your-password")
SQL_INSTANCE_CONNECTION_NAME = os.getenv("SQL_INSTANCE_CONNECTION_NAME", "project-id:region:instance-name")
POOL_SIZE = int(os.getenv("POOL_SIZE", "50"))

# Concurrency Configuration
COOLDOWN_SECONDS_DEFAULT = float(os.getenv("COOLDOWN_SECONDS", "1.0"))
MAX_RETRIES_DEFAULT = int(os.getenv("MAX_RETRIES", "8"))
BATCH_TIMEOUT_HOURS_DEFAULT = int(os.getenv("BATCH_TIMEOUT_HOURS", "3"))

# Round-robin Configuration
ROUND_ROBIN_ENABLED_DEFAULT = os.getenv("ROUND_ROBIN_ENABLED", "false").lower() == "true"
ROUND_ROBIN_OPTIONS_DEFAULT: List[str] = [
    "us-central1",
    "us-west1",
    "us-east1",
    "us-west4",
    "us-east4",
    "us-east5",
    "us-south1",
]

# Test Configuration
TEST_PROJECT_ID = os.getenv("TEST_PROJECT_ID", GCP_PROJECT_ID)
TEST_REGION = os.getenv("TEST_REGION", GCP_REGION)
TEST_BUCKET = os.getenv("TEST_BUCKET", VERTEX_BUCKET)

def reload_config():
    """
    Reload all configuration values from environment variables.
    This function should be called after setting new environment variables
    if the package was already imported.
    """
    global GCP_PROJECT_ID, GCP_PROJECT_NUM, GCP_REGION, VERTEX_BUCKET
    global TABLE_NAME, SQL_DATABASE_NAME, SQL_USER, SQL_PASSWORD, SQL_INSTANCE_CONNECTION_NAME
    global COOLDOWN_SECONDS_DEFAULT, MAX_RETRIES_DEFAULT, BATCH_TIMEOUT_HOURS_DEFAULT
    global ROUND_ROBIN_ENABLED_DEFAULT
    global TEST_PROJECT_ID, TEST_REGION, TEST_BUCKET

    # Google Cloud Platform Configuration
    GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-project-id")
    GCP_PROJECT_NUM = os.getenv("GCP_PROJECT_NUM", "123456789012")
    GCP_REGION = os.getenv("GCP_REGION", "us-central1")
    VERTEX_BUCKET = os.getenv("VERTEX_BUCKET", "your-gcs-bucket")

    # SQL Configuration
    TABLE_NAME = os.getenv("TABLE_NAME", "your-table")
    SQL_DATABASE_NAME = os.getenv("SQL_DATABASE_NAME", "your-database")
    SQL_USER = os.getenv("SQL_USER", "db-user")
    SQL_PASSWORD = os.getenv("SQL_PASSWORD", "your-password")
    SQL_INSTANCE_CONNECTION_NAME = os.getenv("SQL_INSTANCE_CONNECTION_NAME", "project-id:region:instance-name")

    # Concurrency Configuration
    COOLDOWN_SECONDS_DEFAULT = float(os.getenv("COOLDOWN_SECONDS", "1.0"))
    MAX_RETRIES_DEFAULT = int(os.getenv("MAX_RETRIES", "8"))
    BATCH_TIMEOUT_HOURS_DEFAULT = int(os.getenv("BATCH_TIMEOUT_HOURS", "3"))

    # Round-robin Configuration
    ROUND_ROBIN_ENABLED_DEFAULT = os.getenv("ROUND_ROBIN_ENABLED", "false").lower() == "true"

    # Test Configuration
    TEST_PROJECT_ID = os.getenv("TEST_PROJECT_ID", GCP_PROJECT_ID)
    TEST_REGION = os.getenv("TEST_REGION", GCP_REGION)
    TEST_BUCKET = os.getenv("TEST_BUCKET", VERTEX_BUCKET) 