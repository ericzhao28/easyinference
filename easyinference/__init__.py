"""Easy inference using Google's Vertex AI and Gemini models."""

from dotenv import load_dotenv

load_dotenv()

import vertexai
from . import config

vertexai.init(project=config.GCP_PROJECT_ID, location=config.GCP_REGION)

from . import cloudsql
from .cloudsql.table_utils import initialize_query_connection
from .inference import (
    inference,
    individual_inference,
    run_chat_inference_async,
    run_clearing_inference,
)

def reload_config():
    """
    Reload all configuration values from environment variables.
    This function should be called after setting new environment variables
    if the package was already imported.
    """
    config.reload_config()
    # Reinitialize Vertex AI with new configuration
    vertexai.init(project=config.GCP_PROJECT_ID, location=config.GCP_REGION)

__version__ = "1.0.0"

__all__ = [
    "inference",
    "individual_inference",
    "run_chat_inference_async",
    "run_clearing_inference",
    "reload_config",
    "initialize_query_connection",
]
