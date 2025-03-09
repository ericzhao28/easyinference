# EasyInference

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> **A robust system for large-scale text inference using Vertex AI (Gemini).**

This repository hosts a modular framework to orchestrate large-scale batch and live inference requests to Gemini models.

## 📑 Contents

- [📦 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [⚙️ Core Functions](#%EF%B8%8F-core-functions)
- [💡 Example Usage](#-example-usage)
- [📚 Package Overview](#-package-overview)
- [🏗️ System Architecture](#%EF%B8%8F-system-architecture)
- [🔑 Key Concepts and Features](#-key-concepts-and-features)
- [🧩 Core Components](#-core-components)
- [📊 Table Schema](#-table-schema)
- [📄 License](#-license)

---

## 📦 Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/ericzhao28/easyinference.git
```

## 🚀 Quick Start

1. Set up your credentials for GCP and Vertex AI:
```bash
gcloud auth application-default login
```

2. Configure the necessary environment variables:

```bash
# Google Cloud Platform Configuration
export GCP_PROJECT_ID="your-project-id"
export GCP_PROJECT_NUM="123456789012"
export GCP_REGION="us-central1"
export VERTEX_BUCKET="your-gcs-bucket"

# SQL Configuration
export TABLE_NAME="your-table"
export SQL_DATABASE_NAME="your-database"
export SQL_USER="db-user"
export SQL_PASSWORD="your-password"
export SQL_INSTANCE_CONNECTION_NAME="project-id:region:instance-name"
export POOL_SIZE="50"

# Additional Configuration
export COOLDOWN_SECONDS="1.0"
export MAX_RETRIES="8"
export BATCH_TIMEOUT_HOURS="3"
export ROUND_ROBIN_ENABLED="false"
```

   Alternatively, you can use the provided `example.env` file:
   - Copy `example.env` to `.env`
   - Update the values in `.env` with your configuration
   - Use `python-dotenv` to load these variables in your code
   - Make sure you set your environment variables before importing `easyinference`.
     Otherwise, you should run `easyinference.reload_config()` after setting your
     environment variables.

3. Initialize the database connection:
```python
from easyinference import initialize_query_connection

# Initialize the database connection before using any inference functions
initialize_query_connection()
```

4. Import and use the package:
```python
from dotenv import load_dotenv  # pip install python-dotenv

# Load environment variables from .env file (if using this approach)
load_dotenv()

from easyinference import inference, individual_inference, run_clearing_inference, reload_config, initialize_query_connection

# Initialize the database connection
initialize_query_connection()
```

## ⚙️ Core Functions

### 1. `inference()`

<details>
<summary>Main async function for batch processing multiple datapoints</summary>

```python
async def inference(
    prompt_functions: List[Callable[[Any], str]],  # Functions that convert datapoints to prompt text
    datapoints: List[Any],                         # List of data items to process
    tags: Optional[List[str]] = None,              # Identifier tags for tracking
    duplication_indices: Optional[List[int]] = None, # Indices for running datapoints multiple times
    run_fast: bool = True,                         # If True, makes direct API calls; if False, queues for batch
    allow_failure: bool = False,                   # If True, continues after max retries with error messages
    attempts_cap: int = 8,                         # Maximum number of retry attempts
    temperature: float = 0,                        # Temperature parameter for generation
    max_output_tokens: int = 8192,                 # Maximum tokens to generate in response
    system_prompt: str = "",                       # System prompt to guide model behavior
    model: str = "publishers/google/models/gemini-1.5-flash-002", # Generative model to use
    batch_size: int = 1000,                        # Max concurrent requests or batch job size
    run_fast_timeout: float = 200,                 # Timeout in seconds for fast mode calls
    cooldown_seconds: float = 1.0,                 # Base wait time between retries
    batch_timeout_hours: int = 3,                  # Max runtime before restarting
    round_robin_enabled: bool = False,             # Whether to cycle through regions
    round_robin_options: List[str] = ["us-central1", "us-west1", "us-east1", "us-west4", "us-east4", "us-east5", "us-south1"], # Region options for cycling
    initial_histories: Optional[List[dict]] = None,   # Starting conversation histories for the inference sessions
) -> tuple[List[tuple], str]                       # Returns ([[[response 1, response 2, ...], [query 1, query 2, ...]], ... for each datapoint], launch_timestamp_tag)
```

</details>

### 2. `individual_inference()`

<details>
<summary>For processing a single datapoint through multiple prompt functions</summary>

```python
async def individual_inference(
    prompt_functions: List[Callable[[Any], str]],  # Functions that convert datapoint to prompt text
    datapoint: Any,                                # Data to process
    tags: Optional[List[str]] = None,              # Identifier tags for tracking
    optional_tags: Optional[List[str]] = None,     # Additional tags not used for lookup
    duplication_index: int = 0,                    # Index to distinguish duplicate runs
    run_fast: bool = True,                         # If True, makes direct API calls; if False, queues for batch
    allow_failure: bool = False,                   # If True, continues after max retries with error messages
    attempts_cap: int = 8,                         # Maximum number of retry attempts
    temperature: float = 0,                        # Temperature parameter for generation
    max_output_tokens: int = 8192,                 # Maximum tokens to generate in response
    system_prompt: str = "",                       # System prompt to guide model behavior
    model: str = "publishers/google/models/gemini-1.5-flash-002", # Generative model to use
    run_fast_timeout: float = 200,                 # Timeout in seconds for fast mode calls
    cooldown_seconds: float = 1.0,                 # Base wait time between retries
    round_robin_enabled: bool = False,             # Whether to cycle through regions
    round_robin_options: List[str] = ["us-central1", "us-west1", "us-east1", "us-west4", "us-east4", "us-east5", "us-south1"], # Region options for cycling
    initial_history_json: Optional[dict] = None,   # Starting conversation history for the inference session
) -> tuple[List[str], List[str]]                   # Returns [[response 1, response 2, ...], [query 1, query 2, ...]]
```

</details>

### 3. `run_clearing_inference()`

<details>
<summary>For managing batch inference jobs</summary>

```python
async def run_clearing_inference(
    tag: str,                                      # Unique identifier tag for the batch
    batch_size: int,                               # Maximum number of requests per batch job
    run_batch_jobs: bool,                          # Whether to launch new batch jobs
    batch_timeout_hours: int = 3                   # Maximum runtime hours before restarting
) -> None
```

</details>

### 4. `reload_config()`

<details>
<summary>For reloading the config after setting environment variables</summary>

```python
def reload_config() -> None
```

</details>

## 💡 Example Usage

```python
import asyncio
from dotenv import load_dotenv
from easyinference import inference, reload_config, initialize_query_connection

load_dotenv()
reload_config()

# Initialize the database connection before using any inference functions
initialize_query_connection()

async def process_data():
    # Define data and prompt function
    datapoints = [
        {"text": "What is machine learning?"},
        {"text": "Explain neural networks"}
    ]
    
    def create_prompt(dp):
        return f"Please explain: {dp['text']}"
    
    # Run inference
    results, timestamp = await inference(
        prompt_functions=[create_prompt],
        datapoints=datapoints,
        tags=["explanation", "v1"],
        run_fast=True
    )
    
    # Process results
    first_datapoint_result, second_datapoint_result = results
    for i, (response, query) in enumerate(first_datapoint_result):
        print(f"Query: {query}")
        print(f"Response: {response}")
    
    return results

# Run the async function
results = asyncio.run(process_data())
```

---

## 📚 Package Overview

**Goal:** We provide a **scalable** and **robust** pipeline to handle:

- ✨ **Conversation-based** inference requests to Gemini models
- ✨ **Failure tracking** and **retry** logic to ensure stable operation
- ✨ **Asynchronous or synchronous** methods for generating text from the model

We accomplish this by:

1. **Storing** every inference "step" in a **PostgreSQL** table, which captures the query text, model parameters, conversation history, and final responses (or errors).  
2. **Separating** "fast" live calls vs. "slow" batch-based calls.  
3. **Monitoring** the status of batch inference jobs, so you can schedule or restart them if they take too long.  
4. **Allowing** different usage patterns: single datapoint or bulk processing, with multi-prompt sequences, concurrency caps, and re-tries.

---

## 🏗️ System Architecture

```
 ┌───────────────────┐
 │  Your Application │
 └─────────┬─────────┘
           │
           ├─────────────────┐
           │                 │
           ▼                 ▼
┌─────────────────────┐    ┌─────────────────────┐
│Individual Inference │    │      Inference      │
│      (Fast)         │◀---│                     │
└──────────┬──────────┘    └──────────┬──────────┘
           │                          │
           │                          ▼
           │               ┌─────────────────────────┐
           │               │     Batch Clearing      │
           │               │      (monitoring)       │
           │               │                         │
           │               └──────────┬──────────────┘
           │                          │
           ▼                          ▼
┌────────────────────────┐  ┌────────────────────────┐
│ Vertex AI (Gemini API) │  │ Vertex AI (Gemini API) │
│     (Live Calls)       │  │     (Batch Job)        │
└────────────────────────┘  └────────────────────────┘
           │                          │
           └──────────┐   ┌───────────┘
                      ▼   ▼
               ┌────────────────────┐
               │     PostgreSQL     │
               │    Master Table    │
               └────────────────────┘
```

1. **`Individual Inference`** manages a single datapoint and a **sequence** of prompts.  
2. **`Inference`** is a bulk orchestrator that calls individual inference on multiple datapoints.  
3. **`Clearing Inference`** takes unprocessed/failed rows and triggers additional attempts (live or batch). It also monitors batch jobs and handles timeouts.

---

## 🔑 Key Concepts and Features

### 1. Conversation History 💬

Stored in PostgreSQL under `history_json` as a JSON object:

```json
{
  "history": [
    {"role": "user", "parts": {"text": "Hello, how are you?"}},
    {"role": "model", "parts": {"text": "I am fine. How can I help?"}}
  ]
}
```

> This helps **Vertex** continue the same conversation context across multiple queries without duplication.

### 2. Generation Parameters ⚙️

Stored under `generation_params_json` (JSON):

```json
{
  "temperature": 0.7,
  "max_output_tokens": 8192,
  "system_prompt": "You are a helpful assistant..."
}
```

### 3. Duplication Index 🔄

An integer marking whether a row is an **exact** duplicate of an earlier row (e.g., a re-run). Defaults to 0.

### 4. Tags 🏷️

A list of strings (alphabetically sorted) representing categories or labels applied to a request (e.g. `["admin", "api-v1"]`).  
This can help in filtering or grouping by usage scenario.

### 5. Request Cause 📡

Either `"intentional"` (explicit user request) or `"backup"` (an automatic fallback).

### 6. Status and Failure Counts 📊

- `Last Status` can be `"PENDING"`, `"RUNNING"`, `"FAILED"`, `"SUCCEEDED"`, `"WAITING"`.  
- `Failure Count` tracks how many attempts have failed so far, and `Attempts Cap` sets the max allowed.

### 7. Content Hash 🔒

A hash of `(Model, History, Query, GenerationParams, DuplicationIndex)` for deduplicating or resuming.

### 8. Modes ⚡

- **Run Fast**: calls the Vertex API directly, returning the result in real-time.  
- **Run Slow**: queues up the request for a **batch job**. The `run_clearing_inference` function handles job submission and monitoring.

### 9. Initialization ⚡

Before using any inference functions, you must initialize the database connection by calling:

```python
from easyinference import initialize_query_connection

initialize_query_connection()
```

This sets up the necessary connections to the PostgreSQL database for tracking inference requests.

---

## 🧩 Core Components

### 1. `cloudsql/schema.py` 📝

- Defines a `ConvoRow` data class that mirrors each column in the table.  
- Enumerations for `RequestStatus` and `RequestCause`.

### 2. `cloudsql/table_utils.py` 🔧

- Helper functions to **insert**, **update**, or **read** rows from PostgreSQL.  
- Includes concurrency checks so you don't overwrite a "SUCCEEDED" row with "FAILED."
- Functions for connecting to PostgreSQL, creating tables, and querying data.

### 3. `inference.py` 🧠

- Implements both `individual_inference` and `inference` functions
- Contains `run_chat_inference_async` for "fast" calls with built-in retry/backoff
- Implements `run_clearing_inference` that handles both batch submission and monitoring
- Manages the logic for deduplicating (by content hash), incrementing failure counts, and handling partial successes

### 4. `config.py` ⚙️

- Configuration settings for database connections, retry logic, and batch operations.
- Contains defaults for constants like `MAX_RETRIES`, `BATCH_TIMEOUT_HOURS`, and various connection parameters.

---

## 📊 Table Schema

Your **master PostgreSQL table** has the following columns:

<table>
  <thead>
    <tr>
      <th>Column Name</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>row_id</strong></td>
      <td>INTEGER</td>
      <td>Auto-incrementing primary key</td>
    </tr>
    <tr>
      <td><strong>content_hash</strong></td>
      <td>STRING</td>
      <td>SHA-256 hash of key fields for deduplication</td>
    </tr>
    <tr>
      <td><strong>history_json</strong></td>
      <td>JSON</td>
      <td>JSON storing prior conversation messages in a format with the key "history"</td>
    </tr>
    <tr>
      <td><strong>query</strong></td>
      <td>STRING</td>
      <td>User's latest query that needs a response</td>
    </tr>
    <tr>
      <td><strong>model</strong></td>
      <td>STRING</td>
      <td>Full path of the model (e.g. <code>"publishers/google/models/gemini-1.5-flash-002"</code>)</td>
    </tr>
    <tr>
      <td><strong>generation_params_json</strong></td>
      <td>JSON</td>
      <td>JSON storing generation settings, e.g. <code>{"temperature":0.7,"max_output_tokens":8192,"system_prompt":"..."}</code></td>
    </tr>
    <tr>
      <td><strong>duplication_index</strong></td>
      <td>INTEGER</td>
      <td>Used to mark re-runs or explicit duplicates. Defaults to 0</td>
    </tr>
    <tr>
      <td><strong>tags</strong></td>
      <td>ARRAY(STRING)</td>
      <td>A sorted list of tags (e.g. <code>["api-v1","testing"]</code>)</td>
    </tr>
    <tr>
      <td><strong>request_cause</strong></td>
      <td>STRING</td>
      <td><code>"intentional"</code> or <code>"backup"</code>. Uses the <code>RequestCause</code> enum</td>
    </tr>
    <tr>
      <td><strong>request_timestamp</strong></td>
      <td>STRING</td>
      <td>ISO 8601 timestamp (<code>"2025-02-25T12:34:56Z"</code>)</td>
    </tr>
    <tr>
      <td><strong>access_timestamps</strong></td>
      <td>ARRAY(STRING)</td>
      <td>List of ISO 8601 timestamps of each read/update</td>
    </tr>
    <tr>
      <td><strong>attempts_metadata_json</strong></td>
      <td>ARRAY(JSON)</td>
      <td>JSON array of prior attempts, storing batch info and error messages</td>
    </tr>
    <tr>
      <td><strong>response_json</strong></td>
      <td>JSON</td>
      <td>JSON containing the final successful response if available. Example: <code>{"text":"...response..."}</code></td>
    </tr>
    <tr>
      <td><strong>current_batch</strong></td>
      <td>STRING</td>
      <td>The ID of any currently running batch job. Can be <code>NULL</code></td>
    </tr>
    <tr>
      <td><strong>last_status</strong></td>
      <td>STRING</td>
      <td><code>"PENDING"</code>, <code>"RUNNING"</code>, <code>"FAILED"</code>, <code>"SUCCEEDED"</code>, or <code>"WAITING"</code></td>
    </tr>
    <tr>
      <td><strong>failure_count</strong></td>
      <td>INTEGER</td>
      <td>How many times this row has failed so far</td>
    </tr>
    <tr>
      <td><strong>attempts_cap</strong></td>
      <td>INTEGER</td>
      <td>The maximum number of times we will re-try</td>
    </tr>
    <tr>
      <td><strong>notes</strong></td>
      <td>STRING</td>
      <td>Optional free-text notes</td>
    </tr>
    <tr>
      <td><strong>insertion_timestamp</strong></td>
      <td>TIMESTAMP</td>
      <td>When the row was inserted into the database</td>
    </tr>
  </tbody>
</table>

### Content Hash 🔐

- **SHA-256** over the combination of `(Model, History, Query, GenerationParams, DuplicationIndex)`.  
- Ensures we don't re-run the same content multiple times unless we want to.

### Tagging 🏷️

- A query can have tags like `["api-v1","admin-request"]`. The system enforces that the tag list is **alphabetically sorted**.  
- For batch mode, a timestamp tag is automatically added for tracking.

---

## 📄 License

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This project is provided under the **MIT License**.  

Feel free to modify or extend the code to suit your deployment and usage requirements.
