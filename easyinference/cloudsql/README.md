# CloudSQL Module

The `cloudsql` module (despite its name) now handles all interactions with PostgreSQL, including:

- Defining the schema for inference request tracking
- Creating and managing PostgreSQL tables
- Inserting new inference requests
- Updating request statuses and results
- Querying for requests by various criteria
- Ensuring data consistency and integrity
- Batch processing support

## Key Components

### `schema.py`

This file defines the data structures that represent rows in the PostgreSQL table:

- **`RequestStatus` Enum**: Defines the possible states of an inference request:
  - `PENDING`: Request is waiting to be processed
  - `RUNNING`: Request is currently being processed
  - `FAILED`: Request processing failed
  - `SUCCEEDED`: Request was successfully processed
  - `WAITING`: Request is queued for batch processing

- **`RequestCause` Enum**: Indicates why a request was created:
  - `INTENTIONAL`: Explicitly requested by a user
  - `BACKUP`: Automatically created as a fallback

- **`ConvoRow` Class**: Represents a single row in the database with attributes:
  - `row_id`: Database-generated primary key
  - `content_hash`: Unique identifier (SHA-256 hash of key fields)
  - `history_json`: Conversation history in Vertex AI format
  - `query`: The current user query
  - `model`: The AI model to use
  - `generation_params_json`: Parameters for text generation
  - `duplication_index`: Counter for intentional duplicates
  - `tags`: List of categorization tags
  - `request_cause`: Why the request was created
  - `request_timestamp`: When the request was created
  - `access_timestamps`: When the row was accessed
  - `attempts_metadata_json`: Details of previous attempts
  - `response_json`: The successful response (if any)
  - `current_batch`: ID of any running batch job
  - `last_status`: Current status of the request
  - `failure_count`: Number of failed attempts
  - `attempts_cap`: Maximum allowed attempts
  - `notes`: Optional free-text notes

### `table_utils.py`

This file provides utility functions for interacting with the PostgreSQL table:

- **`connect_with_connector()`**: Establishes a connection pool to Cloud SQL
- **`insert_row(row)`**: Inserts a new row or updates an existing row
- **`find_existing_row_by_content_hash(content_hash, tag_subset, tag_superset)`**: Finds a row by content hash with optional tag filtering
- **`get_batch_ids_by_status_and_tag(status_list, tag)`**: Gets unique batch IDs for rows with specified status and tag
- **`get_rows_by_status_and_tag_and_batch(status_list, tag, current_batch)`**: Retrieves rows by status, tag, and batch
- **`stream_rows_by_status_and_tag(status_list, tag, batch_size)`**: Streams rows in batches to handle large datasets
- **`create_table_if_not_exists()`**: Creates the PostgreSQL table if it doesn't exist
- **`refresh_row(row)`**: Fetches the current state of a row from the database

## Table Schema

The PostgreSQL table has the following schema:

| Column Name | Type | Description |
|-------------|------|-------------|
| **row_id** | INTEGER | Auto-incrementing primary key |
| **content_hash** | STRING | Unique ID based on content hash |
| **history_json** | JSON | Conversation history |
| **query** | STRING | User's latest query |
| **model** | STRING | Full path of the model |
| **generation_params_json** | JSON | Generation settings |
| **duplication_index** | INTEGER | Counter for intentional duplicates |
| **tags** | ARRAY(STRING) | List of categorization tags |
| **request_cause** | STRING | "intentional" or "backup" |
| **request_timestamp** | STRING | ISO 8601 timestamp of request creation |
| **access_timestamps** | ARRAY(STRING) | List of ISO 8601 timestamps of each access |
| **attempts_metadata_json** | ARRAY(JSON) | List of prior attempts |
| **response_json** | JSON | The final successful response |
| **current_batch** | STRING | ID of any currently running batch job |
| **last_status** | STRING | Current status |
| **failure_count** | INTEGER | Number of failed attempts |
| **attempts_cap** | INTEGER | Maximum number of retry attempts allowed |
| **notes** | STRING | Optional free-text notes |
| **insertion_timestamp** | TIMESTAMP | When the row was inserted |

## Usage Examples

### Creating the Table

```python
from easyinference.cloudsql import table_utils

# Create the table if it doesn't exist
table_utils.create_table_if_not_exists()
```

### Creating and Inserting a Row

```python
from easyinference.cloudsql.schema import ConvoRow, RequestStatus, RequestCause

# Create a new row
row = ConvoRow(
    history_json={"contents": [{"role": "user", "parts": {"text": "Hello"}}]},
    query="What is machine learning?",
    model="publishers/google/models/gemini-1.5-pro-002",
    generation_params_json={"temperature": 0.7, "max_tokens": 512},
    tags=["api-v1", "education"],
    duplication_index=0,
    request_cause=RequestCause.INTENTIONAL,
    last_status=RequestStatus.WAITING,
    attempts_cap=3
)

# Insert the row into PostgreSQL
from easyinference.cloudsql import table_utils
row_id = table_utils.insert_row(row)
print(f"Inserted row with ID: {row_id}")
```

### Finding Rows by Status, Tag, and Batch

```python
from easyinference.cloudsql import table_utils
from easyinference.cloudsql.schema import RequestStatus

# Get all waiting and running rows with the "api-v1" tag in a specific batch
rows = table_utils.get_rows_by_status_and_tag_and_batch(
    status_list=[RequestStatus.WAITING, RequestStatus.RUNNING],
    tag="api-v1",
    current_batch="batch-123"
)

# Process the rows
for row in rows:
    print(f"Content hash: {row.content_hash}, Status: {row.last_status.value}")
```

### Streaming Large Result Sets

```python
from easyinference.cloudsql import table_utils
from easyinference.cloudsql.schema import RequestStatus

# Stream all failed rows with the "api-v1" tag in batches of 100
batches = table_utils.stream_rows_by_status_and_tag(
    status_list=[RequestStatus.FAILED],
    tag="api-v1",
    batch_size=100
)

# Process each batch
for batch in batches:
    print(f"Processing batch of {len(batch)} rows")
    for row in batch:
        # Process each row
        print(f"Processing row with content hash: {row.content_hash}")
```

### Refreshing a Row

```python
from easyinference.cloudsql import table_utils

# Get the latest state of a row
updated_row = table_utils.refresh_row(row)
if updated_row:
    print(f"Current status: {updated_row.last_status.value}")
```

## Implementation Details

### Content Hash Generation

The `content_hash` property of a `ConvoRow` is automatically generated as a SHA-256 hash of the following fields:
- model
- history_json
- query
- generation_params_json
- duplication_index

This hash serves as a unique identifier and enables efficient deduplication of identical requests.

### Row Versioning

The system maintains multiple versions of a row with the same content hash but different `insertion_timestamp` values. Query functions like `find_existing_row_by_content_hash` and `get_rows_by_status_and_tag_and_batch` automatically return only the most recent version of each row.
