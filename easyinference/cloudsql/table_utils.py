"""
table_utils.py

Provides helper functions to interact with the Google Cloud SQL database, including:
 - Insert new rows
 - Find existing rows by content hash
 - Retrieve rows by status and tag
"""

from typing import Optional, List, Iterable
import logging
import asyncpg
from datetime import datetime
import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy import text, MetaData, Table, Column, String, Integer, TIMESTAMP, JSON
from sqlalchemy.dialects.postgresql import ARRAY
from google.cloud.sql.connector import Connector, IPTypes, create_async_connector
from contextlib import asynccontextmanager, contextmanager
import json  # Add this import at the top of the file
import asyncio # Import asyncio for semaphore


from easyinference.cloudsql.schema import ConvoRow, RequestStatus
import easyinference.config as config

# Configure logging
logger = logging.getLogger(__name__)
pool = None
db_semaphore = None # Initialize semaphore here

async def init_connection_pool(connector: Connector) -> AsyncEngine:
    """
    Initializes a connection pool for a Cloud SQL instance of Postgres.

    Uses the Cloud SQL Python Connector package.
    """
    def getconn() -> asyncpg.Connection:
        conn: asyncpg.Connection = connector.connect_async(
            config.SQL_INSTANCE_CONNECTION_NAME,
            "asyncpg",
            user=config.SQL_USER,
            password=config.SQL_PASSWORD,
            db=config.SQL_DATABASE_NAME,
            ip_type=IPTypes.PUBLIC,
        )
        return conn

    pool = create_async_engine(
        "postgresql+asyncpg://",
        async_creator=getconn,
        pool_size=config.POOL_SIZE,
        max_overflow=int(2 * config.POOL_SIZE),
        pool_timeout=60,
        pool_recycle=300
    )
    return pool

# Create the engine once as a module-level variable
async def initialize_query_connection():
    global pool, db_semaphore # Add db_semaphore to global scope
    connector = await create_async_connector()
    pool = await init_connection_pool(connector)
    db_semaphore = asyncio.Semaphore(config.POOL_SIZE) # Initialize semaphore with a limit, adjust as needed.

metadata = MetaData()

@asynccontextmanager
async def get_connection():
    """Async context manager for database connections"""
    if pool is None:
        raise ValueError("Connection pool not initialized. Run initialize_query_connection() first.")
    async with db_semaphore: # Acquire semaphore here
        async with pool.connect() as connection:
            yield connection


async def insert_row(row: ConvoRow) -> Optional[int]:
    """
    Insert a new row into the SQL database or replace an existing row if there's one with
    the same row_id.

    Args:
        row (ConvoRow): The row object to insert.

    Returns:
        Optional[int]: The primary key (row_id) of the inserted row, or None if an error occurred.

    Raises:
        Exception if the insert fails for any reason.
    """
    logger.info(f"Attempting to insert or replace row with content hash: {row.content_hash}")
    try:
        row_dict = row.to_dict()
        row_dict["insertion_timestamp"] = datetime.now()
        
        # Check if we have a row_id to potentially update
        has_row_id = "row_id" in row_dict and row_dict["row_id"] is not None
        row_id_value = row_dict.get("row_id")
        
        # Remove row_id if present for insertion - we'll add it back for update
        if "row_id" in row_dict:
            del row_dict["row_id"]
        
        # Convert JSON objects to JSON strings for asyncpg
        row_dict["history_json"] = json.dumps(row_dict["history_json"])
        row_dict["generation_params_json"] = json.dumps(row_dict["generation_params_json"])
        row_dict["attempts_metadata_json"] = [json.dumps(t) for t in row_dict["attempts_metadata_json"]]
        row_dict["response_json"] = json.dumps(row_dict["response_json"])
        
        # Insert or replace based on row_id
        async with get_connection() as conn:
            if has_row_id:
                # Check if a row with this row_id exists
                check_row_id_query = text(f"""
                    SELECT row_id FROM "{config.TABLE_NAME}" WHERE row_id = :row_id
                """)
                result = await conn.execute(check_row_id_query, {"row_id": row_id_value})
                existing_by_row_id = result.fetchone()
                
                if existing_by_row_id:
                    # Update the specific row with matching row_id
                    logger.info(f"Found existing row with row_id {row_id_value}. Will replace it.")
                    update_query = text(f"""
                        UPDATE "{config.TABLE_NAME}" SET
                            content_hash = :content_hash, 
                            history_json = :history_json,
                            query = :query,
                            model = :model,
                            generation_params_json = :generation_params_json,
                            duplication_index = :duplication_index,
                            tags = :tags,
                            request_cause = :request_cause,
                            request_timestamp = :request_timestamp,
                            access_timestamps = :access_timestamps,
                            attempts_metadata_json = :attempts_metadata_json,
                            response_json = :response_json,
                            current_batch = :current_batch,
                            last_status = :last_status,
                            failure_count = :failure_count,
                            attempts_cap = :attempts_cap,
                            notes = :notes,
                            insertion_timestamp = :insertion_timestamp
                        WHERE row_id = :row_id
                        RETURNING row_id
                    """)
                    
                    # Add row_id back to parameters for the UPDATE
                    row_dict["row_id"] = row_id_value
                    result = await conn.execute(update_query, row_dict)
                    row_id = result.scalar_one()
                else:
                    # Row_id provided but not found - insert new row with default row_id
                    row_id = await _insert_new_row(conn, row_dict)
            else:
                # No row_id provided - simple insert
                row_id = await _insert_new_row(conn, row_dict)
            
            await conn.commit()
        
        logger.info(f"Successfully inserted/replaced row with content hash: {row.content_hash}, assigned row_id: {row_id}")
        # Update the row object with the new row_id
        row.row_id = row_id
        return row_id
    except Exception as e:
        error_message = f"Failed to insert/replace row {row.content_hash}: {e}"
        logger.error(error_message, exc_info=True)
        raise


async def _insert_new_row(conn, row_dict) -> int:
    """Helper function to insert a new row and return its row_id"""
    insert_query = text(f"""
        INSERT INTO "{config.TABLE_NAME}" (
            content_hash, history_json, query, model, generation_params_json, 
            duplication_index, tags, request_cause, request_timestamp, 
            access_timestamps, attempts_metadata_json, response_json, 
            current_batch, last_status, failure_count, attempts_cap, 
            notes, insertion_timestamp
        ) VALUES (
            :content_hash, :history_json, :query, :model, :generation_params_json,
            :duplication_index, :tags, :request_cause, :request_timestamp,
            :access_timestamps, :attempts_metadata_json, :response_json,
            :current_batch, :last_status, :failure_count, :attempts_cap,
            :notes, :insertion_timestamp
        ) RETURNING row_id
    """)
    
    result = await conn.execute(insert_query, row_dict)
    return result.scalar_one()


async def find_existing_row_by_content_hash(content_hash: str, tag_subset: Optional[List[str]] = None, tag_superset: Optional[List[str]] = None) -> Optional[ConvoRow]:
    """
    Search the SQL database for an existing row matching the given content hash.
    Optionally, enforce that:
    - The row's tags contain all tags in tag_subset (if provided)
    - The row's tags are contained within tag_superset (if provided)
    If multiple rows match the criteria, the most recent row is returned.

    Args:
        content_hash (str): The content hash of the row.
        tag_subset (Optional[List[str]]): If provided, filter to rows whose tags contain all these tags.
        tag_superset (Optional[List[str]]): If provided, filter to rows whose tags are all contained in this list.

    Returns:
        Optional[ConvoRow]: The found row if it exists, otherwise None.
    """
    try:
        # Start building the query
        query_str = f"""
        WITH ranked_rows AS (
            SELECT
                *,
                ROW_NUMBER() OVER(PARTITION BY content_hash ORDER BY insertion_timestamp DESC) as row_num
            FROM "{config.TABLE_NAME}"
            WHERE content_hash = :content_hash
        """
        
        # Add tag filtering if needed
        params = {"content_hash": content_hash}
        if tag_subset and len(tag_subset) > 0:
            query_str += " AND tags @> :tag_subset"
            params["tag_subset"] = tag_subset
        
        if tag_superset and len(tag_superset) > 0:
            query_str += " AND tags <@ :tag_superset"
            params["tag_superset"] = tag_superset
        
        # Complete the query
        query_str += """
        )
        SELECT * FROM ranked_rows WHERE row_num = 1
        """
        
        async with get_connection() as conn:
            result = await conn.execute(text(query_str), params)
            row = result.fetchone()
            
        if row:
            # Convert row to dictionary
            row_dict = dict(row._mapping)
            
            return ConvoRow.from_dict(row_dict)
        return None
    except Exception as e:
        logger.error(f"Failed to query existing row: {e}")
        raise


async def get_batch_ids_by_status_and_tag(status_list: List[RequestStatus], tag: Optional[str] = None) -> set[str]:
    """
    Retrieve the set of unique current_batch values from the most recent rows (by content_hash) that match any of the provided statuses and, optionally, contain a specific tag.

    This function efficiently fetches batch IDs by only considering the latest entry for each content hash,
    addressing the issue of redundant older entries in the table.

    Args:
        status_list (List[RequestStatus]): The statuses to match.
        tag (Optional[str]): If provided, filter rows that contain this tag.

    Returns:
        set[str]: A set of unique current_batch values from the most recent matching rows.
                 Returns an empty set if no matching rows or no current_batch values are found.
    """
    log_message = f"Fetching unique current_batch values from most recent rows by statuses: {status_list}"
    if tag:
        log_message += f", tag: {tag}"
    logger.info(log_message)

    if not isinstance(status_list, list):
        raise ValueError(f"Expected status_list to be a list, but got {type(status_list)}.")
    if tag is not None and not isinstance(tag, str):
        raise ValueError(f"Expected tag to be a string or None, but got {type(tag)}.")
    
    try:
        # Convert status_list to string values for the query
        status_values = [status.value for status in status_list]
        
        # Start building the query
        query_str = f"""
        WITH ranked_rows AS (
            SELECT
                content_hash,
                current_batch,
                ROW_NUMBER() OVER(PARTITION BY content_hash ORDER BY insertion_timestamp DESC) as row_num
            FROM "{config.TABLE_NAME}"
            WHERE last_status = ANY(:status_list)
        """
        
        # Add tag filtering if needed
        params = {"status_list": status_values}
        if tag:
            query_str += " AND :tag = ANY(tags)"
            params["tag"] = tag
        
        # Complete the query
        query_str += """
        )
        SELECT DISTINCT current_batch
        FROM ranked_rows
        WHERE row_num = 1
        AND current_batch IS NOT NULL
        """
        
        async with get_connection() as conn:
            result = await conn.execute(text(query_str), params)
            results = result.fetchall()
        
        unique_batches = {row[0] for row in results if row[0] is not None}
        logger.info(f"Retrieved {len(unique_batches)} unique current_batch values from most recent rows with statuses: {status_list} and tag: {tag}")
        return unique_batches
    except Exception as e:
        logger.error(e)
        raise


async def get_rows_by_status_and_tag_and_batch(status_list: List[RequestStatus], tag: Optional[str] = None, current_batch: Optional[str] = None) -> List[ConvoRow]:
    """
    Retrieve rows that match any of the provided statuses and, optionally, contain a specific tag and/or current_batch,
    ignoring older rows if a more recent row with the same content_hash and tags exists.

    Args:
        status_list (List[RequestStatus]): The statuses to match.
        tag (Optional[str]): If provided, filter rows that contain this tag.
        current_batch (Optional[str]): If provided, filter rows that match this current_batch value.

    Returns:
        List[ConvoRow]: The most recent rows that match the criteria, one per content_hash and tags combination.
    """
    log_message = f"Fetching most recent rows by statuses: {status_list}"
    if tag:
        log_message += f", tag: {tag}"
    if current_batch:
        log_message += f", current_batch: {current_batch}"
    logger.info(log_message)

    if not isinstance(status_list, list):
        raise ValueError(f"Expected status_list to be a list, but got {type(status_list)}.")
    if tag is not None and not isinstance(tag, str):
        raise ValueError(f"Expected tag to be a string or None, but got {type(tag)}.")
    if current_batch is not None and not isinstance(current_batch, str):
        raise ValueError(f"Expected current_batch to be a string or None, but got {type(current_batch)}.")

    try:
        # Convert status_list to string values for the query
        status_values = [status.value for status in status_list]
        
        # Start building the query
        query_str = f"""
        WITH ranked_rows AS (
            SELECT
                *,
                ROW_NUMBER() OVER(PARTITION BY content_hash ORDER BY insertion_timestamp DESC) as row_num
            FROM "{config.TABLE_NAME}"
            WHERE last_status = ANY(:status_list)
        """
        
        # Add tag and batch filtering if needed
        params = {"status_list": status_values}
        if tag:
            query_str += " AND :tag = ANY(tags)"
            params["tag"] = tag
        
        if current_batch:
            query_str += " AND current_batch = :current_batch"
            params["current_batch"] = current_batch
        
        # Complete the query
        query_str += """
        )
        SELECT * FROM ranked_rows WHERE row_num = 1
        """
        
        async with get_connection() as conn:
            result = await conn.execute(text(query_str), params)
            results = result.fetchall()
        
        # Convert rows to ConvoRow objects
        bq_rows = []
        for row in results:
            # Convert each row to a dictionary
            row_dict = dict(row._mapping)
            
            bq_rows.append(ConvoRow.from_dict(row_dict))
        
        logger.info(f"Retrieved {len(bq_rows)} most recent rows with statuses: {status_list}, tag: {tag}, current_batch: {current_batch}")
        return bq_rows
    except Exception as e:
        logger.error(f"Failed to retrieve rows by status, tag and batch. Statuses: {status_list}, Tag: {tag}, Current Batch: {current_batch}. Error: {e}")
        return []


async def stream_rows_by_status_and_tag(status_list: List[RequestStatus], tag: Optional[str] = None, batch_size: int = 1000) -> Iterable[List[ConvoRow]]:
    """
    Retrieve rows that match any of the provided statuses and, optionally, contain a specific tag,
    in pages to efficiently handle large datasets, ignoring older rows with the same content_hash and tags.

    Args:
        status_list (List[RequestStatus]): The statuses to match.
        tag (Optional[str]): If provided, filter rows that contain this tag.
        batch_size (int): The number of rows to retrieve per page.

    Returns:
        Iterable[List[ConvoRow]]: An iterable of lists of ConvoRow objects that match the criteria.
    """
    log_message = f"Fetching most recent rows by statuses: {status_list}"
    if tag:
        log_message += f", tag: {tag}"
    logger.info(log_message)

    if not isinstance(status_list, list):
        raise ValueError(f"Expected status_list to be a list, but got {type(status_list)}.")
    if tag is not None and not isinstance(tag, str):
        raise ValueError(f"Expected tag to be a string or None, but got {type(tag)}.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"Expected batch_size to be a positive integer, but got {batch_size}.")

    try:
        # Convert status_list to string values for the query
        status_values = [status.value for status in status_list]
        
        # Start building the query
        query_str = f"""
        WITH ranked_rows AS (
            SELECT
                *,
                ROW_NUMBER() OVER(PARTITION BY content_hash ORDER BY insertion_timestamp DESC) as row_num
            FROM "{config.TABLE_NAME}"
            WHERE last_status = ANY(:status_list)
        """
        
        # Add tag filtering if needed
        params = {"status_list": status_values}
        if tag:
            query_str += " AND :tag = ANY(tags)"
            params["tag"] = tag
        
        # Complete the query
        query_str += """
        )
        SELECT * FROM ranked_rows WHERE row_num = 1
        """
        
        # Execute query with pagination
        async with get_connection() as conn:
            result = await conn.execute(text(query_str), params)
            
            # Process rows in batches
            current_batch = []
            retrieved_count = 0
            
            for row in result:
                # Convert row to a dictionary
                row_dict = dict(row._mapping)
                
                current_batch.append(ConvoRow.from_dict(row_dict))
                retrieved_count += 1
                
                # When we reach the batch size, yield the entire batch
                if len(current_batch) >= batch_size:
                    yield current_batch
                    current_batch = []  # Reset the batch
            
            # Don't forget to yield any remaining items in the last batch
            if current_batch:
                yield current_batch

        logger.info(f"Retrieved {retrieved_count} most recent rows with statuses: {status_list}, tag: {tag} in batches of size: {batch_size}")

    except Exception as e:
        logger.error(f"Failed to retrieve rows by status and tag. Statuses: {status_list}, Tag: {tag}. Error: {e}")
        yield []  # return empty list in case of error


async def create_table_if_not_exists() -> None:
    """
    Creates the SQL table if it doesn't already exist.
    Uses the schema defined in the README.

    Raises:
        Exception if table creation fails.
    """
    try:
        logger.info(f"Checking if table '{config.TABLE_NAME}' exists")
        
        # Check if table exists using asyncpg-compatible app
        async with get_connection() as conn:
            # Query the information schema to check if table exists
            query = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = :table_name
                )
            """)
            result = await conn.execute(query, {"table_name": config.TABLE_NAME})
            table_exists = result.scalar()
            
            if table_exists:
                logger.info(f"Table '{config.TABLE_NAME}' already exists")
                return
        
        logger.info(f"Table '{config.TABLE_NAME}' does not exist, creating now")
        
        # Define the table
        conversations = Table(
            config.TABLE_NAME,
            metadata,
            Column("row_id", Integer, primary_key=True, autoincrement=True),
            Column("content_hash", String, nullable=False, index=True),
            Column("history_json", JSON, nullable=False),
            Column("query", String, nullable=False),
            Column("model", String, nullable=False),
            Column("generation_params_json", JSON, nullable=False),
            Column("duplication_index", Integer, nullable=False),
            Column("tags", ARRAY(String), nullable=True),
            Column("request_cause", String, nullable=False),
            Column("request_timestamp", String, nullable=False),
            Column("access_timestamps", ARRAY(String), nullable=True),
            Column("attempts_metadata_json", ARRAY(JSON), nullable=False),
            Column("response_json", JSON, nullable=False),
            Column("current_batch", String, nullable=True),
            Column("last_status", String, nullable=False),
            Column("failure_count", Integer, nullable=False),
            Column("attempts_cap", Integer, nullable=False),
            Column("notes", String, nullable=False),
            Column("insertion_timestamp", TIMESTAMP, nullable=False),
        )

        # Create the table using SQLAlchemy's create_all
        async with get_connection() as conn:
            await conn.run_sync(lambda conn: metadata.create_all(conn, tables=[conversations]))
        
        logger.info(f"Successfully created table '{config.TABLE_NAME}'")
    
    except Exception as e:
        logger.error(f"Failed to create table {config.TABLE_NAME}: {e}")


async def refresh_row(row: ConvoRow) -> Optional[ConvoRow]:
    """
    Fetch the current state of a row from the database using its row_id.
    
    Args:
        row (ConvoRow): The row object to refresh.
        
    Returns:
        Optional[ConvoRow]: The current row from the database if it exists, otherwise None.
    """
    if not row.row_id:
        logger.warning("Cannot refresh row without a row_id")
        return None
        
    try:
        query_str = f"""
        SELECT * FROM "{config.TABLE_NAME}" WHERE row_id = :row_id
        """
        
        params = {"row_id": row.row_id}
        
        async with get_connection() as conn:
            result = await conn.execute(text(query_str), params)
            row_data = result.fetchone()
            
        if row_data:
            # Convert row to dictionary
            row_dict = dict(row_data._mapping)
            for json_field in ['history_json', 'generation_params_json', 'attempts_metadata_json', 'response_json']:
                if isinstance(row_dict[json_field], dict):
                    row_dict[json_field] = row_dict[json_field]
            
            return ConvoRow.from_dict(row_dict)
        
        logger.info(f"No row found with row_id: {row.row_id}")
        return None
    except Exception as e:
        logger.error(f"Failed to refresh row with row_id {row.row_id}: {e}")
        raise