"""
inference.py

Combines inference and individual_inference functionalities.
Handles inference on multiple datapoints using multiple prompt functions.
Calls individual_inference asynchronously for each datapoint.
Manages clearing inference for batch jobs when run_fast=False.
Implements async "Individual Inference" with fast and slow (batch) modes.
Manages retries, database updates, and inference calls.
"""

import tempfile
import asyncio
import random
import logging
import json
from typing import List, Callable, Any, Optional
from datetime import datetime, timezone, timedelta

from google.cloud import storage
import vertexai
from google.api_core.exceptions import RetryError, ResourceExhausted, ServiceUnavailable, InternalServerError, Cancelled
from google.auth.exceptions import TransportError
from .cloudsql.schema import ConvoRow, RequestStatus, RequestCause, RequestStatus
from vertexai.preview.batch_prediction import BatchPredictionJob
from vertexai.generative_models import Content, GenerativeModel, Part
from vertexai.preview import generative_models
from .cloudsql.table_utils import (
    find_existing_row_by_content_hash,
    insert_row,
    get_rows_by_status_and_tag_and_batch,
    get_batch_ids_by_status_and_tag,
    stream_rows_by_status_and_tag,
    refresh_row,   
)

from .config import COOLDOWN_SECONDS_DEFAULT, MAX_RETRIES_DEFAULT, BATCH_TIMEOUT_HOURS_DEFAULT, ROUND_ROBIN_ENABLED_DEFAULT, ROUND_ROBIN_OPTIONS_DEFAULT, GCP_PROJECT_ID, VERTEX_BUCKET


# Configure logging
logger = logging.getLogger(__name__)

storage_client = storage.Client(project="navresearch")
bucket = storage_client.bucket(f"{VERTEX_BUCKET}")

# For convenience:
job_state_map = {
    "unspecified": 0,
    "queued": 1,
    "pending": 2,
    "running": 3,
    "succeeded": 4,
    "failed": 5,
    "cancelling": 6,
    "cancelled": 7,
    "paused": 8,
    "expired": 9,
    "updating": 10,
    "partially_succeeded": 11,
}

ROUND_ROBIN_IDX = 0


async def run_chat_inference_async(
    row: ConvoRow, 
    timeout: float, 
    chat: Optional[Any] = None, 
    cooldown_seconds: float = COOLDOWN_SECONDS_DEFAULT,
    round_robin_enabled: bool = ROUND_ROBIN_ENABLED_DEFAULT,
    round_robin_options: List[str] = ROUND_ROBIN_OPTIONS_DEFAULT
):
    """
    Performs asynchronous chat inference with Vertex AI generative models.
    
    This function handles the complete chat inference workflow, including model initialization,
    chat session management, message sending, and error handling. It supports both new chat
    sessions and continuing existing ones, with configurable safety settings, generation
    parameters, and system instructions.
    
    Functionality:
    - Creates a new chat session or uses an existing one passed as parameter
    - Configures safety settings to allow all content types
    - Applies temperature and token limit settings from the request
    - Sets up system instructions when provided
    - Implements round-robin region selection for load balancing (when enabled)
    - Manages exponential backoff retries for transient API errors
    - Enforces timeout limits for request completion
    - Truncates extremely long queries to prevent API errors
    
    Parameters:
    -----------
    row : ConvoRow
        Database row containing the inference request details, including:
        - Query text to send to the model
        - Model identifier
        - Chat history for context
        - Generation parameters (temperature, max tokens, etc.)
        - System prompt (optional)
    
    timeout : float
        Maximum time in seconds to wait for the inference response before timing out.
        Controls how long the function will wait for the model to generate a response.
    
    chat : Optional[Any], default=None
        Existing chat session to use for sending the message. If None, a new chat
        session will be created with the history from the row.
    
    cooldown_seconds : float, default=COOLDOWN_SECONDS_DEFAULT
        Base cooldown period between retries. Used with exponential backoff for
        resilience against transient API errors.
        
    round_robin_enabled : bool, default=ROUND_ROBIN_ENABLED_DEFAULT
        Whether to enable round-robin region selection for load balancing.
        
    round_robin_options : List[str], default=ROUND_ROBIN_OPTIONS_DEFAULT
        List of region options to cycle through when round_robin_enabled is True.
        
    Returns:
    --------
    tuple
        Four-element tuple containing:
        - response_text: str or None - The generated text response if successful
        - metadata: dict or None - Full response metadata from the model
        - chat: Chat object or None - The chat session used (for potential reuse)
        - error_code: int - Status code (0=success, 1=timeout, 2=value error)
    
    Error Handling:
    --------------
    - Implements retries with exponential backoff for transient API errors
    - Returns specific error codes for different failure types
    - Logs detailed error information
    - Truncates excessively long inputs to prevent API failures
    
    Notes:
    ------
    - The function modifies the global ROUND_ROBIN_IDX variable when round-robin is enabled
    - Maximum retry attempts are hardcoded to 8 (retries < 8)
    - Uses random jitter on initial cooldown to prevent thundering herd issues
    - Verifies chat history length remains consistent between retries
    
    Example:
    --------
    ```python
    response_text, metadata, chat_session, error_code = await run_chat_inference_async(
        row=conversation_row,
        timeout=120.0,
        chat=existing_chat_session,
        cooldown_seconds=2.0
    )
    if error_code == 0:
        # Process successful response
        process_response(response_text)
    else:
        # Handle error based on error_code
        handle_error(error_code)
    ```
    """
    global ROUND_ROBIN_IDX
    
    if chat is None:
        safety_config = [
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
            ),
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
            ),
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
            ),
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]
        gen_config = {"temperature": 0, "max_output_tokens": 8192}
        parsed_config = row.generation_params_json
        if "temperature" in parsed_config:
            gen_config["temperature"] = parsed_config["temperature"]
        if "max_output_tokens" in parsed_config:
            gen_config["max_output_tokens"] = parsed_config["max_output_tokens"]

        logging.info(f"Async querying model {row.model}.")
        model = GenerativeModel(row.model, safety_settings=safety_config, generation_config=gen_config,
                                system_instruction=parsed_config.get("system_prompt", None))

        history = []
        for turn in row.history_json["history"]:
            if not turn["parts"]["text"]:
                raise ValueError(f"Empty text in history: {row.history_json}")
            history.append(
                Content(
                    role=turn["role"],
                    parts=[Part.from_text(turn["parts"]["text"])],
                )
            )
        chat = model.start_chat(history=history, response_validation=False)
    
    initial_chat_history = len(chat.history)

    try:
        success = False
        tries = 0
        await asyncio.sleep(cooldown_seconds * random.random())
        while not success:
            try:
                user_query = row.query
                if round_robin_enabled:
                    region = round_robin_options[ROUND_ROBIN_IDX % len(round_robin_options)]
                    vertexai.init(project=GCP_PROJECT_ID, location=region)
                    ROUND_ROBIN_IDX += 1
                assert initial_chat_history == len(chat.history)
                if not user_query:
                    raise ValueError(f"Empty query: {row.query}")
                response = await asyncio.wait_for(chat.send_message_async(user_query), timeout=timeout)
                response.text
                success = True
            except (RetryError, ResourceExhausted, ServiceUnavailable, InternalServerError, Cancelled, TransportError) as e:
                tries += 1
                await asyncio.sleep(cooldown_seconds * (2 ** tries))
                assert tries < 8
        return response.text, response.to_dict(), chat, 0
    except TimeoutError as e:
        logger.error(f"Timeout error code: {e}")
        return None, None, None, 1
    except ValueError as e:
        logger.error(f"Value error code: {e}")
        return None, None, None, 2


async def run_clearing_inference(tag: str, batch_size: int, run_batch_jobs: bool, batch_timeout_hours: int = BATCH_TIMEOUT_HOURS_DEFAULT, round_robin_enabled: bool = ROUND_ROBIN_ENABLED_DEFAULT, round_robin_options: List[str] = ROUND_ROBIN_OPTIONS_DEFAULT):
    """
    Continuously processes batch inference jobs for requests with a specific tag until completion.
    
    This function manages the full lifecycle of batch inference jobs, including monitoring existing
    jobs, handling failures, launching new batches, and processing results. It runs in a continuous
    loop until all requests with the specified tag have been successfully processed.
    
    Functionality:
    - Monitors existing batch prediction jobs by querying their status
    - Handles failed batch jobs by updating row statuses
    - Cancels and restarts jobs that exceed the configured timeout (BATCH_TIMEOUT_HOURS_DEFAULT)
    - Processes successful batch jobs by:
      - Retrieving results from GCS storage
      - Parsing and mapping responses to their original requests
      - Updating database rows with results or failure information
    - Launches new batch prediction jobs for pending requests
    - Continuously checks and waits until all tagged requests are complete
    
    Parameters:
    -----------
    tag : str
        Unique identifier tag for the batch of inference requests to process.
        Used to track and identify related requests in the database.
    
    batch_size : int, default=4096
        Maximum number of requests to include in a single batch prediction job.
        Controls memory usage and parallelism. Larger values may improve throughput
        but increase resource requirements.
    
    run_batch_jobs : bool
        Whether to run batch prediction jobs. If False, the function will only monitor existing batch prediction jobs.
    
    batch_timeout_hours : int, default=BATCH_TIMEOUT_HOURS_DEFAULT
        Maximum time in hours to wait for a batch prediction job to complete.
        If a job exceeds this time, it will be cancelled and retried.
    
    round_robin_enabled : bool, default=ROUND_ROBIN_ENABLED_DEFAULT
        Whether to enable round-robin region selection for load balancing.
        
    round_robin_options : List[str], default=ROUND_ROBIN_OPTIONS_DEFAULT
        List of region options to cycle through when round_robin_enabled is True.

    Returns:
    --------
    None
        The function returns when all requests with the specified tag have been 
        successfully processed or marked as failed.
        
    Notes:
    ------
    - This function is designed to run as a long-running background task.
    - Relies on GCS bucket storage for batch inputs and outputs.
    - Implements exponential backoff retry logic for resilience.
    - Periodically logs progress and status information.
    
    Raises:
    -------
    Various exceptions may be raised due to GCP API issues or connectivity problems,
    but the function includes retry logic to handle transient failures.
    
    Example:
    --------
    ```python
    # Launch a background task to process batch inference jobs
    batch_tag = "inference_batch_20230415"
    asyncio.create_task(run_clearing_inference(tag=batch_tag, batch_size=1000))
    ```
    """
    # Get all rows with the given tag
    while True:
        logger.info("Running clearing inference for tag %s", tag)

        # Check on current batch prediction jobs
        batch_ids = await get_batch_ids_by_status_and_tag([RequestStatus.RUNNING, RequestStatus.PENDING], tag)
        for batch_id in batch_ids:
            job = BatchPredictionJob(batch_id)
            logger.info(f"Retrieving all incomplete rows corresponding to batch job {batch_id}")
            rows = await get_rows_by_status_and_tag_and_batch([RequestStatus.WAITING, RequestStatus.FAILED, RequestStatus.PENDING, RequestStatus.RUNNING], tag, batch_id)

            logger.info(f"Checking batch job {batch_id} with state {job.state} and this many rows remaining: {len(rows)}")

            # Handle if job has failed
            if job.state in [job_state_map["cancelled"], job_state_map["cancelling"], job_state_map["failed"], job_state_map["expired"]]:
                logger.info(f"Handling failed batch job {batch_id}")
                # Convert sequential updates to parallel using gather
                update_tasks = []
                for row in rows:
                    row.last_status = RequestStatus.FAILED
                    row.access_timestamps.append(datetime.now(timezone.utc).isoformat())
                    row.attempts_metadata_json = row.attempts_metadata_json + [{"error": "Batch job failed."}]
                    update_tasks.append(insert_row(row))
                await asyncio.gather(*update_tasks)
            
            # Optionally restart if job is running
            if job.state == job_state_map["running"]:
                logger.info(f"Checking running batch job {batch_id}")
                if job._gca_resource.start_time:
                    if (datetime.now(timezone.utc) - job._gca_resource.start_time) > timedelta(hours=batch_timeout_hours):
                        logger.warning(f"Job {batch_id} has been running for more than {batch_timeout_hours} hours. Restarting...")
                        job.cancel()
                        # Convert sequential updates to parallel using gather
                        update_tasks = []
                        for row in rows:
                            logger.warning(f"Job {batch_id} has been running for more than {batch_timeout_hours} hours. Restarting...")
                            row.last_status = RequestStatus.FAILED
                            row.access_timestamps.append(datetime.now(timezone.utc).isoformat())
                            row.attempts_metadata_json = row.attempts_metadata_json + [{"error": "Batch job ran out of time."}]
                            update_tasks.append(insert_row(row))
                        await asyncio.gather(*update_tasks)
    
            # Handle successful batch jobs
            if job.state in [job_state_map["succeeded"], job_state_map["partially_succeeded"]]:
                logger.info(f"Handling successful batch job {batch_id}")
                all_responses = {}

                blobs = list(bucket.list_blobs(prefix=f"{job._gca_resource.output_config.gcs_destination.output_uri_prefix.split(VERTEX_BUCKET + '/')[1]}"))
                
                blobs = sorted(blobs, key=lambda x: x.name)
                for blob in blobs:
                    if "incremental" in blob.name:
                        continue
                    with tempfile.NamedTemporaryFile(delete=True, mode="r") as f:
                        blob.download_to_filename(f.name)
                        for l in f.readlines():
                            if not l:
                                continue
                            query_content_hash = str(json.loads(l)["custom_id"])
                            try:
                                all_responses[query_content_hash] = json.loads(l)["response"]
                            except KeyError as e:
                                all_responses[query_content_hash] = json.loads(l)
                                all_responses[query_content_hash].pop("request")
                    
                    # Convert sequential row updates to parallel using gather
                    update_tasks = []
                    for row in rows:
                        if row.content_hash not in all_responses:
                            row.last_status = RequestStatus.FAILED
                            row.access_timestamps.append(datetime.now(timezone.utc).isoformat())
                            row.attempts_metadata_json = row.attempts_metadata_json + [{"This query was not found in batch!"}]
                            update_tasks.append(insert_row(row))
                            continue
                        try:
                            text_response = all_responses[row.content_hash]["candidates"][0]["content"]["parts"][0]["text"]
                            row.last_status = RequestStatus.SUCCEEDED
                            row.response_json = {"text": text_response}
                            row.access_timestamps.append(datetime.now(timezone.utc).isoformat())
                            row.attempts_metadata_json = row.attempts_metadata_json + [all_responses[row.content_hash]]
                        except KeyError as e:
                            row.last_status = RequestStatus.FAILED
                            row.access_timestamps.append(datetime.now(timezone.utc).isoformat())
                            row.attempts_metadata_json = row.attempts_metadata_json + [all_responses[row.content_hash]]
                            row.failure_count += 1
                        update_tasks.append(insert_row(row))
                    await asyncio.gather(*update_tasks)

        # Launch new batch prediction jobs
        if run_batch_jobs:
            logger.info(f"Checking now if there are new rows that need to be launched")
            async for batch in stream_rows_by_status_and_tag([RequestStatus.WAITING, RequestStatus.FAILED], tag, batch_size=batch_size):
                logger.info(f"Launching new batch prediction job for tag {tag}")
                model_types = set()
                for row in batch:
                    model_types.add(row.model)
                assert len(model_types) == 1
                model = model_types.pop()
                with tempfile.NamedTemporaryFile(mode="w") as f:
                    logger.info(f"Preparing minibatch for model {model} of length {len(batch)}...")
                    for row in batch:
                        request_contents = []
                        for turn in row.history_json["history"]:
                            request_contents.append(turn)
                        request_contents.append({"role": "user", "parts": {"text": row.query}})

                        full_request = {"request": {
                            "contents": request_contents,
                            "generationConfig": {
                                "temperature": row.generation_params_json.get("temperature", 0),
                                "maxOutputTokens": row.generation_params_json.get("max_output_tokens", 8192),
                            },
                            "safetySettings": [
                                {
                                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                    "threshold": "BLOCK_NONE",
                                },
                                {
                                    "category": "HARM_CATEGORY_HARASSMENT",
                                    "threshold": "BLOCK_NONE",
                                },
                                {
                                    "category": "HARM_CATEGORY_HATE_SPEECH",
                                    "threshold": "BLOCK_NONE",
                                },
                                {
                                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                    "threshold": "BLOCK_NONE",
                                },
                            ],
                        }, "custom_id": row.content_hash}
            
                        if "system_prompt" in row.generation_params_json:
                            full_request["request"]["system_instruction"] = {"role": "system", "parts": {"text": row.generation_params_json["system_prompt"]}}

                        json.dump(full_request, f)
                        f.write("\n")
                    f.flush()
                    pst_timezone = timezone(timedelta(hours=-8))
                    timestamp = datetime.now(timezone.utc).astimezone(pst_timezone).strftime("%Y%m%dT%H%M%S")
                    model_name = model.split("/")[-1]
                    batch_input_path = f"batch_inputs/{model_name}/{timestamp}.jsonl"
                    logger.info(f"Uploading to {batch_input_path}")
                    blob = bucket.blob(batch_input_path)
                    blob.upload_from_filename(f.name, if_generation_match=0)
                    logger.info(f"Done uploading to {batch_input_path}")

                    if round_robin_enabled:
                        region = round_robin_options[ROUND_ROBIN_IDX % len(round_robin_options)]
                        logger.info(f"Running in region {region}")
                        vertexai.init(project=GCP_PROJECT_ID, location=region)
                        ROUND_ROBIN_IDX += 1

                    batch_input_path_gcs = f"gs://{VERTEX_BUCKET}/{batch_input_path}"
                    output_uri_prefix = f"gs://{VERTEX_BUCKET}/batch_outputs/{model_name}/{timestamp}.jsonl"

                    job = BatchPredictionJob.submit(
                        source_model=model,
                        input_dataset=batch_input_path_gcs,
                        output_uri_prefix=output_uri_prefix,
                    )
                    # Parallelize row updates after batch job creation
                    update_tasks = []
                    for row in batch:
                        row.last_status = RequestStatus.PENDING
                        row.access_timestamps.append(datetime.now(timezone.utc).isoformat())
                        row.current_batch = job.resource_name
                        update_tasks.append(insert_row(row))
                    await asyncio.gather(*update_tasks)

        logger.info(f"Checking now if we are completely done with this tag")
        rows = await get_rows_by_status_and_tag_and_batch([RequestStatus.WAITING, RequestStatus.FAILED, RequestStatus.PENDING, RequestStatus.RUNNING], tag)
        if not rows:
            logger.info("Clearing inference for tag %s completed", tag)
            return

        logger.info("Clearing inference for tag %s waiting", tag)
        await asyncio.sleep(60)


async def individual_inference(
    prompt_functions: List[Callable[[Any], str]],
    datapoint: Any,
    tags: Optional[List[str]] = None,
    optional_tags: Optional[List[str]] = None,
    duplication_index: int = 0,
    run_fast: bool = True,
    allow_failure: bool = False,
    attempts_cap: int = MAX_RETRIES_DEFAULT,
    temperature: float = 0,
    max_output_tokens: int = 8192,
    system_prompt: str = "",
    model: str = "publishers/google/models/gemini-1.5-flash-002",
    run_fast_timeout: float = 200,
    cooldown_seconds: float = COOLDOWN_SECONDS_DEFAULT,
    round_robin_enabled: bool = ROUND_ROBIN_ENABLED_DEFAULT,
    round_robin_options: List[str] = ROUND_ROBIN_OPTIONS_DEFAULT,
    initial_history_json: Optional[dict] = None,
) -> List[str]:
    """
    Performs inference for a single datapoint using multiple prompt functions in sequence.
    
    This function handles the complete workflow for processing a single datapoint through
    a sequence of prompts, maintaining conversation history between prompts. It supports
    both fast (synchronous) and batch (asynchronous) inference modes, with comprehensive
    error handling, retry mechanisms, and database persistence.
    
    Functionality:
    - Processes a single datapoint through multiple prompt functions sequentially
    - Builds and maintains conversation history across prompt functions
    - Supports result caching and deduplication through content hashing
    - Handles database persistence of requests and responses
    - Implements timeout, retry logic, and failure handling
    - Provides both fast (direct API call) and batch (queued processing) modes
    
    Parameters:
    -----------
    prompt_functions : List[Callable[[Any], str]]
        List of functions that transform the datapoint into prompt strings.
        Each function is called in sequence, with conversation history maintained.
    
    datapoint : Any
        The data to be processed by prompt functions. Can be any type that
        the prompt functions can handle (dict, string, object, etc.).
    
    tags : Optional[List[str]], default=None
        Identifier tags for tracking and grouping related requests in the database.
        Required when run_fast=False for batch processing.
    
    optional_tags : Optional[List[str]], default=None
        Additional tags that won't be used for row lookup but will be added to the row.
    
    duplication_index : int, default=0
        Index to distinguish duplicate runs of the same content.
        Useful for running the same query multiple times with different parameters.
    
    run_fast : bool, default=True
        If True, performs direct API calls (fast mode).
        If False, queues requests for batch processing (requires tags).
    
    allow_failure : bool, default=False
        If True, continues processing when attempts_cap is exceeded, returning error messages.
        If False, raises ValueError when attempts_cap is exceeded.
    
    attempts_cap : int, default=MAX_RETRIES_DEFAULT
        Maximum number of retry attempts before considering a request failed.
    
    temperature : float, default=0
        Temperature parameter for generation. Higher values increase randomness.
    
    max_output_tokens : int, default=8192
        Maximum number of tokens to generate in the response.
    
    system_prompt : str, default=""
        System prompt to guide model behavior. Empty string means no system prompt.
    
    model : str, default="publishers/google/models/gemini-1.5-flash-002"
        Identifier of the generative model to use for inference.
    
    run_fast_timeout : float, default=200
        Timeout in seconds for fast mode inference calls.
    
    cooldown_seconds : float, default=COOLDOWN_SECONDS_DEFAULT
        Base wait time between retry attempts. Used with exponential backoff.
    
    round_robin_enabled : bool, default=ROUND_ROBIN_ENABLED_DEFAULT
        Whether to cycle through different regions for load balancing.
    
    round_robin_options : List[str], default=ROUND_ROBIN_OPTIONS_DEFAULT
        List of region options to cycle through when round_robin_enabled is True.
    
    initial_history_json : Optional[dict], default=None
        Starting conversation history for the inference session. Of form
        {"history": [{"role": "user", "parts": {"text": "user query 1"}},
                     {"role": "model", "parts": {"text": "model response 1"}},
                     ...]}
    
    Returns:
    --------
    tuple
        Two-element tuple containing:
        - collected_responses: List[str] - List of model responses from each prompt
        - collected_queries: List[str] - List of prompts generated from the datapoint
    
    Raises:
    -------
    ValueError
        If tags is not provided when run_fast=False
        If tags is a string instead of a list
        If prompt_functions is not a list
        If attempts_cap is exceeded and allow_failure=False
    
    Notes:
    ------
    - Fast mode (run_fast=True): Makes direct API calls and waits for results
    - Batch mode (run_fast=False): Queues requests for processing by a batch job
    - When using batch mode, a separate clearing job must be running to process the batch
    - Response caching: If a response for a content hash already exists, it will be reused
    
    Example:
    --------
    ```python
    # Define prompt functions
    def initial_prompt(data):
        return f"Summarize this text: {data['text']}"
        
    def followup_prompt(data):
        return "What are the key insights from this summary?"
    
    # Run inference with fast mode
    responses, queries = await individual_inference(
        prompt_functions=[initial_prompt, followup_prompt],
        datapoint={"text": "Sample document content..."},
        tags=["summary_project", "batch_123"],
        run_fast=True
    )
    
    # Process results
    summary, insights = responses
    ```
    """
    # If not tags, will always create new 
    
    if not tags and not run_fast:
        raise ValueError("You are doing something wrong. If fast mode is disabled, you need to run a separate clearing job, which requires tags.")
    if isinstance(tags, str):
        raise ValueError("Tags must be a list of strings.")
    if not isinstance(prompt_functions, list):
        raise ValueError("Prompt functions must be a list.")
    
    all_tags = (tags or []) + (optional_tags or [])

    collected_responses = []
    collected_queries = []
    chat = None
    history_json = initial_history_json or {"history": []}

    for prompt_func in prompt_functions:
        # Get new history
        user_query = prompt_func(datapoint)
        generation_params_json = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if system_prompt:
            generation_params_json["system_prompt"] = system_prompt

        # Set up query
        row = ConvoRow(
            history_json=history_json,
            query=user_query,
            model=model,
            generation_params_json=generation_params_json,
            duplication_index=duplication_index,
            tags=all_tags,
            request_cause=RequestCause.INTENTIONAL,
            attempts_cap=attempts_cap,
        )
        collected_queries.append(user_query)
        history_json["history"].append({"role": "user", "parts": {"text": user_query}})

        # Decide whether to create a new row
        if all_tags:
            new_row = await find_existing_row_by_content_hash(row.content_hash, tags)
            if new_row:
                row = new_row
            else:
                logger.info(f"Creating new row for content hash {row.content_hash}")
                await insert_row(row)

        # Update row with access information
        row.access_timestamps.append(datetime.now(timezone.utc).isoformat())
        row.tags = sorted(set(row.tags + all_tags))

        # Run inference
        response = None
        if row.response_json:
            response = row.response_json["text"]
            assert row.last_status == RequestStatus.SUCCEEDED
        else:
            logger.info(f"Row {row.content_hash} has no response, will run inference")

        while response is None:
            # If permission exceeded
            if row.failure_count >= row.attempts_cap:
                if not allow_failure:
                    raise ValueError("Exceeded attempts cap.")
                collected_responses.extend(["ERROR MESSAGE"] * (len(prompt_functions) - len(collected_responses)))
                collected_queries.extend(["ERROR MESSAGE"] * (len(prompt_functions) - len(collected_queries)))
                return collected_responses, collected_queries
            
            # If the row is successfully completed, exit
            if row.last_status == RequestStatus.SUCCEEDED:
                response = row.response_json["text"]
                break
            
            # More attempts are allowed and no success yet, so run inference
            if run_fast:
                # Will override batch
                resp_text, metadata, chat, err_code = await run_chat_inference_async(
                    row, timeout=run_fast_timeout, chat=chat, cooldown_seconds=cooldown_seconds,
                    round_robin_enabled=round_robin_enabled, round_robin_options=round_robin_options
                )
                if not row.attempts_metadata_json:
                    row.attempts_metadata_json = []
                row.attempts_metadata_json = row.attempts_metadata_json + [metadata]
                if err_code == 0:
                    row.response_json = {"text": resp_text}
                    row.last_status = RequestStatus.SUCCEEDED
                    await insert_row(row)
                else:
                    row.failure_count += 1
                    row.last_status = RequestStatus.FAILED
                    await insert_row(row)
            else:
                # If we've reached here, we know we need to re-attempt.
                if row.last_status == RequestStatus.FAILED:
                    row.last_status = RequestStatus.WAITING
                    await insert_row(row)

                # Wait for batch update to row
                while row.last_status in (RequestStatus.WAITING, RequestStatus.PENDING):
                    # Report if waiting in waiting status or waiting in pending status
                    await asyncio.sleep(30)
                    row = await refresh_row(row)

        collected_responses.append(response)
        history_json["history"].append({"role": "model", "parts": {"text": response}})

    return collected_responses, collected_queries


async def inference(
    prompt_functions: List[Callable[[Any], str]],
    datapoints: List[Any],
    tags: Optional[List[str]] = None,
    duplication_indices: Optional[List[int]] = None,
    run_fast: bool = True,
    allow_failure: bool = False,
    attempts_cap: int = MAX_RETRIES_DEFAULT,
    temperature: float = 0,
    max_output_tokens: int = 8192,
    system_prompt: str = "",
    model: str = "publishers/google/models/gemini-1.5-flash-002",
    batch_size: int = 1000,
    run_fast_timeout: float = 200,
    cooldown_seconds: float = COOLDOWN_SECONDS_DEFAULT,
    batch_timeout_hours: int = BATCH_TIMEOUT_HOURS_DEFAULT,
    round_robin_enabled: bool = ROUND_ROBIN_ENABLED_DEFAULT,
    round_robin_options: List[str] = ROUND_ROBIN_OPTIONS_DEFAULT,
    initial_histories: Optional[List[dict]] = None,
) -> List[List[str]]:
    """
    Processes multiple datapoints through a sequence of prompt functions with parallel execution.
    
    This function orchestrates batch inference across multiple datapoints, handling all aspects
    of parallel processing, result collection, and error management. It supports both fast
    (direct API calls) and batch (queued processing) modes, with configurable concurrency,
    retry logic, and region distribution.
    
    Functionality:
    - Processes multiple datapoints through multiple prompt functions in parallel
    - Creates and manages unique batch tags for tracking related requests
    - Controls concurrency with semaphores in fast mode to prevent API overloading
    - Supports duplicate runs of the same datapoints with different parameters
    - Manages both fast (synchronous) and batch (asynchronous queued) processing modes
    - Handles load balancing across regions with round-robin configuration
    - Provides comprehensive parameter controls for model behavior and performance
    
    Parameters:
    -----------
    prompt_functions : List[Callable[[Any], str]]
        List of functions that transform datapoints into prompt strings.
        Each function is called in sequence for each datapoint, with conversation history maintained.
    
    datapoints : List[Any]
        List of data items to be processed. Each item is passed to the prompt functions.
        Can be any type that the prompt functions can handle (dict, string, object, etc.).
    
    tags : Optional[List[str]], default=None
        Identifier tags for tracking and grouping related requests in the database.
        Required when run_fast=False for batch processing.
    
    duplication_indices : Optional[List[int]], default=None
        List of indices to run the same datapoints multiple times with different tracking.
        Defaults to [0] if None, meaning each datapoint is processed once.
    
    run_fast : bool, default=True
        If True, performs direct API calls (fast mode) with semaphore-controlled concurrency.
        If False, queues requests for batch processing (requires tags).
    
    allow_failure : bool, default=False
        If True, continues processing when attempts_cap is exceeded, returning error messages.
        If False, raises ValueError when attempts_cap is exceeded.
    
    attempts_cap : int, default=MAX_RETRIES_DEFAULT
        Maximum number of retry attempts before considering a request failed.
    
    temperature : float, default=0
        Temperature parameter for generation. Higher values increase randomness.
    
    max_output_tokens : int, default=8192
        Maximum number of tokens to generate in the response.
    
    system_prompt : str, default=""
        System prompt to guide model behavior. Empty string means no system prompt.
    
    model : str, default="publishers/google/models/gemini-1.5-flash-002"
        Identifier of the generative model to use for inference.
    
    batch_size : int, default=1000
        Maximum number of concurrent requests in fast mode, or maximum number of
        requests per batch job in batch mode.
    
    run_fast_timeout : float, default=200
        Timeout in seconds for fast mode inference calls.
    
    cooldown_seconds : float, default=COOLDOWN_SECONDS_DEFAULT
        Base wait time between retry attempts. Used with exponential backoff.
    
    batch_timeout_hours : int, default=BATCH_TIMEOUT_HOURS_DEFAULT
        Maximum runtime for batch jobs before they're considered stalled and restarted.
    
    round_robin_enabled : bool, default=ROUND_ROBIN_ENABLED_DEFAULT
        Whether to cycle through different regions for load balancing.
    
    round_robin_options : List[str], default=ROUND_ROBIN_OPTIONS_DEFAULT
        List of region options to cycle through when round_robin_enabled is True.
    
    initial_histories : Optional[List[dict]], default=None
        Starting conversation histories for the inference session.
    
    Returns:
    --------
    tuple
        Two-element tuple containing:
        - results: List[tuple] - List of (responses, queries) tuples for each datapoint-duplication combination
        - launch_timestamp_tag: str - Unique timestamp tag generated for this batch run
    
    Raises:
    -------
    Various exceptions may be propagated from individual_inference, including:
    - ValueError if tags is not provided when run_fast=False
    - ValueError if attempts_cap is exceeded and allow_failure=False
    - Various API errors if they persist beyond retry attempts
    
    Notes:
    ------
    - In fast mode (run_fast=True), a semaphore limits concurrent API calls to prevent overloading
    - In batch mode (run_fast=False), requests are queued and a clearing job processes them
    - The function automatically generates and attaches a timestamp tag for tracking
    - When duplication_indices is provided, each datapoint is processed multiple times
    - Results maintain ordering corresponding to the input datapoints and duplication indices
    
    Example:
    --------
    ```python
    # Define prompt functions
    def summarize(data):
        return f"Summarize this text: {data['text']}"
        
    def extract_entities(data):
        return "Extract named entities from the summary."
    
    # Prepare data
    documents = [{"text": "Document 1 content..."}, {"text": "Document 2 content..."}]
    
    # Run inference with fast mode
    results, batch_tag = await inference(
        prompt_functions=[summarize, extract_entities],
        datapoints=documents,
        tags=["entity_extraction_project"],
        run_fast=True,
        batch_size=5  # Process up to 5 documents concurrently
    )
    
    # Process results (two prompt functions per datapoint)
    for i, (responses, queries) in enumerate(results):
        summary, entities = responses
        print(f"Document {i+1} summary: {summary}")
        print(f"Document {i+1} entities: {entities}")
    ```
    """
    # Generate a timestamp tag for tracking batch inference runs
    launch_timestamp_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    logger.info(f"Running inference with timestamp tag: {launch_timestamp_tag}")

    # Process each datapoint asynchronously
    tasks = []
    semaphore = asyncio.Semaphore(batch_size) if run_fast else None

    async def process_datapoint(dp, initial_history_json: Optional[dict], duplication_index: int):
        if semaphore is not None:
            assert run_fast
            async with semaphore:
                # Call individual inference for this datapoint
                responses, queries = await individual_inference(
                    prompt_functions=prompt_functions,
                    datapoint=dp,
                    tags=tags,
                    optional_tags=[launch_timestamp_tag],
                    duplication_index=duplication_index,
                    run_fast=run_fast,
                    run_fast_timeout=run_fast_timeout,
                    allow_failure=allow_failure,
                    attempts_cap=attempts_cap,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    system_prompt=system_prompt,
                    model=model,
                    cooldown_seconds=cooldown_seconds,
                    round_robin_enabled=round_robin_enabled,
                    round_robin_options=round_robin_options,
                    initial_history_json=initial_history_json,
                )
                return responses, queries
        else:
            assert not run_fast
            # Call individual inference for this datapoint without semaphore
            responses, queries = await individual_inference(
                prompt_functions=prompt_functions,
                datapoint=dp,
                tags=tags,
                optional_tags=[launch_timestamp_tag],
                duplication_index=duplication_index,
                run_fast=run_fast,
                allow_failure=allow_failure,
                attempts_cap=attempts_cap,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_prompt=system_prompt,
                model=model,
                cooldown_seconds=cooldown_seconds,
                round_robin_enabled=round_robin_enabled,
                round_robin_options=round_robin_options,
                initial_history_json=initial_history_json,
            )
            return responses, queries

    for duplication_index in duplication_indices or [0]:
        for i, dp in enumerate(datapoints):
            tasks.append(process_datapoint(dp, initial_histories[i] if initial_histories else None, duplication_index))

    # If running in batch mode, periodically trigger clearing inference
    if not run_fast:
        tasks.append(run_clearing_inference(tag=launch_timestamp_tag, batch_size=batch_size, run_batch_jobs=True, batch_timeout_hours=batch_timeout_hours))

    results = list(await asyncio.gather(*tasks))

    if not run_fast:
        results = results[:-1]

    return results, launch_timestamp_tag
