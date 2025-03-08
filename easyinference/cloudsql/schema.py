"""
schema.py

Defines the BigQuery schema and associated data classes/enums for tracking inference requests.
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict
from enum import Enum
import json
import hashlib


class RequestStatus(str, Enum):
    """
    Enumeration of valid statuses for the Last Status column in BigQuery.
    """
    PENDING = "Pending"
    RUNNING = "Running"
    FAILED = "Failed"
    SUCCEEDED = "Succeeded"
    WAITING = "Waiting"  # Used for batch mode before processing


class RequestCause(str, Enum):
    """
    Indicates whether a request was intentionally made or automatically triggered as a backup.
    """
    INTENTIONAL = "intentional"
    BACKUP = "backup"


class ConvoRow:
    """
    Represents a single row in the BigQuery master table.

    Attributes:
        row_id (Optional[int]): Database-generated primary key for the row.
        content_hash (str): Immutable unique identifier for the row.
        history_json (Dict): Conversation history in Vertex AI's format.
        query (str): User's latest query requiring inference.
        model (str): Full path of the model needed to infer.
        generation_params_json (Dict): Generation parameters.
        duplication_index (int): Integer for marking intentional duplications (default 0).
        tags (List[str]): Alphabetically sorted list of tags associated with the query.
        request_cause (RequestCause): Whether the request was intentional or a backup.
        request_timestamp (str): Timestamp of the request in ISO 8601 format.
        access_timestamps (List[str]): List of ISO 8601 timestamps marking when the row was accessed.
        attempts_metadata_json (List): List storing metadata of past attempts, errors, or responses.
        response_json (str): JSON string containing the successful response, if available.
        current_batch (Optional[str]): Batch ID if the request is pending/running as part of a batch.
        last_status (RequestStatus): The latest known status of the request.
        failure_count (int): Number of recorded failures for this request.
        attempts_cap (int): Maximum number of attempts allowed before stopping.
        content_hash (str): Hash of (Model, History, Query, Generation parameters, Duplication index) for deduplication.
        notes (str): Optional field for notes, default empty.
    """

    def __init__(
        self,
        history_json: Dict,
        query: str,
        model: str,
        generation_params_json: Dict,
        tags: List[str] = None,
        duplication_index: int = 0,
        request_cause: RequestCause = RequestCause.INTENTIONAL,
        request_timestamp: str = None,
        access_timestamps: List[str] = None,
        attempts_metadata_json: List = None,
        response_json: str = None,
        current_batch: Optional[str] = None,
        last_status: RequestStatus = RequestStatus.WAITING,
        failure_count: int = 0,
        attempts_cap: int = 3,
        notes: str = "",
        row_id: Optional[int] = None,
    ):
        self.history_json = history_json
        self.query = query
        self.model = model
        self.generation_params_json = generation_params_json
        self.duplication_index = duplication_index
        self.tags = sorted(tags or [])
        self.request_cause = request_cause
        self.request_timestamp = request_timestamp or datetime.now(timezone.utc).isoformat()
        self.access_timestamps = access_timestamps or [self.request_timestamp]
        self.attempts_metadata_json = attempts_metadata_json or []
        self.response_json = response_json or {}
        self.current_batch = current_batch
        self.last_status = last_status
        self.failure_count = failure_count
        self.attempts_cap = attempts_cap
        self.notes = notes
        self.row_id = row_id

    @property
    def content_hash(self) -> str:
        """
        Computes a stable hash of (model, history_json, query, generation_params_json, duplication_index).
        Used for deduplication and resuming previous responses.
        """
        canonical_obj = {
            "model": self.model,
            "history_json": self.history_json,
            "query": self.query,
            "generation_params_json": self.generation_params_json,
            "duplication_index": self.duplication_index,
        }
        s = json.dumps(canonical_obj, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, any]:
        """
        Convert this row to a Python dictionary for BigQuery operations.

        Returns:
            Dict[str, any]: Dictionary representation of the BigQuery row.
        """
        result = {
            "content_hash": self.content_hash,
            "history_json": self.history_json,
            "query": self.query,
            "model": self.model,
            "generation_params_json": self.generation_params_json,
            "duplication_index": self.duplication_index,
            "tags": self.tags,
            "request_cause": self.request_cause.value,
            "request_timestamp": self.request_timestamp,
            "access_timestamps": self.access_timestamps,
            "attempts_metadata_json": self.attempts_metadata_json,
            "response_json": self.response_json,
            "current_batch": self.current_batch,
            "last_status": self.last_status.value,
            "failure_count": self.failure_count,
            "attempts_cap": self.attempts_cap,
            "notes": self.notes,
            "row_id": self.row_id,
        }
            
        return result

    @staticmethod
    def from_dict(row: Dict[str, any]) -> "ConvoRow":
        """
        Construct a ConvoRow object from a dictionary.

        Args:
            row (Dict[str, any]): Dictionary containing row data from BigQuery.

        Returns:
            ConvoRow: A ConvoRow instance populated from the dictionary.
        """
        new_row = row.copy()
        if isinstance(new_row["history_json"], str):
            new_row["history_json"] = json.loads(new_row["history_json"])
        if isinstance(new_row["generation_params_json"], str):
            new_row["generation_params_json"] = json.loads(new_row["generation_params_json"])
        if isinstance(new_row["attempts_metadata_json"], str):
            new_row["attempts_metadata_json"] = [json.loads(t) for t in new_row["attempts_metadata_json"]]
        if isinstance(new_row["response_json"], str):
            new_row["response_json"] = json.loads(new_row["response_json"])
        return ConvoRow(
            history_json=new_row["history_json"],
            query=new_row["query"],
            model=new_row["model"],
            generation_params_json=new_row["generation_params_json"],
            duplication_index=new_row["duplication_index"],
            tags=sorted(new_row["tags"]),
            request_cause=RequestCause(new_row["request_cause"]),
            request_timestamp=new_row["request_timestamp"],
            access_timestamps=new_row["access_timestamps"],
            attempts_metadata_json=new_row["attempts_metadata_json"],
            response_json=new_row["response_json"],
            current_batch=new_row["current_batch"] if new_row["current_batch"] else None,
            last_status=RequestStatus(new_row["last_status"]),
            failure_count=new_row["failure_count"],
            attempts_cap=new_row["attempts_cap"],
            notes=new_row["notes"],
            row_id=new_row["row_id"],
        )