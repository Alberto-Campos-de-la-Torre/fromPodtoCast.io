"""
Modelos Pydantic para validación estructurada del pipeline.
"""
from .llm_schemas import (
    LLMCorrectionResponse,
    LLMCorrectionBatchResponse,
    CorrectionRequest,
    CorrectionBatchRequest,
    LLMCorrectionMetadata
)
from .metadata_schemas import (
    SegmentMetadata,
    PodcastMetadata,
    ProcessingStats,
    SpeakerInfo
)

__all__ = [
    # LLM Schemas
    'LLMCorrectionResponse',
    'LLMCorrectionBatchResponse',
    'CorrectionRequest',
    'CorrectionBatchRequest',
    'LLMCorrectionMetadata',
    # Metadata Schemas
    'SegmentMetadata',
    'PodcastMetadata',
    'ProcessingStats',
    'SpeakerInfo'
]

