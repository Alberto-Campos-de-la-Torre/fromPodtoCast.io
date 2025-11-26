"""
Schemas Pydantic para validación de metadata de podcasts.
Garantiza estructura correcta antes de guardar JSON.
"""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pathlib import Path


class SpeakerInfo(BaseModel):
    """
    Información de un hablante identificado.
    """
    speaker_id: str = Field(..., description="ID único del hablante")
    global_id: Optional[str] = Field(None, description="ID global del voice bank")
    segments_count: int = Field(default=0, ge=0, description="Número de segmentos")
    total_duration: float = Field(default=0.0, ge=0.0, description="Duración total en segundos")
    
    @field_validator('total_duration')
    @classmethod
    def round_duration(cls, v: float) -> float:
        return round(v, 2)


class LLMCorrectionInfo(BaseModel):
    """
    Información de corrección LLM almacenada en un segmento.
    """
    original: str = Field(..., description="Texto original")
    cambios: List[str] = Field(default_factory=list)
    confianza: float = Field(ge=0.0, le=1.0)


class SegmentMetadata(BaseModel):
    """
    Metadata completa de un segmento de audio procesado.
    
    Esta es la estructura final que se guarda en el JSON de metadata.
    """
    audio_path: str = Field(..., description="Ruta al archivo de audio normalizado")
    speaker: str = Field(..., description="ID del hablante")
    text: str = Field(..., description="Texto transcrito y corregido")
    text_original: Optional[str] = Field(None, description="Texto original antes de correcciones")
    duration: float = Field(..., gt=0, description="Duración en segundos")
    sample_rate: int = Field(default=22050, description="Sample rate del audio")
    start_time: Optional[float] = Field(None, ge=0, description="Tiempo de inicio en el audio original")
    end_time: Optional[float] = Field(None, ge=0, description="Tiempo de fin en el audio original")
    llm_correction: Optional[LLMCorrectionInfo] = Field(None, description="Info de corrección LLM")
    preprocessing_applied: bool = Field(default=False, description="Si se aplicó preprocesamiento")
    
    @field_validator('duration', 'start_time', 'end_time')
    @classmethod
    def round_times(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            return round(v, 3)
        return v
    
    @model_validator(mode='after')
    def validate_times(self) -> 'SegmentMetadata':
        """Verifica que end_time > start_time si ambos están presentes."""
        if self.start_time is not None and self.end_time is not None:
            if self.end_time <= self.start_time:
                raise ValueError(f"end_time ({self.end_time}) debe ser mayor que start_time ({self.start_time})")
        return self


class ProcessingStats(BaseModel):
    """
    Estadísticas del procesamiento de un podcast.
    """
    total_duration: float = Field(..., ge=0, description="Duración total del audio original")
    useful_duration: float = Field(..., ge=0, description="Duración útil procesada")
    segments_count: int = Field(..., ge=0, description="Número de segmentos generados")
    speakers_count: int = Field(..., ge=0, description="Número de hablantes identificados")
    transcription_time: Optional[float] = Field(None, ge=0, description="Tiempo de transcripción")
    llm_correction_time: Optional[float] = Field(None, ge=0, description="Tiempo de corrección LLM")
    total_processing_time: float = Field(..., ge=0, description="Tiempo total de procesamiento")
    efficiency: float = Field(default=0.0, ge=0.0, le=100.0, description="Porcentaje de audio útil")
    
    # Estadísticas de LLM
    llm_corrected_count: int = Field(default=0, ge=0)
    llm_failed_count: int = Field(default=0, ge=0)
    llm_avg_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_cache_hits: int = Field(default=0, ge=0, description="Correcciones recuperadas del caché")
    
    @field_validator('efficiency')
    @classmethod
    def calculate_efficiency(cls, v: float) -> float:
        return round(v, 2)
    
    @model_validator(mode='after')
    def compute_efficiency(self) -> 'ProcessingStats':
        """Calcula eficiencia si no está establecida."""
        if self.efficiency == 0.0 and self.total_duration > 0:
            self.efficiency = round((self.useful_duration / self.total_duration) * 100, 2)
        return self


class DiarizationStats(BaseModel):
    """Estadísticas de diarización."""
    speakers_detected: int = Field(..., ge=0)
    total_speech_duration: float = Field(..., ge=0)
    processing_time: float = Field(..., ge=0)


class TranscriptionStats(BaseModel):
    """Estadísticas de transcripción."""
    segments_transcribed: int = Field(..., ge=0)
    total_text_length: int = Field(..., ge=0)
    detected_language: str = Field(default="es")
    processing_time: float = Field(..., ge=0)


class LLMCorrectionStats(BaseModel):
    """Estadísticas de corrección LLM."""
    enabled: bool = Field(default=True)
    corrected_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    avg_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    total_changes: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    batch_calls: int = Field(default=0, ge=0, description="Número de llamadas batch al LLM")


class PodcastMetadata(BaseModel):
    """
    Metadata completa de un podcast procesado.
    Este es el schema del archivo JSON de salida.
    """
    podcast_id: str = Field(..., description="ID único del podcast")
    source_file: str = Field(..., description="Archivo de audio original")
    processed_at: datetime = Field(default_factory=datetime.now)
    segments: List[SegmentMetadata] = Field(default_factory=list)
    speakers: Dict[str, SpeakerInfo] = Field(default_factory=dict)
    processing_stats: Optional[ProcessingStats] = None
    
    # Estadísticas detalladas opcionales
    diarization_stats: Optional[DiarizationStats] = None
    transcription_stats: Optional[TranscriptionStats] = None
    llm_stats: Optional[LLMCorrectionStats] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @model_validator(mode='after')
    def compute_speaker_stats(self) -> 'PodcastMetadata':
        """Computa estadísticas de hablantes basadas en segmentos."""
        if self.segments and not self.speakers:
            speaker_stats: Dict[str, Dict] = {}
            for seg in self.segments:
                if seg.speaker not in speaker_stats:
                    speaker_stats[seg.speaker] = {
                        'segments_count': 0,
                        'total_duration': 0.0
                    }
                speaker_stats[seg.speaker]['segments_count'] += 1
                speaker_stats[seg.speaker]['total_duration'] += seg.duration
            
            self.speakers = {
                spk: SpeakerInfo(
                    speaker_id=spk,
                    segments_count=stats['segments_count'],
                    total_duration=stats['total_duration']
                )
                for spk, stats in speaker_stats.items()
            }
        return self
    
    def to_json_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario compatible con JSON."""
        return self.model_dump(mode='json', exclude_none=True)


class MetadataValidationResult(BaseModel):
    """
    Resultado de validación de metadata.
    """
    is_valid: bool = Field(..., description="Si la metadata es válida")
    errors: List[str] = Field(default_factory=list, description="Lista de errores encontrados")
    warnings: List[str] = Field(default_factory=list, description="Lista de advertencias")
    segments_validated: int = Field(default=0, ge=0)
    
    def add_error(self, error: str) -> None:
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)

