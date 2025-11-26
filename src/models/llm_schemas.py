"""
Schemas Pydantic para validación de respuestas del LLM.
Garantiza estructura correcta y tipos válidos en las correcciones.
"""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class LLMCorrectionResponse(BaseModel):
    """
    Respuesta estructurada de una corrección LLM individual.
    
    Attributes:
        texto_corregido: El texto después de la corrección
        cambios: Lista de cambios aplicados (máximo 10)
        confianza: Nivel de confianza entre 0 y 1
    """
    texto_corregido: str = Field(..., min_length=1, description="Texto corregido")
    cambios: List[str] = Field(default_factory=list, max_length=10, description="Lista de cambios aplicados")
    confianza: float = Field(ge=0.0, le=1.0, description="Confianza de la corrección")
    
    @field_validator('confianza')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Redondea la confianza a 2 decimales."""
        return round(v, 2)
    
    @field_validator('texto_corregido')
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Limpia espacios extra del texto."""
        return v.strip()
    
    @field_validator('cambios')
    @classmethod
    def validate_cambios(cls, v: List[str]) -> List[str]:
        """Filtra cambios vacíos."""
        return [c.strip() for c in v if c and c.strip()]


class LLMCorrectionBatchItem(BaseModel):
    """Item individual en una respuesta de batch."""
    id: int = Field(..., ge=0, description="Índice del texto en el batch")
    texto_corregido: str = Field(..., min_length=1)
    cambios: List[str] = Field(default_factory=list, max_length=10)
    confianza: float = Field(ge=0.0, le=1.0)
    
    @field_validator('confianza')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return round(v, 2)


class LLMCorrectionBatchResponse(BaseModel):
    """
    Respuesta estructurada de correcciones en batch.
    Permite procesar múltiples textos en una sola llamada.
    """
    correcciones: List[LLMCorrectionBatchItem] = Field(
        ..., 
        min_length=1,
        description="Lista de correcciones para cada texto del batch"
    )
    
    @model_validator(mode='after')
    def validate_order(self) -> 'LLMCorrectionBatchResponse':
        """Verifica que los IDs estén en orden y sean consecutivos."""
        ids = [c.id for c in self.correcciones]
        expected = list(range(len(ids)))
        if sorted(ids) != expected:
            # Reordenar si es necesario
            self.correcciones = sorted(self.correcciones, key=lambda x: x.id)
        return self


class CorrectionRequest(BaseModel):
    """
    Solicitud de corrección individual.
    
    Attributes:
        text: Texto a corregir
        segment_id: Identificador único del segmento
        speaker: Hablante opcional para contexto
    """
    text: str = Field(..., min_length=1, description="Texto a corregir")
    segment_id: str = Field(..., description="ID único del segmento")
    speaker: Optional[str] = Field(None, description="Identificador del hablante")
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        return v.strip()


class CorrectionBatchRequest(BaseModel):
    """
    Solicitud de corrección en batch.
    Agrupa múltiples textos para procesamiento eficiente.
    """
    texts: List[str] = Field(
        ..., 
        min_length=1, 
        max_length=10,
        description="Lista de textos a corregir"
    )
    segment_ids: Optional[List[str]] = Field(
        None, 
        description="IDs opcionales de segmentos"
    )
    
    @model_validator(mode='after')
    def validate_ids_length(self) -> 'CorrectionBatchRequest':
        """Verifica que segment_ids tenga la misma longitud que texts."""
        if self.segment_ids is not None:
            if len(self.segment_ids) != len(self.texts):
                raise ValueError(
                    f"segment_ids ({len(self.segment_ids)}) debe tener la misma "
                    f"longitud que texts ({len(self.texts)})"
                )
        return self


class LLMCorrectionMetadata(BaseModel):
    """
    Metadata de una corrección LLM para almacenar en el segmento.
    """
    original: str = Field(..., description="Texto original antes de corrección")
    cambios: List[str] = Field(default_factory=list, description="Cambios aplicados")
    confianza: float = Field(ge=0.0, le=1.0, description="Confianza de la corrección")
    modelo: str = Field(default="qwen3:8b", description="Modelo LLM usado")
    intentos: int = Field(default=1, ge=1, description="Número de intentos")
    timestamp: datetime = Field(default_factory=datetime.now, description="Momento de la corrección")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CacheEntry(BaseModel):
    """
    Entrada de caché para correcciones.
    """
    text_hash: str = Field(..., description="Hash del texto original")
    response: LLMCorrectionResponse = Field(..., description="Respuesta cacheada")
    created_at: datetime = Field(default_factory=datetime.now)
    hits: int = Field(default=0, ge=0, description="Número de veces usada")
    
    def increment_hits(self) -> None:
        """Incrementa el contador de hits."""
        self.hits += 1

