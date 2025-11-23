"""
Módulo para identificar y etiquetar diferentes narradores (speakers) en audio.
"""
import os
from pathlib import Path
from typing import List, Dict, Optional
import torch
import torchaudio
import numpy as np

# Importación opcional de pyannote.audio
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError as e:
    PYANNOTE_AVAILABLE = False
    Pipeline = None
    print(f"Advertencia: pyannote.audio no está disponible: {e}")
    print("La diarización avanzada no estará disponible, se usará método simple.")


class SpeakerDiarizer:
    """Clase para realizar diarización de hablantes."""
    
    def __init__(self, hf_token: Optional[str] = None, device: Optional[str] = None):
        """
        Inicializa el diarizador de hablantes.
        
        Args:
            hf_token: Token de Hugging Face (necesario para modelos privados)
            device: Dispositivo a usar ('cuda', 'cpu', o None para auto-detectar)
        """
        # Detectar dispositivo
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.hf_token = hf_token
        
        # Cargar pipeline de diarización
        if not PYANNOTE_AVAILABLE:
            print("pyannote.audio no está disponible, usando método simple de diarización.")
            self.pipeline = None
            return
        
        print("Cargando pipeline de diarización...")
        try:
            if hf_token:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
            else:
                # Intentar cargar sin token (puede fallar si el modelo es privado)
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1"
                )
            self.pipeline.to(torch.device(self.device))
            print("Pipeline de diarización cargado exitosamente.")
        except Exception as e:
            print(f"Error cargando pipeline de diarización: {e}")
            print("Usando método alternativo basado en energía...")
            self.pipeline = None
    
    def diarize(self, audio_path: str, min_speakers: Optional[int] = None,
                max_speakers: Optional[int] = None) -> List[Dict]:
        """
        Realiza diarización de hablantes en un archivo de audio.
        
        Args:
            audio_path: Ruta al archivo de audio
            min_speakers: Número mínimo de hablantes (opcional)
            max_speakers: Número máximo de hablantes (opcional)
        
        Returns:
            Lista de diccionarios con información de segmentos y hablantes
        """
        if self.pipeline is None:
            # Método alternativo: asignar speaker_id basado en energía
            return self._simple_diarization(audio_path)
        
        # Usar pipeline de pyannote
        try:
            diarization = self.pipeline(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            # Convertir resultado a formato estándar
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker,
                    'duration': turn.end - turn.start
                })
            
            return segments
        except Exception as e:
            print(f"Error en diarización con pyannote: {e}")
            return self._simple_diarization(audio_path)
    
    def _simple_diarization(self, audio_path: str) -> List[Dict]:
        """
        Método alternativo simple de diarización basado en energía.
        Asigna un speaker_id único por segmento (útil cuando no hay acceso a pyannote).
        """
        # Cargar audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convertir a mono si es estéreo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Calcular energía en ventanas
        window_size = int(sample_rate * 0.5)  # 0.5 segundos
        energy = []
        for i in range(0, waveform.shape[1], window_size):
            window = waveform[0, i:i+window_size]
            energy.append(torch.mean(window**2).item())
        
        # Asignar speaker_id basado en cambios de energía (simplificado)
        # En producción, usar pyannote o similar
        segments = [{
            'start': 0.0,
            'end': len(waveform[0]) / sample_rate,
            'speaker': 'SPEAKER_00',
            'duration': len(waveform[0]) / sample_rate
        }]
        
        return segments
    
    def assign_speaker_to_segment(self, segment_path: str, 
                                  diarization_result: List[Dict]) -> str:
        """
        Asigna un speaker_id a un segmento de audio basado en resultados de diarización.
        
        Args:
            segment_path: Ruta al segmento de audio
            diarization_result: Resultado de diarización del audio completo
        
        Returns:
            ID del hablante asignado
        """
        # Cargar información del segmento
        waveform, sample_rate = torchaudio.load(segment_path)
        segment_duration = len(waveform[0]) / sample_rate
        
        # Obtener timestamp del nombre del archivo si está disponible
        # (asumiendo que el nombre contiene información de tiempo)
        # Por ahora, usar el speaker más común en el rango temporal del segmento
        
        if not diarization_result:
            return "SPEAKER_00"
        
        # Encontrar el speaker que más tiempo ocupa en el segmento
        # (simplificado: usar el primer speaker encontrado)
        # En producción, calcular el overlap temporal real
        
        speaker_counts = {}
        for seg in diarization_result:
            speaker = seg.get('speaker', 'SPEAKER_00')
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + seg.get('duration', 0)
        
        # Retornar el speaker más común
        if speaker_counts:
            most_common_speaker = max(speaker_counts, key=speaker_counts.get)
            return most_common_speaker
        
        return "SPEAKER_00"
    
    def get_speaker_id(self, speaker_label: str) -> int:
        """
        Convierte una etiqueta de speaker (ej: 'SPEAKER_00') a un ID numérico.
        
        Args:
            speaker_label: Etiqueta del hablante
        
        Returns:
            ID numérico del hablante
        """
        try:
            # Extraer número de la etiqueta
            if 'SPEAKER_' in speaker_label:
                return int(speaker_label.split('_')[1])
            else:
                # Si no tiene formato estándar, usar hash
                return hash(speaker_label) % 1000
        except:
            return 0

