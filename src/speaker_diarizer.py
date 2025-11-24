"""
Módulo para identificar y etiquetar diferentes narradores (speakers) en audio.
"""
import os
from pathlib import Path
from typing import List, Dict, Optional
import torch
import torchaudio
import numpy as np
import warnings

# Suprimir warnings de deprecación de torchaudio
warnings.filterwarnings('ignore', category=UserWarning, module='torchaudio')

# Importación opcional de pyannote.audio
try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment
    PYANNOTE_AVAILABLE = True
except ImportError as e:
    PYANNOTE_AVAILABLE = False
    Pipeline = None
    Segment = None


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
            print("⚠️  pyannote.audio no está disponible, usando método simple de diarización.")
            self.pipeline = None
            return
        
        # Intentar cargar pyannote solo si hay token o está explícitamente solicitado
        if hf_token:
            print("🔄 Cargando pipeline de diarización de pyannote.audio...")
            try:
                model_name = "pyannote/speaker-diarization-3.1"
                print(f"   Usando token de Hugging Face para {model_name}")
                self.pipeline = Pipeline.from_pretrained(
                    model_name,
                    use_auth_token=hf_token
                )
                
                # Mover pipeline al dispositivo correcto
                if self.device == "cuda" and torch.cuda.is_available():
                    self.pipeline = self.pipeline.to(torch.device(self.device))
                    print(f"   ✓ Pipeline cargado en {self.device}")
                else:
                    self.pipeline = self.pipeline.to(torch.device("cpu"))
                    print(f"   ✓ Pipeline cargado en CPU")
                
                print("✅ Pipeline de diarización cargado exitosamente.")
            except Exception as e:
                print(f"⚠️  Error cargando pipeline de pyannote: {e}")
                print("   Usando método simple de diarización basado en energía...")
                self.pipeline = None
        else:
            # Sin token, usar método simple directamente (sin intentar cargar pyannote)
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
        Asigna diferentes speaker_ids basados en cambios de energía y volumen.
        """
        try:
            # Cargar audio usando la nueva API de TorchCodec si está disponible
            try:
                # Intentar usar torchaudio.load_with_torchcodec (nueva API)
                waveform, sample_rate = torchaudio.load_with_torchcodec(audio_path)
            except (AttributeError, TypeError):
                # Fallback a la API antigua si la nueva no está disponible
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convertir a mono si es estéreo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            duration = len(waveform[0]) / sample_rate
            
            # Calcular energía en ventanas de 2 segundos
            window_size = int(sample_rate * 2.0)  # 2 segundos
            energy = []
            timestamps = []
            
            for i in range(0, waveform.shape[1], window_size):
                window = waveform[0, i:i+window_size]
                energy_value = torch.mean(window**2).item()
                energy.append(energy_value)
                timestamps.append(i / sample_rate)
            
            if len(energy) == 0:
                # Si no hay energía calculada, retornar un solo segmento
                return [{
                    'start': 0.0,
                    'end': duration,
                    'speaker': 'SPEAKER_00',
                    'duration': duration
                }]
            
            # Calcular umbral de energía (mediana)
            energy_array = np.array(energy)
            energy_threshold = np.median(energy_array)
            
            # Identificar cambios significativos de energía (posibles cambios de hablante)
            speaker_changes = []
            current_speaker = 0
            
            for i, e in enumerate(energy):
                if i == 0:
                    speaker_changes.append((timestamps[i], current_speaker))
                else:
                    # Si hay un cambio significativo de energía, cambiar de speaker
                    energy_diff = abs(e - energy[i-1])
                    if energy_diff > energy_threshold * 0.5:  # Cambio significativo
                        current_speaker = (current_speaker + 1) % 3  # Máximo 3 speakers
                    speaker_changes.append((timestamps[i], current_speaker))
            
            # Crear segmentos de diarización
            segments = []
            for i in range(len(speaker_changes)):
                start_time = speaker_changes[i][0]
                speaker_id = speaker_changes[i][1]
                
                if i < len(speaker_changes) - 1:
                    end_time = speaker_changes[i+1][0]
                else:
                    end_time = duration
                
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'speaker': f'SPEAKER_{speaker_id:02d}',
                    'duration': end_time - start_time
                })
            
            return segments
            
        except Exception as e:
            print(f"   ⚠️  Error en diarización simple: {e}")
            # Retornar un solo segmento como fallback
            try:
                try:
                    waveform, sample_rate = torchaudio.load_with_torchcodec(audio_path)
                except (AttributeError, TypeError):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        waveform, sample_rate = torchaudio.load(audio_path)
                duration = len(waveform[0]) / sample_rate if waveform.shape[0] > 0 else 0.0
            except:
                duration = 0.0
            
            return [{
                'start': 0.0,
                'end': duration,
                'speaker': 'SPEAKER_00',
                'duration': duration
            }]
    
    def assign_speaker_to_segment(self, segment_path: str, 
                                  diarization_result: List[Dict],
                                  segment_start: float = 0.0,
                                  segment_end: Optional[float] = None) -> str:
        """
        Asigna un speaker_id a un segmento de audio basado en resultados de diarización.
        
        Args:
            segment_path: Ruta al segmento de audio
            diarization_result: Resultado de diarización del audio completo
            segment_start: Tiempo de inicio del segmento en el audio original (segundos)
            segment_end: Tiempo de fin del segmento en el audio original (segundos)
        
        Returns:
            ID del hablante asignado
        """
        if not diarization_result:
            return "SPEAKER_00"
        
        # Si no se proporciona segment_end, calcularlo desde el archivo
        if segment_end is None:
            try:
                waveform, sample_rate = torchaudio.load(segment_path)
                segment_duration = len(waveform[0]) / sample_rate
                segment_end = segment_start + segment_duration
            except Exception:
                segment_end = segment_start + 15.0  # Asumir 15 segundos por defecto
        
        # Encontrar el speaker que más tiempo ocupa en el rango del segmento
        speaker_overlap = {}
        
        for seg in diarization_result:
            seg_start = seg.get('start', 0.0)
            seg_end = seg.get('end', 0.0)
            speaker = seg.get('speaker', 'SPEAKER_00')
            
            # Calcular overlap entre el segmento y el resultado de diarización
            overlap_start = max(segment_start, seg_start)
            overlap_end = min(segment_end, seg_end)
            
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                speaker_overlap[speaker] = speaker_overlap.get(speaker, 0.0) + overlap_duration
        
        # Retornar el speaker con mayor overlap
        if speaker_overlap:
            most_common_speaker = max(speaker_overlap, key=speaker_overlap.get)
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

