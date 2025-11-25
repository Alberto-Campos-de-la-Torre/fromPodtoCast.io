"""
Módulo para segunda etapa de revisión de segmentos de audio.
Detecta múltiples hablantes en segmentos individuales y los divide si es necesario.
"""
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

from speaker_diarizer import SpeakerDiarizer


class SegmentReviewer:
    """Clase para revisar y refinar segmentos de audio detectando múltiples hablantes."""
    
    def __init__(self, diarizer: Optional[SpeakerDiarizer] = None,
                 min_segment_duration: float = 5.0,
                 max_speakers_per_segment: int = 2,
                 transcriber: Optional[object] = None,
                 retranscribe_split: bool = False):
        """
        Inicializa el revisor de segmentos.
        
        Args:
            diarizer: Instancia de SpeakerDiarizer para detectar hablantes
            min_segment_duration: Duración mínima en segundos para mantener un segmento (default: 5.0)
            max_speakers_per_segment: Número máximo de hablantes permitidos por segmento (default: 2)
            transcriber: Instancia de AudioTranscriber para re-transcribir segmentos divididos (opcional)
            retranscribe_split: Si True, re-transcribe automáticamente los segmentos divididos (default: False)
        """
        self.diarizer = diarizer
        self.min_segment_duration = min_segment_duration
        self.max_speakers_per_segment = max_speakers_per_segment
        self.transcriber = transcriber
        self.retranscribe_split = retranscribe_split
    
    def review_segments(self, metadata: List[Dict], 
                       normalized_dir: str,
                       output_dir: Optional[str] = None,
                       diarization_result: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Revisa segmentos de audio para detectar múltiples hablantes y dividirlos si es necesario.
        
        Args:
            metadata: Lista de diccionarios con metadatos de segmentos
            normalized_dir: Directorio donde están los segmentos normalizados
            output_dir: Directorio donde guardar nuevos segmentos divididos (opcional)
            diarization_result: Resultado de diarización del audio completo (opcional)
        
        Returns:
            Lista actualizada de metadatos con segmentos revisados y divididos
        """
        if output_dir is None:
            output_dir = normalized_dir
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        reviewed_metadata = []
        segment_counter = {}
        
        print(f"\n{'='*60}")
        print("Segunda Etapa: Revisión de Segmentos")
        print(f"{'='*60}\n")
        print(f"Revisando {len(metadata)} segmentos...")
        print(f"  - Duración mínima: {self.min_segment_duration}s")
        print(f"  - Máximo hablantes por segmento: {self.max_speakers_per_segment}")
        
        global_diarization_available = bool(diarization_result)
        diarization_ready = self._is_diarization_ready()
        
        if global_diarization_available:
            print("  - Usando resultados de diarización global para la revisión")
        elif diarization_ready:
            print("  - Usando diarización directa por segmento (pyannote)")
        else:
            print("  - ⚠️  Diarización avanzada no disponible. Segunda etapa omitida.")
            print("        Activa 'use_diarization' y configura 'hf_token' para habilitarla.\n")
            return metadata
        
        print()
        
        for i, segment_meta in enumerate(tqdm(metadata, desc="Revisando segmentos")):
            segment_path = segment_meta.get('path', '')
            
            if not os.path.exists(segment_path):
                print(f"   ⚠️  Segmento no encontrado: {segment_path}")
                continue
            
            # Obtener podcast_id para mantener contador por podcast
            podcast_id = segment_meta.get('podcast_id', 'unknown')
            if podcast_id not in segment_counter:
                segment_counter[podcast_id] = 0
            
            segment_start = float(segment_meta.get('start', 0.0))
            segment_end = float(segment_meta.get('end', segment_start))
            
            # Revisar el segmento
            reviewed_segments = self._review_single_segment(
                segment_meta,
                segment_path,
                output_dir,
                podcast_id,
                segment_counter[podcast_id],
                segment_start,
                segment_end,
                diarization_result,
                global_diarization_available
            )
            
            # Actualizar contador
            segment_counter[podcast_id] += len(reviewed_segments)
            
            # Agregar segmentos revisados (solo los que cumplen con duración mínima)
            for reviewed_seg in reviewed_segments:
                if reviewed_seg['duration'] >= self.min_segment_duration:
                    reviewed_metadata.append(reviewed_seg)
                else:
                    # Eliminar archivo si es muy corto
                    if os.path.exists(reviewed_seg['path']):
                        try:
                            os.remove(reviewed_seg['path'])
                        except:
                            pass
        
        print(f"\n✓ Revisión completada:")
        print(f"  - Segmentos originales: {len(metadata)}")
        print(f"  - Segmentos finales: {len(reviewed_metadata)}")
        print(f"  - Segmentos descartados: {len(metadata) - len(reviewed_metadata)}\n")
        
        return reviewed_metadata
    
    def _review_single_segment(self, segment_meta: Dict, 
                               segment_path: str,
                               output_dir: str,
                               podcast_id: str,
                               base_segment_idx: int,
                               segment_start_absolute: float,
                               segment_end_absolute: float,
                               global_diarization: Optional[List[Dict]] = None,
                               global_diarization_available: bool = False) -> List[Dict]:
        """
        Revisa un segmento individual para detectar múltiples hablantes.
        
        Args:
            segment_meta: Metadatos del segmento original
            segment_path: Ruta al archivo de audio del segmento
            output_dir: Directorio donde guardar segmentos divididos
            podcast_id: ID del podcast
            base_segment_idx: Índice base para numerar nuevos segmentos
        
        Returns:
            Lista de metadatos de segmentos (puede ser 1 o más si se dividió)
        """
        # Cargar audio para obtener duración
        try:
            audio, sr = librosa.load(segment_path, sr=None, mono=True)
            duration = len(audio) / sr
        except Exception as e:
            print(f"   ✗ Error cargando {Path(segment_path).name}: {e}")
            return []
        
        # Si el segmento es muy corto, descartarlo directamente
        if duration < self.min_segment_duration:
            return []
        
        # Intentar utilizar resultados globales primero
        diarization_segments = []
        if global_diarization_available and global_diarization:
            diarization_segments = self._get_segment_diarization_from_global(
                global_diarization, segment_start_absolute, segment_end_absolute
            )
        
        # Si no hay resultados globales, ejecutar diarización directa
        diarization_source = "global" if diarization_segments else "segment"
        if not diarization_segments:
            if not self._is_diarization_ready():
                return [self._create_segment_metadata(
                    segment_meta, segment_path, 0.0, duration, base_segment_idx, podcast_id
                )]
            
            try:
                diarization_segments = self.diarizer.diarize(segment_path)
            except Exception as e:
                print(f"   ⚠️  Error en diarización de {Path(segment_path).name}: {e}")
                return [self._create_segment_metadata(
                    segment_meta, segment_path, 0.0, duration, base_segment_idx, podcast_id
                )]
        
        if not diarization_segments:
            return [self._create_segment_metadata(
                segment_meta, segment_path, 0.0, duration, base_segment_idx, podcast_id
            )]
        
        # Contar hablantes únicos
        unique_speakers = set(seg.get('speaker', 'SPEAKER_00') for seg in diarization_segments)
        num_speakers = len(unique_speakers)
        
        # Si hay más hablantes de los permitidos, dividir el segmento
        if num_speakers > self.max_speakers_per_segment:
            if diarization_source == "global":
                print(f"   ↪ Segmento {Path(segment_path).name} dividido usando diarización global.")
            return self._split_segment_by_speakers(
                segment_meta, segment_path, audio, sr, diarization_segments,
                output_dir, podcast_id, base_segment_idx, segment_start_absolute
            )
        else:
            # Si tiene 2 o menos hablantes, mantener el segmento pero actualizar speaker
            speaker_label = self._get_dominant_speaker(diarization_segments)
            speaker_id = self.diarizer.get_speaker_id(speaker_label) if self.diarizer else segment_meta.get('speaker', 0)
            
            segment_meta_copy = segment_meta.copy()
            segment_meta_copy['speaker'] = speaker_id
            segment_meta_copy['speaker_label'] = speaker_label
            
            return [segment_meta_copy]
    
    def _split_segment_by_speakers(self, segment_meta: Dict,
                                   segment_path: str,
                                   audio: np.ndarray,
                                   sr: int,
                                   diarization_result: List[Dict],
                                   output_dir: str,
                                   podcast_id: str,
                                   base_segment_idx: int,
                                   segment_start_absolute: float) -> List[Dict]:
        """
        Divide un segmento basándose en los cambios de hablantes detectados.
        
        Args:
            segment_meta: Metadatos del segmento original
            segment_path: Ruta al archivo de audio original
            audio: Array de audio cargado
            sr: Sample rate
            diarization_result: Resultado de diarización
            output_dir: Directorio donde guardar segmentos divididos
            podcast_id: ID del podcast
            base_segment_idx: Índice base para numerar nuevos segmentos
        
        Returns:
            Lista de metadatos de segmentos divididos
        """
        # Ordenar segmentos de diarización por tiempo
        sorted_segments = sorted(diarization_result, key=lambda x: x.get('start', 0.0))
        
        new_segments = []
        segment_idx = base_segment_idx
        
        for diar_seg in sorted_segments:
            start_time = diar_seg.get('start', 0.0)
            end_time = diar_seg.get('end', 0.0)
            speaker_label = diar_seg.get('speaker', 'SPEAKER_00')
            seg_duration = end_time - start_time
            
            # Solo crear segmento si cumple con duración mínima
            if seg_duration < self.min_segment_duration:
                continue
            
            # Extraer audio del segmento
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Guardar nuevo segmento
            new_segment_path = os.path.join(
                output_dir,
                f"seg_{segment_idx:04d}.wav"
            )
            
            try:
                sf.write(new_segment_path, segment_audio, sr)
                
                # Crear metadatos del nuevo segmento
                speaker_id = self.diarizer.get_speaker_id(speaker_label)
                
                # Transcribir el nuevo segmento si está habilitado
                segment_text = ''
                segment_language = segment_meta.get('language', 'unknown')
                
                if self.retranscribe_split and self.transcriber:
                    try:
                        transcription_result = self.transcriber.transcribe(new_segment_path)
                        segment_text = transcription_result.get('text', '')
                        segment_language = transcription_result.get('language', segment_language)
                    except Exception as e:
                        print(f"   ⚠️  Error transcribiendo segmento dividido: {e}")
                        segment_text = segment_meta.get('text', '')  # Fallback al texto original
                else:
                    # Marcar que necesita transcripción
                    segment_text = segment_meta.get('text', '')
                    if not segment_text:
                        segment_text = '[NEEDS_TRANSCRIPTION]'
                
                # Crear nuevo metadata
                new_meta = {
                    'text': segment_text,
                    'path': os.path.abspath(new_segment_path),
                    'speaker': speaker_id,
                    'speaker_label': speaker_label,
                    'start': float(segment_start_absolute + start_time),
                    'end': float(segment_start_absolute + end_time),
                    'duration': float(seg_duration),
                    'language': segment_language,
                    'podcast_id': podcast_id,
                    'segment_id': f"seg_{segment_idx:04d}",
                    'original_segment_id': segment_meta.get('segment_id', 'unknown'),
                    'split_from': Path(segment_path).name
                }
                
                new_segments.append(new_meta)
                segment_idx += 1
            
            except Exception as e:
                print(f"   ✗ Error guardando segmento dividido: {e}")
                continue
        
        # Si se dividió el segmento, eliminar el archivo original si es diferente
        if len(new_segments) > 0 and segment_path != new_segments[0]['path']:
            try:
                if os.path.exists(segment_path):
                    os.remove(segment_path)
            except:
                pass
        
        return new_segments
    
    def _get_dominant_speaker(self, diarization_result: List[Dict]) -> str:
        """
        Obtiene el hablante que más tiempo ocupa en el segmento.
        
        Args:
            diarization_result: Resultado de diarización
        
        Returns:
            Etiqueta del hablante dominante
        """
        speaker_duration = {}
        
        for seg in diarization_result:
            speaker = seg.get('speaker', 'SPEAKER_00')
            duration = seg.get('duration', 0.0)
            speaker_duration[speaker] = speaker_duration.get(speaker, 0.0) + duration
        
        if speaker_duration:
            return max(speaker_duration, key=speaker_duration.get)
        return 'SPEAKER_00'
    
    def _create_segment_metadata(self, original_meta: Dict,
                                segment_path: str,
                                start: float,
                                end: float,
                                segment_idx: int,
                                podcast_id: str) -> Dict:
        """
        Crea metadatos para un segmento.
        
        Args:
            original_meta: Metadatos originales
            segment_path: Ruta al archivo de audio
            start: Tiempo de inicio
            end: Tiempo de fin
            segment_idx: Índice del segmento
            podcast_id: ID del podcast
        
        Returns:
            Diccionario con metadatos del segmento
        """
        duration = end - start
        return {
            'text': original_meta.get('text', ''),
            'path': os.path.abspath(segment_path),
            'speaker': original_meta.get('speaker', 0),
            'speaker_label': original_meta.get('speaker_label', 'SPEAKER_00'),
            'start': float(start),
            'end': float(end),
            'duration': float(duration),
            'language': original_meta.get('language', 'unknown'),
            'podcast_id': podcast_id,
            'segment_id': f"seg_{segment_idx:04d}"
        }

    def _get_segment_diarization_from_global(self,
                                             diarization_result: List[Dict],
                                             segment_start: float,
                                             segment_end: float) -> List[Dict]:
        """Extrae y normaliza los segmentos de diarización global que se solapan con el segmento."""
        if not diarization_result:
            return []
        
        local_segments = []
        for seg in diarization_result:
            diar_start = seg.get('start', 0.0)
            diar_end = seg.get('end', 0.0)
            speaker = seg.get('speaker', 'SPEAKER_00')
            
            overlap_start = max(segment_start, diar_start)
            overlap_end = min(segment_end, diar_end)
            
            if overlap_start >= overlap_end:
                continue
            
            local_segments.append({
                'start': overlap_start - segment_start,
                'end': overlap_end - segment_start,
                'speaker': speaker,
                'duration': overlap_end - overlap_start
            })
        
        return local_segments

    def _is_diarization_ready(self) -> bool:
        """Indica si existe un diarizador avanzado disponible."""
        return self.diarizer is not None and getattr(self.diarizer, 'pipeline', None) is not None

