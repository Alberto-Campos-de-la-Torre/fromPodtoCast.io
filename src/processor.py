"""
Script principal que orquesta todo el proceso de preparación de datos para TTS.
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from audio_segmenter import AudioSegmenter
from audio_normalizer import AudioNormalizer
from transcriber import AudioTranscriber
from speaker_diarizer import SpeakerDiarizer


class PodcastProcessor:
    """Clase principal para procesar podcasts y generar datos de entrenamiento."""
    
    def __init__(self, config: Dict):
        """
        Inicializa el procesador de podcasts.
        
        Args:
            config: Diccionario con configuración del procesador
        """
        self.config = config
        
        # Inicializar componentes
        self.segmenter = AudioSegmenter(
            min_duration=config.get('min_duration', 10.0),
            max_duration=config.get('max_duration', 15.0),
            silence_thresh=config.get('silence_thresh', -40.0),
            min_silence_len=config.get('min_silence_len', 500)
        )
        
        self.normalizer = AudioNormalizer(
            target_sr=config.get('target_sr', 22050),
            target_lufs=config.get('target_lufs', -23.0),
            normalize_peak=config.get('normalize_peak', True)
        )
        
        self.transcriber = AudioTranscriber(
            model_name=config.get('whisper_model', 'base'),
            device=config.get('device', None),
            language=config.get('language', None)
        )
        
        # Diarizador opcional (puede requerir token de HF)
        hf_token = config.get('hf_token', None)
        if hf_token or config.get('use_diarization', False):
            try:
                self.diarizer = SpeakerDiarizer(
                    hf_token=hf_token,
                    device=config.get('device', None)
                )
            except Exception as e:
                print(f"Advertencia: No se pudo inicializar diarizador: {e}")
                self.diarizer = None
        else:
            self.diarizer = None
    
    def process_podcast(self, input_audio_path: str, output_dir: str, 
                       podcast_id: Optional[str] = None) -> List[Dict]:
        """
        Procesa un archivo de podcast completo.
        
        Args:
            input_audio_path: Ruta al archivo de podcast
            output_dir: Directorio donde guardar los resultados
            podcast_id: ID único del podcast (opcional)
        
        Returns:
            Lista de diccionarios con metadatos de los segmentos procesados
        """
        # Crear estructura de directorios
        podcast_name = podcast_id or Path(input_audio_path).stem
        segments_dir = os.path.join(output_dir, 'segments', podcast_name)
        normalized_dir = os.path.join(output_dir, 'normalized', podcast_name)
        Path(segments_dir).mkdir(parents=True, exist_ok=True)
        Path(normalized_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Procesando podcast: {Path(input_audio_path).name}")
        print(f"{'='*60}\n")
        
        # Paso 1: Segmentar audio
        print("1. Segmentando audio...")
        segments = self.segmenter.segment_audio(
            input_audio_path, 
            segments_dir,
            base_name=podcast_name
        )
        print(f"   ✓ Generados {len(segments)} segmentos\n")
        
        # Paso 2: Normalizar segmentos
        print("2. Normalizando segmentos...")
        normalized_segments = []
        for segment_path, start, end in tqdm(segments, desc="   Normalizando"):
            segment_name = Path(segment_path).name
            normalized_path = os.path.join(normalized_dir, segment_name)
            
            try:
                self.normalizer.normalize_audio(segment_path, normalized_path)
                normalized_segments.append((normalized_path, start, end))
            except Exception as e:
                print(f"   ✗ Error normalizando {segment_name}: {e}")
        
        print(f"   ✓ Normalizados {len(normalized_segments)} segmentos\n")
        
        # Paso 3: Diarización del audio original (si está habilitado)
        diarization_result = None
        if self.diarizer:
            print("3. Realizando diarización de hablantes...")
            try:
                diarization_result = self.diarizer.diarize(input_audio_path)
                print(f"   ✓ Identificados {len(set(s['speaker'] for s in diarization_result))} hablantes\n")
            except Exception as e:
                print(f"   ✗ Error en diarización: {e}\n")
        
        # Paso 4: Transcribir segmentos normalizados
        print("4. Transcribiendo segmentos...")
        transcriptions = []
        segment_files = [path for path, _, _ in normalized_segments]
        
        for i, segment_path in enumerate(tqdm(segment_files, desc="   Transcribiendo")):
            try:
                result = self.transcriber.transcribe(segment_path)
                transcriptions.append({
                    'segment_path': segment_path,
                    'text': result['text'],
                    'language': result.get('language', 'unknown')
                })
            except Exception as e:
                print(f"   ✗ Error transcribiendo {Path(segment_path).name}: {e}")
                transcriptions.append({
                    'segment_path': segment_path,
                    'text': '',
                    'language': 'unknown',
                    'error': str(e)
                })
        
        print(f"   ✓ Transcritos {len(transcriptions)} segmentos\n")
        
        # Paso 5: Asignar speakers y generar metadatos finales
        print("5. Generando metadatos finales...")
        metadata = []
        
        for i, (norm_path, start, end) in enumerate(normalized_segments):
            transcription = transcriptions[i] if i < len(transcriptions) else {'text': '', 'language': 'unknown'}
            
            # Asignar speaker_id
            speaker_id = 0
            speaker_label = "SPEAKER_00"
            
            if self.diarizer and diarization_result:
                try:
                    speaker_label = self.diarizer.assign_speaker_to_segment(
                        norm_path, diarization_result
                    )
                    speaker_id = self.diarizer.get_speaker_id(speaker_label)
                except Exception as e:
                    print(f"   Advertencia: Error asignando speaker: {e}")
            
            # Crear entrada de metadata
            entry = {
                'text': transcription['text'],
                'path': os.path.abspath(norm_path),
                'speaker': speaker_id,
                'speaker_label': speaker_label,
                'start': float(start),
                'end': float(end),
                'duration': float(end - start),
                'language': transcription.get('language', 'unknown'),
                'podcast_id': podcast_name
            }
            
            # Solo agregar si tiene transcripción válida
            if transcription['text'].strip():
                metadata.append(entry)
        
        print(f"   ✓ Generados {len(metadata)} registros de metadata\n")
        
        return metadata
    
    def process_batch(self, input_audio_files: List[str], output_dir: str) -> List[Dict]:
        """
        Procesa múltiples archivos de podcast.
        
        Args:
            input_audio_files: Lista de rutas a archivos de podcast
            output_dir: Directorio donde guardar los resultados
        
        Returns:
            Lista de todos los metadatos generados
        """
        all_metadata = []
        
        for i, audio_file in enumerate(input_audio_files, 1):
            print(f"\n[{i}/{len(input_audio_files)}] Procesando: {Path(audio_file).name}")
            
            try:
                metadata = self.process_podcast(audio_file, output_dir)
                all_metadata.extend(metadata)
            except Exception as e:
                print(f"✗ Error procesando {audio_file}: {e}")
        
        return all_metadata
    
    def save_metadata(self, metadata: List[Dict], output_path: str):
        """
        Guarda los metadatos en un archivo JSON.
        
        Args:
            metadata: Lista de diccionarios con metadatos
            output_path: Ruta donde guardar el archivo JSON
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"✓ Metadata guardado en: {output_path}")
        print(f"  Total de registros: {len(metadata)}")
        print(f"{'='*60}\n")

