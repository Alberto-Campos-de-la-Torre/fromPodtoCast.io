"""
Script principal que orquesta todo el proceso de preparación de datos para TTS.
"""
import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from audio_segmenter import AudioSegmenter
from audio_normalizer import AudioNormalizer
from transcriber import AudioTranscriber
from speaker_diarizer import SpeakerDiarizer
from segment_reviewer import SegmentReviewer
from voice_bank import VoiceBankManager


class PodcastProcessor:
    """Clase principal para procesar podcasts y generar datos de entrenamiento."""
    
    def __init__(self, config: Dict):
        """
        Inicializa el procesador de podcasts.
        
        Args:
            config: Diccionario con configuración del procesador
        """
        self.config = config
        self.voice_bank_manager = None
        
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
        
        # Banco de voces (opcional)
        use_voice_bank = config.get('use_voice_bank', False)
        if use_voice_bank:
            voice_bank_path = config.get('voice_bank_path', './data/output/voice_bank.json')
            voice_match_threshold = config.get('voice_match_threshold', 0.85)
            try:
                self.voice_bank_manager = VoiceBankManager(
                    bank_path=voice_bank_path,
                    match_threshold=voice_match_threshold
                )
            except Exception as e:
                print(f"⚠️  No se pudo inicializar VoiceBank ({voice_bank_path}): {e}")
                self.voice_bank_manager = None
                use_voice_bank = False
        else:
            print("Voice bank deshabilitado (use_voice_bank=false).")
        
        # Diarizador opcional (puede requerir token de HF)
        hf_token = config.get('hf_token', None)
        use_diarization = config.get('use_diarization', False)
        
        # Inicializar diarizador siempre (usará método simple si pyannote no está disponible)
        try:
            self.diarizer = SpeakerDiarizer(
                hf_token=hf_token if (use_diarization or hf_token) else None,
                device=config.get('device', None),
                voice_bank_manager=self.voice_bank_manager if use_voice_bank else None
            )
            # El diarizador siempre está disponible (método simple por defecto)
        except Exception as e:
            print(f"⚠️  Advertencia: No se pudo inicializar diarizador: {e}")
            print("   Continuando sin diarización...")
            self.diarizer = None
        
        # Inicializar revisor de segmentos (segunda etapa)
        use_segment_review = config.get('use_segment_review', True)
        if use_segment_review:
            self.segment_reviewer = SegmentReviewer(
                diarizer=self.diarizer,
                min_segment_duration=config.get('review_min_duration', 5.0),
                max_speakers_per_segment=config.get('review_max_speakers', 2),
                transcriber=self.transcriber,
                retranscribe_split=config.get('review_retranscribe', False)
            )
        else:
            self.segment_reviewer = None
    
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
        # Usar un ID único basado en el nombre del archivo (sin caracteres especiales)
        podcast_name = podcast_id or Path(input_audio_path).stem
        # Crear ID limpio para el directorio (solo alfanuméricos y guiones)
        podcast_id_clean = re.sub(r'[^a-zA-Z0-9_-]', '_', podcast_name)[:50]  # Limitar longitud
        
        segments_dir = os.path.join(output_dir, 'segments', podcast_id_clean)
        normalized_dir = os.path.join(output_dir, 'normalized', podcast_id_clean)
        Path(segments_dir).mkdir(parents=True, exist_ok=True)
        Path(normalized_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Procesando podcast: {Path(input_audio_path).name}")
        print(f"ID del podcast: {podcast_id_clean}")
        print(f"{'='*60}\n")
        
        metrics = {
            'podcast_id': podcast_id_clean,
            'input_audio': os.path.abspath(input_audio_path),
            'timestamp': datetime.utcnow().isoformat(),
            'segments': {},
            'diarization': {
                'enabled': bool(self.diarizer),
                'unique_speakers': 0,
                'segments': 0,
                'status': 'skipped'
            },
            'voice_bank': {
                'enabled': bool(self.voice_bank_manager),
                'matched': 0,
                'created': 0
            },
            'second_stage': {
                'enabled': bool(self.segment_reviewer),
                'applied': False,
                'segments_before': 0,
                'segments_after': 0
            }
        }
        
        # Paso 1: Segmentar audio
        print("1. Segmentando audio...")
        segments = self.segmenter.segment_audio(
            input_audio_path, 
            segments_dir,
            base_name=""  # Ya no usamos el nombre en los archivos
        )
        print(f"   ✓ Generados {len(segments)} segmentos\n")
        metrics['segments']['raw'] = len(segments)
        
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
        metrics['segments']['normalized'] = len(normalized_segments)
        
        # Paso 3: Diarización del audio original
        diarization_result = None
        if self.diarizer:
            print("3. Realizando diarización de hablantes...")
            try:
                diarization_result = self.diarizer.diarize(input_audio_path)
                if diarization_result:
                    unique_speakers = len(set(s.get('speaker', 'SPEAKER_00') for s in diarization_result))
                    print(f"   ✓ Identificados {unique_speakers} hablantes")
                    print(f"   ✓ Generados {len(diarization_result)} segmentos de diarización\n")
                    metrics['diarization'] = {
                        'enabled': True,
                        'unique_speakers': unique_speakers,
                        'segments': len(diarization_result),
                        'status': 'success'
                    }
                else:
                    print("   ⚠️  Diarización no generó resultados\n")
                    metrics['diarization']['status'] = 'empty_result'
            except Exception as e:
                print(f"   ✗ Error en diarización: {e}")
                import traceback
                traceback.print_exc()
                print()
                metrics['diarization']['status'] = 'error'
        else:
            print("3. Diarización de hablantes (saltada - no disponible)\n")
            metrics['diarization']['status'] = 'disabled'
        
        # Actualizar métricas del voice bank si corresponde
        if self.diarizer and getattr(self.diarizer, 'voice_bank_stats', None):
            vb_stats = self.diarizer.voice_bank_stats
            metrics['voice_bank'].update(vb_stats)
        
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
        metrics['segments']['transcribed'] = len(transcriptions)
        
        # Paso 5: Asignar speakers y generar metadatos finales
        print("5. Generando metadatos finales...")
        metadata = []
        
        for i, (norm_path, start, end) in enumerate(normalized_segments):
            transcription = transcriptions[i] if i < len(transcriptions) else {'text': '', 'language': 'unknown'}
            
            # Asignar speaker_id
            speaker_id = 0
            speaker_label = "SPEAKER_00"
            
            if self.diarizer and diarization_result and len(diarization_result) > 0:
                try:
                    speaker_label = self.diarizer.assign_speaker_to_segment(
                        norm_path, diarization_result,
                        segment_start=start,
                        segment_end=end
                    )
                    speaker_id = self.diarizer.get_speaker_id(speaker_label)
                except Exception as e:
                    print(f"   ⚠️  Advertencia: Error asignando speaker para {Path(norm_path).name}: {e}")
                    # Usar método alternativo: buscar speaker más cercano temporalmente
                    try:
                        # Encontrar el segmento de diarización que más se solapa con este segmento
                        best_speaker = "SPEAKER_00"
                        max_overlap = 0.0
                        for diar_seg in diarization_result:
                            diar_start = diar_seg.get('start', 0.0)
                            diar_end = diar_seg.get('end', 0.0)
                            overlap_start = max(start, diar_start)
                            overlap_end = min(end, diar_end)
                            if overlap_start < overlap_end:
                                overlap = overlap_end - overlap_start
                                if overlap > max_overlap:
                                    max_overlap = overlap
                                    best_speaker = diar_seg.get('speaker', 'SPEAKER_00')
                        speaker_label = best_speaker
                        speaker_id = self.diarizer.get_speaker_id(speaker_label)
                    except:
                        pass
            
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
                'podcast_id': podcast_id_clean,
                'segment_id': f"seg_{i:04d}"
            }
            
            # Solo agregar si tiene transcripción válida
            if transcription['text'].strip():
                metadata.append(entry)
        
        print(f"   ✓ Generados {len(metadata)} registros de metadata\n")
        metrics['segments']['metadata_before_review'] = len(metadata)
        metrics['second_stage']['segments_before'] = len(metadata)
        
        # Paso 6: Segunda etapa - Revisión de segmentos (opcional)
        if self.segment_reviewer:
            print("6. Segunda etapa: Revisando segmentos para detectar múltiples hablantes...")
            try:
                metadata = self.segment_reviewer.review_segments(
                    metadata,
                    normalized_dir,
                    normalized_dir,  # Guardar segmentos divididos en el mismo directorio
                    diarization_result=diarization_result
                )
                print(f"   ✓ Revisión completada: {len(metadata)} segmentos finales\n")
                metrics['second_stage']['applied'] = True
                metrics['second_stage']['segments_after'] = len(metadata)
            except Exception as e:
                print(f"   ✗ Error en revisión de segmentos: {e}")
                import traceback
                traceback.print_exc()
                print("   Continuando con segmentos originales...\n")
                metrics['second_stage']['segments_after'] = metrics['second_stage']['segments_before']
        else:
            print("6. Segunda etapa de revisión (saltada - deshabilitada)\n")
            metrics['second_stage']['segments_after'] = metrics['second_stage']['segments_before']
            metrics['second_stage']['applied'] = False
        
        # Guardar metadata específica del podcast
        podcast_metadata_path = self._save_podcast_metadata(metadata, output_dir, podcast_id_clean)
        metrics['outputs'] = {
            'metadata_path': podcast_metadata_path
        }
        metrics['segments']['metadata_after_review'] = len(metadata)
        
        # Guardar log de métricas
        metrics_path = self._write_podcast_metrics(metrics, output_dir, podcast_id_clean)
        metrics['outputs']['metrics_log'] = metrics_path
        
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

    def _save_podcast_metadata(self, metadata: List[Dict], output_dir: str, podcast_id: str) -> str:
        """Guarda metadata específica de un podcast."""
        metadata_dir = os.path.join(output_dir, 'metadata')
        Path(metadata_dir).mkdir(parents=True, exist_ok=True)
        podcast_metadata_path = os.path.join(metadata_dir, f"{podcast_id}.json")
        
        with open(podcast_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ Metadata del podcast guardada en: {podcast_metadata_path}")
        return podcast_metadata_path

    def _write_podcast_metrics(self, metrics: Dict, output_dir: str, podcast_id: str) -> str:
        """Guarda un log con métricas del procesamiento del podcast."""
        logs_dir = os.path.join(output_dir, 'logs')
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
        metrics_path = os.path.join(logs_dir, f"{podcast_id}.log")
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ Log de métricas guardado en: {metrics_path}")
        return metrics_path

