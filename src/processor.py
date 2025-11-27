"""
Script principal que orquesta todo el proceso de preparación de datos para TTS.

Optimizado con:
- Procesamiento por lotes para corrección LLM
- Caché de correcciones
- Paralelización opcional
"""
import os
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from audio_segmenter import AudioSegmenter
from audio_normalizer import AudioNormalizer
from transcriber import AudioTranscriber
from speaker_diarizer import SpeakerDiarizer
from segment_reviewer import SegmentReviewer
from voice_bank import VoiceBankManager
from text_preprocessor import TextPreprocessor
from text_corrector_llm import TextCorrectorLLM
from correction_cache import BatchCorrectionCache, get_global_cache


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
                max_speakers_per_segment=config.get('review_max_speakers', 1),
                transcriber=self.transcriber,
                retranscribe_split=config.get('review_retranscribe', False),
                min_speaker_purity=config.get('review_min_speaker_purity', 0.8)
            )
        else:
            self.segment_reviewer = None
        
        # Inicializar pre-procesador de texto
        text_config = config.get('text_preprocessing', {})
        if text_config.get('enabled', True):
            self.text_preprocessor = TextPreprocessor(
                glosario_path=text_config.get('glosario_path'),
                fix_punctuation=text_config.get('fix_punctuation', True),
                normalize_numbers=text_config.get('normalize_numbers', True),
                fix_spacing=text_config.get('fix_spacing', True),
                fix_capitalization=text_config.get('fix_capitalization', True)
            )
        else:
            self.text_preprocessor = None
        
        # Inicializar corrector LLM (opcional) con optimizaciones
        llm_config = config.get('llm_correction', {})
        self.llm_corrector = None
        self.llm_cache = None
        self.llm_use_batch = llm_config.get('use_batch', True)
        self.llm_batch_size = llm_config.get('batch_size', 5)
        self.llm_use_parallel = llm_config.get('use_parallel', False)
        self.llm_max_workers = llm_config.get('max_workers', 2)
        
        if llm_config.get('enabled', False):
            try:
                # Inicializar caché si está habilitado
                cache_enabled = llm_config.get('enable_cache', True)
                cache_file = llm_config.get('cache_file')
                
                if cache_enabled:
                    if not cache_file:
                        cache_file = os.path.join(
                            config.get('output_dir', './data/output'),
                            'llm_cache.json'
                        )
                    self.llm_cache = BatchCorrectionCache(
                        cache_file=cache_file,
                        max_entries=llm_config.get('cache_max_entries', 10000),
                        expire_days=llm_config.get('cache_expire_days', 30)
                    )
                
                self.llm_corrector = TextCorrectorLLM(
                    ollama_host=llm_config.get('ollama_host', 'http://192.168.1.81:11434'),
                    model=llm_config.get('model', 'qwen3:8b'),
                    glosario_path=text_config.get('glosario_path'),
                    timeout=llm_config.get('timeout', 120),
                    max_retries=llm_config.get('max_retries', 3),
                    batch_size=self.llm_batch_size,
                    enable_cache=cache_enabled,
                    cache_file=cache_file,
                    max_workers=self.llm_max_workers
                )
                self.llm_min_confidence = llm_config.get('min_confidence', 0.7)
                
                mode = "batch" if self.llm_use_batch else ("paralelo" if self.llm_use_parallel else "secuencial")
                cache_status = f", caché={'ON' if cache_enabled else 'OFF'}"
                print(f"✓ Corrector LLM inicializado ({llm_config.get('model', 'qwen3:8b')}, modo={mode}{cache_status})")
            except Exception as e:
                print(f"⚠️  No se pudo inicializar corrector LLM: {e}")
                self.llm_corrector = None
    
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
        
        # Paso 1: Diarización del audio original (antes de segmentar)
        diarization_result = None
        use_diarization_segmentation = self.config.get('segment_by_diarization', True)
        
        if self.diarizer:
            print("1. Realizando diarización de hablantes...")
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
                    use_diarization_segmentation = False
            except Exception as e:
                print(f"   ✗ Error en diarización: {e}")
                import traceback
                traceback.print_exc()
                print()
                metrics['diarization']['status'] = 'error'
                use_diarization_segmentation = False
        else:
            print("1. Diarización de hablantes (saltada - no disponible)\n")
            metrics['diarization']['status'] = 'disabled'
            use_diarization_segmentation = False
        
        # Paso 2: Segmentar audio (por diarización o por silencios)
        if use_diarization_segmentation and diarization_result:
            print("2. Segmentando audio por segmentos de habla...")
            segments = self.segmenter.segment_by_diarization(
                input_audio_path,
                segments_dir,
                diarization_result,
                base_name=""
            )
            # Los segmentos incluyen speaker: (path, start, end, speaker)
            print(f"   ✓ Generados {len(segments)} segmentos basados en diarización\n")
            metrics['segments']['raw'] = len(segments)
            metrics['segments']['method'] = 'diarization'
        else:
            print("2. Segmentando audio por detección de silencios...")
            raw_segments = self.segmenter.segment_audio(
                input_audio_path, 
                segments_dir,
                base_name=""
            )
            # Convertir a formato con speaker: (path, start, end, speaker)
            segments = [(path, start, end, 'SPEAKER_00') for path, start, end in raw_segments]
            print(f"   ✓ Generados {len(segments)} segmentos\n")
            metrics['segments']['raw'] = len(segments)
            metrics['segments']['method'] = 'silence'
        
        # Paso 2.5: Revisión de pureza de hablantes (ANTES de normalizar)
        # Detecta y divide segmentos con múltiples hablantes
        if self.segment_reviewer and diarization_result:
            print("2.5. Revisando pureza de hablantes...")
            metrics['second_stage']['segments_before'] = len(segments)
            try:
                segments = self.segment_reviewer.review_raw_segments(
                    segments,
                    segments_dir,
                    diarization_result
                )
                metrics['second_stage']['applied'] = True
                metrics['second_stage']['segments_after'] = len(segments)
            except Exception as e:
                print(f"   ✗ Error en revisión de pureza: {e}")
                import traceback
                traceback.print_exc()
                print("   Continuando con segmentos originales...\n")
                metrics['second_stage']['segments_after'] = metrics['second_stage']['segments_before']
        else:
            metrics['second_stage']['applied'] = False
            metrics['second_stage']['segments_before'] = len(segments)
            metrics['second_stage']['segments_after'] = len(segments)
        
        # Paso 3: Normalizar segmentos (y eliminar segmentos temporales)
        print("3. Normalizando segmentos...")
        normalized_segments = []
        for idx, segment_tuple in enumerate(tqdm(segments, desc="   Normalizando")):
            if len(segment_tuple) == 4:
                segment_path, start, end, speaker = segment_tuple
            else:
                segment_path, start, end = segment_tuple
                speaker = 'SPEAKER_00'
            
            # Simplificar speaker label para el nombre del archivo
            simplified_speaker = self._simplify_speaker_label(speaker)
            segment_name = f"seg_{idx:04d}_{simplified_speaker}.wav"
            normalized_path = os.path.join(normalized_dir, segment_name)
            
            try:
                self.normalizer.normalize_audio(segment_path, normalized_path)
                normalized_segments.append((normalized_path, start, end, speaker))
                # Eliminar segmento temporal después de normalizar
                if os.path.exists(segment_path) and segment_path != normalized_path:
                    os.remove(segment_path)
            except Exception as e:
                print(f"   ✗ Error normalizando {segment_name}: {e}")
        
        # Limpiar directorio de segmentos temporales si está vacío
        try:
            if os.path.exists(segments_dir) and not os.listdir(segments_dir):
                os.rmdir(segments_dir)
        except Exception:
            pass
        
        print(f"   ✓ Normalizados {len(normalized_segments)} segmentos\n")
        metrics['segments']['normalized'] = len(normalized_segments)
        
        # Actualizar métricas del voice bank si corresponde
        if self.diarizer and getattr(self.diarizer, 'voice_bank_stats', None):
            vb_stats = self.diarizer.voice_bank_stats
            metrics['voice_bank'].update(vb_stats)
        
        # Paso 4: Transcribir segmentos normalizados
        print("4. Transcribiendo segmentos...")
        transcriptions = []
        segment_files = [seg[0] for seg in normalized_segments]  # Extraer solo paths
        
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
        
        # Paso 4.5: Pre-procesamiento de texto
        if self.text_preprocessor:
            print("4.5. Pre-procesando transcripciones...")
            preprocessed_count = 0
            for trans in transcriptions:
                if trans.get('text'):
                    original = trans['text']
                    corrected, changes = self.text_preprocessor.preprocess(original)
                    trans['text'] = corrected
                    if changes:
                        trans['text_original'] = original
                        trans['text_changes'] = changes
                        preprocessed_count += 1
            
            preprocess_stats = self.text_preprocessor.get_stats()
            print(f"   ✓ Pre-procesados {preprocessed_count} textos")
            print(f"   ✓ Correcciones: puntuación={preprocess_stats.get('punctuation_fixed', 0)}, "
                  f"números={preprocess_stats.get('numbers_normalized', 0)}, "
                  f"glosario={preprocess_stats.get('glosario_applied', 0)}\n")
            
            metrics['text_preprocessing'] = {
                'enabled': True,
                'processed': preprocess_stats.get('processed', 0),
                'corrected': preprocessed_count,
                'stats': preprocess_stats
            }
        else:
            metrics['text_preprocessing'] = {'enabled': False}
        
        # Paso 4.6: Corrección con LLM (opcional) - OPTIMIZADO
        if self.llm_corrector:
            llm_start_time = time.time()
            print("4.6. Corrigiendo transcripciones con LLM...")
            
            # Extraer textos válidos con sus índices
            valid_indices = []
            texts_to_correct = []
            for i, trans in enumerate(transcriptions):
                if trans.get('text', '').strip():
                    valid_indices.append(i)
                    texts_to_correct.append(trans['text'])
            
            llm_corrected_count = 0
            llm_failed_count = 0
            cache_hits = 0
            
            if texts_to_correct:
                # Usar procesamiento por lotes optimizado
                if self.llm_use_batch:
                    print(f"   Modo: batch (tamaño={self.llm_batch_size})")
                    corrections = self.llm_corrector.correct_batch_optimized(
                        texts_to_correct,
                        batch_size=self.llm_batch_size
                    )
                elif self.llm_use_parallel:
                    print(f"   Modo: paralelo (workers={self.llm_max_workers})")
                    corrections = self.llm_corrector.correct_parallel(
                        texts_to_correct,
                        max_workers=self.llm_max_workers
                    )
                else:
                    print("   Modo: secuencial")
                    corrections = []
                    for text in tqdm(texts_to_correct, desc="   Corrigiendo"):
                        corrections.append(self.llm_corrector.correct(text))
                
                # Aplicar correcciones
                for idx, (original_idx, (corrected, meta)) in enumerate(zip(valid_indices, corrections)):
                    trans = transcriptions[original_idx]
                    original = trans['text']
                    
                    # Verificar si vino del caché
                    if meta.get('from_cache'):
                        cache_hits += 1
                    
                    # Solo aplicar si la confianza es suficiente
                    confianza = meta.get('confianza', 0)
                    if 'error' not in meta and confianza >= self.llm_min_confidence:
                        trans['text'] = corrected
                        if corrected != original:
                            trans['llm_correction'] = {
                                'original': original if 'text_original' not in trans else trans.get('text_original'),
                                'cambios': meta.get('cambios', []),
                                'confianza': confianza
                            }
                            llm_corrected_count += 1
                    elif 'error' in meta:
                        llm_failed_count += 1
            
            llm_elapsed = time.time() - llm_start_time
            llm_stats = self.llm_corrector.get_stats()
            
            print(f"   ✓ Corregidos {llm_corrected_count} textos con LLM")
            print(f"   ✓ Confianza promedio: {llm_stats.get('avg_confidence', 0):.2f}")
            print(f"   ✓ Tiempo: {llm_elapsed:.1f}s ({len(texts_to_correct)/max(llm_elapsed, 0.1):.1f} textos/s)")
            if cache_hits > 0:
                print(f"   ✓ Caché hits: {cache_hits}")
            if llm_stats.get('batch_calls', 0) > 0:
                print(f"   ✓ Llamadas batch: {llm_stats.get('batch_calls', 0)}")
            if llm_stats.get('pydantic_validations', 0) > 0:
                print(f"   🔷 Pydantic: {llm_stats.get('pydantic_validations', 0)} respuestas validadas")
            if llm_failed_count > 0:
                print(f"   ⚠️  Fallaron {llm_failed_count} correcciones\n")
            else:
                print()
            
            metrics['llm_correction'] = {
                'enabled': True,
                'model': self.llm_corrector.model,
                'corrected': llm_corrected_count,
                'failed': llm_failed_count,
                'avg_confidence': llm_stats.get('avg_confidence', 0),
                'total_changes': llm_stats.get('total_changes', 0),
                'cache_hits': cache_hits,
                'batch_calls': llm_stats.get('batch_calls', 0),
                'pydantic_validations': llm_stats.get('pydantic_validations', 0),
                'processing_time': round(llm_elapsed, 2),
                'mode': 'batch' if self.llm_use_batch else ('parallel' if self.llm_use_parallel else 'sequential')
            }
        else:
            metrics['llm_correction'] = {'enabled': False}
        
        # Paso 5: Generar metadatos finales
        print("5. Generando metadatos finales...")
        metadata = []
        
        for i, segment_tuple in enumerate(normalized_segments):
            # Extraer información del segmento
            if len(segment_tuple) == 4:
                norm_path, start, end, speaker_label = segment_tuple
            else:
                norm_path, start, end = segment_tuple
                speaker_label = 'SPEAKER_00'
            
            transcription = transcriptions[i] if i < len(transcriptions) else {'text': '', 'language': 'unknown'}
            
            # Obtener speaker_id numérico
            speaker_id = 0
            if self.diarizer:
                try:
                    speaker_id = self.diarizer.get_speaker_id(speaker_label)
                except:
                    # Extraer número del label si es posible
                    match = re.search(r'(\d+)', speaker_label)
                    if match:
                        speaker_id = int(match.group(1))
            
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
        # Nota: second_stage ya fue procesada en paso 2.5 (antes de normalizar)
        
        # Paso 6: Limpieza y renombrado de segmentos
        print("6. Limpiando y estandarizando nombres de segmentos...")
        metadata, cleanup_stats = self._cleanup_segments(metadata, normalized_dir)
        print(f"   ✓ Segmentos finales: {len(metadata)}")
        if cleanup_stats['removed'] > 0:
            print(f"   ✓ Eliminados {cleanup_stats['removed']} segmentos sin speaker válido")
        if cleanup_stats['renamed'] > 0:
            print(f"   ✓ Renombrados {cleanup_stats['renamed']} segmentos\n")
        else:
            print()
        
        # Guardar metadata específica del podcast
        podcast_metadata_path = self._save_podcast_metadata(metadata, output_dir, podcast_id_clean)
        metrics['outputs'] = {
            'metadata_path': podcast_metadata_path
        }
        metrics['segments']['metadata_after_review'] = len(metadata)
        metrics['cleanup'] = cleanup_stats
        
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
    
    def _cleanup_segments(self, metadata: List[Dict], normalized_dir: str) -> Tuple[List[Dict], Dict]:
        """
        Limpia y estandariza los segmentos:
        - Elimina segmentos sin speaker válido
        - Renombra archivos con formato consistente: seg_XXXX_SPK_YY.wav
        
        Returns:
            Tuple (metadata_limpia, estadísticas)
        """
        stats = {'removed': 0, 'renamed': 0, 'kept': 0}
        cleaned_metadata = []
        
        for i, entry in enumerate(metadata):
            speaker_label = entry.get('speaker_label', '')
            old_path = entry.get('path', '')
            
            # Verificar que el archivo existe
            if not os.path.exists(old_path):
                stats['removed'] += 1
                continue
            
            # Verificar que tiene speaker válido
            if not speaker_label or speaker_label == 'SPEAKER_00':
                # Intentar obtener speaker del nombre del archivo
                filename = Path(old_path).stem
                match = re.search(r'SPEAKER_(?:GLOBAL_)?(\d+)|SPK_(\d+)', filename)
                if match:
                    num = match.group(1) or match.group(2)
                    speaker_label = f"SPK_{int(num):02d}"
                else:
                    # Sin speaker válido, eliminar
                    try:
                        os.remove(old_path)
                    except:
                        pass
                    stats['removed'] += 1
                    continue
            
            # Simplificar speaker label
            simplified_label = self._simplify_speaker_label(speaker_label)
            
            # Generar nuevo nombre de archivo
            new_filename = f"seg_{i:04d}_{simplified_label}.wav"
            new_path = os.path.join(normalized_dir, new_filename)
            
            # Renombrar si es diferente
            if old_path != new_path:
                try:
                    if os.path.exists(new_path):
                        os.remove(new_path)
                    os.rename(old_path, new_path)
                    stats['renamed'] += 1
                except Exception as e:
                    # Si falla el renombrado, mantener el original
                    new_path = old_path
            
            # Actualizar metadata
            entry['path'] = new_path
            entry['speaker_label'] = simplified_label
            entry['segment_id'] = f"seg_{i:04d}_{simplified_label}"
            
            cleaned_metadata.append(entry)
            stats['kept'] += 1
        
        return cleaned_metadata, stats
    
    def _simplify_speaker_label(self, label: str) -> str:
        """Simplifica el label del speaker a formato SPK_XX."""
        # Si ya está en formato SPK_XX, mantenerlo
        if re.match(r'^SPK_\d{2}$', label):
            return label
        
        # Extraer número de SPEAKER_GLOBAL_XXX o SPEAKER_XX
        match = re.search(r'SPEAKER_(?:GLOBAL_)?(\d+)', label)
        if match:
            num = int(match.group(1))
            return f"SPK_{num:02d}"
        
        # Extraer cualquier número
        match = re.search(r'(\d+)', label)
        if match:
            num = int(match.group(1))
            return f"SPK_{num:02d}"
        
        # Fallback
        return "SPK_00"

