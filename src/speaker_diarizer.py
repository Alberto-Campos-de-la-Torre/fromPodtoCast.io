"""
Módulo para identificar y etiquetar diferentes narradores (speakers) en audio.
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
import torchaudio
import numpy as np
import warnings

# Suprimir warnings de deprecación de torchaudio
warnings.filterwarnings('ignore', category=UserWarning, module='torchaudio')
warnings.filterwarnings('ignore', category=UserWarning, module='pyannote')
warnings.filterwarnings('ignore', category=UserWarning, module='lightning')

# Habilitar TF32 para mejor rendimiento en GPUs Ampere+ (suprime el warning)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Importación opcional de pyannote.audio
try:
    from pyannote.audio import Pipeline, Model, Inference
    from pyannote.core import Segment
    PYANNOTE_AVAILABLE = True
    PYANNOTE_EMBED_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    PYANNOTE_EMBED_AVAILABLE = False
    Pipeline = None
    Model = None
    Inference = None
    Segment = None

from voice_bank import VoiceBankManager


class SpeakerDiarizer:
    """Clase para realizar diarización de hablantes."""
    
    def __init__(self, hf_token: Optional[str] = None, device: Optional[str] = None,
                 voice_bank_manager: Optional[VoiceBankManager] = None):
        """
        Inicializa el diarizador de hablantes.
        
        Args:
            hf_token: Token de Hugging Face (necesario para modelos privados)
            device: Dispositivo a usar ('cuda', 'cpu', o None para auto-detectar)
            voice_bank_manager: Manejador del banco global de voces
        """
        # Detectar dispositivo
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.hf_token = hf_token
        self.voice_bank_manager = voice_bank_manager
        self.embedding_inference: Optional[Inference] = None
        self.voice_bank_stats = {
            'enabled': bool(voice_bank_manager),
            'matched': 0,
            'created': 0
        }
        
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
                    token=hf_token
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

        self._init_embedding_model()
    
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
            
            # pyannote 3.x puede devolver diferentes tipos de objetos
            annotation = None
            
            if hasattr(diarization, 'speaker_diarization'):
                # DiarizeOutput object (pyannote 3.x) - acceder a la anotación directamente
                annotation = diarization.speaker_diarization
            elif hasattr(diarization, 'itertracks'):
                # Annotation object (pyannote < 3.x o algunos casos)
                annotation = diarization
            
            if annotation is not None and hasattr(annotation, 'itertracks'):
                for turn, _, speaker in annotation.itertracks(yield_label=True):
                    segments.append({
                        'start': turn.start,
                        'end': turn.end,
                        'speaker': speaker,
                        'duration': turn.end - turn.start
                    })
            
            if not segments:
                print("   ⚠️  pyannote no devolvió segmentos, usando método simple")
                return self._simple_diarization(audio_path)
            
            segments = self._maybe_assign_global_ids(audio_path, segments)
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
                try:
                    waveform, sample_rate = torchaudio.load_with_torchcodec(segment_path)
                except (AttributeError, TypeError):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
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

    # ------------------------------------------------------------------ #
    # Voice bank helpers
    # ------------------------------------------------------------------ #
    def _init_embedding_model(self):
        """Inicializa el modelo de embeddings si hay banco de voces."""
        if not self.voice_bank_manager:
            return
        if not PYANNOTE_EMBED_AVAILABLE or Model is None or Inference is None:
            print("⚠️  pyannote/embedding no disponible, voice bank deshabilitado.")
            self.voice_bank_manager = None
            self.voice_bank_stats['enabled'] = False
            return
        if not self.hf_token:
            print("⚠️  Voice bank requiere hf_token para cargar pyannote/embedding.")
            self.voice_bank_manager = None
            self.voice_bank_stats['enabled'] = False
            return
        try:
            model = Model.from_pretrained(
                "pyannote/embedding",
                token=self.hf_token
            )
            # pyannote 3.x requiere torch.device en lugar de string
            device = torch.device(self.device) if isinstance(self.device, str) else self.device
            self.embedding_inference = Inference(model, window="whole", device=device)
            print("   ✓ Modelo de embeddings cargado para voice bank.")
        except Exception as e:
            print(f"⚠️  No se pudo cargar pyannote/embedding: {e}")
            self.voice_bank_manager = None
            self.voice_bank_stats['enabled'] = False

    def _maybe_assign_global_ids(self, audio_path: str, segments: List[Dict]) -> List[Dict]:
        """Asigna IDs globales mediante el banco de voces."""
        if not self.voice_bank_manager or not self.embedding_inference or Segment is None:
            return segments
        
        speaker_groups: Dict[str, List[Dict]] = {}
        for seg in segments:
            speaker_groups.setdefault(seg['speaker'], []).append(seg)
        
        assignments: Dict[str, str] = {}
        matched = created = 0
        
        for local_label, segs in speaker_groups.items():
            embedding = self._compute_embedding(audio_path, segs)
            if embedding is None:
                continue
            
            # Validar embedding antes de procesar
            if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                print(f"   ⚠️  Embedding inválido para {local_label}, saltando...")
                continue
            
            best_id, score = self.voice_bank_manager.find_best_match(embedding)
            if best_id:
                self.voice_bank_manager.update_speaker(best_id, embedding)
                assignments[local_label] = best_id
                matched += 1
            else:
                new_id = self.voice_bank_manager.add_speaker(embedding)
                if new_id:  # add_speaker puede retornar None si el embedding es inválido
                    assignments[local_label] = new_id
                    created += 1
        
        # Aplicar asignaciones y crear fallback para speakers sin global ID
        if assignments:
            for seg in segments:
                original_label = seg['speaker']
                seg['speaker_local'] = original_label
                if original_label in assignments:
                    seg['speaker'] = assignments[original_label]
                else:
                    # Crear ID simplificado para speakers sin embedding válido
                    # Extraer número del label local
                    match = re.search(r'(\d+)', original_label)
                    if match:
                        num = int(match.group(1))
                        seg['speaker'] = f"SPK_{num:02d}"
                    else:
                        seg['speaker'] = original_label
        
        if matched or created:
            print(f"   Voice bank: {matched} hablantes reutilizados, {created} nuevos.")
        self.voice_bank_stats['matched'] = matched
        self.voice_bank_stats['created'] = created
        return segments

    def _compute_embedding(self, audio_path: str, speaker_segments: List[Dict]) -> Optional[np.ndarray]:
        """Calcula embedding promedio para los segmentos de un hablante."""
        if not self.embedding_inference:
            return None
        
        embeddings = []
        MIN_SEGMENT_FOR_EMBEDDING = 2.0  # Mínimo 2 segundos para extraer embedding confiable
        
        # Cargar audio completo una vez
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    waveform, sample_rate = torchaudio.load_with_torchcodec(audio_path)
                except (AttributeError, TypeError):
                    waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            print(f"   ⚠️  Error cargando audio para embeddings: {e}")
            return None
        
        # Filtrar segmentos muy cortos y ordenar por duración (más largos primero)
        valid_segments = []
        for seg in speaker_segments:
            start = float(seg.get('start', 0.0))
            end = float(seg.get('end', start))
            duration = end - start
            if duration >= MIN_SEGMENT_FOR_EMBEDDING:
                valid_segments.append((seg, duration))
        
        # Ordenar por duración descendente y tomar máximo 5 segmentos más largos
        valid_segments.sort(key=lambda x: x[1], reverse=True)
        valid_segments = valid_segments[:5]
        
        if not valid_segments:
            # Si no hay segmentos largos, intentar con todos los disponibles
            for seg in speaker_segments:
                start = float(seg.get('start', 0.0))
                end = float(seg.get('end', start))
                if end - start >= 1.0:  # Mínimo 1 segundo
                    valid_segments.append((seg, end - start))
        
        if not valid_segments:
            return None
        
        import tempfile
        for seg, duration in valid_segments:
            start = float(seg.get('start', 0.0))
            end = float(seg.get('end', start))
            
            try:
                # Extraer segmento del waveform
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                segment_waveform = waveform[:, start_sample:end_sample]
                
                # Verificar que el segmento tiene contenido
                if segment_waveform.numel() < sample_rate:  # Menos de 1 segundo
                    continue
                
                # Guardar segmento temporal
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    torchaudio.save(tmp_path, segment_waveform, sample_rate)
                
                    # Extraer embedding del segmento
                    emb = self.embedding_inference({'uri': Path(tmp_path).stem, 'audio': tmp_path})
                
                emb_array = np.array(emb, dtype=np.float32)
                
                # Validar embedding extraído
                if not np.any(np.isnan(emb_array)) and not np.any(np.isinf(emb_array)):
                    embeddings.append(emb_array)
                
                # Limpiar archivo temporal
                os.remove(tmp_path)
                
            except Exception as e:
                continue
        
        if not embeddings:
            return None
        
        # Calcular embedding promedio
        mean_emb = np.mean(embeddings, axis=0)
        
        # Validar resultado
        if np.any(np.isnan(mean_emb)) or np.any(np.isinf(mean_emb)):
            return None
        
        norm = np.linalg.norm(mean_emb)
        if norm == 0 or np.isnan(norm):
            return None
        
        return mean_emb / norm

    def get_voice_bank_stats(self) -> Dict:
        """
        Retorna estadísticas del voice bank para el último audio procesado.
        
        Returns:
            Dict con 'enabled', 'matched' y 'created'
        """
        return self.voice_bank_stats.copy()

