"""
Módulo para segmentar archivos de audio de podcast en fragmentos de 10-15 segundos.
Soporta segmentación por silencios o por resultados de diarización.
"""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence


class AudioSegmenter:
    """Clase para segmentar audio en fragmentos de duración específica."""
    
    def __init__(self, min_duration: float = 10.0, max_duration: float = 15.0, 
                 silence_thresh: float = -40.0, min_silence_len: int = 500):
        """
        Inicializa el segmentador de audio.
        
        Args:
            min_duration: Duración mínima de los segmentos en segundos (default: 10.0)
            max_duration: Duración máxima de los segmentos en segundos (default: 15.0)
            silence_thresh: Umbral de silencio en dB (default: -40.0)
            min_silence_len: Longitud mínima de silencio en ms para hacer un corte (default: 500)
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.silence_thresh = silence_thresh
        self.min_silence_len = min_silence_len
    
    def segment_audio(self, input_path: str, output_dir: str, 
                     base_name: str = None) -> List[Tuple[str, float, float]]:
        """
        Segmenta un archivo de audio en fragmentos de 10-15 segundos.
        
        Args:
            input_path: Ruta al archivo de audio de entrada
            output_dir: Directorio donde guardar los segmentos
            base_name: Nombre base para los archivos de salida (opcional)
        
        Returns:
            Lista de tuplas (ruta_segmento, inicio, fin) en segundos
        """
        # Crear directorio de salida si no existe
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Obtener nombre base del archivo si no se proporciona
        if base_name is None:
            base_name = Path(input_path).stem
        
        # Cargar audio - usar pydub primero para archivos grandes (mejor manejo de memoria)
        audio = None
        audio_segment = None
        sr = None
        duration = 0.0
        
        try:
            # Intentar primero con pydub (mejor para archivos grandes)
            print(f"   Cargando audio (puede tardar para archivos grandes)...")
            audio_segment = AudioSegment.from_file(input_path)
            sr = audio_segment.frame_rate
            duration = len(audio_segment) / 1000.0  # pydub usa milisegundos
            print(f"   ✓ Audio cargado con pydub: {duration:.2f}s, {sr}Hz")
        except Exception as e1:
            print(f"   ⚠️  Error cargando con pydub: {e1}")
            print(f"   Intentando con librosa...")
            try:
                # Fallback a librosa
                audio, sr = librosa.load(input_path, sr=None, mono=True)
                duration = len(audio) / sr
                print(f"   ✓ Audio cargado con librosa: {duration:.2f}s, {sr}Hz")
            except Exception as e2:
                print(f"   ✗ Error cargando audio con ambos métodos")
                raise Exception(f"No se pudo cargar el archivo de audio. Pydub: {e1}, Librosa: {e2}")
        
        # Si el audio es más corto que la duración mínima, crear un solo segmento
        if duration < self.min_duration:
            print(f"   ⚠️  Audio muy corto ({duration:.2f}s < {self.min_duration}s), creando un solo segmento")
            if audio_segment is not None:
                # Tenemos audio_segment de pydub
                output_path, start, end = self._save_segment(
                    audio_segment, output_dir, base_name, 0, sr
                )
            elif audio is not None:
                # Tenemos audio de librosa
                output_path, start, end = self._save_segment_from_array(
                    audio, sr, output_dir, base_name, 0
                )
            else:
                raise Exception("No se pudo cargar el audio")
            return [(output_path, 0.0, duration)]
        
        # Si no tenemos audio_segment pero sí audio de librosa, convertir
        if audio_segment is None and audio is not None:
            print(f"   Convirtiendo audio de librosa a formato pydub para segmentación...")
            # Normalizar a int16 para pydub
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_segment = AudioSegment(
                audio_int16.tobytes(),
                frame_rate=sr,
                sample_width=2,
                channels=1
            )
        
        # Verificar que tenemos audio_segment
        if audio_segment is None:
            raise Exception("No se pudo obtener audio_segment para segmentación")
        
        # Intentar segmentar por silencios primero
        chunks = split_on_silence(
            audio_segment,
            min_silence_len=self.min_silence_len,
            silence_thresh=self.silence_thresh,
            keep_silence=200  # Mantener 200ms de silencio al inicio/fin
        )
        
        segments = []
        current_chunk = AudioSegment.empty()
        segment_idx = 0
        
        for chunk in chunks:
            chunk_duration = len(chunk) / 1000.0  # Convertir a segundos
            
            # Si el chunk actual + nuevo chunk excede el máximo, guardar el actual
            if len(current_chunk) > 0:
                current_duration = len(current_chunk) / 1000.0
                if current_duration + chunk_duration > self.max_duration:
                    # Guardar el chunk actual si cumple con la duración mínima
                    if current_duration >= self.min_duration:
                        output_path, start, end = self._save_segment(
                            current_chunk, output_dir, base_name, segment_idx, sr
                        )
                        segments.append((output_path, start, end))
                        segment_idx += 1
                    current_chunk = AudioSegment.empty()
            
            # Agregar chunk al actual
            current_chunk += chunk
            current_duration = len(current_chunk) / 1000.0
            
            # Si el chunk actual alcanza la duración mínima, considerar guardarlo
            if current_duration >= self.min_duration:
                # Si está cerca del máximo o el siguiente chunk lo excedería, guardar
                if current_duration >= self.max_duration * 0.9:
                    output_path, start, end = self._save_segment(
                        current_chunk, output_dir, base_name, segment_idx, sr
                    )
                    segments.append((output_path, start, end))
                    segment_idx += 1
                    current_chunk = AudioSegment.empty()
        
        # Guardar el último chunk si cumple con la duración mínima
        if len(current_chunk) > 0:
            current_duration = len(current_chunk) / 1000.0
            # Si el último chunk excede el máximo, dividirlo
            if current_duration > self.max_duration:
                # Dividir en segmentos de tamaño máximo
                remaining = current_chunk
                while len(remaining) > 0:
                    segment_duration = min(self.max_duration * 1000, len(remaining))
                    segment_chunk = remaining[:int(segment_duration)]
                    remaining = remaining[int(segment_duration):]
                    
                    seg_duration = len(segment_chunk) / 1000.0
                    if seg_duration >= self.min_duration:
                        output_path, start, end = self._save_segment(
                            segment_chunk, output_dir, base_name, segment_idx, sr
                        )
                        segments.append((output_path, start, end))
                        segment_idx += 1
            elif current_duration >= self.min_duration:
                output_path, start, end = self._save_segment(
                    current_chunk, output_dir, base_name, segment_idx, sr
                )
                segments.append((output_path, start, end))
        
        # Si no se generaron segmentos con el método de silencios, usar segmentación fija
        if len(segments) == 0:
            segments = self._fixed_segmentation(audio, sr, output_dir, base_name)
        
        # Verificar que ningún segmento exceda el máximo
        final_segments = []
        cumulative_time = 0.0
        for seg_path, start, end in segments:
            duration = end - start
            if duration > self.max_duration:
                print(f"   ⚠️  Segmento {Path(seg_path).name} excede máximo ({duration:.2f}s > {self.max_duration}s), dividiendo...")
                # Recargar y dividir el segmento
                seg_audio, seg_sr = librosa.load(seg_path, sr=sr)
                sub_segments = self._fixed_segmentation(seg_audio, seg_sr, output_dir, base_name)
                # Ajustar timestamps
                for sub_path, sub_start, sub_end in sub_segments:
                    final_segments.append((sub_path, cumulative_time + sub_start, cumulative_time + sub_end))
                cumulative_time += duration
            else:
                final_segments.append((seg_path, cumulative_time, cumulative_time + duration))
                cumulative_time += duration
        
        return final_segments
    
    def _save_segment(self, audio_segment: AudioSegment, output_dir: str, 
                     base_name: str, segment_idx: int, sample_rate: int) -> Tuple[str, float, float]:
        """
        Guarda un segmento de audio en formato WAV.
        
        Returns:
            Tupla (ruta_archivo, inicio, fin) en segundos
        """
        # Formato escalable: seg_0000.wav, seg_0001.wav, etc.
        output_path = os.path.join(output_dir, f"seg_{segment_idx:04d}.wav")
        
        # Convertir a numpy array y normalizar
        audio_array = np.array(audio_segment.get_array_of_samples())
        if audio_segment.channels == 2:
            audio_array = audio_array.reshape((-1, 2)).mean(axis=1)
        
        # Normalizar a float32
        if audio_segment.sample_width == 1:
            audio_array = audio_array.astype(np.float32) / 128.0 - 1.0
        elif audio_segment.sample_width == 2:
            audio_array = audio_array.astype(np.float32) / 32768.0
        elif audio_segment.sample_width == 4:
            audio_array = audio_array.astype(np.float32) / 2147483648.0
        
        # Guardar con soundfile
        sf.write(output_path, audio_array, sample_rate)
        
        duration = len(audio_segment) / 1000.0
        return (output_path, 0.0, duration)
    
    def _save_segment_from_array(self, audio: np.ndarray, sr: int, output_dir: str,
                                base_name: str, segment_idx: int) -> Tuple[str, float, float]:
        """
        Guarda un segmento de audio desde un array numpy.
        
        Returns:
            Tupla (ruta_archivo, inicio, fin) en segundos
        """
        # Formato escalable: seg_0000.wav, seg_0001.wav, etc.
        output_path = os.path.join(output_dir, f"seg_{segment_idx:04d}.wav")
        duration = len(audio) / sr
        sf.write(output_path, audio, sr)
        return (output_path, 0.0, duration)
    
    def _fixed_segmentation(self, audio: np.ndarray, sr: int, output_dir: str, 
                           base_name: str) -> List[Tuple[str, float, float]]:
        """
        Segmentación fija cuando no se pueden detectar silencios apropiados.
        Divide el audio en segmentos de duración máxima, ajustando para evitar cortes bruscos.
        """
        segments = []
        duration = len(audio) / sr
        segment_idx = 0
        current_start = 0.0
        
        # Si el audio completo es más corto que min_duration, crear un solo segmento
        if duration < self.min_duration:
            output_path, start, end = self._save_segment_from_array(
                audio, sr, output_dir, base_name, segment_idx
            )
            return [(output_path, 0.0, duration)]
        
        while current_start < duration:
            # Calcular el final del segmento
            segment_end = min(current_start + self.max_duration, duration)
            segment_duration = segment_end - current_start
            
            # Solo guardar si cumple con la duración mínima
            if segment_duration >= self.min_duration:
                # Extraer segmento
                start_sample = int(current_start * sr)
                end_sample = int(segment_end * sr)
                segment_audio = audio[start_sample:end_sample]
                
                # Guardar segmento
                output_path = os.path.join(output_dir, f"seg_{segment_idx:04d}.wav")
                sf.write(output_path, segment_audio, sr)
                
                segments.append((output_path, current_start, segment_end))
                segment_idx += 1
            
            # Avanzar al siguiente segmento
            current_start = segment_end
        
        # Si quedó un segmento final más corto que min_duration pero mayor que 0,
        # agregarlo al último segmento si existe, o crear uno nuevo si es el único
        if current_start < duration and len(segments) > 0:
            # Agregar el resto al último segmento
            last_seg_path, last_start, last_end = segments[-1]
            remaining_duration = duration - last_end
            if remaining_duration > 0:
                # Recargar y extender el último segmento
                last_audio, _ = librosa.load(last_seg_path, sr=sr)
                remaining_start = int(last_end * sr)
                remaining_audio = audio[remaining_start:]
                extended_audio = np.concatenate([last_audio, remaining_audio])
                sf.write(last_seg_path, extended_audio, sr)
                segments[-1] = (last_seg_path, last_start, duration)
        
        return segments
    
    def segment_by_diarization(self, input_path: str, output_dir: str,
                               diarization_segments: List[Dict],
                               base_name: str = None) -> List[Tuple[str, float, float, str]]:
        """
        Segmenta audio basándose en resultados de diarización de hablantes.
        
        Args:
            input_path: Ruta al archivo de audio de entrada
            output_dir: Directorio donde guardar los segmentos
            diarization_segments: Lista de segmentos de diarización con 'start', 'end', 'speaker'
            base_name: Nombre base para los archivos de salida (opcional)
        
        Returns:
            Lista de tuplas (ruta_segmento, inicio, fin, speaker_id)
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if base_name is None:
            base_name = Path(input_path).stem
        
        # Validar y reparar audio si es necesario
        working_path = self._validate_and_prepare_audio(input_path)
        
        # Cargar audio
        print(f"   Cargando audio para segmentación por diarización...")
        try:
            audio, sr = librosa.load(working_path, sr=None, mono=True)
            duration = len(audio) / sr
            print(f"   ✓ Audio cargado: {duration:.2f}s, {sr}Hz")
        except Exception as e:
            raise Exception(f"No se pudo cargar el audio: {e}")
        
        if not diarization_segments:
            print("   ⚠️  Sin segmentos de diarización, usando segmentación por silencios")
            return self.segment_audio(input_path, output_dir, base_name)
        
        # Ordenar segmentos por tiempo de inicio
        sorted_segments = sorted(diarization_segments, key=lambda x: x.get('start', 0))
        
        # Procesar segmentos de diarización
        result_segments = []
        segment_idx = 0
        stats = {'kept': 0, 'discarded_short': 0, 'split': 0}
        
        print(f"   Procesando {len(sorted_segments)} segmentos de diarización...")
        
        for diar_seg in sorted_segments:
            start = float(diar_seg.get('start', 0))
            end = float(diar_seg.get('end', start))
            speaker = diar_seg.get('speaker', 'SPEAKER_00')
            seg_duration = end - start
            
            # Validar timestamps
            if start < 0:
                start = 0
            if end > duration:
                end = duration
            if end <= start:
                continue
            
            seg_duration = end - start
            
            # Descartar segmentos muy cortos
            if seg_duration < self.min_duration:
                stats['discarded_short'] += 1
                continue
            
            # Dividir segmentos muy largos
            if seg_duration > self.max_duration:
                sub_segments = self._split_long_segment(
                    audio, sr, start, end, speaker, output_dir, segment_idx
                )
                for sub_seg in sub_segments:
                    result_segments.append(sub_seg)
                    segment_idx += 1
                stats['split'] += 1
            else:
                # Guardar segmento normal
                seg_path = self._save_diarization_segment(
                    audio, sr, start, end, speaker, output_dir, segment_idx
                )
                if seg_path:
                    result_segments.append((seg_path, start, end, speaker))
                    segment_idx += 1
                    stats['kept'] += 1
        
        # Mostrar estadísticas
        print(f"   ✓ Segmentación completada:")
        print(f"     - Segmentos guardados: {stats['kept']}")
        print(f"     - Segmentos divididos: {stats['split']}")
        print(f"     - Descartados (< {self.min_duration}s): {stats['discarded_short']}")
        print(f"     - Total final: {len(result_segments)}")
        
        # Limpiar audio temporal si se creó
        if working_path != input_path and os.path.exists(working_path):
            try:
                os.remove(working_path)
            except:
                pass
        
        return result_segments
    
    def _validate_and_prepare_audio(self, audio_path: str) -> str:
        """
        Valida un archivo de audio y crea una copia limpia si está corrupto.
        
        Returns:
            Ruta al audio válido (original o reparado)
        """
        import shutil
        
        if not shutil.which('ffprobe') or not shutil.which('ffmpeg'):
            return audio_path
        
        # Verificar integridad con ffmpeg
        try:
            result = subprocess.run(
                ['ffmpeg', '-v', 'error', '-i', audio_path, '-f', 'null', '-'],
                capture_output=True, text=True, timeout=120
            )
            
            if 'Invalid data' in result.stderr or 'corrupt' in result.stderr.lower():
                print(f"   ⚠️  Audio corrupto detectado, reparando...")
                return self._repair_audio(audio_path)
            
            return audio_path
            
        except subprocess.TimeoutExpired:
            print(f"   ⚠️  Timeout validando audio, intentando reparar...")
            return self._repair_audio(audio_path)
        except Exception:
            return audio_path
    
    def _repair_audio(self, audio_path: str) -> str:
        """Repara un archivo de audio corrupto."""
        try:
            # Crear archivo temporal
            tmp_path = tempfile.mktemp(suffix='.wav')
            
            with open(tmp_path, 'wb') as out_file:
                process = subprocess.Popen(
                    ['ffmpeg', '-y', '-err_detect', 'ignore_err',
                     '-i', audio_path, '-vn', '-ar', '22050', '-ac', '1',
                     '-f', 'wav', '-'],
                    stdout=out_file, stderr=subprocess.PIPE
                )
                process.communicate(timeout=600)
            
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 10000:
                print(f"   ✓ Audio reparado exitosamente")
                return tmp_path
            
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return audio_path
            
        except Exception as e:
            print(f"   ⚠️  Error reparando audio: {e}")
            return audio_path
    
    def _save_diarization_segment(self, audio: np.ndarray, sr: int,
                                   start: float, end: float, speaker: str,
                                   output_dir: str, segment_idx: int) -> Optional[str]:
        """
        Guarda un segmento de audio basado en diarización.
        
        Returns:
            Ruta al archivo guardado o None si falló
        """
        try:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            # Validar índices
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if end_sample <= start_sample:
                return None
            
            segment_audio = audio[start_sample:end_sample]
            
            # Nombre del archivo: seg_XXXX_SPEAKER.wav
            # Limpiamos el nombre del speaker para evitar caracteres problemáticos
            speaker_clean = speaker.replace('/', '_').replace('\\', '_')
            output_path = os.path.join(output_dir, f"seg_{segment_idx:04d}_{speaker_clean}.wav")
            
            sf.write(output_path, segment_audio, sr)
            return output_path
            
        except Exception as e:
            print(f"   ⚠️  Error guardando segmento {segment_idx}: {e}")
            return None
    
    def _split_long_segment(self, audio: np.ndarray, sr: int,
                            start: float, end: float, speaker: str,
                            output_dir: str, start_idx: int) -> List[Tuple[str, float, float, str]]:
        """
        Divide un segmento largo en sub-segmentos de tamaño máximo.
        Intenta cortar en puntos de baja energía (silencios naturales).
        
        Returns:
            Lista de tuplas (path, start, end, speaker)
        """
        seg_duration = end - start
        result = []
        current_start = start
        sub_idx = 0
        
        while current_start < end:
            # Calcular fin del sub-segmento
            sub_end = min(current_start + self.max_duration, end)
            sub_duration = sub_end - current_start
            
            # Solo guardar si cumple duración mínima
            if sub_duration >= self.min_duration:
                # Intentar ajustar el corte a un punto de baja energía
                if sub_end < end:
                    sub_end = self._find_best_cut_point(
                        audio, sr, current_start, sub_end,
                        search_window=1.0  # Buscar en ±1 segundo
                    )
                
                seg_path = self._save_diarization_segment(
                    audio, sr, current_start, sub_end, speaker,
                    output_dir, start_idx + sub_idx
                )
                
                if seg_path:
                    result.append((seg_path, current_start, sub_end, speaker))
                    sub_idx += 1
            
            current_start = sub_end
        
        return result
    
    def _find_best_cut_point(self, audio: np.ndarray, sr: int,
                             start: float, target_end: float,
                             search_window: float = 1.0) -> float:
        """
        Encuentra el mejor punto de corte cerca de target_end basándose en energía.
        
        Returns:
            Tiempo óptimo de corte
        """
        # Definir ventana de búsqueda
        search_start = max(start + self.min_duration, target_end - search_window)
        search_end = min(target_end + search_window, len(audio) / sr)
        
        if search_end <= search_start:
            return target_end
        
        # Extraer audio de la ventana de búsqueda
        start_sample = int(search_start * sr)
        end_sample = int(search_end * sr)
        search_audio = audio[start_sample:end_sample]
        
        if len(search_audio) == 0:
            return target_end
        
        # Calcular energía en ventanas pequeñas (50ms)
        window_samples = int(0.05 * sr)
        energies = []
        
        for i in range(0, len(search_audio) - window_samples, window_samples // 2):
            window = search_audio[i:i + window_samples]
            energy = np.sqrt(np.mean(window ** 2))
            energies.append((i, energy))
        
        if not energies:
            return target_end
        
        # Encontrar el punto de mínima energía
        min_energy_idx = min(energies, key=lambda x: x[1])[0]
        best_cut = search_start + (min_energy_idx / sr)
        
        # Asegurar que el corte produce segmentos válidos
        if best_cut - start < self.min_duration:
            return target_end
        
        return best_cut

