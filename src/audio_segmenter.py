"""
Módulo para segmentar archivos de audio de podcast en fragmentos de 10-15 segundos.
"""
import os
from pathlib import Path
from typing import List, Tuple
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
        
        # Cargar audio con librosa para obtener información
        audio, sr = librosa.load(input_path, sr=None)
        duration = len(audio) / sr
        
        # Si el audio es más corto que la duración mínima, crear un solo segmento
        if duration < self.min_duration:
            print(f"   ⚠️  Audio muy corto ({duration:.2f}s < {self.min_duration}s), creando un solo segmento")
            output_path, start, end = self._save_segment_from_array(
                audio, sr, output_dir, base_name, 0
            )
            return [(output_path, 0.0, duration)]
        
        # Cargar con pydub para segmentación inteligente
        audio_segment = AudioSegment.from_file(input_path)
        
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
            if current_duration >= self.min_duration:
                output_path, start, end = self._save_segment(
                    current_chunk, output_dir, base_name, segment_idx, sr
                )
                segments.append((output_path, start, end))
        
        # Si no se generaron segmentos con el método de silencios, usar segmentación fija
        if len(segments) == 0:
            segments = self._fixed_segmentation(audio, sr, output_dir, base_name)
        
        return segments
    
    def _save_segment(self, audio_segment: AudioSegment, output_dir: str, 
                     base_name: str, segment_idx: int, sample_rate: int) -> Tuple[str, float, float]:
        """
        Guarda un segmento de audio en formato WAV.
        
        Returns:
            Tupla (ruta_archivo, inicio, fin) en segundos
        """
        output_path = os.path.join(output_dir, f"{base_name}_segment_{segment_idx:04d}.wav")
        
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
        output_path = os.path.join(output_dir, f"{base_name}_segment_{segment_idx:04d}.wav")
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
                output_path = os.path.join(output_dir, f"{base_name}_segment_{segment_idx:04d}.wav")
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

