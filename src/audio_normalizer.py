"""
Módulo para normalizar archivos de audio: bitrate, sample rate y niveles de audio.
"""
import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional


class AudioNormalizer:
    """Clase para normalizar audio a estándares consistentes."""
    
    def __init__(self, target_sr: int = 22050, target_bitrate: Optional[int] = None,
                 target_lufs: float = -23.0, normalize_peak: bool = True):
        """
        Inicializa el normalizador de audio.
        
        Args:
            target_sr: Sample rate objetivo en Hz (default: 22050)
            target_bitrate: Bitrate objetivo (no usado directamente, pero documentado)
            target_lufs: Nivel LUFS objetivo para normalización de loudness (default: -23.0)
            normalize_peak: Si True, normaliza el pico a -1.0 dB (default: True)
        """
        self.target_sr = target_sr
        self.target_bitrate = target_bitrate
        self.target_lufs = target_lufs
        self.normalize_peak = normalize_peak
    
    def normalize_audio(self, input_path: str, output_path: str) -> dict:
        """
        Normaliza un archivo de audio.
        
        Args:
            input_path: Ruta al archivo de audio de entrada
            output_path: Ruta donde guardar el audio normalizado
        
        Returns:
            Diccionario con información de la normalización realizada
        """
        # Crear directorio de salida si no existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Cargar audio
        audio, sr = librosa.load(input_path, sr=None, mono=True)
        
        # Información original
        original_sr = sr
        original_duration = len(audio) / sr
        original_peak = np.max(np.abs(audio))
        original_rms = np.sqrt(np.mean(audio**2))
        
        # Resamplear si es necesario
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        
        # Normalizar niveles de audio
        # Primero, normalización de loudness (LUFS aproximado usando RMS)
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms > 0:
            # Convertir RMS a LUFS aproximado (simplificado)
            # LUFS ≈ 20 * log10(RMS) - 0.691 (aproximación)
            current_lufs = 20 * np.log10(current_rms) - 0.691
            target_rms = 10 ** ((self.target_lufs + 0.691) / 20)
            
            # Aplicar ganancia para alcanzar el nivel objetivo
            if current_rms > 0:
                gain = target_rms / current_rms
                audio = audio * gain
        
        # Normalización de pico (evitar clipping)
        if self.normalize_peak:
            peak = np.max(np.abs(audio))
            if peak > 0:
                # Normalizar a -1.0 dB (0.89125 en escala lineal)
                max_peak = 10 ** (-1.0 / 20)
                if peak > max_peak:
                    audio = audio * (max_peak / peak)
        
        # Asegurar que el audio esté en el rango [-1, 1]
        audio = np.clip(audio, -1.0, 1.0)
        
        # Guardar audio normalizado
        sf.write(output_path, audio, sr, subtype='PCM_16')  # 16-bit PCM
        
        # Información de normalización
        final_peak = np.max(np.abs(audio))
        final_rms = np.sqrt(np.mean(audio**2))
        final_lufs = 20 * np.log10(final_rms) - 0.691 if final_rms > 0 else -np.inf
        
        return {
            'original_sample_rate': original_sr,
            'target_sample_rate': self.target_sr,
            'original_peak': float(original_peak),
            'final_peak': float(final_peak),
            'original_rms': float(original_rms),
            'final_rms': float(final_rms),
            'final_lufs': float(final_lufs),
            'duration': float(original_duration),
            'output_path': output_path
        }
    
    def normalize_batch(self, input_files: list, output_dir: str) -> list:
        """
        Normaliza múltiples archivos de audio.
        
        Args:
            input_files: Lista de rutas a archivos de audio
            output_dir: Directorio donde guardar los archivos normalizados
        
        Returns:
            Lista de diccionarios con información de normalización
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = []
        for input_path in input_files:
            filename = Path(input_path).name
            output_path = os.path.join(output_dir, filename)
            
            try:
                result = self.normalize_audio(input_path, output_path)
                results.append(result)
            except Exception as e:
                print(f"Error normalizando {input_path}: {e}")
                results.append({
                    'input_path': input_path,
                    'error': str(e)
                })
        
        return results

