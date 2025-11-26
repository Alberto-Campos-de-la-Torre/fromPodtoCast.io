"""
Módulo para transcribir audio a texto usando modelos de STT.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import whisper
import torch

# Forzar uso de /usr/bin/ffmpeg en lugar de snap
if os.path.exists('/usr/bin/ffmpeg'):
    os.environ['PATH'] = '/usr/bin:' + os.environ.get('PATH', '')


class AudioTranscriber:
    """Clase para transcribir audio a texto."""
    
    def __init__(self, model_name: str = "base", device: Optional[str] = None, 
                 language: Optional[str] = None):
        """
        Inicializa el transcriptor de audio.
        
        Args:
            model_name: Nombre del modelo Whisper a usar (tiny, base, small, medium, large)
            device: Dispositivo a usar ('cuda', 'cpu', o None para auto-detectar)
            language: Idioma del audio (None para auto-detectar)
        """
        self.model_name = model_name
        self.language = language
        
        # Detectar dispositivo
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Cargar modelo
        print(f"Cargando modelo Whisper '{model_name}' en {self.device}...")
        self.model = whisper.load_model(model_name, device=self.device)
        print("Modelo cargado exitosamente.")
    
    def _prepare_audio_path(self, audio_path: str) -> tuple:
        """
        Prepara el archivo de audio para transcripción.
        Si está en un sistema de archivos NTFS (disco externo), lo copia a /tmp
        para evitar problemas con ffmpeg snap.
        
        Returns:
            Tuple (path_a_usar, es_temporal)
        """
        # Verificar si está en /media (típicamente NTFS)
        if audio_path.startswith('/media/'):
            # Copiar a /tmp para evitar problemas con ffmpeg snap
            tmp_path = os.path.join(tempfile.gettempdir(), f"whisper_audio_{os.getpid()}_{Path(audio_path).name}")
            try:
                shutil.copy2(audio_path, tmp_path)
                return tmp_path, True
            except Exception:
                return audio_path, False
        return audio_path, False
    
    def transcribe(self, audio_path: str, **kwargs) -> Dict:
        """
        Transcribe un archivo de audio a texto.
        
        Args:
            audio_path: Ruta al archivo de audio
            **kwargs: Argumentos adicionales para whisper.transcribe()
        
        Returns:
            Diccionario con la transcripción y metadatos
        """
        # Preparar audio (copiar a /tmp si es necesario)
        working_path, is_temp = self._prepare_audio_path(audio_path)
        
        try:
            # Configuración por defecto
            transcribe_options = {
                'language': self.language,
                'task': 'transcribe',
                'fp16': self.device == 'cuda',
                **kwargs
            }
            
            # Transcribir
            result = self.model.transcribe(working_path, **transcribe_options)
            
            return {
                'text': result['text'].strip(),
                'language': result.get('language', 'unknown'),
                'segments': result.get('segments', []),
                'audio_path': audio_path
            }
        finally:
            # Limpiar archivo temporal
            if is_temp and os.path.exists(working_path):
                try:
                    os.remove(working_path)
                except:
                    pass
    
    def transcribe_batch(self, audio_files: list, **kwargs) -> list:
        """
        Transcribe múltiples archivos de audio.
        
        Args:
            audio_files: Lista de rutas a archivos de audio
            **kwargs: Argumentos adicionales para whisper.transcribe()
        
        Returns:
            Lista de diccionarios con transcripciones
        """
        results = []
        for i, audio_path in enumerate(audio_files, 1):
            print(f"Transcribiendo {i}/{len(audio_files)}: {Path(audio_path).name}")
            try:
                result = self.transcribe(audio_path, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Error transcribiendo {audio_path}: {e}")
                results.append({
                    'audio_path': audio_path,
                    'text': '',
                    'error': str(e)
                })
        
        return results

