"""
Módulo para transcribir audio a texto usando modelos de STT.
"""
import os
from pathlib import Path
from typing import Optional, Dict
import whisper
import torch


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
    
    def transcribe(self, audio_path: str, **kwargs) -> Dict:
        """
        Transcribe un archivo de audio a texto.
        
        Args:
            audio_path: Ruta al archivo de audio
            **kwargs: Argumentos adicionales para whisper.transcribe()
        
        Returns:
            Diccionario con la transcripción y metadatos
        """
        # Configuración por defecto
        transcribe_options = {
            'language': self.language,
            'task': 'transcribe',
            'fp16': self.device == 'cuda',
            **kwargs
        }
        
        # Transcribir
        result = self.model.transcribe(audio_path, **transcribe_options)
        
        return {
            'text': result['text'].strip(),
            'language': result.get('language', 'unknown'),
            'segments': result.get('segments', []),
            'audio_path': audio_path
        }
    
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

