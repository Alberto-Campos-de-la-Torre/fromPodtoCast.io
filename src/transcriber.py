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
                 language: Optional[str] = None, force_language: bool = False):
        """
        Inicializa el transcriptor de audio.
        
        Args:
            model_name: Nombre del modelo Whisper a usar (tiny, base, small, medium, large)
            device: Dispositivo a usar ('cuda', 'cpu', o None para auto-detectar)
            language: Idioma del audio (None para auto-detectar, 'es' para español)
            force_language: Si True, fuerza el idioma especificado ignorando auto-detección.
                           Útil para evitar falsos positivos de catalán/gallego en español.
        """
        self.model_name = model_name
        self.language = language
        self.force_language = force_language
        
        # Si force_language está activo pero no hay idioma, usar español por defecto
        if self.force_language and self.language is None:
            self.language = 'es'
            print(f"⚠️  force_language activo sin idioma especificado, usando 'es' (español)")
        
        # Detectar dispositivo
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Cargar modelo
        lang_msg = f", idioma={'forzado: ' + self.language if self.force_language else 'auto-detectar'}"
        print(f"Cargando modelo Whisper '{model_name}' en {self.device}{lang_msg}...")
        
        self.use_hf = False
        if os.path.isdir(model_name):
            print(f"📂 Detectado modelo local Hugging Face: {model_name}")
            try:
                from transformers import pipeline as hf_pipeline
                
                # Configurar tipo de dato para torch
                torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
                
                self.pipeline = hf_pipeline(
                    "automatic-speech-recognition",
                    model=model_name,
                    device=self.device,
                    torch_dtype=torch_dtype,
                    chunk_length_s=30,
                )
                self.use_hf = True
                print("✓ Modelo HF cargado exitosamente.")
            except ImportError as e:
                print(f"❌ Error: transformers no está instalado. Ejecuta: pip install transformers")
                raise RuntimeError(f"transformers es requerido para modelos HF locales: {e}")
            except Exception as e:
                print(f"⚠️  Error cargando modelo HF local: {e}")
                print("   Intentando cargar con validación safetensors relajada...")
                try:
                    from transformers import pipeline as hf_pipeline
                    # Intento secundario desactivando safetensors si falla
                    self.pipeline = hf_pipeline(
                        "automatic-speech-recognition",
                        model=model_name,
                        device=self.device,
                        torch_dtype=torch_dtype,
                        chunk_length_s=30,
                        model_kwargs={"use_safetensors": False}
                    )
                    self.use_hf = True
                    print("✓ Modelo HF cargado exitosamente (sin safetensors).")
                except Exception as e2:
                    raise RuntimeError(f"No se pudo cargar el modelo HF: {e2}")
        else:
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
            if self.use_hf:
                # --- Lógica para Hugging Face Pipeline ---
                generate_kwargs = {"task": "transcribe"}
                if self.language:
                    generate_kwargs["language"] = self.language
                
                # Ejecutar pipeline
                result = self.pipeline(
                    working_path, 
                    return_timestamps=True, 
                    generate_kwargs=generate_kwargs
                )
                
                # Adaptar formato de segmentos (HF chunks -> Whisper segments)
                segments = []
                for chunk in result.get('chunks', []):
                    # timestamp puede ser (start, end) o None
                    ts = chunk.get('timestamp', (0.0, 0.0))
                    if isinstance(ts, (list, tuple)) and len(ts) == 2:
                        start, end = ts
                    else:
                        start, end = 0.0, 0.0
                        
                    segments.append({
                        'start': start if start is not None else 0.0,
                        'end': end if end is not None else 0.0,
                        'text': chunk.get('text', ''),
                        'seek': 0
                    })
                
                return {
                    'text': result['text'].strip(),
                    'language': self.language if self.language else 'unknown',
                    'segments': segments,
                    'audio_path': audio_path
                }
            else:
                # --- Lógica original OpenAI Whisper ---
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

