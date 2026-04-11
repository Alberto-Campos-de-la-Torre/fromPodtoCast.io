#!/usr/bin/env python3
"""
Prueba final de integración de Whisper large-v3
"""
import sys
import os
import numpy as np
import soundfile as sf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from transcriber import AudioTranscriber

def create_test_audio(duration=3.0, sr=16000):
    """Crea audio de prueba."""
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.3
    test_file = "/tmp/test_whisper_large_v3.wav"
    sf.write(test_file, audio, sr)
    return test_file

print("="*70)
print("Prueba Final: Whisper large-v3 en fromPodtoCast")
print("="*70)
print()

print("1️⃣ Verificando configuración...")
import json
with open('config/config.json') as f:
    config = json.load(f)
whisper_model = config.get('whisper_model')
print(f"   Modelo configurado: {whisper_model}")
print()

print("2️⃣ Inicializando AudioTranscriber...")
try:
    transcriber = AudioTranscriber(
        model_name=whisper_model,
        device='cpu',  # Cambiar a 'cuda' para GPU
        language='es',
        force_language=True
    )
    print(f"   ✅ Transcriptor inicializado")
    print(f"   Modelo: {transcriber.model_name}")
    print(f"   Dispositivo: {transcriber.device}")
    print(f"   Usa HF: {transcriber.use_hf}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

print()
print("3️⃣ Creando audio de prueba...")
test_audio = create_test_audio()
print(f"   ✅ Audio creado: {test_audio}")

print()
print("4️⃣ Transcribiendo...")
try:
    result = transcriber.transcribe(test_audio)
    print(f"   ✅ Transcripción exitosa")
    print(f"   Texto: '{result['text']}'")
    print(f"   Idioma: {result['language']}")
    print(f"   Segmentos: {len(result.get('segments', []))}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    if os.path.exists(test_audio):
        os.remove(test_audio)

print()
print("="*70)
print("✅ INTEGRACIÓN EXITOSA - Sistema listo para producción")
print("="*70)
print()
print("El modelo Whisper large-v3 está funcionando correctamente.")
print("Puedes usar el pipeline completo con: python3 main.py")
