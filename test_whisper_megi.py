#!/usr/bin/env python3
"""
Script de prueba para verificar que el modelo Whisper_Megi_IA funciona correctamente.
"""
import sys
import os
import numpy as np
import soundfile as sf
from pathlib import Path

# Añadir el directorio src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from transcriber import AudioTranscriber

def create_test_audio(duration=3.0, sr=16000):
    """Crea un archivo de audio de prueba con un tono simple."""
    t = np.linspace(0, duration, int(sr * duration))
    # Tono de 440 Hz (La)
    audio = np.sin(2 * np.pi * 440 * t) * 0.3
    
    test_file = "/tmp/test_audio_whisper.wav"
    sf.write(test_file, audio, sr)
    return test_file

def main():
    print("=" * 70)
    print("Verificando Whisper_Megi_IA")
    print("=" * 70)
    
    model_path = "/media/ttech-main/8464b39b-ba5b-49c5-9e05-010122b9874d/Whisper_Megi_IA_HF"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: No se encuentra el modelo en {model_path}")
        return 1
    
    print(f"\n1. Inicializando AudioTranscriber con modelo: {model_path}")
    try:
        transcriber = AudioTranscriber(
            model_name=model_path,
            device="cpu",  # Usar CPU para la prueba rápida
            language="es",
            force_language=True
        )
        print("   ✓ Transcriptor inicializado correctamente")
    except Exception as e:
        print(f"   ❌ Error al inicializar: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n2. Creando audio de prueba...")
    test_audio = create_test_audio()
    print(f"   ✓ Audio creado: {test_audio}")
    
    print("\n3. Transcribiendo audio de prueba...")
    try:
        result = transcriber.transcribe(test_audio)
        print(f"   ✓ Transcripción exitosa!")
        print(f"   - Texto: '{result['text']}'")
        print(f"   - Idioma detectado: {result['language']}")
        print(f"   - Segmentos: {len(result.get('segments', []))}")
    except Exception as e:
        print(f"   ❌ Error al transcribir: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Limpiar
        if os.path.exists(test_audio):
            os.remove(test_audio)
    
    print("\n" + "=" * 70)
    print("✅ VERIFICACIÓN EXITOSA - El modelo está funcionando correctamente")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
