#!/usr/bin/env python3
"""
Script de prueba rápida para fromPodtoCast.
Verifica que todo funcione correctamente con un ejemplo mínimo.
"""
import sys
import os
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_imports():
    """Verifica que todos los módulos se puedan importar."""
    print("🔍 Verificando imports...")
    try:
        from audio_segmenter import AudioSegmenter
        from audio_normalizer import AudioNormalizer
        from transcriber import AudioTranscriber
        from speaker_diarizer import SpeakerDiarizer
        from processor import PodcastProcessor
        print("✅ Todos los módulos se importaron correctamente")
        return True
    except ImportError as e:
        print(f"❌ Error importando módulos: {e}")
        return False

def test_audio_segmenter():
    """Prueba básica del segmentador."""
    print("\n🔍 Probando AudioSegmenter...")
    try:
        from audio_segmenter import AudioSegmenter
        segmenter = AudioSegmenter(min_duration=10.0, max_duration=15.0)
        print("✅ AudioSegmenter inicializado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en AudioSegmenter: {e}")
        return False

def test_audio_normalizer():
    """Prueba básica del normalizador."""
    print("\n🔍 Probando AudioNormalizer...")
    try:
        from audio_normalizer import AudioNormalizer
        normalizer = AudioNormalizer(target_sr=22050)
        print("✅ AudioNormalizer inicializado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en AudioNormalizer: {e}")
        return False

def test_transcriber():
    """Prueba básica del transcriptor (sin cargar modelo completo)."""
    print("\n🔍 Probando AudioTranscriber...")
    try:
        from transcriber import AudioTranscriber
        # No inicializamos el modelo completo para ahorrar tiempo
        print("✅ AudioTranscriber se puede importar (modelo se carga al usarlo)")
        return True
    except Exception as e:
        print(f"❌ Error en AudioTranscriber: {e}")
        return False

def check_data_directories():
    """Verifica que los directorios necesarios existan."""
    print("\n🔍 Verificando directorios de datos...")
    base_dir = Path(__file__).parent.parent
    required_dirs = [
        base_dir / 'data' / 'input',
        base_dir / 'data' / 'output',
        base_dir / 'config'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"✅ {dir_path} existe")
        else:
            print(f"❌ {dir_path} no existe")
            all_exist = False
    
    return all_exist

def check_config_file():
    """Verifica que el archivo de configuración exista."""
    print("\n🔍 Verificando archivo de configuración...")
    config_path = Path(__file__).parent.parent / 'config' / 'config.json'
    if config_path.exists():
        print(f"✅ {config_path} existe")
        return True
    else:
        print(f"❌ {config_path} no existe")
        return False

def main():
    print("="*60)
    print("🧪 Prueba Rápida de fromPodtoCast")
    print("="*60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("AudioSegmenter", test_audio_segmenter()))
    results.append(("AudioNormalizer", test_audio_normalizer()))
    results.append(("AudioTranscriber", test_transcriber()))
    results.append(("Directorios", check_data_directories()))
    results.append(("Configuración", check_config_file()))
    
    print("\n" + "="*60)
    print("📊 Resumen de Pruebas")
    print("="*60)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20s} {status}")
        if not result:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 ¡Todas las pruebas pasaron! El proyecto está listo para usar.")
        print("\n📝 Próximos pasos:")
        print("   1. Coloca un archivo de audio en data/input/")
        print("   2. Ejecuta: python3 main.py data/input/tu_archivo.mp3 -o data/output")
        return 0
    else:
        print("\n⚠️  Algunas pruebas fallaron. Revisa los errores arriba.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

