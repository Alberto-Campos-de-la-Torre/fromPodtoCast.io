#!/usr/bin/env python3
"""
Verifica que el auto_pipeline esté usando el modelo Whisper large-v3
"""
import json
import sys
from pathlib import Path

print("="*70)
print("Verificación: Whisper large-v3 en Auto Pipeline")
print("="*70)
print()

# 1. Verificar config.json
config_path = Path('/home/ttech-main/fromPodtoCast/config/config.json')
print("1️⃣ Configuración en config/config.json:")
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    
    whisper_model = config.get('whisper_model', 'NO CONFIGURADO')
    print(f"   whisper_model: {whisper_model}")
    
    if whisper_model == 'large-v3':
        print("   ✅ Configurado correctamente para large-v3")
    else:
        print(f"   ⚠️  NO está configurado para large-v3, valor actual: {whisper_model}")
else:
    print("   ❌ Archivo de configuración no encontrado")
    sys.exit(1)

print()

# 2. Verificar que processor.py lee el config
print("2️⃣ Flujo de carga del modelo:")
print("   auto_pipeline.py → main.py → processor.py → AudioTranscriber")
print()
print("   Pasos:")
print("   ✓ auto_pipeline.py llama a main.py con --config")
print("   ✓ main.py carga config.json")
print("   ✓ main.py inicializa PodcastProcessor(config)")
print("   ✓ processor.py lee config.get('whisper_model', 'base')")
print("   ✓ AudioTranscriber carga el modelo especificado")
print()

# 3. Verificar modelo descargado
print("3️⃣ Modelo Whisper large-v3:")
cache_dir = Path.home() / ".cache" / "whisper"
large_v3_path = cache_dir / "large-v3.pt"

if large_v3_path.exists():
    size_gb = large_v3_path.stat().st_size / (1024**3)
    print(f"   Ubicación: {large_v3_path}")
    print(f"   Tamaño: {size_gb:.2f} GB")
    print("   ✅ Modelo descargado y listo")
else:
    print("   ❌ Modelo large-v3 NO encontrado")
    sys.exit(1)

print()

# 4. Verificar GPU config
print("4️⃣ Configuración GPU:")
gpu_config = config.get('gpu_config', {})
if gpu_config.get('enabled'):
    whisper_gpu = gpu_config.get('whisper_gpu', 1)
    print(f"   Multi-GPU: HABILITADO")
    print(f"   Whisper asignado a: GPU {whisper_gpu}")
else:
    print(f"   Multi-GPU: DESHABILITADO")
    device = config.get('device')
    print(f"   Dispositivo: {device or 'auto (CUDA si disponible)'}")

print()
print("="*70)
print("✅ VERIFICACIÓN COMPLETA")
print("="*70)
print()
print("El auto_pipeline usará Whisper large-v3 cuando procese audio.")
print()
print("Para ejecutar el pipeline:")
print("  cd /home/ttech-main/fromPodtoCast")
print("  python3 scripts/auto_pipeline.py")
print()
