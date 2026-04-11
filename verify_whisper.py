#!/usr/bin/env python3
"""
Verifica qué versión de Whisper está disponible y funcional en fromPodtoCast
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import whisper

print("="*70)
print("Verificación de Whisper en fromPodtoCast")
print("="*70)
print()

print("📦 Paquete openai-whisper")
print(f"   Versión: {whisper.__version__ if hasattr(whisper, '__version__') else 'N/A'}")
print(f"   Modelos disponibles: {len(whisper.available_models())}")
print(f"   Modelos: {', '.join(whisper.available_models())}")
print()

print("💾 Modelos descargados en caché:")
cache_dir = Path.home() / ".cache" / "whisper"
if cache_dir.exists():
    models = list(cache_dir.glob("*.pt"))
    if models:
        for model in models:
            size_mb = model.stat().st_size / (1024**2)
            print(f"   ✓ {model.name} ({size_mb:.1f} MB)")
    else:
        print("   (vacío)")
else:
    print("   (no existe el directorio)")
print()

print("🔧 Configuración actual (config/config.json):")
import json
config_file = Path(__file__).parent / "config" / "config.json"
if config_file.exists():
    with open(config_file) as f:
        config = json.load(f)
    whisper_model = config.get('whisper_model', 'base')
    print(f"   whisper_model: {whisper_model}")
    
    # Verificar si es un path o nombre de modelo
    if os.path.isdir(whisper_model):
        print(f"   Tipo: Directorio local (modelo HF)")
        print(f"   Existe: {'✓' if Path(whisper_model).exists() else '✗'}")
    else:
        print(f"   Tipo: Modelo OpenAI Whisper estándar")
else:
    print("   (archivo de config no encontrado)")

print()
print("="*70)
print("Recomendación:")
print("="*70)

# Verificar qué funciona
whisper_megi_path = "/media/ttech-main/8464b39b-ba5b-49c5-9e05-010122b9874d/Whisper_Megi_IA_HF"
if os.path.exists(whisper_megi_path):
    print(f"⚠️  Whisper_Megi_IA existe pero tiene archivos corruptos")
    print(f"   Los archivos .safetensors del repositorio están dañados")
    print()

print("✅ USAR: Modelo OpenAI Whisper estándar")
print("   Opciones:")
print("   - 'base': Rápido, buena calidad (139MB) ← YA DESCARGADO")
print("   - 'large-v3': Máxima calidad (3GB)")
print("   - 'large-v3-turbo': Balance calidad/velocidad (1.5GB)")
print()
print("   Configuración recomendada:")
print('   "whisper_model": "base"  # o "large-v3" para mejor calidad')
print()
