#!/usr/bin/env python3
"""
Descarga el modelo Whisper large-v3
"""
import whisper

print("="*70)
print("Descargando Whisper large-v3")
print("="*70)
print()
print("📥 Iniciando descarga del modelo large-v3...")
print("   Tamaño: ~3GB")
print("   Esto puede tardar varios minutos...")
print()

try:
    model = whisper.load_model("large-v3")
    print()
    print("="*70)
    print("✅ Modelo large-v3 descargado exitosamente!")
    print("="*70)
    print()
    print("Información del modelo:")
    print(f"   Dispositivo: {model.device}")
    print(f"   Tipo: {type(model).__name__}")
    print()
    
except Exception as e:
    print()
    print("="*70)
    print(f"❌ Error al descargar: {e}")
    print("="*70)
    raise
