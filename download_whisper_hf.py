#!/usr/bin/env python3
"""
Script para descargar Whisper_Megi_IA directamente desde Hugging Face Hub
usando huggingface_hub en lugar de git clone
"""
import os
from pathlib import Path

# Instalar huggingface_hub si no está disponible
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("📦 Instalando huggingface_hub...")
    os.system("pip install -q huggingface_hub")
    from huggingface_hub import snapshot_download

# Configuración
repo_id = "MrZeggers/Whisper_Megi_IA"
local_dir = "/media/ttech-main/8464b39b-ba5b-49c5-9e05-010122b9874d/Whisper_Megi_IA_HF"

print("="*70)
print(f"Descargando modelo: {repo_id}")
print(f"Destino: {local_dir}")
print("="*70)
print()

# Crear directorio si no existe
Path(local_dir).mkdir(parents=True, exist_ok=True)

print("🔄 Iniciando descarga con huggingface_hub...")
print("   (Esto puede tardar varios minutos...)")
print()

try:
    # Descargar todo el repositorio
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    
    print()
    print("="*70)
    print("✅ Descarga completada exitosamente!")
    print(f"📂 Modelo guardado en: {local_dir}")
    print("="*70)
    
except Exception as e:
    print()
    print("="*70)
    print(f"❌ Error durante la descarga: {e}")
    print("="*70)
    raise
