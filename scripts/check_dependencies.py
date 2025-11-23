#!/usr/bin/env python3
"""
Script para verificar que todas las dependencias estén instaladas correctamente.
"""
import sys
from importlib import import_module

def check_dependency(module_name, package_name=None):
    """Verifica si un módulo está instalado."""
    if package_name is None:
        package_name = module_name
    
    try:
        import_module(module_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} - NO INSTALADO")
        return False

def main():
    print("Verificando dependencias de fromPodcast...\n")
    
    dependencies = [
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("pydub", "pydub"),
        ("whisper", "openai-whisper"),
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
        ("tqdm", "tqdm"),
        ("numpy", "numpy"),
    ]
    
    optional_dependencies = [
        ("pyannote", "pyannote.audio (opcional)"),
    ]
    
    all_ok = True
    
    print("Dependencias requeridas:")
    for module, name in dependencies:
        if not check_dependency(module, name):
            all_ok = False
    
    print("\nDependencias opcionales:")
    for module, name in optional_dependencies:
        check_dependency(module, name)
    
    print("\n" + "="*50)
    if all_ok:
        print("✓ Todas las dependencias requeridas están instaladas")
        return 0
    else:
        print("✗ Faltan algunas dependencias requeridas")
        print("\nInstala las dependencias faltantes con:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == '__main__':
    sys.exit(main())

