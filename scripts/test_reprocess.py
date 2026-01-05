#!/usr/bin/env python3
"""
Script de prueba para reprocess_metadata.py

Genera metadata de prueba y verifica el re-procesamiento.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Test data
TEST_METADATA = [
    {
        "text": "Hola mundo esto es una prueva",
        "path": "/dummy/path/seg_0000_SPK_00.wav",
        "speaker": 0,
        "speaker_label": "SPK_00",
        "start": 0.0,
        "end": 3.5,
        "duration": 3.5,
        "language": "es",
        "podcast_id": "test_podcast",
        "segment_id": "seg_0000"
    },
    {
        "text": "Este es un texto con hérores de transcripción",
        "path": "/dummy/path/seg_0001_SPK_01.wav",
        "speaker": 1,
        "speaker_label": "SPK_01",
        "start": 3.5,
        "end": 7.2,
        "duration": 3.7,
        "language": "es",
        "podcast_id": "test_podcast",
        "segment_id": "seg_0001"
    },
    {
        "text": "La inteligencia artificial esta cambiando el mundo",
        "path": "/dummy/path/seg_0002_SPK_00.wav",
        "speaker": 0,
        "speaker_label": "SPK_00",
        "start": 7.2,
        "end": 11.0,
        "duration": 3.8,
        "language": "es",
        "podcast_id": "test_podcast",
        "segment_id": "seg_0002"
    }
]

def create_test_metadata(output_path: str):
    """Crea archivo de metadata de prueba."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(TEST_METADATA, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Metadata de prueba creado: {output_path}")
    return output_path

def verify_reprocessing(metadata_path: str):
    """Verifica que el re-procesamiento funcionó."""
    if not os.path.exists(metadata_path):
        print(f"✗ Error: No se encontró {metadata_path}")
        return False
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n📊 Verificando resultados...")
    
    # Verificar estructura
    if not isinstance(data, list):
        print("✗ La estructura no es una lista")
        return False
    
    print(f"✓ Estructura correcta (lista con {len(data)} entradas)")
    
    # Verificar correcciones
    corrected_count = 0
    for entry in data:
        if 'llm_correction' in entry:
            corrected_count += 1
            print(f"\n  Entrada {entry.get('segment_id', '?')}:")
            print(f"    Original: {entry['llm_correction'].get('original', '?')[:50]}...")
            print(f"    Corregido: {entry.get('text', '?')[:50]}...")
            if 'mcp_verified' in entry['llm_correction']:
                print(f"    ✓ Verificado por MCP")
            if 'mcp_reverted' in entry['llm_correction']:
                print(f"    ⚠️  Revertido por MCP")
    
    print(f"\n✓ {corrected_count}/{len(data)} entradas procesadas con LLM")
    
    return True

def main():
    print("🧪 Test de reprocess_metadata.py\n")
    
    # Crear metadata de prueba
    test_file = "/tmp/test_metadata_reprocess.json"
    create_test_metadata(test_file)
    
    # Mostrar contenido original
    print("\n📄 Metadata original:")
    with open(test_file, 'r', encoding='utf-8') as f:
        original = json.load(f)
    for entry in original:
        print(f"  - {entry['text'][:60]}...")
    
    # Ejecutar re-procesamiento
    print(f"\n🚀 Ejecutando re-procesamiento...")
    print(f"   (Esto puede tardar unos segundos...)\n")
    
    cmd = f"python scripts/reprocess_metadata.py {test_file} --config config/config_example_mcp.json"
    print(f"Comando: {cmd}\n")
    
    result = os.system(cmd)
    
    if result != 0:
        print(f"\n✗ El script retornó código de error: {result}")
        return False
    
    # Verificar resultados
    if verify_reprocessing(test_file):
        print("\n✅ Test completado exitosamente!")
        
        # Mostrar resultado final
        print("\n📄 Metadata final:")
        with open(test_file, 'r', encoding='utf-8') as f:
            final = json.load(f)
        for entry in final:
            print(f"  - {entry['text'][:60]}...")
        
        # Mostrar backup
        import glob
        backups = glob.glob(f"{test_file}.backup_*")
        if backups:
            print(f"\n📦 Backups creados:")
            for backup in backups:
                print(f"  - {backup}")
        
        return True
    else:
        print("\n❌ Test fallido!")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
