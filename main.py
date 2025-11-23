#!/usr/bin/env python3
"""
fromPodtoCast - Script principal para procesar podcasts y generar datos de entrenamiento para TTS.
"""
import argparse
import json
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from processor import PodcastProcessor


def load_config(config_path: str) -> dict:
    """Carga configuración desde un archivo JSON."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Procesa podcasts para generar datos de entrenamiento TTS'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Ruta al archivo de audio o directorio con archivos de podcast'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='./data/output',
        help='Directorio de salida (default: ./data/output)'
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='./config/config.json',
        help='Ruta al archivo de configuración (default: ./config/config.json)'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default='./data/output/metadata.json',
        help='Ruta donde guardar el archivo JSON de metadata (default: ./data/output/metadata.json)'
    )
    
    args = parser.parse_args()
    
    # Cargar configuración
    config_path = Path(args.config)
    if config_path.exists():
        print(f"Cargando configuración desde: {config_path}")
        config = load_config(str(config_path))
    else:
        print(f"Advertencia: Archivo de configuración no encontrado: {config_path}")
        print("Usando configuración por defecto...")
        config = {}
    
    # Inicializar procesador
    processor = PodcastProcessor(config)
    
    # Determinar archivos a procesar
    input_path = Path(args.input)
    if input_path.is_file():
        audio_files = [str(input_path)]
    elif input_path.is_dir():
        # Buscar archivos de audio comunes
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(f'*{ext}'))
            audio_files.extend(input_path.glob(f'**/*{ext}'))
        audio_files = [str(f) for f in audio_files]
        
        if not audio_files:
            print(f"Error: No se encontraron archivos de audio en {input_path}")
            sys.exit(1)
    else:
        print(f"Error: {input_path} no es un archivo ni un directorio válido")
        sys.exit(1)
    
    print(f"\nArchivos a procesar: {len(audio_files)}")
    for f in audio_files:
        print(f"  - {f}")
    
    # Procesar archivos
    metadata = processor.process_batch(audio_files, args.output)
    
    # Guardar metadata
    if metadata:
        processor.save_metadata(metadata, args.metadata)
        print(f"\n✓ Proceso completado exitosamente!")
        print(f"  Archivos procesados: {len(audio_files)}")
        print(f"  Segmentos generados: {len(metadata)}")
    else:
        print("\n✗ No se generaron metadatos. Revisa los errores anteriores.")
        sys.exit(1)


if __name__ == '__main__':
    main()

