#!/usr/bin/env python3
"""
fromPodtoCast - Script principal para procesar podcasts y generar datos de entrenamiento para TTS.
"""
import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from processor import PodcastProcessor


def load_config(config_path: str) -> dict:
    """Carga configuración desde un archivo JSON."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_metadata(metadata_path: str) -> list:
    """Carga metadata existente desde un archivo JSON."""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_metadata(metadata: list, output_path: str):
    """Guarda metadata en un archivo JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def resume_llm_correction(metadata_path: str, config: dict, output_path: str = None):
    """
    Retoma solo la fase de corrección con LLM sobre metadata existente.
    
    Args:
        metadata_path: Ruta al archivo de metadata existente
        config: Configuración del procesador
        output_path: Ruta de salida (por defecto sobrescribe el original)
    """
    from text_corrector_llm import TextCorrectorLLM
    from text_preprocessor import TextPreprocessor
    
    print(f"\n{'='*60}")
    print("  RETOMANDO FASE 4.6: Corrección con LLM")
    print(f"{'='*60}\n")
    
    # Cargar metadata existente
    print(f"📂 Cargando metadata desde: {metadata_path}")
    metadata = load_metadata(metadata_path)
    print(f"   ✓ Cargados {len(metadata)} segmentos\n")
    
    # Inicializar componentes
    llm_config = config.get('llm_correction', {})
    text_config = config.get('text_preprocessing', {})
    
    if not llm_config.get('enabled', False):
        print("⚠️  LLM correction está deshabilitado en config.json")
        print("   Habilítalo con: llm_correction.enabled = true")
        sys.exit(1)
    
    # Inicializar pre-procesador (opcional, para aplicar antes del LLM)
    text_preprocessor = None
    if text_config.get('enabled', True):
        text_preprocessor = TextPreprocessor(
            glosario_path=text_config.get('glosario_path'),
            fix_punctuation=text_config.get('fix_punctuation', True),
            normalize_numbers=text_config.get('normalize_numbers', True),
            fix_spacing=text_config.get('fix_spacing', True),
            fix_capitalization=text_config.get('fix_capitalization', True)
        )
    
    # Inicializar corrector LLM con optimizaciones
    try:
        llm_corrector = TextCorrectorLLM(
            ollama_host=llm_config.get('ollama_host', 'http://192.168.1.81:11434'),
            model=llm_config.get('model', 'qwen3:8b'),
            glosario_path=text_config.get('glosario_path'),
            timeout=llm_config.get('timeout', 90),
            max_retries=llm_config.get('max_retries', 2),
            batch_size=llm_config.get('batch_size', 3),
            parallel_workers=llm_config.get('parallel_workers', 2),
            smart_filter=llm_config.get('smart_filter', True)
        )
        min_confidence = llm_config.get('min_confidence', 0.7)
        batch_size = llm_config.get('batch_size', 3)
        workers = llm_config.get('parallel_workers', 2)
        print(f"✓ Corrector LLM inicializado ({llm_config.get('model', 'qwen3:8b')})")
        print(f"  Optimizaciones: batch={batch_size}, workers={workers}, smart_filter=ON")
    except Exception as e:
        print(f"❌ Error inicializando corrector LLM: {e}")
        sys.exit(1)
    
    # Filtrar segmentos que ya tienen corrección LLM (opcional)
    skip_already_corrected = True
    segments_to_process = []
    already_corrected = 0
    indices_to_process = []
    
    for i, entry in enumerate(metadata):
        if skip_already_corrected and 'llm_correction' in entry:
            already_corrected += 1
        else:
            segments_to_process.append(entry)
            indices_to_process.append(i)
    
    if already_corrected > 0:
        print(f"   ⏭️  Saltando {already_corrected} segmentos ya corregidos")
    
    print(f"   📝 Procesando {len(segments_to_process)} segmentos\n")
    
    # Pre-procesamiento con reglas (si no se hizo antes)
    preprocess_count = 0
    if text_preprocessor:
        for entry in segments_to_process:
            text = entry.get('text', '')
            if text and 'text_changes' not in entry:
                corrected, changes = text_preprocessor.preprocess(text)
                if changes:
                    entry['text_original'] = text
                    entry['text_changes'] = changes
                    entry['text'] = corrected
                    preprocess_count += 1
    
    # Procesar con LLM en paralelo y batches
    import time
    start_time = time.time()
    
    total_batches = (len(segments_to_process) + batch_size - 1) // batch_size
    
    print(f"   🚀 Iniciando corrección ({total_batches} batches)...\n")
    
    with tqdm(total=len(segments_to_process), desc="   Corrigiendo con LLM", unit="seg") as pbar:
        processed_count = [0]  # Usar lista para mutabilidad en closure
        
        def update_progress(batch_done, total):
            increment = min(batch_size, len(segments_to_process) - processed_count[0])
            pbar.update(increment)
            processed_count[0] += increment
        
        corrected_entries = llm_corrector.correct_parallel(
            segments_to_process,
            text_field='text',
            min_confidence=min_confidence,
            progress_callback=update_progress
        )
        
        # Completar barra si quedó incompleta
        remaining = len(segments_to_process) - pbar.n
        if remaining > 0:
            pbar.update(remaining)
    
    elapsed = time.time() - start_time
    
    # Aplicar correcciones al metadata original
    llm_corrected_count = 0
    llm_skipped_count = 0
    
    for j, corrected_entry in enumerate(corrected_entries):
        orig_idx = indices_to_process[j]
        metadata[orig_idx] = corrected_entry
        
        if 'llm_correction' in corrected_entry:
            llm_corrected_count += 1
        elif corrected_entry.get('llm_skipped'):
            llm_skipped_count += 1
    
    # Estadísticas
    llm_stats = llm_corrector.get_stats()
    
    print(f"\n{'='*60}")
    print("  RESULTADOS")
    print(f"{'='*60}")
    print(f"  📊 Total segmentos:     {len(metadata)}")
    print(f"  ✅ Corregidos con LLM:  {llm_corrected_count}")
    print(f"  ⏭️  Saltados (sin errores): {llm_skipped_count}")
    print(f"  📝 Pre-procesados:      {preprocess_count}")
    print(f"  ⏭️  Ya corregidos:       {already_corrected}")
    print(f"  ❌ Fallidos:            {llm_stats.get('failed', 0)}")
    print(f"  📈 Confianza promedio:  {llm_stats.get('avg_confidence', 0):.2f}")
    print(f"  ⏱️  Tiempo total:        {elapsed:.1f}s")
    print(f"  📦 Llamadas batch:      {llm_stats.get('batch_calls', 0)}")
    print(f"{'='*60}\n")
    
    # Guardar metadata actualizada
    output = output_path or metadata_path
    save_metadata(metadata, output)
    print(f"✓ Metadata guardada en: {output}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Procesa podcasts para generar datos de entrenamiento TTS'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Ruta al archivo de audio, directorio, o archivo de metadata (con --resume-llm)'
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
    parser.add_argument(
        '--resume-llm',
        action='store_true',
        help='Retomar solo la fase 4.6 (corrección LLM) sobre metadata existente'
    )
    parser.add_argument(
        '--reprocess-all',
        action='store_true',
        help='Con --resume-llm: reprocesar todos los segmentos, incluso los ya corregidos'
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
    
    # Modo: Retomar solo corrección LLM
    if args.resume_llm:
        input_path = Path(args.input)
        
        # Determinar archivo de metadata
        if input_path.suffix == '.json':
            metadata_path = str(input_path)
        else:
            # Buscar metadata del podcast
            import re
            podcast_name = input_path.stem if input_path.is_file() else input_path.name
            podcast_id_clean = re.sub(r'[^a-zA-Z0-9_-]', '_', podcast_name)[:50]
            metadata_path = f"{args.output}/metadata/{podcast_id_clean}.json"
        
        if not Path(metadata_path).exists():
            print(f"❌ Error: No se encontró metadata en: {metadata_path}")
            print("   Primero procesa el podcast completo o especifica la ruta al JSON.")
            sys.exit(1)
        
        # Determinar salida
        output_path = args.metadata if args.metadata != './data/output/metadata.json' else None
        
        metadata = resume_llm_correction(metadata_path, config, output_path)
        
        print(f"\n✓ Corrección LLM completada!")
        print(f"  Segmentos procesados: {len(metadata)}")
        return
    
    # Modo normal: Procesar podcast completo
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
