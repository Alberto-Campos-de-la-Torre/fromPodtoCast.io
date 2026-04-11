#!/usr/bin/env python3
"""
Diagnóstico de estado de reprocesamiento de metadata.

Escanea todos los archivos de metadata y clasifica su estado:
- completed: Todas las entries tienen reprocessed_at
- partial: Algunas entries tienen reprocessed_at, otras no
- pending: Ninguna entry tiene reprocessed_at
- empty: Sin entries con texto

Actualiza reprocess_progress.json con el estado detectado.

Uso:
    python check_reprocess_status.py --data-dir /path/to/data [--update-progress] [--verbose]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def analyze_metadata_file(filepath: str) -> Dict:
    """
    Analiza un archivo de metadata y determina su estado de corrección.
    
    Returns:
        Dict con estadísticas del archivo
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        return {
            'file': os.path.basename(filepath),
            'path': filepath,
            'status': 'error',
            'error': str(e)
        }
    
    # Extraer entries
    if isinstance(metadata, dict):
        if 'entries' in metadata:
            entries = metadata['entries']
        elif 'segments' in metadata:
            entries = metadata['segments']
        else:
            entries = [metadata]
    else:
        entries = metadata
    
    total = len(entries)
    with_text = 0
    with_llm_correction = 0
    with_reprocessed_at = 0
    with_original = 0
    confidences = []
    
    for entry in entries:
        text = entry.get('text', '').strip()
        if not text:
            continue
        with_text += 1
        
        llm = entry.get('llm_correction', {})
        if llm:
            with_llm_correction += 1
            if llm.get('reprocessed_at'):
                with_reprocessed_at += 1
            conf = llm.get('confianza', 0)
            if conf:
                confidences.append(conf)
        
        if entry.get('text_original') or llm.get('original'):
            with_original += 1
    
    # Determinar estado
    if with_text == 0:
        status = 'empty'
    elif with_reprocessed_at == with_text:
        status = 'completed'
    elif with_reprocessed_at > 0:
        status = 'partial'
    elif with_llm_correction == with_text and with_llm_correction > 0:
        status = 'corrected_old'  # Tiene correcciones pero de un run anterior
    elif with_llm_correction > 0:
        status = 'partial_old'  # Correcciones parciales de un run anterior
    else:
        status = 'pending'
    
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    return {
        'file': os.path.basename(filepath),
        'path': filepath,
        'status': status,
        'total_entries': total,
        'with_text': with_text,
        'with_llm_correction': with_llm_correction,
        'with_reprocessed_at': with_reprocessed_at,
        'with_original': with_original,
        'avg_confidence': round(avg_conf, 3),
        'pct_corrected': round(with_reprocessed_at / with_text * 100, 1) if with_text > 0 else 0
    }


def scan_metadata_directory(data_dir: str, verbose: bool = False) -> List[Dict]:
    """Escanea todos los archivos de metadata en un directorio."""
    results = []
    data_path = Path(data_dir)
    
    # Buscar en subdirectorio metadata/
    metadata_files = []
    metadata_subdir = data_path / 'metadata'
    if metadata_subdir.exists():
        metadata_files.extend(list(metadata_subdir.glob('*.json')))
    
    # Buscar en raíz
    metadata_files.extend([f for f in data_path.glob('*.json') if f.is_file()])
    
    # Eliminar duplicados y backups
    metadata_files = list(set(metadata_files))
    metadata_files = [f for f in metadata_files if '.backup_' not in str(f)]
    metadata_files.sort(key=lambda f: f.name)
    
    print(f"\n📁 Escaneando {len(metadata_files)} archivos de metadata...")
    print(f"   Directorio: {data_dir}\n")
    
    for i, filepath in enumerate(metadata_files):
        if verbose and (i + 1) % 10 == 0:
            print(f"   Analizando {i+1}/{len(metadata_files)}...", flush=True)
        
        result = analyze_metadata_file(str(filepath))
        results.append(result)
    
    return results


def print_report(results: List[Dict], verbose: bool = False):
    """Imprime reporte de estado."""
    # Contar por estado
    status_counts = {}
    for r in results:
        s = r['status']
        status_counts[s] = status_counts.get(s, 0) + 1
    
    status_icons = {
        'completed': '✅',
        'partial': '⚠️ ',
        'partial_old': '⚠️ ',
        'corrected_old': '🔄',
        'pending': '⏳',
        'empty': '⊘',
        'error': '❌'
    }
    
    print(f"{'='*60}")
    print(f"  REPORTE DE ESTADO DE REPROCESAMIENTO")
    print(f"{'='*60}\n")
    
    for status, count in sorted(status_counts.items()):
        icon = status_icons.get(status, '?')
        print(f"  {icon} {status:20s}: {count:4d} archivos")
    
    print(f"\n  Total: {len(results)} archivos\n")
    
    # Mostrar archivos parciales
    partials = [r for r in results if r['status'] in ('partial', 'partial_old')]
    if partials:
        print(f"{'─'*60}")
        print(f"  ARCHIVOS PARCIALMENTE CORREGIDOS ({len(partials)}):")
        print(f"{'─'*60}")
        for r in partials:
            pct = r['pct_corrected']
            bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            print(f"  {bar} {pct:5.1f}%  {r['with_reprocessed_at']:3d}/{r['with_text']:3d}  {r['file'][:55]}")
        print()
    
    if verbose:
        # Mostrar todos los archivos
        print(f"{'─'*60}")
        print(f"  DETALLE POR ARCHIVO:")
        print(f"{'─'*60}")
        for r in results:
            icon = status_icons.get(r['status'], '?')
            print(f"  {icon} [{r['status']:13s}] {r.get('with_reprocessed_at', 0):3d}/{r.get('with_text', 0):3d} entries  conf:{r.get('avg_confidence', 0):.2f}  {r['file'][:40]}")


def update_progress_file(results: List[Dict], progress_path: str):
    """Actualiza el archivo de progreso con el estado detectado."""
    # Cargar progreso existente
    progress = {'completed': [], 'failed': [], 'partial': []}
    if os.path.exists(progress_path):
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                progress = json.load(f)
        except:
            pass
    
    # Preservar completed existentes (por basename)
    existing_completed = {c['file'] for c in progress.get('completed', [])}
    
    # Actualizar con resultados del scan
    new_completed = []
    new_partial = []
    
    for r in results:
        if r['status'] == 'error':
            continue
        
        if r['status'] == 'completed':
            if r['file'] not in existing_completed:
                new_completed.append({
                    'file': r['file'],
                    'path': r['path'],
                    'timestamp': datetime.now().isoformat(),
                    'entries': r['total_entries'],
                    'corrected': r['with_reprocessed_at'],
                    'detected_by': 'scan'
                })
        elif r['status'] in ('partial', 'partial_old'):
            new_partial.append({
                'file': r['file'],
                'path': r['path'],
                'timestamp': datetime.now().isoformat(),
                'total_entries': r['with_text'],
                'corrected_entries': r['with_reprocessed_at'],
                'pct_corrected': r['pct_corrected'],
                'status': r['status']
            })
    
    # Merge
    progress.setdefault('completed', []).extend(new_completed)
    progress['partial'] = new_partial  # Replace, not append
    progress['scan_timestamp'] = datetime.now().isoformat()
    progress['scan_summary'] = {
        s: sum(1 for r in results if r['status'] == s)
        for s in set(r['status'] for r in results)
    }
    
    # Guardar
    Path(progress_path).parent.mkdir(parents=True, exist_ok=True)
    with open(progress_path, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Progreso actualizado: {progress_path}")
    print(f"  Nuevos completed: {len(new_completed)}")
    print(f"  Parciales: {len(new_partial)}")


def main():
    parser = argparse.ArgumentParser(
        description='Diagnóstico de estado de reprocesamiento de metadata'
    )
    parser.add_argument(
        '--data-dir',
        default='/media/ttech-main/8464b39b-ba5b-49c5-9e05-010122b9874d',
        help='Directorio donde buscar archivos de metadata'
    )
    parser.add_argument(
        '--update-progress',
        action='store_true',
        help='Actualizar reprocess_progress.json con el estado detectado'
    )
    parser.add_argument(
        '--progress-file',
        default='./data/reprocess_progress.json',
        help='Archivo de progreso (default: ./data/reprocess_progress.json)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar detalle por archivo'
    )
    
    args = parser.parse_args()
    
    results = scan_metadata_directory(args.data_dir, verbose=args.verbose)
    print_report(results, verbose=args.verbose)
    
    if args.update_progress:
        update_progress_file(results, args.progress_file)


if __name__ == '__main__':
    main()
