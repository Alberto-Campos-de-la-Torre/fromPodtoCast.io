#!/usr/bin/env python3
"""
Detecta y re-transcribe segmentos con texto corrupto (repeticiones).

Flujo:
1. Escanea metadata buscando entries con texto repetido (corrupción del LLM)
2. Re-transcribe esos segmentos desde el audio WAV con Whisper
3. Limpia el llm_correction corrupto y guarda el texto fresco de Whisper
4. Genera un reporte de lo que se encontró y corrigió

Uso:
    python retranscribe_corrupted.py --data-dir /path/to/data [--dry-run] [--min-repetitions 3]
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter

# Agregar src al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def detect_repetition(text: str, min_phrase_len: int = 4, min_repetitions: int = 3) -> Optional[Dict]:
    """
    Detecta si un texto tiene frases repetidas excesivamente (corrupción).
    
    Returns:
        Dict con info de la repetición si se detecta, None si el texto está limpio
    """
    if not text or len(text) < 50:
        return None
    
    words = text.split()
    if len(words) < 15:
        return None
    
    # Buscar frases de diferentes longitudes repetidas
    for phrase_len in [8, 6, 5, 4]:
        for i in range(len(words) - phrase_len):
            phrase = ' '.join(words[i:i+phrase_len])
            if len(phrase) < 12:  # Ignorar frases muy cortas
                continue
            
            count = text.count(phrase)
            if count >= min_repetitions:
                # Calcular qué porcentaje del texto es la repetición
                repeated_chars = len(phrase) * count
                pct = repeated_chars / len(text) * 100
                
                return {
                    'phrase': phrase[:80],
                    'count': count,
                    'phrase_len_words': phrase_len,
                    'pct_of_text': round(pct, 1),
                    'text_len': len(text),
                    'severity': 'critical' if pct > 50 else ('high' if pct > 25 else 'medium')
                }
    
    return None


def scan_metadata_for_corruption(data_dir: str, min_repetitions: int = 3) -> List[Dict]:
    """
    Escanea todos los archivos de metadata buscando texto corrupto.
    
    Returns:
        Lista de entries corruptas con su ubicación
    """
    metadata_dir = Path(data_dir) / 'metadata'
    if not metadata_dir.exists():
        print(f"❌ No se encontró directorio metadata: {metadata_dir}")
        return []
    
    files = sorted(metadata_dir.glob('*.json'))
    files = [f for f in files if '.backup' not in str(f)]
    
    print(f"\n📁 Escaneando {len(files)} archivos de metadata...")
    
    corrupted = []
    files_with_corruption = set()
    
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            continue
        
        # Extraer entries
        if isinstance(data, dict):
            entries = data.get('entries', data.get('segments', []))
            if not entries and 'text' in data:
                entries = [data]
        elif isinstance(data, list):
            entries = data
        else:
            continue
        
        for i, entry in enumerate(entries):
            text = entry.get('text', '')
            
            repetition = detect_repetition(text, min_repetitions=min_repetitions)
            if repetition:
                audio_path = entry.get('path', '')
                audio_exists = os.path.exists(audio_path) if audio_path else False
                
                # Verificar si hay original preservado
                has_whisper_original = False
                original_text = None
                if 'text_original' in entry:
                    original_text = entry['text_original']
                    has_whisper_original = True
                elif 'llm_correction' in entry and 'original' in entry['llm_correction']:
                    original_text = entry['llm_correction']['original']
                    # Verificar si el original TAMBIÉN está corrupto
                    orig_rep = detect_repetition(original_text, min_repetitions=min_repetitions)
                    has_whisper_original = orig_rep is None  # Solo si el original NO está corrupto
                
                corrupted.append({
                    'file': str(filepath),
                    'filename': filepath.name,
                    'entry_index': i,
                    'audio_path': audio_path,
                    'audio_exists': audio_exists,
                    'has_whisper_original': has_whisper_original,
                    'original_also_corrupted': original_text is not None and detect_repetition(original_text, min_repetitions=min_repetitions) is not None,
                    'repetition': repetition,
                    'current_text_preview': text[:100] + '...' if len(text) > 100 else text
                })
                files_with_corruption.add(filepath.name)
    
    return corrupted


def retranscribe_segments(corrupted_entries: List[Dict], config_path: str = None, 
                          dry_run: bool = False) -> Dict:
    """
    Re-transcribe los segmentos corruptos usando Whisper.
    
    Returns:
        Estadísticas del proceso
    """
    stats = {
        'total_corrupted': len(corrupted_entries),
        'recoverable_from_original': 0,
        'retranscribed': 0,
        'audio_missing': 0,
        'failed': 0,
        'files_modified': set()
    }
    
    # Separar por estrategia de recuperación
    needs_retranscription = []
    recoverable_from_original = []
    
    for entry in corrupted_entries:
        if entry['has_whisper_original'] and not entry['original_also_corrupted']:
            recoverable_from_original.append(entry)
        elif entry['audio_exists']:
            needs_retranscription.append(entry)
        else:
            stats['audio_missing'] += 1
    
    print(f"\n{'='*60}")
    print(f"  PLAN DE RECUPERACIÓN")
    print(f"{'='*60}")
    print(f"  Total corruptos:          {len(corrupted_entries)}")
    print(f"  Recuperables de original: {len(recoverable_from_original)}")
    print(f"  Necesitan re-transcribir: {len(needs_retranscription)}")
    print(f"  Audio faltante:           {stats['audio_missing']}")
    print(f"{'='*60}\n")
    
    if dry_run:
        print("🔍 [DRY RUN] No se realizarán cambios\n")
        return stats
    
    # Paso 1: Recuperar desde original preservado
    if recoverable_from_original:
        print(f"📋 Recuperando {len(recoverable_from_original)} entries desde texto original...")
        _recover_from_original(recoverable_from_original, stats)
    
    # Paso 2: Re-transcribir con Whisper
    if needs_retranscription:
        print(f"\n🎤 Re-transcribiendo {len(needs_retranscription)} segmentos con Whisper...")
        
        # Cargar config para Whisper
        config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        _retranscribe_with_whisper(needs_retranscription, config, stats)
    
    # Guardar archivos modificados
    print(f"\n💾 Guardando {len(stats['files_modified'])} archivos modificados...")
    stats['files_modified'] = list(stats['files_modified'])
    
    return stats


def _recover_from_original(entries: List[Dict], stats: Dict):
    """Recupera texto desde el campo original preservado."""
    # Agrupar por archivo para hacer un solo read/write por archivo
    by_file = {}
    for entry in entries:
        f = entry['file']
        if f not in by_file:
            by_file[f] = []
        by_file[f].append(entry)
    
    for filepath, file_entries in by_file.items():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                entries_list = data.get('entries', data.get('segments', []))
            elif isinstance(data, list):
                entries_list = data
            else:
                continue
            
            modified = False
            for entry_info in file_entries:
                idx = entry_info['entry_index']
                if idx >= len(entries_list):
                    continue
                
                entry = entries_list[idx]
                
                # Recuperar original
                if 'text_original' in entry:
                    original = entry['text_original']
                elif 'llm_correction' in entry and 'original' in entry['llm_correction']:
                    original = entry['llm_correction']['original']
                else:
                    continue
                
                # Reemplazar texto corrupto con original
                entry['text'] = original
                entry['corruption_recovered'] = {
                    'method': 'original_preserved',
                    'timestamp': datetime.now().isoformat(),
                    'corrupted_preview': entry_info['current_text_preview'][:80]
                }
                # Limpiar la corrección LLM corrupta
                if 'llm_correction' in entry:
                    del entry['llm_correction']
                if 'text_original' in entry:
                    del entry['text_original']
                
                modified = True
                stats['recoverable_from_original'] += 1
            
            if modified:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                stats['files_modified'].add(filepath)
                print(f"   ✅ {Path(filepath).name}: {len(file_entries)} entries recuperadas")
        
        except Exception as e:
            print(f"   ❌ Error en {Path(filepath).name}: {e}")
            stats['failed'] += len(file_entries)


def _retranscribe_with_whisper(entries: List[Dict], config: Dict, stats: Dict):
    """Re-transcribe segmentos usando Whisper."""
    try:
        from transcriber import AudioTranscriber
    except ImportError:
        print("   ❌ No se pudo importar AudioTranscriber")
        stats['failed'] += len(entries)
        return
    
    # Inicializar Whisper con el modelo del config
    whisper_model = config.get('whisper_model', 'large-v3')
    language = config.get('language', 'es') or 'es'  # Nunca None
    
    print(f"   Modelo Whisper: {whisper_model}, idioma: {language}")
    transcriber = AudioTranscriber(
        model_name=whisper_model,
        language=language,
        force_language=True
    )
    
    # Agrupar por archivo
    by_file = {}
    for entry in entries:
        f = entry['file']
        if f not in by_file:
            by_file[f] = []
        by_file[f].append(entry)
    
    for filepath, file_entries in by_file.items():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                entries_list = data.get('entries', data.get('segments', []))
            elif isinstance(data, list):
                entries_list = data
            else:
                continue
            
            modified = False
            retranscribed_count = 0
            
            for entry_info in file_entries:
                idx = entry_info['entry_index']
                if idx >= len(entries_list):
                    continue
                
                entry = entries_list[idx]
                audio_path = entry.get('path', '')
                
                if not os.path.exists(audio_path):
                    stats['audio_missing'] += 1
                    continue
                
                try:
                    # Re-transcribir
                    result = transcriber.transcribe(audio_path)
                    new_text = result.get('text', '').strip()
                    
                    if new_text:
                        # Guardar texto re-transcripto
                        entry['text'] = new_text
                        entry['corruption_recovered'] = {
                            'method': 'retranscribed_whisper',
                            'whisper_model': whisper_model,
                            'timestamp': datetime.now().isoformat(),
                            'corrupted_preview': entry_info['current_text_preview'][:80]
                        }
                        # Limpiar correcciones LLM corruptas
                        if 'llm_correction' in entry:
                            del entry['llm_correction']
                        if 'text_original' in entry:
                            del entry['text_original']
                        # Quitar reprocessed_at para que el reprocess lo corrija de nuevo
                        
                        modified = True
                        retranscribed_count += 1
                        stats['retranscribed'] += 1
                    else:
                        stats['failed'] += 1
                
                except Exception as e:
                    print(f"      ❌ Error transcribiendo {Path(audio_path).name}: {e}")
                    stats['failed'] += 1
            
            if modified:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                stats['files_modified'].add(filepath)
                print(f"   🎤 {Path(filepath).name}: {retranscribed_count} entries re-transcritas")
        
        except Exception as e:
            print(f"   ❌ Error en {Path(filepath).name}: {e}")
            stats['failed'] += len(file_entries)


def print_corruption_report(corrupted: List[Dict]):
    """Imprime reporte detallado de corrupción."""
    if not corrupted:
        print("\n✅ No se encontró texto corrupto")
        return
    
    # Agrupar por archivo
    by_file = {}
    for c in corrupted:
        fn = c['filename']
        if fn not in by_file:
            by_file[fn] = []
        by_file[fn].append(c)
    
    # Contar por severidad
    severity_counts = Counter(c['repetition']['severity'] for c in corrupted)
    
    # Contar por estrategia
    with_original = sum(1 for c in corrupted if c['has_whisper_original'] and not c['original_also_corrupted'])
    needs_whisper = sum(1 for c in corrupted if not c['has_whisper_original'] and c['audio_exists'])
    unrecoverable = sum(1 for c in corrupted if not c['has_whisper_original'] and not c['audio_exists'])
    both_corrupted = sum(1 for c in corrupted if c['original_also_corrupted'])
    
    print(f"\n{'='*60}")
    print(f"  REPORTE DE CORRUPCIÓN DE TEXTO")
    print(f"{'='*60}")
    print(f"\n  Total entries corruptas:  {len(corrupted)}")
    print(f"  Archivos afectados:      {len(by_file)}")
    print(f"\n  Por severidad:")
    print(f"    ❌ Crítica (>50%):      {severity_counts.get('critical', 0)}")
    print(f"    ⚠️  Alta (25-50%):      {severity_counts.get('high', 0)}")
    print(f"    ⚡ Media (<25%):        {severity_counts.get('medium', 0)}")
    print(f"\n  Estrategia de recuperación:")
    print(f"    📋 Desde original:      {with_original}")
    print(f"    🎤 Re-transcribir:      {needs_whisper}")
    print(f"    ❌ Original tb corrupto: {both_corrupted}")
    print(f"    ⛔ Sin audio:           {unrecoverable}")
    
    print(f"\n{'─'*60}")
    print(f"  DETALLE POR ARCHIVO:")
    print(f"{'─'*60}")
    
    for filename, entries in sorted(by_file.items(), key=lambda x: -len(x[1])):
        print(f"\n  📄 {filename[:55]} ({len(entries)} corruptas)")
        for e in entries[:5]:
            rep = e['repetition']
            strategy = "📋original" if e['has_whisper_original'] and not e['original_also_corrupted'] else (
                "🎤whisper" if e['audio_exists'] else "⛔perdido"
            )
            print(f"     [{strategy}] entry#{e['entry_index']:3d}  {rep['severity']:8s}  {rep['count']}x \"{rep['phrase'][:40]}...\"")
        if len(entries) > 5:
            print(f"     ... y {len(entries)-5} más")


def main():
    parser = argparse.ArgumentParser(
        description='Detecta y re-transcribe segmentos con texto corrupto'
    )
    parser.add_argument(
        '--data-dir',
        default='/media/ttech-main/8464b39b-ba5b-49c5-9e05-010122b9874d',
        help='Directorio de datos'
    )
    parser.add_argument(
        '--config',
        default=str(PROJECT_ROOT / 'config' / 'config.json'),
        help='Archivo de configuración (para Whisper model)'
    )
    parser.add_argument(
        '--min-repetitions',
        type=int,
        default=3,
        help='Mínimo de repeticiones para considerar corrupto (default: 3)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Solo escanear y reportar, no modificar archivos'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Aplicar correcciones (recuperar de original + re-transcribir)'
    )
    
    args = parser.parse_args()
    
    # Paso 1: Escanear
    corrupted = scan_metadata_for_corruption(args.data_dir, min_repetitions=args.min_repetitions)
    
    # Paso 2: Reportar
    print_corruption_report(corrupted)
    
    # Paso 3: Corregir
    if args.fix and corrupted:
        print(f"\n{'='*60}")
        print(f"  INICIANDO RECUPERACIÓN")
        print(f"{'='*60}")
        stats = retranscribe_segments(corrupted, config_path=args.config, dry_run=args.dry_run)
        
        print(f"\n{'='*60}")
        print(f"  RESULTADOS FINALES")
        print(f"{'='*60}")
        print(f"  Recuperados de original:  {stats['recoverable_from_original']}")
        print(f"  Re-transcritos (Whisper): {stats['retranscribed']}")
        print(f"  Audio faltante:           {stats['audio_missing']}")
        print(f"  Fallidos:                 {stats['failed']}")
        print(f"  Archivos modificados:     {len(stats['files_modified'])}")
    elif corrupted and not args.fix:
        print(f"\n💡 Usa --fix para aplicar correcciones")
        print(f"   Primero prueba con --dry-run --fix para ver el plan")


if __name__ == '__main__':
    main()
