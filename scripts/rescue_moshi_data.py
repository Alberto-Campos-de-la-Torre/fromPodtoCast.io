#!/usr/bin/env python3
"""
rescue_moshi_data.py - Rescate de datos para entrenamiento Moshi conversacional.

Propuesta A+: Revertir TODAS las correcciones LLM al texto Whisper original
(la verdad acústica) y filtrar por calidad conversacional.

Principio: El texto crudo de Whisper es mejor que el texto sanitizado para TTS.
Si el hablante dijo "FEVI disminuida", el texto DEBE decir eso, no "fracción de eyección".

Uso:
    python scripts/rescue_moshi_data.py --data-dir /path/to/data
    python scripts/rescue_moshi_data.py --data-dir /path/to/data --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime


def rescue_conversational_dataset(data_dir: str, dry_run: bool = False):
    """
    Rescata datos conversacionales para entrenamiento Moshi.
    
    1. Revierte text → text_original (Whisper puro)
    2. Filtra mono-hablantes (requiere ≥2 speakers)
    3. Filtra baja interacción (2do speaker ≥10% del tiempo)
    4. Filtra segmentos envenenados (bug del preprocesador)
    5. Genera metadata limpia en metadata_moshi_ready/
    """
    base_path = Path(data_dir)
    metadata_dir = base_path / "metadata"
    output_dir = base_path / "metadata_moshi_ready"
    
    if not metadata_dir.exists():
        print(f"❌ Directorio de metadata no encontrado: {metadata_dir}")
        sys.exit(1)
    
    if not dry_run:
        output_dir.mkdir(exist_ok=True)

    stats = {
        "archivos_analizados": 0,
        "descartados_no_json": 0,
        "descartados_vacio": 0,
        "descartados_monologo": 0,
        "descartados_baja_interaccion": 0,
        "descartados_pocos_segmentos": 0,
        "archivos_validos": 0,
        "segmentos_revertidos": 0,
        "segmentos_sin_cambio": 0,
        "segmentos_envenenados": 0,
        "segmentos_totales": 0,
        "speakers_distribution": Counter(),
    }
    
    # Textos envenenados conocidos (del bug del preprocesador/RCA anterior)
    POISON_MARKERS = [
        "Listado de enfermedades",
    ]

    json_files = sorted(metadata_dir.glob("*.json"))
    # Excluir backups y archivos del sistema
    json_files = [f for f in json_files if ".backup" not in f.name 
                  and f.name != "processed_videos.json"
                  and f.name != "reprocess_progress.json"]
    
    print(f"\n{'='*60}")
    print(f"  🚑 RESCATE DE DATOS MOSHI — Propuesta A+")
    print(f"  📁 Origen: {metadata_dir}")
    print(f"  📁 Destino: {output_dir}")
    print(f"  📄 Archivos encontrados: {len(json_files)}")
    if dry_run:
        print(f"  ⚠️  MODO DRY-RUN — no se escribirá nada")
    print(f"{'='*60}\n")

    for json_file in json_files:
        # Cargar archivo
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            stats["descartados_no_json"] += 1
            continue
        
        # Extraer entries
        if isinstance(data, dict):
            entries = data.get('segments', data.get('entries', []))
        elif isinstance(data, list):
            entries = data
        else:
            stats["descartados_no_json"] += 1
            continue
            
        if not isinstance(entries, list) or len(entries) == 0:
            stats["descartados_vacio"] += 1
            continue
        
        stats["archivos_analizados"] += 1

        # ─── 1. FILTRO CONVERSACIONAL ESTRICTO ───
        speakers = [e.get("speaker_label", e.get("speaker", "SPK_00")) for e in entries]
        unique_speakers = set(speakers)
        stats["speakers_distribution"][len(unique_speakers)] += 1
        
        if len(unique_speakers) < 2:
            stats["descartados_monologo"] += 1
            continue
        
        # Calcular tiempo por hablante
        speaker_times = Counter()
        for e in entries:
            spk = e.get('speaker_label', e.get('speaker', 'SPK_00'))
            duration = e.get('duration', 0)
            if duration == 0:
                # Intentar calcular de start/end
                start = e.get('start', 0)
                end = e.get('end', 0)
                duration = end - start
            speaker_times[spk] += max(duration, 0)
        
        total_time = sum(speaker_times.values())
        if total_time <= 0:
            stats["descartados_baja_interaccion"] += 1
            continue
            
        top_speakers = speaker_times.most_common()
        
        # El segundo hablante debe tener al menos 10% del tiempo total
        if len(top_speakers) < 2 or (top_speakers[1][1] / total_time) < 0.10:
            stats["descartados_baja_interaccion"] += 1
            continue

        # ─── 2. REVERSIÓN A WHISPER ORIGINAL ───
        clean_entries = []
        for entry in entries:
            # Detectar texto envenenado
            text_check = entry.get('text', '') + entry.get('text_original', '')
            if any(marker in text_check for marker in POISON_MARKERS):
                stats["segmentos_envenenados"] += 1
                continue
            
            # Intentar revertir al texto original de Whisper
            texto_whisper = None
            
            # Prioridad 1: text_original (guardado por nuestro pipeline)
            if 'text_original' in entry:
                texto_whisper = entry['text_original']
                stats["segmentos_revertidos"] += 1
            # Prioridad 2: llm_correction.original (formato alternativo)
            elif 'llm_correction' in entry and isinstance(entry['llm_correction'], dict):
                texto_whisper = entry['llm_correction'].get('original')
                if texto_whisper:
                    stats["segmentos_revertidos"] += 1
            
            if texto_whisper:
                entry['text'] = texto_whisper
            else:
                # Nunca fue tocado por LLM — usar text tal cual
                stats["segmentos_sin_cambio"] += 1
            
            # Verificar que hay texto válido
            if not entry.get('text', '').strip():
                continue
            
            # Purgar metadata LLM residual
            for key in ['text_original', 'llm_correction', 'text_changes', 
                        'llm_error', 'preprocessing_applied', 'mcp_verified',
                        'mcp_reverted', 'mcp_razon', 'mcp_confianza',
                        'mcp_validaciones', 'mcp_verified_at',
                        'reprocessing_info']:
                entry.pop(key, None)
            
            clean_entries.append(entry)
            stats["segmentos_totales"] += 1

        # Mínimo 10 segmentos para ser útil
        if len(clean_entries) < 10:
            stats["descartados_pocos_segmentos"] += 1
            continue

        # ─── 3. GUARDAR RESULTADO LIMPIO ───
        if not dry_run:
            if isinstance(data, dict):
                if 'segments' in data:
                    data['segments'] = clean_entries
                elif 'entries' in data:
                    data['entries'] = clean_entries
                # Purgar metadata de reprocesamiento
                data.pop('reprocessing_info', None)
                save_data = data
            else:
                save_data = clean_entries
            
            # Agregar metadata de rescate
            if isinstance(save_data, dict):
                save_data['_rescue_info'] = {
                    'rescued_at': datetime.now().isoformat(),
                    'method': 'propuesta_a_plus',
                    'speakers': len(unique_speakers),
                    'top_speakers': {spk: round(t, 1) for spk, t in top_speakers[:3]},
                    'total_segments': len(clean_entries),
                    'reverted_to_whisper': stats["segmentos_revertidos"],
                }

            with open(output_dir / json_file.name, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        stats["archivos_validos"] += 1
        
        # Log por archivo
        interaction_pct = (top_speakers[1][1] / total_time * 100) if total_time > 0 else 0
        print(f"  ✅ {json_file.name[:55]:55s} | {len(clean_entries):3d} segs | "
              f"{len(unique_speakers)} spk | 2nd={interaction_pct:.0f}%")

    # ─── REPORTE FINAL ───
    print(f"\n{'='*60}")
    print(f"  🎯 REPORTE DE RESCATE DE DATOS PARA MOSHI")
    print(f"{'='*60}")
    print(f"  Archivos analizados:            {stats['archivos_analizados']}")
    print(f"  ❌ Descartados (no JSON):        {stats['descartados_no_json']}")
    print(f"  ❌ Descartados (vacíos):          {stats['descartados_vacio']}")
    print(f"  ❌ Descartados (monólogos):       {stats['descartados_monologo']}")
    print(f"  ❌ Descartados (falso diálogo):   {stats['descartados_baja_interaccion']}")
    print(f"  ❌ Descartados (pocos segmentos): {stats['descartados_pocos_segmentos']}")
    print(f"  ────────────────────────────────────────")
    print(f"  ✅ Archivos válidos (Moshi-ready): {stats['archivos_validos']}")
    print(f"  ────────────────────────────────────────")
    print(f"  📝 Segmentos totales:          {stats['segmentos_totales']}")
    print(f"  🔄 Revertidos a Whisper:       {stats['segmentos_revertidos']}")
    print(f"  ✓  Sin cambio (ya limpios):    {stats['segmentos_sin_cambio']}")
    print(f"  ☠️  Envenenados (descartados):  {stats['segmentos_envenenados']}")
    print(f"  ────────────────────────────────────────")
    print(f"  📊 Distribución de speakers:")
    for n_spk, count in sorted(stats['speakers_distribution'].items()):
        label = "⭐" if n_spk >= 2 else "  "
        print(f"     {label} {n_spk} speaker(s): {count} archivos")
    
    if not dry_run:
        print(f"\n  📁 Datos listos en: {output_dir}")
    else:
        print(f"\n  ⚠️  DRY-RUN completado — no se escribió nada")
    
    print(f"{'='*60}\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Rescate de datos para entrenamiento Moshi conversacional (Propuesta A+)'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default='/media/ttech-main/8464b39b-ba5b-49c5-9e05-010122b9874d',
        help='Directorio base de datos'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Solo analizar, no escribir archivos'
    )
    
    args = parser.parse_args()
    rescue_conversational_dataset(args.data_dir, args.dry_run)


if __name__ == "__main__":
    main()
