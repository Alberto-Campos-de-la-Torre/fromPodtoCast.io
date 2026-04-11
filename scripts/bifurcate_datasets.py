#!/usr/bin/env python3
"""
bifurcate_datasets.py — Genera datasets limpios y separados para:
  - Moshi (conversacional): 2+ speakers, tolera overlap
  - Kyutai TTS (Local csm): incluye monólogos, rechaza overlap

Ambos: revertir correcciones LLM a Whisper crudo (verdad acústica).
"""

import json
import argparse
from pathlib import Path
from collections import Counter

# Patrones de texto envenenado (del bug de text_preprocessor.py)
POISONED_PATTERNS = [
    "Listado de enfermedades",
    "Lista de enfermedades",
    "Enfermedades del aparato",
    "Enfermedades del sistema",
    "Clasificación de enfermedades",
]


def is_poisoned(text: str) -> bool:
    return any(p in text for p in POISONED_PATTERNS)


def revert_to_whisper(entry: dict) -> str:
    """Recupera el texto original de Whisper (verdad acústica)."""
    return (
        entry.get('text_original')
        or entry.get('llm_correction', {}).get('original')
        or entry.get('text', '')
    )


def clean_entry(entry: dict) -> dict:
    """Reverte a Whisper y purga metadata LLM."""
    entry['text'] = revert_to_whisper(entry)
    for key in ['text_original', 'llm_correction', 'text_changes',
                'preprocessing_applied', 'reprocessing_info']:
        entry.pop(key, None)
    return entry


def bifurcate(data_dir: str, dry_run: bool = False):
    base = Path(data_dir)
    metadata_dir = base / "metadata"

    moshi_out = base / "metadata_moshi_ready"
    tts_out = base / "metadata_tts_ready"

    if not dry_run:
        moshi_out.mkdir(exist_ok=True)
        tts_out.mkdir(exist_ok=True)

    stats = {
        "analizados": 0, "vacios": 0, "no_json": 0,
        "tts_guardados": 0, "moshi_guardados": 0,
        "tts_segments": 0, "moshi_segments": 0,
        "tts_monologo": 0, "tts_conversacion": 0,
        "revertidos": 0, "envenenados": 0,
        "overlap_filtered": 0,
    }

    json_files = sorted(f for f in metadata_dir.iterdir()
                        if f.suffix == '.json' and '.backup' not in f.name)

    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            stats["no_json"] += 1
            continue

        # Normalize to list
        if isinstance(data, dict):
            entries = data.get('segments', data.get('entries', []))
        elif isinstance(data, list):
            entries = data
        else:
            stats["no_json"] += 1
            continue

        if not entries or len(entries) < 3:
            stats["vacios"] += 1
            continue

        stats["analizados"] += 1

        # ── Phase 1: Base cleanup (both models) ──
        clean = []
        for e in entries:
            original_text = e.get('text', '')
            whisper_text = revert_to_whisper(e)

            if is_poisoned(whisper_text):
                stats["envenenados"] += 1
                continue

            if whisper_text != original_text:
                stats["revertidos"] += 1

            clean.append(clean_entry(e.copy()))

        if len(clean) < 3:
            continue

        # ── Phase 2: TTS dataset (includes monologues, rejects overlap) ──
        MAX_OVERLAP = 0.15
        tts_entries = [e for e in clean
                       if e.get('overlap_ratio', 0) < MAX_OVERLAP]
        stats["overlap_filtered"] += len(clean) - len(tts_entries)

        if tts_entries:
            spk_counts = Counter(e.get('speaker_label', 'SPK_00') for e in tts_entries)
            if len(spk_counts) <= 1:
                stats["tts_monologo"] += 1
            else:
                stats["tts_conversacion"] += 1

            stats["tts_segments"] += len(tts_entries)
            stats["tts_guardados"] += 1

            mark = "MONO" if len(spk_counts) <= 1 else f"{len(spk_counts)}spk"
            print(f"  TTS  {jf.name:60s} | {len(tts_entries):4d} segs | {mark}")

            if not dry_run:
                with open(tts_out / jf.name, 'w', encoding='utf-8') as f:
                    json.dump(tts_entries, f, indent=2, ensure_ascii=False)

        # ── Phase 3: Moshi dataset (2+ speakers, 10% min 2nd) ──
        speaker_times = Counter()
        for e in clean:
            speaker_times[e.get('speaker_label', 'SPK_00')] += e.get('duration', 0)

        top = speaker_times.most_common(2)
        total_time = sum(speaker_times.values())

        if (len(top) >= 2 and total_time > 0
                and (top[1][1] / total_time) >= 0.10):
            stats["moshi_segments"] += len(clean)
            stats["moshi_guardados"] += 1

            pct = int(100 * top[1][1] / total_time)
            print(f"  MOSHI {jf.name:58s} | {len(clean):4d} segs | "
                  f"{len(speaker_times)} spk | 2nd={pct}%")

            if not dry_run:
                with open(moshi_out / jf.name, 'w', encoding='utf-8') as f:
                    json.dump(clean, f, indent=2, ensure_ascii=False)

    # ── Report ──
    print("\n" + "=" * 70)
    print("  BIFURCACION DE DATASETS")
    print("=" * 70)
    print(f"  Archivos analizados:            {stats['analizados']}")
    print(f"  Descartados (no JSON):          {stats['no_json']}")
    print(f"  Descartados (vacios):           {stats['vacios']}")
    print(f"  Segmentos revertidos:           {stats['revertidos']}")
    print(f"  Envenenados eliminados:         {stats['envenenados']}")
    print(f"  Filtrados por overlap:          {stats['overlap_filtered']}")
    print("  " + "-" * 50)
    print(f"  TTS (Local csm):                {stats['tts_guardados']} archivos")
    print(f"     Monologos (oro puro):        {stats['tts_monologo']}")
    print(f"     Conversaciones:              {stats['tts_conversacion']}")
    print(f"     Segmentos totales:           {stats['tts_segments']}")
    print(f"  Moshi (conversacional):         {stats['moshi_guardados']} archivos")
    print(f"     Segmentos totales:           {stats['moshi_segments']}")
    print("  " + "-" * 50)

    if dry_run:
        print("  DRY-RUN — no se escribio nada")
    else:
        print(f"  TTS -> {tts_out}")
        print(f"  Moshi -> {moshi_out}")

    print("=" * 70)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bifurcar dataset para TTS y Moshi")
    parser.add_argument("--data-dir", type=str,
                        default="/media/ttech-main/8464b39b-ba5b-49c5-9e05-010122b9874d",
                        help="Directorio raiz con metadata/")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo analizar, no escribir")
    args = parser.parse_args()
    bifurcate(args.data_dir, dry_run=args.dry_run)
