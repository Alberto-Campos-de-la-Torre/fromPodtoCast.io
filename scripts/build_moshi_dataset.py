#!/usr/bin/env python3
"""
build_moshi_dataset.py - Extractor de Pares Conversacionales para Moshiko

Escanea todos los JSON de metadata generados por fromPodtoCast y extrae
pares conversacionales (A→B) usando 3 reglas de oro:
1. Cambio de turno real (A ≠ B)
2. Gap temporal natural (-1.0s ≤ gap ≤ 1.5s)
3. Filtrado de respuestas cortas/basura

Genera archivos .pt con tokens de audio Mimi y tokens de texto SPM.
"""
import json
import os
import sys
import argparse
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# --- CONFIGURACIÓN POR DEFECTO ---
DEFAULT_METADATA_DIR = "/media/ttech-main/8464b39b-ba5b-49c5-9e05-010122b9874d/metadata"
DEFAULT_OUTPUT_DIR = "/media/ttech-main/8464b39b-ba5b-49c5-9e05-010122b9874d/moshi_dataset_100k/train"
DEFAULT_DEVICE = "cuda:0"

MAX_GAP = 1.5   # Máximo silencio permitido entre turnos (segundos)
MIN_GAP = -1.0  # Máximo solapamiento permitido (segundos, negativo = se pisan)
MIN_WORDS = 3   # Mínimo de palabras por turno
MIN_DURATION = 1.5  # Duración mínima de segmento (segundos)

# Textos basura generados por el pipeline
GARBAGE_PATTERNS = [
    "Listado de enfermedades",
    "Texto verificado",
    "Subtítulos realizados",
    "Suscríbete",
    "Dale like",
    "Gracias por ver",
]


def load_tokenizers(device):
    """Carga Mimi y SPM tokenizer de Moshiko."""
    print("🔧 Cargando Mimi y SPM Tokenizer...")
    from moshi.models import loaders
    
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
    spm = checkpoint_info.get_text_tokenizer()
    mimi = loaders.get_mimi(
        filename=loaders.hf_hub_download("kyutai/mimi", "model.safetensors"),
        device=device
    )
    mimi.eval()
    print("✅ Tokenizers cargados")
    return mimi, spm


def is_garbage_text(text):
    """Detecta texto basura/plantilla."""
    for pattern in GARBAGE_PATTERNS:
        if pattern.lower() in text.lower():
            return True
    return False


def extract_pairs_from_metadata(json_path, max_gap=None, min_gap=None):
    """
    Extrae pares conversacionales de un archivo de metadata.
    
    Returns:
        Lista de tuplas (seg_a, seg_b) que pasan los filtros
    """
    _max_gap = max_gap if max_gap is not None else MAX_GAP
    _min_gap = min_gap if min_gap is not None else MIN_GAP
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, Exception):
        return []
    
    # Manejar diferentes formatos de metadata
    if isinstance(data, dict):
        entries = data.get('segments', data.get('entries', []))
    elif isinstance(data, list):
        entries = data
    else:
        return []
    
    if not isinstance(entries, list) or len(entries) < 2:
        return []
    
    # Ordenar cronológicamente
    entries = sorted(entries, key=lambda x: x.get('start', 0.0))
    
    pairs = []
    for i in range(len(entries) - 1):
        seg_a = entries[i]
        seg_b = entries[i + 1]
        
        # Regla 1: Cambio de turno real
        if seg_a.get('speaker_label') == seg_b.get('speaker_label'):
            continue
        
        # Regla 2: Gap temporal natural
        gap = seg_b.get('start', 0.0) - seg_a.get('end', 0.0)
        if gap > _max_gap or gap < _min_gap:
            continue
        
        # Regla 3: Filtrado de calidad
        text_a = seg_a.get('text_original', seg_a.get('text', '')).strip()
        text_b = seg_b.get('text_original', seg_b.get('text', '')).strip()
        
        if len(text_a.split()) < MIN_WORDS or len(text_b.split()) < MIN_WORDS:
            continue
        
        # Filtro de duración mínima
        dur_a = seg_a.get('duration', seg_a.get('end', 0) - seg_a.get('start', 0))
        dur_b = seg_b.get('duration', seg_b.get('end', 0) - seg_b.get('start', 0))
        if dur_a < MIN_DURATION or dur_b < MIN_DURATION:
            continue
        
        # Filtro de texto basura
        if is_garbage_text(text_a) or is_garbage_text(text_b):
            continue
        
        # Verificar que los archivos de audio existen
        if not os.path.exists(seg_a.get('path', '')) or not os.path.exists(seg_b.get('path', '')):
            continue
        
        pairs.append((seg_a, seg_b, gap))
    
    return pairs


def tokenize_pair(seg_a, seg_b, mimi, spm, device):
    """
    Tokeniza un par conversacional con Mimi (audio) y SPM (texto).
    
    Returns:
        dict con tokens o None si falla
    """
    try:
        text_a = seg_a.get('text_original', seg_a.get('text', '')).strip()
        text_b = seg_b.get('text_original', seg_b.get('text', '')).strip()
        
        wav_a, sr_a = torchaudio.load(seg_a['path'])
        wav_b, sr_b = torchaudio.load(seg_b['path'])
        
        # Resamplear si es necesario
        if sr_a != mimi.sample_rate:
            wav_a = torchaudio.functional.resample(wav_a, sr_a, mimi.sample_rate)
        if sr_b != mimi.sample_rate:
            wav_b = torchaudio.functional.resample(wav_b, sr_b, mimi.sample_rate)
        
        with torch.no_grad():
            # Moshiko usa 8 codebooks por usuario (total 16)
            codes_u = mimi.encode(wav_a.mean(0, keepdim=True).unsqueeze(0).to(device)).cpu()[0, :8, :]
            codes_s = mimi.encode(wav_b.mean(0, keepdim=True).unsqueeze(0).to(device)).cpu()[0, :8, :]
        
        # Tokenizar texto con SPM 32K oficial
        tt_u = torch.tensor(spm.encode(text_a), dtype=torch.long)
        tt_s = torch.tensor(spm.encode(text_b), dtype=torch.long)
        
        return {
            "codes_user": codes_u,
            "codes_system": codes_s,
            "text_tokens_user": tt_u,
            "text_tokens_system": tt_s,
            "text_user": text_a,
            "text_system": text_b,
        }
    except Exception as e:
        return None


def scan_only(metadata_dir):
    """Modo scan: solo cuenta pares sin tokenizar (rápido)."""
    json_files = sorted(Path(metadata_dir).glob("*.json"))
    
    stats = {
        "podcasts": 0,
        "total_segments": 0,
        "pares_encontrados": 0,
        "rechazos_monologo": 0,
        "rechazos_gap": 0,
        "rechazos_calidad": 0,
        "rechazos_archivo": 0,
        "gaps": [],
    }
    
    print(f"🔍 Escaneando {len(json_files)} archivos de metadata...")
    
    for json_file in tqdm(json_files, desc="Escaneando"):
        if ".backup" in json_file.name:
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            continue
        
        entries = data if isinstance(data, list) else data.get('segments', data.get('entries', []))
        if not isinstance(entries, list):
            continue
        
        entries = sorted(entries, key=lambda x: x.get('start', 0.0))
        stats["total_segments"] += len(entries)
        
        for i in range(len(entries) - 1):
            seg_a = entries[i]
            seg_b = entries[i + 1]
            
            if seg_a.get('speaker_label') == seg_b.get('speaker_label'):
                stats["rechazos_monologo"] += 1
                continue
            
            gap = seg_b.get('start', 0.0) - seg_a.get('end', 0.0)
            if gap > MAX_GAP or gap < MIN_GAP:
                stats["rechazos_gap"] += 1
                continue
            
            text_a = seg_a.get('text_original', seg_a.get('text', '')).strip()
            text_b = seg_b.get('text_original', seg_b.get('text', '')).strip()
            
            if len(text_a.split()) < MIN_WORDS or len(text_b.split()) < MIN_WORDS:
                stats["rechazos_calidad"] += 1
                continue
            
            dur_a = seg_a.get('duration', seg_a.get('end', 0) - seg_a.get('start', 0))
            dur_b = seg_b.get('duration', seg_b.get('end', 0) - seg_b.get('start', 0))
            if dur_a < MIN_DURATION or dur_b < MIN_DURATION:
                stats["rechazos_calidad"] += 1
                continue
            
            if is_garbage_text(text_a) or is_garbage_text(text_b):
                stats["rechazos_calidad"] += 1
                continue
            
            if not os.path.exists(seg_a.get('path', '')) or not os.path.exists(seg_b.get('path', '')):
                stats["rechazos_archivo"] += 1
                continue
            
            stats["pares_encontrados"] += 1
            stats["gaps"].append(gap)
        
        stats["podcasts"] += 1
    
    # Reporte
    print("\n" + "=" * 60)
    print("🔍 REPORTE DE ESCANEO (sin tokenización)")
    print("=" * 60)
    print(f"📁 Podcasts analizados:       {stats['podcasts']}")
    print(f"🔊 Segmentos totales:         {stats['total_segments']}")
    print(f"✅ PARES CONVERSACIONALES:     {stats['pares_encontrados']} 🔥")
    print(f"❌ Rechazos (monólogo):        {stats['rechazos_monologo']}")
    print(f"❌ Rechazos (gap temporal):    {stats['rechazos_gap']}")
    print(f"❌ Rechazos (calidad/corto):   {stats['rechazos_calidad']}")
    print(f"❌ Rechazos (archivo falta):   {stats['rechazos_archivo']}")
    
    if stats["gaps"]:
        import statistics
        gaps = stats["gaps"]
        print(f"\n📊 Distribución de gaps:")
        print(f"   Media:   {statistics.mean(gaps):.3f}s")
        print(f"   Mediana: {statistics.median(gaps):.3f}s")
        print(f"   Min:     {min(gaps):.3f}s")
        print(f"   Max:     {max(gaps):.3f}s")
        overlaps = sum(1 for g in gaps if g < 0)
        print(f"   Solapamientos (full-duplex): {overlaps} ({100*overlaps/len(gaps):.1f}%)")
    
    return stats


def create_dataset(metadata_dir, output_dir, device):
    """Modo build: extrae y tokeniza todos los pares conversacionales."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    mimi, spm = load_tokenizers(device)
    
    json_files = sorted(Path(metadata_dir).glob("*.json"))
    
    stats = {
        "podcasts": 0,
        "pares_creados": 0,
        "pares_fallidos": 0,
        "rechazos_monologo": 0,
        "rechazos_gap": 0,
    }
    
    print(f"\n🏭 Procesando {len(json_files)} archivos de metadata...")
    
    for json_file in tqdm(json_files, desc="Construyendo dataset"):
        if ".backup" in json_file.name:
            continue
        
        pairs = extract_pairs_from_metadata(json_file)
        
        for idx, (seg_a, seg_b, gap) in enumerate(pairs):
            pair_data = tokenize_pair(seg_a, seg_b, mimi, spm, device)
            
            if pair_data is not None:
                out_name = f"{json_file.stem}_pair_{idx:04d}.pt"
                torch.save(pair_data, Path(output_dir) / out_name)
                stats["pares_creados"] += 1
            else:
                stats["pares_fallidos"] += 1
        
        stats["podcasts"] += 1
        
        # Limpiar cache CUDA periódicamente
        if stats["podcasts"] % 50 == 0:
            torch.cuda.empty_cache()
    
    # Reporte final
    print("\n" + "=" * 60)
    print("🎙️  DATASET CONVERSACIONAL MASIVO COMPLETADO")
    print("=" * 60)
    print(f"📁 Podcasts analizados:       {stats['podcasts']}")
    print(f"✅ PARES PERFECTOS CREADOS:    {stats['pares_creados']} 🔥")
    print(f"❌ Pares fallidos (audio):     {stats['pares_fallidos']}")
    print(f"💾 Guardados en:              {output_dir}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extractor de pares conversacionales para Moshiko"
    )
    parser.add_argument(
        '--metadata-dir', '-m',
        default=DEFAULT_METADATA_DIR,
        help=f'Directorio con metadata JSONs (default: {DEFAULT_METADATA_DIR})'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=DEFAULT_OUTPUT_DIR,
        help=f'Directorio de salida para .pt (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--device', '-d',
        default=DEFAULT_DEVICE,
        help=f'Dispositivo CUDA (default: {DEFAULT_DEVICE})'
    )
    parser.add_argument(
        '--scan-only', '-s',
        action='store_true',
        help='Solo escanear y reportar sin tokenizar (rápido)'
    )
    parser.add_argument(
        '--max-gap',
        type=float, default=MAX_GAP,
        help=f'Máximo silencio entre turnos en segundos (default: {MAX_GAP})'
    )
    parser.add_argument(
        '--min-gap',
        type=float, default=MIN_GAP,
        help=f'Máximo solapamiento en segundos (default: {MIN_GAP})'
    )
    
    args = parser.parse_args()
    
    # Use args for gap values in scan/build
    max_gap_val = args.max_gap
    min_gap_val = args.min_gap
    
    print("=" * 60)
    print("🎙️  Moshiko Conversational Dataset Builder")
    print("=" * 60)
    print(f"📁 Metadata: {args.metadata_dir}")
    print(f"⚙️  Gap range: [{args.min_gap}s, {args.max_gap}s]")
    print(f"⏱️  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if args.scan_only:
        scan_only(args.metadata_dir)
    else:
        print(f"💾 Output: {args.output_dir}")
        print(f"🖥️  Device: {args.device}")
        create_dataset(args.metadata_dir, args.output_dir, args.device)


if __name__ == "__main__":
    main()
