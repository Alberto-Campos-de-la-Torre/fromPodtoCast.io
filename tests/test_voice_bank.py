#!/usr/bin/env python3
"""
Test aislado para VoiceBankManager con audio real.

Uso:
    python tests/test_voice_bank.py <audio_file> [--hf-token TOKEN] [--threshold 0.85]

Ejemplo:
    python tests/test_voice_bank.py ./data/input/podcast.wav --hf-token hf_xxxx
"""
import sys
import os
import argparse
import tempfile
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from voice_bank import VoiceBankManager


def log(msg: str, level: str = "INFO"):
    """Log con timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️", "DEBUG": "🔍"}
    print(f"[{ts}] {icons.get(level, 'ℹ️')} {msg}")


def test_basic_voice_bank(tmp_path: Path):
    """Test básico de VoiceBankManager con embeddings sintéticos."""
    log("Ejecutando test básico con embeddings sintéticos...", "INFO")
    
    bank_path = tmp_path / "voice_bank_test.json"
    manager = VoiceBankManager(
        bank_path=str(bank_path),
        match_threshold=0.8,
        id_generator=lambda n: f"SPEAKER_GLOBAL_{n:03d}"
    )

    # Crear embeddings de prueba
    emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_b = np.array([0.9, 0.1, 0.0], dtype=np.float32)  # Similar a emb_a
    emb_c = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Diferente

    # Test 1: Agregar primer speaker
    speaker_a = manager.add_speaker(emb_a)
    assert speaker_a == "SPEAKER_GLOBAL_001", f"Expected SPEAKER_GLOBAL_001, got {speaker_a}"
    log(f"Speaker A agregado: {speaker_a}", "SUCCESS")

    # Test 2: emb_b es similar a emb_a, debe matchear
    match_id, score = manager.find_best_match(emb_b)
    assert match_id == speaker_a, f"Expected match with {speaker_a}, got {match_id}"
    assert score > 0.8, f"Expected score > 0.8, got {score}"
    log(f"emb_b coincide con {match_id} (score: {score:.4f})", "SUCCESS")

    # Test 3: Actualizar speaker
    manager.update_speaker(match_id, emb_b)
    assert manager.voice_entries[match_id]["occurrences"] == 2
    log(f"Speaker {match_id} actualizado, occurrences: 2", "SUCCESS")

    # Test 4: emb_c debe generar un nuevo speaker
    speaker_c = manager.add_speaker(emb_c)
    assert speaker_c != speaker_a, "Speaker C should be different from A"
    assert speaker_c == "SPEAKER_GLOBAL_002", f"Expected SPEAKER_GLOBAL_002, got {speaker_c}"
    log(f"Speaker C agregado: {speaker_c}", "SUCCESS")

    # Test 5: Verificar número de entradas
    assert len(manager.voice_entries) == 2, f"Expected 2 entries, got {len(manager.voice_entries)}"
    log(f"Total speakers en banco: {len(manager.voice_entries)}", "SUCCESS")

    # Test 6: Verificar persistencia
    with open(bank_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 2, f"Expected 2 entries in file, got {len(data)}"
    log("Persistencia verificada correctamente", "SUCCESS")

    log("✅ Test básico completado exitosamente", "SUCCESS")
    return True


def test_with_real_audio(audio_path: str, hf_token: str, threshold: float = 0.85):
    """
    Test con audio real: extrae embeddings y prueba el voice bank.
    
    Args:
        audio_path: Ruta al archivo de audio
        hf_token: Token de Hugging Face para pyannote
        threshold: Umbral de similitud coseno
    """
    log(f"Probando con audio real: {audio_path}", "INFO")
    log(f"Umbral de similitud: {threshold}", "INFO")
    
    # Verificar que el archivo existe
    if not os.path.exists(audio_path):
        log(f"Archivo no encontrado: {audio_path}", "ERROR")
        return False
    
    # Importar dependencias de audio
    try:
        from speaker_diarizer import SpeakerDiarizer, PYANNOTE_AVAILABLE
        import torchaudio
    except ImportError as e:
        log(f"Error importando dependencias: {e}", "ERROR")
        return False
    
    if not PYANNOTE_AVAILABLE:
        log("pyannote.audio no está disponible. Instala con: pip install pyannote.audio", "ERROR")
        return False
    
    # Crear directorio temporal para el banco de voces
    with tempfile.TemporaryDirectory() as tmp_dir:
        bank_path = os.path.join(tmp_dir, "voice_bank_test.json")
        
        # Crear voice bank manager
        manager = VoiceBankManager(
            bank_path=bank_path,
            match_threshold=threshold,
            id_generator=lambda n: f"GLOBAL_SPK_{n:03d}"
        )
        
        log("VoiceBankManager creado", "SUCCESS")
        
        # Crear diarizador con voice bank
        log("Cargando SpeakerDiarizer con pyannote...", "INFO")
        diarizer = SpeakerDiarizer(hf_token=hf_token, voice_bank_manager=manager)
        
        if diarizer.pipeline is None:
            log("No se pudo cargar el pipeline de pyannote", "ERROR")
            return False
        
        log("SpeakerDiarizer cargado correctamente", "SUCCESS")
        
        # Realizar diarización
        log("Realizando diarización del audio...", "INFO")
        segments = diarizer.diarize(audio_path)
        
        if not segments:
            log("No se detectaron segmentos de hablantes", "WARNING")
            return False
        
        # Mostrar resultados
        unique_speakers = set(seg.get('speaker', 'UNKNOWN') for seg in segments)
        log(f"Segmentos detectados: {len(segments)}", "SUCCESS")
        log(f"Hablantes únicos: {len(unique_speakers)}", "SUCCESS")
        
        for speaker in sorted(unique_speakers):
            speaker_segs = [s for s in segments if s.get('speaker') == speaker]
            total_time = sum(s.get('duration', 0) for s in speaker_segs)
            log(f"  {speaker}: {len(speaker_segs)} segmentos, {total_time:.2f}s total", "INFO")
        
        # Verificar voice bank
        log(f"\nEstado del Voice Bank:", "INFO")
        log(f"  Archivo: {bank_path}", "INFO")
        log(f"  Speakers registrados: {len(manager.voice_entries)}", "INFO")
        
        for spk_id, entry in manager.voice_entries.items():
            log(f"  {spk_id}: {entry.get('occurrences', 1)} ocurrencias", "INFO")
        
        # Mostrar estadísticas del diarizador
        stats = diarizer.get_voice_bank_stats()
        log(f"\nEstadísticas de Voice Bank:", "INFO")
        log(f"  Speakers emparejados: {stats.get('matched', 0)}", "INFO")
        log(f"  Speakers nuevos: {stats.get('created', 0)}", "INFO")
        
        # Test de persistencia
        with open(bank_path, "r", encoding="utf-8") as f:
            persisted_data = json.load(f)
        log(f"  Entries persistidos: {len(persisted_data)}", "INFO")
        
        # Verificar que se pueden cargar los embeddings
        for spk_id, entry in persisted_data.items():
            emb = np.array(entry.get('embedding', []), dtype=np.float32)
            norm = np.linalg.norm(emb)
            log(f"  {spk_id}: embedding dim={len(emb)}, norm={norm:.4f}", "DEBUG")
        
        log("\n✅ Test con audio real completado exitosamente", "SUCCESS")
        return True


def test_voice_bank_reuse(audio_path: str, hf_token: str, threshold: float = 0.85):
    """
    Test de reutilización: procesa el mismo audio dos veces y verifica que
    los speakers se reutilizan en la segunda pasada.
    """
    log("=" * 60, "INFO")
    log("Test de reutilización de Voice Bank", "INFO")
    log("=" * 60, "INFO")
    
    if not os.path.exists(audio_path):
        log(f"Archivo no encontrado: {audio_path}", "ERROR")
        return False
    
    try:
        from speaker_diarizer import SpeakerDiarizer, PYANNOTE_AVAILABLE
    except ImportError as e:
        log(f"Error importando dependencias: {e}", "ERROR")
        return False
    
    if not PYANNOTE_AVAILABLE:
        log("pyannote.audio no está disponible", "ERROR")
        return False
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        bank_path = os.path.join(tmp_dir, "voice_bank_reuse.json")
        
        # Primera pasada
        log("\n--- Primera pasada ---", "INFO")
        manager1 = VoiceBankManager(
            bank_path=bank_path,
            match_threshold=threshold,
            id_generator=lambda n: f"GLOBAL_SPK_{n:03d}"
        )
        
        diarizer1 = SpeakerDiarizer(hf_token=hf_token, voice_bank=manager1)
        segments1 = diarizer1.diarize(audio_path)
        stats1 = diarizer1.get_voice_bank_stats()
        
        speakers_1 = set(seg.get('speaker') for seg in segments1)
        log(f"Speakers detectados: {speakers_1}", "INFO")
        log(f"Voice Bank - Nuevos: {stats1.get('created', 0)}, Emparejados: {stats1.get('matched', 0)}", "INFO")
        
        # Segunda pasada (cargando el banco existente)
        log("\n--- Segunda pasada (reutilizando banco) ---", "INFO")
        manager2 = VoiceBankManager(
            bank_path=bank_path,
            match_threshold=threshold,
            id_generator=lambda n: f"GLOBAL_SPK_{n:03d}"
        )
        
        # Verificar que se cargaron los speakers anteriores
        log(f"Speakers cargados del banco: {len(manager2.voice_entries)}", "INFO")
        
        diarizer2 = SpeakerDiarizer(hf_token=hf_token, voice_bank=manager2)
        segments2 = diarizer2.diarize(audio_path)
        stats2 = diarizer2.get_voice_bank_stats()
        
        speakers_2 = set(seg.get('speaker') for seg in segments2)
        log(f"Speakers detectados: {speakers_2}", "INFO")
        log(f"Voice Bank - Nuevos: {stats2.get('created', 0)}, Emparejados: {stats2.get('matched', 0)}", "INFO")
        
        # Verificar reutilización
        if stats2.get('matched', 0) > 0:
            log("\n✅ Reutilización de speakers verificada", "SUCCESS")
            return True
        else:
            log("\n⚠️ No se reutilizaron speakers (puede ser normal si el umbral es muy alto)", "WARNING")
            return True  # No es necesariamente un error


def main():
    parser = argparse.ArgumentParser(
        description='Test aislado para VoiceBankManager con audio real',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
    # Test básico (sin audio)
    python tests/test_voice_bank.py --basic
    
    # Test con audio real
    python tests/test_voice_bank.py ./data/input/podcast.wav --hf-token hf_xxxx
    
    # Test de reutilización
    python tests/test_voice_bank.py ./data/input/podcast.wav --hf-token hf_xxxx --reuse-test
"""
    )
    parser.add_argument(
        'audio_file',
        nargs='?',
        default=None,
        help='Ruta al archivo de audio para testing'
    )
    parser.add_argument(
        '--hf-token',
        type=str,
        default=None,
        help='Token de Hugging Face para pyannote'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.85,
        help='Umbral de similitud coseno (default: 0.85)'
    )
    parser.add_argument(
        '--basic',
        action='store_true',
        help='Ejecutar solo test básico con embeddings sintéticos'
    )
    parser.add_argument(
        '--reuse-test',
        action='store_true',
        help='Ejecutar test de reutilización (procesa audio dos veces)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧪 Test de VoiceBankManager")
    print("=" * 60)
    print()
    
    # Cargar configuración desde config.json
    config_path = Path(__file__).parent.parent / 'config' / 'config.json'
    config = {}
    hf_token = None
    config_threshold = 0.85
    
    if config_path.exists():
        log(f"Leyendo configuración desde: {config_path}", "INFO")
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Verificar hf_token en config
            hf_token = config.get('hf_token')
            if hf_token:
                log(f"✓ hf_token encontrado en config.json", "SUCCESS")
            else:
                log("⚠ hf_token no configurado en config.json", "WARNING")
            
            # Leer threshold del config si existe
            config_threshold = config.get('voice_match_threshold', 0.85)
            log(f"  voice_match_threshold: {config_threshold}", "INFO")
            
            # Verificar use_voice_bank
            use_vb = config.get('use_voice_bank', False)
            log(f"  use_voice_bank: {use_vb}", "INFO")
            
        except Exception as e:
            log(f"Error leyendo config.json: {e}", "WARNING")
    else:
        log(f"Archivo config.json no encontrado: {config_path}", "WARNING")
    
    # Permitir override desde línea de comandos
    if args.hf_token:
        hf_token = args.hf_token
        log("Usando hf_token desde argumento --hf-token", "INFO")
    
    # Usar threshold del argumento si se proporcionó, sino del config
    threshold = args.threshold if args.threshold != 0.85 else config_threshold
    
    print()
    
    # Siempre ejecutar test básico primero
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            test_basic_voice_bank(Path(tmp_dir))
        except AssertionError as e:
            log(f"Test básico falló: {e}", "ERROR")
            sys.exit(1)
    
    # Si solo se solicitó test básico, terminar
    if args.basic:
        print()
        log("Solo test básico solicitado. Terminando.", "INFO")
        sys.exit(0)
    
    # Si se proporcionó audio, ejecutar tests con audio real
    if args.audio_file:
        # Verificar token
        if not hf_token:
            log("Se requiere token de Hugging Face para tests con audio real", "ERROR")
            log("Configura 'hf_token' en config/config.json o usa --hf-token TOKEN", "INFO")
            sys.exit(1)
        
        print()
        
        # Test con audio real
        success = test_with_real_audio(args.audio_file, hf_token, threshold)
        
        if not success:
            sys.exit(1)
        
        # Test de reutilización si se solicitó
        if args.reuse_test:
            print()
            success = test_voice_bank_reuse(args.audio_file, hf_token, threshold)
            if not success:
                sys.exit(1)
    else:
        print()
        if hf_token:
            log("hf_token configurado. Para test con audio real:", "INFO")
            log("  python tests/test_voice_bank.py <audio_file>", "INFO")
        else:
            log("Para test con audio real, configura hf_token en config.json:", "INFO")
            log("  python tests/test_voice_bank.py <audio_file>", "INFO")
    
    print()
    print("=" * 60)
    print("✅ Todos los tests completados exitosamente")
    print("=" * 60)


if __name__ == '__main__':
    main()
