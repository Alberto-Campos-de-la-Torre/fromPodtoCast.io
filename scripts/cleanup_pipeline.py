#!/usr/bin/env python3
"""
Cleanup Pipeline - Herramienta de limpieza y recuperación
==========================================

Este script limpia archivos temporales y recupera el estado del pipeline
cuando se traba inesperadamente durante el procesamiento de un audio.

Funcionalidades:
1. Limpia archivos temporales (.tmp, .temp, .part)
2. Libera memoria GPU
3. Mata procesos colgados de ffmpeg/whisper/ollama
4. Limpia directorios de segmentos incompletos
5. Muestra el estado del último procesamiento
6. Permite recuperar el procesamiento desde el último video exitoso

Uso:
    python scripts/cleanup_pipeline.py                  # Análisis y limpieza básica
    python scripts/cleanup_pipeline.py --deep           # Limpieza profunda
    python scripts/cleanup_pipeline.py --kill-processes # Matar procesos colgados
    python scripts/cleanup_pipeline.py --show-status    # Ver estado del pipeline
    python scripts/cleanup_pipeline.py --recover        # Recuperar procesamiento
"""

import argparse
import json
import os
import sys
import subprocess
import signal
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Configuración de paths
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_PATH = '/media/ttech-main/8464b39b-ba5b-49c5-9e05-010122b9874d'

# Colores ANSI
class Colors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'


def log(message: str, level: str = "INFO"):
    """Log con timestamp y colores."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    icons = {
        "INFO": ("ℹ️ ", Colors.CYAN),
        "SUCCESS": ("✅", Colors.GREEN),
        "WARNING": ("⚠️ ", Colors.YELLOW),
        "ERROR": ("❌", Colors.RED),
        "CLEAN": ("🧹", Colors.BLUE),
    }
    icon, color = icons.get(level, ("", ""))
    print(f"{color}[{timestamp}] {icon} {message}{Colors.RESET}")


def cleanup_temp_files(data_path: str, deep: bool = False) -> Dict:
    """
    Limpia archivos temporales del directorio de datos.
    
    Args:
        data_path: Ruta al directorio de datos
        deep: Si True, limpia también archivos de video/audio descargados parcialmente
    
    Returns:
        Diccionario con estadísticas de limpieza
    """
    stats = {
        'temp_files': 0,
        'temp_size': 0,
        'partial_downloads': 0,
        'partial_size': 0,
        'empty_dirs': 0,
    }
    
    log("Limpiando archivos temporales...", "CLEAN")
    
    # Patrones de archivos temporales
    temp_patterns = ['*.tmp', '*.temp', '*.part', '*~']
    
    # Directorios a revisar
    dirs_to_check = [
        os.path.join(data_path, 'input'),
        os.path.join(data_path, 'segments'),
        os.path.join(data_path, 'normalized'),
        tempfile.gettempdir(),
    ]
    
    for directory in dirs_to_check:
        if not os.path.exists(directory):
            continue
        
        log(f"   Revisando: {directory}", "INFO")
        
        for pattern in temp_patterns:
            for file_path in Path(directory).rglob(pattern):
                try:
                    size = file_path.stat().st_size
                    file_path.unlink()
                    stats['temp_files'] += 1
                    stats['temp_size'] += size
                    log(f"      🗑️  {file_path.name} ({size/1024/1024:.1f} MB)", "CLEAN")
                except Exception as e:
                    log(f"      ⚠️  No se pudo eliminar {file_path.name}: {e}", "WARNING")
    
    # Limpieza profunda (archivos de descarga incompletos)
    if deep:
        log("   Limpieza profunda activada...", "INFO")
        input_dir = os.path.join(data_path, 'input')
        
        if os.path.exists(input_dir):
            # Eliminar videos/audios que no están en el registro
            video_patterns = ['*.mp4', '*.webm', '*.m4a', '*.mkv']
            for pattern in video_patterns:
                for file_path in Path(input_dir).glob(pattern):
                    try:
                        size = file_path.stat().st_size
                        file_path.unlink()
                        stats['partial_downloads'] += 1
                        stats['partial_size'] += size
                        log(f"      🗑️  {file_path.name} ({size/1024/1024:.1f} MB)", "CLEAN")
                    except Exception as e:
                        log(f"      ⚠️  No se pudo eliminar {file_path.name}: {e}", "WARNING")
    
    # Limpiar directorios vacíos en segments/
    segments_base = os.path.join(data_path, 'segments')
    if os.path.exists(segments_base):
        for subdir in Path(segments_base).iterdir():
            if subdir.is_dir() and not any(subdir.iterdir()):
                try:
                    subdir.rmdir()
                    stats['empty_dirs'] += 1
                    log(f"      📁🗑️  Directorio vacío eliminado: {subdir.name}", "CLEAN")
                except Exception as e:
                    log(f"      ⚠️  No se pudo eliminar {subdir.name}: {e}", "WARNING")
    
    return stats


def cleanup_gpu_memory() -> bool:
    """
    Libera memoria GPU usando PyTorch si está disponible.
    
    Returns:
        True si se limpió la memoria GPU, False si no hay GPU/PyTorch
    """
    log("Limpiando memoria GPU...", "CLEAN")
    
    try:
        import torch
        import gc
        
        if not torch.cuda.is_available():
            log("   No hay GPU disponible", "INFO")
            return False
        
        # Limpiar memoria de Python
        gc.collect()
        
        # Limpiar memoria de cada GPU
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        log(f"   ✓ Memoria GPU liberada en {device_count} dispositivo(s)", "SUCCESS")
        return True
        
    except ImportError:
        log("   PyTorch no disponible, saltando limpieza GPU", "INFO")
        return False
    except Exception as e:
        log(f"   ⚠️  Error limpiando GPU: {e}", "WARNING")
        return False


def kill_stuck_processes(dry_run: bool = False) -> Dict:
    """
    Mata procesos que pueden quedarse colgados.
    
    Args:
        dry_run: Si True, solo muestra los procesos sin matarlos
    
    Returns:
        Diccionario con estadísticas de procesos
    """
    stats = {
        'ffmpeg': 0,
        'whisper': 0,
        'python': 0,
        'yt-dlp': 0,
    }
    
    log("Buscando procesos colgados...", "CLEAN")
    
    # Procesos a buscar
    process_patterns = {
        'ffmpeg': ['ffmpeg'],
        'yt-dlp': ['yt-dlp', 'youtube-dl'],
        'whisper': ['whisper'],
        'python': ['main.py', 'auto_pipeline.py'],
    }
    
    try:
        # Obtener lista de procesos
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        
        lines = result.stdout.split('\n')
        
        for category, patterns in process_patterns.items():
            for line in lines:
                # Saltar el propio proceso de cleanup
                if 'cleanup_pipeline.py' in line:
                    continue
                
                for pattern in patterns:
                    if pattern in line.lower():
                        # Extraer PID
                        parts = line.split()
                        if len(parts) > 1:
                            try:
                                pid = int(parts[1])
                                
                                if dry_run:
                                    log(f"   [DRY-RUN] Proceso: {category} (PID: {pid})", "WARNING")
                                    log(f"            {line[:100]}", "INFO")
                                else:
                                    try:
                                        os.kill(pid, signal.SIGTERM)
                                        stats[category] += 1
                                        log(f"   ☠️  Matado: {category} (PID: {pid})", "CLEAN")
                                    except ProcessLookupError:
                                        pass  # El proceso ya no existe
                                    except PermissionError:
                                        log(f"   ⚠️  Sin permisos para matar PID {pid}", "WARNING")
                            except (ValueError, IndexError):
                                pass
        
    except Exception as e:
        log(f"   ⚠️  Error buscando procesos: {e}", "WARNING")
    
    return stats


def verify_segments_consistency(data_path: str) -> Dict:
    """
    Verifica consistencia entre segmentos, logs y registro de videos.
    
    Compara:
    - Directorios en segments/ y normalized/
    - Logs en logs/
    - Registro en processed_videos.json
    
    Args:
        data_path: Ruta al directorio de datos
    
    Returns:
        Diccionario con estadísticas de consistencia
    """
    stats = {
        'orphan_segments': [],      # Segmentos sin log correspondiente
        'orphan_normalized': [],    # Normalized sin log correspondiente
        'incomplete_processing': [], # Directorio existe pero log indica fallo
        'missing_segments': [],     # Log existe pero faltan segmentos
        'consistent': [],           # Procesamiento consistente
    }
    
    log("Verificando consistencia de segmentos...", "INFO")
    
    # 1. Obtener directorios de segmentos
    segments_base = os.path.join(data_path, 'segments')
    normalized_base = os.path.join(data_path, 'normalized')
    logs_base = os.path.join(data_path, 'logs')
    metadata_base = os.path.join(data_path, 'metadata')
    
    segment_dirs = set()
    normalized_dirs = set()
    log_files = set()
    metadata_files = set()
    
    if os.path.exists(segments_base):
        segment_dirs = {d.name for d in Path(segments_base).iterdir() if d.is_dir()}
    
    if os.path.exists(normalized_base):
        normalized_dirs = {d.name for d in Path(normalized_base).iterdir() if d.is_dir()}
    
    if os.path.exists(logs_base):
        log_files = {f.stem for f in Path(logs_base).iterdir() if f.suffix == '.log'}
    
    if os.path.exists(metadata_base):
        metadata_files = {f.stem for f in Path(metadata_base).iterdir() if f.suffix == '.json'}
    
    # 2. Leer registro de videos procesados
    registry_path = os.path.join(data_path, 'processed_videos.json')
    processed_ids = set()
    failed_ids = set()
    
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        # Extraer podcast_ids de los procesados
        for video_id, info in registry.get('processed', {}).items():
            # El podcast_id puede estar en el log
            if os.path.exists(logs_base):
                log_path = os.path.join(logs_base, f"{video_id}.log")
                # Buscar el podcast_id en metadata o logs
                for log_file in log_files:
                    processed_ids.add(log_file)
        
        for video_id in registry.get('failed', {}):
            failed_ids.add(video_id)
    
    # 3. Verificar segmentos huérfanos (directorios sin log/metadata correspondiente)
    all_segment_dirs = segment_dirs | normalized_dirs
    
    for podcast_id in segment_dirs:
        if podcast_id not in log_files and podcast_id not in metadata_files:
            # Contar archivos en el directorio
            seg_path = os.path.join(segments_base, podcast_id)
            file_count = sum(1 for _ in Path(seg_path).rglob('*.wav'))
            size = sum(f.stat().st_size for f in Path(seg_path).rglob('*.wav'))
            
            stats['orphan_segments'].append({
                'podcast_id': podcast_id,
                'files': file_count,
                'size_mb': size / 1024 / 1024,
                'path': seg_path
            })
    
    for podcast_id in normalized_dirs:
        if podcast_id not in log_files and podcast_id not in metadata_files:
            norm_path = os.path.join(normalized_base, podcast_id)
            file_count = sum(1 for _ in Path(norm_path).rglob('*.wav'))
            size = sum(f.stat().st_size for f in Path(norm_path).rglob('*.wav'))
            
            stats['orphan_normalized'].append({
                'podcast_id': podcast_id,
                'files': file_count,
                'size_mb': size / 1024 / 1024,
                'path': norm_path
            })
    
    # 4. Verificar procesamiento incompleto (log existe pero indica fallo o está incompleto)
    for podcast_id in log_files:
        log_path = os.path.join(logs_base, f"{podcast_id}.log")
        
        try:
            with open(log_path, 'r') as f:
                log_data = json.load(f)
            
            # Verificar si el log indica procesamiento exitoso
            segments_metadata = log_data.get('segments', {}).get('metadata_after_review', 0)
            
            if segments_metadata == 0:
                # Procesamiento incompleto
                has_segments = podcast_id in segment_dirs
                has_normalized = podcast_id in normalized_dirs
                
                stats['incomplete_processing'].append({
                    'podcast_id': podcast_id,
                    'has_segments': has_segments,
                    'has_normalized': has_normalized,
                    'log_path': log_path
                })
            else:
                # Verificar que los segmentos existan
                if podcast_id not in normalized_dirs:
                    stats['missing_segments'].append({
                        'podcast_id': podcast_id,
                        'expected_segments': segments_metadata,
                        'log_path': log_path
                    })
                else:
                    # Todo consistente
                    norm_path = os.path.join(normalized_base, podcast_id)
                    file_count = sum(1 for _ in Path(norm_path).rglob('*.wav'))
                    
                    stats['consistent'].append({
                        'podcast_id': podcast_id,
                        'segments': file_count,
                        'expected': segments_metadata
                    })
        
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            # Log corrupto o incompleto
            stats['incomplete_processing'].append({
                'podcast_id': podcast_id,
                'has_segments': podcast_id in segment_dirs,
                'has_normalized': podcast_id in normalized_dirs,
                'log_path': log_path,
                'error': 'Log corrupto o incompleto'
            })
    
    return stats


def show_pipeline_status(data_path: str, verify_consistency: bool = False):
    """
    Muestra el estado actual del pipeline.
    
    Args:
        data_path: Ruta al directorio de datos
        verify_consistency: Si True, verifica consistencia entre segmentos y logs
    """
    log("Estado del Pipeline", "INFO")
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    
    # 1. Leer registro de videos procesados
    registry_path = os.path.join(data_path, 'processed_videos.json')
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        processed = registry.get('processed', {})
        failed = registry.get('failed', {})
        
        print(f"{Colors.GREEN}✓ Videos procesados: {len(processed)}{Colors.RESET}")
        print(f"{Colors.RED}✗ Videos fallidos: {len(failed)}{Colors.RESET}")
        
        # Último video procesado
        if processed:
            last_video = max(processed.items(), key=lambda x: x[1].get('timestamp', ''))
            print(f"\n{Colors.BOLD}Último video exitoso:{Colors.RESET}")
            print(f"   Título: {last_video[1].get('title', 'N/A')[:60]}")
            print(f"   Fecha: {last_video[1].get('timestamp', 'N/A')}")
        
        # Último fallo
        if failed:
            last_fail = max(failed.items(), key=lambda x: x[1].get('timestamp', ''))
            print(f"\n{Colors.BOLD}Último fallo:{Colors.RESET}")
            print(f"   Título: {last_fail[1].get('title', 'N/A')[:60]}")
            print(f"   Error: {last_fail[1].get('error', 'N/A')}")
            print(f"   Etapa: {last_fail[1].get('stage', 'N/A')}")
            print(f"   Fecha: {last_fail[1].get('timestamp', 'N/A')}")
    else:
        print(f"{Colors.YELLOW}⚠️  No se encontró registro de procesamiento{Colors.RESET}")
    
    # 2. Verificar directorios
    print(f"\n{Colors.BOLD}Directorios:{Colors.RESET}")
    
    dirs_to_check = {
        'input': 'Audios descargados',
        'segments': 'Segmentos temporales',
        'normalized': 'Segmentos normalizados',
        'metadata': 'Archivos de metadata',
        'logs': 'Logs de procesamiento',
    }
    
    for dir_name, description in dirs_to_check.items():
        dir_path = os.path.join(data_path, dir_name)
        if os.path.exists(dir_path):
            # Contar archivos
            file_count = sum(1 for _ in Path(dir_path).rglob('*') if _.is_file())
            dir_size = sum(f.stat().st_size for f in Path(dir_path).rglob('*') if f.is_file())
            
            print(f"   {description:30s}: {file_count:5d} archivos ({dir_size/1024/1024:.1f} MB)")
        else:
            print(f"   {description:30s}: {Colors.YELLOW}No existe{Colors.RESET}")
    
    # 3. Verificar caché LLM
    cache_path = os.path.join(data_path, 'llm_cache.json')
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            cache_size = len(cache_data.get('entries', {}))
            print(f"\n{Colors.BOLD}Caché LLM:{Colors.RESET}")
            print(f"   Entradas: {cache_size}")
            print(f"   Tamaño: {os.path.getsize(cache_path)/1024/1024:.1f} MB")
        except:
            print(f"\n{Colors.YELLOW}⚠️  Caché LLM corrupto{Colors.RESET}")
    
    # 4. Verificar consistencia si se solicitó
    if verify_consistency:
        print(f"\n{Colors.BOLD}Verificación de Consistencia:{Colors.RESET}")
        consistency_stats = verify_segments_consistency(data_path)
        
        # Mostrar segmentos huérfanos
        if consistency_stats['orphan_segments']:
            print(f"\n{Colors.YELLOW}⚠️  Segmentos temporales huérfanos (sin log):{Colors.RESET}")
            total_size = 0
            for item in consistency_stats['orphan_segments'][:5]:  # Mostrar solo primeros 5
                print(f"   • {item['podcast_id']}: {item['files']} archivos ({item['size_mb']:.1f} MB)")
                total_size += item['size_mb']
            if len(consistency_stats['orphan_segments']) > 5:
                print(f"   ... y {len(consistency_stats['orphan_segments']) - 5} más")
            print(f"   Total a limpiar: {total_size:.1f} MB")
        
        if consistency_stats['orphan_normalized']:
            print(f"\n{Colors.YELLOW}⚠️  Segmentos normalizados huérfanos (sin log):{Colors.RESET}")
            total_size = 0
            for item in consistency_stats['orphan_normalized'][:5]:
                print(f"   • {item['podcast_id']}: {item['files']} archivos ({item['size_mb']:.1f} MB)")
                total_size += item['size_mb']
            if len(consistency_stats['orphan_normalized']) > 5:
                print(f"   ... y {len(consistency_stats['orphan_normalized']) - 5} más")
            print(f"   Total a limpiar: {total_size:.1f} MB")
        
        # Mostrar procesamiento incompleto
        if consistency_stats['incomplete_processing']:
            print(f"\n{Colors.RED}⚠️  Procesamiento incompleto:{Colors.RESET}")
            for item in consistency_stats['incomplete_processing'][:5]:
                status = []
                if item['has_segments']:
                    status.append("segments✓")
                if item['has_normalized']:
                    status.append("normalized✓")
                status_str = ", ".join(status) if status else "sin archivos"
                error = f" ({item.get('error', 'sin metadata')})" if 'error' in item else ""
                print(f"   • {item['podcast_id']}: {status_str}{error}")
            if len(consistency_stats['incomplete_processing']) > 5:
                print(f"   ... y {len(consistency_stats['incomplete_processing']) - 5} más")
        
        # Mostrar resumen
        print(f"\n{Colors.BOLD}Resumen de Consistencia:{Colors.RESET}")
        print(f"   ✓ Procesados correctamente: {len(consistency_stats['consistent'])}")
        print(f"   ⚠️  Segmentos huérfanos: {len(consistency_stats['orphan_segments']) + len(consistency_stats['orphan_normalized'])}")
        print(f"   ⚠️  Procesamiento incompleto: {len(consistency_stats['incomplete_processing'])}")
        print(f"   ⚠️  Segmentos faltantes: {len(consistency_stats['missing_segments'])}")
        
        # Sugerencias de limpieza
        if consistency_stats['orphan_segments'] or consistency_stats['orphan_normalized']:
            print(f"\n{Colors.CYAN}💡 Sugerencia:{Colors.RESET} Ejecutar con --clean-orphans para eliminar segmentos huérfanos")
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}\n")




def clean_orphan_segments(data_path: str, dry_run: bool = False) -> Dict:
    """
    Limpia segmentos huérfanos identificados por la verificación de consistencia.
    
    Args:
        data_path: Ruta al directorio de datos
        dry_run: Si True, solo muestra qué se limpiaría sin hacerlo
    
    Returns:
        Diccionario con estadísticas de limpieza
    """
    stats = {
        'segments_cleaned': 0,
        'normalized_cleaned': 0,
        'space_freed': 0,
        'errors': []
    }
    
    log("Limpiando segmentos huérfanos...", "CLEAN")
    
    # Obtener lista de segmentos huérfanos
    consistency_stats = verify_segments_consistency(data_path)
    
    # Limpiar segmentos temporales
    for item in consistency_stats['orphan_segments']:
        podcast_id = item['podcast_id']
        path = item['path']
        size = item['size_mb'] * 1024 * 1024
        
        if dry_run:
            log(f"   [DRY-RUN] Eliminaría: {podcast_id} ({item['size_mb']:.1f} MB)", "WARNING")
        else:
            try:
                shutil.rmtree(path)
                stats['segments_cleaned'] += 1
                stats['space_freed'] += size
                log(f"   🗑️  Eliminado: {podcast_id} ({item['size_mb']:.1f} MB)", "CLEAN")
            except Exception as e:
                stats['errors'].append(f"Error eliminando {podcast_id}: {e}")
                log(f"   ⚠️  Error eliminando {podcast_id}: {e}", "WARNING")
    
    # Limpiar segmentos normalizados huérfanos
    for item in consistency_stats['orphan_normalized']:
        podcast_id = item['podcast_id']
        path = item['path']
        size = item['size_mb'] * 1024 * 1024
        
        if dry_run:
            log(f"   [DRY-RUN] Eliminaría normalized: {podcast_id} ({item['size_mb']:.1f} MB)", "WARNING")
        else:
            try:
                shutil.rmtree(path)
                stats['normalized_cleaned'] += 1
                stats['space_freed'] += size
                log(f"   🗑️  Eliminado normalized: {podcast_id} ({item['size_mb']:.1f} MB)", "CLEAN")
            except Exception as e:
                stats['errors'].append(f"Error eliminando normalized {podcast_id}: {e}")
                log(f"   ⚠️  Error eliminando {podcast_id}: {e}", "WARNING")
    
    return stats


def recover_pipeline(data_path: str) -> Optional[str]:
    """
    Intenta recuperar el procesamiento desde el último estado válido.
    
    Args:
        data_path: Ruta al directorio de datos
    
    Returns:
        Comando sugerido para continuar el procesamiento, o None
    """
    log("Analizando opciones de recuperación...", "INFO")
    
    registry_path = os.path.join(data_path, 'processed_videos.json')
    if not os.path.exists(registry_path):
        log("   No se encontró registro de videos procesados", "ERROR")
        return None
    
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    failed = registry.get('failed', {})
    processing_failures = {k: v for k, v in failed.items() if v.get('stage') == 'processing'}
    
    if not processing_failures:
        log("   No hay videos con fallas de procesamiento para recuperar", "INFO")
        return None
    
    log(f"   Encontrados {len(processing_failures)} videos con fallas de procesamiento", "SUCCESS")
    
    # Sugerir comando de recuperación
    print(f"\n{Colors.BOLD}Opciones de recuperación:{Colors.RESET}\n")
    print(f"1. Reintentar videos fallidos:")
    print(f"   {Colors.CYAN}python scripts/auto_pipeline.py --retry-failed{Colors.RESET}")
    
    print(f"\n2. Continuar con nuevos videos:")
    print(f"   {Colors.CYAN}python scripts/auto_pipeline.py{Colors.RESET}")
    
    print(f"\n3. Procesar solo los audios ya descargados:")
    print(f"   {Colors.CYAN}python scripts/auto_pipeline.py --process-only{Colors.RESET}")
    
    return "python scripts/auto_pipeline.py --retry-failed"


def main():
    parser = argparse.ArgumentParser(
        description='Limpia y recupera el pipeline cuando se traba',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f'Ruta al directorio de datos (default: {DEFAULT_DATA_PATH})'
    )
    
    parser.add_argument(
        '--deep',
        action='store_true',
        help='Limpieza profunda (incluye videos/audios parciales)'
    )
    
    parser.add_argument(
        '--kill-processes',
        action='store_true',
        help='Matar procesos colgados (ffmpeg, whisper, etc.)'
    )
    
    parser.add_argument(
        '--show-status',
        action='store_true',
        help='Mostrar estado del pipeline sin limpiar'
    )
    
    parser.add_argument(
        '--recover',
        action='store_true',
        help='Sugerir comandos para recuperar el procesamiento'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verificar consistencia entre segmentos, logs y metadatos'
    )
    
    parser.add_argument(
        '--clean-orphans',
        action='store_true',
        help='Limpiar segmentos huérfanos (sin log/metadata correspondiente)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Modo dry-run: mostrar qué se haría sin hacerlo'
    )
    
    args = parser.parse_args()
    
    # Header
    print()
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}  🧹 fromPodtoCast - Pipeline Cleanup{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*60}{Colors.RESET}")
    print(f"  📁 Directorio: {args.data_path}")
    print(f"  ⏱️  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{Colors.MAGENTA}{'─'*60}{Colors.RESET}\n")
    
    # Solo mostrar estado (con verificación si se solicitó)
    if args.show_status:
        show_pipeline_status(args.data_path, verify_consistency=args.verify)
        return
    
    # Solo verificar
    if args.verify:
        show_pipeline_status(args.data_path, verify_consistency=True)
        return
    
    # Solo recuperar
    if args.recover:
        show_pipeline_status(args.data_path, verify_consistency=True)
        recover_pipeline(args.data_path)
        return
    
    # Solo limpiar huérfanos
    if args.clean_orphans:
        orphan_stats = clean_orphan_segments(args.data_path, dry_run=args.dry_run)
        print()
        log(f"Directorios de segmentos limpiados: {orphan_stats['segments_cleaned']}", "SUCCESS")
        log(f"Directorios de normalized limpiados: {orphan_stats['normalized_cleaned']}", "SUCCESS")
        log(f"Espacio liberado: {orphan_stats['space_freed']/1024/1024:.1f} MB", "SUCCESS")
        if orphan_stats['errors']:
            log(f"Errores: {len(orphan_stats['errors'])}", "WARNING")
        print()
        show_pipeline_status(args.data_path, verify_consistency=True)
        return
    
    # Limpieza completa
    total_cleaned = 0
    
    # 1. Limpiar archivos temporales
    temp_stats = cleanup_temp_files(args.data_path, deep=args.deep)
    total_cleaned += temp_stats['temp_size'] + temp_stats['partial_size']
    
    print()
    log(f"Archivos temporales eliminados: {temp_stats['temp_files']}", "SUCCESS")
    log(f"Descargas parciales eliminadas: {temp_stats['partial_downloads']}", "SUCCESS")
    log(f"Directorios vacíos eliminados: {temp_stats['empty_dirs']}", "SUCCESS")
    log(f"Espacio liberado: {total_cleaned/1024/1024:.1f} MB", "SUCCESS")
    print()
    
    # 2. Limpiar memoria GPU
    cleanup_gpu_memory()
    print()
    
    # 3. Matar procesos colgados (si se solicitó)
    if args.kill_processes:
        process_stats = kill_stuck_processes(dry_run=args.dry_run)
        print()
        
        total_killed = sum(process_stats.values())
        if total_killed > 0:
            log(f"Procesos terminados: {total_killed}", "SUCCESS")
            for proc_type, count in process_stats.items():
                if count > 0:
                    log(f"   {proc_type}: {count}", "INFO")
        else:
            log("No se encontraron procesos colgados", "INFO")
        print()
    
    # 4. Mostrar estado final con verificación de consistencia
    show_pipeline_status(args.data_path, verify_consistency=True)
    
    # 5. Sugerir recuperación si hay fallos
    if not args.dry_run:
        recover_pipeline(args.data_path)
    
    # Footer
    print(f"\n{Colors.BOLD}{Colors.GREEN}✓ Limpieza completada{Colors.RESET}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⚠️  Limpieza interrumpida por el usuario{Colors.RESET}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{Colors.RED}❌ Error inesperado: {e}{Colors.RESET}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
