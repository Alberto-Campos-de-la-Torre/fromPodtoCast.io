#!/usr/bin/env python3
"""
Auto Pipeline - Sistema automático de búsqueda, descarga y procesamiento de podcasts.

Flujo:
1. Busca videos en YouTube según categorías configuradas
2. Filtra por duración, idioma y criterios de exclusión
3. Descarga el audio de videos no procesados anteriormente
4. Procesa cada audio con el pipeline completo (diarización, transcripción, etc.)
5. Registra videos procesados para evitar duplicados

Uso:
    python scripts/auto_pipeline.py                    # Ejecutar con config por defecto
    python scripts/auto_pipeline.py --dry-run          # Solo mostrar qué se descargaría
    python scripts/auto_pipeline.py --category podcasts_negocios  # Solo una categoría
    python scripts/auto_pipeline.py --limit 3          # Máximo 3 videos por categoría
"""
import argparse
import json
import os
import sys
import subprocess
import hashlib
import re
import select
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time

# Para barras de progreso
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

# Para gráficas
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Configuración de paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Ruta base para datos
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
        "SEARCH": ("🔍", Colors.BLUE),
        "DOWNLOAD": ("📥", Colors.MAGENTA),
        "PROCESS": ("⚙️ ", Colors.CYAN),
        "SKIP": ("⏭️ ", Colors.YELLOW),
    }
    icon, color = icons.get(level, ("", ""))
    print(f"{color}[{timestamp}] {icon} {message}{Colors.RESET}")


def load_config(config_path: str) -> Dict:
    """Carga configuración de búsqueda."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_processed_registry(registry_path: str) -> Dict:
    """Carga registro de videos ya procesados."""
    if os.path.exists(registry_path):
        with open(registry_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"processed": {}, "failed": {}, "skipped": {}}


def save_processed_registry(registry: Dict, registry_path: str):
    """Guarda registro de videos procesados."""
    Path(registry_path).parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def get_video_id(url: str) -> Optional[str]:
    """Extrae el ID del video de una URL de YouTube."""
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:embed/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def search_youtube(query: str, max_results: int = 5, 
                   min_duration: int = 600, max_duration: int = 10800,
                   upload_date: str = None) -> List[Dict]:
    """
    Busca videos en YouTube usando yt-dlp.
    
    Args:
        query: Término de búsqueda
        max_results: Número máximo de resultados
        min_duration: Duración mínima en segundos
        max_duration: Duración máxima en segundos
        upload_date: Filtro de fecha (today, week, month, year)
    
    Returns:
        Lista de diccionarios con información de videos
    """
    log(f"Buscando: '{query}'", "SEARCH")
    
    # Construir comando yt-dlp para búsqueda (flat-playlist ya incluye duración)
    cmd = [
        'yt-dlp',
        f'ytsearch{max_results * 4}:{query}',  # Buscar más para compensar unavailable
        '--dump-json',
        '--flat-playlist',
        '--no-warnings',
        '--ignore-errors',
        '--cookies-from-browser', 'chrome',  # Evitar bloqueo de YouTube
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        # Parsear resultados directamente (flat-playlist ya incluye duración)
        valid_videos = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            try:
                video = json.loads(line)
                video_id = video.get('id')
                duration = video.get('duration', 0)
                availability = video.get('availability')
                live_status = video.get('live_status')
                
                if not video_id:
                    continue
                
                # Filtrar videos no disponibles o en vivo
                if availability and availability != 'public':
                    continue
                if live_status and live_status != 'not_live':
                    continue
                
                # Filtrar por duración directamente
                if duration and min_duration <= duration <= max_duration:
                    valid_videos.append({
                        'id': video_id,
                        'url': f'https://www.youtube.com/watch?v={video_id}',
                        'title': video.get('title', 'Sin título'),
                        'duration': duration,
                        'duration_string': video.get('duration_string', ''),
                        'channel': video.get('channel', 'Desconocido'),
                        'upload_date': video.get('upload_date', ''),
                        'view_count': video.get('view_count', 0),
                        'description': video.get('description', '')[:500] if video.get('description') else '',
                    })
                    
                    if len(valid_videos) >= max_results:
                        break
                        
            except json.JSONDecodeError:
                continue
        
        log(f"   Encontrados {len(valid_videos)} videos válidos", "INFO")
        return valid_videos
        
    except subprocess.TimeoutExpired:
        log(f"   Timeout buscando '{query}'", "WARNING")
        return []
    except Exception as e:
        log(f"   Error buscando: {e}", "ERROR")
        return []


def should_exclude(video: Dict, exclude_keywords: List[str], 
                   blacklist_channels: List[str]) -> Tuple[bool, str]:
    """
    Verifica si un video debe ser excluido.
    
    Returns:
        Tuple (excluir, razón)
    """
    title = video.get('title', '').lower()
    channel = video.get('channel', '').lower()
    
    # Verificar keywords de exclusión
    for keyword in exclude_keywords:
        if keyword.lower() in title:
            return True, f"Keyword excluido: '{keyword}'"
    
    # Verificar canales en blacklist
    for blocked in blacklist_channels:
        if blocked.lower() in channel:
            return True, f"Canal en blacklist: '{blocked}'"
    
    return False, ""


def convert_to_wav(input_path: str, output_path: str) -> bool:
    """Convierte un archivo de video/audio a WAV usando ffmpeg."""
    try:
        # Copiar a /tmp para evitar problemas con snap ffmpeg y NTFS
        import shutil
        tmp_input = f"/tmp/convert_input_{os.getpid()}.mp4"
        tmp_output = f"/tmp/convert_output_{os.getpid()}.wav"
        
        shutil.copy2(input_path, tmp_input)
        
        # Usar /usr/bin/ffmpeg para evitar snap
        ffmpeg_path = '/usr/bin/ffmpeg' if os.path.exists('/usr/bin/ffmpeg') else 'ffmpeg'
        
        cmd = [
            ffmpeg_path, '-y', '-i', tmp_input,
            '-vn',  # Sin video
            '-acodec', 'pcm_s16le',  # Códec WAV estándar
            '-ar', '22050',  # Sample rate
            '-ac', '1',  # Mono
            tmp_output
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0 and os.path.exists(tmp_output):
            shutil.move(tmp_output, output_path)
            os.remove(tmp_input)
            return True
        
        # Limpiar temporales
        for f in [tmp_input, tmp_output]:
            if os.path.exists(f):
                os.remove(f)
        return False
    except Exception as e:
        log(f"   Error en conversión: {e}", "ERROR")
        return False


def cleanup_corrupt_files(output_dir: str) -> int:
    """
    Elimina archivos corruptos o incompletos del directorio.
    
    Returns:
        Número de archivos eliminados
    """
    removed = 0
    patterns = ['*.part', '*.mp4', '*.webm', '*.m4a', '*.temp', '*.tmp']
    
    for pattern in patterns:
        for file_path in Path(output_dir).glob(pattern):
            try:
                file_path.unlink()
                log(f"   🗑️ Eliminado archivo corrupto: {file_path.name}", "WARNING")
                removed += 1
            except Exception as e:
                log(f"   ⚠️ No se pudo eliminar {file_path.name}: {e}", "WARNING")
    
    return removed


def _normalize_title_for_search(title: str) -> str:
    """Normaliza un título para búsqueda de archivos."""
    import unicodedata
    # Normalizar unicode y quitar caracteres especiales
    normalized = unicodedata.normalize('NFKC', title.lower())
    # Quitar caracteres especiales comunes
    for char in ['｜', '|', '#', '?', '？', ':', '：', '"', "'", '/', '\\']:
        normalized = normalized.replace(char, ' ')
    # Reducir espacios múltiples
    return ' '.join(normalized.split())[:50]


def download_audio(url: str, output_dir: str, video_title: str) -> Tuple[bool, str]:
    """
    Descarga el audio de un video usando el script download_video.py.
    
    Returns:
        Tuple (éxito, ruta_archivo o mensaje_error)
    """
    download_script = PROJECT_ROOT / 'scripts' / 'download_video.py'
    
    # Limpiar archivos corruptos antes de descargar
    cleanup_corrupt_files(output_dir)
    
    # Guardar archivos existentes antes de descargar
    existing_files = set(Path(output_dir).glob('*'))
    
    cmd = [
        sys.executable,
        str(download_script),
        url,
        '-o', output_dir,
        '--format', 'wav'
    ]
    
    log(f"📥 {video_title[:55]}...", "DOWNLOAD")
    
    try:
        # Ejecutar descarga en silencio (solo capturar errores críticos)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Leer output sin mostrarlo (suprimir errores de yt-dlp)
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                output_lines.append(line)
        
        process.wait(timeout=3600)  # 1 hora máximo
        
        # Funciones de búsqueda de archivos
        audio_extensions = ['.wav', '.mp3', '.m4a', '.opus', '.ogg']
        video_extensions = ['.mp4', '.mkv', '.webm']
        all_extensions = audio_extensions + video_extensions
        
        # 1. Buscar archivos nuevos
        new_files = set()
        for ext in ['*' + e for e in all_extensions]:
            new_files.update(Path(output_dir).glob(ext))
        new_files = new_files - existing_files
        
        # 2. Buscar audio en archivos nuevos
        for f in new_files:
            if f.suffix.lower() in audio_extensions:
                log(f"   ✅ {f.name[:60]}", "SUCCESS")
                return True, str(f)
        
        # 3. Buscar video para convertir
        for f in new_files:
            if f.suffix.lower() in video_extensions:
                wav_path = f.with_suffix('.wav')
                if convert_to_wav(str(f), str(wav_path)):
                    try:
                        f.unlink()
                    except:
                        pass
                    log(f"   ✅ {wav_path.name[:60]}", "SUCCESS")
                    return True, str(wav_path)
        
        # 4. Buscar por nombre similar al título (puede que yt-dlp falló pero el archivo existe)
        title_normalized = _normalize_title_for_search(video_title)
        for f in Path(output_dir).glob('*.wav'):
            file_normalized = _normalize_title_for_search(f.stem)
            # Si el nombre del archivo contiene parte del título
            if title_normalized[:20] in file_normalized or file_normalized[:20] in title_normalized:
                log(f"   ✅ {f.name[:60]}", "SUCCESS")
                return True, str(f)
        
        # 5. Buscar archivos recientes (últimos 10 minutos)
        for f in Path(output_dir).iterdir():
            if f.suffix.lower() in all_extensions:
                if (datetime.now().timestamp() - f.stat().st_mtime) < 600:
                    if f.suffix.lower() in video_extensions:
                        wav_path = f.with_suffix('.wav')
                        if convert_to_wav(str(f), str(wav_path)):
                            try:
                                f.unlink()
                            except:
                                pass
                            log(f"   ✅ {wav_path.name[:60]}", "SUCCESS")
                            return True, str(wav_path)
                    log(f"   ✅ {f.name[:60]}", "SUCCESS")
                    return True, str(f)
        
        # Solo si realmente no hay archivo
        log(f"   ❌ No se pudo descargar", "ERROR")
        return False, "Archivo no encontrado"
            
    except subprocess.TimeoutExpired:
        process.kill()
        log("   ❌ Timeout (1 hora)", "ERROR")
        return False, "Timeout"
    except Exception as e:
        log(f"   ❌ {str(e)[:50]}", "ERROR")
        return False, str(e)


def process_audio(audio_path: str, output_dir: str, config_path: str, 
                  show_progress: bool = True) -> Tuple[bool, str]:
    """
    Procesa un audio con el pipeline principal (main.py).
    Muestra el progreso en tiempo real incluyendo barras de tqdm.
    
    Returns:
        Tuple (éxito, mensaje)
    """
    main_script = PROJECT_ROOT / 'main.py'
    
    print()  # Línea vacía antes del procesamiento
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}Procesando: {Path(audio_path).name}{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}")
    
    cmd = [
        sys.executable,
        '-u',  # Unbuffered output para ver progreso en tiempo real
        str(main_script),
        audio_path,
        '-o', output_dir,
        '-c', config_path
    ]
    
    try:
        # Ejecutar mostrando salida en tiempo real
        if show_progress:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            output_lines = []
            num_segments = None
            
            # Leer y mostrar salida en tiempo real
            for line in iter(process.stdout.readline, ''):
                if line:
                    output_lines.append(line)
                    
                    # Mostrar línea (filtrar algunas muy verbosas)
                    line_stripped = line.strip()
                    
                    # Buscar segmentos generados
                    match = re.search(r'Segmentos generados: (\d+)', line)
                    if match:
                        num_segments = match.group(1)
                    
                    # Buscar metadata generada
                    match = re.search(r'Generados (\d+) registros de metadata', line)
                    if match:
                        num_segments = match.group(1)
                    
                    # Siempre mostrar la línea
                    print(line, end='', flush=True)
            
            process.wait()
            returncode = process.returncode
            stdout_text = ''.join(output_lines)
        else:
            # Modo silencioso
            result = subprocess.run(cmd, capture_output=True, text=True)
            returncode = result.returncode
            stdout_text = result.stdout
            
            # Buscar segmentos
            match = re.search(r'Segmentos generados: (\d+)', stdout_text)
            if match:
                num_segments = match.group(1)
            else:
                match = re.search(r'Generados (\d+) registros de metadata', stdout_text)
                num_segments = match.group(1) if match else None
        
        print(f"{Colors.CYAN}{'='*60}{Colors.RESET}")
        
        # Evaluar resultado
        if num_segments:
            print(f"{Colors.GREEN}✓ Procesado: {num_segments} segmentos generados{Colors.RESET}")
            return True, f"{num_segments} segmentos"
        
        if returncode == 0:
            print(f"{Colors.GREEN}✓ Procesado correctamente{Colors.RESET}")
            return True, "OK"
        
        print(f"{Colors.RED}✗ Error en procesamiento (código: {returncode}){Colors.RESET}")
        return False, f"Exit code: {returncode}"
            
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.RESET}")
        return False, str(e)


def run_pipeline(config: Dict, registry: Dict, registry_path: str,
                 data_path: str, category_filter: str = None,
                 limit_per_category: int = None, dry_run: bool = False,
                 process_only: bool = False, max_total_videos: int = 10) -> Dict:
    """
    Ejecuta el pipeline completo de búsqueda, descarga y procesamiento.
    
    Returns:
        Estadísticas de la ejecución
    """
    stats = {
        'searched': 0,
        'found': 0,
        'downloaded': 0,
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'errors': [],
        # Métricas de tiempo
        'start_time': time.time(),
        'total_audio_duration': 0,  # Duración total de audio descargado (segundos)
        'useful_audio_duration': 0,  # Audio útil procesado (segmentos)
        'processing_times': [],  # Lista de tiempos de procesamiento por video
        'video_details': []  # Detalles por video para la gráfica
    }
    
    search_settings = config.get('search_settings', {})
    max_results = limit_per_category or search_settings.get('max_results_per_query', 5)
    min_duration = search_settings.get('min_duration_minutes', 10) * 60
    max_duration = search_settings.get('max_duration_minutes', 180) * 60
    upload_filter = search_settings.get('upload_date_filter', 'year')
    
    channels_blacklist = config.get('channels_blacklist', [])
    
    input_dir = os.path.join(data_path, 'input')
    config_path = str(PROJECT_ROOT / 'config' / 'config.json')
    
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    
    # Header del pipeline
    print()
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'═'*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}  🎙️  fromPodtoCast - Auto Pipeline{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'═'*60}{Colors.RESET}")
    print(f"  📁 Destino: {data_path}")
    print(f"  🎯 Videos a procesar: {max_total_videos}")
    print(f"  ⏱️  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{Colors.MAGENTA}{'─'*60}{Colors.RESET}")
    
    categories = config.get('categories', [])
    
    for category in categories:
        cat_name = category.get('name', 'unknown')
        
        # Filtrar por categoría si se especificó
        if category_filter and cat_name != category_filter:
            continue
        
        if not category.get('enabled', True):
            log(f"Categoría '{cat_name}' deshabilitada, saltando...", "SKIP")
            continue
        
        log(f"\n{'='*60}", "INFO")
        log(f"Categoría: {cat_name}", "INFO")
        log(f"{'='*60}", "INFO")
        
        queries = category.get('queries', [])
        exclude_keywords = category.get('exclude_keywords', [])
        
        category_videos = []
        
        for query in queries:
            stats['searched'] += 1
            
            videos = search_youtube(
                query,
                max_results=max_results,
                min_duration=min_duration,
                max_duration=max_duration,
                upload_date=upload_filter
            )
            
            for video in videos:
                video_id = video.get('id')
                
                # Verificar si ya fue procesado
                if video_id in registry.get('processed', {}):
                    log(f"   Ya procesado: {video.get('title', '')[:50]}", "SKIP")
                    stats['skipped'] += 1
                    continue
                
                if video_id in registry.get('failed', {}):
                    log(f"   Falló anteriormente: {video.get('title', '')[:50]}", "SKIP")
                    stats['skipped'] += 1
                    continue
                
                # Verificar exclusiones
                should_skip, reason = should_exclude(
                    video, exclude_keywords, channels_blacklist
                )
                if should_skip:
                    log(f"   Excluido ({reason}): {video.get('title', '')[:50]}", "SKIP")
                    stats['skipped'] += 1
                    continue
                
                category_videos.append(video)
                stats['found'] += 1
        
        # Eliminar duplicados
        seen_ids = set()
        unique_videos = []
        for v in category_videos:
            if v['id'] not in seen_ids:
                seen_ids.add(v['id'])
                unique_videos.append(v)
        
        log(f"\nVideos a procesar en '{cat_name}': {len(unique_videos)}", "INFO")
        
        if dry_run:
            for v in unique_videos:
                log(f"   [DRY-RUN] {v.get('title', '')[:60]} ({v.get('duration_string', '')})", "INFO")
            continue
        
        # Procesar videos con barra de progreso (respetando límite total)
        remaining_quota = max_total_videos - (stats['downloaded'] + stats['processed'])
        videos_to_process = unique_videos[:remaining_quota] if remaining_quota > 0 else []
        
        if TQDM_AVAILABLE and videos_to_process:
            print(f"\n{Colors.BOLD}📊 Progreso de categoría: {cat_name}{Colors.RESET}")
            video_iterator = tqdm(
                videos_to_process,
                desc=f"   Procesando",
                unit="video",
                ncols=80,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        else:
            video_iterator = videos_to_process
        
        for idx, video in enumerate(video_iterator):
            video_id = video['id']
            title = video.get('title', 'Sin título')
            video_duration = video.get('duration', 0)
            
            # Actualizar descripción de barra de progreso
            if TQDM_AVAILABLE and hasattr(video_iterator, 'set_description'):
                video_iterator.set_description(f"   [{idx+1}/{len(videos_to_process)}] {title[:30]}")
            
            print()
            print(f"{Colors.BOLD}{'─'*60}{Colors.RESET}")
            print(f"{Colors.BOLD}📹 Video {idx+1}/{len(videos_to_process)}: {title[:55]}{Colors.RESET}")
            print(f"   📺 Canal: {video.get('channel', 'Desconocido')}")
            print(f"   ⏱️  Duración: {video.get('duration_string', 'N/A')}")
            
            video_start_time = time.time()
            
            # Paso 1: Descargar
            success, result = download_audio(video['url'], input_dir, title)
            
            if not success:
                registry.setdefault('failed', {})[video_id] = {
                    'title': title,
                    'error': result,
                    'timestamp': datetime.now().isoformat(),
                    'stage': 'download'
                }
                save_processed_registry(registry, registry_path)
                stats['failed'] += 1
                stats['errors'].append(f"Download: {title[:40]} - {result}")
                continue
            
            audio_path = result
            stats['downloaded'] += 1
            stats['total_audio_duration'] += video_duration
            
            if process_only:
                log("   Modo download-only, saltando procesamiento", "INFO")
                registry.setdefault('processed', {})[video_id] = {
                    'title': title,
                    'audio_path': audio_path,
                    'timestamp': datetime.now().isoformat(),
                    'stage': 'downloaded'
                }
                save_processed_registry(registry, registry_path)
                # Registrar métricas del video
                video_end_time = time.time()
                stats['video_details'].append({
                    'title': title[:40],
                    'audio_duration': video_duration,
                    'useful_duration': 0,
                    'processing_time': video_end_time - video_start_time,
                    'category': cat_name
                })
                continue
            
            # Paso 2: Procesar con pipeline
            success, result = process_audio(audio_path, data_path, config_path)
            
            video_end_time = time.time()
            processing_time = video_end_time - video_start_time
            
            if not success:
                registry.setdefault('failed', {})[video_id] = {
                    'title': title,
                    'audio_path': audio_path,
                    'error': result,
                    'timestamp': datetime.now().isoformat(),
                    'stage': 'processing'
                }
                save_processed_registry(registry, registry_path)
                stats['failed'] += 1
                stats['errors'].append(f"Process: {title[:40]} - {result}")
                continue
            
            # Extraer número de segmentos del resultado
            num_segments = 0
            if result and 'segmentos' in result:
                try:
                    num_segments = int(result.split()[0])
                except:
                    pass
            
            # Estimar audio útil (asumiendo ~12s por segmento promedio)
            useful_duration = num_segments * 12
            stats['useful_audio_duration'] += useful_duration
            stats['processing_times'].append(processing_time)
            
            # Registrar métricas detalladas del video
            stats['video_details'].append({
                'title': title[:40],
                'audio_duration': video_duration,
                'useful_duration': useful_duration,
                'processing_time': processing_time,
                'segments': num_segments,
                'category': cat_name
            })
            
            registry.setdefault('processed', {})[video_id] = {
                'title': title,
                'channel': video.get('channel', ''),
                'duration': video_duration,
                'audio_path': audio_path,
                'segments': result,
                'timestamp': datetime.now().isoformat(),
                'category': cat_name
            }
            save_processed_registry(registry, registry_path)
            stats['processed'] += 1
            
            # === LIMPIEZA DE MEMORIA ENTRE VIDEOS ===
            # Previene acumulación que causa congelamiento después de 4-5 videos
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
            
            # Pequeña pausa entre videos para no sobrecargar
            time.sleep(2)
    
    stats['end_time'] = time.time()
    return stats


def format_duration(seconds: float) -> str:
    """Formatea segundos a formato legible (HH:MM:SS)."""
    if seconds < 0:
        return "0:00"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def generate_report_chart(stats: Dict, output_path: str):
    """
    Genera una gráfica visual del reporte de procesamiento.
    
    Args:
        stats: Diccionario con estadísticas del procesamiento
        output_path: Ruta donde guardar la imagen
    """
    if not MATPLOTLIB_AVAILABLE:
        log("matplotlib no disponible, saltando generación de gráfica", "WARNING")
        return None
    
    video_details = stats.get('video_details', [])
    if not video_details:
        log("Sin videos procesados para generar gráfica", "WARNING")
        return None
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'ggplot')
    
    # Crear figura con subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('📊 Reporte de Procesamiento - Auto Pipeline', fontsize=16, fontweight='bold', y=0.98)
    
    # Colores
    colors = {
        'audio_total': '#3498db',
        'audio_util': '#2ecc71', 
        'processing': '#e74c3c',
        'categories': ['#9b59b6', '#f39c12', '#1abc9c', '#e91e63', '#00bcd4']
    }
    
    # === Subplot 1: Resumen general (arriba izquierda) ===
    ax1 = fig.add_subplot(2, 2, 1)
    
    total_time = stats.get('end_time', time.time()) - stats.get('start_time', time.time())
    total_audio = stats.get('total_audio_duration', 0)
    useful_audio = stats.get('useful_audio_duration', 0)
    
    metrics = ['Videos\nProcesados', 'Audio Total\n(descargado)', 'Audio Útil\n(segmentos)', 'Tiempo\nProcesamiento']
    values = [
        stats.get('processed', 0) + stats.get('downloaded', 0),
        total_audio / 60,  # En minutos
        useful_audio / 60,  # En minutos
        total_time / 60  # En minutos
    ]
    bar_colors = ['#3498db', colors['audio_total'], colors['audio_util'], colors['processing']]
    
    bars = ax1.bar(metrics, values, color=bar_colors, edgecolor='white', linewidth=2)
    ax1.set_ylabel('Cantidad / Minutos', fontsize=10)
    ax1.set_title('Resumen General', fontsize=12, fontweight='bold')
    
    # Añadir valores sobre las barras
    for bar, val, metric in zip(bars, values, metrics):
        height = bar.get_height()
        if 'Videos' in metric:
            label = f'{int(val)}'
        else:
            label = f'{val:.1f} min'
        ax1.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # === Subplot 2: Tiempo por video (arriba derecha) ===
    ax2 = fig.add_subplot(2, 2, 2)
    
    titles = [v.get('title', '')[:25] + '...' if len(v.get('title', '')) > 25 else v.get('title', '') 
              for v in video_details]
    audio_durations = [v.get('audio_duration', 0) / 60 for v in video_details]
    processing_times = [v.get('processing_time', 0) / 60 for v in video_details]
    
    x = range(len(titles))
    width = 0.35
    
    bars1 = ax2.bar([i - width/2 for i in x], audio_durations, width, 
                    label='Duración Audio', color=colors['audio_total'], alpha=0.8)
    bars2 = ax2.bar([i + width/2 for i in x], processing_times, width,
                    label='Tiempo Proceso', color=colors['processing'], alpha=0.8)
    
    ax2.set_ylabel('Minutos', fontsize=10)
    ax2.set_title('Duración vs Tiempo de Procesamiento por Video', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(titles, rotation=45, ha='right', fontsize=8)
    ax2.legend(loc='upper right')
    
    # === Subplot 3: Audio útil vs total (abajo izquierda) ===
    ax3 = fig.add_subplot(2, 2, 3)
    
    useful_durations = [v.get('useful_duration', 0) / 60 for v in video_details]
    
    x = range(len(titles))
    ax3.bar(x, audio_durations, label='Audio Total', color=colors['audio_total'], alpha=0.6)
    ax3.bar(x, useful_durations, label='Audio Útil', color=colors['audio_util'], alpha=0.9)
    
    ax3.set_ylabel('Minutos', fontsize=10)
    ax3.set_xlabel('Videos', fontsize=10)
    ax3.set_title('Audio Total vs Audio Útil (Segmentos)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(titles, rotation=45, ha='right', fontsize=8)
    ax3.legend(loc='upper right')
    
    # === Subplot 4: Estadísticas de texto (abajo derecha) ===
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # Calcular estadísticas adicionales
    efficiency = (useful_audio / total_audio * 100) if total_audio > 0 else 0
    avg_processing = sum(processing_times) / len(processing_times) if processing_times else 0
    total_segments = sum(v.get('segments', 0) for v in video_details)
    
    # Crear tabla de estadísticas
    stats_text = f"""
    ╔══════════════════════════════════════════════════╗
    ║           📈 ESTADÍSTICAS DETALLADAS             ║
    ╠══════════════════════════════════════════════════╣
    ║  Videos procesados:      {stats.get('processed', 0):>6}                  ║
    ║  Videos descargados:     {stats.get('downloaded', 0):>6}                  ║
    ║  Videos fallidos:        {stats.get('failed', 0):>6}                  ║
    ╠══════════════════════════════════════════════════╣
    ║  Audio total descargado: {format_duration(total_audio):>12}            ║
    ║  Audio útil procesado:   {format_duration(useful_audio):>12}            ║
    ║  Eficiencia:             {efficiency:>6.1f}%                 ║
    ╠══════════════════════════════════════════════════╣
    ║  Tiempo total ejecución: {format_duration(total_time):>12}            ║
    ║  Promedio por video:     {format_duration(avg_processing * 60):>12}            ║
    ║  Total segmentos:        {total_segments:>6}                  ║
    ╚══════════════════════════════════════════════════╝
    """
    
    ax4.text(0.5, 0.5, stats_text, transform=ax4.transAxes, 
             fontsize=11, fontfamily='monospace',
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='#2c3e50', alpha=0.1))
    
    # Ajustar layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Guardar
    chart_path = os.path.join(output_path, f'pipeline_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    log(f"Gráfica guardada en: {chart_path}", "SUCCESS")
    return chart_path


def retry_failed_videos(registry: Dict, registry_path: str, data_path: str, 
                        config_path: str) -> Dict:
    """
    Reprocesa videos que fallaron en la etapa de procesamiento.
    Solo procesa los que ya tienen audio descargado.
    
    Returns:
        Estadísticas de la ejecución
    """
    stats = {
        'searched': 0,
        'found': 0,
        'downloaded': 0,
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'errors': [],
        'start_time': time.time(),
        'total_audio_duration': 0,
        'useful_audio_duration': 0,
        'processing_times': [],
        'video_details': []
    }
    
    failed_videos = registry.get('failed', {})
    
    if not failed_videos:
        log("No hay videos fallidos para reprocesar", "INFO")
        stats['end_time'] = time.time()
        return stats
    
    # Filtrar videos que fallaron en procesamiento O están solo descargados (ya tienen audio)
    to_retry = []
    for video_id, info in failed_videos.items():
        stage = info.get('stage', '')
        # Incluir 'processing' (falló procesando) y 'downloaded' (solo descargado, no procesado)
        if stage in ('processing', 'downloaded') and info.get('audio_path'):
            audio_path = info.get('audio_path')
            if os.path.exists(audio_path):
                to_retry.append((video_id, info))
            else:
                log(f"Audio no encontrado: {audio_path}", "WARNING")
    
    # También buscar en 'processed' los que solo tienen stage='downloaded'
    processed_videos = registry.get('processed', {})
    for video_id, info in processed_videos.items():
        if info.get('stage') == 'downloaded' and info.get('audio_path'):
            audio_path = info.get('audio_path')
            if os.path.exists(audio_path):
                to_retry.append((video_id, info))
                log(f"Agregando video solo descargado: {info.get('title', '')[:40]}", "INFO")
    
    log(f"Videos a reprocesar: {len(to_retry)}", "INFO")
    stats['found'] = len(to_retry)
    
    for video_id, info in to_retry:
        title = info.get('title', 'Sin título')
        audio_path = info.get('audio_path')
        
        log(f"\n--- Reprocesando: {title[:60]} ---", "INFO")
        
        video_start_time = time.time()
        
        # Obtener duración del audio
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
                capture_output=True, text=True, timeout=30
            )
            video_duration = float(result.stdout.strip()) if result.returncode == 0 else 0
        except:
            video_duration = 0
        
        stats['total_audio_duration'] += video_duration
        
        # Procesar
        success, result = process_audio(audio_path, data_path, config_path)
        
        video_end_time = time.time()
        processing_time = video_end_time - video_start_time
        
        if not success:
            log(f"   ❌ Falló de nuevo: {result[:100]}", "ERROR")
            stats['failed'] += 1
            stats['errors'].append(f"Retry: {title[:40]} - {result}")
            continue
        
        # Extraer número de segmentos
        num_segments = 0
        if result and 'segmentos' in result:
            try:
                num_segments = int(result.split()[0])
            except:
                pass
        
        useful_duration = num_segments * 12
        stats['useful_audio_duration'] += useful_duration
        stats['processing_times'].append(processing_time)
        
        stats['video_details'].append({
            'title': title[:40],
            'audio_duration': video_duration,
            'useful_duration': useful_duration,
            'processing_time': processing_time,
            'segments': num_segments,
            'category': 'retry'
        })
        
        # Mover de failed a processed
        registry.setdefault('processed', {})[video_id] = {
            'title': title,
            'audio_path': audio_path,
            'segments': result,
            'timestamp': datetime.now().isoformat(),
            'retried': True
        }
        del registry['failed'][video_id]
        save_processed_registry(registry, registry_path)
        
        stats['processed'] += 1
        log(f"   ✓ Procesado: {result}", "SUCCESS")
    
    stats['end_time'] = time.time()
    return stats


def print_summary(stats: Dict, data_path: str = None):
    """Imprime resumen de la ejecución y genera gráfica."""
    total_time = stats.get('end_time', time.time()) - stats.get('start_time', time.time())
    total_audio = stats.get('total_audio_duration', 0)
    useful_audio = stats.get('useful_audio_duration', 0)
    
    print()
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'═'*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}  📊 RESUMEN DE EJECUCIÓN{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'═'*60}{Colors.RESET}")
    
    # Estadísticas de videos
    print(f"\n{Colors.BOLD}📹 VIDEOS:{Colors.RESET}")
    print(f"   ├─ Búsquedas realizadas: {stats['searched']}")
    print(f"   ├─ Encontrados:          {stats['found']}")
    print(f"   ├─ Saltados:             {stats['skipped']}")
    print(f"   ├─ {Colors.GREEN}Descargados:          {stats['downloaded']}{Colors.RESET}")
    print(f"   ├─ {Colors.GREEN}Procesados:           {stats['processed']}{Colors.RESET}")
    print(f"   └─ {Colors.RED}Fallidos:             {stats['failed']}{Colors.RESET}")
    
    # Tiempos y eficiencia
    print(f"\n{Colors.BOLD}⏱️  TIEMPOS:{Colors.RESET}")
    print(f"   ├─ Audio descargado:     {format_duration(total_audio)}")
    print(f"   ├─ Audio útil:           {format_duration(useful_audio)}")
    print(f"   └─ Tiempo total:         {format_duration(total_time)}")
    
    if total_audio > 0:
        efficiency = (useful_audio / total_audio) * 100
        eff_color = Colors.GREEN if efficiency > 50 else (Colors.YELLOW if efficiency > 25 else Colors.RED)
        print(f"\n{Colors.BOLD}📈 EFICIENCIA:{Colors.RESET}")
        print(f"   └─ {eff_color}{efficiency:.1f}% de audio útil{Colors.RESET}")
    
    # Velocidad de procesamiento
    if stats['processed'] > 0 and total_time > 0:
        videos_per_hour = (stats['processed'] / total_time) * 3600
        print(f"\n{Colors.BOLD}🚀 RENDIMIENTO:{Colors.RESET}")
        print(f"   └─ {videos_per_hour:.1f} videos/hora")
    
    if stats['errors']:
        print(f"\n{Colors.BOLD}{Colors.YELLOW}⚠️  ERRORES:{Colors.RESET}")
        for err in stats['errors'][:5]:
            print(f"   └─ {err}")
        if len(stats['errors']) > 5:
            print(f"   └─ ... y {len(stats['errors']) - 5} errores más")
    
    print(f"\n{Colors.MAGENTA}{'═'*60}{Colors.RESET}")
    
    # Generar gráfica si hay datos y ruta
    if data_path and stats.get('video_details'):
        generate_report_chart(stats, data_path)


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline automático de búsqueda, descarga y procesamiento de podcasts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/auto_pipeline.py --videos 20         # Descargar y procesar 20 videos
  python scripts/auto_pipeline.py --dry-run           # Ver qué se descargaría sin hacerlo
  python scripts/auto_pipeline.py --category podcasts_negocios --videos 5  # 5 videos de una categoría
  python scripts/auto_pipeline.py --limit 2           # Máximo 2 videos por query
  python scripts/auto_pipeline.py --download-only --videos 10  # Solo descargar 10, sin procesar
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=str(PROJECT_ROOT / 'config' / 'search_queries.json'),
        help='Ruta al archivo de configuración de búsqueda'
    )
    parser.add_argument(
        '--data-path', '-d',
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f'Ruta base para datos (default: {DEFAULT_DATA_PATH})'
    )
    parser.add_argument(
        '--category',
        type=str,
        help='Procesar solo esta categoría'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Límite de videos por query'
    )
    parser.add_argument(
        '--videos', '-n',
        type=int,
        default=10,
        help='Número total de videos a descargar (default: 10)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Solo mostrar qué se descargaría, sin ejecutar'
    )
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Solo descargar audios, no procesar con pipeline'
    )
    parser.add_argument(
        '--reset-failed',
        action='store_true',
        help='Limpiar registro de videos fallidos (no reprocesa)'
    )
    parser.add_argument(
        '--retry-failed',
        action='store_true',
        help='Reprocesar videos que fallaron en etapa de procesamiento (ya descargados)'
    )
    
    args = parser.parse_args()
    
    # Banner
    print(f"""
{Colors.CYAN}{'='*60}
   🎙️  AUTO PIPELINE - fromPodtoCast
   Búsqueda, descarga y procesamiento automático
{'='*60}{Colors.RESET}
""")
    
    # Cargar configuración
    if not os.path.exists(args.config):
        log(f"Archivo de configuración no encontrado: {args.config}", "ERROR")
        sys.exit(1)
    
    config = load_config(args.config)
    log(f"Configuración cargada: {args.config}", "SUCCESS")
    
    # Verificar directorio de datos
    if not os.path.exists(args.data_path):
        log(f"Directorio de datos no encontrado: {args.data_path}", "ERROR")
        log("Creando directorio...", "INFO")
        Path(args.data_path).mkdir(parents=True, exist_ok=True)
    
    # Cargar registro de procesados
    registry_path = os.path.join(args.data_path, 'processed_videos.json')
    registry = load_processed_registry(registry_path)
    
    # Reset failed si se solicita
    if args.reset_failed:
        failed_count = len(registry.get('failed', {}))
        registry['failed'] = {}
        save_processed_registry(registry, registry_path)
        log(f"Limpiados {failed_count} registros de videos fallidos", "SUCCESS")
    
    processed_count = len(registry.get('processed', {}))
    failed_count = len(registry.get('failed', {}))
    log(f"Videos ya procesados: {processed_count}", "INFO")
    log(f"Videos fallidos: {failed_count}", "INFO")
    
    if args.dry_run:
        log("MODO DRY-RUN: No se descargará ni procesará nada", "WARNING")
    
    # Modo retry-failed: reprocesar videos fallidos
    if args.retry_failed:
        log("MODO RETRY-FAILED: Reprocesando videos fallidos...", "INFO")
        config_path = str(PROJECT_ROOT / 'config' / 'config.json')
        
        stats = retry_failed_videos(
            registry=registry,
            registry_path=registry_path,
            data_path=args.data_path,
            config_path=config_path
        )
        
        # Mostrar resumen y generar gráfica
        print_summary(stats, args.data_path)
        
        if stats['processed'] > 0:
            log("Retry completado exitosamente", "SUCCESS")
        else:
            log("No se reprocesaron videos", "WARNING")
        return
    
    # Ejecutar pipeline normal
    log(f"Objetivo: descargar hasta {args.videos} videos", "INFO")
    
    stats = run_pipeline(
        config=config,
        registry=registry,
        registry_path=registry_path,
        data_path=args.data_path,
        category_filter=args.category,
        limit_per_category=args.limit,
        dry_run=args.dry_run,
        process_only=args.download_only,
        max_total_videos=args.videos
    )
    
    # Mostrar resumen y generar gráfica
    print_summary(stats, args.data_path)
    
    if stats['processed'] > 0:
        log("Pipeline completado exitosamente", "SUCCESS")
    elif stats['downloaded'] > 0:
        log("Descargas completadas (sin procesamiento)", "SUCCESS")
    elif args.dry_run:
        log("Dry-run completado", "SUCCESS")
    else:
        log("No se procesaron nuevos videos", "WARNING")


if __name__ == '__main__':
    main()

