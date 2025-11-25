#!/usr/bin/env python3
"""
Script para descargar videos desde URLs y extraer el audio automáticamente.
Soporta YouTube, Vimeo, y otras plataformas compatibles con yt-dlp.
"""
import argparse
import sys
import os
import shutil
import time
from pathlib import Path
import subprocess
import json
from datetime import datetime

# Agregar src al path si es necesario
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))


def log_status(message: str, level: str = "INFO"):
    """Registra un mensaje con timestamp y nivel."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✅",
        "WARNING": "⚠️ ",
        "ERROR": "❌",
        "DEBUG": "🔍"
    }.get(level, "ℹ️ ")
    print(f"[{timestamp}] {prefix} {message}")


def log_error(message: str, error: Exception = None):
    """Registra un error con detalles."""
    log_status(message, "ERROR")
    if error:
        log_status(f"   Detalle: {str(error)}", "DEBUG")
        if hasattr(error, 'stderr') and error.stderr:
            log_status(f"   stderr: {error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr}", "DEBUG")
        if hasattr(error, 'stdout') and error.stdout:
            log_status(f"   stdout: {error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout}", "DEBUG")


def check_ytdlp():
    """Verifica si yt-dlp está instalado."""
    log_status("Verificando yt-dlp...", "INFO")
    try:
        result = subprocess.run(
            ['yt-dlp', '--version'], 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=10
        )
        version = result.stdout.strip()
        log_status(f"yt-dlp encontrado: versión {version}", "SUCCESS")
        return True, version
    except FileNotFoundError:
        log_status("yt-dlp no está instalado o no está en PATH", "ERROR")
        return False, None
    except subprocess.TimeoutExpired:
        log_status("Timeout al verificar yt-dlp", "ERROR")
        return False, None
    except subprocess.CalledProcessError as e:
        log_error(f"Error ejecutando yt-dlp: código {e.returncode}", e)
        return False, None
    except Exception as e:
        log_error("Error inesperado verificando yt-dlp", e)
        return False, None


def check_ffmpeg():
    """Verifica si ffmpeg está instalado (necesario para conversión de audio)."""
    log_status("Verificando ffmpeg...", "INFO")
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'], 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=5
        )
        # Extraer versión de la primera línea
        version_line = result.stdout.split('\n')[0] if result.stdout else "ffmpeg disponible"
        log_status(f"ffmpeg encontrado: {version_line}", "SUCCESS")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        log_status("ffmpeg no encontrado (necesario para conversión de audio)", "WARNING")
        log_status("   Instala con: sudo apt-get install ffmpeg", "INFO")
        return False
    except Exception as e:
        log_error("Error verificando ffmpeg", e)
        return False


def check_disk_space(path: Path, min_gb: float = 1.0):
    """Verifica que haya suficiente espacio en disco."""
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024**3)
        log_status(f"Espacio disponible en disco: {free_gb:.2f} GB", "INFO")
        if free_gb < min_gb:
            log_status(f"⚠️  Advertencia: Solo {free_gb:.2f} GB disponibles (recomendado: {min_gb} GB)", "WARNING")
            return False
        return True
    except Exception as e:
        log_error("Error verificando espacio en disco", e)
        return True  # Continuar si no se puede verificar


def check_write_permissions(path: Path):
    """Verifica permisos de escritura en el directorio."""
    try:
        test_file = path / ".write_test"
        test_file.touch()
        test_file.unlink()
        log_status(f"Permisos de escritura OK en: {path}", "SUCCESS")
        return True
    except PermissionError:
        log_error(f"Sin permisos de escritura en: {path}")
        return False
    except Exception as e:
        log_error("Error verificando permisos de escritura", e)
        return False


def install_ytdlp():
    """Instala yt-dlp si no está disponible."""
    log_status("yt-dlp no está instalado. Instalando...", "INFO")
    try:
        log_status("Ejecutando: pip install yt-dlp", "DEBUG")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'yt-dlp'], 
            check=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutos máximo
        )
        log_status("yt-dlp instalado correctamente", "SUCCESS")
        if result.stdout:
            log_status(f"Salida de instalación: {result.stdout[:200]}...", "DEBUG")
        return True
    except subprocess.TimeoutExpired:
        log_error("Timeout instalando yt-dlp (más de 5 minutos)")
        return False
    except subprocess.CalledProcessError as e:
        log_error("Error instalando yt-dlp", e)
        log_status("   Instala manualmente con: pip install yt-dlp", "INFO")
        return False
    except Exception as e:
        log_error("Error inesperado instalando yt-dlp", e)
        return False


def download_video(url: str, output_dir: str, audio_only: bool = True,
                   audio_format: str = 'wav', audio_quality: str = 'best') -> dict:
    """
    Descarga un video desde una URL y extrae el audio.
    
    Args:
        url: URL del video
        output_dir: Directorio donde guardar el archivo (se crea automáticamente si no existe)
        audio_only: Si True, solo descarga el audio
        audio_format: Formato de audio (wav, mp3, m4a, etc.)
        audio_quality: Calidad de audio (best, worst, o formato específico)
    
    Returns:
        Diccionario con información del archivo descargado
    """
    log_status(f"Iniciando descarga desde: {url}", "INFO")
    
    # Verificaciones previas
    output_path = Path(output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        log_status(f"Directorio de salida: {output_path.absolute()}", "INFO")
    except Exception as e:
        log_error(f"No se pudo crear el directorio {output_path}", e)
        return {'success': False, 'url': url, 'error': f'Cannot create directory: {str(e)}'}
    
    # Verificar permisos de escritura
    if not check_write_permissions(output_path):
        return {'success': False, 'url': url, 'error': 'No write permissions'}
    
    # Verificar espacio en disco
    check_disk_space(output_path, min_gb=1.0)
    
    # Verificar ffmpeg si se necesita conversión de audio
    if audio_only and audio_format != 'webm' and audio_format != 'opus':
        if not check_ffmpeg():
            log_status("Continuando sin ffmpeg (puede fallar la conversión)", "WARNING")
    
    # Construir comando yt-dlp
    cmd = ['yt-dlp']
    
    # Agregar opciones para manejar timeouts y conexiones lentas
    cmd.extend([
        '--socket-timeout', '60',  # Timeout de socket: 60 segundos
        '--retries', '3',  # Reintentar hasta 3 veces
        '--fragment-retries', '3',  # Reintentar fragmentos
        '--file-access-retries', '3',  # Reintentar acceso a archivos
        '--extractor-retries', '3',  # Reintentar extractores
        '--no-check-certificate',  # En algunos casos ayuda con conexiones lentas
    ])
    
    if audio_only:
        cmd.extend([
            '--extract-audio',
            '--audio-format', audio_format,
            '--audio-quality', audio_quality
        ])
        log_status(f"Modo: Solo audio ({audio_format}, calidad: {audio_quality})", "INFO")
    
    # Configuración de salida
    output_template = str(output_path / '%(title)s.%(ext)s')
    cmd.extend(['-o', output_template])
    
    # Limpiar URL si tiene parámetros de playlist (solo usar el video específico)
    clean_url = url
    if '&list=' in url or '?list=' in url:
        log_status("URL contiene parámetros de playlist, extrayendo solo el video...", "INFO")
        # Extraer solo la parte del video (antes de &list= o ?list=)
        if '&list=' in url:
            clean_url = url.split('&list=')[0]
        elif '?list=' in url:
            clean_url = url.split('?list=')[0]
        log_status(f"URL limpia: {clean_url}", "DEBUG")
    
    # Obtener información del video sin descargar con retry
    info_cmd = cmd + ['--dump-json', '--no-download', clean_url]
    log_status("Obteniendo información del video...", "INFO")
    log_status(f"   URL: {clean_url}", "DEBUG")
    log_status(f"   Timeout: 120 segundos", "DEBUG")
    
    video_info = None
    max_retries = 3
    retry_delay = 5  # segundos entre reintentos
    
    for attempt in range(1, max_retries + 1):
        try:
            log_status(f"Intento {attempt}/{max_retries}...", "INFO")
            result = subprocess.run(
                info_cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                timeout=120  # Aumentado a 120 segundos
            )
        
            if not result.stdout:
                log_error("No se recibió información del video")
                if attempt < max_retries:
                    log_status(f"Reintentando en {retry_delay} segundos...", "WARNING")
                    time.sleep(retry_delay)
                    continue
                return {'success': False, 'url': url, 'error': 'Empty response from yt-dlp'}
            
            video_info = json.loads(result.stdout)
            log_status(f"Información obtenida exitosamente en intento {attempt}", "SUCCESS")
            break  # Salir del loop si fue exitoso
        
        except subprocess.TimeoutExpired:
            log_error(f"Timeout obteniendo información del video (más de 120 segundos) - Intento {attempt}/{max_retries}")
            if attempt < max_retries:
                log_status(f"Reintentando en {retry_delay} segundos...", "WARNING")
                time.sleep(retry_delay)
                continue
            return {'success': False, 'url': url, 'error': 'Timeout getting video info after retries'}
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            log_error(f"Error obteniendo información del video - Intento {attempt}/{max_retries}", e)
            log_status(f"   Comando: {' '.join(info_cmd)}", "DEBUG")
            
            # Algunos errores no deben reintentarse
            if "Private video" in error_msg or "Video unavailable" in error_msg:
                return {'success': False, 'url': url, 'error': 'Video unavailable or private'}
            elif "HTTP Error 403" in error_msg or "HTTP Error 404" in error_msg:
                return {'success': False, 'url': url, 'error': 'HTTP Error: Video not accessible'}
            elif "Sign in to confirm your age" in error_msg:
                return {'success': False, 'url': url, 'error': 'Video requires age verification'}
            
            # Reintentar para otros errores
            if attempt < max_retries:
                log_status(f"Reintentando en {retry_delay} segundos...", "WARNING")
                time.sleep(retry_delay)
                continue
            return {'success': False, 'url': url, 'error': f'Failed to get video info after {max_retries} attempts: {error_msg[:200]}'}
        except json.JSONDecodeError as e:
            log_error(f"Error parseando información del video (JSON inválido) - Intento {attempt}/{max_retries}", e)
            log_status(f"   Respuesta recibida: {result.stdout[:200] if 'result' in locals() and result.stdout else 'None'}...", "DEBUG")
            if attempt < max_retries:
                log_status(f"Reintentando en {retry_delay} segundos...", "WARNING")
                time.sleep(retry_delay)
                continue
            return {'success': False, 'url': url, 'error': 'Invalid JSON response from yt-dlp'}
        except Exception as e:
            log_error(f"Error inesperado obteniendo información del video - Intento {attempt}/{max_retries}", e)
            if attempt < max_retries:
                log_status(f"Reintentando en {retry_delay} segundos...", "WARNING")
                time.sleep(retry_delay)
                continue
            return {'success': False, 'url': url, 'error': f'Unexpected error: {str(e)}'}
    
    if video_info is None:
        log_error("No se pudo obtener información del video después de todos los reintentos")
        return {'success': False, 'url': url, 'error': 'Failed to get video info after all retries'}
    
    # Mostrar información del video obtenido
    log_status("Video encontrado:", "SUCCESS")
    log_status(f"   Título: {video_info.get('title', 'N/A')}", "INFO")
    duration = video_info.get('duration', 0)
    if duration:
        log_status(f"   Duración: {duration / 60:.2f} minutos ({duration:.0f} segundos)", "INFO")
    log_status(f"   Canal: {video_info.get('uploader', 'N/A')}", "INFO")
    log_status(f"   ID: {video_info.get('id', 'N/A')}", "INFO")
    
    # Verificar si el video está disponible
    availability = video_info.get('availability', 'unknown')
    if availability != 'public':
        log_status(f"   Disponibilidad: {availability}", "WARNING")
    
    # Obtener tamaño estimado si está disponible
    filesize = video_info.get('filesize') or video_info.get('filesize_approx')
    if filesize:
        size_mb = filesize / (1024 * 1024)
        log_status(f"   Tamaño estimado: {size_mb:.2f} MB", "INFO")
    
    # Descargar
    log_status("Iniciando descarga...", "INFO")
    download_cmd = cmd + [clean_url]  # Usar URL limpia
    
    # Guardar timestamp antes de la descarga
    files_before = set(output_path.glob("*"))
    start_time = time.time()
    
    try:
        # Ejecutar descarga con timeout (2 horas para videos muy largos)
        process = subprocess.Popen(
            download_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Monitorear progreso (simplificado)
        log_status("Descarga en progreso... (esto puede tardar varios minutos)", "INFO")
        
        stdout, stderr = process.communicate(timeout=7200)  # 2 horas máximo
        
        if process.returncode != 0:
            log_error(f"Error en descarga (código {process.returncode})")
            if stderr:
                log_status(f"   Error: {stderr[:500]}", "DEBUG")
            return {'success': False, 'url': url, 'error': f'Download failed: {stderr[:200] if stderr else "Unknown error"}'}
        
        elapsed_time = time.time() - start_time
        log_status(f"Descarga completada en {elapsed_time:.1f} segundos", "SUCCESS")
        
    except subprocess.TimeoutExpired:
        process.kill()
        log_error("Timeout en descarga (más de 2 horas)")
        return {'success': False, 'url': url, 'error': 'Download timeout (2 hours)'}
    except Exception as e:
        log_error("Error inesperado durante la descarga", e)
        return {'success': False, 'url': url, 'error': f'Download error: {str(e)}'}
    
    # Encontrar el archivo descargado
    log_status("Buscando archivo descargado...", "INFO")
    files_after = set(output_path.glob("*"))
    new_files = files_after - files_before
    
    downloaded_file = None
    title_safe = video_info.get('title', 'video')
    
    # Buscar por nombre del archivo esperado
    if new_files:
        # Ordenar por tiempo de modificación (más reciente primero)
        new_files_sorted = sorted(new_files, key=lambda p: p.stat().st_mtime, reverse=True)
        downloaded_file = str(new_files_sorted[0])
        log_status(f"Archivo nuevo encontrado: {Path(downloaded_file).name}", "SUCCESS")
    else:
        # Buscar archivos con el título (yt-dlp puede haber limpiado caracteres)
        log_status("Buscando archivo por título...", "DEBUG")
        for ext in [audio_format, 'mp3', 'm4a', 'webm', 'opus', 'ogg', 'wav']:
            pattern = f"*{title_safe[:30]}*.{ext}"
            files = list(output_path.glob(pattern))
            if files:
                downloaded_file = str(files[0])
                log_status(f"Archivo encontrado por patrón: {Path(downloaded_file).name}", "SUCCESS")
                break
        
        if not downloaded_file:
            # Buscar el archivo más reciente en el directorio
            log_status("Buscando archivo más reciente...", "DEBUG")
            for ext in [audio_format, 'mp3', 'm4a', 'webm', 'opus', 'ogg', 'wav']:
                files = list(output_path.glob(f"*.{ext}"))
                if files:
                    downloaded_file = str(max(files, key=lambda p: p.stat().st_mtime))
                    log_status(f"Archivo más reciente encontrado: {Path(downloaded_file).name}", "SUCCESS")
                    break
    
    if downloaded_file and os.path.exists(downloaded_file):
        try:
            file_size = os.path.getsize(downloaded_file) / (1024 * 1024)  # MB
            log_status(f"Descarga completada exitosamente", "SUCCESS")
            log_status(f"   Archivo: {Path(downloaded_file).name}", "INFO")
            log_status(f"   Tamaño: {file_size:.2f} MB", "INFO")
            log_status(f"   Ruta completa: {downloaded_file}", "INFO")
            
            return {
                'success': True,
                'url': url,
                'title': video_info.get('title', ''),
                'file_path': downloaded_file,
                'duration': duration,
                'size_mb': file_size
            }
        except Exception as e:
            log_error("Error obteniendo información del archivo descargado", e)
            return {'success': False, 'url': url, 'error': f'Error reading file: {str(e)}'}
    else:
        log_error("Archivo descargado pero no encontrado en el directorio")
        log_status(f"   Directorio: {output_path}", "DEBUG")
        log_status(f"   Archivos en directorio: {list(output_path.glob('*'))}", "DEBUG")
        return {'success': False, 'url': url, 'error': 'File not found after download'}


def download_batch(urls: list, output_dir: str, **kwargs) -> list:
    """
    Descarga múltiples videos.
    
    Args:
        urls: Lista de URLs
        output_dir: Directorio de salida
        **kwargs: Argumentos adicionales para download_video
    
    Returns:
        Lista de resultados
    """
    results = []
    total = len(urls)
    
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{total}] Procesando: {url}")
        result = download_video(url, output_dir, **kwargs)
        results.append(result)
    
    return results


def main():
    # Obtener ruta absoluta del directorio del proyecto
    project_root = Path(__file__).parent.parent
    default_output = str(project_root / 'data' / 'input')
    
    parser = argparse.ArgumentParser(
        description='Descarga videos desde URLs y extrae el audio para procesamiento'
    )
    parser.add_argument(
        'urls',
        nargs='+',
        help='URL(s) del video a descargar (YouTube, Vimeo, etc.)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=default_output,
        help=f'Directorio donde guardar los archivos (default: {default_output})'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='wav',
        choices=['wav', 'mp3', 'm4a', 'opus', 'webm'],
        help='Formato de audio de salida (default: wav)'
    )
    parser.add_argument(
        '--quality',
        type=str,
        default='best',
        help='Calidad de audio: best, worst, o formato específico (default: best)'
    )
    parser.add_argument(
        '--video',
        action='store_true',
        help='Descargar video completo en lugar de solo audio'
    )
    parser.add_argument(
        '--install-ytdlp',
        action='store_true',
        help='Instalar yt-dlp si no está disponible'
    )
    
    args = parser.parse_args()
    
    log_status("="*60, "INFO")
    log_status("Iniciando proceso de descarga", "INFO")
    log_status("="*60, "INFO")
    
    # Verificar yt-dlp
    ytdlp_available, version = check_ytdlp()
    
    if not ytdlp_available:
        if args.install_ytdlp:
            if not install_ytdlp():
                log_status("No se pudo instalar yt-dlp. Abortando.", "ERROR")
                sys.exit(1)
            # Verificar nuevamente después de la instalación
            ytdlp_available, version = check_ytdlp()
            if not ytdlp_available:
                log_status("yt-dlp aún no está disponible después de la instalación", "ERROR")
                sys.exit(1)
        else:
            log_status("yt-dlp no está instalado.", "ERROR")
            log_status("   Instálalo con: pip install yt-dlp", "INFO")
            log_status("   O usa: python3 scripts/download_video.py --install-ytdlp <url>", "INFO")
            sys.exit(1)
    else:
        log_status(f"yt-dlp {version} disponible y listo", "SUCCESS")
    
    # Verificar URLs
    log_status(f"URLs a procesar: {len(args.urls)}", "INFO")
    for i, url in enumerate(args.urls, 1):
        log_status(f"   {i}. {url}", "INFO")
    
    # Descargar videos
    log_status("Iniciando descargas...", "INFO")
    results = download_batch(
        args.urls,
        args.output,
        audio_only=not args.video,
        audio_format=args.format,
        audio_quality=args.quality
    )
    
    # Resumen
    log_status("="*60, "INFO")
    log_status("Resumen de Descargas", "INFO")
    log_status("="*60, "INFO")
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    log_status(f"Exitosas: {len(successful)}/{len(results)}", "SUCCESS" if successful else "INFO")
    if failed:
        log_status(f"Fallidas: {len(failed)}/{len(results)}", "ERROR")
    
    if successful:
        log_status("", "INFO")
        log_status("Archivos descargados exitosamente:", "SUCCESS")
        for result in successful:
            log_status(f"   ✓ {Path(result['file_path']).name}", "SUCCESS")
            log_status(f"     Ruta: {result['file_path']}", "INFO")
            if 'size_mb' in result:
                log_status(f"     Tamaño: {result['size_mb']:.2f} MB", "INFO")
            if 'duration' in result and result['duration']:
                log_status(f"     Duración: {result['duration'] / 60:.2f} minutos", "INFO")
    
    if failed:
        log_status("", "INFO")
        log_status("Errores encontrados:", "ERROR")
        for result in failed:
            log_status(f"   ✗ {result['url']}", "ERROR")
            error_msg = result.get('error', 'Unknown error')
            log_status(f"     Error: {error_msg}", "ERROR")
    
    # Mostrar ruta absoluta del directorio de salida
    output_abs = Path(args.output).absolute()
    log_status("", "INFO")
    log_status(f"Archivos guardados en: {output_abs}", "INFO")
    log_status("", "INFO")
    log_status("Próximo paso: Procesar los archivos con:", "INFO")
    log_status(f"   python3 main.py {output_abs} -o ./data/output", "INFO")
    log_status(f"   O simplemente:", "INFO")
    log_status(f"   python3 main.py {args.output} -o ./data/output", "INFO")


if __name__ == '__main__':
    main()

