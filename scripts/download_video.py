#!/usr/bin/env python3
"""
Script para descargar videos desde URLs y extraer el audio automáticamente.
Soporta YouTube, Vimeo, y otras plataformas compatibles con yt-dlp.
"""
import argparse
import sys
import os
# Ensure deno is in PATH for yt-dlp JavaScript runtime
os.environ["PATH"] = os.path.expanduser("~/.deno/bin") + ":" + os.environ.get("PATH", "")
import shutil
import time
import re
import threading
from pathlib import Path
import subprocess
import json
from datetime import datetime
from typing import Optional, Tuple

# Agregar src al path si es necesario
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Resolver path de yt-dlp (puede no estar en PATH si se invoca sin activar venv)
YTDLP_PATH = shutil.which('yt-dlp') or str(project_root / 'venv' / 'bin' / 'yt-dlp')

# Colores ANSI para terminal
class Colors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'


def log_status(message: str, level: str = "INFO"):
    """Registra un mensaje con timestamp y nivel."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✅",
        "WARNING": "⚠️ ",
        "ERROR": "❌",
        "DEBUG": "🔍",
        "PROGRESS": "📥"
    }.get(level, "ℹ️ ")
    
    # Colores para terminal
    color = {
        "INFO": Colors.CYAN,
        "SUCCESS": Colors.GREEN,
        "WARNING": Colors.YELLOW,
        "ERROR": Colors.RED,
        "DEBUG": Colors.BLUE,
        "PROGRESS": Colors.CYAN
    }.get(level, "")
    
    print(f"{color}[{timestamp}] {prefix} {message}{Colors.RESET}")


def log_error(message: str, error: Exception = None):
    """Registra un error con detalles."""
    log_status(message, "ERROR")
    if error:
        log_status(f"   Detalle: {str(error)}", "DEBUG")
        if hasattr(error, 'stderr') and error.stderr:
            stderr_content = error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr
            # Mostrar las primeras líneas del error
            for line in stderr_content.split('\n')[:5]:
                if line.strip():
                    log_status(f"   stderr: {line.strip()}", "DEBUG")
        if hasattr(error, 'stdout') and error.stdout:
            stdout_content = error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout
            for line in stdout_content.split('\n')[:5]:
                if line.strip():
                    log_status(f"   stdout: {line.strip()}", "DEBUG")


def validate_audio_file(file_path: str) -> Tuple[bool, str, Optional[float]]:
    """
    Valida un archivo de audio descargado.
    
    Returns:
        Tuple[bool, str, Optional[float]]: (es_valido, mensaje, duracion_segundos)
    """
    if not os.path.exists(file_path):
        return False, "Archivo no existe", None
    
    file_size = os.path.getsize(file_path)
    if file_size < 1000:  # Menos de 1KB
        return False, f"Archivo demasiado pequeño ({file_size} bytes)", None
    
    # Verificar con ffprobe
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 
             'format=duration,size,bit_rate', '-show_entries',
             'stream=codec_name,sample_rate,channels',
             '-of', 'json', file_path],
            capture_output=True, text=True, timeout=30
        )
        
        if result.returncode != 0:
            return False, f"ffprobe error: {result.stderr[:100]}", None
        
        info = json.loads(result.stdout)
        
        # Extraer información
        format_info = info.get('format', {})
        duration = float(format_info.get('duration', 0))
        
        streams = info.get('streams', [])
        audio_stream = None
        for stream in streams:
            if stream.get('codec_name'):
                audio_stream = stream
                break
        
        if duration < 1:
            return False, "Duración menor a 1 segundo", duration
        
        # Verificar integridad (detectar corrupción)
        check_result = subprocess.run(
            ['ffmpeg', '-v', 'error', '-i', file_path, '-f', 'null', '-'],
            capture_output=True, text=True, timeout=120
        )
        
        if 'Invalid data' in check_result.stderr or 'corrupt' in check_result.stderr.lower():
            return False, "Audio contiene datos corruptos", duration
        
        # Audio válido
        codec = audio_stream.get('codec_name', 'unknown') if audio_stream else 'unknown'
        sample_rate = audio_stream.get('sample_rate', 'unknown') if audio_stream else 'unknown'
        channels = audio_stream.get('channels', 'unknown') if audio_stream else 'unknown'
        
        return True, f"OK (codec={codec}, sr={sample_rate}Hz, ch={channels}, dur={duration:.1f}s)", duration
        
    except subprocess.TimeoutExpired:
        return False, "Timeout validando audio", None
    except json.JSONDecodeError:
        return False, "Error parseando información de audio", None
    except Exception as e:
        return False, f"Error: {str(e)}", None


def repair_audio_file(input_path: str, output_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Intenta reparar un archivo de audio corrupto.
    
    Returns:
        Tuple[bool, str]: (exitoso, mensaje_o_ruta)
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_repaired{ext}"
    
    log_status(f"Intentando reparar audio...", "INFO")
    
    try:
        # Usar pipe stdout para preservar salida incluso con errores
        with open(output_path, 'wb') as out_file:
            process = subprocess.Popen(
                ['ffmpeg', '-y',
                 '-err_detect', 'ignore_err',
                 '-i', input_path,
                 '-vn',
                 '-ar', '16000',
                 '-ac', '1',
                 '-f', 'wav',
                 '-'],
                stdout=out_file,
                stderr=subprocess.PIPE,
                text=False
            )
            
            _, stderr = process.communicate(timeout=600)
        
        # Verificar resultado
        if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
            is_valid, msg, duration = validate_audio_file(output_path)
            if is_valid:
                log_status(f"Audio reparado exitosamente: {msg}", "SUCCESS")
                return True, output_path
            else:
                os.remove(output_path)
                return False, f"Reparación falló: {msg}"
        else:
            if os.path.exists(output_path):
                os.remove(output_path)
            return False, "No se pudo crear archivo reparado"
            
    except Exception as e:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        return False, f"Error reparando: {str(e)}"


def check_ytdlp():
    """Verifica si yt-dlp está instalado."""
    log_status("Verificando yt-dlp...", "INFO")
    try:
        result = subprocess.run(
            [YTDLP_PATH, '--version'], 
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


def check_nodejs():
    """Verifica si Node.js está instalado (recomendado para YouTube)."""
    try:
        result = subprocess.run(
            ['node', '--version'], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            log_status(f"Node.js encontrado: {result.stdout.strip()}", "SUCCESS")
            return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    except Exception:
        pass
    
    # Node.js no encontrado - solo advertir, no es crítico
    log_status("Node.js no encontrado (recomendado para mejor extracción de YouTube)", "WARNING")
    log_status("   Instala con: sudo apt-get install nodejs", "INFO")
    log_status("   Se usará cliente Android/Web como alternativa", "INFO")
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


def update_ytdlp():
    """Actualiza yt-dlp a la última versión."""
    log_status("Actualizando yt-dlp...", "INFO")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', 'yt-dlp'], 
            check=True,
            capture_output=True,
            text=True,
            timeout=300
        )
        log_status("yt-dlp actualizado correctamente", "SUCCESS")
        
        # Verificar nueva versión
        check_result = subprocess.run(
            [YTDLP_PATH, '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if check_result.returncode == 0:
            log_status(f"Nueva versión: {check_result.stdout.strip()}", "INFO")
        return True
    except Exception as e:
        log_error("Error actualizando yt-dlp", e)
        return False


def validate_existing_files(directory: str) -> list:
    """Valida todos los archivos de audio en un directorio."""
    log_status(f"Validando archivos en: {directory}", "INFO")
    
    path = Path(directory)
    if not path.exists():
        log_error(f"Directorio no existe: {directory}")
        return []
    
    results = []
    audio_extensions = ['.wav', '.mp3', '.m4a', '.opus', '.webm', '.ogg', '.flac']
    
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(path.glob(f"*{ext}"))
    
    if not audio_files:
        log_status("No se encontraron archivos de audio", "WARNING")
        return []
    
    log_status(f"Encontrados {len(audio_files)} archivos de audio", "INFO")
    
    for audio_file in sorted(audio_files):
        log_status(f"Validando: {audio_file.name}", "INFO")
        is_valid, msg, duration = validate_audio_file(str(audio_file))
        
        result = {
            'file': str(audio_file),
            'valid': is_valid,
            'message': msg,
            'duration': duration
        }
        
        if is_valid:
            log_status(f"   ✓ {msg}", "SUCCESS")
        else:
            log_status(f"   ✗ {msg}", "ERROR")
            
            # Ofrecer reparación
            log_status("   Intentando reparar...", "INFO")
            success, repair_msg = repair_audio_file(str(audio_file))
            
            if success:
                # Reemplazar con reparado
                repaired_path = repair_msg
                backup_path = str(audio_file) + ".bak"
                os.rename(str(audio_file), backup_path)
                os.rename(repaired_path, str(audio_file))
                os.remove(backup_path)
                
                result['repaired'] = True
                log_status(f"   ✓ Archivo reparado exitosamente", "SUCCESS")
            else:
                result['repair_failed'] = True
                log_status(f"   ✗ No se pudo reparar: {repair_msg}", "ERROR")
        
        results.append(result)
    
    return results


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
    cmd = [YTDLP_PATH]
    
    # Agregar opciones para manejar timeouts y conexiones lentas
    cmd.extend([
        '--socket-timeout', '60',  # Timeout de socket: 60 segundos
        '--retries', '3',  # Reintentar hasta 3 veces
        '--fragment-retries', '3',  # Reintentar fragmentos
        '--file-access-retries', '3',  # Reintentar acceso a archivos
        '--extractor-retries', '3',  # Reintentar extractores
        '--no-check-certificate',  # En algunos casos ayuda con conexiones lentas
        # Cookies disabled - YouTube flags the Chrome session as bot
        # '--cookies-from-browser', 'chrome',
        '--remote-components', 'ejs:github',
        '--no-check-formats',  # Skip format validation to avoid 'format not available' errors
        '--no-warnings',  # Suprimir warnings de formatos faltantes
        '--restrict-filenames', # ASCII filenames para evitar problemas de filesystem/ffmpeg
        '--sleep-requests', '3',  # Delay between requests to avoid rate limiting
        '--sleep-interval', '5',  # Wait before downloading
    ])
    
    if audio_only:
        # Evitar post-procesamiento de ffmpeg dentro de yt-dlp que está fallando por permisos/rutas
        # Descargar simplemente el mejor audio disponible
        cmd.extend([
            '-f', 'bestaudio/best',
            # '--extract-audio', # Deshabilitado para evitar fallo de ffmpeg interno
            # '--audio-format', audio_format,
            # '--audio-quality', audio_quality
        ])
        log_status(f"Modo: Mejor audio disponible (sin conversión forzada)", "INFO")
    
    # Configuración de salida
    import tempfile
    
    # Usar directorio temporal local para descarga y conversión
    # Esto evita problemas de permisos en unidades externas (/media) y mejora velocidad
    temp_dir_obj = tempfile.TemporaryDirectory(prefix="ytdlp_temp_")
    temp_output_path = Path(temp_dir_obj.name)
    log_status(f"Usando directorio temporal: {temp_output_path}", "DEBUG")
    
    # Incluir ID para evitar problemas con títulos duplicados o caracteres especiales
    output_template = str(temp_output_path / '%(title)s [%(id)s].%(ext)s')
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
    # Usar comando base sin opciones de audio para obtener info
    info_cmd = [
        YTDLP_PATH,
        '--socket-timeout', '60',
        '--no-check-certificate',
        # '--cookies-from-browser', 'chrome',  # Disabled - flagged by YouTube
        '--remote-components', 'ejs:github',
        '--no-check-formats',
        '--no-warnings',
        '--dump-json',
        '--no-download',
        clean_url
    ]
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
    
    # Agregar opción de progreso para yt-dlp
    download_cmd = cmd + ['--newline', '--progress', clean_url]  # Usar URL limpia con progreso
    
    # Guardar timestamp antes de la descarga
    files_before = set(output_path.glob("*"))
    start_time = time.time()
    
    try:
        # Ejecutar descarga con monitoreo de progreso en tiempo real
        process = subprocess.Popen(
            download_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combinar stdout y stderr
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        log_status("Descarga en progreso...", "PROGRESS")
        
        last_progress = ""
        download_errors = []
        
        # Leer salida línea por línea para mostrar progreso
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if not line:
                continue
            
            # Detectar progreso de descarga
            if '[download]' in line:
                # Extraer porcentaje si está disponible
                match = re.search(r'(\d+\.?\d*)%', line)
                if match:
                    progress = match.group(1)
                    if progress != last_progress:
                        # Mostrar cada 10% o cambio significativo
                        try:
                            pct = float(progress)
                            if pct % 10 < 1 or pct > 99:
                                log_status(f"   Progreso: {progress}% - {line[11:80]}...", "PROGRESS")
                        except:
                            pass
                        last_progress = progress
                elif 'Destination' in line:
                    log_status(f"   {line}", "INFO")
                elif 'has already been downloaded' in line:
                    log_status(f"   {line}", "WARNING")
            
            # Detectar extracción de audio
            elif '[ExtractAudio]' in line:
                log_status(f"   Extrayendo audio: {line[14:60]}...", "INFO")
            
            # Detectar conversión
            elif 'Converting' in line or 'ffmpeg' in line.lower():
                log_status(f"   Convirtiendo: {line[:60]}...", "INFO")
            
            # Detectar errores
            elif 'ERROR' in line or 'error' in line.lower():
                download_errors.append(line)
                log_status(f"   ⚠️ {line}", "WARNING")
            
            # Detectar advertencias importantes
            elif 'WARNING' in line:
                log_status(f"   {line}", "WARNING")
        
        process.wait(timeout=7200)  # 2 horas máximo
        
        if process.returncode != 0:
            log_error(f"Error en descarga (código {process.returncode})")
            if download_errors:
                for err in download_errors[:3]:
                    log_status(f"   {err}", "ERROR")
            return {'success': False, 'url': url, 'error': f'Download failed with code {process.returncode}'}
        
        elapsed_time = time.time() - start_time
        log_status(f"Descarga completada en {elapsed_time:.1f} segundos", "SUCCESS")
        
        if download_errors:
            log_status(f"   Advertencia: Se encontraron {len(download_errors)} errores menores durante la descarga", "WARNING")
        
    except subprocess.TimeoutExpired:
        process.kill()
        log_error("Timeout en descarga (más de 2 horas)")
        return {'success': False, 'url': url, 'error': 'Download timeout (2 hours)'}
    except Exception as e:
        log_error("Error inesperado durante la descarga", e)
        return {'success': False, 'url': url, 'error': f'Download error: {str(e)}'}
    
    # Encontrar el archivo descargado en el directorio temporal
    log_status("Buscando archivo descargado en temporal...", "INFO")
    
    # En temp dir, todo archivo es nuevo porque estaba vacío
    # Buscar por ID para mayor seguridad
    video_id = video_info.get('id')
    temp_files = []
    
    if video_id:
        # Priorizar archivo con ID
        temp_files.extend(list(temp_output_path.glob(f"*[{video_id}]*")))
        if not temp_files:
            temp_files.extend(list(temp_output_path.glob(f"*{video_id}*")))
    
    # Si no encuentra por ID, listar todo (debe ser el único archivo ahí)
    if not temp_files:
        temp_files = list(temp_output_path.glob("*"))
        
    downloaded_file = None
    
    if temp_files:
        # Ordenar por tiempo de modificación (más reciente primero)
        temp_files_sorted = sorted(temp_files, key=lambda p: p.stat().st_mtime, reverse=True)
        temp_file = str(temp_files_sorted[0])
        log_status(f"Archivo temporal encontrado: {Path(temp_file).name}", "SUCCESS")
        
        # Mover al destino final
        try:
            dest_file = output_path / Path(temp_file).name
            shutil.move(temp_file, str(dest_file))
            downloaded_file = str(dest_file)
            log_status(f"Archivo movido a destino final: {Path(downloaded_file).name}", "SUCCESS")
        except Exception as e:
            log_error(f"Error moviendo archivo desde temporal: {e}")
            return {'success': False, 'url': url, 'error': f'Failed to move file: {str(e)}'}
            
    else:
        log_error("No se encontró ningún archivo en el directorio temporal")
        return {'success': False, 'url': url, 'error': 'File not found in temp dir'}
    
    # Limpieza manual del directorio temporal (aunque el obj lo hace al final)
    try:
        temp_dir_obj.cleanup()
    except:
        pass

    
    if downloaded_file and os.path.exists(downloaded_file):
        try:
            file_size = os.path.getsize(downloaded_file) / (1024 * 1024)  # MB
            log_status(f"Archivo descargado: {Path(downloaded_file).name}", "SUCCESS")
            log_status(f"   Tamaño: {file_size:.2f} MB", "INFO")
            
            # Validar integridad del audio
            log_status("Validando integridad del audio...", "INFO")
            is_valid, validation_msg, actual_duration = validate_audio_file(downloaded_file)
            
            if is_valid:
                log_status(f"   Validación: {validation_msg}", "SUCCESS")
                
                return {
                    'success': True,
                    'url': url,
                    'title': video_info.get('title', ''),
                    'file_path': downloaded_file,
                    'duration': actual_duration or duration,
                    'size_mb': file_size,
                    'validated': True
                }
            else:
                log_status(f"   Validación falló: {validation_msg}", "WARNING")
                
                # Intentar reparar
                success, repair_result = repair_audio_file(downloaded_file)
                
                if success:
                    # Reemplazar archivo original con el reparado
                    repaired_path = repair_result
                    os.replace(repaired_path, downloaded_file)
                    
                    # Re-validar
                    is_valid, validation_msg, actual_duration = validate_audio_file(downloaded_file)
                    file_size = os.path.getsize(downloaded_file) / (1024 * 1024)
                    
                    log_status(f"   Audio reparado: {validation_msg}", "SUCCESS")
                    
                    return {
                        'success': True,
                        'url': url,
                        'title': video_info.get('title', ''),
                        'file_path': downloaded_file,
                        'duration': actual_duration or duration,
                        'size_mb': file_size,
                        'validated': True,
                        'repaired': True
                    }
                else:
                    log_status(f"   No se pudo reparar: {repair_result}", "ERROR")
                    log_status(f"   El archivo puede tener problemas pero se conserva", "WARNING")
                    
                    return {
                        'success': True,  # Descarga exitosa pero archivo con problemas
                        'url': url,
                        'title': video_info.get('title', ''),
                        'file_path': downloaded_file,
                        'duration': duration,
                        'size_mb': file_size,
                        'validated': False,
                        'validation_error': validation_msg
                    }
                    
        except Exception as e:
            log_error("Error procesando archivo descargado", e)
            return {'success': False, 'url': url, 'error': f'Error processing file: {str(e)}'}
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
    # Ruta base para datos de voz
    default_output = '/media/ttech-main/42A4266DA426639F/Base de Datos - Voz/input'
    
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
    parser.add_argument(
        '--update-ytdlp',
        action='store_true',
        help='Actualizar yt-dlp a la última versión'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Solo validar archivos existentes sin descargar'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar información detallada de depuración'
    )
    
    args = parser.parse_args()
    
    # Configurar nivel de verbosidad global
    global VERBOSE
    VERBOSE = args.verbose
    
    log_status("="*60, "INFO")
    log_status("Iniciando proceso de descarga", "INFO")
    log_status("="*60, "INFO")
    
    # Modo de solo validación
    if args.validate_only:
        log_status("Modo: Solo validación de archivos existentes", "INFO")
        results = validate_existing_files(args.output)
        
        valid_count = sum(1 for r in results if r['valid'])
        repaired_count = sum(1 for r in results if r.get('repaired'))
        failed_count = sum(1 for r in results if not r['valid'] and not r.get('repaired'))
        
        log_status("="*60, "INFO")
        log_status("Resumen de validación:", "INFO")
        log_status(f"   Válidos: {valid_count}", "SUCCESS" if valid_count else "INFO")
        if repaired_count:
            log_status(f"   Reparados: {repaired_count}", "WARNING")
        if failed_count:
            log_status(f"   Con errores: {failed_count}", "ERROR")
        log_status("="*60, "INFO")
        sys.exit(0 if failed_count == 0 else 1)
    
    # Verificar yt-dlp
    ytdlp_available, version = check_ytdlp()
    
    # Actualizar yt-dlp si se solicita
    if args.update_ytdlp and ytdlp_available:
        update_ytdlp()
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
    
    # Verificar Node.js (recomendado pero no requerido)
    check_nodejs()
    
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
            status_icon = "✓"
            if result.get('repaired'):
                status_icon = "🔧"  # Reparado
            elif not result.get('validated', True):
                status_icon = "⚠️"  # Con problemas
            
            log_status(f"   {status_icon} {Path(result['file_path']).name}", "SUCCESS")
            log_status(f"     Ruta: {result['file_path']}", "INFO")
            if 'size_mb' in result:
                log_status(f"     Tamaño: {result['size_mb']:.2f} MB", "INFO")
            if 'duration' in result and result['duration']:
                log_status(f"     Duración: {result['duration'] / 60:.2f} minutos", "INFO")
            
            # Mostrar estado de validación
            if result.get('repaired'):
                log_status(f"     Estado: Reparado automáticamente", "WARNING")
            elif result.get('validated'):
                log_status(f"     Estado: Validado ✓", "SUCCESS")
            elif 'validation_error' in result:
                log_status(f"     Estado: Requiere atención - {result['validation_error']}", "WARNING")
    
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

