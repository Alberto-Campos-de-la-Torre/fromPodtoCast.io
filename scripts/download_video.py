#!/usr/bin/env python3
"""
Script para descargar videos desde URLs y extraer el audio automáticamente.
Soporta YouTube, Vimeo, y otras plataformas compatibles con yt-dlp.
"""
import argparse
import sys
import os
from pathlib import Path
import subprocess
import json

# Agregar src al path si es necesario
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))


def check_ytdlp():
    """Verifica si yt-dlp está instalado."""
    try:
        result = subprocess.run(['yt-dlp', '--version'], 
                               capture_output=True, text=True, check=True)
        return True, result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False, None


def install_ytdlp():
    """Instala yt-dlp si no está disponible."""
    print("📦 yt-dlp no está instalado. Instalando...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'yt-dlp'], 
                      check=True)
        print("✅ yt-dlp instalado correctamente")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error instalando yt-dlp. Instala manualmente: pip install yt-dlp")
        return False


def download_video(url: str, output_dir: str, audio_only: bool = True,
                   audio_format: str = 'wav', audio_quality: str = 'best') -> dict:
    """
    Descarga un video desde una URL y extrae el audio.
    
    Args:
        url: URL del video
        output_dir: Directorio donde guardar el archivo
        audio_only: Si True, solo descarga el audio
        audio_format: Formato de audio (wav, mp3, m4a, etc.)
        audio_quality: Calidad de audio (best, worst, o formato específico)
    
    Returns:
        Diccionario con información del archivo descargado
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Construir comando yt-dlp
    cmd = ['yt-dlp']
    
    if audio_only:
        cmd.extend([
            '--extract-audio',
            '--audio-format', audio_format,
            '--audio-quality', audio_quality
        ])
    
    # Configuración de salida
    output_template = os.path.join(output_dir, '%(title)s.%(ext)s')
    # Limpiar caracteres problemáticos del nombre
    output_template = output_template.replace(' ', '_').replace('/', '_')
    cmd.extend(['-o', output_template])
    
    # Obtener información del video sin descargar
    info_cmd = cmd + ['--dump-json', '--no-download', url]
    
    try:
        result = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
        video_info = json.loads(result.stdout)
        
        print(f"\n📹 Video encontrado:")
        print(f"   Título: {video_info.get('title', 'N/A')}")
        print(f"   Duración: {video_info.get('duration', 0) / 60:.2f} minutos")
        print(f"   Canal: {video_info.get('uploader', 'N/A')}")
        
        # Descargar
        print(f"\n⬇️  Descargando...")
        download_cmd = cmd + [url]
        subprocess.run(download_cmd, check=True)
        
        # Encontrar el archivo descargado
        title_safe = video_info.get('title', 'video').replace(' ', '_').replace('/', '_')
        downloaded_file = None
        
        for ext in [audio_format, 'mp3', 'm4a', 'webm', 'opus']:
            potential_file = os.path.join(output_dir, f"{title_safe}.{ext}")
            if os.path.exists(potential_file):
                downloaded_file = potential_file
                break
        
        if not downloaded_file:
            # Buscar el archivo más reciente en el directorio
            files = list(Path(output_dir).glob(f"*.{audio_format}"))
            if not files:
                files = list(Path(output_dir).glob("*.*"))
            if files:
                downloaded_file = str(max(files, key=os.path.getctime))
        
        if downloaded_file:
            file_size = os.path.getsize(downloaded_file) / (1024 * 1024)  # MB
            print(f"✅ Descarga completada: {Path(downloaded_file).name}")
            print(f"   Tamaño: {file_size:.2f} MB")
            
            return {
                'success': True,
                'url': url,
                'title': video_info.get('title', ''),
                'file_path': downloaded_file,
                'duration': video_info.get('duration', 0),
                'size_mb': file_size
            }
        else:
            print("⚠️  Archivo descargado pero no encontrado")
            return {'success': False, 'url': url, 'error': 'File not found'}
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error descargando {url}: {e.stderr}")
        return {'success': False, 'url': url, 'error': str(e)}
    except json.JSONDecodeError:
        print(f"❌ Error obteniendo información del video")
        return {'success': False, 'url': url, 'error': 'Failed to get video info'}


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
        default='./data/input',
        help='Directorio donde guardar los archivos (default: ./data/input)'
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
    
    # Verificar yt-dlp
    ytdlp_available, version = check_ytdlp()
    
    if not ytdlp_available:
        if args.install_ytdlp:
            if not install_ytdlp():
                sys.exit(1)
        else:
            print("❌ yt-dlp no está instalado.")
            print("   Instálalo con: pip install yt-dlp")
            print("   O usa: python3 scripts/download_video.py --install-ytdlp <url>")
            sys.exit(1)
    else:
        print(f"✅ yt-dlp {version} disponible")
    
    # Descargar videos
    results = download_batch(
        args.urls,
        args.output,
        audio_only=not args.video,
        audio_format=args.format,
        audio_quality=args.quality
    )
    
    # Resumen
    print("\n" + "="*60)
    print("📊 Resumen de Descargas")
    print("="*60)
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"✅ Exitosas: {len(successful)}/{len(results)}")
    print(f"❌ Fallidas: {len(failed)}/{len(results)}")
    
    if successful:
        print("\n📁 Archivos descargados:")
        for result in successful:
            print(f"   - {Path(result['file_path']).name}")
            print(f"     {result['file_path']}")
    
    if failed:
        print("\n⚠️  Errores:")
        for result in failed:
            print(f"   - {result['url']}: {result.get('error', 'Unknown error')}")
    
    print("\n💡 Próximo paso: Procesar los archivos con:")
    print(f"   python3 main.py {args.output} -o ./data/output")


if __name__ == '__main__':
    main()

