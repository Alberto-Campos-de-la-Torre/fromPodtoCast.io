#!/usr/bin/env python3
"""
download_podcast_rss.py - Descargador de podcasts desde RSS feeds

Descarga episodios de podcast directamente desde RSS feeds (MP3/M4A).
Sin anti-bot, sin tokens, sin rate-limiting — solo HTTP directo.

Uso:
    python download_podcast_rss.py                          # Descargar de todos los feeds
    python download_podcast_rss.py --feed "La Cotorrisa"    # Solo un podcast
    python download_podcast_rss.py --max-episodes 20        # Limitar episodios por feed
    python download_podcast_rss.py --process                # Descargar + procesar con fromPodtoCast
"""
import json
import os
import sys
import re
import time
import argparse
import hashlib
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
FEEDS_CONFIG = PROJECT_ROOT / "config" / "podcast_feeds.json"
PROCESSED_VIDEOS_PATH = None  # Set from config
DEFAULT_OUTPUT_DIR = "/media/ttech-main/8464b39b-ba5b-49c5-9e05-010122b9874d/input"


def sanitize_filename(name: str, max_len: int = 80) -> str:
    """Convierte un nombre a un filename ASCII seguro."""
    # Remove or replace special characters
    name = re.sub(r'[^\w\s\-.]', '_', name)
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    if len(name) > max_len:
        name = name[:max_len]
    return name


def load_config():
    """Carga la configuración de feeds."""
    if not FEEDS_CONFIG.exists():
        print(f"❌ No se encontró {FEEDS_CONFIG}")
        sys.exit(1)
    with open(FEEDS_CONFIG, 'r') as f:
        return json.load(f)


def load_processed_ids(output_dir: str) -> set:
    """Carga IDs de episodios ya procesados."""
    pv_path = Path(output_dir).parent / "processed_videos.json"
    processed = set()
    if pv_path.exists():
        try:
            with open(pv_path, 'r') as f:
                data = json.load(f)
                # Handle both string list and dict list formats
                for key in ["processed", "failed"]:
                    items = data.get(key, [])
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, str):
                                processed.add(item)
                            elif isinstance(item, dict):
                                processed.add(item.get("id", ""))
                    elif isinstance(items, dict):
                        processed.update(items.keys())
        except:
            pass

    # Also check existing files in output dir
    output_path = Path(output_dir)
    if output_path.exists():
        for f in output_path.iterdir():
            if f.suffix in ('.wav', '.mp3', '.m4a', '.ogg'):
                processed.add(f.stem)

    return processed


def parse_rss_feed(feed_url: str, max_episodes: int = 50,
                   min_duration_min: int = 20, max_duration_min: int = 180) -> list:
    """
    Parsea un RSS feed y extrae información de episodios.
    
    Returns:
        Lista de dicts con: title, audio_url, duration, pub_date, episode_id
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) fromPodtoCast/1.0",
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }

    try:
        req = Request(feed_url, headers=headers)
        with urlopen(req, timeout=30) as response:
            content = response.read()
    except (URLError, HTTPError) as e:
        print(f"  ❌ Error descargando feed: {e}")
        return []

    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        print(f"  ❌ Error parseando XML: {e}")
        return []

    # RSS namespaces
    ns = {
        'itunes': 'http://www.itunes.com/dtds/podcast-1.0.dtd',
        'content': 'http://purl.org/rss/1.0/modules/content/',
        'media': 'http://search.yahoo.com/mrss/',
    }

    episodes = []
    channel = root.find('channel')
    if channel is None:
        print("  ❌ No se encontró <channel> en el RSS")
        return []

    items = channel.findall('item')

    for item in items[:max_episodes * 2]:  # Get more than needed to filter
        if len(episodes) >= max_episodes:
            break

        title = item.findtext('title', '').strip()
        if not title:
            continue

        # Get audio URL from enclosure
        enclosure = item.find('enclosure')
        audio_url = None
        if enclosure is not None:
            audio_url = enclosure.get('url', '')
            mime = enclosure.get('type', '')
            if audio_url and ('audio' in mime or audio_url.endswith(('.mp3', '.m4a', '.ogg', '.wav'))):
                pass
            elif audio_url:
                # Some feeds don't set MIME type correctly
                pass
            else:
                audio_url = None

        # Try media:content as fallback
        if not audio_url:
            media_content = item.find('media:content', ns)
            if media_content is not None:
                audio_url = media_content.get('url', '')

        if not audio_url:
            continue

        # Parse duration
        duration_str = item.findtext(f'{{{ns["itunes"]}}}duration', '')
        duration_sec = 0
        if duration_str:
            try:
                parts = duration_str.split(':')
                if len(parts) == 3:
                    duration_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                elif len(parts) == 2:
                    duration_sec = int(parts[0]) * 60 + int(parts[1])
                else:
                    duration_sec = int(parts[0])
            except ValueError:
                duration_sec = 0

        duration_min = duration_sec / 60

        # Filter by duration
        if duration_sec > 0 and (duration_min < min_duration_min or duration_min > max_duration_min):
            continue

        # Generate unique episode ID
        guid = item.findtext('guid', '')
        episode_id = hashlib.md5(
            (guid or audio_url or title).encode()
        ).hexdigest()[:12]

        pub_date = item.findtext('pubDate', '')

        episodes.append({
            'title': title,
            'audio_url': audio_url,
            'duration_sec': duration_sec,
            'duration_min': round(duration_min, 1),
            'pub_date': pub_date,
            'episode_id': episode_id,
            'guid': guid,
        })

    return episodes


def download_episode(episode: dict, output_dir: str, podcast_name: str) -> str:
    """
    Descarga un episodio de podcast via HTTP.
    
    Returns:
        Path al archivo descargado o None si falla
    """
    title = episode['title']
    audio_url = episode['audio_url']

    # Determine file extension from URL
    url_lower = audio_url.split('?')[0].lower()
    if '.m4a' in url_lower:
        ext = '.m4a'
    elif '.ogg' in url_lower:
        ext = '.ogg'
    elif '.wav' in url_lower:
        ext = '.wav'
    else:
        ext = '.mp3'

    # Create safe filename
    safe_title = sanitize_filename(f"{podcast_name}_{title}")
    filename = f"{safe_title}_{episode['episode_id']}{ext}"
    filepath = Path(output_dir) / filename

    if filepath.exists() and filepath.stat().st_size > 0:
        print(f"  ⏭️  Ya existe: {filename}")
        return str(filepath)

    # Download with wget for reliability (handles redirects, resume, etc.)
    try:
        cmd = [
            'wget', '-q', '--show-progress',
            '--timeout=60',
            '--tries=3',
            '--user-agent=Mozilla/5.0 (X11; Linux x86_64) fromPodtoCast/1.0',
            '-O', str(filepath),
            audio_url
        ]
        result = subprocess.run(cmd, timeout=600, capture_output=True, text=True)

        if result.returncode != 0:
            # Fallback to Python urllib
            print(f"  ⚠️  wget falló, intentando urllib...")
            headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) fromPodtoCast/1.0"}
            req = Request(audio_url, headers=headers)
            with urlopen(req, timeout=120) as response:
                with open(filepath, 'wb') as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)

        # Validate file
        if filepath.exists() and filepath.stat().st_size > 10000:  # > 10KB
            # Ensure file is readable/writable
            os.chmod(str(filepath), 0o644)
            return str(filepath)
        else:
            filepath.unlink(missing_ok=True)
            print(f"  ❌ Archivo vacío o muy pequeño, eliminado")
            return None

    except Exception as e:
        filepath.unlink(missing_ok=True)
        print(f"  ❌ Error descargando: {e}")
        return None


def convert_to_wav(filepath: str) -> str:
    """Convierte MP3/M4A a WAV 24kHz mono para el pipeline."""
    wav_path = filepath.rsplit('.', 1)[0] + '.wav'
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 10000:
        return wav_path

    try:
        # Use /usr/bin/ffmpeg explicitly (snap ffmpeg can't access /media/ paths)
        ffmpeg_bin = '/usr/bin/ffmpeg' if os.path.exists('/usr/bin/ffmpeg') else 'ffmpeg'
        cmd = [
            ffmpeg_bin, '-y', '-i', filepath,
            '-ar', '24000',  # 24kHz sample rate (matching pipeline config)
            '-ac', '1',      # Mono
            '-acodec', 'pcm_s16le',
            '-loglevel', 'error',
            wav_path
        ]
        result = subprocess.run(cmd, timeout=600, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 10000:
            # Remove original to save space
            os.remove(filepath)
            return wav_path
        else:
            err = result.stderr[:300] if result.stderr else "unknown error"
            print(f"  ❌ ffmpeg error: {err}")
            return None
    except Exception as e:
        print(f"  ❌ Error convirtiendo: {e}")
        return None


def register_download(episode: dict, podcast_name: str, output_dir: str, filepath: str):
    """Registra el episodio descargado en processed_videos.json."""
    pv_path = Path(output_dir).parent / "processed_videos.json"
    data = {"processed": [], "failed": []}

    if pv_path.exists():
        try:
            with open(pv_path, 'r') as f:
                data = json.load(f)
        except:
            pass

    # Check if already registered
    existing_ids = set()
    for key in ["processed", "failed"]:
        items = data.get(key, [])
        if isinstance(items, list):
            for item in items:
                if isinstance(item, str):
                    existing_ids.add(item)
                elif isinstance(item, dict):
                    existing_ids.add(item.get("id", ""))
        elif isinstance(items, dict):
            existing_ids.update(items.keys())

    ep_id = f"rss_{episode['episode_id']}"
    if ep_id in existing_ids:
        return

    entry = {
        "id": ep_id,
        "title": episode['title'],
        "source": "rss",
        "podcast": podcast_name,
        "file": filepath,
        "duration": episode['duration_sec'],
        "downloaded_at": datetime.now().isoformat(),
    }

    # Handle both list and dict formats for "processed"
    proc = data.get("processed", [])
    if isinstance(proc, list):
        proc.append(ep_id)
        data["processed"] = proc
    elif isinstance(proc, dict):
        proc[ep_id] = entry
        data["processed"] = proc
    else:
        data["processed"] = [ep_id]

    with open(pv_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Descargador de podcasts desde RSS feeds"
    )
    parser.add_argument(
        '--feed', '-f',
        help='Nombre del podcast a descargar (default: todos)',
    )
    parser.add_argument(
        '--max-episodes', '-m',
        type=int, default=None,
        help='Máximo episodios por feed (default: from config)',
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Directorio de salida (default: from config)',
    )
    parser.add_argument(
        '--list-feeds',
        action='store_true',
        help='Solo listar feeds disponibles',
    )
    parser.add_argument(
        '--list-episodes',
        action='store_true',
        help='Solo listar episodios sin descargar',
    )
    parser.add_argument(
        '--process',
        action='store_true',
        help='Procesar con fromPodtoCast después de descargar',
    )
    parser.add_argument(
        '--no-convert',
        action='store_true',
        help='No convertir a WAV (mantener MP3/M4A)',
    )

    args = parser.parse_args()
    config = load_config()
    settings = config.get("settings", {})
    feeds = config.get("feeds", [])

    output_dir = args.output_dir or settings.get("output_dir", DEFAULT_OUTPUT_DIR)
    max_episodes = args.max_episodes or settings.get("max_episodes_per_feed", 50)
    min_dur = settings.get("min_duration_minutes", 20)
    max_dur = settings.get("max_duration_minutes", 180)
    delay = settings.get("download_delay_seconds", 2)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if args.list_feeds:
        print("\n📻 Feeds configurados:")
        for i, feed in enumerate(feeds, 1):
            print(f"  {i:2d}. {feed['name']} ({feed['type']})")
        print(f"\n  Total: {len(feeds)} feeds")
        return

    # Filter to specific feed if requested
    if args.feed:
        feeds = [f for f in feeds if args.feed.lower() in f['name'].lower()]
        if not feeds:
            print(f"❌ No se encontró feed: {args.feed}")
            return

    processed_ids = load_processed_ids(output_dir)

    print("=" * 60)
    print("📻 fromPodtoCast - RSS Podcast Downloader")
    print("=" * 60)
    print(f"📁 Output: {output_dir}")
    print(f"📊 Feeds: {len(feeds)}")
    print(f"🔢 Max episodes/feed: {max_episodes}")
    print(f"⏱️  Duration filter: {min_dur}-{max_dur} min")
    print(f"📦 Already downloaded: {len(processed_ids)}")
    print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    total_downloaded = 0
    total_converted = 0
    total_skipped = 0
    total_failed = 0
    downloaded_files = []

    for feed_info in feeds:
        name = feed_info['name']
        url = feed_info['url']

        print(f"\n{'='*60}")
        print(f"🎙️  {name}")
        print(f"    {url}")
        print(f"{'='*60}")

        episodes = parse_rss_feed(url, max_episodes, min_dur, max_dur)

        if not episodes:
            print(f"  ⚠️  No se encontraron episodios válidos")
            continue

        print(f"  📋 {len(episodes)} episodios encontrados")

        if args.list_episodes:
            for ep in episodes:
                dur_str = f"{ep['duration_min']}m" if ep['duration_min'] > 0 else "?"
                print(f"    • [{dur_str}] {ep['title'][:60]}")
            continue

        for i, episode in enumerate(episodes):
            # Check if already downloaded
            ep_id = episode['episode_id']
            safe_name = sanitize_filename(f"{name}_{episode['title']}")
            if ep_id in processed_ids or safe_name in processed_ids:
                total_skipped += 1
                continue

            dur_str = f"{episode['duration_min']}m" if episode['duration_min'] > 0 else "?"
            print(f"\n  📥 [{i+1}/{len(episodes)}] [{dur_str}] {episode['title'][:55]}...")

            filepath = download_episode(episode, output_dir, name)

            if filepath:
                total_downloaded += 1
                print(f"  ✅ Descargado: {Path(filepath).name}")

                # Convert to WAV
                if not args.no_convert and not filepath.endswith('.wav'):
                    print(f"  🔄 Convirtiendo a WAV...")
                    wav_path = convert_to_wav(filepath)
                    if wav_path:
                        filepath = wav_path
                        total_converted += 1
                        print(f"  ✅ Convertido: {Path(wav_path).name}")
                    else:
                        print(f"  ⚠️  Conversión falló, manteniendo original")

                downloaded_files.append(filepath)
                register_download(episode, name, output_dir, filepath)
                processed_ids.add(ep_id)
            else:
                total_failed += 1

            # Rate limiting
            time.sleep(delay)

    # Summary
    print(f"\n{'='*60}")
    print(f"📊 RESUMEN")
    print(f"{'='*60}")
    print(f"  ✅ Descargados:  {total_downloaded}")
    print(f"  🔄 Convertidos:  {total_converted}")
    print(f"  ⏭️  Saltados:     {total_skipped}")
    print(f"  ❌ Fallidos:     {total_failed}")
    print(f"  📁 Archivos en:  {output_dir}")

    # Optionally launch processing
    if args.process and downloaded_files:
        print(f"\n🚀 Lanzando procesamiento de {len(downloaded_files)} archivos...")
        main_script = PROJECT_ROOT / "main.py"
        if main_script.exists():
            cmd = [
                sys.executable, str(main_script),
                output_dir, '-o',
                str(Path(output_dir).parent / "output")
            ]
            subprocess.run(cmd)


if __name__ == "__main__":
    main()
