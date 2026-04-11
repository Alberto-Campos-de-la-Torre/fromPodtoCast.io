import os, sys, json, argparse
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from processor import PodcastProcessor

parser = argparse.ArgumentParser()
parser.add_argument('fraction_start', type=float, help='Fraction of files to start from (e.g. 0.25)')
args = parser.parse_args()

def load_config(path):
    with open(path) as f: return json.load(f)

config = load_config('./config/config.json')
processor = PodcastProcessor(config)

input_path = Path("/media/ttech-main/8464b39b-ba5b-49c5-9e05-010122b9874d/input")
output_dir = "/media/ttech-main/42A4266DA426639F/Training and Test/segments"

audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
audio_files = []
for ext in audio_extensions:
    audio_files.extend(input_path.glob(f'*{ext}'))
    audio_files.extend(input_path.glob(f'**/*{ext}'))
audio_files = [str(f) for f in audio_files]

start_idx = int(len(audio_files) * args.fraction_start)
audio_files = audio_files[start_idx:]

print(f"Worker iniciando en índice {start_idx} (Fracción: {args.fraction_start}). Archivos a procesar: {len(audio_files)}")

for i, audio_file in enumerate(audio_files, 1):
    podcast_name = Path(audio_file).stem
    import re
    podcast_id_clean = re.sub(r'[^a-zA-Z0-9_-]', '_', podcast_name)[:50]
    
    # Check if already processed
    metadata_path = os.path.join(output_dir, 'metadata', f"{podcast_id_clean}.json")
    if os.path.exists(metadata_path):
        print(f"[{i}/{len(audio_files)}] ⏭️  Saltando {podcast_id_clean}, ya existe.")
        continue
        
    print(f"\n[{i}/{len(audio_files)}] Procesando: {Path(audio_file).name}")
    try:
        processor.process_podcast(audio_file, output_dir)
    except Exception as e:
        print(f"✗ Error procesando {audio_file}: {e}")
        
    import gc
    gc.collect()
    if hasattr(processor, 'gpu_manager'):
        processor.gpu_manager.cleanup_memory()
