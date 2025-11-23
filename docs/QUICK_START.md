# 🚀 Guía Rápida - Probar fromPodtoCast

## Paso 1: Verificar Dependencias

```bash
cd /home/ttech-main/fromPodtoCast
python3 scripts/check_dependencies.py
```

Si faltan dependencias, instálalas:
```bash
pip install -r requirements-minimal.txt  # Si ya tienes PyTorch
# O
pip install -r requirements.txt  # Instalación completa
```

## Paso 2: Preparar Archivo de Audio de Prueba

Tienes dos opciones:

### Opción A: Usar un archivo existente
Coloca tu archivo de podcast en `data/input/`:
```bash
# Copiar un archivo de audio (ejemplo)
cp /ruta/a/tu/podcast.mp3 data/input/
```

### Opción B: Usar archivos de prueba del proyecto Local-csm
Si tienes archivos de audio en Local-csm:
```bash
# Copiar archivos de prueba
cp /home/ttech-main/csm/Local-csm/data/audio/*.wav data/input/ 2>/dev/null || echo "No hay archivos en Local-csm/data/audio"
```

## Paso 3: Configurar (Opcional)

Edita `config/config.json` si necesitas ajustar parámetros:
- `whisper_model`: "tiny" (rápido), "base" (recomendado), "small", "medium", "large"
- `min_duration` / `max_duration`: Duración de segmentos (10-15 segundos por defecto)
- `language`: Idioma del audio (null = auto-detectar)

## Paso 4: Ejecutar el Procesador

### Procesar un archivo individual:
```bash
python3 main.py data/input/tu_archivo.mp3 -o data/output
```

### Procesar todos los archivos de un directorio:
```bash
python3 main.py data/input/ -o data/output
```

### Con metadata personalizado:
```bash
python3 main.py data/input/tu_archivo.mp3 -o data/output --metadata data/output/train_metadata.json
```

## Paso 5: Verificar Resultados

Después de la ejecución, deberías ver:

```
data/output/
├── segments/
│   └── nombre_podcast/
│       ├── nombre_podcast_segment_0000.wav
│       ├── nombre_podcast_segment_0001.wav
│       └── ...
├── normalized/
│   └── nombre_podcast/
│       ├── nombre_podcast_segment_0000.wav
│       ├── nombre_podcast_segment_0001.wav
│       └── ...
└── metadata.json
```

### Ver el archivo de metadata:
```bash
cat data/output/metadata.json | head -50
```

### Verificar un segmento de audio:
```bash
# Reproducir un segmento (si tienes un reproductor instalado)
# o verificar con librosa
python3 -c "import librosa; import soundfile as sf; audio, sr = librosa.load('data/output/normalized/nombre_podcast/nombre_podcast_segment_0000.wav'); print(f'Sample rate: {sr} Hz, Duración: {len(audio)/sr:.2f}s')"
```

## Paso 6: Usar el Metadata para Entrenamiento

El archivo `metadata.json` generado es compatible con:
- **Sesame1b**: Usa directamente el archivo JSON
- **Kyutai TTS**: Puede requerir ajustes menores según el formato específico

### Ejemplo de uso con Sesame1b:
```bash
# El archivo metadata.json ya está en el formato correcto
# Puedes usarlo directamente en pretokenize.py de Sesame
python pretokenize.py --train_data data/output/metadata.json --val_data data/output/metadata.json --output tokenized_data.hdf5
```

## Solución de Problemas

### Error: "No module named 'whisper'"
```bash
pip install openai-whisper
```

### Error: "ffmpeg not found"
```bash
sudo apt-get install ffmpeg
```

### Error: "CUDA out of memory"
- Usa un modelo Whisper más pequeño: `"whisper_model": "tiny"` en config.json
- O procesa archivos más pequeños

### Audio sin transcripción
- Verifica que el audio tenga suficiente volumen
- Prueba con un modelo Whisper más grande
- Verifica que el archivo de audio no esté corrupto

## Próximos Pasos

1. ✅ Procesar tus primeros podcasts
2. ✅ Revisar la calidad de las transcripciones
3. ✅ Ajustar parámetros en `config/config.json` según necesites
4. ✅ Usar el metadata.json para entrenar tus modelos TTS

## Notas

- El primer uso de Whisper descargará el modelo (puede tardar)
- El procesamiento puede tardar según la duración del audio
- Los segmentos se guardan en `data/output/segments/` y `data/output/normalized/`
- El archivo final `metadata.json` contiene todas las rutas absolutas

