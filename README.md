# fromPodtoCast

Solución para crear archivos de entrenamiento para modelos de TTS (Text-to-Speech) como Kyutai TTS y Sesame1b a partir de audios de podcast pregrabados.

## Características

- **Segmentación inteligente**: Divide podcasts en segmentos de 10-15 segundos, respetando pausas naturales
- **Normalización de audio**: Normaliza bitrate, sample rate y niveles de audio (LUFS) para consistencia
- **Transcripción automática**: Utiliza Whisper para transcribir audio a texto
- **Diarización de hablantes**: Identifica y etiqueta diferentes narradores (opcional, requiere token de Hugging Face)
- **Voice Bank global**: Reutiliza embeddings de hablantes conocidos para asignar IDs consistentes en múltiples podcasts
- **Generación de metadata**: Crea archivos JSON compatibles con formatos de entrenamiento TTS

## Requisitos

- Python 3.8+
- FFmpeg (para procesamiento de audio)
- CUDA (opcional, para aceleración GPU)

## Instalación

1. Clonar o descargar el proyecto:
```bash
cd /home/ttech-main/fromPodcast
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Instalar FFmpeg (si no está instalado):
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

## Configuración

Edita el archivo `config/config.json` para ajustar los parámetros:

```json
{
  "min_duration": 10.0,          // Duración mínima de segmentos (segundos)
  "max_duration": 15.0,           // Duración máxima de segmentos (segundos)
  "silence_thresh": -40.0,        // Umbral de silencio para segmentación (dB)
  "min_silence_len": 500,         // Longitud mínima de silencio (ms)
  "target_sr": 22050,             // Sample rate objetivo (Hz)
  "target_lufs": -23.0,           // Nivel LUFS objetivo
  "normalize_peak": true,         // Normalizar pico a -1.0 dB
  "whisper_model": "base",        // Modelo Whisper (tiny, base, small, medium, large)
  "device": null,                 // Dispositivo (null = auto, "cuda" o "cpu")
  "language": null,                // Idioma (null = auto-detectar)
  "use_diarization": false,       // Habilitar diarización de hablantes
  "hf_token": null,               // Token de Hugging Face (necesario para diarización)
  "use_voice_bank": true,         // Reutilizar voces conocidas mediante embeddings
  "voice_bank_path": "./data/output/voice_bank.json", // Ubicación del banco global
  "voice_match_threshold": 0.85   // Umbral de similitud coseno para reutilizar IDs
}
```

### Configuración de Diarización y Voice Bank (Opcional)

Para usar la diarización de hablantes, necesitas:

1. Obtener un token de Hugging Face:
   - Crear cuenta en https://huggingface.co
   - Aceptar los términos de uso de los modelos de pyannote
   - Generar un token en https://huggingface.co/settings/tokens

2. Configurar en `config/config.json`:
```json
{
  "use_diarization": true,
  "hf_token": "tu_token_aqui",
  "use_voice_bank": true,
  "voice_match_threshold": 0.85
}
```

> Consulta `docs/VOICE_BANK.md` para entender el flujo interno de embeddings y el formato de `voice_bank.json`.

## Uso

### Procesar un archivo de podcast

```bash
python main.py /ruta/al/podcast.mp3 -o ./data/output
```

### Procesar múltiples archivos

```bash
python main.py /ruta/a/directorio/con/podcasts -o ./data/output
```

### Especificar archivo de metadata de salida

```bash
python main.py /ruta/al/podcast.mp3 -o ./data/output --metadata ./data/train_metadata.json
```

### Usar configuración personalizada

```bash
python main.py /ruta/al/podcast.mp3 -c ./config/mi_config.json
```

## Estructura de Salida

El procesador genera la siguiente estructura:

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

## Formato de Metadata

El archivo `metadata.json` generado sigue el formato compatible con Sesame1b y otros modelos TTS:

```json
[
  {
    "text": "Transcripción del segmento de audio",
    "path": "/ruta/absoluta/al/archivo.wav",
    "speaker": 0,
    "speaker_label": "SPEAKER_00",
    "start": 0.0,
    "end": 12.5,
    "duration": 12.5,
    "language": "es",
    "podcast_id": "nombre_podcast"
  }
]
```

## Componentes del Proyecto

- **audio_segmenter.py**: Segmenta audio en fragmentos de duración específica
- **audio_normalizer.py**: Normaliza audio (sample rate, niveles, LUFS)
- **transcriber.py**: Transcribe audio a texto usando Whisper
- **speaker_diarizer.py**: Identifica y etiqueta diferentes hablantes
- **processor.py**: Orquesta todo el proceso de pipeline
- **main.py**: Script principal de línea de comandos

## Modelos Whisper

Puedes elegir diferentes modelos de Whisper según tus necesidades:

- **tiny**: Más rápido, menor precisión
- **base**: Balance velocidad/precisión (recomendado)
- **small**: Mejor precisión, más lento
- **medium**: Alta precisión
- **large**: Máxima precisión, muy lento

## Notas

- Los archivos de audio deben estar en formatos compatibles (WAV, MP3, FLAC, M4A, OGG)
- El procesamiento puede tardar según la duración de los podcasts y el modelo de Whisper usado
- Para mejor rendimiento, usa GPU (CUDA) si está disponible
- La diarización de hablantes es opcional y requiere token de Hugging Face

## Solución de Problemas

### Error: "No module named 'whisper'"
```bash
pip install openai-whisper
```

### Error: "ffmpeg not found"
Instala FFmpeg según tu sistema operativo (ver sección de instalación).

### Error en diarización: "Authentication required"
Asegúrate de tener un token válido de Hugging Face y de haber aceptado los términos de uso de los modelos de pyannote.

### Audio sin transcripción
Verifica que el audio tenga suficiente volumen y claridad. Puedes probar con un modelo Whisper más grande.

## Licencia

Este proyecto está bajo la licencia MIT.

## Referencias

- [Kyutai TTS](https://github.com/kyutai-labs/delayed-streams-modeling)
- [Sesame1b](https://github.com/SesameAILabs/csm)
- [Whisper](https://github.com/openai/whisper)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)

