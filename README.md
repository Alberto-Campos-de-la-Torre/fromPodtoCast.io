# fromPodtoCast

Solución completa para crear datasets de entrenamiento para modelos TTS (Text-to-Speech) a partir de podcasts. Incluye búsqueda automática en YouTube, descarga, procesamiento y generación de datos de entrenamiento.

## 🚀 Características Principales

### Pipeline Automático (`auto_pipeline.py`)
- **Búsqueda en YouTube**: Busca podcasts por categorías configurables
- **Descarga automática**: Descarga y convierte a WAV automáticamente
- **Procesamiento completo**: Diarización, segmentación, transcripción y verificación
- **Registro de progreso**: Evita duplicados y permite retomar procesos fallidos
- **Generación de reportes**: Gráficas visuales del procesamiento

### Procesamiento de Audio (`main.py`)
- **Segmentación inteligente**: Divide podcasts en segmentos de 5-15 segundos
- **Normalización de audio**: Ajusta sample rate (22050 Hz), niveles LUFS (-23.0)
- **Diarización de hablantes**: Identifica y etiqueta diferentes narradores
- **Voice Bank global**: Reutiliza IDs de hablantes entre podcasts

### Transcripción y Texto
- **Whisper**: Transcripción automática con detección de idioma
- **Preprocesamiento**: Corrección de puntuación, números, espaciado
- **Corrección LLM**: Verificación y corrección con modelos de lenguaje (Ollama)

### 🚀 Optimizaciones de Rendimiento (Nuevo)

#### Validación con Pydantic
El proyecto utiliza **Pydantic v2** para validación estructurada de datos:
- `LLMCorrectionResponse`: Valida respuestas del modelo de lenguaje
- `LLMCorrectionBatchResponse`: Valida respuestas en lote
- `SegmentMetadata` / `PodcastMetadata`: Valida estructura de salida
- Detección automática de errores antes de guardar JSON
- Serialización/deserialización tipada

#### Procesamiento LLM Optimizado
- **Batch Processing**: Agrupa 5 textos por llamada HTTP (80% menos requests)
- **Caché Persistente**: Almacena correcciones en JSON con expiración configurable
- **ThreadPoolExecutor**: Procesamiento paralelo opcional (3-4x más rápido)
- **Reintentos Inteligentes**: Fallback automático si el batch falla

#### Barras de Progreso
- Progreso en tiempo real para cada etapa del pipeline
- Indicadores visuales con colores (tqdm)
- Resumen detallado al finalizar con estadísticas

## 📁 Estructura del Proyecto

```
fromPodtoCast/
├── main.py                    # Procesador principal de audio
├── config/
│   ├── config.json            # Configuración del procesador
│   ├── search_queries.json    # Categorías de búsqueda para auto_pipeline
│   └── glosario_terminos.json # Términos técnicos para corrección
├── scripts/
│   ├── auto_pipeline.py       # Pipeline automático completo
│   ├── download_video.py      # Descarga de videos/audio
│   ├── reprocess_metadata.py  # Re-procesar metadata con LLM/MCP
│   ├── cleanup_pipeline.py    # Limpieza y recuperación del pipeline
│   └── check_dependencies.py  # Verificador de dependencias
├── src/
│   ├── processor.py           # Orquestador del pipeline
│   ├── audio_segmenter.py     # Segmentación de audio
│   ├── audio_normalizer.py    # Normalización de audio
│   ├── transcriber.py         # Transcripción con Whisper
│   ├── speaker_diarizer.py    # Diarización de hablantes
│   ├── segment_reviewer.py    # Revisión de segmentos
│   ├── voice_bank.py          # Gestión de voces conocidas
│   ├── text_preprocessor.py   # Preprocesamiento de texto
│   ├── text_corrector_llm.py  # Corrección con LLM (optimizado)
│   ├── correction_cache.py    # Caché de correcciones LLM
│   └── models/                # Schemas Pydantic
│       ├── llm_schemas.py     # Validación de respuestas LLM
│       └── metadata_schemas.py # Validación de metadata
└── docs/                      # Documentación adicional
```

## 🛠️ Instalación

### Requisitos
- Python 3.8+
- FFmpeg (para procesamiento de audio)
- CUDA (opcional, para aceleración GPU)

### Pasos

```bash
# 1. Clonar el proyecto
cd /home/ttech-main/fromPodtoCast

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar dependencias
python scripts/check_dependencies.py
```

## 📖 Uso

### 1. Pipeline Automático (Recomendado)

El script `auto_pipeline.py` automatiza todo el proceso: búsqueda, descarga y procesamiento.

```bash
# Buscar, descargar y procesar 20 videos
python scripts/auto_pipeline.py --videos 20

# Ver qué videos se encontrarían (sin descargar)
python scripts/auto_pipeline.py --dry-run --videos 10

# Solo descargar, sin procesar
python scripts/auto_pipeline.py --download-only --videos 5

# Procesar solo una categoría
python scripts/auto_pipeline.py --category podcasts_medicina --videos 10

# Reprocesar videos que fallaron
python scripts/auto_pipeline.py --retry-failed

# Limpiar registro de fallidos
python scripts/auto_pipeline.py --reset-failed
```

#### Opciones del Auto Pipeline

| Opción | Descripción |
|--------|-------------|
| `--videos N` | Número total de videos a descargar (default: 10) |
| `--dry-run` | Solo mostrar qué se descargaría |
| `--download-only` | Solo descargar, no procesar |
| `--category NAME` | Procesar solo una categoría |
| `--limit N` | Máximo videos por query de búsqueda |
| `--retry-failed` | Reprocesar videos fallidos |
| `--reset-failed` | Limpiar registro de videos fallidos |
| `--data-path PATH` | Ruta donde guardar datos |

### 2. Procesamiento Manual

Para procesar archivos de audio existentes:

```bash
# Procesar un archivo
python main.py /ruta/al/podcast.wav -o ./output

# Procesar un directorio
python main.py /ruta/a/directorio/ -o ./output

# Con configuración personalizada
python main.py archivo.wav -o ./output -c ./config/mi_config.json
```

### 3. Descarga de Videos

Para descargar videos individualmente:

```bash
python scripts/download_video.py "URL_DEL_VIDEO" -o ./data/input --format wav
```

### 4. Re-procesamiento de Metadata

Para re-aplicar correcciones LLM y verificación MCP a metadata ya generado (sin re-procesar audio):

```bash
# Re-procesar un archivo de metadata específico
python scripts/reprocess_metadata.py data/output/metadata/mi_podcast.json

# Re-procesar todos los archivos en un directorio
python scripts/reprocess_metadata.py --all --data-dir data/output

# Con configuración personalizada
python scripts/reprocess_metadata.py --all --config config/config.json

# Sin crear backups (no recomendado)
python scripts/reprocess_metadata.py archivo.json --no-backup
```

**Casos de uso:**
- Cambiaste el modelo LLM y quieres mejorar las correcciones
- Actualizaste el diccionario MCP con nuevos términos
- Ajustaste los parámetros de confianza
- Encontraste errores en correcciones previas

Ver [REPROCESS_METADATA.md](docs/REPROCESS_METADATA.md) para más detalles.


## ⚙️ Configuración

### config/config.json

```json
{
  "min_duration": 5.0,              // Duración mínima de segmentos (segundos)
  "max_duration": 15.0,             // Duración máxima de segmentos
  "target_sr": 22050,               // Sample rate objetivo (Hz)
  "target_lufs": -23.0,             // Nivel LUFS objetivo
  "whisper_model": "base",          // Modelo Whisper: tiny, base, small, medium, large
  "use_diarization": true,          // Habilitar diarización
  "hf_token": "hf_xxx",             // Token de Hugging Face
  "use_voice_bank": true,           // Reutilizar voces conocidas
  "use_segment_review": true,       // Segunda etapa de revisión
  "text_preprocessing": {
    "enabled": true,
    "fix_punctuation": true,
    "normalize_numbers": true
  },
  "llm_correction": {
    "enabled": true,
    "ollama_host": "http://localhost:11434",
    "model": "qwen3:8b",
    "use_batch": true,
    "batch_size": 5,
    "enable_cache": true,
    "cache_file": "./llm_cache.json"
  }
}
```

### config/search_queries.json

Define las categorías de búsqueda para el auto pipeline:

```json
{
  "search_settings": {
    "max_results_per_query": 5,
    "min_duration_minutes": 10,
    "max_duration_minutes": 180
  },
  "categories": [
    {
      "name": "podcasts_negocios",
      "enabled": true,
      "queries": [
        "podcast emprendimiento español",
        "podcast marketing digital español"
      ],
      "exclude_keywords": ["shorts", "clip"]
    }
  ]
}
```

## 📊 Salida Generada

```
Base de Datos - Voz/
├── input/                         # Audios descargados
│   └── podcast_ejemplo.wav
├── normalized/                    # Segmentos procesados
│   └── podcast_ejemplo/
│       ├── seg_0000_SPK_00.wav
│       ├── seg_0001_SPK_01.wav
│       └── ...
├── metadata/                      # Metadata por podcast
│   └── podcast_ejemplo.json
├── logs/                          # Logs de procesamiento
│   └── podcast_ejemplo.log
├── metadata.json                  # Metadata consolidada
├── voice_bank.json                # Banco de voces conocidas
├── processed_videos.json          # Registro de videos procesados
└── pipeline_report_*.png          # Gráficas de reporte
```

### Formato de Metadata

```json
[
  {
    "text": "Transcripción del segmento",
    "path": "/ruta/absoluta/al/archivo.wav",
    "speaker": 0,
    "speaker_label": "SPK_00",
    "start": 0.0,
    "end": 12.5,
    "duration": 12.5,
    "language": "es",
    "podcast_id": "nombre_podcast",
    "segment_id": "seg_0000_SPK_00"
  }
]
```

## 📈 Reportes y Gráficas

Al finalizar el procesamiento, se genera automáticamente una gráfica con:

- **Resumen General**: Videos procesados, audio total, audio útil
- **Duración vs Tiempo de Procesamiento**: Por cada video
- **Audio Total vs Audio Útil**: Eficiencia del procesamiento
- **Estadísticas Detalladas**: Métricas completas

## 🔧 Pipeline de Procesamiento

1. **Diarización** → Identifica hablantes en el audio
2. **Segmentación** → Divide en clips de 5-15 segundos
3. **Normalización** → Ajusta volumen y sample rate
4. **Transcripción** → Convierte audio a texto (Whisper)
5. **Preprocesamiento** → Limpia puntuación, números (diccionarios)
6. **Corrección LLM** → Verifica y corrige texto (batch + caché)
7. **Validación** → Verifica estructura con Pydantic
8. **Generación Metadata** → Crea archivos JSON

### 🔧 Optimizaciones del Pipeline

#### Rendimiento LLM

| Característica | Descripción | Impacto |
|----------------|-------------|---------|
| **Batch Processing** | Agrupa 5 textos por llamada | 80% menos HTTP calls |
| **Caché Persistente** | Guarda correcciones en JSON | Instant en repetidos |
| **Paralelo** | ThreadPoolExecutor opcional | 3-4x más rápido |
| **Pydantic** | Validación de respuestas | <1% errores parsing |

#### Schemas Pydantic

```python
# src/models/llm_schemas.py
class LLMCorrectionResponse(BaseModel):
    texto_corregido: str = Field(..., min_length=1)
    cambios: List[str] = Field(default_factory=list, max_length=10)
    confianza: float = Field(ge=0.0, le=1.0)

# src/models/metadata_schemas.py  
class SegmentMetadata(BaseModel):
    audio_path: str
    speaker: str
    text: str
    duration: float = Field(gt=0)
    sample_rate: int = Field(default=22050)
```

#### Métricas de Rendimiento

| Métrica | Sin Optimización | Con Optimización |
|---------|------------------|------------------|
| Llamadas HTTP/podcast | ~200 | ~40 |
| Tiempo LLM/podcast | ~15 min | ~4 min |
| Errores de parsing | ~5% | <1% |
| Textos repetidos | Reprocesados | Desde caché |

## 🐛 Solución de Problemas

### Error de conversión de audio (ffmpeg snap)
El script automáticamente usa `/usr/bin/ffmpeg` en lugar del ffmpeg de snap para evitar problemas de permisos.

### Warnings de Lightning
Los warnings de PyTorch Lightning son filtrados automáticamente y no detienen el procesamiento.

### Videos fallidos
Usa `--retry-failed` para reprocesar videos que fallaron:
```bash
python scripts/auto_pipeline.py --retry-failed
```

### Token de Hugging Face
Para diarización, necesitas un token de HuggingFace:
1. Crea cuenta en https://huggingface.co
2. Acepta términos de pyannote/speaker-diarization
3. Genera token en https://huggingface.co/settings/tokens
4. Añádelo a config.json

## 📝 Licencia

Este proyecto está bajo la licencia MIT.

## 🔗 Referencias

- [Whisper](https://github.com/openai/whisper) - Transcripción de audio
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Diarización de hablantes
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Descarga de videos
- [Ollama](https://ollama.ai/) - Corrección con LLM local
- [Pydantic](https://docs.pydantic.dev/) - Validación de datos estructurados
- [tqdm](https://github.com/tqdm/tqdm) - Barras de progreso
