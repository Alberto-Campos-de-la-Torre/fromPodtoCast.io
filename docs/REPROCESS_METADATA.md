# Re-procesamiento de Metadata

## Descripción General

El script `reprocess_metadata.py` permite re-aplicar correcciones LLM y verificación MCP a archivos de metadata ya generados, **sin necesidad de re-procesar el audio**.

Esto es útil cuando:
- Cambiaste el modelo LLM y quieres mejorar las correcciones
- Actualizaste el diccionario MCP con nuevos términos
- Ajustaste los parámetros de confianza
- Encontraste errores en las correcciones y quieres corregirlos

## 🚀 Uso Rápido

### Re-procesar un archivo específico
```bash
python scripts/reprocess_metadata.py data/output/metadata/mi_podcast.json
```

### Re-procesar todos los archivos en un directorio
```bash
python scripts/reprocess_metadata.py --all --data-dir data/output
```

### Con configuración personalizada
```bash
python scripts/reprocess_metadata.py --all --data-dir data/output --config config/config.json
```

## 📋 Opciones de Línea de Comandos

| Opción | Descripción |
|--------|-------------|
| `metadata_file` | Archivo de metadata JSON a re-procesar (opcional si se usa `--all`) |
| `--all` | Re-procesar todos los archivos de metadata en un directorio |
| `--data-dir DIR` | Directorio donde buscar archivos (default: `./data/output`) |
| `--config FILE` | Archivo de configuración JSON personalizado |
| `--no-backup` | No crear backup antes de modificar (no recomendado) |

## ⚙️ Configuración

### Configuración por Defecto

Si no se especifica archivo de configuración, se usan estos valores:

```json
{
  "text_preprocessing": {
    "glosario_path": "config/glosario_terminos.json"
  },
  "llm_correction": {
    "enabled": true,
    "ollama_host": "http://localhost:11434",
    "model": "qwen3:14b",
    "timeout": 120,
    "max_retries": 3,
    "batch_size": 5,
    "min_confidence": 0.7,
    "enable_cache": true,
    "max_workers": 2,
    "mcp_verification": {
      "enabled": true,
      "model": "qwen3:14b",
      "dictionary_path": "data/diccionario_base.json",
      "timeout": 60,
      "confidence_threshold": 0.80
    }
  }
}
```

### Configuración Personalizada

Crea un archivo JSON con las configuraciones que quieres cambiar:

```json
{
  "llm_correction": {
    "model": "llama3:70b",
    "batch_size": 10,
    "min_confidence": 0.75,
    "mcp_verification": {
      "model": "llama3:70b",
      "confidence_threshold": 0.85
    }
  }
}
```

Y úsalo con:
```bash
python scripts/reprocess_metadata.py --all --config mi_config.json
```

## 🔄 Flujo de Procesamiento

1. **Backup**: Crea copia de seguridad del archivo original (opcional)
2. **Carga**: Lee el archivo de metadata JSON
3. **Extracción**: Identifica textos para re-procesar
4. **Corrección LLM**: Aplica correcciones usando el modelo especificado
5. **Verificación MCP**: Verifica correcciones usando diccionario MCP
6. **Actualización**: Actualiza el metadata con nuevas correcciones
7. **Guardado**: Guarda el archivo actualizado

## 📊 Estructura del Metadata

### Antes del Re-procesamiento
```json
{
  "text": "Este es el texto transcripto",
  "path": "/ruta/al/audio.wav",
  "speaker": 0,
  "duration": 5.2,
  ...
}
```

### Después del Re-procesamiento
```json
{
  "text": "Este es el texto transcrito",
  "text_original": "Este es el texto transcripto",
  "path": "/ruta/al/audio.wav",
  "speaker": 0,
  "duration": 5.2,
  "llm_correction": {
    "original": "Este es el texto transcripto",
    "cambios": [
      {
        "tipo": "ortografía",
        "original": "transcripto",
        "corregido": "transcrito"
      }
    ],
    "confianza": 0.95,
    "reprocessed_at": "2026-01-05T07:30:00",
    "mcp_verified": true,
    "mcp_confianza": 0.92,
    "mcp_validaciones": [...],
    "mcp_verified_at": "2026-01-05T07:30:05"
  },
  ...
}
```

## 🎯 Casos de Uso

### Caso 1: Cambio de Modelo LLM

Has actualizado tu infraestructura y ahora usas un modelo más potente:

```bash
# Crear configuración temporal
cat > /tmp/nuevo_modelo.json << 'EOF'
{
  "llm_correction": {
    "model": "qwen3:32b",
    "batch_size": 3,
    "mcp_verification": {
      "model": "qwen3:32b"
    }
  }
}
EOF

# Re-procesar todo
python scripts/reprocess_metadata.py --all \
  --data-dir /media/ttech-main/8464b39b-ba5b-49c5-9e05-010122b9874d \
  --config /tmp/nuevo_modelo.json
```

### Caso 2: Actualización del Diccionario MCP

Has añadido 100 nuevos términos a tu diccionario MCP:

```bash
# Los nuevos términos ya están en data/diccionario_base.json
# Solo re-procesar para aplicar las nuevas validaciones
python scripts/reprocess_metadata.py --all --data-dir data/output
```

### Caso 3: Ajuste de Confianza

Quieres ser más estricto con las correcciones aceptadas:

```bash
cat > /tmp/config_estricta.json << 'EOF'
{
  "llm_correction": {
    "min_confidence": 0.85,
    "mcp_verification": {
      "confidence_threshold": 0.90
    }
  }
}
EOF

python scripts/reprocess_metadata.py --all --config /tmp/config_estricta.json
```

### Caso 4: Corregir un Podcast Específico

Detectaste errores en un podcast en particular:

```bash
# Encuentra el archivo de metadata
ls data/output/metadata/

# Re-procesa solo ese archivo
python scripts/reprocess_metadata.py data/output/metadata/nombre_podcast.json
```

## 📁 Gestión de Backups

### Formato de Backup
Los backups se crean automáticamente con este formato:
```
archivo.json.backup_YYYYMMDD_HHMMSS
```

Ejemplo:
```
podcast_123.json                    # Archivo actual
podcast_123.json.backup_20260105_073000  # Backup
podcast_123.json.backup_20260104_153000  # Backup anterior
```

### Restaurar desde Backup
```bash
# Si necesitas revertir los cambios
cp archivo.json.backup_20260105_073000 archivo.json
```

### Limpiar Backups Antiguos
```bash
# Eliminar backups de más de 7 días
find data/output/metadata -name "*.backup_*" -mtime +7 -delete
```

## 📈 Estadísticas de Salida

El script muestra estadísticas detalladas al finalizar:

```
============================================================
RESUMEN DE RE-PROCESAMIENTO
============================================================
Archivos procesados: 5/5

Total de entradas:   1250
Entradas vacías:     12

Corrección LLM:
  Corregidas:        843
  Fallidas:          8
  Cache hits:        127

Verificación MCP:
  Verificadas:       823
  Revertidas:        20
============================================================
```

## ⚠️ Consideraciones Importantes

### Preservación de Datos
- El script **preserva** los archivos de audio originales
- El texto original se guarda en `text_original` o `llm_correction.original`
- Los timestamps de procesamiento se añaden automáticamente

### Procesamiento Batch
- Se usa procesamiento por lotes para máxima eficiencia
- El tamaño de lote se puede ajustar con `batch_size`
- Reduce significativamente el tiempo de procesamiento

### Cache
- Las correcciones LLM se cachean automáticamente
- Si re-procesas el mismo texto, se usa el cache
- Útil para re-procesar múltiples veces con diferentes parámetros MCP

### Compatibilidad
- Funciona con diferentes estructuras de metadata:
  - Lista de entradas: `[{...}, {...}]`
  - Dict con 'entries': `{"entries": [{...}]}`
  - Dict con 'segments': `{"segments": [{...}]}`

## 🔧 Troubleshooting

### Error: "No hay procesadores disponibles"
Verifica que Ollama esté corriendo:
```bash
curl http://localhost:11434/api/tags
```

### Error: "Archivo no encontrado"
Verifica la ruta:
```bash
ls -la data/output/metadata/
```

### Correcciones no se aplican
Revisa la configuración de `min_confidence`:
- Si está muy alta, pocas correcciones se aplicarán
- Valor recomendado: 0.7 - 0.8

### MCP revierte muchas correcciones
- Esto es normal para regionalismos/términos técnicos
- El diccionario MCP protege estos términos
- Revisa `llm_correction.mcp_razon` para ver por qué se revirtió

## 🔗 Scripts Relacionados

- `auto_pipeline.py` - Pipeline principal de procesamiento
- `cleanup_pipeline.py` - Limpieza y recuperación del pipeline
- `test_mcp_system.py` - Pruebas del sistema MCP

## 📝 Ejemplos Adicionales

Ver `scripts/example_reprocess.sh` para más ejemplos de uso.
