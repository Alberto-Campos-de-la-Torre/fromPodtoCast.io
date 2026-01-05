# 🔄 Re-procesamiento de Metadata - Resumen de Implementación

## ✅ Archivos Creados

### Scripts Principales

1. **`scripts/reprocess_metadata.py`** (643 líneas)
   - Script principal para re-procesamiento de metadata
   - Soporta archivos individuales o procesamiento en lote
   - Integración completa con LLM y MCP
   - Sistema de backups automático
   - Estadísticas detalladas de procesamiento

2. **`scripts/example_reprocess.sh`**
   - Ejemplos de uso del script
   - Casos de uso comunes documentados
   - Configuraciones personalizadas de ejemplo

3. **`scripts/test_reprocess.py`**
   - Script de prueba automatizado
   - Genera metadata de prueba
   - Verifica funcionamiento del re-procesador

### Documentación

4. **`docs/REPROCESS_METADATA.md`**
   - Documentación completa del sistema
   - Casos de uso detallados
   - Guías de configuración
   - Troubleshooting

5. **`README.md`** (actualizado)
   - Sección nueva sobre re-procesamiento
   - Integración en la estructura del proyecto

## 🎯 Funcionalidades Implementadas

### Procesamiento de Metadata

✅ **Lectura de archivos JSON**
   - Soporte para múltiples estructuras (listas, dicts)
   - Manejo de diferentes formatos de metadata
   - Preservación de estructura original

✅ **Corrección LLM**
   - Uso del modelo configurado (qwen3:14b por defecto)
   - Procesamiento batch optimizado
   - System de caché para evitar re-procesar textos iguales
   - Preservación de texto original

✅ **Verificación MCP**
   - Integración con diccionario MCP
   - Detección de regionalismos/términos protegidos
   - Reversión inteligente de correcciones incorrectas
   - Consultas al diccionario documentadas

✅ **Gestión de Backups**
   - Backups automáticos con timestamp
   - Formato: `archivo.json.backup_YYYYMMDD_HHMMSS`
   - Opción para desactivar backups

✅ **Estadísticas Detalladas**
   - Conteo de correcciones aplicadas
   - Cache hits
   - Verificaciones MCP
   - Reversiones por regionalismo
   - Tiempo de procesamiento

## 🚀 Casos de Uso

### 1. Cambio de Modelo LLM
```bash
# Has actualizado de qwen3:14b a qwen3:32b
python scripts/reprocess_metadata.py --all \
  --data-dir data/output \
  --config config/config_nuevo_modelo.json
```

### 2. Actualización del Diccionario MCP
```bash
# Agregaste 100 nuevos términos técnicos
python scripts/reprocess_metadata.py --all --data-dir data/output
```

### 3. Ajuste de Parámetros de Confianza
```bash
# Quieres ser más estricto con las correcciones
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

### 4. Corrección de Podcast Específico
```bash
# Detectaste errores en un podcast particular
python scripts/reprocess_metadata.py data/output/metadata/podcast_123.json
```

## 📊 Estructura de Metadata

### Antes del Re-procesamiento
```json
{
  "text": "transcripción original",
  "path": "/ruta/audio.wav",
  "speaker": 0,
  "duration": 5.2
}
```

### Después del Re-procesamiento
```json
{
  "text": "transcripción corregida",
  "text_original": "transcripción original",
  "path": "/ruta/audio.wav",
  "speaker": 0,
  "duration": 5.2,
  "llm_correction": {
    "original": "transcripción original",
    "cambios": [{...}],
    "confianza": 0.95,
    "reprocessed_at": "2026-01-05T08:47:00",
    "mcp_verified": true,
    "mcp_confianza": 0.92,
    "mcp_validaciones": [{...}],
    "mcp_verified_at": "2026-01-05T08:47:05"
  }
}
```

## ⚙️ Configuración por Defecto

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

## 🔍 Características Técnicas

### Optimizaciones
- ✅ **Batch processing**: Reduce llamadas HTTP al LLM
- ✅ **Cache persistente**: Evita re-procesar textos iguales
- ✅ **Preservación de datos**: Nunca se pierden textos originales
- ✅ **Backups automáticos**: Seguridad ante errores
- ✅ **Procesamiento paralelo**: Configurable con `max_workers`

### Compatibilidad
- ✅ Múltiples estructuras JSON (lista, dict con 'entries', etc.)
- ✅ Metadata con o sin correcciones previas
- ✅ Integración con sistema de cache existente
- ✅ Compatible con todos los modelos Ollama

### Validaciones
- ✅ Verificación de existencia de archivos
- ✅ Validación de estructura JSON
- ✅ Manejo de errores robusto
- ✅ Mensajes informativos de progreso

## 📈 Estadísticas de Salida

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

## 🧪 Testing

### Ejecutar Test Automatizado
```bash
# Test completo del sistema
python scripts/test_reprocess.py
```

El test:
1. Crea metadata de prueba con errores intencionales
2. Ejecuta el re-procesador
3. Verifica que las correcciones se aplicaron
4. Muestra diferencias entre original y corregido
5. Lista backups creados

## 📝 Próximos Pasos Sugeridos

1. **Ejecutar Test**
   ```bash
   python scripts/test_reprocess.py
   ```

2. **Re-procesar Metadata Existente**
   ```bash
   # Encuentra tus archivos de metadata
   find /media/ttech-main/8464b39b-ba5b-49c5-9e05-010122b9874d -name "*.json" -type f
   
   # Re-procesa todo
   python scripts/reprocess_metadata.py --all --data-dir /ruta/a/tus/datos
   ```

3. **Revisar Backups**
   ```bash
   # Ver backups creados
   find data/output/metadata -name "*.backup_*"
   ```

4. **Optimizar Configuración**
   - Ajustar `batch_size` según tu hardware
   - Modificar `min_confidence` según calidad deseada
   - Configurar `confidence_threshold` de MCP

## 🔗 Referencias

- **Documentación completa**: `docs/REPROCESS_METADATA.md`
- **Ejemplos de uso**: `scripts/example_reprocess.sh`
- **Script de test**: `scripts/test_reprocess.py`
- **Configuración ejemplo**: `config/config_example_mcp.json`

## ✨ Ventajas del Sistema

1. **Sin re-procesar audio** → Ahorra tiempo y recursos
2. **Backups automáticos** → Seguridad total
3. **Batch processing** → Eficiencia máxima
4. **Cache inteligente** → No repite trabajo
5. **MCP verification** → Protege regionalismos
6. **Estadísticas detalladas** → Transparencia total

---

**¡El sistema está listo para usar!** 🎉

Para empezar, ejecuta el test o comienza a re-procesar tus archivos de metadata existentes.
