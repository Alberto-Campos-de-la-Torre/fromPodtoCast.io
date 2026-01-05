# Manual de Limpieza y Recuperación del Pipeline

## 🧹 ¿Cuándo usar la limpieza?

El script de limpieza `cleanup_pipeline.py` es útil cuando:

1. **El pipeline se traba inesperadamente** durante el procesamiento de un audio
2. **Se queda sin memoria** (GPU o RAM)
3. **Hay procesos colgados** que no responden (ffmpeg, whisper, etc.)
4. **Fallos de red** dejan archivos temporales ocupando espacio
5. **Quieres verificar el estado** del procesamiento actual

## 📋 Comandos Básicos

### Ver estado del pipeline
```bash
source venv/bin/activate
python scripts/cleanup_pipeline.py --show-status
```

Este comando muestra:
- ✅ Videos procesados exitosamente
- ❌ Videos fallidos
- 📊 Detalles del último video procesado
- 📊 Detalles del último fallo
- 📁 Estado de directorios (archivos y tamaño)
- 💾 Estado del caché LLM

### Limpieza básica
```bash
python scripts/cleanup_pipeline.py
```

Realiza:
- Elimina archivos temporales (`.tmp`, `.temp`, `.part`)
- Limpia directorios vacíos
- Libera memoria GPU
- Muestra estado final y opciones de recuperación

### Limpieza profunda
```bash
python scripts/cleanup_pipeline.py --deep
```

Además de la limpieza básica:
- Elimina videos/audios parcialmente descargados (`.mp4`, `.webm`, `.m4a`)
- **⚠️ CUIDADO:** Solo usar si estás seguro de que hay descargas incompletas

### Matar procesos colgados
```bash
python scripts/cleanup_pipeline.py --kill-processes
```

Busca y termina procesos que pueden quedarse colgados:
- `ffmpeg` (conversión de audio)
- `yt-dlp` (descarga de videos)
- `whisper` (transcripción)
- `python` procesando scripts del pipeline

### Recuperar procesamiento
```bash
python scripts/cleanup_pipeline.py --recover
```

Analiza el estado y sugiere comandos para:
- Reintentar videos que fallaron en procesamiento
- Continuar con nuevos videos
- Procesar solo audios ya descargados

### Modo dry-run (simulación)
```bash
python scripts/cleanup_pipeline.py --dry-run --kill-processes
```

Muestra qué procesos se matarían **sin matarlos realmente**. Útil para revisar antes de ejecutar.

## 🔄 Flujo de Recuperación Recomendado

### Caso 1: Pipeline se traba durante procesamiento

1. **CTRL+C** para interrumpir el pipeline
2. **Limpiar y verificar estado:**
   ```bash
   python scripts/cleanup_pipeline.py --kill-processes
   ```
3. **Ver qué falló:**
   ```bash
   python scripts/cleanup_pipeline.py --show-status
   ```
4. **Recuperar procesamiento:**
   - Si falló la descarga: Continuar normalmente
     ```bash
     python scripts/auto_pipeline.py
     ```
   - Si falló el procesamiento: Reintentar videos fallidos
     ```bash
     python scripts/auto_pipeline.py --retry-failed
     ```

### Caso 2: Quedarse sin memoria

1. **Matar procesos y limpiar GPU:**
   ```bash
   python scripts/cleanup_pipeline.py --kill-processes
   ```
2. **Revisar configuración** en `config/config.json`:
   - Reducir `batch_size` de LLM
   - Activar `gpu_config.enabled` para distribución multi-GPU
3. **Continuar procesamiento:**
   ```bash
   python scripts/auto_pipeline.py
   ```

### Caso 3: Disco lleno o archivos temporales

1. **Limpieza profunda:**
   ```bash
   python scripts/cleanup_pipeline.py --deep
   ```
2. **Revisar espacio liberado:**
   ```bash
   python scripts/cleanup_pipeline.py --show-status
   ```
3. **Continuar:**
   ```bash
   python scripts/auto_pipeline.py
   ```

### Caso 4: ffmpeg o yt-dlp colgados

1. **Ver procesos activos (dry-run):**
   ```bash
   python scripts/cleanup_pipeline.py --kill-processes --dry-run
   ```
2. **Matar procesos colgados:**
   ```bash
   python scripts/cleanup_pipeline.py --kill-processes
   ```
3. **Continuar:**
   ```bash
   python scripts/auto_pipeline.py
   ```

## 📊 Interpretando el Estado

### Videos Procesados
```
✓ Videos procesados: 111
✗ Videos fallidos: 9
```
- **Procesados:** Videos completamente procesados (transcripción, diarización, corrección LLM)
- **Fallidos:** Videos con errores en descarga o procesamiento

### Último Video Exitoso
```
Último video exitoso:
   Título: PREGUNTAS y RESPUESTAS comentadas del examen MIR 2023...
   Fecha: 2026-01-04T12:46:10.064544
```
Muestra el último video procesado exitosamente y cuándo se completó.

### Último Fallo
```
Último fallo:
   Título: Clinical Cases - Internal Medicine...
   Error: Archivo no encontrado
   Etapa: download
   Fecha: 2026-01-04T11:02:25.021145
```
- **Error:** Descripción del problema
- **Etapa:** `download` (descarga) o `processing` (procesamiento)

### Directorios
```
Directorios:
   Audios descargados            :   121 archivos (20596.6 MB)
   Segmentos temporales          :   557 archivos (224.3 MB)
   Segmentos normalizados        : 30054 archivos (12909.9 MB)
```

- **input:** Audios WAV descargados de YouTube
- **segments:** Segmentos temporales (deben estar vacíos después del procesamiento)
- **normalized:** Segmentos procesados y listos para entrenamiento TTS
- **metadata:** Archivos JSON con información de cada podcast
- **logs:** Logs de procesamiento de cada video

⚠️ **Nota:** Si `segments` tiene muchos archivos (>1000), puede indicar que el pipeline se trabó antes de normalizar. Usar `--deep` para limpiar.

## 🎯 Comandos de Recuperación

Después de la limpieza, el script sugiere cómo continuar:

### 1. Reintentar videos fallidos
```bash
python scripts/auto_pipeline.py --retry-failed
```
Reprocesa solo los videos que fallaron en la etapa de **procesamiento** (no descarga).

### 2. Continuar con nuevos videos
```bash
python scripts/auto_pipeline.py
```
Busca y procesa nuevos videos según la configuración.

### 3. Procesar audios ya descargados
```bash
python scripts/auto_pipeline.py --process-only
```
Útil cuando tienes audios descargados pero no procesados.

## 🔍 Solución de Problemas

### Error: "No se encontró registro de videos procesados"
**Causa:** No existe `processed_videos.json`  
**Solución:** 
```bash
# Crear registro vacío
echo '{"processed": {}, "failed": {}, "skipped": {}}' > /ruta/a/data/processed_videos.json
```

### Caché LLM corrupto
**Causa:** Interrupción durante escritura del caché  
**Solución:**
```bash
# Eliminar caché (se regenerará)
rm /ruta/a/data/llm_cache.json
```

### GPU sin liberar memoria
**Causa:** PyTorch no disponible o drivers CUDA incorrectos  
**Solución:**
```bash
# Verificar CUDA
nvidia-smi

# Reiniciar si es necesario
sudo systemctl restart nvidia-persistenced
```

### Procesos no se matan
**Causa:** Sin permisos suficientes  
**Solución:**
```bash
# Ejecutar con sudo (CUIDADO)
sudo python scripts/cleanup_pipeline.py --kill-processes
```

## ⚙️ Automatización

### Script de limpieza automática (cron)
```bash
# Limpiar diariamente a las 3 AM
0 3 * * * cd /home/user/fromPodtoCast && source venv/bin/activate && python scripts/cleanup_pipeline.py
```

### Pre/Post procesamiento
```bash
# Antes de ejecutar pipeline
python scripts/cleanup_pipeline.py --kill-processes

# Ejecutar pipeline
python scripts/auto_pipeline.py

# Después del pipeline
python scripts/cleanup_pipeline.py
```

## 📈 Mejores Prácticas

1. **Ejecutar limpieza regularmente** para evitar acumulación de archivos temporales
2. **Ver estado antes de procesar** un nuevo batch de videos
3. **Usar `--dry-run`** antes de operaciones destructivas
4. **Hacer backup** del archivo `processed_videos.json` periódicamente
5. **Monitorear espacio en disco** especialmente en directorios `input/` y `normalized/`
6. **Limpiar GPU entre procesamiento** de videos largos o complejos

## 🚨 Advertencias

- ⚠️ **`--deep`**: Elimina videos/audios. Asegúrate de que sean descargas incompletas.
- ⚠️ **`--kill-processes`**: Puede interrumpir procesos legítimos. Verificar con `--dry-run` primero.
- ⚠️ **No ejecutar durante procesamiento activo**: Puede causar corrupción de datos.

## 📝 Logs y Diagnóstico

Los logs de cada video están en:
```
/ruta/a/data/logs/<video_id>.log
```

Para investigar un fallo específico:
```bash
# Ver log del último video fallido
cat /ruta/a/data/logs/<ultimo_video_fallido>.log | jq
```

## 🔗 Ver También

- [README.md](../README.md) - Documentación principal
- [docs/GUIA_INSTALACION.md](GUIA_INSTALACION.md) - Instalación del sistema
- [scripts/auto_pipeline.py](../scripts/auto_pipeline.py) - Pipeline principal
