# 🧹 Sistema de Limpieza y Recuperación del Pipeline

## Resumen

Cuando el pipeline de procesamiento de podcasts se traba inesperadamente, tenemos un sistema completo de limpieza y recuperación que:

1. ✅ **Detecta segmentos huérfanos** comparando directorios con logs
2. ✅ **Limpia archivos temporales** automáticamente  
3. ✅ **Libera memoria GPU** para evitar congelamiento
4. ✅ **Mata procesos colgados** (ffmpeg, whisper, etc.)
5. ✅ **Sugiere opciones de recuperación** basadas en el estado actual

## 🚀 Uso Rápido

### Ver estado actual y verificar consistencia
```bash
source venv/bin/activate
python scripts/cleanup_pipeline.py --verify
```

### Limpiar segmentos huérfanos (simulación primero)
```bash
python scripts/cleanup_pipeline.py --clean-orphans --dry-run
python scripts/cleanup_pipeline.py --clean-orphans
```

### Recuperar después de un crash
```bash
python scripts/cleanup_pipeline.py --kill-processes
python scripts/cleanup_pipeline.py --recover
```

## 📊 Verificación de Consistencia

El sistema verifica automáticamente la consistencia entre:
- **Directorios de segmentos** (`segments/` y `normalized/`)
- **Logs de procesamiento** (`logs/*.log`)
- **Metadata** (`metadata/*.json`)
- **Registro de videos** (`processed_videos.json`)

### ¿Quué detecta?

- **Segmentos huérfanos**: Directorios sin log/metadata correspondiente
- **Procesamiento incompleto**: Log indica fallo pero archivos existen
- **Segmentos faltantes**: Log indica éxito pero archivos no existen
- **Procesamiento consistente**: Todo en orden ✅

## 🔧 Comandos Disponibles

### Verificación
```bash
# Ver estado básico
python scripts/cleanup_pipeline.py --show-status

# Ver estado + verificación de consistencia
python scripts/cleanup_pipeline.py --verify

# Ver estado + opciones de recuperación
python scripts/cleanup_pipeline.py --recover
```

### Limpieza
```bash
# Limpieza básica (archivos .tmp, .temp, .part)
python scripts/cleanup_pipeline.py

# Limpieza profunda (incluye .mp4, .webm parciales)
python scripts/cleanup_pipeline.py --deep

# Limpiar segmentos huérfanos
python scripts/cleanup_pipeline.py --clean-orphans

# Matar procesos colgados
python scripts/cleanup_pipeline.py --kill-processes
```

### Simulación (Dry-run)
```bash
# Ver qué se limpiaría sin hacerlo
python scripts/cleanup_pipeline.py --clean-orphans --dry-run
python scripts/cleanup_pipeline.py --kill-processes --dry-run
```

## 📖 Ejemplo de Output

```
============================================================
  🧹 fromPodtoCast - Pipeline Cleanup
============================================================
  📁 Directorio: /media/ttech-main/8464b39b-ba5b-49c5-9e05-010122b9874d
  ⏱️  Inicio: 2026-01-04 13:36:21
────────────────────────────────────────────────────────────

Estado del Pipeline
============================================================
✓ Videos procesados: 111
✗ Videos fallidos: 9

Verificación de Consistencia:
Verificando consistencia de segmentos...

⚠️  Segmentos temporales huérfanos (sin log):
   • Guia_COMPLETA_de_INTELIGENCIA_Artificial: 557 archivos (224.3 MB)
   Total a limpiar: 224.3 MB

⚠️  Segmentos normalizados huérfanos (sin log):
   • Como_reconfigurar_nuestro_cerebro: 229 archivos (99.6 MB)
   • TODA_LA_VERDAD_sobre_el_MARKETING: 239 archivos (97.4 MB)
   ... y 3 más
   Total a limpiar: 541.4 MB

Resumen de Consistencia:
   ✓ Procesados correctamente: 107
   ⚠️  Segmentos huérfanos: 9
   ⚠️  Procesamiento incompleto: 3
   ⚠️  Segmentos faltantes: 0

💡 Sugerencia: Ejecutar con --clean-orphans para eliminar segmentos huérfanos
```

## 🔄 Flujo de Recuperación Recomendado

1. **Pipeline se traba** → `Ctrl+C` para interrumpir

2. **Verificar estado**:
   ```bash
   python scripts/cleanup_pipeline.py --verify
   ```

3. **Limpiar recursos**:
   ```bash
   # Matar procesos colgados
   python scripts/cleanup_pipeline.py --kill-processes
   
   # Limpiar segmentos huérfanos
   python scripts/cleanup_pipeline.py --clean-orphans
   ```

4. **Recuperar procesamiento**:
   ```bash
   # Opción recomendada basada en el estado
   python scripts/cleanup_pipeline.py --recover
   
   # Luego ejecutar el comando sugerido, por ejemplo:
   python scripts/auto_pipeline.py --retry-failed
   ```

## 📝 Archivos Creados

1. **`scripts/cleanup_pipeline.py`**: Script principal de limpieza
2. **`docs/MANUAL_LIMPIEZA.md`**: Manual completo de uso
3. **`scripts/example_cleanup.sh`**: Ejemplos interactivos
4. **`docs/cleanup_pipeline_workflow.png`**: Diagrama del flujo

## 🎯 Casos de Uso

### Caso 1: Pipeline se traba procesando
```bash
python scripts/cleanup_pipeline.py --kill-processes
python scripts/auto_pipeline.py --retry-failed
```

### Caso 2: Memoria GPU llena
```bash
python scripts/cleanup_pipeline.py  # libera GPU automáticamente
python scripts/auto_pipeline.py
```

### Caso 3: Disco lleno
```bash
python scripts/cleanup_pipeline.py --verify  # ver qué se puede limpiar
python scripts/cleanup_pipeline.py --clean-orphans
python scripts/cleanup_pipeline.py --deep
```

### Caso 4: Muchos archivos temporales
```bash
python scripts/cleanup_pipeline.py  # limpieza básica
```

## ⚠️ Notas Importantes

- **Siempre usar `--verify`** antes de limpiar para saber qué se eliminará
- **Usar `--dry-run`** para simular antes de operaciones destructivas
- **Los segmentos huérfanos** se detectan comparando con logs, no con el registro
- **Backup** el archivo `processed_videos.json` regularmente

## 📚 Documentación Completa

Para más detalles, consulta:
- [`docs/MANUAL_LIMPIEZA.md`](../docs/MANUAL_LIMPIEZA.md) - Manual completo
- [`scripts/cleanup_pipeline.py --help`](../scripts/cleanup_pipeline.py) - Ayuda del comando
- [`scripts/example_cleanup.sh`](../scripts/example_cleanup.sh) - Ejemplos de uso

## 🎨 Diagrama del Flujo

![Pipeline Recovery Workflow](cleanup_pipeline_workflow.png)

## 🤝 Contribuciones

El sistema de limpieza verifica:
- ✅ Consistencia entre segmentos y logs
- ✅ Procesamiento exitoso vs fallido
- ✅ Archivos huérfanos sin referencia
- ✅ Estado de memoria GPU
- ✅ Procesos colgados

---

**Última actualización**: 2026-01-04
