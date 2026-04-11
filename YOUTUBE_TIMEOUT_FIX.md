# Solución a Timeouts en Auto Pipeline

## Problema
Timeouts frecuentes durante la búsqueda de videos en YouTube.

## Causas Principales
1. **YouTube rate limiting** - Demasiadas búsquedas rápidas
2. **Problemas con cookies de Chrome** - Puede fallar la extracción
3. **Timeout muy corto** - 120s no es suficiente con conexión lenta
4. **Problemas de red** - Latencia alta o conexión inestable

## ✅ Solución Aplicada

### Cambios en `auto_pipeline.py`:

```diff
- timeout=120
+ timeout=180  # 3 minutos

+ '--socket-timeout', '30',  # Timeout por socket
+ '--retries', '3',  # Reintentar 3 veces  
+ '--sleep-requests', '1',  # Esperar 1s entre requests
```

### Mejoras Implementadas:
1. ✅ **Timeout aumentado**: 120s → 180s (3 minutos)
2. ✅ **Socket timeout**: 30s por socket individual
3. ✅ **Reintentos automáticos**: 3 intentos por búsqueda
4. ✅ **Delay entre requests**: 1s para evitar rate limiting

## Soluciones Adicionales

### 1. Si persisten los timeouts

**Aumentar aún más el timeout:**
```python
# En auto_pipeline.py línea ~180
timeout=300  # 5 minutos
```

### 2. Reducir número de búsquedas simultáneas

**Editar `config/search_queries.json`:**
```json
{
  "search_settings": {
    "max_results_per_query": 3,  // Reducir de 5 a 3
  }
}
```

### 3. Usar proxy o VPN

Si YouTube está bloqueando tu IP:
```bash
# Opción 1: Con proxy
export HTTP_PROXY=http://tu-proxy:8080
python3 scripts/auto_pipeline.py

# Opción 2: Con VPN activa
python3 scripts/auto_pipeline.py
```

### 4. Desactivar cookies de Chrome

Si el problema es con las cookies:

**Editar `auto_pipeline.py` línea ~145:**
```python
# Comentar o eliminar esta línea:
# '--cookies-from-browser', 'chrome',
```

### 5. Usar archivo de cookies manual

**Exportar cookies de YouTube:**
```bash
# Instalar extensión "Get cookies.txt" en Chrome
# Exportar cookies de youtube.com a cookies.txt
```

**Usar en auto_pipeline.py:**
```python
'--cookies', '/path/to/cookies.txt',
```

### 6. Reducir categorías activas

**Editar `config/search_queries.json`:**
```json
{
  "categories": [
    {
      "name": "entrevistas",
      "enabled": true  // Dejar solo 1-2 categorías activas
    },
    {
      "name": "debates",
      "enabled": false  // Deshabilitar el resto
    }
  ]
}
```

### 7. Ejecutar con delay entre categorías

**Modificar `auto_pipeline.py` línea ~576:**
```python
# Después de procesar cada categoría
time.sleep(10)  # Esperar 10s entre categorías
```

## Diagnóstico

### Verificar si es problema de red:
```bash
# Test de velocidad
curl -o /dev/null -s -w '%{time_total}s\\n' https://www.youtube.com
```

### Verificar si es rate limiting:
```bash
# Probar búsqueda manual
yt-dlp "ytsearch5:podcast español" --dump-json --flat-playlist
```

### Ver logs detallados:
```bash
# Ejecutar con verbose
CUDA_VISIBLE_DEVICES=1 python3 scripts/auto_pipeline.py --max-videos 3 -v
```

## Recomendaciones

### Para conexión lenta:
- Usar `timeout=300` (5 minutos)
- Reducir `max_results_per_query` a 2-3
- Procesar 1 categoría a la vez

### Para evitar bloqueos:
- Activar VPN
- Usar `--sleep-requests 3` (3 segundos)
- Reducir número de videos simultáneos

### Para ejecuciones largas:
```bash
# Ejecutar en background con nohup
nohup CUDA_VISIBLE_DEVICES=1 python3 scripts/auto_pipeline.py --max-videos 20 > pipeline.log 2>&1 &

# Ver progreso
tail -f pipeline.log
```

## Restaurar Versión Anterior

Si hay problemas con los cambios:
```bash
cd /home/ttech-main/fromPodtoCast/scripts
cp auto_pipeline.py.backup auto_pipeline.py
```

---

**Última actualización:** 2026-01-27  
**Estado:** ✅ Parche aplicado
