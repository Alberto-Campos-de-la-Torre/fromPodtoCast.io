# 🔧 Solución de Problemas con Ollama (Errores 500 y Timeouts)

## 📊 Diagnóstico del Problema

### Síntomas Detectados
Basado en los logs de tu sistema (2026-01-07):

```
[GIN] 2026/01/07 - 09:19:03 | 500 | 30.030159479s | POST "/api/generate"
[GIN] 2026/01/07 - 09:19:21 | 500 |  9.020783409s | POST "/api/generate"
time=2026-01-07T09:16:44 level=WARN msg="truncating input prompt" limit=4096 prompt=5010
time=2026-01-07T09:17:27 level=WARN msg="truncating input prompt" limit=4096 prompt=4561
time=2026-01-07T09:17:57 level=WARN msg="truncating input prompt" limit=4096 prompt=4999
```

### 🔴 Problemas Identificados

1. **Errores 500 (Internal Server Error)**
   - Ocurren después de 9-30 segundos
   - Indica que Ollama crashea internamente

2. **Truncamiento de Prompts**
   - Los prompts exceden **4096 tokens** (límite del modelo)
   - Prompts observados: **4532, 4561, 4999, 5010 tokens**
   - Se pierden **900-1000 tokens** de contexto

3. **Batch Size Agresivo**
   - Configuración actual: `batch_size: 5`
   - Env

iando 5 correcciones simultáneas sobrecarga Ollama

4. **Timeout Corto**
   - Configurado a **120 segundos**
   - Algunos requests toman 30-35 segundos cada uno
   - En batch de 5: **35s × 5 = 175s** > 120s timeout

## 🎯 Causas Raíz

### Causa #1: Prompts Demasiado Largos
El prompt de corrección LLM incluye:
- Instrucciones del sistema (~500 tokens)
- Glosario completo (~1000-2000 tokens si es grande)
- Contexto de verificación (~500 tokens)
- Texto a corregir (~1000-2000 tokens)
- **Total: 3000-5000+ tokens**

Cuando excede 4096, Ollama:
1. Trunca el prompt (pierde contexto importante)
2. Puede generar respuestas inválidas
3. Causa errores 500 al parsear respuestas

### Causa #2: Sobrecarga de Memoria GPU
- Ollama usa **12.5 GB** de la GPU 0 (RTX 5090)
- Modelo qwen3:14b cargado en memoria
- 5 requests simultáneos generan contexto masivo
- **Memoria insuficiente** → crashes → Error 500

### Causa #3: Procesamiento en Batch Agresivo
```python
# Config actual
{
  "batch_size": 5,        # 5 simultáneos
  "timeout": 120,         # Solo 2 minutos
  "max_retries": 3
}
```

Si cada corrección toma 20-30s:
- **Batch de 5**: 5 × 30s = 150s > 120s timeout ❌
- **Retries**: 3 × 150s = 450s de espera total

## ✅ Soluciones Recomendadas

### Solución #1: Reducir Batch Size (INMEDIATO)

**Antes:**
```json
{
  "batch_size": 5,
  "timeout": 120
}
```

**Después:**
```json
{
  "batch_size": 2,        # 2 simultáneos máximo
  "timeout": 180,         # 3 minutos
  "max_retries": 2        # Menos reintentos
}
```

**Comando rápido:**
```bash
# Editar config
nano config/config.json

# Cambiar batch_size de 5 a 2
# Cambiar timeout de 120 a 180
# Guardar (Ctrl+O, Enter, Ctrl+X)
```

### Solución #2: Optimizar Tamaño del Glosario

El glosario se incluye en **cada prompt**. Si es muy grande, lo trunca.

**Revisar tamaño:**
```bash
# Ver tamaño del glosario
cat data/diccionario_base.json | jq 'length'

# Ver tokens aproximados (500 chars = ~125 tokens)
wc -c data/diccionario_base.json
```

**Opciones:**

A) **Usar solo términos relevantes** (filtrar por frecuencia)
```bash
# Crear glosario reducido con top 200 términos
python scripts/expand_glosario.py --filter-top 200
```

B) **Dividir glosario por categorías**
```json
{
  "glosario_path": "./data/diccionario_medico.json",  # Solo términos médicos
  "glosario_business_path": "./data/diccionario_negocios.json"
}
```

C) **Enviar glosario solo en el system prompt**
```python
# Modificar text_corrector_llm.py
# Mover glosario a system message en vez de cada user message
```

### Solución #3: Aumentar Límite de Contexto de Ollama

Por defecto, qwen3:14b tiene límite de **4096 tokens**. Podemos aumentarlo:

```bash
# Crear Modelfile personalizado
cat > ~/qwen3-extended.Modelfile << 'EOF'
FROM qwen3:14b

# Aumentar contexto a 8192 tokens
PARAMETER num_ctx 8192

# Ajustar otros parámetros
PARAMETER temperature 0.1
PARAMETER top_p 0.95
EOF

# Crear modelo personalizado
ollama create qwen3-extended -f ~/qwen3-extended.Modelfile

# Actualizar config.json
# "model": "qwen3-extended"
```

**⚠️ NOTA:** Más contexto = más memoria GPU. Monitorea con `nvidia-smi`.

### Solución #4: Modo Secuencial en Vez de Batch

Para máxima estabilidad, deshabilitar batch processing:

```json
{
  "use_batch": false,       # Procesar uno por uno
  "use_parallel": false,    # Sin paralelismo
  "batch_size": 1,
  "timeout": 60             # Timeout más corto por request individual
}
```

**Ventajas:**
- ✅ Sin errores 500
- ✅ Memoria GPU estable
- ✅ Fácil de debuggear

**Desventajas:**
- ⏱️ Más lento (pero más confiable)

### Solución #5: Reiniciar Ollama Periódicamente

Ollama puede acumular **memory leaks** después de muchos requests.

**Script de reinicio automático:**
```bash
#!/bin/bash
# ~/restart_ollama.sh

echo "Reiniciando Ollama..."
sudo systemctl restart ollama

# Esperar a que esté listo
sleep 5

# Verificar
curl -s http://localhost:11434/api/tags | jq -r '.models[].name'

echo "Ollama reiniciado correctamente"
```

**Ejecutar antes de procesar batch largo:**
```bash
# Antes de auto_pipeline.py
bash ~/restart_ollama.sh

# Luego procesar
python scripts/auto_pipeline.py
```

### Solución #6: Monitoreo en Tiempo Real

Crear script para monitorear errores:

```bash
#!/bin/bash
# ~/monitor_ollama.sh

# Seguir logs en tiempo real
journalctl -u ollama -f --since "1 minute ago" | \
  grep -E "(ERROR|500|timeout|truncating)" --color=always
```

**Ejecutar en otra terminal:**
```bash
bash ~/monitor_ollama.sh
```

### Solución #7: Configuración avanzada de Ollama

Editar configuración del servicio systemd:

```bash
# Ver configuración actual
sudo systemctl cat ollama

# Editar
sudo systemctl edit ollama --full

# Agregar variables de entorno
[Service]
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_QUEUE=5"
Environment="OLLAMA_FLASH_ATTENTION=1"

# Guardar y reiniciar
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

## 🚀 Plan de Implementación Recomendado

### Paso 1: Cambios Inmediatos (5 minutos)

```bash
# 1. Editar configuración
nano config/config.json

# Cambiar:
# "batch_size": 5  →  "batch_size": 2
# "timeout": 120   →  "timeout": 180
# "max_retries": 3 →  "max_retries": 2

# 2. Reiniciar Ollama
sudo systemctl restart ollama

# 3. Probar con un video
python scripts/auto_pipeline.py --limit 1
```

### Paso 2: Optimización del Glosario (15 minutos)

```bash
# 1. Revisar tamaño
cat data/diccionario_base.json | jq 'length'

# Si > 300 términos:
# 2. Crear versión reducida
python scripts/expand_glosario.py --filter-top 200

# 3. Respaldo del original
cp data/diccionario_base.json data/diccionario_base_full.json

# 4. Usar versión reducida
# (editar config.json si es necesario)
```

### Paso 3: Modelo con Más Contexto (20 minutos)

```bash
# 1. Crear Modelfile
cat > ~/qwen3-extended.Modelfile << 'EOF'
FROM qwen3:14b
PARAMETER num_ctx 8192
PARAMETER temperature 0.1
PARAMETER top_p 0.95
EOF

# 2. Crear modelo
ollama create qwen3-extended -f ~/qwen3-extended.Modelfile

# 3. Actualizar config.json
# "model": "qwen3-extended"

# 4. Probar
python scripts/auto_pipeline.py --limit 1
```

### Paso 4: Monitoreo (5 minutos)

```bash
# Terminal 1: Pipeline
python scripts/auto_pipeline.py

# Terminal 2: Monitoreo
journalctl -u ollama -f | grep -E "(ERROR|500|truncating)" --color
```

## 📈 Configuración Recomendada Final

```json
{
  "llm_correction": {
    "enabled": true,
    "ollama_host": "http://localhost:11434",
    "model": "qwen3-extended",  // Modelo con más contexto
    "min_confidence": 0.7,
    "timeout": 180,              // 3 minutos
    "max_retries": 2,            // Menos reintentos
    "use_batch": true,
    "batch_size": 2,             // Solo 2 simultáneos
    "use_parallel": false,
    "max_workers": 1,
    "enable_cache": true,
    "cache_file": "/path/to/llm_cache.json",
    "cache_max_entries": 10000,
    "cache_expire_days": 30
  }
}
```

## 🔍 Cómo Verificar si Funciona

### Test 1: Sin Errores 500
```bash
# Procesar 1 video
python scripts/auto_pipeline.py --limit 1

# Revisar logs (no debe haber 500)
journalctl -u ollama -n 100 | grep -c "500"
# Debe retornar: 0
```

### Test 2: Sin Truncamiento
```bash
# Revisar logs (no debe truncar)
journalctl -u ollama -n 100 | grep -c "truncating"
# Debe retornar: 0
```

### Test 3: Memoria GPU Estable
```bash
# Antes de procesar
nvidia-smi

# Durante procesamiento (otra terminal)
watch -n 1 nvidia-smi

# La memoria debe mantenerse < 95% de la GPU
```

### Test 4: Tiempo de Respuesta
```bash
# Probar request individual
time curl -X POST http://localhost:11434/api/generate \
  -d '{"model":"qwen3:14b","prompt":"Hola","stream":false}'

# Debe completar en < 5 segundos
```

## 🆘 Troubleshooting Adicional

### Si Sigue Fallando: Modo Degradado
```json
{
  "llm_correction": {
    "enabled": false  // Deshabilitar temporalmente
  }
}
```

Luego procesar sin LLM:
```bash
python scripts/auto_pipeline.py
```

### Si Ollama No Responde
```bash
# 1. Matar todos los procesos
sudo pkill -9 ollama

# 2. Limpiar memoria GPU
python scripts/cleanup_pipeline.py

# 3. Reiniciar servicio
sudo systemctl restart ollama

# 4. Verificar
curl http://localhost:11434/api/tags
```

### Si GPU Queda sin Memoria
```bash
# Forzar limpieza
python -c "
import torch
import gc
gc.collect()
for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
"
```

## 📚 Referencias

- [Ollama Model Parameters](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
- [Qwen3 Context Window](https://qwenlm.github.io/blog/qwen3/)
- [GPU Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)

## 📝 Resumen de Cambios Recomendados

| Parámetro | Actual | Recomendado | Razón |
|-----------|--------|-------------|-------|
| `batch_size` | 5 | **2** | Reduce sobrecarga GPU |
| `timeout` | 120s | **180s** | Permite completar batch |
| `max_retries` | 3 | **2** | Menos espera en fallos |
| `model` | qwen3:14b | **qwen3-extended** | Más contexto (8192) |
| `use_parallel` | false | **false** | Mantener |
| Glosario | full | **top-200** | Reduce tokens |

---

**Última actualización**: 2026-01-07  
**Estado Ollama**: ✅ Activo (memoria GPU: 12.8GB/32.6GB en GPU 0)
