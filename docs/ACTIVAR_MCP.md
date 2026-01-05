# Guía Rápida: Activar MCP en Auto Pipeline

## ¿El MCP se activa con auto_pipeline?

**Respuesta:** SÍ, pero requiere configuración previa.

## Cómo Activar MCP

### Opción 1: Editar config.json (Recomendado)

Edita `config/config.json` y añade esta sección dentro de `llm_correction`:

```json
{
  "llm_correction": {
    "enabled": true,
    "model": "qwen3:14b",
    
    "mcp_verification": {
      "enabled": true,
      "model": "qwen3:14b",
      "dictionary_path": "./data/diccionario_base.json",
      "timeout": 60,
      "confidence_threshold": 0.80
    }
  }
}
```

### Opción 2: Copiar config_example_mcp.json

```bash
# Respaldar config actual
cp config/config.json config/config.json.backup

# Copiar configuración con MCP
cp config/config_example_mcp.json config/config.json

# Ajustar tu token de HuggingFace en config.json
```

## Ejecutar Auto Pipeline con MCP

Una vez configurado:

```bash
# Activar entorno virtual
source venv/bin/activate

# Ejecutar auto pipeline (usará MCP automáticamente)
python scripts/auto_pipeline.py --videos 10
```

## Verificar que MCP está Activo

Durante la ejecución verás estos mensajes:

```
✓ Corrector LLM inicializado (qwen3:14b, modo=batch, caché=ON, verificación=ON)
✓ Verificador MCP inicializado (modelo=qwen3:14b)

...

4.6. Corrigiendo transcripciones con LLM...
4.7. Verificando correcciones con MCP...  <-- Aquí confirmas que MCP está activo
   ✓ Verificados 42 textos con MCP
   ⚠️  Revertidos 3 textos (regionalismos/términos protegidos)
```

## Flujo Completo

```
Auto Pipeline → usa config/config.json
                    ↓
            Si llm_correction.mcp_verification.enabled = true
                    ↓
            Fase 1: LLM Corrección (qwen3:14b)
                    ↓
            Fase 2: MCP Verificación (diccionario + qwen3:14b)
                    ↓
            Texto final verificado
```

## Verificar Configuración Actual

```bash
# Ver si MCP está configurado en config.json
grep -A 5 "mcp_verification" config/config.json
```

Si no aparece nada, necesitas añadir la configuración.
