#!/bin/bash
# Ejemplos de uso del script reprocess_metadata.py

# ============================================================================
# EJEMPLOS DE USO
# ============================================================================

# 1. Re-procesar un archivo de metadata específico
python scripts/reprocess_metadata.py data/output/metadata/mi_podcast.json

# 2. Re-procesar con configuración personalizada
python scripts/reprocess_metadata.py data/output/metadata/mi_podcast.json --config config/config.json

# 3. Re-procesar sin crear backup (no recomendado)
python scripts/reprocess_metadata.py data/output/metadata/mi_podcast.json --no-backup

# 4. Re-procesar TODOS los archivos de metadata en un directorio
python scripts/reprocess_metadata.py --all --data-dir data/output

# 5. Re-procesar todos con configuración personalizada
python scripts/reprocess_metadata.py --all --data-dir data/output --config config/config.json

# ============================================================================
# CONFIGURACIÓN PERSONALIZADA
# ============================================================================

# Puedes crear un archivo de configuración JSON temporal para re-procesamiento:
cat > /tmp/reprocess_config.json << 'EOF'
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
    "batch_size": 10,
    "min_confidence": 0.75,
    "enable_cache": true,
    "mcp_verification": {
      "enabled": true,
      "model": "qwen3:14b",
      "dictionary_path": "data/diccionario_base.json",
      "timeout": 60,
      "confidence_threshold": 0.85
    }
  }
}
EOF

# Usar la configuración temporal
python scripts/reprocess_metadata.py --all --data-dir data/output --config /tmp/reprocess_config.json

# ============================================================================
# CASOS DE USO COMUNES
# ============================================================================

# Caso 1: Cambiaste el modelo LLM y quieres re-procesar todo
# - Edita config/config.json para usar el nuevo modelo
# - Ejecuta:
python scripts/reprocess_metadata.py --all --data-dir /media/ttech-main/8464b39b-ba5b-49c5-9e05-010122b9874d

# Caso 2: Actualizaste el diccionario MCP y quieres re-verificar
# - Asegúrate que el diccionario esté actualizado en data/diccionario_base.json
# - Ejecuta:
python scripts/reprocess_metadata.py --all --data-dir data/output

# Caso 3: Corregir errores en un podcast específico
# - Identifica el archivo de metadata del podcast
# - Ejecuta:
python scripts/reprocess_metadata.py data/output/metadata/nombre_del_podcast.json

# Caso 4: Re-procesar solo con modelo diferente (sin cambiar config principal)
cat > /tmp/modelo_mejorado.json << 'EOF'
{
  "llm_correction": {
    "enabled": true,
    "model": "llama3:70b",
    "batch_size": 3,
    "mcp_verification": {
      "enabled": true,
      "model": "llama3:70b"
    }
  }
}
EOF
python scripts/reprocess_metadata.py --all --data-dir data/output --config /tmp/modelo_mejorado.json

# ============================================================================
# NOTAS IMPORTANTES
# ============================================================================

# 1. El script crea backups automáticos con timestamp:
#    archivo.json -> archivo.json.backup_20260105_073000
#
# 2. Los backups se crean en el mismo directorio que el archivo original
#
# 3. El script preserva la estructura original del JSON:
#    - Si es una lista, sigue siendo una lista
#    - Si es un dict con 'entries', mantiene esa estructura
#
# 4. Los textos originales se preservan en:
#    - Campo 'text_original' si no existía corrección previa
#    - Campo 'llm_correction.original' si ya había corrección
#
# 5. El script usa procesamiento batch para máxima eficiencia
#
# 6. Los timestamps de re-procesamiento se guardan en:
#    - llm_correction.reprocessed_at
#    - llm_correction.mcp_verified_at
