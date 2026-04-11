#!/bin/bash
# Script para aplicar fix de configuración Ollama
# Soluciona errores 500 y timeouts

set -e

echo "═══════════════════════════════════════════════════════════"
echo "  🔧 Aplicando Fix de Configuración Ollama"
echo "═══════════════════════════════════════════════════════════"
echo ""

CONFIG_FILE="config/config.json"

# Verificar que existe el archivo
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: No se encontró $CONFIG_FILE"
    exit 1
fi

# Hacer backup
BACKUP_FILE="config/config.json.backup.$(date +%Y%m%d_%H%M%S)"
echo "📦 Creando backup: $BACKUP_FILE"
cp "$CONFIG_FILE" "$BACKUP_FILE"

# Mostrar valores actuales
echo ""
echo "Valores ACTUALES:"
echo "  batch_size:  $(cat $CONFIG_FILE | jq '.llm_correction.batch_size')"
echo "  timeout:     $(cat $CONFIG_FILE | jq '.llm_correction.timeout')"
echo "  max_retries: $(cat $CONFIG_FILE | jq '.llm_correction.max_retries')"
echo ""

# Aplicar cambios
echo "Aplicando NUEVOS valores:"
echo "  batch_size:  5 → 2  (reduce sobrecarga GPU)"
echo "  timeout:     120 → 180 (permite completar batch)"
echo "  max_retries: 3 → 2  (menos espera en fallos)"
echo ""

# Usar jq para modificar el JSON
jq '.llm_correction.batch_size = 2 | 
    .llm_correction.timeout = 180 | 
    .llm_correction.max_retries = 2' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp"

# Reemplazar el archivo original
mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"

# Verificar cambios
echo "✅ Cambios aplicados:"
echo "  batch_size:  $(cat $CONFIG_FILE | jq '.llm_correction.batch_size')"
echo "  timeout:     $(cat $CONFIG_FILE | jq '.llm_correction.timeout')"
echo "  max_retries: $(cat $CONFIG_FILE | jq '.llm_correction.max_retries')"
echo ""

# Reiniciar Ollama
echo "🔄 ¿Deseas reiniciar Ollama ahora? [s/N]"
read -t 10 -n 1 answer || answer="n"
echo ""

if [[ "$answer" == "s" ]] || [[ "$answer" == "S" ]]; then
    echo "Reiniciando Ollama..."
    sudo systemctl restart ollama
    sleep 3
    
    # Verificar estado
    if systemctl is-active --quiet ollama; then
        echo "✅ Ollama reiniciado correctamente"
        
        # Verificar conectividad
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✅ Ollama responde correctamente"
        else
            echo "⚠️  Ollama no responde, esperando 5s más..."
            sleep 5
            if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
                echo "✅ Ollama responde correctamente"
            else
                echo "❌ Ollama no responde. Revisar con: sudo systemctl status ollama"
            fi
        fi
    else
        echo "❌ Error reiniciando Ollama. Revisar con: sudo systemctl status ollama"
    fi
else
    echo "⏭️  Reinicio omitido. Reinicia manualmente con: sudo systemctl restart ollama"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ Configuración Actualizada"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "📝 Backup guardado en: $BACKUP_FILE"
echo ""
echo "🧪 Probar con:"
echo "   python scripts/auto_pipeline.py --limit 1"
echo ""
echo "📊 Monitorear errores:"
echo "   journalctl -u ollama -f | grep -E '(ERROR|500|truncating)'"
echo ""
