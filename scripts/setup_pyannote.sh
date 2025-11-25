#!/bin/bash
# Script para instalar y configurar pyannote.audio

echo "=========================================="
echo "Configuración de pyannote.audio"
echo "=========================================="
echo ""

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    echo "Activando entorno virtual..."
    source venv/bin/activate
fi

# Verificar si pyannote.audio está instalado
if python3 -c "import pyannote.audio" 2>/dev/null; then
    echo "✅ pyannote.audio ya está instalado"
    python3 -c "import pyannote.audio; print(f'Versión: {pyannote.audio.__version__}')" 2>/dev/null || echo "Versión no disponible"
else
    echo "📦 Instalando pyannote.audio..."
    pip install pyannote.audio
    
    if [ $? -eq 0 ]; then
        echo "✅ pyannote.audio instalado correctamente"
    else
        echo "❌ Error instalando pyannote.audio"
        echo "   Intenta manualmente: pip install pyannote.audio"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Configuración de Hugging Face"
echo "=========================================="
echo ""
echo "Para usar la diarización avanzada, necesitas:"
echo ""
echo "1. Crear una cuenta en https://huggingface.co"
echo "2. Aceptar los términos de uso en:"
echo "   https://huggingface.co/pyannote/speaker-diarization-3.1"
echo "   https://huggingface.co/pyannote/segmentation-3.0"
echo "   https://huggingface.co/pyannote/embedding"
echo ""
echo "3. Generar un token en:"
echo "   https://huggingface.co/settings/tokens"
echo ""
echo "4. Configurar el token en config/config.json:"
echo "   {"
echo "     \"use_diarization\": true,"
echo "     \"hf_token\": \"tu_token_aqui\""
echo "   }"
echo ""
echo "O exportar como variable de entorno:"
echo "   export HUGGINGFACE_TOKEN=tu_token_aqui"
echo ""





