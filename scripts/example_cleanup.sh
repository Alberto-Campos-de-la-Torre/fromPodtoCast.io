#!/bin/bash
# Ejemplo de uso del sistema de limpieza y recuperación
# Este script demuestra los casos de uso más comunes

set -e  # Salir si hay errores

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Ejemplo de Uso: Sistema de Limpieza y Recuperación${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

# Activar ambiente virtual
echo -e "${YELLOW}1. Activando ambiente virtual...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Ambiente virtual activado${NC}\n"

# =============================================================================
# EJEMPLO 1: Ver estado del pipeline
# =============================================================================
echo -e "${BLUE}═══ Ejemplo 1: Ver Estado del Pipeline ═══${NC}\n"
echo -e "${YELLOW}Comando:${NC} python scripts/cleanup_pipeline.py --show-status\n"

python scripts/cleanup_pipeline.py --show-status

echo -e "\n${GREEN}Este comando muestra:${NC}"
echo -e "  • Número de videos procesados y fallidos"
echo -e "  • Detalles del último video exitoso"
echo -e "  • Detalles del último fallo"
echo -e "  • Estado de directorios y archivos"
echo -e "  • Estado del caché LLM\n"

read -p "Presiona Enter para continuar..."
echo ""

# =============================================================================
# EJEMPLO 2: Limpieza básica (simulación)
# =============================================================================
echo -e "${BLUE}═══ Ejemplo 2: Limpieza Básica (Simulación) ═══${NC}\n"
echo -e "${YELLOW}Comando:${NC} python scripts/cleanup_pipeline.py --dry-run\n"

echo -e "${GREEN}Descripción:${NC}"
echo -e "  • Muestra qué archivos se limpiarían SIN eliminarlos"
echo -e "  • Útil para revisar antes de hacer limpieza real"
echo -e "  • No hace cambios en el sistema\n"

echo -e "${YELLOW}(Saltado en este ejemplo para evitar salida larga)${NC}\n"

read -p "Presiona Enter para continuar..."
echo ""

# =============================================================================
# EJEMPLO 3: Verificar procesos (sin matarlos)
# =============================================================================
echo -e "${BLUE}═══ Ejemplo 3: Verificar Procesos Colgados ═══${NC}\n"
echo -e "${YELLOW}Comando:${NC} python scripts/cleanup_pipeline.py --kill-processes --dry-run\n"

python scripts/cleanup_pipeline.py --kill-processes --dry-run 2>/dev/null || true

echo -e "\n${GREEN}Este comando muestra:${NC}"
echo -e "  • Procesos de ffmpeg que podrían estar colgados"
echo -e "  • Procesos de whisper activos"
echo -e "  • Procesos de python ejecutando scripts del pipeline"
echo -e "  • PIDs de cada proceso"
echo -e "  • SIN matar ningún proceso (--dry-run)\n"

read -p "Presiona Enter para continuar..."
echo ""

# =============================================================================
# EJEMPLO 4: Recuperación después de un fallo
# =============================================================================
echo -e "${BLUE}═══ Ejemplo 4: Analizar Opciones de Recuperación ═══${NC}\n"
echo -e "${YELLOW}Comando:${NC} python scripts/cleanup_pipeline.py --recover\n"

python scripts/cleanup_pipeline.py --recover

echo -e "\n${GREEN}Este comando:${NC}"
echo -e "  • Analiza el registro de videos procesados"
echo -e "  • Identifica videos que fallaron en procesamiento"
echo -e "  • Sugiere comandos específicos para recuperar"
echo -e "  • Muestra 3 opciones: retry, continue, process-only\n"

read -p "Presiona Enter para continuar..."
echo ""

# =============================================================================
# RESUMEN DE CASOS DE USO
# =============================================================================
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Resumen de Casos de Uso${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

echo -e "${GREEN}🔍 CASO 1: Pipeline se traba durante procesamiento${NC}"
echo -e "   1. Ctrl+C para interrumpir"
echo -e "   2. python scripts/cleanup_pipeline.py --kill-processes"
echo -e "   3. python scripts/auto_pipeline.py --retry-failed\n"

echo -e "${GREEN}🔍 CASO 2: Se queda sin memoria${NC}"
echo -e "   1. python scripts/cleanup_pipeline.py --kill-processes"
echo -e "   2. Revisar config/config.json (reducir batch_size)"
echo -e "   3. python scripts/auto_pipeline.py\n"

echo -e "${GREEN}🔍 CASO 3: Disco lleno${NC}"
echo -e "   1. python scripts/cleanup_pipeline.py --show-status"
echo -e "   2. python scripts/cleanup_pipeline.py --deep"
echo -e "   3. python scripts/auto_pipeline.py\n"

echo -e "${GREEN}🔍 CASO 4: ffmpeg/yt-dlp colgados${NC}"
echo -e "   1. python scripts/cleanup_pipeline.py --kill-processes --dry-run"
echo -e "   2. python scripts/cleanup_pipeline.py --kill-processes"
echo -e "   3. python scripts/auto_pipeline.py\n"

echo -e "${GREEN}🔍 CASO 5: Verificación periódica${NC}"
echo -e "   • python scripts/cleanup_pipeline.py --show-status"
echo -e "   • Revisar número de archivos en segments/"
echo -e "   • Revisar tamaño del caché LLM\n"

# =============================================================================
# COMANDOS ÚTILES ADICIONALES
# =============================================================================
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Comandos Útiles Adicionales${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

echo -e "${YELLOW}Ver espacio en disco:${NC}"
echo -e "   df -h /media/ttech-main/8464b39b-ba5b-49c5-9e05-010122b9874d\n"

echo -e "${YELLOW}Ver archivos temporales grandes:${NC}"
echo -e "   find /tmp -name '*whisper*' -o -name '*ffmpeg*' 2>/dev/null | xargs du -sh\n"

echo -e "${YELLOW}Ver memoria GPU:${NC}"
echo -e "   nvidia-smi\n"

echo -e "${YELLOW}Ver procesos de python:${NC}"
echo -e "   ps aux | grep python | grep -v grep\n"

echo -e "${YELLOW}Ver logs de un video específico:${NC}"
echo -e "   cat /ruta/a/data/logs/<video_id>.log | jq\n"

echo -e "${YELLOW}Backup del registro de videos:${NC}"
echo -e "   cp /ruta/a/data/processed_videos.json /ruta/a/data/processed_videos.backup.json\n"

echo -e "${GREEN}✓ Ejemplos completados${NC}\n"

echo -e "${BLUE}Para más información, consulta:${NC}"
echo -e "   • docs/MANUAL_LIMPIEZA.md"
echo -e "   • scripts/cleanup_pipeline.py --help\n"
