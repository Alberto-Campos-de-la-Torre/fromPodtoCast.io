# 📥 Descargar Videos para Procesamiento

El script `download_video.py` permite descargar videos desde URLs (YouTube, Vimeo, etc.) y extraer automáticamente el audio para procesamiento.

## 🚀 Uso Básico

### Descargar un video de YouTube

```bash
cd /home/ttech-main/fromPodtoCast
source venv/bin/activate

python3 scripts/download_video.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Descargar múltiples videos

```bash
python3 scripts/download_video.py \
  "https://www.youtube.com/watch?v=VIDEO1" \
  "https://www.youtube.com/watch?v=VIDEO2" \
  "https://vimeo.com/VIDEO3"
```

## 📋 Opciones Disponibles

### Formato de audio

```bash
# WAV (recomendado para procesamiento)
python3 scripts/download_video.py --format wav "URL"

# MP3
python3 scripts/download_video.py --format mp3 "URL"

# M4A
python3 scripts/download_video.py --format m4a "URL"
```

### Calidad de audio

```bash
# Mejor calidad (default)
python3 scripts/download_video.py --quality best "URL"

# Peor calidad (más rápido, menos espacio)
python3 scripts/download_video.py --quality worst "URL"
```

### Directorio de salida

```bash
python3 scripts/download_video.py -o ./data/input "URL"
```

### Descargar video completo (no solo audio)

```bash
python3 scripts/download_video.py --video "URL"
```

### Instalar yt-dlp automáticamente

```bash
python3 scripts/download_video.py --install-ytdlp "URL"
```

## 🔧 Instalación

### Opción 1: Instalación automática

```bash
cd /home/ttech-main/fromPodtoCast
source venv/bin/activate
pip install yt-dlp
```

### Opción 2: Durante la descarga

```bash
python3 scripts/download_video.py --install-ytdlp "URL"
```

## 📝 Ejemplo Completo

```bash
# 1. Activar entorno virtual
cd /home/ttech-main/fromPodtoCast
source venv/bin/activate

# 2. Descargar podcast de YouTube
python3 scripts/download_video.py \
  --format wav \
  --quality best \
  -o ./data/input \
  "https://www.youtube.com/watch?v=VIDEO_ID"

# 3. Procesar el audio descargado
python3 main.py ./data/input -o ./data/output
```

## 🎯 Flujo de Trabajo Recomendado

1. **Descargar videos:**
   ```bash
   python3 scripts/download_video.py "URL1" "URL2" "URL3"
   ```

2. **Verificar archivos descargados:**
   ```bash
   ls -lh ./data/input/
   ```

3. **Procesar todos los archivos:**
   ```bash
   python3 main.py ./data/input -o ./data/output
   ```

4. **Usar el metadata generado:**
   ```bash
   cat ./data/output/metadata.json
   ```

## ⚠️ Notas Importantes

- **Formato recomendado**: WAV para mejor calidad en procesamiento
- **Duración**: Los videos largos pueden tardar en descargarse
- **Espacio en disco**: Asegúrate de tener suficiente espacio
- **Términos de servicio**: Respeta los términos de servicio de las plataformas

## 🔗 Plataformas Soportadas

yt-dlp soporta más de 1000 plataformas, incluyendo:
- YouTube
- Vimeo
- Twitch
- SoundCloud
- Y muchas más

Ver la [lista completa](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)

## 🐛 Solución de Problemas

### Error: "yt-dlp no está instalado"

```bash
pip install yt-dlp
```

### Error: "ffmpeg not found"

```bash
sudo apt-get install ffmpeg
```

### Error: "Video unavailable" o "Private video"

- Verifica que el video sea público
- Algunos videos pueden tener restricciones geográficas
- Algunos videos pueden requerir autenticación

### Descarga muy lenta

- Usa `--quality worst` para descargas más rápidas
- Verifica tu conexión a internet
- Algunos servidores pueden estar sobrecargados

## 📚 Referencias

- [yt-dlp GitHub](https://github.com/yt-dlp/yt-dlp)
- [Documentación yt-dlp](https://github.com/yt-dlp/yt-dlp#usage-and-options)

