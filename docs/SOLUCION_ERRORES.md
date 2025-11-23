# 🔧 Solución de Errores Comunes

## ✅ Error Resuelto: pyannote.audio / torchvision conflict

**Error original:**
```
RuntimeError: operator torchvision::nms does not exist
```

**Solución aplicada:**
- `pyannote.audio` ahora es **opcional**
- El proyecto funciona sin diarización avanzada si hay conflictos
- Se usa método simple de diarización como fallback

**Estado:** ✅ **RESUELTO**

---

## 📋 Pasos para Instalar Dependencias

### Paso 1: Instalar dependencias básicas

```bash
cd /home/ttech-main/fromPodtoCast
pip install -r requirements-minimal.txt
```

O si prefieres instalación completa:
```bash
pip install -r requirements.txt
```

### Paso 2: Verificar instalación

```bash
python3 scripts/check_dependencies.py
```

Deberías ver:
```
✓ librosa
✓ soundfile
✓ pydub
✓ openai-whisper
✓ torch
✓ torchaudio
✓ tqdm
✓ numpy
```

### Paso 3: Ejecutar prueba

```bash
python3 scripts/test_example.py
```

---

## ⚠️ Si aún hay problemas

### Error: "No module named 'librosa'"
```bash
pip install librosa soundfile
```

### Error: "No module named 'whisper'"
```bash
pip install openai-whisper
```

### Error: "ffmpeg not found"
```bash
sudo apt-get install ffmpeg
```

### Error: Conflictos con torch/torchvision
- Usa `requirements-minimal.txt` en lugar de `requirements.txt`
- O crea un entorno virtual separado:
```bash
python3 -m venv venv_frompodtocast
source venv_frompodtocast/bin/activate
pip install -r requirements-minimal.txt
```

---

## 🎯 Nota sobre Diarización

La diarización de hablantes (identificación de diferentes narradores) es **opcional**:

- **Sin pyannote.audio**: El proyecto funciona, pero asigna el mismo speaker_id a todos los segmentos
- **Con pyannote.audio**: Identifica diferentes hablantes automáticamente (requiere token de Hugging Face)

Para usar diarización avanzada:
1. Instala pyannote.audio: `pip install pyannote.audio`
2. Obtén token de Hugging Face
3. Configura en `config/config.json`:
```json
{
  "use_diarization": true,
  "hf_token": "tu_token_aqui"
}
```

---

## ✅ Estado Actual

- ✅ Error de pyannote.audio resuelto (ahora es opcional)
- ⏳ Pendiente: Instalar dependencias básicas
- ⏳ Pendiente: Probar con archivo de audio real

---

## 🚀 Próximos Pasos

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements-minimal.txt
   ```

2. **Verificar que todo funciona:**
   ```bash
   python3 scripts/test_example.py
   ```

3. **Probar con un archivo de audio:**
   ```bash
   cp /home/ttech-main/csm/Local-csm/data/audio/audio20.wav data/input/
   python3 main.py data/input/audio20.wav -o data/output
   ```

