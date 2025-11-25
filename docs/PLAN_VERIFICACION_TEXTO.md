# 📋 Plan Completo: Verificación y Corrección de Transcripciones

## 🎯 Objetivo

Implementar una fase de verificación de transcripciones en español para corregir errores de Whisper y preparar los datos en los formatos requeridos por Sesame CSM y Kyutai TTS.

---

## 📊 Formatos de Exportación Soportados

### 1. Sesame CSM (Conversational Speech Model)

```json
[
  {
    "text": "Transcripción corregida y limpia",
    "path": "/ruta/absoluta/al/audio.wav",
    "speaker": 0,
    "start": 0.0,
    "end": 10.5
  }
]
```

**Características:**
- Formato: JSON
- Sample rate: Flexible
- Multi-speaker: Sí (speaker ID entero)
- Timestamps: Opcionales

### 2. Kyutai TTS

```jsonl
{"audio_path": "/path/audio.wav", "text": "transcripción", "speaker_id": "spk_001", "duration": 10.5, "language": "es"}
```

**Características:**
- Formato: JSONL
- Sample rate: 24kHz (recomendado para Mimi codec)
- Duración: 1-30 segundos por segmento
- Voice cloning: Muestra de 10s por speaker
- Config adicional: YAML

---

## 🔍 Problemas Comunes de Whisper en Español

| Tipo | Ejemplo Error | Ejemplo Correcto |
|------|---------------|------------------|
| Homofonía | "haber" vs "a ver" | Contexto determina |
| Regionalismos | "güey" → "buey" | Mantener original |
| Marcas/Nombres | "Gemina" → "Gemini" | Corrección manual |
| Acentos | "como" vs "cómo" | Contexto sintáctico |
| Anglicismos | "marketing" → "márketing" | Normalizar |
| Puntuación | falta de "¿" "¡" | Añadir automático |
| Números | "5" vs "cinco" | Estandarizar |
| Muletillas | "este...", "eh..." | Eliminar o mantener |

---

## 🏗️ Arquitectura Propuesta

```
┌─────────────────────────────────────────────────────────────┐
│                    FASE DE VERIFICACIÓN                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Paso 1      │    │  Paso 2      │    │  Paso 3      │  │
│  │  Pre-proceso │───▶│  Corrección  │───▶│  Validación  │  │
│  │  Automático  │    │  LLM/Manual  │    │  Calidad     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ - Puntuación │    │ - GPT/Claude │    │ - Longitud   │  │
│  │ - Números    │    │ - Revisión   │    │ - Caracteres │  │
│  │ - Espacios   │    │   humana     │    │ - Coherencia │  │
│  │ - Mayúsculas │    │ - Glosario   │    │ - Audio-Text │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │      Exportación Multi-TTS    │
              ├───────────────────────────────┤
              │  • sesame_csm/train.json      │
              │  • kyutai_tts/train.jsonl     │
              └───────────────────────────────┘
```

---

## 📝 Plan de Implementación por Fases

### **Fase 1: Pre-procesamiento Automático**

**Archivo:** `src/text_preprocessor.py`

```python
class TextPreprocessor:
    """Correcciones automáticas basadas en reglas."""
    
    def preprocess(self, text: str) -> str:
        text = self._fix_spanish_punctuation(text)  # ¿? ¡!
        text = self._normalize_numbers(text)         # "5" → "cinco"
        text = self._fix_spacing(text)               # espacios dobles
        text = self._fix_capitalization(text)        # inicio oraciones
        text = self._remove_filler_words(text)       # "eh...", "este..."
        text = self._fix_common_errors(text)         # diccionario errores
        return text
```

**Reglas automáticas:**
1. Añadir `¿` al inicio de preguntas
2. Añadir `¡` al inicio de exclamaciones
3. Normalizar espacios y puntuación
4. Corregir errores comunes (diccionario)
5. Capitalización después de puntos

---

### **Fase 2: Corrección con LLM** (Opcional)

**Archivo:** `src/text_corrector_llm.py`

```python
class LLMTextCorrector:
    """Corrección usando modelo de lenguaje local o API."""
    
    def __init__(self, model: str = "local", glosario_path: str = None):
        self.model = model  # "local" (Ollama), "openai", "anthropic"
        self.glosario = self._load_glosario(glosario_path)
    
    def correct(self, text: str, audio_context: dict) -> str:
        prompt = f"""Corrige errores de transcripción en español.
        
        Texto original: {text}
        Contexto: Podcast sobre {audio_context.get('topic', 'desconocido')}
        
        Glosario de términos correctos:
        {self.glosario}
        
        Reglas:
        1. Mantener el significado original
        2. Corregir solo errores obvios
        3. Respetar regionalismos mexicanos
        4. NO parafrasear
        
        Texto corregido:"""
        
        return self._call_llm(prompt)
```

**Opciones de modelo:**
- **Local**: Ollama con Llama3/Mistral (sin costo)
- **API**: OpenAI GPT-4o-mini / Claude Haiku (bajo costo)
- **Híbrido**: Local para bulk, API para casos difíciles

---

### **Fase 3: Glosario de Términos**

**Archivo:** `config/glosario_terminos.json`

```json
{
  "correcciones": {
    "Gemina": "Gemini",
    "güey": "güey",
    "que es": "qué es",
    "por que": "por qué",
    "IA": "inteligencia artificial"
  },
  "marcas": [
    "Google", "YouTube", "TikTok", "Instagram", 
    "Gemini", "ChatGPT", "Meta"
  ],
  "mantener": [
    "güey", "chido", "neta", "chamba"
  ],
  "eliminar": [
    "este...", "eh...", "mmm..."
  ]
}
```

---

### **Fase 4: Validación de Calidad**

**Archivo:** `src/text_validator.py`

```python
class TextValidator:
    """Valida calidad de transcripciones."""
    
    def validate(self, entry: dict) -> Tuple[bool, List[str]]:
        issues = []
        
        # Validaciones
        if len(entry['text']) < 10:
            issues.append("texto_muy_corto")
        
        if not self._has_valid_punctuation(entry['text']):
            issues.append("puntuacion_incorrecta")
        
        if self._has_repetitions(entry['text']):
            issues.append("repeticiones_detectadas")
        
        if not self._text_audio_ratio_valid(entry):
            issues.append("ratio_texto_audio_invalido")
        
        return len(issues) == 0, issues
    
    def _text_audio_ratio_valid(self, entry: dict) -> bool:
        """Verifica que el texto tenga longitud coherente con la duración."""
        words = len(entry['text'].split())
        duration = entry.get('duration', 0)
        wpm = words / (duration / 60) if duration > 0 else 0
        # Español normal: 120-180 palabras por minuto
        return 80 <= wpm <= 220
```

---

### **Fase 5: Exportadores Multi-Formato**

**Archivo:** `src/tts_exporter.py`

#### Sesame CSM Exporter

```python
class SesameCMSExporter(TTSExporter):
    """Exportador para Sesame CSM."""
    
    def export(self, metadata: List[Dict], output_dir: str, 
               copy_audio: bool = False) -> Dict:
        # Convierte a formato CSM
        # Divide en train/val (90/10)
        # Guarda train.json y val.json
        pass
```

#### Kyutai TTS Exporter

```python
class KyutaiTTSExporter(TTSExporter):
    """Exportador para Kyutai TTS."""
    
    def export(self, metadata: List[Dict], output_dir: str,
               include_speaker_samples: bool = True) -> Dict:
        # Convierte a formato JSONL
        # Crea muestras de 10s por speaker
        # Genera config.yaml
        # Guarda train.jsonl y val.jsonl
        pass
```

#### Multi-Format Exporter

```python
class MultiFormatExporter:
    """Exporta a múltiples formatos TTS."""
    
    EXPORTERS = {
        'sesame_csm': SesameCMSExporter,
        'kyutai_tts': KyutaiTTSExporter
    }
    
    def export_all(self, metadata, output_dir) -> Dict:
        # Exporta a todos los formatos configurados
        pass
```

---

## 📁 Estructura de Archivos

```
fromPodtoCast/
├── src/
│   ├── text_preprocessor.py      # Correcciones automáticas
│   ├── text_corrector_llm.py     # Corrección con LLM
│   ├── text_validator.py         # Validación de calidad
│   └── tts_exporter.py           # Exportadores multi-formato
├── config/
│   ├── config.json               # + nuevos parámetros
│   └── glosario_terminos.json    # Diccionario correcciones
└── data/output/
    ├── metadata/                  # Formato interno
    └── tts_ready/                 # Formatos TTS listos
        ├── sesame_csm/
        │   ├── train.json
        │   ├── val.json
        │   └── audio/             # (opcional)
        └── kyutai_tts/
            ├── train.jsonl
            ├── val.jsonl
            ├── config.yaml
            └── speaker_samples/
                ├── spk_001_sample.wav
                └── spk_002_sample.wav
```

---

## ⚙️ Configuración Propuesta

**Añadir a `config.json`:**

```json
{
  "text_verification": {
    "enabled": true,
    "auto_preprocess": true,
    "use_llm_correction": false,
    "llm_provider": "local",
    "glosario_path": "./config/glosario_terminos.json",
    "min_words_per_segment": 5,
    "max_wpm": 220,
    "min_wpm": 80,
    "remove_fillers": true,
    "fix_punctuation": true
  },
  "tts_export": {
    "enabled": true,
    "formats": ["sesame_csm", "kyutai_tts"],
    "output_dir": "./data/output/tts_ready",
    "train_split": 0.9,
    "copy_audio": false,
    "sesame_csm": {
      "shuffle": true
    },
    "kyutai_tts": {
      "target_sample_rate": 24000,
      "min_duration": 1.0,
      "max_duration": 30.0,
      "include_speaker_samples": true
    }
  }
}
```

---

## 📊 Métricas de Verificación

El log de cada podcast incluirá:

```json
{
  "text_verification": {
    "total_segments": 150,
    "auto_corrected": 45,
    "llm_corrected": 12,
    "validation_passed": 142,
    "validation_failed": 8,
    "issues": {
      "texto_muy_corto": 3,
      "ratio_invalido": 5
    },
    "avg_wpm": 145.3
  },
  "tts_export": {
    "sesame_csm": {
      "train_entries": 128,
      "val_entries": 14
    },
    "kyutai_tts": {
      "train_entries": 125,
      "val_entries": 13,
      "speakers": ["spk_001", "spk_002"]
    }
  }
}
```

---

## 🚀 Flujo de Procesamiento Actualizado

```
Audio → Diarización → Segmentación → Normalización
                                          ↓
                                    Transcripción (Whisper)
                                          ↓
                                    Segunda Etapa (Pureza)
                                          ↓
                              ┌───────────────────────┐
                              │ NUEVA FASE            │
                              │ Verificación de Texto │
                              ├───────────────────────┤
                              │ 1. Pre-proceso auto   │
                              │ 2. Corrección LLM     │
                              │ 3. Validación         │
                              └───────────────────────┘
                                          ↓
                              ┌───────────────────────┐
                              │ Exportación Multi-TTS │
                              ├───────────────────────┤
                              │ • Sesame CSM          │
                              │ • Kyutai TTS          │
                              └───────────────────────┘
                                          ↓
                        Datos listos para fine-tuning TTS
```

---

## 📌 Prioridad de Implementación

| Prioridad | Componente | Complejidad | Impacto |
|-----------|-----------|-------------|---------|
| 🔴 Alta | TextPreprocessor (automático) | Baja | Alto |
| 🔴 Alta | TTSExporter (multi-formato) | Media | Alto |
| 🟡 Media | TextValidator | Media | Alto |
| 🟡 Media | Glosario de términos | Baja | Medio |
| 🟢 Baja | LLMTextCorrector | Alta | Medio |

---

## 📊 Comparación de Formatos TTS

| Característica | Sesame CSM | Kyutai TTS |
|----------------|------------|------------|
| **Formato archivo** | JSON | JSONL |
| **Sample rate** | Flexible | 24kHz (Mimi) |
| **Duración segmento** | Flexible | 1-30s |
| **Voice cloning** | No requerido | 10s sample/speaker |
| **Multi-speaker** | ✓ (speaker ID) | ✓ (speaker_id string) |
| **Timestamps** | Opcional | No usado |
| **Config adicional** | No | YAML |

---

## ❓ Decisiones Pendientes

1. **¿Usar LLM para corrección?**
   - Sí, pero primero hay que utilizar la version de reglas simples→ Mayor precisión, costo/latencia

2. **¿Revisión humana?**
   - Interfaz para revisar casos flaggeados

3. **¿Normalizar números?**
   - "5" → "cinco" (mejor para TTS)

4. **¿Eliminar muletillas?**
   - mantener para naturalidad

---

## 📅 Fecha de Creación

Noviembre 2025

