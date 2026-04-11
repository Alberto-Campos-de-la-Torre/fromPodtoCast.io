# Análisis Crítico: Calidad de Datos para Modelo Conversacional TTS

## Fecha: 2026-02-26

---

## 1. Modelo Objetivo: Moshiko (Kyutai) — Full-Duplex Conversacional

### 1.1 Arquitectura

Estamos re-entrenando (fine-tuning) el modelo **kyutai/moshiko-pytorch-bf16**, un modelo conversacional de voz basado en la arquitectura **Moshi** de Kyutai Labs. Sus características clave:

- **LMModel con Depformer**: 32 capas transformer principales + 6 capas depformer
- **Full-Duplex**: El modelo procesa audio del usuario y genera audio del sistema **simultáneamente**, no en turnos alternos. Esto simula una conversación natural donde ambos participantes pueden hablar al mismo tiempo
- **Cross-Attention**: El backbone transformer usa cross-attention entre los streams de texto y audio para alinear la generación de voz con el contenido textual. El decoder posterior genera codebooks de audio condicionado por las representaciones del backbone
- **Dimensión**: dim=4096, con vocabularios text_vocab=32000, audio_vocab=2049

### 1.2 Estructura de Tokens (17 Canales)

El modelo opera con **17 codebooks simultáneos** en cada frame temporal:

```
Canal  0       : Texto (Inner Monologue) — tokens de texto del hablante activo
Canales 1-8    : Audio del Sistema       — 8 codebooks Mimi del audio generado
Canales 9-16   : Audio del Usuario       — 8 codebooks Mimi del audio de entrada
```

Cada muestra de entrenamiento se construye como un tensor `[17, T]` donde `T` es la dimensión temporal:

```
Tiempo →  [  T_usuario  |  T_sistema  ]

Canal 0:  [ texto_user  | texto_system ]     ← Inner Monologue
Canales 1-8:  [ silencio    | audio_sistema  ]   ← Sistema habla SOLO aquí
Canales 9-16: [ audio_user  | silencio       ]   ← Usuario habla SOLO aquí
```

### 1.3 Tokenización del Audio (Mimi Codec)

El audio se tokeniza usando el codec **Mimi** de Kyutai:
- **8 codebooks** de audio (dep_q=8) por stream
- Cada codebook produce tokens con vocabulario de 2049
- La tasa de frames es ~12.5 Hz (80ms por frame)

### 1.4 Función de Pérdida

El entrenamiento usa una pérdida combinada:
```python
total_loss = audio_loss + 0.1 * text_loss
```

- **audio_loss**: Cross-entropy sobre los canales 1-8 (audio del sistema), enmascarado para calcular SOLO durante el turno del sistema
- **text_loss**: Cross-entropy auxiliar sobre el canal 0 (texto), ponderado al 10%
- **loss_mask**: Máscara booleana `[B, T]` que es `True` solo en las posiciones del turno del sistema

### 1.5 ¿Por qué la alineación audio↔texto es CRÍTICA?

En este modelo, el canal de texto (canal 0) funciona como **"inner monologue"** — es la representación textual de lo que el sistema está diciendo mientras genera audio. El mecanismo de cross-attention del backbone **alinea frame por frame** los tokens de texto con los tokens de audio.

**Si el texto dice "fracción de eyección disminuida" pero el audio dice "FEVI disminuida":**
- El canal 0 tendría 4 tokens de texto más que frames de audio disponibles
- La cross-attention intentaría alinear "fracción" con el fonema /fe/, "de" con /vi/, etc.
- El modelo aprendería una alineación **incorrecta** entre texto y fonemas
- La voz sintética producida sería incoherente — podría decir "FEVI" cuando el texto dice "fracción"

**Gravedad**: Este no es un error estético — es un error de entrenamiento que **corrompe los pesos del modelo**.

---

## 2. Problemas Identificados en los Datos

### 2.1 Correcciones LLM que rompen alineación audio↔texto

El corrector LLM (qwen3-32b/30b) con el prompt anterior aplicó correcciones que **alteran lo que se dijo**, destruyendo la correspondencia 1:1 entre audio y texto:

| Tipo de error | Ejemplo | Impacto en tokens [17, T] |
|---------------|---------|---------------------------|
| **Expansión de abreviaturas** | "FEVI disminuida" → "fracción de eyección disminuida" | Canal 0 tiene más tokens de texto que frames de audio en canales 1-8 |
| **Sinonimización** | "menos caro" → "más barato" | Fonemas de canal 0 no corresponden a canales 1-8 |
| **Adición de palabras** | "llegaba" → "llegaba otro" | Token "otro" en canal 0 sin correspondencia en audio |
| **Reordenamiento** | "les quiero hacer" → "quiero hacerles" | Cross-attention aprende alineación temporal incorrecta |
| **Correcciones fantasma** | 48-76% de "correcciones" sin cambio real | Desperdicio de procesamiento |

**Alcance del daño**: ~42+ archivos ya reprocesados con el prompt anterior. Las correcciones malas están mezcladas con correcciones buenas (tildes, puntuación) dentro de los mismos archivos.

### 2.2 Videos de un solo hablante

El modelo Moshi es **full-duplex y conversacional** — requiere datos con **≥2 hablantes** para construir la estructura de canales:
- Canales 1-8 = audio del sistema (hablante B)
- Canales 9-16 = audio del usuario (hablante A)
- Canal 0 = texto del hablante activo

Un video con **un solo hablante** (conferencia, monólogo, clase) produce datos donde:
- Solo un stream de audio tiene contenido
- No hay turnos de conversación
- No se puede generar la alternancia `T_usuario | T_sistema`

Categorías probablemente mono-hablante e inútiles:
- Conferencias médicas (ENARM, diplomados)
- Clases/tutoriales
- Monólogos informativos
- Documentales narrados

### 2.3 Re-verificación sin audio: IMPOSIBLE

Las correcciones malas del LLM son **español gramaticalmente correcto** — ningún modelo de texto puede detectarlas como errores sin acceso al audio original:
- "fracción de eyección disminuida" es texto perfecto
- Solo escuchando el audio se puede saber que el hablante dijo "FEVI disminuida"
- Un LLM de re-verificación simplemente confirmaría la corrección incorrecta

---

## 3. Propuestas de Solución

### Propuesta A: Revertir + re-corregir (RECOMENDADA — acción inmediata)

**Concepto**: Los `text_original` (transcripción Whisper pura) están preservados en la metadata. Revertir las correcciones LLM malas y re-aplicar con el prompt estricto nuevo.

**Pasos**:
1. **Identificar archivos contaminados**: los que tienen `text_original` en sus entries
2. **Revertir**: `text = text_original`, borrar `llm_correction`
3. **Filtrar multi-hablante**: verificar ≥2 speakers en diarización
4. **Re-corregir**: con el prompt estricto que preserva alineación audio↔texto

**Script conceptual**:
```python
for archivo in metadata_contaminada:
    for entry in archivo.entries:
        if 'text_original' in entry:
            entry['text'] = entry['text_original']  # Restaurar Whisper
            del entry['text_original']
            del entry['llm_correction']
    # Re-corregir con prompt estricto
    corrector.correct_batch(archivo, prompt=PROMPT_ESTRICTO_TTS)
```

**Ventaja**: No requiere re-transcribir audio con Whisper (lo más costoso). Solo deshace las correcciones LLM malas.
**Costo**: ~2-4 horas LLM.

### Propuesta B: Re-transcripción completa desde audio

**Concepto**: Descartar toda metadata y re-transcribir desde cero.

**Pasos**:
1. Whisper + diarización fresca
2. Filtrar ≥2 speakers
3. Corregir con prompt estricto

**Ventaja**: Datos 100% limpios.
**Desventaja**: ~24-48h de Whisper large-v3 (GPU intensivo).

### Propuesta C: Validación fonética con forced alignment

**Concepto**: Usar whisperx/forced alignment para detectar desalineaciones automáticamente comparando el audio real con el texto corregido.

**Pasos**:
1. Para cada segmento, ejecutar forced alignment sobre el audio
2. Comparar palabras detectadas vs texto corregido
3. Si hay palabras en el texto sin correspondencia en audio → reverter esa corrección
4. Re-corregir las revertidas con prompt estricto

**Ventaja**: Detección precisa sin re-transcribir todo.
**Desventaja**: Requiere whisperx, ~8-12h procesamiento.

### Propuesta D: Descarte selectivo + foco en nuevos datos conversacionales

**Concepto**: Aceptar la contaminación de los ~42 archivos, descartarlos, y enfocarse en los ~224 pendientes + nuevas descargas con criterios estrictos.

**Pasos**:
1. Descartar los 42 archivos ya corregidos con prompt viejo
2. Mantener solo archivos sin correcciones (pendientes) + nuevos del autopipeline
3. Filtrar por ≥2 speakers
4. Corregir solo estos con prompt estricto
5. Buscar activamente entrevistas/debates con 2+ personas claras

**Ventaja**: Data limpia garantizada.
**Desventaja**: Se pierde el trabajo de procesamiento previo.

---

## 4. Comparativa de Propuestas

| Propuesta | Calidad datos | Tiempo | Riesgo | Datos recuperados |
|-----------|--------------|--------|--------|-------------------|
| **A: Revertir + re-corregir** | ⭐⭐⭐⭐ Alta | ~2-4h | Bajo | ~42 archivos |
| **B: Re-transcribir todo** | ⭐⭐⭐⭐⭐ Máxima | ~24-48h | Nulo | Todos |
| **C: Forced alignment** | ⭐⭐⭐⭐ Alta | ~8-12h | Medio | ~42 archivos |
| **D: Descarte + nuevos** | ⭐⭐⭐⭐⭐ Máxima | ~4-8h | Bajo | Solo nuevos |

---

## 5. Recomendación

**Propuesta A como acción inmediata** + **filtrado multi-hablante** como proceso paralelo.

### Justificación técnica:
1. Los `text_original` preservan la transcripción Whisper original — la fuente de verdad fonética
2. El prompt estricto YA está implementado (SYSTEM_PROMPT + BATCH_SYSTEM_PROMPT + Modelfiles Ollama)
3. Solo se necesita un script de reversión → es la opción de menor costo con mayor recuperación
4. El filtrado de multi-hablante elimina automáticamente los videos mono-hablante

### Pasos concretos:
1. ✅ Prompt estricto implementado (preserva alineación audio↔texto)
2. ✅ Modelfiles de Ollama actualizados (32b local + 30b remoto)
3. 🔲 Script de reversión: restaurar `text = text_original` en archivos ya corregidos
4. 🔲 Filtro multi-hablante: identificar archivos con ≥2 speakers en diarización
5. 🔲 Re-corregir solo multi-hablante válidos con prompt estricto
6. 🔲 Validar muestra: verificar que `len(words_corregido) == len(words_original)` ±1

---

## 6. Formato de Datos Requerido para Tokenización Moshi

Para que los datos lleguen al tokenizador que prepara tensores `[17, T]`, cada muestra necesita:

```json
{
  "conversation_id": "video_xyz",
  "turn": {
    "speaker_user": {
      "text": "texto exacto del usuario (lo que DIJO, no lo que debería haber dicho)",
      "audio_path": "segmento_audio_user.wav",
      "text_tokens": [124, 5621, 8903, ...],
      "audio_codes": [[tok1_cb1, ...], [tok1_cb2, ...], ...]
    },
    "speaker_system": {
      "text": "texto exacto del sistema (lo que DIJO)",
      "audio_path": "segmento_audio_system.wav",
      "text_tokens": [7821, 331, 9102, ...],
      "audio_codes": [[tok1_cb1, ...], [tok1_cb2, ...], ...]
    }
  }
}
```

**Regla de oro**: Si el canal 0 (texto) tiene el token para "fracción" pero el canal 1-8 (audio) tiene el fonema /FE-vi/, la cross-attention del backbone aprenderá que "fracción" suena como "FEVI" — **corrompiendo permanentemente la generación de voz del modelo**.
