# Voice Bank y Reutilización de Hablantes

Este módulo permite que la diarización reconozca y reutilice voces ya vistas en podcasts anteriores, reduciendo latencia y evitando reasignar IDs locales.

## Flujo Interno

1. **Diarización global**: Pyannote detecta segmentos y hablantes.  
2. **Extracción de embeddings**: Para cada hablante se calcula un embedding de alta dimensionalidad con `pyannote/embedding`.  
3. **Voice Bank (`voice_bank.json`)**:
   - Se normaliza el embedding y se busca el speaker más parecido usando similitud coseno.
   - Si `cos_sim ≥ voice_match_threshold` se reutiliza el ID global y se actualiza su embedding promedio.
   - Si no, se crea un nuevo ID `SPEAKER_GLOBAL_xxx` y se almacena junto al embedding.
4. **Propagación**: Los nuevos IDs globales sustituyen las etiquetas locales antes de que el pipeline genere metadata o ejecute la segunda etapa.
5. **Persistencia**: El banco se escribe en `data/output/voice_bank.json` (configurable) y se enriquece cada vez que aparece un hablante nuevo.

## Configuración

En `config/config.json`:

```json
{
  "use_voice_bank": true,
  "voice_bank_path": "./data/output/voice_bank.json",
  "voice_match_threshold": 0.85
}
```

- `use_voice_bank`: activa/desactiva la funcionalidad.
- `voice_match_threshold (τ)`: umbral de similitud coseno para considerar una voz existente.
- Se requiere `hf_token` válido para cargar `pyannote/embedding`.

## Métricas

En el log por podcast (`data/output/logs/<id>.log`) se registran:
- `voice_bank.enabled`
- `voice_bank.matched`: hablantes reconocidos
- `voice_bank.created`: nuevos hablantes añadidos

## Casos de uso

- **Entrenamientos multi-podcast**: evita volver a etiquetar voces recurrentes.
- **Monitor de speakers**: extrae estadísticas de aparición usando `voice_bank.json`.

## Limitaciones y fallback

- Si `pyannote/embedding` no está disponible o falta `hf_token`, el Voice Bank se desactiva automáticamente y el pipeline continúa con IDs locales.
- El método simple de diarización (fallback sin pyannote) no genera embeddings; en ese caso el banco tampoco se usa.



