"""
Módulo para corrección avanzada de transcripciones usando LLM local (Ollama).
Utiliza el modelo qwen3:8b para correcciones contextuales de alta calidad.

Optimizado con:
- Procesamiento por lotes (batching)
- Validación Pydantic
- Caché de correcciones
- Paralelización opcional
"""
import json
import re
import hashlib
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime

# Barras de progreso
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from pydantic import ValidationError
    from .models.llm_schemas import (
        LLMCorrectionResponse,
        LLMCorrectionBatchResponse,
        LLMCorrectionBatchItem,
        LLMCorrectionMetadata,
        CacheEntry
    )
    PYDANTIC_AVAILABLE = True
except ImportError:
    try:
        from pydantic import ValidationError
        from models.llm_schemas import (
            LLMCorrectionResponse,
            LLMCorrectionBatchResponse,
            LLMCorrectionBatchItem,
            LLMCorrectionMetadata,
            CacheEntry
        )
        PYDANTIC_AVAILABLE = True
    except ImportError:
        PYDANTIC_AVAILABLE = False


# Verificador de transcripciones
try:
    from .transcription_verifier import TranscriptionVerifier, VerificationResult
    VERIFIER_AVAILABLE = True
except ImportError:
    try:
        from transcription_verifier import TranscriptionVerifier, VerificationResult
        VERIFIER_AVAILABLE = True
    except ImportError:
        VERIFIER_AVAILABLE = False
        TranscriptionVerifier = None
        VerificationResult = None


class TextCorrectorLLM:
    """
    Corrector de texto usando LLM local via Ollama.
    
    Características:
    - Corrección contextual de errores de transcripción
    - Preservación de regionalismos y expresiones coloquiales
    - Puntuación y acentuación correcta
    - Formato de salida estructurado JSON
    - Procesamiento por lotes optimizado
    - Caché de correcciones
    - Verificación anti-alucinación
    """
    
    # Master prompt para el modelo (individual) - OPTIMIZADO CON FORZADO DE IDIOMA
    SYSTEM_PROMPT = """Eres un corrector experto de transcripciones de audio en español con 20 años de experiencia en podcasts.

## TU ROL
Actúas como un editor profesional especializado en transcripciones automáticas de Whisper para podcasts EN ESPAÑOL.

## ⚠️ IDIOMA OBJETIVO: ESPAÑOL (SIEMPRE)
- El idioma de salida SIEMPRE debe ser ESPAÑOL
- El transcriptor a veces detecta erróneamente español como catalán (ca) o gallego (gl)
- Si ves texto que parece catalán/gallego pero el contexto sugiere español, NORMALIZA AL ESPAÑOL
- Ejemplos: "És per la corazon" → "Es por el corazón", "hagam" → "hagamos"

## TAREA PRINCIPAL
Corregir ÚNICAMENTE errores de transcripción, preservando el contenido exacto del hablante.

## ⚠️ REGLA CRÍTICA: NO AGREGAR NI ELIMINAR CONTENIDO
- NUNCA agregues información que no está en el texto original
- NUNCA elimines palabras o frases del original
- NUNCA cambies el significado o intención del hablante
- NUNCA parafrasees o resumas el contenido
- El texto corregido DEBE tener aproximadamente la misma longitud que el original

## CORRECCIONES PERMITIDAS (SOLO ESTAS)

### ✅ CORREGIR:
1. **Ortografía**: tildes (qué, cómo, más), letras incorrectas
2. **Puntuación**: agregar ¿?, ¡!, comas, puntos donde faltan
3. **Mayúsculas**: nombres propios, inicio de oración
4. **Nombres de marcas**: YouTube, TikTok, Instagram, ChatGPT, Google, WhatsApp
5. **Acrónimos**: IA, SEO, API, URL, PDF, CEO, NFT
6. **Errores comunes de Whisper**:
   - "gemina" → "Gemini"
   - "chat gpt" → "ChatGPT"
   - "ai" → "IA"
   - "que es" (al inicio) → "¿Qué es"
   - "por que" (pregunta) → "por qué"
7. **Texto mal transcrito como catalán/gallego**: normalizar al español correcto

### ❌ NO CORREGIR (MANTENER TAL CUAL):
- Regionalismos: güey, chido, neta, órale, chamba, morro, chale, fresa
- Muletillas: pues, este, o sea, bueno, ¿no?, ¿verdad?
- Expresiones coloquiales: no manches, qué onda, está cañón, ni modo
- Estilo informal del hablante
- Repeticiones intencionales
- Pausas o titubeos representados
- NO traducir español a catalán/gallego

## GLOSARIO ESPECÍFICO
{glosario_context}

## FORMATO DE RESPUESTA (JSON ESTRICTO)
Responde ÚNICAMENTE con este JSON, sin texto antes ni después:
{{
  "texto_corregido": "El texto corregido completo",
  "cambios": ["cambio1", "cambio2"],
  "confianza": 0.95
}}

### Reglas del JSON:
- `texto_corregido`: Texto final (misma longitud aproximada del original)
- `cambios`: Lista de hasta 5 correcciones principales aplicadas
- `confianza`: Número entre 0.0 y 1.0

### Si el texto está correcto:
{{
  "texto_corregido": "[mismo texto sin cambios]",
  "cambios": [],
  "confianza": 0.99
}}

RESPONDE SOLO JSON. NO EXPLIQUES."""

    # Prompt para batch processing - OPTIMIZADO CON FORZADO DE IDIOMA
    BATCH_SYSTEM_PROMPT = """Eres un corrector experto de transcripciones de audio en español.

## ROL
Editor profesional especializado en corrección de transcripciones automáticas de podcasts EN ESPAÑOL.

## ⚠️ IDIOMA OBJETIVO: ESPAÑOL (SIEMPRE)
- El idioma de salida SIEMPRE debe ser ESPAÑOL
- El transcriptor a veces detecta erróneamente español como catalán (ca) o gallego (gl)
- Si ves texto que parece catalán/gallego pero el contexto sugiere español, NORMALIZA AL ESPAÑOL
- Usa el CONTEXTO del lote (todos los textos juntos son una conversación) para inferir el idioma real
- Ejemplos de errores comunes de transcripción a corregir:
  * "És per la corazon" → "Es por el corazón" (catalán falso → español)
  * "Estic deseant" → "Estoy deseando" (catalán falso → español)
  * "hagam" → "hagamos" (NO traducir al catalán)
  * "nutrició" → "nutrición" (NO traducir al catalán)

## ⚠️ REGLA CRÍTICA: PRESERVAR EL CONTENIDO ORIGINAL
- NUNCA agregues información nueva
- NUNCA elimines contenido del original  
- NUNCA cambies el significado
- La longitud de cada texto corregido debe ser similar al original
- Solo corrige errores de ortografía, puntuación y formato
- Si el texto está muy corrupto (parece otro idioma), intenta reconstruir el español original

## CORRECCIONES PERMITIDAS

### ✅ CORREGIR:
- Tildes y ortografía
- Puntuación (¿?, ¡!, comas, puntos)
- Mayúsculas (nombres propios, inicio de oración)
- Marcas: YouTube, TikTok, Instagram, ChatGPT, Google
- Acrónimos: IA, SEO, API, URL
- Texto erróneamente transcrito como catalán/gallego → español correcto

### ❌ NO CORREGIR:
- Regionalismos mexicanos (güey, chido, neta, órale)
- Muletillas naturales (pues, este, o sea)
- Expresiones coloquiales
- NO traducir español a catalán/gallego (error común del transcriptor)

## GLOSARIO
{glosario_context}

## EJEMPLOS DE CORRECCIÓN CORRECTA

Entrada 0: "que es el marketing digital y por que es importante"
Salida 0: {{"id": 0, "texto_corregido": "¿Qué es el marketing digital y por qué es importante?", "cambios": ["Añadido ¿", "qué con tilde", "por qué separado"], "confianza": 0.95}}

Entrada 1: "vamos a hablar de chat gpt y de youtube"
Salida 1: {{"id": 1, "texto_corregido": "Vamos a hablar de ChatGPT y de YouTube.", "cambios": ["ChatGPT", "YouTube", "punto final"], "confianza": 0.92}}

Entrada 2: "pues si guey esta bien chido el podcast"
Salida 2: {{"id": 2, "texto_corregido": "Pues sí güey, está bien chido el podcast.", "cambios": ["sí con tilde", "Mayúscula inicial", "coma después de güey"], "confianza": 0.90}}

Entrada 3: "És per la corazon que et canviat"
Salida 3: {{"id": 3, "texto_corregido": "Es por el corazón que te ha cambiado.", "cambios": ["normalizado de catalán falso a español"], "confianza": 0.85}}

## FORMATO DE RESPUESTA (CRÍTICO)
Responde ÚNICAMENTE con JSON válido. NINGÚN texto antes ni después.

Estructura EXACTA:
{{
  "correcciones": [
    {{"id": 0, "texto_corregido": "texto1", "cambios": ["cambio1"], "confianza": 0.95}},
    {{"id": 1, "texto_corregido": "texto2", "cambios": [], "confianza": 0.98}}
  ]
}}

## REGLAS JSON
- Usa COMA entre objetos del array (excepto el último)
- IDs consecutivos de 0 a N-1
- confianza es NÚMERO (0.0-1.0), NO string
- Solo comillas dobles
- Escapa comillas internas: \\"

RESPONDE SOLO JSON."""

    # Prompt para FASE 2: Reconstrucción de textos muy corruptos
    RECONSTRUCTION_PROMPT = """Eres un experto en reconstrucción de transcripciones corruptas de audio en español.

## CONTEXTO
El texto que recibirás proviene de una transcripción automática con MUCHOS ERRORES.
Whisper (el transcriptor) cometió errores graves que hacen el texto ininteligible.

## TU TAREA
RECONSTRUIR el significado probable del texto basándote en:
1. Las palabras reconocibles que encuentres
2. El contexto del podcast (tema general indicado)
3. Estructura gramatical española correcta

## REGLAS DE RECONSTRUCCIÓN
1. Usa SOLO las palabras reconocibles como base
2. Completa las partes corruptas con lo MÁS PROBABLE según el contexto
3. Si una palabra es irreconocible, intenta inferirla del sonido similar
4. Mantén la LONGITUD SIMILAR al original (±20%)
5. El resultado DEBE ser español gramaticalmente correcto
6. Marca con confianza BAJA (0.5-0.7) si tuviste que adivinar mucho

## GLOSARIO ESPECÍFICO
{glosario_context}

## EJEMPLOS DE RECONSTRUCCIÓN

Original corrupto: "Lo que recupere y que te hago una salud, que sea el tiempo por ti"
Reconstruido: "Lo que te quiero decir es que cuides tu salud, que sea prioridad para ti"
Confianza: 0.55

Original corrupto: "certilletas y esa efectividad tojanicas la hostIA"
Reconstruido: "certificaciones y esa efectividad, tomando en cuenta la historia"
Confianza: 0.60

## FORMATO DE RESPUESTA (JSON ESTRICTO)
{{
  "texto_reconstruido": "El texto reconstruido completo",
  "palabras_base": ["lista", "de", "palabras", "originales", "usadas"],
  "cambios_principales": ["cambio1", "cambio2"],
  "confianza": 0.55
}}

RESPONDE SOLO JSON. SIN EXPLICACIONES."""

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "qwen3:14b",
        glosario_path: Optional[str] = None,
        timeout: int = 600,
        max_retries: int = 4,
        batch_size: int = 4,
        enable_cache: bool = True,
        cache_file: Optional[str] = None,
        max_workers: int = 4,
        enable_verification: bool = True,
        verification_config: Optional[Dict] = None
    ):
        """
        Inicializa el corrector LLM.
        
        Args:
            ollama_host: URL del servidor Ollama
            model: Nombre del modelo a usar
            glosario_path: Ruta al archivo de glosario JSON
            timeout: Timeout para requests en segundos
            max_retries: Número máximo de reintentos
            batch_size: Tamaño del lote para procesamiento batch
            enable_cache: Habilitar caché de correcciones
            cache_file: Ruta al archivo de caché
            max_workers: Workers para paralelización
            enable_verification: Habilitar verificación de correcciones
            verification_config: Configuración del verificador (optional)
        """
        if not ollama_host.startswith(('http://', 'https://')):
            ollama_host = f"http://{ollama_host}"
        self.ollama_host = ollama_host.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # Cargar glosario
        self.glosario = self._load_glosario(glosario_path)
        self.glosario_context = self._format_glosario_context()
        
        # Cache
        self.enable_cache = enable_cache
        self.cache: Dict[str, Dict] = {}
        self.cache_file = cache_file
        if enable_cache and cache_file:
            self._load_cache()
        
        # Verificador de transcripciones
        self.enable_verification = enable_verification and VERIFIER_AVAILABLE
        self.verifier = None
        if self.enable_verification:
            v_config = verification_config or {}
            self.verifier = TranscriptionVerifier(
                min_length_ratio=v_config.get('min_length_ratio', 0.70),
                max_length_ratio=v_config.get('max_length_ratio', 1.40),
                min_word_preservation=v_config.get('min_word_preservation', 0.80),
                max_new_word_ratio=v_config.get('max_new_word_ratio', 0.25),
                min_sequence_similarity=v_config.get('min_sequence_similarity', 0.60),
                logger=self.logger
            )
            self.logger.info("✓ Verificador de transcripciones habilitado")
        
        # Estadísticas extendidas
        self.stats = {
            'processed': 0,
            'corrected': 0,
            'failed': 0,
            'avg_confidence': 0.0,
            'total_changes': 0,
            'cache_hits': 0,
            'batch_calls': 0,
            'individual_calls': 0,
            'pydantic_validations': 0,
            'verification_passed': 0,
            'verification_failed': 0,
            'verification_reverted': 0
        }
        
        # Dual-model (deshabilitado por defecto, se activa via configure_dual_model)
        self.dual_model_enabled = False

        # Verificar conexión
        self._verify_connection()
    
    def _load_glosario(self, path: Optional[str]) -> Dict:
        """Carga el glosario de términos."""
        default = {'correcciones': {}, 'mantener': []}
        
        if path and Path(path).exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Error cargando glosario: {e}")
        
        return default
    
    def _format_glosario_context(self) -> str:
        """Formatea el glosario como contexto para el prompt."""
        lines = []
        
        # Correcciones más relevantes (primeras 30)
        correcciones = self.glosario.get('correcciones', {})
        if correcciones:
            lines.append("### Correcciones obligatorias:")
            for i, (error, correccion) in enumerate(list(correcciones.items())[:30]):
                lines.append(f"  - \"{error}\" → \"{correccion}\"")
        
        # Términos a mantener (primeros 30)
        mantener = self.glosario.get('mantener', [])
        if mantener:
            lines.append("\n### Expresiones a MANTENER (no corregir):")
            lines.append(f"  {', '.join(mantener[:30])}")
        
        return '\n'.join(lines)
    
    def _verify_connection(self) -> bool:
        """Verifica la conexión con el servidor Ollama."""
        try:
            response = requests.get(
                f"{self.ollama_host}/api/tags",
                timeout=10
            )
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                if not any(self.model in name for name in model_names):
                    self.logger.warning(
                        f"⚠️  Modelo {self.model} no encontrado. "
                        f"Disponibles: {model_names}"
                    )
                else:
                    self.logger.info(f"✓ Conectado a Ollama ({self.ollama_host})")
                return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ No se puede conectar a Ollama: {e}")
            return False
        
        return False
    
    # ==================== CACHE ====================
    
    def _get_text_hash(self, text: str) -> str:
        """Genera un hash único para el texto."""
        return hashlib.md5(text.strip().lower().encode()).hexdigest()
    
    def _load_cache(self) -> None:
        """Carga el caché desde archivo."""
        if self.cache_file and Path(self.cache_file).exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                self.logger.info(f"✓ Caché cargado: {len(self.cache)} entradas")
            except Exception as e:
                self.logger.warning(f"Error cargando caché: {e}")
                self.cache = {}
    
    def _save_cache(self) -> None:
        """Guarda el caché a archivo."""
        if self.cache_file:
            try:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
            except Exception as e:
                self.logger.warning(f"Error guardando caché: {e}")
    
    def _get_from_cache(self, text: str) -> Optional[Dict]:
        """Busca un texto en el caché."""
        if not self.enable_cache:
            return None
        
        text_hash = self._get_text_hash(text)
        if text_hash in self.cache:
            self.stats['cache_hits'] += 1
            entry = self.cache[text_hash]
            entry['hits'] = entry.get('hits', 0) + 1
            return entry.get('response')
        return None
    
    def _add_to_cache(self, text: str, response: Dict) -> None:
        """Agrega una respuesta al caché."""
        if not self.enable_cache:
            return

        text_hash = self._get_text_hash(text)
        self.cache[text_hash] = {
            'response': response,
            'created_at': datetime.now().isoformat(),
            'hits': 0
        }

    # ==================== DETECCIÓN DE TEXTOS CORRUPTOS ====================

    # Lista de palabras válidas comunes en español (para detección rápida)
    COMMON_SPANISH_WORDS = {
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'de', 'del',
        'en', 'a', 'al', 'y', 'o', 'que', 'es', 'son', 'no', 'sí', 'si',
        'por', 'para', 'con', 'como', 'pero', 'más', 'muy', 'ya', 'yo',
        'tu', 'tú', 'su', 'mi', 'me', 'te', 'se', 'lo', 'le', 'nos',
        'este', 'esta', 'esto', 'ese', 'esa', 'eso', 'hay', 'ser', 'estar',
        'tiene', 'tienen', 'hace', 'hacen', 'puede', 'pueden', 'va', 'van',
        'todo', 'toda', 'todos', 'todas', 'otro', 'otra', 'otros', 'otras',
        'bien', 'mal', 'mucho', 'poco', 'algo', 'nada', 'siempre', 'nunca',
        'cuando', 'donde', 'porque', 'aunque', 'entonces', 'también', 'ahora',
        'hoy', 'ayer', 'mañana', 'aquí', 'allí', 'así', 'bueno', 'pues',
        'sobre', 'entre', 'hasta', 'desde', 'sin', 'hacia', 'durante',
        'cada', 'cual', 'quien', 'cuyo', 'tanto', 'tan', 'mismo', 'propio',
        'primero', 'segundo', 'tercero', 'último', 'nuevo', 'viejo', 'grande',
        'pequeño', 'mejor', 'peor', 'mayor', 'menor', 'solo', 'sólo',
    }

    def _is_text_corrupt(
        self,
        text: str,
        min_valid_word_ratio: float = 0.50,
        min_coherence_score: float = 0.40
    ) -> Tuple[bool, float, str]:
        """
        Detecta si un texto está muy corrupto y necesita reconstrucción.

        Criterios de corrupción:
        1. Menos del 50% de palabras son reconocibles en español
        2. Estructura sintáctica rota (secuencias de palabras sin sentido)
        3. Palabras inventadas o mal transcritas

        Returns:
            Tuple (is_corrupt, corruption_score, reason)
        """
        if not text or not text.strip():
            return False, 0.0, "Texto vacío"

        # Normalizar y extraer palabras
        words = re.findall(r'\b[a-záéíóúüñ]+\b', text.lower())

        if len(words) < 3:
            return False, 0.0, "Texto muy corto"

        # 1. Contar palabras válidas en español
        valid_words = 0
        invalid_words = []

        for word in words:
            if len(word) <= 2:  # Ignorar palabras muy cortas
                valid_words += 1
                continue

            # Verificar si está en palabras comunes
            if word in self.COMMON_SPANISH_WORDS:
                valid_words += 1
            # Verificar si está en el glosario (correcciones o mantener)
            elif word in self.glosario.get('correcciones', {}):
                valid_words += 1
            elif word in self.glosario.get('mantener', []):
                valid_words += 1
            # Verificar si parece palabra válida (no tiene secuencias extrañas)
            elif self._looks_like_valid_spanish_word(word):
                valid_words += 1
            else:
                invalid_words.append(word)

        valid_ratio = valid_words / len(words) if words else 1.0

        # 2. Verificar coherencia sintáctica básica
        coherence_score = self._check_basic_syntax(text)

        # 3. Determinar si es corrupto
        is_corrupt = (
            valid_ratio < min_valid_word_ratio or
            coherence_score < min_coherence_score
        )

        corruption_score = 1.0 - (valid_ratio * 0.6 + coherence_score * 0.4)

        reason = ""
        if is_corrupt:
            reasons = []
            if valid_ratio < min_valid_word_ratio:
                reasons.append(f"solo {valid_ratio:.0%} palabras válidas")
            if coherence_score < min_coherence_score:
                reasons.append(f"coherencia baja ({coherence_score:.0%})")
            if invalid_words:
                reasons.append(f"palabras sospechosas: {invalid_words[:3]}")
            reason = "; ".join(reasons)

        return is_corrupt, corruption_score, reason

    def _looks_like_valid_spanish_word(self, word: str) -> bool:
        """Verifica si una palabra parece válida en español por su estructura."""
        if len(word) < 2:
            return True

        # Secuencias de consonantes inválidas en español
        invalid_patterns = [
            r'[bcdfghjklmnpqrstvwxyz]{4,}',  # 4+ consonantes seguidas
            r'^[bcdfghjklmnpqrstvwxyz]{3}',   # 3 consonantes al inicio
            r'[bcdfghjklmnpqrstvwxyz]{3}$',   # 3 consonantes al final
            r'[aeiouáéíóú]{4,}',              # 4+ vocales seguidas
            r'[qwxz]{2,}',                     # Letras raras repetidas
        ]

        for pattern in invalid_patterns:
            if re.search(pattern, word):
                return False

        return True

    def _check_basic_syntax(self, text: str) -> float:
        """
        Verifica coherencia sintáctica básica.
        Retorna un score de 0.0 a 1.0.
        """
        score = 1.0

        # Patrones que indican problemas sintácticos
        problems = [
            (r'\b(el|la|los|las)\s+(el|la|los|las)\b', 0.15),  # Artículos repetidos
            (r'\b(de|a|en|por|para)\s+(de|a|en|por|para)\b', 0.15),  # Preposiciones repetidas
            (r'\b(y|o|que)\s+(y|o|que)\b', 0.10),  # Conectores repetidos
            (r'[,\.]{2,}', 0.10),  # Puntuación repetida
            (r'\s{3,}', 0.05),  # Espacios múltiples
        ]

        for pattern, penalty in problems:
            if re.search(pattern, text.lower()):
                score -= penalty

        # Bonus por estructura correcta
        if re.search(r'^[A-ZÁÉÍÓÚ¿¡]', text):  # Empieza con mayúscula
            score += 0.05
        if re.search(r'[.!?]$', text.strip()):  # Termina con puntuación
            score += 0.05

        return max(0.0, min(1.0, score))

    def _reconstruct_text(
        self,
        text: str,
        context_hint: str = "podcast en español"
    ) -> Tuple[str, Dict]:
        """
        Fase 2: Reconstruye un texto muy corrupto usando LLM.

        Args:
            text: Texto corrupto a reconstruir
            context_hint: Contexto del podcast para ayudar

        Returns:
            Tuple (texto_reconstruido, metadata)
        """
        self.stats['reconstruction_attempts'] = self.stats.get('reconstruction_attempts', 0) + 1

        system_prompt = self.RECONSTRUCTION_PROMPT.format(
            glosario_context=self.glosario_context
        )

        user_prompt = f"""Contexto del podcast: {context_hint}

Texto corrupto a reconstruir:
"{text}"

Reconstruye el significado probable. Responde SOLO con JSON."""

        try:
            response = self._call_ollama(system_prompt, user_prompt, timeout=self.timeout * 2)

            if response:
                result = self._parse_reconstruction_response(response, text)
                if result['success']:
                    self.stats['reconstruction_success'] = self.stats.get('reconstruction_success', 0) + 1
                    return result['texto_reconstruido'], {
                        'fase': 2,
                        'reconstruido': True,
                        'palabras_base': result.get('palabras_base', []),
                        'cambios': result.get('cambios_principales', []),
                        'confianza': result.get('confianza', 0.5),
                        'modelo': self.model
                    }

        except Exception as e:
            self.logger.warning(f"Error en reconstrucción: {e}")

        # Fallback: devolver original con advertencia
        return text, {
            'fase': 2,
            'reconstruido': False,
            'error': 'reconstruccion_fallida',
            'confianza': 0.3
        }

    def _parse_reconstruction_response(self, response: str, original: str) -> Dict:
        """Parsea la respuesta de reconstrucción."""
        try:
            cleaned = self._clean_json_response(response)
            data = json.loads(cleaned)

            texto = data.get('texto_reconstruido', original)
            confianza = float(data.get('confianza', 0.5))
            confianza = max(0.0, min(1.0, confianza))

            return {
                'success': True,
                'texto_reconstruido': texto,
                'palabras_base': data.get('palabras_base', []),
                'cambios_principales': data.get('cambios_principales', []),
                'confianza': confianza
            }

        except (json.JSONDecodeError, Exception) as e:
            self.logger.warning(f"Error parseando reconstrucción: {e}")
            return {'success': False, 'error': str(e)}

    # ==================== DUAL-MODEL: CLASIFICACIÓN POR DIFICULTAD ====================

    def configure_dual_model(
        self,
        local_host: str,
        local_model: str,
        local_timeout: int = 120,
        local_batch_size: int = 8,
        difficulty_threshold: float = 0.40,
        unknown_word_threshold: float = 0.25,
        min_words_for_hard: int = 5,
        fallback_to_remote: bool = True
    ):
        """
        Configura el procesamiento dual-model.

        Textos fáciles van al modelo local (rápido), difíciles al remoto (potente).
        Ambos se procesan en paralelo con ThreadPoolExecutor.

        Args:
            local_host: URL del Ollama local (ej: http://localhost:11434)
            local_model: Modelo local (ej: qwen3:14b)
            local_timeout: Timeout para el modelo local
            local_batch_size: Batch size para el modelo local
            difficulty_threshold: Score >= threshold → hard
            unknown_word_threshold: Ratio de palabras desconocidas para forzar hard
            min_words_for_hard: Mínimo de palabras desconocidas para forzar hard
            fallback_to_remote: Reintentar en remoto si local falla
        """
        self.dual_model_enabled = False

        # Verificar conexión al host local
        try:
            resp = requests.get(f"{local_host}/api/tags", timeout=10)
            if resp.status_code != 200:
                self.logger.warning(f"⚠️  Dual-model: host local {local_host} no responde, desactivando")
                return
            models = [m.get('name', '') for m in resp.json().get('models', [])]
            if not any(local_model in name for name in models):
                self.logger.warning(
                    f"⚠️  Dual-model: modelo {local_model} no encontrado en {local_host}. "
                    f"Disponibles: {models}"
                )
                return
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"⚠️  Dual-model: no se puede conectar a {local_host}: {e}")
            return

        # Preload del modelo local para evitar cold start (~147s)
        self.logger.info(f"Precargando modelo local {local_model} en {local_host}...")
        try:
            resp = requests.post(
                f"{local_host}/api/generate",
                json={"model": local_model, "prompt": "", "stream": False,
                      "options": {"num_predict": 1}},
                timeout=180  # Cold start puede tomar ~147s
            )
            if resp.status_code == 200:
                self.logger.info(f"✓ Modelo local {local_model} precargado")
            else:
                self.logger.warning(f"⚠️  Preload falló (status {resp.status_code}), desactivando dual-model")
                return
        except Exception as e:
            self.logger.warning(f"⚠️  Preload falló: {e}, desactivando dual-model")
            return

        # Todo OK, activar dual-model
        self.dual_model_enabled = True
        self.local_host = local_host
        self.local_model = local_model
        self.local_timeout = local_timeout
        self.local_batch_size = local_batch_size
        self.difficulty_threshold = difficulty_threshold
        self.unknown_word_threshold = unknown_word_threshold
        self.min_words_for_hard = min_words_for_hard
        self.fallback_to_remote = fallback_to_remote

        # Stats para dual-model
        self.stats.update({
            'easy_count': 0,
            'hard_count': 0,
            'easy_time': 0.0,
            'hard_time': 0.0,
            'local_failures': 0,
            'local_fallback_to_remote': 0
        })

        self.logger.info(
            f"✓ Dual-model activado: easy→{local_model}@{local_host} | "
            f"hard→{self.model}@{self.ollama_host} | threshold={difficulty_threshold}"
        )

    # Extended vocabulary for difficulty classifier (supplements COMMON_SPANISH_WORDS)
    # Includes common verbs, nouns, adjectives that Whisper transcribes correctly
    EXTENDED_SPANISH_WORDS = {
        # Verbos comunes (conjugaciones frecuentes)
        'ser', 'estar', 'está', 'estás', 'están', 'estoy', 'estamos',
        'haber', 'tener', 'hacer', 'poder', 'decir', 'dar', 'ver',
        'saber', 'querer', 'llegar', 'pasar', 'deber', 'poner', 'parecer', 'quedar',
        'creer', 'hablar', 'llevar', 'dejar', 'seguir', 'encontrar', 'llamar',
        'venir', 'pensar', 'salir', 'volver', 'tomar', 'conocer', 'vivir', 'sentir',
        'tratar', 'mirar', 'contar', 'empezar', 'esperar', 'buscar', 'existir',
        'entrar', 'trabajar', 'escribir', 'perder', 'producir', 'ocurrir', 'entender',
        'pedir', 'recibir', 'recordar', 'terminar', 'permitir', 'aparecer', 'conseguir',
        'comenzar', 'servir', 'sacar', 'necesitar', 'mantener', 'resultar', 'leer',
        'caer', 'cambiar', 'presentar', 'crear', 'abrir', 'considerar', 'oír',
        'acabar', 'convertir', 'ganar', 'formar', 'traer', 'partir', 'morir',
        'aceptar', 'realizar', 'suponer', 'comprender', 'lograr', 'explicar',
        'preguntar', 'tocar', 'reconocer', 'estudiar', 'alcanzar', 'nacer',
        'dirigir', 'correr', 'utilizar', 'pagar', 'ayudar', 'gustar', 'jugar',
        'escuchar', 'cumplir', 'ofrecer', 'descubrir', 'levantar', 'intentar',
        'usar', 'meter', 'ocupar', 'aprender', 'casar', 'crecer', 'subir',
        'cambiando', 'haciendo', 'diciendo', 'siendo', 'teniendo', 'pudiendo',
        'hablando', 'pasando', 'viviendo', 'trabajando', 'buscando',
        'hace', 'hizo', 'tienen', 'puede', 'pueden', 'dice', 'quiere', 'viene',
        'sale', 'pone', 'sabe', 'cree', 'lleva', 'deja', 'sigue', 'piensa',
        'vamos', 'habla', 'mira', 'cuenta', 'empieza', 'espera', 'entra',
        # Sustantivos comunes
        'vida', 'tiempo', 'año', 'años', 'día', 'días', 'mundo', 'casa', 'país',
        'parte', 'momento', 'forma', 'hombre', 'mujer', 'hijo', 'hija', 'agua',
        'mano', 'hecho', 'ejemplo', 'gobierno', 'ciudad', 'nombre', 'trabajo',
        'punto', 'grupo', 'problema', 'medio', 'caso', 'pueblo', 'tipo',
        'manera', 'cuerpo', 'padre', 'madre', 'tierra', 'familia', 'cabeza',
        'historia', 'dinero', 'lugar', 'persona', 'gente', 'cosa', 'cosas',
        'número', 'paso', 'cuenta', 'razón', 'estado', 'noche', 'doctor',
        'verdad', 'programa', 'pregunta', 'libro', 'orden', 'nivel', 'lado',
        'final', 'fuerza', 'efecto', 'cambio', 'idea', 'muerte', 'palabra',
        'tema', 'clase', 'mes', 'hora', 'juego', 'guerra', 'salud', 'calle',
        'mesa', 'sangre', 'niño', 'niña', 'minuto', 'minutos', 'semana',
        'proceso', 'centro', 'espacio', 'base', 'arte', 'campo', 'hospital',
        'video', 'podcast', 'internet', 'contenido', 'plataforma', 'canal',
        'marca', 'negocio', 'cliente', 'producto', 'servicio', 'mercado',
        'redes', 'sociales', 'digital', 'estrategia', 'marketing',
        # Adjetivos comunes
        'bueno', 'buena', 'buenos', 'buenas', 'malo', 'mala', 'malos', 'malas',
        'grande', 'grandes', 'pequeño', 'pequeña', 'largo', 'corto', 'alto', 'bajo',
        'joven', 'viejo', 'nuevo', 'nueva', 'nuevos', 'antiguo', 'diferente',
        'importante', 'posible', 'imposible', 'necesario', 'claro', 'cierto',
        'seguro', 'libre', 'fuerte', 'real', 'social', 'político', 'público',
        'humano', 'general', 'común', 'único', 'simple', 'fácil', 'difícil',
        'mejor', 'peor', 'mayor', 'menor', 'anterior', 'siguiente', 'último',
        'primero', 'segundo', 'tercero', 'cuarto', 'quinto', 'medio',
        'solo', 'sola', 'solos', 'solas', 'propio', 'propia',
        'bonito', 'lindo', 'chido', 'padre', 'chingón', 'exitoso', 'interesante',
        # Adverbios y conectores
        'después', 'antes', 'luego', 'ahí', 'acá', 'allá', 'arriba', 'abajo',
        'dentro', 'fuera', 'lejos', 'cerca', 'rápido', 'lento', 'bastante',
        'demasiado', 'realmente', 'prácticamente', 'especialmente', 'exactamente',
        'obviamente', 'básicamente', 'simplemente', 'solamente', 'totalmente',
        'completamente', 'actualmente', 'normalmente', 'generalmente',
        'incluso', 'además', 'tampoco', 'quizás', 'tal', 'vez', 'mientras',
        'según', 'junto', 'contra', 'tras', 'mediante', 'respecto',
        # Palabras de podcasts/contenido digital
        'episodio', 'suscribir', 'suscríbete', 'comentario', 'comentarios',
        'audiencia', 'comunidad', 'plataforma', 'aplicación', 'tecnología',
        'herramienta', 'herramientas', 'información', 'experiencia', 'proyecto',
        'empresa', 'compañía', 'equipo', 'sistema', 'desarrollo', 'resultado',
        'resultados', 'solución', 'oportunidad', 'comunicación',
    }

    def _classify_difficulty(self, text: str) -> Tuple[str, float, Dict]:
        """
        Clasifica un texto como 'easy' o 'hard' usando heurística rápida (cero LLM).

        Factores del score (0.0 = trivial, 1.0 = muy difícil):
        - 45%: Ratio de palabras desconocidas
        - 25%: Densidad de términos técnicos del glosario
        - 15%: Longitud del texto
        - 15%: Coherencia sintáctica inversa

        Returns:
            Tuple ('easy'|'hard', score 0.0-1.0, details_dict)
        """
        if not text or not text.strip():
            return 'easy', 0.0, {'reason': 'empty'}

        words = re.findall(r'\b[a-záéíóúüñ]+\b', text.lower())

        # Textos muy cortos → siempre easy
        if len(words) < 3:
            return 'easy', 0.0, {'reason': 'too_short', 'word_count': len(words)}

        # --- Factor 1 (45%): Ratio de palabras desconocidas ---
        # Uses COMMON + EXTENDED vocabularies and glosario. Does NOT use
        # _looks_like_valid_spanish_word (too permissive for garbled Whisper output)
        unknown_words = []
        known_words = self.COMMON_SPANISH_WORDS | self.EXTENDED_SPANISH_WORDS
        glosario_corrections = set(self.glosario.get('correcciones', {}).keys())
        glosario_mantener = set(self.glosario.get('mantener', []))

        for word in words:
            if len(word) <= 2:
                continue
            if word in known_words:
                continue
            if word in glosario_corrections or word in glosario_mantener:
                continue
            unknown_words.append(word)

        countable_words = [w for w in words if len(w) > 2]
        unknown_ratio = len(unknown_words) / max(len(countable_words), 1)
        unknown_score = min(unknown_ratio / 0.50, 1.0)  # Normalize: 50% unknown → score 1.0

        # --- Factor 2 (25%): Densidad de términos técnicos ---
        technical_count = 0
        for word in words:
            if word in glosario_corrections:
                technical_count += 1
        technical_density = technical_count / max(len(words), 1)
        technical_score = min(technical_density / 0.20, 1.0)  # 20% technical → score 1.0

        # --- Factor 3 (15%): Longitud del texto ---
        length_score = min(len(words) / 50.0, 1.0)  # 50+ words → score 1.0

        # --- Factor 4 (15%): Coherencia sintáctica inversa ---
        coherence = self._check_basic_syntax(text)
        incoherence_score = 1.0 - coherence

        # --- Score final ponderado ---
        score = (
            0.45 * unknown_score +
            0.25 * technical_score +
            0.15 * length_score +
            0.15 * incoherence_score
        )

        # --- Regla de override: muchas palabras desconocidas → forzar hard ---
        force_hard = (
            unknown_ratio > self.unknown_word_threshold and
            len(unknown_words) > self.min_words_for_hard
        )

        difficulty = 'hard' if (score >= self.difficulty_threshold or force_hard) else 'easy'

        details = {
            'score': round(score, 3),
            'unknown_ratio': round(unknown_ratio, 3),
            'unknown_words': unknown_words[:5],
            'technical_density': round(technical_density, 3),
            'word_count': len(words),
            'coherence': round(coherence, 3),
            'force_hard': force_hard
        }

        return difficulty, score, details

    def _process_batch_on_host(
        self,
        texts: List[str],
        host: str,
        model: str,
        timeout: int
    ) -> List[Tuple[str, Dict]]:
        """
        Procesa un batch de textos en un host/modelo específico.
        Idéntico a _process_batch() pero con host/model override.

        Args:
            texts: Lista de textos del batch
            host: URL del host Ollama
            model: Nombre del modelo
            timeout: Timeout base para la llamada
        """
        self.stats['batch_calls'] += 1

        system_prompt = self.BATCH_SYSTEM_PROMPT.format(
            glosario_context=self.glosario_context
        )

        texts_formatted = "\n".join(
            f'{i}. "{text}"' for i, text in enumerate(texts)
        )

        user_prompt = f"""Corrige las siguientes {len(texts)} transcripciones:

{texts_formatted}

Responde con el JSON que contiene las correcciones para TODOS los textos."""

        for attempt in range(self.max_retries):
            try:
                response = self._call_ollama(
                    system_prompt,
                    user_prompt,
                    timeout=timeout * 2,
                    host_override=host,
                    model_override=model
                )

                if response:
                    batch_result = self._parse_batch_response(response, texts)

                    if batch_result:
                        for corrected_text, meta in batch_result:
                            if 'error' not in meta:
                                meta['modelo'] = model
                                self.stats['processed'] += 1
                                self.stats['corrected'] += 1
                                self.stats['total_changes'] += len(meta.get('cambios', []))

                                conf = meta.get('confianza', 0.5)
                                n = self.stats['corrected']
                                self.stats['avg_confidence'] = (
                                    (self.stats['avg_confidence'] * (n - 1) + conf) / n
                                )

                        return batch_result

            except Exception as e:
                self.logger.warning(
                    f"Batch en {host}/{model} intento {attempt + 1}/{self.max_retries} falló: {e}"
                )
                continue

        # Fallback: procesar individualmente
        self.logger.warning(f"Batch en {host}/{model} falló, procesando individualmente...")
        results = []
        for text in texts:
            corrected, meta = self.correct(text)
            meta['modelo'] = model
            results.append((corrected, meta))
        return results

    # ==================== CORRECCIÓN INDIVIDUAL ====================
    
    def correct(self, text: str) -> Tuple[str, Dict]:
        """
        Corrige un texto usando el LLM.
        
        Args:
            text: Texto a corregir
            
        Returns:
            Tuple (texto_corregido, metadata)
        """
        if not text or not text.strip():
            return text, {'error': 'texto_vacío'}
        
        # Verificar caché primero
        cached = self._get_from_cache(text)
        if cached:
            return cached.get('texto_corregido', text), cached
        
        self.stats['processed'] += 1
        self.stats['individual_calls'] += 1
        
        # Construir prompt con contexto del glosario
        system_prompt = self.SYSTEM_PROMPT.format(
            glosario_context=self.glosario_context
        )
        
        user_prompt = f"""Corrige la siguiente transcripción:

"{text}"

Recuerda: Responde SOLO con el JSON estructurado."""

        # Intentar corrección con reintentos
        for attempt in range(self.max_retries):
            try:
                response = self._call_ollama(system_prompt, user_prompt)
                
                if response:
                    result = self._parse_response(response, text)
                    
                    if result['success']:
                        self.stats['corrected'] += 1
                        self.stats['total_changes'] += len(result.get('cambios', []))
                        
                        # Actualizar promedio de confianza
                        conf = result.get('confianza', 0.5)
                        n = self.stats['corrected']
                        self.stats['avg_confidence'] = (
                            (self.stats['avg_confidence'] * (n - 1) + conf) / n
                        )
                        
                        metadata = {
                            'cambios': result.get('cambios', []),
                            'confianza': conf,
                            'modelo': self.model,
                            'intentos': attempt + 1
                        }
                        
                        # Guardar en caché
                        cache_response = {
                            'texto_corregido': result['texto_corregido'],
                            **metadata
                        }
                        self._add_to_cache(text, cache_response)
                        
                        return result['texto_corregido'], metadata
                
            except Exception as e:
                self.logger.warning(
                    f"Intento {attempt + 1}/{self.max_retries} falló: {e}"
                )
                continue
        
        # Si fallan todos los intentos, devolver texto original
        self.stats['failed'] += 1
        return text, {'error': 'max_retries_exceeded', 'original': True}
    
    # ==================== PROCESAMIENTO POR LOTES ====================
    
    def correct_batch_optimized(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        verify_corrections: Optional[bool] = None
    ) -> List[Tuple[str, Dict]]:
        """
        Corrige múltiples textos en lotes optimizados.
        Reduce llamadas HTTP agrupando textos.
        
        Args:
            texts: Lista de textos a corrigir
            batch_size: Tamaño del lote (usa self.batch_size si no se especifica)
            verify_corrections: Verificar correcciones (usa self.enable_verification si no se especifica)
            
        Returns:
            Lista de tuplas (texto_corregido, metadata) en el mismo orden
        """
        if not texts:
            return []
        
        batch_size = batch_size or self.batch_size
        should_verify = verify_corrections if verify_corrections is not None else self.enable_verification
        
        results: List[Tuple[str, Dict]] = [None] * len(texts)  # type: ignore
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []
        
        # Primero verificar caché para todos
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = (text, {'error': 'texto_vacío'})
                continue
            
            cached = self._get_from_cache(text)
            if cached:
                results[i] = (cached.get('texto_corregido', text), cached)
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # Procesar textos no cacheados
        if uncached_texts:
            if getattr(self, 'dual_model_enabled', False):
                self._process_dual_model(
                    uncached_texts, uncached_indices, results,
                    batch_size, should_verify
                )
            else:
                self._process_single_model(
                    uncached_texts, uncached_indices, results,
                    batch_size, should_verify
                )
        
        # Guardar caché al final
        if self.enable_cache:
            self._save_cache()
        
        # Asegurar que no hay None en los resultados (safety check)
        for i, result in enumerate(results):
            if result is None:
                # Si por alguna razón un resultado quedó como None, usar texto original
                original_text = texts[i] if i < len(texts) else ""
                results[i] = (original_text, {'error': 'resultado_no_procesado', 'fallback': True})
                self.logger.warning(f"Resultado None en índice {i}, usando texto original como fallback")
        
        return results
    
    def _process_batch(self, texts: List[str]) -> List[Tuple[str, Dict]]:
        """
        Procesa un batch de textos en una sola llamada al LLM.
        
        Args:
            texts: Lista de textos (máximo batch_size)
            
        Returns:
            Lista de (texto_corregido, metadata)
        """
        self.stats['batch_calls'] += 1
        
        system_prompt = self.BATCH_SYSTEM_PROMPT.format(
            glosario_context=self.glosario_context
        )
        
        # Construir prompt con textos numerados
        texts_formatted = "\n".join(
            f'{i}. "{text}"' for i, text in enumerate(texts)
        )
        
        user_prompt = f"""Corrige las siguientes {len(texts)} transcripciones:

{texts_formatted}

Responde con el JSON que contiene las correcciones para TODOS los textos."""

        # Intentar con reintentos
        for attempt in range(self.max_retries):
            try:
                response = self._call_ollama(
                    system_prompt, 
                    user_prompt,
                    timeout=self.timeout * 2  # Más tiempo para batches
                )
                
                if response:
                    batch_result = self._parse_batch_response(response, texts)
                    
                    if batch_result:
                        # Actualizar estadísticas
                        for _, meta in batch_result:
                            if 'error' not in meta:
                                self.stats['processed'] += 1
                                self.stats['corrected'] += 1
                                self.stats['total_changes'] += len(meta.get('cambios', []))
                                
                                conf = meta.get('confianza', 0.5)
                                n = self.stats['corrected']
                                self.stats['avg_confidence'] = (
                                    (self.stats['avg_confidence'] * (n - 1) + conf) / n
                                )
                        
                        return batch_result
                        
            except Exception as e:
                self.logger.warning(
                    f"Batch intento {attempt + 1}/{self.max_retries} falló: {e}"
                )
                continue
        
        # Fallback: procesar individualmente
        self.logger.warning("Batch falló, procesando individualmente...")
        return [self.correct(text) for text in texts]
    
    def _clean_json_response(self, response: str) -> str:
        """
        Limpia y repara JSON malformado del LLM.
        Maneja los errores más comunes de formato.
        """
        response = response.strip()
        
        # Extraer solo el JSON (ignorar texto antes/después)
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            response = json_match.group(0)
        
        # Remover prefijos comunes del LLM
        response = re.sub(r'^```json\s*', '', response)
        response = re.sub(r'\s*```$', '', response)
        response = re.sub(r'^Here is.*?:', '', response, flags=re.IGNORECASE)
        response = re.sub(r'^JSON:?\s*', '', response, flags=re.IGNORECASE)
        
        # Remover comentarios // o /* */
        response = re.sub(r'//.*?$', '', response, flags=re.MULTILINE)
        response = re.sub(r'/\*.*?\*/', '', response, flags=re.DOTALL)
        
        # Remover comas trailing antes de } o ]
        response = re.sub(r',(\s*[}\]])', r'\1', response)
        
        # Agregar comas faltantes entre objetos en array: }{ → },{
        response = re.sub(r'\}(\s*)\{', r'},\1{', response)
        
        # Agregar comas faltantes entre propiedades en misma línea
        # "valor"   "siguiente" → "valor", "siguiente"
        response = re.sub(r'(")\s+(")', r'\1, \2', response)
        
        # Agregar comas faltantes después de números seguidos de "
        # 0.95   "cambios" → 0.95, "cambios"
        response = re.sub(r'(\d)(\s+)(")', r'\1,\2\3', response)
        
        # Agregar comas faltantes después de ] seguido de "
        # ]   "siguiente" → ], "siguiente"
        response = re.sub(r'(\])(\s+)(")', r'\1,\2\3', response)
        
        # Agregar comas faltantes entre líneas: valor\n"campo"
        response = re.sub(r'(\d|"|\])\s*\n\s*"', r'\1,\n"', response)
        
        # Arreglar strings no terminadas (problema común)
        # Buscar líneas que empiezan con " pero no terminan con ",
        lines = response.split('\n')
        fixed_lines = []
        for line in lines:
            stripped = line.strip()
            # Si la línea tiene un string no cerrado, intentar cerrarlo
            if stripped.count('"') % 2 == 1:
                # Número impar de comillas, agregar una al final
                if not stripped.endswith('"'):
                    line = line.rstrip() + '"'
            fixed_lines.append(line)
        response = '\n'.join(fixed_lines)
        
        # Remover trailing content después del último }
        last_brace = response.rfind('}')
        if last_brace != -1:
            response = response[:last_brace + 1]
        
        return response
    
    def _request_json_correction(
        self, 
        original_response: str, 
        error_msg: str,
        original_texts: List[str]
    ) -> Optional[str]:
        """
        Pide al LLM que corrija su respuesta JSON malformada.
        """
        correction_prompt = f"""Tu respuesta anterior tenía un error de formato JSON:
ERROR: {error_msg}

Tu respuesta fue:
```
{original_response[:500]}...
```

Por favor, responde ÚNICAMENTE con el JSON válido corregido.
El JSON debe tener esta estructura exacta:
{{
  "correcciones": [
    {{"id": 0, "texto_corregido": "...", "cambios": [], "confianza": 0.95}},
    {{"id": 1, "texto_corregido": "...", "cambios": [], "confianza": 0.90}}
  ]
}}

IMPORTANTE:
- NO incluyas texto fuera del JSON
- Usa comas entre cada objeto del array
- Los IDs deben ser 0, 1, 2... hasta {len(original_texts) - 1}
- La confianza debe ser un número entre 0 y 1
"""
        
        try:
            return self._call_ollama(
                "Eres un corrector de JSON. Responde SOLO con JSON válido.",
                correction_prompt,
                timeout=30
            )
        except Exception as e:
            self.logger.warning(f"Error pidiendo corrección JSON: {e}")
            return None
    
    def _parse_batch_response(
        self, 
        response: str, 
        original_texts: List[str]
    ) -> Optional[List[Tuple[str, Dict]]]:
        """Parsea la respuesta de un batch con reparación y reintentos."""
        
        max_retries = 3
        last_error = ""
        json_parse_failed = 0
        pydantic_failed = 0
        
        for attempt in range(max_retries):
            try:
                # Limpiar JSON
                cleaned = self._clean_json_response(response)
                
                # Intentar parsear
                data = json.loads(cleaned)
                
                # JSON parseó correctamente, ahora validar con Pydantic
                if PYDANTIC_AVAILABLE:
                    try:
                        validated = LLMCorrectionBatchResponse(**data)
                        correcciones = [
                            (c.texto_corregido, {
                                'cambios': c.cambios,
                                'confianza': c.confianza,
                                'modelo': self.model,
                                'batch': True,
                                'pydantic_validated': True,
                                'repair_attempts': attempt
                            })
                            for c in validated.correcciones
                        ]
                        self.stats['pydantic_validations'] = self.stats.get('pydantic_validations', 0) + len(correcciones)
                        
                        # Mostrar mensaje de éxito con Pydantic
                        if attempt > 0:
                            print(f"   🔷 Pydantic: JSON reparado en intento {attempt + 1}, {len(correcciones)} items validados")
                        
                        return correcciones
                        
                    except ValidationError as e:
                        pydantic_failed += 1
                        last_error = f"Pydantic: {str(e)[:200]}"
                        print(f"   ⚠️ Pydantic (intento {attempt + 1}/{max_retries}): estructura inválida")
                        
                        # Pedir al LLM que corrija
                        if attempt < max_retries - 1:
                            new_response = self._request_json_correction(response, last_error, original_texts)
                            if new_response:
                                response = new_response
                                continue
                else:
                    # Sin Pydantic, usar fallback
                    correcciones_raw = data.get('correcciones', [])
                    results = []
                    
                    for i, text in enumerate(original_texts):
                        corr = next((c for c in correcciones_raw if c.get('id') == i), None)
                        
                        if corr:
                            results.append((
                                corr.get('texto_corregido', text),
                                {
                                    'cambios': corr.get('cambios', []),
                                    'confianza': max(0.0, min(1.0, float(corr.get('confianza', 0.5)))),
                                    'modelo': self.model,
                                    'batch': True
                                }
                            ))
                        else:
                            results.append((text, {'error': 'missing_in_batch'}))
                    
                    return results
                    
            except json.JSONDecodeError as e:
                json_parse_failed += 1
                last_error = f"JSON: {str(e)}"
                # Mostrar en consola para visibilidad
                print(f"   ⚠️ JSON inválido (intento {attempt + 1}/{max_retries})")
                
                # Pedir al LLM que corrija
                if attempt < max_retries - 1:
                    new_response = self._request_json_correction(response, last_error, original_texts)
                    if new_response:
                        response = new_response
                        continue
                        
            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"Error procesando batch (intento {attempt + 1}/{max_retries}): {e}")
        
        # Después de 3 intentos, retornar None para caer al fallback
        if json_parse_failed > 0 and pydantic_failed == 0:
            print(f"   ❌ Batch falló: JSON malformado ({json_parse_failed} intentos)")
        elif pydantic_failed > 0:
            print(f"   ❌ Batch falló: Pydantic rechazó estructura ({pydantic_failed} intentos)")
        else:
            print(f"   ❌ Batch falló: {last_error[:50]}")
        
        return None
    
    # ==================== SINGLE/DUAL MODEL PROCESSING ====================

    def _verify_batch_results(
        self,
        batch_texts: List[str],
        batch_results: List[Tuple[str, Dict]],
        should_verify: bool
    ) -> Tuple[List[Tuple[str, Dict]], Dict]:
        """
        Aplica verificación anti-alucinación a resultados de un batch.
        Extracted para reusar en single y dual model.

        Returns:
            Tuple (verified_results, counters_dict)
        """
        counters = {
            'verification_passed': 0,
            'verification_failed': 0,
            'verification_reverted': 0,
            'reconstruction_attempted': 0,
            'reconstruction_success': 0,
            'pydantic_validations': 0
        }

        if not (should_verify and self.verifier):
            # Count pydantic validations even without verification
            for _, meta in batch_results:
                if 'error' not in meta and meta.get('pydantic_validated'):
                    counters['pydantic_validations'] += 1
            return batch_results, counters

        verified_results = []
        for idx, (original_text, (corrected_text, meta)) in enumerate(zip(batch_texts, batch_results)):
            if 'error' not in meta:
                if meta.get('pydantic_validated'):
                    counters['pydantic_validations'] += 1

                llm_conf = meta.get('confianza', 1.0)
                v_result = self.verifier.verify(original_text, corrected_text, llm_conf)

                if v_result.is_valid:
                    counters['verification_passed'] += 1
                    meta['verification'] = {
                        'passed': True,
                        'score': v_result.confidence_score,
                        'checks_passed': v_result.checks_passed
                    }
                    verified_results.append((corrected_text, meta))
                else:
                    counters['verification_failed'] += 1

                    # FASE 2: Reconstrucción de textos corruptos
                    is_corrupt, corruption_score, corruption_reason = self._is_text_corrupt(original_text)

                    if is_corrupt:
                        counters['reconstruction_attempted'] += 1
                        self.logger.info(
                            f"Texto corrupto detectado (score={corruption_score:.2f}): {corruption_reason}. "
                            f"Intentando reconstrucción..."
                        )

                        reconstructed, recon_meta = self._reconstruct_text(original_text)

                        if recon_meta.get('success', False):
                            recon_conf = recon_meta.get('confianza', 0.5)
                            v_recon = self.verifier.verify(original_text, reconstructed, recon_conf)

                            if v_recon.is_valid or v_recon.confidence_score >= 0.5:
                                counters['reconstruction_success'] += 1
                                counters['verification_passed'] += 1
                                meta['verification'] = {
                                    'passed': True,
                                    'phase': 2,
                                    'reconstruction': True,
                                    'score': v_recon.confidence_score,
                                    'corruption_score': corruption_score,
                                    'checks_passed': v_recon.checks_passed
                                }
                                meta['reconstruction'] = recon_meta
                                self.logger.info(
                                    f"Reconstrucción exitosa: '{original_text[:30]}...' -> '{reconstructed[:30]}...'"
                                )
                                verified_results.append((reconstructed, meta))
                                continue
                            else:
                                self.logger.debug(
                                    f"Reconstrucción no pasó verificación: score={v_recon.confidence_score}"
                                )
                        else:
                            self.logger.debug(
                                f"Reconstrucción falló: {recon_meta.get('error', 'unknown')}"
                            )

                    # Revertir al original
                    counters['verification_reverted'] += 1
                    meta['verification'] = {
                        'passed': False,
                        'score': v_result.confidence_score,
                        'checks_failed': v_result.checks_failed,
                        'warnings': v_result.warnings,
                        'reverted': True
                    }
                    self.logger.debug(
                        f"Verificación falló para: '{original_text[:50]}...' -> "
                        f"Revirtiendo. Checks: {v_result.checks_failed}"
                    )
                    verified_results.append((original_text, meta))
            else:
                verified_results.append((corrected_text, meta))

        return verified_results, counters

    def _store_batch_results(
        self,
        batch_texts: List[str],
        batch_indices: List[int],
        batch_results: List[Tuple[str, Dict]],
        results: List,
        should_verify: bool
    ):
        """Stores batch results into the main results array and caches them."""
        for idx, (original_idx, result) in enumerate(zip(batch_indices, batch_results)):
            results[original_idx] = result

            if 'error' not in result[1]:
                verification_info = result[1].get('verification', {})
                if not should_verify or verification_info.get('passed', True):
                    self._add_to_cache(
                        batch_texts[idx],
                        {'texto_corregido': result[0], **result[1]}
                    )

    def _print_processing_summary(self, counters: Dict, should_verify: bool):
        """Prints verification and pydantic summary."""
        if PYDANTIC_AVAILABLE and counters.get('pydantic_validations', 0) > 0:
            print(f"   🔷 Pydantic: {counters['pydantic_validations']} respuestas validadas correctamente")

        if should_verify and self.verifier:
            vp = counters.get('verification_passed', 0)
            vf = counters.get('verification_failed', 0)
            vr = counters.get('verification_reverted', 0)
            if vp > 0 or vf > 0:
                print(f"   🔍 Verificación: {vp} OK, {vf} fallaron ({vr} revertidos)")

        ra = counters.get('reconstruction_attempted', 0)
        rs = counters.get('reconstruction_success', 0)
        if ra > 0:
            print(f"   🔄 Reconstrucción (Fase 2): {rs}/{ra} textos corruptos recuperados")

    def _update_verification_stats(self, counters: Dict):
        """Updates global stats from batch counters."""
        self.stats['verification_passed'] += counters.get('verification_passed', 0)
        self.stats['verification_failed'] += counters.get('verification_failed', 0)
        self.stats['verification_reverted'] += counters.get('verification_reverted', 0)
        self.stats['reconstruction_attempted'] = (
            self.stats.get('reconstruction_attempted', 0) + counters.get('reconstruction_attempted', 0)
        )
        self.stats['reconstruction_success'] = (
            self.stats.get('reconstruction_success', 0) + counters.get('reconstruction_success', 0)
        )

    def _process_single_model(
        self,
        uncached_texts: List[str],
        uncached_indices: List[int],
        results: List,
        batch_size: int,
        should_verify: bool
    ):
        """
        Procesa todos los textos usando un solo modelo (comportamiento original).
        Extraído de correct_batch_optimized() para claridad.
        """
        num_batches = (len(uncached_texts) + batch_size - 1) // batch_size
        cache_msg = f" (caché: {self.stats['cache_hits']})" if self.stats['cache_hits'] > 0 else ""
        verify_msg = " [verificación ON]" if should_verify and self.verifier else ""
        print(f"   📦 Procesando {len(uncached_texts)} textos en {num_batches} batches{cache_msg}{verify_msg}")

        batch_ranges = range(0, len(uncached_texts), batch_size)
        if TQDM_AVAILABLE:
            batch_iterator = tqdm(
                batch_ranges,
                desc="   Corrigiendo con LLM",
                unit="batch",
                ncols=80
            )
        else:
            batch_iterator = batch_ranges

        all_counters = {
            'pydantic_validations': 0,
            'verification_passed': 0,
            'verification_failed': 0,
            'verification_reverted': 0,
            'reconstruction_attempted': 0,
            'reconstruction_success': 0
        }

        for batch_start in batch_iterator:
            batch_end = min(batch_start + batch_size, len(uncached_texts))
            batch_texts = uncached_texts[batch_start:batch_end]
            batch_indices = uncached_indices[batch_start:batch_end]

            batch_results = self._process_batch(batch_texts)

            # Verify
            batch_results, counters = self._verify_batch_results(
                batch_texts, batch_results, should_verify
            )
            for k in all_counters:
                all_counters[k] += counters.get(k, 0)

            # Store
            self._store_batch_results(
                batch_texts, batch_indices, batch_results, results, should_verify
            )

        self._update_verification_stats(all_counters)
        self._print_processing_summary(all_counters, should_verify)

    def _process_dual_model(
        self,
        uncached_texts: List[str],
        uncached_indices: List[int],
        results: List,
        batch_size: int,
        should_verify: bool
    ):
        """
        Procesa textos en paralelo usando modelo local (easy) y remoto (hard).

        1. Clasifica todos los textos por dificultad
        2. Despacha easy→local y hard→remoto en paralelo con ThreadPoolExecutor
        3. Fallback: textos que fallan en local se reintentan en remoto
        4. Verifica TODOS los resultados
        5. Mapea resultados de vuelta al orden original
        """
        import time as time_module

        # --- 1. Clasificar ---
        easy_items = []  # (pos_in_uncached, text)
        hard_items = []
        for i, text in enumerate(uncached_texts):
            difficulty, score, details = self._classify_difficulty(text)
            if difficulty == 'easy':
                easy_items.append((i, text))
            else:
                hard_items.append((i, text))

        self.stats['easy_count'] += len(easy_items)
        self.stats['hard_count'] += len(hard_items)

        cache_msg = f" (caché: {self.stats['cache_hits']})" if self.stats['cache_hits'] > 0 else ""
        verify_msg = " [verificación ON]" if should_verify and self.verifier else ""
        print(
            f"   📦 Dual-model: {len(uncached_texts)} textos "
            f"({len(easy_items)} easy→{self.local_model}, "
            f"{len(hard_items)} hard→{self.model}){cache_msg}{verify_msg}"
        )

        # --- 2. Parallel dispatch ---
        # all_results[pos_in_uncached] = (corrected_text, meta)
        all_results: List[Optional[Tuple[str, Dict]]] = [None] * len(uncached_texts)
        local_failures = []  # positions that failed on local

        def _worker_easy():
            """Procesa textos fáciles en el modelo local."""
            t0 = time_module.time()
            local_bs = self.local_batch_size
            for batch_start in range(0, len(easy_items), local_bs):
                batch = easy_items[batch_start:batch_start + local_bs]
                batch_texts = [text for _, text in batch]
                batch_positions = [pos for pos, _ in batch]

                try:
                    batch_results = self._process_batch_on_host(
                        batch_texts, self.local_host, self.local_model, self.local_timeout
                    )
                    for pos, result in zip(batch_positions, batch_results):
                        if 'error' in result[1]:
                            local_failures.append((pos, uncached_texts[pos]))
                        else:
                            all_results[pos] = result
                except Exception as e:
                    self.logger.warning(f"Worker easy falló en batch: {e}")
                    for pos, text in batch:
                        local_failures.append((pos, text))

            elapsed = time_module.time() - t0
            self.stats['easy_time'] += elapsed
            return elapsed

        def _worker_hard():
            """Procesa textos difíciles en el modelo remoto."""
            t0 = time_module.time()
            for batch_start in range(0, len(hard_items), batch_size):
                batch = hard_items[batch_start:batch_start + batch_size]
                batch_texts = [text for _, text in batch]
                batch_positions = [pos for pos, _ in batch]

                try:
                    batch_results = self._process_batch_on_host(
                        batch_texts, self.ollama_host, self.model, self.timeout
                    )
                    for pos, result in zip(batch_positions, batch_results):
                        all_results[pos] = result
                except Exception as e:
                    self.logger.warning(f"Worker hard falló en batch: {e}")
                    for pos, text in batch:
                        all_results[pos] = (text, {'error': f'hard_worker_failed: {e}'})

            elapsed = time_module.time() - t0
            self.stats['hard_time'] += elapsed
            return elapsed

        # Execute workers in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            if easy_items:
                futures.append(executor.submit(_worker_easy))
            if hard_items:
                futures.append(executor.submit(_worker_hard))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Worker falló: {e}")

        # --- 3. Fallback: retry local failures on remote ---
        if local_failures and self.fallback_to_remote:
            self.stats['local_failures'] += len(local_failures)
            print(f"   🔄 Reintentando {len(local_failures)} textos fallidos en remoto...")

            for fb_start in range(0, len(local_failures), batch_size):
                fb_batch = local_failures[fb_start:fb_start + batch_size]
                fb_texts = [text for _, text in fb_batch]
                fb_positions = [pos for pos, _ in fb_batch]

                try:
                    fb_results = self._process_batch_on_host(
                        fb_texts, self.ollama_host, self.model, self.timeout
                    )
                    for pos, result in zip(fb_positions, fb_results):
                        all_results[pos] = result
                        self.stats['local_fallback_to_remote'] += 1
                except Exception as e:
                    self.logger.warning(f"Fallback remoto falló: {e}")
                    for pos, text in fb_batch:
                        all_results[pos] = (text, {'error': f'fallback_failed: {e}'})
        elif local_failures:
            self.stats['local_failures'] += len(local_failures)
            for pos, text in local_failures:
                all_results[pos] = (text, {'error': 'local_failed_no_fallback'})

        # --- 4. Verify ALL results ---
        all_counters = {
            'pydantic_validations': 0,
            'verification_passed': 0,
            'verification_failed': 0,
            'verification_reverted': 0,
            'reconstruction_attempted': 0,
            'reconstruction_success': 0
        }

        for i in range(len(uncached_texts)):
            if all_results[i] is None:
                all_results[i] = (uncached_texts[i], {'error': 'not_processed'})

        # Process in batch-sized chunks for verification
        for chunk_start in range(0, len(uncached_texts), batch_size):
            chunk_end = min(chunk_start + batch_size, len(uncached_texts))
            chunk_texts = uncached_texts[chunk_start:chunk_end]
            chunk_results = all_results[chunk_start:chunk_end]
            chunk_indices = uncached_indices[chunk_start:chunk_end]

            # Verify
            verified, counters = self._verify_batch_results(
                chunk_texts, chunk_results, should_verify
            )
            for k in all_counters:
                all_counters[k] += counters.get(k, 0)

            # Store into results
            self._store_batch_results(
                chunk_texts, chunk_indices, verified, results, should_verify
            )

        self._update_verification_stats(all_counters)

        # --- 5. Print summary ---
        easy_t = self.stats.get('easy_time', 0)
        hard_t = self.stats.get('hard_time', 0)
        lf = self.stats.get('local_failures', 0)
        fb = self.stats.get('local_fallback_to_remote', 0)
        print(
            f"   ⏱️  Tiempos: easy={easy_t:.1f}s ({self.local_model}), "
            f"hard={hard_t:.1f}s ({self.model})"
        )
        if lf > 0:
            print(f"   ⚠️  Local failures: {lf} ({fb} recuperados vía fallback)")

        self._print_processing_summary(all_counters, should_verify)

    # ==================== PROCESAMIENTO PARALELO ====================
    
    def correct_parallel(
        self,
        texts: List[str],
        max_workers: Optional[int] = None
    ) -> List[Tuple[str, Dict]]:
        """
        Corrige textos en paralelo usando ThreadPoolExecutor.
        Útil cuando el batch processing no es viable.
        
        Args:
            texts: Lista de textos
            max_workers: Número de workers paralelos
            
        Returns:
            Lista de (texto_corregido, metadata)
        """
        max_workers = max_workers or self.max_workers
        results: List[Optional[Tuple[str, Dict]]] = [None] * len(texts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.correct, text): i
                for i, text in enumerate(texts)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    self.logger.error(f"Error en paralelo idx {idx}: {e}")
                    results[idx] = (texts[idx], {'error': str(e)})
        
        return results  # type: ignore
    
    # ==================== UTILIDADES ====================
    
    def _call_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout: Optional[int] = None,
        max_retries: int = 3,
        host_override: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> Optional[str]:
        """
        Llama a la API de Ollama con reintentos y backoff exponencial.

        Usa /api/chat para qwen3 (separa thinking de content correctamente).
        Maneja error 500, timeouts y problemas de conexión automáticamente.

        Args:
            host_override: URL del host Ollama alternativo (para dual-model)
            model_override: Nombre del modelo alternativo (para dual-model)
        """
        import time as time_module

        host = host_override or self.ollama_host
        model = model_override or self.model
        actual_timeout = timeout or self.timeout
        is_qwen3 = 'qwen3' in model.lower()

        # qwen3 thinking mode consume ~3000 tokens antes del contenido real
        # necesita num_predict alto para que queden tokens para la respuesta
        num_predict = 16384 if is_qwen3 else 2048

        for attempt in range(max_retries):
            try:
                if is_qwen3:
                    # Usar /api/chat para qwen3: separa thinking de content
                    # /api/generate pone TODO en thinking y response queda vacío
                    response = requests.post(
                        f"{host}/api/chat",
                        json={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            "stream": False,
                            "options": {
                                "temperature": 0.3,
                                "top_p": 0.9,
                                "num_predict": num_predict,
                            }
                        },
                        timeout=actual_timeout
                    )
                else:
                    response = requests.post(
                        f"{host}/api/generate",
                        json={
                            "model": model,
                            "prompt": user_prompt,
                            "system": system_prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.3,
                                "top_p": 0.9,
                                "num_predict": num_predict,
                            }
                        },
                        timeout=actual_timeout
                    )

                if response.status_code == 200:
                    data = response.json()

                    if is_qwen3:
                        # /api/chat: content is in message.content, thinking in message.thinking
                        msg = data.get('message', {})
                        result = msg.get('content', '')
                        if not result:
                            # Fallback: extract JSON from thinking if content is empty
                            thinking = msg.get('thinking', '')
                            if thinking:
                                json_match = re.search(r'\{[\s\S]*\}', thinking)
                                if json_match:
                                    self.logger.debug("Extrayendo JSON del campo thinking")
                                    result = json_match.group(0)
                    else:
                        result = data.get('response', '')

                    return result
                
                elif response.status_code == 500:
                    # Error interno del servidor - intentar precargar modelo
                    self.logger.warning(
                        f"Error 500 en Ollama (intento {attempt + 1}/{max_retries}). "
                        "Puede que el modelo no esté cargado."
                    )
                    if attempt == 0:
                        # Intentar precargar el modelo en el primer error
                        self._preload_model()
                    
                    # Backoff exponencial
                    wait_time = (2 ** attempt) * 2  # 2, 4, 8 segundos
                    self.logger.info(f"Esperando {wait_time}s antes de reintentar...")
                    time_module.sleep(wait_time)
                    continue
                
                elif response.status_code == 503:
                    # Servicio no disponible - el servidor está sobrecargado
                    self.logger.warning(
                        f"Ollama sobrecargado (503), intento {attempt + 1}/{max_retries}"
                    )
                    wait_time = (2 ** attempt) * 3  # 3, 6, 12 segundos
                    time_module.sleep(wait_time)
                    continue
                
                else:
                    self.logger.error(
                        f"Error Ollama: {response.status_code} - {response.text[:200]}"
                    )
                    return None
                    
            except requests.exceptions.Timeout:
                self.logger.warning(
                    f"Timeout en Ollama ({actual_timeout}s), intento {attempt + 1}/{max_retries}"
                )
                # Aumentar timeout progresivamente
                actual_timeout = int(actual_timeout * 1.5)
                continue
                
            except requests.exceptions.ConnectionError as e:
                self.logger.error(f"Error de conexión con Ollama: {e}")
                # Verificar si Ollama está corriendo
                if attempt == 0:
                    self.logger.info("Verificando estado de Ollama...")
                    if not self._verify_connection():
                        self.logger.error("Ollama no está disponible")
                        return None
                wait_time = (2 ** attempt) * 2
                time_module.sleep(wait_time)
                continue
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error en request: {e}")
                return None
        
        self.logger.error(f"Falló después de {max_retries} intentos")
        return None
    
    def _preload_model(self) -> bool:
        """Precarga el modelo en memoria para evitar cold starts."""
        try:
            self.logger.info(f"Precargando modelo {self.model}...")
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "",  # Prompt vacío solo para cargar
                    "stream": False,
                    "options": {"num_predict": 1}
                },
                timeout=120  # 2 minutos para cargar modelo grande
            )
            if response.status_code == 200:
                self.logger.info(f"✓ Modelo {self.model} precargado")
                return True
            else:
                self.logger.warning(f"Error precargando modelo: {response.status_code}")
                return False
        except Exception as e:
            self.logger.warning(f"Error precargando modelo: {e}")
            return False
    
    def _parse_response(self, response: str, original_text: str) -> Dict:
        """Parsea la respuesta JSON del LLM."""
        try:
            response = response.strip()
            
            # Extraer JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                response = json_match.group(0)
            
            data = json.loads(response)
            
            # Validar con Pydantic si está disponible
            if PYDANTIC_AVAILABLE:
                try:
                    validated = LLMCorrectionResponse(**data)
                    self.stats['pydantic_validations'] = self.stats.get('pydantic_validations', 0) + 1
                    return {
                        'success': True,
                        'texto_corregido': validated.texto_corregido,
                        'cambios': validated.cambios,
                        'confianza': validated.confianza,
                        'pydantic_validated': True
                    }
                except ValidationError as e:
                    self.logger.debug(f"Validación Pydantic falló, usando fallback: {e}")
            
            # Fallback sin Pydantic
            texto = data.get('texto_corregido', original_text)
            cambios = data.get('cambios', [])
            confianza = float(data.get('confianza', 0.5))
            confianza = max(0.0, min(1.0, confianza))
            
            return {
                'success': True,
                'texto_corregido': texto,
                'cambios': cambios if isinstance(cambios, list) else [],
                'confianza': confianza
            }
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Error parseando JSON: {e}")
            
            # Intentar extraer texto de respuesta malformada
            if '"texto_corregido"' in response:
                match = re.search(
                    r'"texto_corregido"\s*:\s*"([^"]*)"',
                    response
                )
                if match:
                    return {
                        'success': True,
                        'texto_corregido': match.group(1),
                        'cambios': ['extracción_parcial'],
                        'confianza': 0.5
                    }
            
            return {'success': False, 'error': str(e)}
        
        except Exception as e:
            self.logger.warning(f"Error procesando respuesta: {e}")
            return {'success': False, 'error': str(e)}
    
    def correct_batch(
        self,
        entries: List[Dict],
        text_field: str = 'text',
        min_confidence: float = 0.7,
        use_batch_api: bool = True
    ) -> List[Dict]:
        """
        Corrige un lote de entradas (wrapper de alto nivel).
        
        Args:
            entries: Lista de diccionarios con campo de texto
            text_field: Nombre del campo que contiene el texto
            min_confidence: Confianza mínima para aceptar corrección
            use_batch_api: Usar API de batch optimizada
            
        Returns:
            Lista de entradas con texto corregido
        """
        if not entries:
            return []
        
        # Extraer textos
        texts = [e.get(text_field, '') for e in entries]
        
        # Procesar
        if use_batch_api:
            corrections = self.correct_batch_optimized(texts)
        else:
            corrections = [self.correct(t) for t in texts]
        
        # Aplicar correcciones
        processed = []
        for entry, (corrected, metadata) in zip(entries, corrections):
            new_entry = entry.copy()
            
            if 'error' not in metadata:
                confianza = metadata.get('confianza', 0)
                
                if confianza >= min_confidence:
                    original = entry.get(text_field, '')
                    new_entry[text_field] = corrected
                    
                    if corrected != original:
                        new_entry['text_original'] = original
                        new_entry['llm_correction'] = {
                            'cambios': metadata.get('cambios', []),
                            'confianza': confianza
                        }
                else:
                    new_entry['llm_low_confidence'] = confianza
            else:
                new_entry['llm_error'] = metadata.get('error')
            
            processed.append(new_entry)
        
        return processed
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas de procesamiento."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reinicia las estadísticas."""
        self.stats = {
            'processed': 0,
            'corrected': 0,
            'failed': 0,
            'avg_confidence': 0.0,
            'total_changes': 0,
            'cache_hits': 0,
            'batch_calls': 0,
            'individual_calls': 0,
            'pydantic_validations': 0,
            'verification_passed': 0,
            'verification_failed': 0,
            'verification_reverted': 0
        }
        if self.dual_model_enabled:
            self.stats.update({
                'easy_count': 0,
                'hard_count': 0,
                'easy_time': 0.0,
                'hard_time': 0.0,
                'local_failures': 0,
                'local_fallback_to_remote': 0
            })


def test_connection(host: str = "http://localhost:11434", model: str = "qwen3:14b"):
    """Prueba la conexión y el modelo."""
    print(f"🔌 Probando conexión a {host}...")
    
    try:
        response = requests.get(f"{host}/api/tags", timeout=10)
        if response.status_code != 200:
            print(f"❌ Error: servidor no responde correctamente")
            return False
        
        models = response.json().get('models', [])
        print(f"✓ Servidor Ollama disponible")
        print(f"  Modelos: {[m.get('name') for m in models]}")
        
        model_available = any(model in m.get('name', '') for m in models)
        if not model_available:
            print(f"⚠️  Modelo {model} no encontrado")
            return False
        
        print(f"✓ Modelo {model} disponible")
        
        # Prueba de corrección individual
        print(f"\n📝 Probando corrección individual...")
        corrector = TextCorrectorLLM(host, model, enable_cache=False)
        
        test_text = "que es el marketing digital y por que es importante"
        corrected, meta = corrector.correct(test_text)
        
        print(f"  Original:  {test_text}")
        print(f"  Corregido: {corrected}")
        print(f"  Confianza: {meta.get('confianza', 'N/A')}")
        
        # Prueba de batch
        print(f"\n📦 Probando corrección en batch...")
        test_texts = [
            "como se hace un podcast exitoso",
            "por que la ia esta cambiando todo",
            "donde puedo aprender mas sobre seo"
        ]
        
        batch_results = corrector.correct_batch_optimized(test_texts)
        for i, (original, (corrected, meta)) in enumerate(zip(test_texts, batch_results)):
            print(f"\n  [{i}] Original:  {original}")
            print(f"      Corregido: {corrected}")
            print(f"      Confianza: {meta.get('confianza', 'N/A')}")
        
        print(f"\n📊 Estadísticas: {corrector.get_stats()}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"❌ No se puede conectar a {host}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == '__main__':
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen3:14b"
    
    print("=" * 60)
    print("  TEST: Text Corrector LLM (Ollama) - Optimizado")
    print("=" * 60)
    print()
    
    if test_connection(host, model):
        print("\n✅ Todas las pruebas pasaron")
    else:
        print("\n❌ Algunas pruebas fallaron")
        sys.exit(1)
