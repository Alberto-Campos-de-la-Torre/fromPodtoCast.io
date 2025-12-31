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
    
    # Master prompt para el modelo (individual) - OPTIMIZADO
    SYSTEM_PROMPT = """Eres un corrector experto de transcripciones de audio en español mexicano con 20 años de experiencia en podcasts.

## TU ROL
Actúas como un editor profesional especializado en transcripciones automáticas de Whisper para podcasts mexicanos.

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

### ❌ NO CORREGIR (MANTENER TAL CUAL):
- Regionalismos: güey, chido, neta, órale, chamba, morro, chale, fresa
- Muletillas: pues, este, o sea, bueno, ¿no?, ¿verdad?
- Expresiones coloquiales: no manches, qué onda, está cañón, ni modo
- Estilo informal del hablante
- Repeticiones intencionales
- Pausas o titubeos representados

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

    # Prompt para batch processing - OPTIMIZADO
    BATCH_SYSTEM_PROMPT = """Eres un corrector experto de transcripciones de audio en español mexicano.

## ROL
Editor profesional especializado en corrección de transcripciones automáticas de podcasts.

## ⚠️ REGLA CRÍTICA: PRESERVAR EL CONTENIDO ORIGINAL
- NUNCA agregues información nueva
- NUNCA elimines contenido del original  
- NUNCA cambies el significado
- La longitud de cada texto corregido debe ser similar al original
- Solo corrige errores de ortografía, puntuación y format

## CORRECCIONES PERMITIDAS

### ✅ CORREGIR:
- Tildes y ortografía
- Puntuación (¿?, ¡!, comas, puntos)
- Mayúsculas (nombres propios, inicio de oración)
- Marcas: YouTube, TikTok, Instagram, ChatGPT, Google
- Acrónimos: IA, SEO, API, URL

### ❌ NO CORREGIR:
- Regionalismos mexicanos (güey, chido, neta, órale)
- Muletillas naturales (pues, este, o sea)
- Expresiones coloquiales

## GLOSARIO
{glosario_context}

## EJEMPLOS DE CORRECCIÓN CORRECTA

Entrada 0: "que es el marketing digital y por que es importante"
Salida 0: {{"id": 0, "texto_corregido": "¿Qué es el marketing digital y por qué es importante?", "cambios": ["Añadido ¿", "qué con tilde", "por qué separado"], "confianza": 0.95}}

Entrada 1: "vamos a hablar de chat gpt y de youtube"
Salida 1: {{"id": 1, "texto_corregido": "Vamos a hablar de ChatGPT y de YouTube.", "cambios": ["ChatGPT", "YouTube", "punto final"], "confianza": 0.92}}

Entrada 2: "pues si guey esta bien chido el podcast"
Salida 2: {{"id": 2, "texto_corregido": "Pues sí güey, está bien chido el podcast.", "cambios": ["sí con tilde", "Mayúscula inicial", "coma después de güey"], "confianza": 0.90}}

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

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "qwen3:14b",
        glosario_path: Optional[str] = None,
        timeout: int = 300,
        max_retries: int = 3,
        batch_size: int = 5,
        enable_cache: bool = True,
        cache_file: Optional[str] = None,
        max_workers: int = 2,
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
        
        # Procesar textos no cacheados en batches
        if uncached_texts:
            num_batches = (len(uncached_texts) + batch_size - 1) // batch_size
            cache_msg = f" (caché: {self.stats['cache_hits']})" if self.stats['cache_hits'] > 0 else ""
            verify_msg = " [verificación ON]" if should_verify and self.verifier else ""
            print(f"   📦 Procesando {len(uncached_texts)} textos en {num_batches} batches{cache_msg}{verify_msg}")
            
            # Crear barra de progreso para batches
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
            
            pydantic_validations = 0
            verification_passed = 0
            verification_failed = 0
            verification_reverted = 0
            
            for batch_start in batch_iterator:
                batch_end = min(batch_start + batch_size, len(uncached_texts))
                batch_texts = uncached_texts[batch_start:batch_end]
                batch_indices = uncached_indices[batch_start:batch_end]
                
                batch_results = self._process_batch(batch_texts)
                
                # Verificar correcciones si está habilitado
                if should_verify and self.verifier:
                    verified_results = []
                    for idx, (original_text, (corrected_text, meta)) in enumerate(zip(batch_texts, batch_results)):
                        if 'error' not in meta:
                            # Verificar la corrección
                            llm_conf = meta.get('confianza', 1.0)
                            v_result = self.verifier.verify(original_text, corrected_text, llm_conf)
                            
                            if v_result.is_valid:
                                verification_passed += 1
                                meta['verification'] = {
                                    'passed': True,
                                    'score': v_result.confidence_score,
                                    'checks_passed': v_result.checks_passed
                                }
                                verified_results.append((corrected_text, meta))
                            else:
                                verification_failed += 1
                                # Revertir al texto original
                                verification_reverted += 1
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
                    
                    batch_results = verified_results
                
                for idx, (original_idx, result) in enumerate(zip(batch_indices, batch_results)):
                    results[original_idx] = result
                    
                    # Contar validaciones Pydantic exitosas
                    if 'error' not in result[1] and result[1].get('pydantic_validated'):
                        pydantic_validations += 1
                    
                    # Guardar en caché si fue exitoso y verificado
                    if 'error' not in result[1]:
                        # Solo cachear si pasó verificación o verificación está deshabilitada
                        verification_info = result[1].get('verification', {})
                        if not should_verify or verification_info.get('passed', True):
                            self._add_to_cache(
                                batch_texts[idx],
                                {'texto_corregido': result[0], **result[1]}
                            )
            
            # Actualizar estadísticas de verificación
            self.stats['verification_passed'] += verification_passed
            self.stats['verification_failed'] += verification_failed
            self.stats['verification_reverted'] += verification_reverted
            
            # Mostrar resumen de validación Pydantic
            if PYDANTIC_AVAILABLE and pydantic_validations > 0:
                print(f"   🔷 Pydantic: {pydantic_validations} respuestas validadas correctamente")
            
            # Mostrar resumen de verificación
            if should_verify and self.verifier and (verification_passed > 0 or verification_failed > 0):
                print(f"   🔍 Verificación: {verification_passed} OK, {verification_failed} fallaron ({verification_reverted} revertidos)")
        
        # Guardar caché al final
        if self.enable_cache:
            self._save_cache()
        
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
        max_retries: int = 3
    ) -> Optional[str]:
        """
        Llama a la API de Ollama con reintentos y backoff exponencial.
        
        Maneja error 500, timeouts y problemas de conexión automáticamente.
        """
        import time as time_module
        
        actual_timeout = timeout or self.timeout
        
        # Qwen3 tiene modo "thinking" activado por defecto
        # Agregar /no_think para desactivarlo y obtener respuesta directa
        is_qwen3 = 'qwen3' in self.model.lower()
        final_system = f"/no_think\n{system_prompt}" if is_qwen3 else system_prompt
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": user_prompt,
                        "system": final_system,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "top_p": 0.9,
                            "num_predict": 4096 if is_qwen3 else 2048,  # Más tokens para qwen3
                        }
                    },
                    timeout=actual_timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Qwen3 puede poner respuesta en 'response' o 'thinking'
                    result = data.get('response', '')
                    if not result and is_qwen3:
                        # Si response está vacío, extraer del thinking
                        thinking = data.get('thinking', '')
                        if thinking:
                            self.logger.debug(f"Usando thinking como respuesta para qwen3")
                            result = thinking
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
