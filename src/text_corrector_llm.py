"""
Módulo para corrección avanzada de transcripciones usando LLM local (Ollama).
Utiliza el modelo qwen3:8b para correcciones contextuales de alta calidad.

Optimizaciones para 3060 Ti 8GB:
- Batch de textos (3 por llamada)
- Workers paralelos (2 simultáneos)
- Filtrado inteligente (skip textos sin errores)
"""
import json
import re
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time


class TextCorrectorLLM:
    """
    Corrector de texto usando LLM local via Ollama.
    
    Características:
    - Corrección contextual de errores de transcripción
    - Preservación de regionalismos y expresiones coloquiales
    - Puntuación y acentuación correcta
    - Formato de salida estructurado JSON
    - Procesamiento en batch y paralelo
    """
    
    # Master prompt para texto individual
    SYSTEM_PROMPT = """Eres un experto corrector de transcripciones de audio en español mexicano.

## TU TAREA
Corregir transcripciones de podcasts manteniendo:
1. La naturalidad del habla oral
2. Los regionalismos mexicanos (NO los corrijas)
3. El estilo y tono del hablante original

## REGLAS DE CORRECCIÓN

### DEBES CORREGIR:
- Errores ortográficos (tildes, letras)
- Puntuación faltante o incorrecta (¿?, ¡!, comas, puntos)
- Mayúsculas incorrectas (nombres propios, inicios de oración)
- Números a texto cuando sea natural (5 → cinco)
- Marcas y nombres: YouTube, TikTok, Instagram, ChatGPT, Google, etc.
- Acrónimos: IA, SEO, API, URL, etc.

### NO DEBES CORREGIR (MANTENER TAL CUAL):
- Regionalismos mexicanos: güey, chido, neta, órale, chamba, etc.
- Muletillas naturales: pues, este, o sea, bueno
- Expresiones coloquiales: no manches, qué onda, está cañón
- El estilo informal si es apropiado al contexto

### ERRORES COMUNES DE WHISPER A CORREGIR:
- "gemina" → "Gemini"
- "que es" al inicio → "¿Qué es"
- "por que" → "por qué" (en preguntas)
- "ai" → "IA"
- Falta de signos de apertura ¿ y ¡

## GLOSARIO DE REFERENCIA
{glosario_context}

## FORMATO DE RESPUESTA
Responde ÚNICAMENTE con un JSON válido con esta estructura exacta:
{{
  "texto_corregido": "El texto corregido completo",
  "cambios": ["cambio1", "cambio2"],
  "confianza": 0.95
}}

- texto_corregido: El texto final corregido
- cambios: Lista de correcciones aplicadas (máximo 5 más relevantes)
- confianza: Número entre 0 y 1 indicando confianza en la corrección

IMPORTANTE: 
- Responde SOLO con el JSON, sin explicaciones adicionales
- Si el texto está correcto, devuelve el mismo texto con cambios vacío
- No inventes contenido, solo corrige lo existente
- NO uses saltos de línea dentro del texto_corregido
- Escapa las comillas dentro del texto con \\"
- Mantén el JSON en UNA SOLA respuesta corta"""

    # Prompt para batch de textos
    BATCH_SYSTEM_PROMPT = """Eres un experto corrector de transcripciones de audio en español mexicano.

## TU TAREA
Corregir MÚLTIPLES transcripciones de podcasts manteniendo:
1. La naturalidad del habla oral
2. Los regionalismos mexicanos (NO los corrijas)
3. El estilo y tono del hablante original

## REGLAS DE CORRECCIÓN
- Corregir: tildes, puntuación (¿?, ¡!), mayúsculas, marcas (YouTube, TikTok, ChatGPT)
- NO corregir: regionalismos mexicanos (güey, chido, neta), muletillas naturales

## GLOSARIO
{glosario_context}

## FORMATO DE RESPUESTA
Responde con un JSON array. Cada elemento corresponde a un texto de entrada EN EL MISMO ORDEN:
[
  {{"texto_corregido": "texto 1 corregido", "cambios": ["cambio1"], "confianza": 0.95}},
  {{"texto_corregido": "texto 2 corregido", "cambios": [], "confianza": 0.98}},
  {{"texto_corregido": "texto 3 corregido", "cambios": ["cambio1"], "confianza": 0.90}}
]

IMPORTANTE:
- Responde SOLO con el JSON array
- Mantén el MISMO ORDEN que los textos de entrada
- Un elemento por cada texto de entrada"""

    # Patrones de errores comunes que requieren corrección
    ERROR_PATTERNS = [
        r'\bque es\b',           # Falta ¿
        r'\bpor que\b',          # Falta tilde
        r'\bcomo es\b',          # Falta tilde
        r'\bai\b',               # IA
        r'\bgemina\b',           # Gemini
        r'\byoutube\b',          # YouTube
        r'\btiktok\b',           # TikTok
        r'\binstagram\b',        # Instagram
        r'\bwhatsapp\b',         # WhatsApp
        r'\bchatgpt\b',          # ChatGPT
        r'\bgoogle\b',           # Google
        r'[?!][^¿¡]',            # Falta signo apertura
        r'\b\d{1,2}\b',          # Números pequeños
        r'\bmas\b',              # más
        r'\btambien\b',          # también
        r'\besta\b(?!\s+(?:cañón|chido))',  # está (no regionalismo)
    ]

    def __init__(
        self,
        ollama_host: str = "http://192.168.1.81:11434",
        model: str = "qwen3:8b",
        glosario_path: Optional[str] = None,
        timeout: int = 90,
        max_retries: int = 2,
        batch_size: int = 3,
        parallel_workers: int = 2,
        smart_filter: bool = True
    ):
        """
        Inicializa el corrector LLM.
        
        Args:
            ollama_host: URL del servidor Ollama
            model: Nombre del modelo a usar
            glosario_path: Ruta al archivo de glosario JSON
            timeout: Timeout para requests en segundos
            max_retries: Número máximo de reintentos
            batch_size: Número de textos por llamada (1-5)
            parallel_workers: Número de workers paralelos (1-4)
            smart_filter: Si True, salta textos sin errores detectables
        """
        self.ollama_host = ollama_host.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = min(max(1, batch_size), 5)  # Limitar 1-5
        self.parallel_workers = min(max(1, parallel_workers), 4)  # Limitar 1-4
        self.smart_filter = smart_filter
        self.logger = logging.getLogger(__name__)
        
        # Cargar glosario
        self.glosario = self._load_glosario(glosario_path)
        self.glosario_context = self._format_glosario_context()
        
        # Compilar patrones de error
        self._error_regex = re.compile(
            '|'.join(self.ERROR_PATTERNS), 
            re.IGNORECASE
        )
        
        # Estadísticas
        self.stats = {
            'processed': 0,
            'corrected': 0,
            'failed': 0,
            'skipped': 0,
            'avg_confidence': 0.0,
            'total_changes': 0,
            'batch_calls': 0,
            'time_saved': 0.0
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
        
        # Correcciones más relevantes (primeras 20 para batch)
        correcciones = self.glosario.get('correcciones', {})
        if correcciones:
            lines.append("Correcciones: " + ", ".join(
                f'"{k}"→"{v}"' for k, v in list(correcciones.items())[:20]
            ))
        
        # Términos a mantener (primeros 15)
        mantener = self.glosario.get('mantener', [])
        if mantener:
            lines.append("Mantener: " + ", ".join(mantener[:15]))
        
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
    
    def _needs_correction(self, text: str) -> bool:
        """Determina si un texto probablemente necesita corrección."""
        if not self.smart_filter:
            return True
        
        # Buscar patrones de error
        if self._error_regex.search(text):
            return True
        
        # Verificar si empieza con minúscula
        if text and text[0].islower():
            return True
        
        # Verificar puntuación básica
        if text.endswith('?') and '¿' not in text:
            return True
        if text.endswith('!') and '¡' not in text:
            return True
        
        return False
    
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
        
        # Filtrado inteligente
        if not self._needs_correction(text):
            self.stats['skipped'] += 1
            return text, {'skipped': True, 'confianza': 1.0}
        
        self.stats['processed'] += 1
        
        # Construir prompt
        system_prompt = self.SYSTEM_PROMPT.format(
            glosario_context=self.glosario_context
        )
        
        user_prompt = f"""Corrige la siguiente transcripción:

"{text}"

Responde SOLO con el JSON."""

        # Intentar corrección
        for attempt in range(self.max_retries):
            try:
                response = self._call_ollama(system_prompt, user_prompt)
                
                if response:
                    result = self._parse_response(response, text)
                    
                    if result['success']:
                        self.stats['corrected'] += 1
                        self.stats['total_changes'] += len(result.get('cambios', []))
                        
                        conf = result.get('confianza', 0.5)
                        n = self.stats['corrected']
                        self.stats['avg_confidence'] = (
                            (self.stats['avg_confidence'] * (n - 1) + conf) / n
                        )
                        
                        return result['texto_corregido'], {
                            'cambios': result.get('cambios', []),
                            'confianza': conf,
                            'modelo': self.model,
                            'intentos': attempt + 1
                        }
                
            except Exception as e:
                self.logger.warning(f"Intento {attempt + 1} falló: {e}")
                continue
        
        self.stats['failed'] += 1
        return text, {'error': 'max_retries_exceeded', 'original': True}
    
    def correct_batch_texts(self, texts: List[str]) -> List[Tuple[str, Dict]]:
        """
        Corrige múltiples textos en una sola llamada al LLM.
        
        Args:
            texts: Lista de textos a corregir (máximo batch_size)
            
        Returns:
            Lista de tuplas (texto_corregido, metadata)
        """
        if not texts:
            return []
        
        # Filtrar textos que necesitan corrección
        needs_correction = []
        results = [None] * len(texts)
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = (text, {'error': 'texto_vacío'})
            elif not self._needs_correction(text):
                results[i] = (text, {'skipped': True, 'confianza': 1.0})
                self.stats['skipped'] += 1
            else:
                needs_correction.append((i, text))
        
        if not needs_correction:
            return results
        
        # Construir prompt para batch
        system_prompt = self.BATCH_SYSTEM_PROMPT.format(
            glosario_context=self.glosario_context
        )
        
        texts_to_correct = [t for _, t in needs_correction]
        user_prompt = "Corrige estos textos:\n\n"
        for i, text in enumerate(texts_to_correct, 1):
            user_prompt += f'{i}. "{text}"\n'
        user_prompt += "\nResponde con el JSON array."
        
        self.stats['batch_calls'] += 1
        
        # Llamar al LLM
        for attempt in range(self.max_retries):
            try:
                response = self._call_ollama(system_prompt, user_prompt)
                
                if response:
                    batch_results = self._parse_batch_response(
                        response, texts_to_correct
                    )
                    
                    # Mapear resultados a posiciones originales
                    for j, (orig_idx, _) in enumerate(needs_correction):
                        if j < len(batch_results):
                            results[orig_idx] = batch_results[j]
                            self.stats['processed'] += 1
                            if batch_results[j][1].get('cambios'):
                                self.stats['corrected'] += 1
                        else:
                            results[orig_idx] = (
                                texts[orig_idx], 
                                {'error': 'batch_incomplete'}
                            )
                    
                    return results
                    
            except Exception as e:
                self.logger.warning(f"Batch intento {attempt + 1} falló: {e}")
                continue
        
        # Fallback: devolver originales
        for orig_idx, text in needs_correction:
            results[orig_idx] = (text, {'error': 'batch_failed'})
            self.stats['failed'] += 1
        
        return results
    
    def correct_parallel(
        self,
        entries: List[Dict],
        text_field: str = 'text',
        min_confidence: float = 0.7,
        progress_callback=None
    ) -> List[Dict]:
        """
        Corrige una lista de entradas en paralelo usando batches.
        
        Args:
            entries: Lista de diccionarios con campo de texto
            text_field: Nombre del campo que contiene el texto
            min_confidence: Confianza mínima para aceptar corrección
            progress_callback: Función callback(processed, total) para progreso
            
        Returns:
            Lista de entradas con texto corregido
        """
        if not entries:
            return entries
        
        start_time = time.time()
        results = [None] * len(entries)
        
        # Agrupar en batches
        batches = []
        current_batch = []
        current_indices = []
        
        for i, entry in enumerate(entries):
            if text_field in entry and entry[text_field]:
                current_batch.append(entry[text_field])
                current_indices.append(i)
                
                if len(current_batch) >= self.batch_size:
                    batches.append((current_indices.copy(), current_batch.copy()))
                    current_batch = []
                    current_indices = []
            else:
                results[i] = entry.copy()
        
        # Último batch
        if current_batch:
            batches.append((current_indices, current_batch))
        
        processed_count = 0
        total_batches = len(batches)
        
        # Procesar batches en paralelo
        def process_batch(batch_data):
            indices, texts = batch_data
            return indices, self.correct_batch_texts(texts)
        
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = {
                executor.submit(process_batch, batch): batch 
                for batch in batches
            }
            
            for future in as_completed(futures):
                try:
                    indices, batch_results = future.result()
                    
                    for j, idx in enumerate(indices):
                        entry = entries[idx].copy()
                        
                        if j < len(batch_results):
                            corrected, meta = batch_results[j]
                            
                            confianza = meta.get('confianza', 0)
                            if confianza >= min_confidence and not meta.get('skipped'):
                                original = entry[text_field]
                                entry[text_field] = corrected
                                
                                if corrected != original:
                                    entry['llm_correction'] = {
                                        'original': original,
                                        'cambios': meta.get('cambios', []),
                                        'confianza': confianza
                                    }
                            elif meta.get('skipped'):
                                entry['llm_skipped'] = True
                        
                        results[idx] = entry
                    
                    processed_count += 1
                    if progress_callback:
                        progress_callback(processed_count, total_batches)
                        
                except Exception as e:
                    self.logger.error(f"Error en batch: {e}")
        
        # Rellenar None con originales
        for i, result in enumerate(results):
            if result is None:
                results[i] = entries[i].copy()
        
        elapsed = time.time() - start_time
        self.stats['time_saved'] = max(0, (len(entries) * 5) - elapsed)  # Estimado
        
        return results
    
    def _call_ollama(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Llama a la API de Ollama."""
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": user_prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Más bajo para consistencia
                        "top_p": 0.85,
                        "num_predict": 2048,  # Más para batches
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                self.logger.error(
                    f"Error Ollama: {response.status_code} - {response.text}"
                )
                return None
                
        except requests.exceptions.Timeout:
            self.logger.error("Timeout en llamada a Ollama")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error de conexión: {e}")
            return None
    
    def _parse_response(self, response: str, original_text: str) -> Dict:
        """Parsea la respuesta JSON del LLM."""
        try:
            response = response.strip()
            
            # Eliminar bloques de código markdown
            response = re.sub(r'^```json\s*', '', response)
            response = re.sub(r'^```\s*', '', response)
            response = re.sub(r'\s*```$', '', response)
            
            # Extraer JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                response = json_match.group(0)
            
            response = self._repair_truncated_json(response)
            data = json.loads(response)
            
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
            
            extracted = self._extract_text_from_malformed(response, original_text)
            if extracted:
                return extracted
            
            return {
                'success': True,
                'texto_corregido': original_text,
                'cambios': ['json_parse_error'],
                'confianza': 0.3
            }
        
        except Exception as e:
            self.logger.warning(f"Error procesando respuesta: {e}")
            return {
                'success': True,
                'texto_corregido': original_text,
                'cambios': ['processing_error'],
                'confianza': 0.3
            }
    
    def _parse_batch_response(
        self, response: str, original_texts: List[str]
    ) -> List[Tuple[str, Dict]]:
        """Parsea la respuesta JSON array del batch."""
        results = []
        
        try:
            response = response.strip()
            
            # Limpiar markdown
            response = re.sub(r'^```json\s*', '', response)
            response = re.sub(r'^```\s*', '', response)
            response = re.sub(r'\s*```$', '', response)
            
            # Extraer array JSON
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                response = json_match.group(0)
            
            # Intentar reparar JSON
            response = self._repair_truncated_json(response)
            data = json.loads(response)
            
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if i >= len(original_texts):
                        break
                    
                    if isinstance(item, dict):
                        texto = item.get('texto_corregido', original_texts[i])
                        cambios = item.get('cambios', [])
                        confianza = float(item.get('confianza', 0.5))
                        confianza = max(0.0, min(1.0, confianza))
                        
                        results.append((texto, {
                            'cambios': cambios if isinstance(cambios, list) else [],
                            'confianza': confianza
                        }))
                    else:
                        results.append((original_texts[i], {'error': 'invalid_item'}))
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Error parseando batch JSON: {e}")
            # Intentar extraer respuestas individuales
            results = self._extract_batch_from_malformed(response, original_texts)
        
        except Exception as e:
            self.logger.warning(f"Error procesando batch: {e}")
        
        # Rellenar faltantes
        while len(results) < len(original_texts):
            idx = len(results)
            results.append((original_texts[idx], {'error': 'missing_result'}))
        
        return results
    
    def _extract_batch_from_malformed(
        self, response: str, original_texts: List[str]
    ) -> List[Tuple[str, Dict]]:
        """Intenta extraer resultados de un batch malformado."""
        results = []
        
        # Buscar todos los texto_corregido
        pattern = r'"texto_corregido"\s*:\s*"((?:[^"\\]|\\.)*)"'
        matches = re.findall(pattern, response)
        
        for i, match in enumerate(matches):
            if i >= len(original_texts):
                break
            texto = match.replace('\\"', '"').replace('\\n', ' ').strip()
            results.append((texto, {'cambios': ['batch_extraction'], 'confianza': 0.5}))
        
        return results
    
    def _repair_truncated_json(self, json_str: str) -> str:
        """Intenta reparar JSON truncado."""
        quote_count = json_str.count('"') - json_str.count('\\"')
        
        if quote_count % 2 != 0:
            json_str = json_str.rstrip()
            if not json_str.endswith('"'):
                json_str += '"'
        
        json_str = json_str.rstrip()
        
        # Cerrar brackets
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        json_str += ']' * (open_brackets - close_brackets)
        
        # Cerrar braces
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        json_str += '}' * (open_braces - close_braces)
        
        return json_str
    
    def _extract_text_from_malformed(self, response: str, original: str) -> Optional[Dict]:
        """Extrae el texto corregido de una respuesta malformada."""
        patterns = [
            r'"texto_corregido"\s*:\s*"((?:[^"\\]|\\.)*)"',
            r'"texto_corregido"\s*:\s*"(.*?)(?:"|$)',
            r'texto_corregido["\s:]+([^"]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                texto = match.group(1)
                texto = texto.replace('\\"', '"')
                texto = texto.replace('\\n', ' ')
                texto = texto.strip()
                
                if texto and len(texto) > 5:
                    return {
                        'success': True,
                        'texto_corregido': texto,
                        'cambios': ['extracción_parcial'],
                        'confianza': 0.5
                    }
        
        return None
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas de procesamiento."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reinicia las estadísticas."""
        self.stats = {
            'processed': 0,
            'corrected': 0,
            'failed': 0,
            'skipped': 0,
            'avg_confidence': 0.0,
            'total_changes': 0,
            'batch_calls': 0,
            'time_saved': 0.0
        }


def test_connection(host: str = "http://192.168.1.81:11434", model: str = "qwen3:8b"):
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
        
        # Prueba individual
        print(f"\n📝 Probando corrección individual...")
        corrector = TextCorrectorLLM(host, model, batch_size=3, parallel_workers=2)
        
        test_text = "que es el marketing digital y por que es importante"
        corrected, meta = corrector.correct(test_text)
        
        print(f"  Original:  {test_text}")
        print(f"  Corregido: {corrected}")
        print(f"  Confianza: {meta.get('confianza', 'N/A')}")
        
        # Prueba batch
        print(f"\n📝 Probando corrección en batch (3 textos)...")
        test_texts = [
            "que es el seo y como funciona",
            "youtube es una plataforma de videos",
            "la ia esta cambiando todo"
        ]
        
        start = time.time()
        batch_results = corrector.correct_batch_texts(test_texts)
        elapsed = time.time() - start
        
        for i, (corr, meta) in enumerate(batch_results):
            print(f"  {i+1}. {corr[:50]}... (conf: {meta.get('confianza', 'N/A')})")
        
        print(f"\n  ⏱️  Tiempo batch: {elapsed:.2f}s (vs ~{len(test_texts)*5}s individual)")
        print(f"  📊 Stats: {corrector.get_stats()}")
        
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
    
    host = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.1.81:11434"
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen3:8b"
    
    print("=" * 60)
    print("  TEST: Text Corrector LLM (Optimizado)")
    print("  Batch size: 3 | Workers: 2 | Smart filter: ON")
    print("=" * 60)
    print()
    
    if test_connection(host, model):
        print("\n✅ Todas las pruebas pasaron")
    else:
        print("\n❌ Algunas pruebas fallaron")
        sys.exit(1)
