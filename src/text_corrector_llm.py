"""
Módulo para corrección avanzada de transcripciones usando LLM local (Ollama).
Utiliza el modelo qwen3:8b para correcciones contextuales de alta calidad.
"""
import json
import re
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging


class TextCorrectorLLM:
    """
    Corrector de texto usando LLM local via Ollama.
    
    Características:
    - Corrección contextual de errores de transcripción
    - Preservación de regionalismos y expresiones coloquiales
    - Puntuación y acentuación correcta
    - Formato de salida estructurado JSON
    """
    
    # Master prompt para el modelo
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
- No inventes contenido, solo corrige lo existente"""

    def __init__(
        self,
        ollama_host: str = "http://192.168.1.81:11434",
        model: str = "qwen3:8b",
        glosario_path: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3
    ):
        """
        Inicializa el corrector LLM.
        
        Args:
            ollama_host: URL del servidor Ollama
            model: Nombre del modelo a usar
            glosario_path: Ruta al archivo de glosario JSON
            timeout: Timeout para requests en segundos
            max_retries: Número máximo de reintentos
        """
        self.ollama_host = ollama_host.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
        
        # Cargar glosario
        self.glosario = self._load_glosario(glosario_path)
        self.glosario_context = self._format_glosario_context()
        
        # Estadísticas
        self.stats = {
            'processed': 0,
            'corrected': 0,
            'failed': 0,
            'avg_confidence': 0.0,
            'total_changes': 0
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
                
                # Verificar si el modelo está disponible
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
        
        self.stats['processed'] += 1
        
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
                        
                        return result['texto_corregido'], {
                            'cambios': result.get('cambios', []),
                            'confianza': conf,
                            'modelo': self.model,
                            'intentos': attempt + 1
                        }
                
            except Exception as e:
                self.logger.warning(
                    f"Intento {attempt + 1}/{self.max_retries} falló: {e}"
                )
                continue
        
        # Si fallan todos los intentos, devolver texto original
        self.stats['failed'] += 1
        return text, {'error': 'max_retries_exceeded', 'original': True}
    
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
                        "temperature": 0.3,  # Baja para consistencia
                        "top_p": 0.9,
                        "num_predict": 1024,
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
            # Limpiar respuesta
            response = response.strip()
            
            # Intentar extraer JSON si hay texto adicional
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                response = json_match.group(0)
            
            # Parsear JSON
            data = json.loads(response)
            
            texto = data.get('texto_corregido', original_text)
            cambios = data.get('cambios', [])
            confianza = float(data.get('confianza', 0.5))
            
            # Validar confianza
            confianza = max(0.0, min(1.0, confianza))
            
            return {
                'success': True,
                'texto_corregido': texto,
                'cambios': cambios if isinstance(cambios, list) else [],
                'confianza': confianza
            }
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Error parseando JSON: {e}")
            
            # Intentar extraer texto corregido de respuesta malformada
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
        min_confidence: float = 0.7
    ) -> List[Dict]:
        """
        Corrige un lote de entradas.
        
        Args:
            entries: Lista de diccionarios con campo de texto
            text_field: Nombre del campo que contiene el texto
            min_confidence: Confianza mínima para aceptar corrección
            
        Returns:
            Lista de entradas con texto corregido
        """
        processed = []
        
        for i, entry in enumerate(entries):
            if text_field not in entry:
                processed.append(entry)
                continue
            
            original = entry[text_field]
            corrected, metadata = self.correct(original)
            
            new_entry = entry.copy()
            
            # Solo aplicar si la confianza es suficiente
            if metadata.get('confianza', 0) >= min_confidence:
                new_entry[text_field] = corrected
                
                if corrected != original:
                    new_entry['text_original'] = original
                    new_entry['llm_correction'] = metadata
            else:
                # Mantener original pero marcar
                new_entry['llm_low_confidence'] = metadata.get('confianza', 0)
            
            processed.append(new_entry)
            
            # Log de progreso
            if (i + 1) % 10 == 0:
                self.logger.info(f"Procesados {i + 1}/{len(entries)} segmentos")
        
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
            'total_changes': 0
        }


def test_connection(host: str = "http://192.168.1.81:11434", model: str = "qwen3:8b"):
    """Prueba la conexión y el modelo."""
    print(f"🔌 Probando conexión a {host}...")
    
    try:
        # Verificar servidor
        response = requests.get(f"{host}/api/tags", timeout=10)
        if response.status_code != 200:
            print(f"❌ Error: servidor no responde correctamente")
            return False
        
        models = response.json().get('models', [])
        print(f"✓ Servidor Ollama disponible")
        print(f"  Modelos: {[m.get('name') for m in models]}")
        
        # Verificar modelo
        model_available = any(model in m.get('name', '') for m in models)
        if not model_available:
            print(f"⚠️  Modelo {model} no encontrado")
            print(f"   Puedes instalarlo con: ollama pull {model}")
            return False
        
        print(f"✓ Modelo {model} disponible")
        
        # Prueba rápida
        print(f"\n📝 Probando corrección...")
        corrector = TextCorrectorLLM(host, model)
        
        test_text = "que es el marketing digital y por que es importante para las empresas"
        corrected, meta = corrector.correct(test_text)
        
        print(f"  Original:  {test_text}")
        print(f"  Corregido: {corrected}")
        print(f"  Confianza: {meta.get('confianza', 'N/A')}")
        print(f"  Cambios:   {meta.get('cambios', [])}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"❌ No se puede conectar a {host}")
        print("   Verifica que Ollama esté corriendo en esa dirección")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == '__main__':
    import sys
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Argumentos opcionales
    host = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.1.81:11434"
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen3:8b"
    
    print("=" * 60)
    print("  TEST: Text Corrector LLM (Ollama)")
    print("=" * 60)
    print()
    
    if test_connection(host, model):
        print("\n✅ Todas las pruebas pasaron")
    else:
        print("\n❌ Algunas pruebas fallaron")
        sys.exit(1)

