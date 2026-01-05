"""
Verificador de texto con Model Context Protocol (MCP).

Segunda fase de verificación que usa el diccionario MCP para validar
y mejorar las correcciones de texto realizadas por el LLM.
"""

import json
import re
import logging
import requests
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass

# Cliente MCP
try:
    from .mcp.mcp_client import SimpleMCPClient
    MCP_CLIENT_AVAILABLE = True
except ImportError:
    try:
        from mcp.mcp_client import SimpleMCPClient
        MCP_CLIENT_AVAILABLE = True
    except ImportError:
        MCP_CLIENT_AVAILABLE = False
        SimpleMCPClient = None


@dataclass
class VerificationResult:
    """Resultado de verificación de una corrección."""
    texto_verificado: str
    cambios_aplicados: List[str]
    cambios_revertidos: List[str]
    confianza: float
    validaciones_mcp: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class TextVerifierMCP:
    """
    Verificador de correcciones usando MCP.
    
    Segunda fase que consulta el diccionario MCP para validar
    las correcciones propuestas por el LLM.
    """
    
    VERIFICATION_PROMPT = """Eres un verificador experto de correcciones de texto.

Tu tarea es VALIDAR las correcciones ya aplicadas a un texto transcrito.

## CONTEXTO DEL DICCIONARIO
{dictionary_context}

## TEXTO ORIGINAL
{original_text}

## TEXTO CORREGIDO (PROPUESTO)
{corrected_text}

## VALIDACIONES DEL DICCIONARIO MCP
{mcp_validations}

## TU TAREA
Revisa las correcciones y determina si son válidas según el diccionario.

### REGLAS CRÍTICAS:
1. Si el diccionario indica "mantener_original: true", REVERTIR esa corrección
2. Si una corrección no está en el diccionario, evaluarla con escepticismo alto
3. Priorizar preservación de regionalismos y expresiones coloquiales
4. Aceptar solo correcciones de ortografía, puntuación y formato de marcas

## FORMATO DE RESPUESTA (JSON)
{{
  "texto_final": "Texto verificado final",
  "cambios_aceptados": ["cambio1", "cambio2"],
  "cambios_revertidos": ["cambio3"],
  "confianza": 0.95,
  "razonamiento": "Breve explicación"
}}

RESPONDE SOLO CON EL JSON."""
    
    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "qwen3:14b",
        dictionary_path: Optional[str] = None,
        timeout: int = 60,
        confidence_threshold: float = 0.80
    ):
        """
        Inicializa el verificador MCP.
        
        Args:
            ollama_host: URL del servidor Ollama
            model: Modelo LLM a usar para verificación
            dictionary_path: Ruta al diccionario MCP
            timeout: Timeout para requests
            confidence_threshold: Umbral de confianza mínima
        """
        if not ollama_host.startswith(('http://', 'https://')):
            ollama_host = f"http://{ollama_host}"
        
        self.ollama_host = ollama_host.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Inicializar cliente MCP
        if MCP_CLIENT_AVAILABLE:
            self.mcp_client = SimpleMCPClient(dictionary_path)
            self.logger.info("✓ Cliente MCP inicializado")
        else:
            self.logger.warning("⚠️  Cliente MCP no disponible, verificación limitada")
            self.mcp_client = None
        
        # Estadísticas
        self.stats = {
            'verificados': 0,
            'cambios_aceptados': 0,
            'cambios_revertidos': 0,
            'consultas_mcp': 0,
            'promedio_confianza': 0.0
        }
    
    def verify_correction(
        self,
        original_text: str,
        corrected_text: str,
        llm_metadata: Optional[Dict] = None
    ) -> VerificationResult:
        """
        Verifica una corrección individual usando MCP.
        
        Args:
            original_text: Texto original transcrito
            corrected_text: Texto corregido por LLM
            llm_metadata: Metadata de la corrección original
            
        Returns:
            Resultado de verificación con texto final
        """
        # Si no hay cambios, retornar directamente
        if original_text == corrected_text:
            return VerificationResult(
                texto_verificado=corrected_text,
                cambios_aplicados=[],
                cambios_revertidos=[],
                confianza=1.0,
                validaciones_mcp=[],
                metadata={'sin_cambios': True}
            )
        
        # Detectar cambios entre original y corregido
        cambios_detectados = self._detect_changes(original_text, corrected_text)
        
        # Consultar diccionario MCP para cada cambio
        validaciones_mcp = []
        if self.mcp_client:
            for cambio in cambios_detectados:
                self.stats['consultas_mcp'] += 1
                
                # Validar uso del término corregido
                validacion = self.mcp_client.validar_uso(
                    cambio['palabra_nueva'],
                    corrected_text
                )
                
                validaciones_mcp.append({
                    'cambio': cambio,
                    'validacion': validacion
                })
        
        # Construir contexto del diccionario para el prompt
        dict_context = self._format_dictionary_context(validaciones_mcp)
        mcp_validations_text = self._format_mcp_validations(validaciones_mcp)
        
        # Llamar al LLM para verificación final
        verification_response = self._call_verifier_llm(
            original_text,
            corrected_text,
            dict_context,
            mcp_validations_text
        )
        
        if verification_response:
            self.stats['verificados'] += 1
            self.stats['cambios_aceptados'] += len(verification_response.get('cambios_aceptados', []))
            self.stats['cambios_revertidos'] += len(verification_response.get('cambios_revertidos', []))
            
            # Actualizar promedio de confianza
            conf = verification_response.get('confianza', 0.8)
            n = self.stats['verificados']
            self.stats['promedio_confianza'] = (
                (self.stats['promedio_confianza'] * (n - 1) + conf) / n
            )
            
            return VerificationResult(
                texto_verificado=verification_response.get('texto_final', corrected_text),
                cambios_aplicados=verification_response.get('cambios_aceptados', []),
                cambios_revertidos=verification_response.get('cambios_revertidos', []),
                confianza=conf,
                validaciones_mcp=validaciones_mcp,
                metadata={
                    'modelo': self.model,
                    'mcp_consultas': len(validaciones_mcp),
                    'razonamiento': verification_response.get('razonamiento', '')
                }
            )
        
        # Si falla verificación, usar heurísticas simples
        return self._fallback_verification(
            original_text,
            corrected_text,
            validaciones_mcp
        )
    
    def verify_batch(
        self,
        corrections: List[Tuple[str, str, Dict]]
    ) -> List[VerificationResult]:
        """
        Verifica múltiples correcciones en lote.
        
        Args:
            corrections: Lista de (original, corregido, metadata)
            
        Returns:
            Lista de resultados de verificación
        """
        results = []
        
        try:
            from tqdm import tqdm
            iterator = tqdm(corrections, desc="   Verificando con MCP", unit="texto")
        except ImportError:
            iterator = corrections
        
        for original, corrected, metadata in iterator:
            result = self.verify_correction(original, corrected, metadata)
            results.append(result)
        
        return results
    
    def _detect_changes(self, original: str, corrected: str) -> List[Dict[str, Any]]:
        """Detecta cambios entre texto original y corregido."""
        cambios = []
        
        # Dividir en palabras (simplificado)
        palabras_orig = original.split()
        palabras_corr = corrected.split()
        
        # Detectar diferencias
        max_len = max(len(palabras_orig), len(palabras_corr))
        
        for i in range(max_len):
            palabra_orig = palabras_orig[i] if i < len(palabras_orig) else ""
            palabra_corr = palabras_corr[i] if i < len(palabras_corr) else ""
            
            if palabra_orig.lower() != palabra_corr.lower():
                cambios.append({
                    'posicion': i,
                    'palabra_original': palabra_orig,
                    'palabra_nueva': palabra_corr,
                    'tipo': self._classify_change(palabra_orig, palabra_corr)
                })
        
        return cambios
    
    def _classify_change(self, original: str, nuevo: str) -> str:
        """Clasifica el tipo de cambio."""
        if not original:
            return 'adicion'
        if not nuevo:
            return 'eliminacion'
        if original.lower() == nuevo.lower():
            return 'mayusculas'
        if re.sub(r'[^\w]', '', original.lower()) == re.sub(r'[^\w]', '', nuevo.lower()):
            return 'puntuacion'
        return 'palabra'
    
    def _format_dictionary_context(self, validaciones: List[Dict]) -> str:
        """Formatea el contexto del diccionario para el prompt."""
        if not validaciones:
            return "No hay información del diccionario disponible."
        
        lines = []
        for val in validaciones[:10]:  # Limitar a 10 para no saturar prompt
            cambio = val['cambio']
            validacion = val['validacion']
            
            palabra = cambio['palabra_nueva']
            
            if validacion.get('mantener_original'):
                lines.append(f"- '{palabra}': MANTENER ORIGINAL (regionalismo)")
            elif validacion.get('valido'):
                lines.append(f"- '{palabra}': VÁLIDO ({validacion.get('tipo', 'término')})")
            else:
                lines.append(f"- '{palabra}': NO ENCONTRADO en diccionario")
        
        return '\n'.join(lines)
    
    def _format_mcp_validations(self, validaciones: List[Dict]) -> str:
        """Formatea las validaciones MCP como texto estructurado."""
        if not validaciones:
            return "Sin validaciones MCP."
        
        return json.dumps(validaciones, ensure_ascii=False, indent=2)
    
    def _call_verifier_llm(
        self,
        original: str,
        corrected: str,
        dict_context: str,
        mcp_validations: str
    ) -> Optional[Dict]:
        """Llama al LLM verificador con contexto MCP."""
        
        prompt = self.VERIFICATION_PROMPT.format(
            dictionary_context=dict_context,
            original_text=original,
            corrected_text=corrected,
            mcp_validations=mcp_validations
        )
        
        try:
            # Qwen3 específico
            is_qwen3 = 'qwen3' in self.model.lower()
            system_prompt = "/no_think\nEres un verificador de correcciones." if is_qwen3 else "Eres un verificador de correcciones."
            
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.85,
                        "num_predict": 2048
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                result_text = data.get('response', '')
                
                # Limpiar y parsear JSON
                result_text = self._clean_json_response(result_text)
                
                try:
                    return json.loads(result_text)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Error parseando respuesta verificador: {e}")
                    return None
        
        except Exception as e:
            self.logger.error(f"Error llamando verificador LLM: {e}")
        
        return None
    
    def _clean_json_response(self, response: str) -> str:
        """Limpia respuesta JSON del LLM."""
        response = response.strip()
        
        # Extraer JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            response = json_match.group(0)
        
        # Remover prefijos comunes
        response = re.sub(r'^```json\s*', '', response)
        response = re.sub(r'\s*```$', '', response)
        
        return response
    
    def _fallback_verification(
        self,
        original: str,
        corrected: str,
        validaciones: List[Dict]
    ) -> VerificationResult:
        """Verificación de fallback sin LLM."""
        # Revisar validaciones MCP
        debe_revertir = False
        cambios_revertidos = []
        
        for val in validaciones:
            if val['validacion'].get('mantener_original'):
                debe_revertir = True
                cambios_revertidos.append(
                    f"Revertido '{val['cambio']['palabra_nueva']}' (regionalismo)"
                )
        
        # Si hay que revertir, usar original
        texto_final = original if debe_revertir else corrected
        
        return VerificationResult(
            texto_verificado=texto_final,
            cambios_aplicados=[] if debe_revertir else ['Cambios aceptados (sin LLM)'],
            cambios_revertidos=cambios_revertidos,
            confianza=0.70,
            validaciones_mcp=validaciones,
            metadata={'fallback': True}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de verificación."""
        return self.stats.copy()
