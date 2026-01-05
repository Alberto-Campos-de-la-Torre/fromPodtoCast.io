"""
Cliente MCP (Model Context Protocol) para comunicarse con el servidor de diccionario.

Proporciona una interfaz Python simple para consultar recursos y usar herramientas
del servidor MCP.
"""

import json
import asyncio
import subprocess
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging


class MCPClient:
    """Cliente para el servidor MCP de diccionario."""
    
    def __init__(
        self,
        server_command: Optional[List[str]] = None,
        cache_enabled: bool = True
    ):
        """
        Inicializa el cliente MCP.
        
        Args:
            server_command: Comando para iniciar el servidor MCP (stdio mode)
            cache_enabled: Habilitar caché de respuestas
        """
        self.server_command = server_command
        self.cache_enabled = cache_enabled
        self.cache: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
        # Si no se especifica comando, usar el por defecto
        if not self.server_command:
            project_root = Path(__file__).parent.parent.parent
            server_script = project_root / "src" / "mcp" / "mcp_dictionary_server.py"
            self.server_command = ["python", str(server_script)]
        
        self.logger.info(f"✓ Cliente MCP inicializado: {' '.join(self.server_command)}")
    
    def _get_cache_key(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Genera clave de caché para una llamada."""
        return f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
    
    def _call_server_sync(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Llama al servidor MCP de forma síncrona (blocking).
        
        Args:
            tool_name: Nombre de la herramienta a ejecutar
            arguments: Argumentos para la herramienta
            
        Returns:
            Resultado de la herramienta
        """
        # Verificar caché
        if self.cache_enabled:
            cache_key = self._get_cache_key(tool_name, arguments)
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        try:
            # Construir request MCP
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            # Ejecutar servidor y enviar request
            process = subprocess.Popen(
                self.server_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Enviar request
            stdout, stderr = process.communicate(
                input=json.dumps(request) + "\n",
                timeout=10
            )
            
            # Parsear respuesta
            if stdout:
                try:
                    # MCP puede devolver múltiples líneas JSON
                    lines = stdout.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            response = json.loads(line)
                            if "result" in response:
                                result_data = response["result"]
                                
                                # Extraer texto de la respuesta MCP
                                if isinstance(result_data, list) and len(result_data) > 0:
                                    text_content = result_data[0].get("text", "{}")
                                    result = json.loads(text_content)
                                else:
                                    result = result_data
                                
                                # Guardar en caché
                                if self.cache_enabled:
                                    self.cache[cache_key] = result
                                
                                return result
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error parseando respuesta MCP: {e}")
                    self.logger.debug(f"Stdout: {stdout}")
            
            if stderr:
                self.logger.warning(f"MCP stderr: {stderr}")
            
            return {"error": "No se pudo obtener respuesta del servidor MCP"}
            
        except subprocess.TimeoutExpired:
            self.logger.error("Timeout esperando respuesta del servidor MCP")
            return {"error": "Timeout"}
        except Exception as e:
            self.logger.error(f"Error llamando servidor MCP: {e}")
            return {"error": str(e)}
    
    # ==================== MÉTODOS DE HERRAMIENTAS ====================
    
    def buscar_termino(self, palabra: str) -> Dict[str, Any]:
        """
        Busca un término exacto en el diccionario.
        
        Args:
            palabra: Palabra a buscar
            
        Returns:
            Información del término o error si no se encuentra
        """
        return self._call_server_sync("buscar_termino", {"palabra": palabra})
    
    def buscar_similar(self, palabra: str, max_resultados: int = 5) -> Dict[str, Any]:
        """
        Encuentra términos similares.
        
        Args:
            palabra: Palabra a buscar
            max_resultados: Número máximo de resultados
            
        Returns:
            Lista de términos similares
        """
        return self._call_server_sync("buscar_similar", {
            "palabra": palabra,
            "max_resultados": max_resultados
        })
    
    def validar_uso(self, palabra: str, contexto: str) -> Dict[str, Any]:
        """
        Valida si un término es correcto en un contexto dado.
        
        Args:
            palabra: Palabra a validar
            contexto: Frase o contexto donde aparece
            
        Returns:
            Validación del uso con sugerencias si es necesario
        """
        return self._call_server_sync("validar_uso", {
            "palabra": palabra,
            "contexto": contexto
        })
    
    def obtener_categoria(self, palabra: str) -> Dict[str, Any]:
        """
        Obtiene la categoría de un término.
        
        Args:
            palabra: Palabra a consultar
            
        Returns:
            Categoría y tipo del término
        """
        return self._call_server_sync("obtener_categoria", {"palabra": palabra})
    
    def verificar_correccion(self, original: str, corregido: str) -> Dict[str, Any]:
        """
        Verifica si una corrección propuesta es válida.
        
        Args:
            original: Texto original
            corregido: Texto corregido propuesto
            
        Returns:
            Validación de la corrección
        """
        return self._call_server_sync("verificar_correccion", {
            "original": original,
            "corregido": corregido
        })
    
    def limpiar_cache(self) -> None:
        """Limpia el caché de respuestas."""
        self.cache.clear()
        self.logger.info("✓ Caché limpiado")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Obtiene estadísticas del caché."""
        return {
            "entradas": len(self.cache),
            "habilitado": self.cache_enabled
        }


# ==================== WRAPPER SÍNCRONO SIMPLIFICADO ====================

class SimpleMCPClient:
    """
    Cliente MCP simplificado sin dependencias de MCP SDK.
    Usa llamadas directas al servidor via subprocess.
    """
    
    def __init__(self, dictionary_path: Optional[str] = None):
        """
        Inicializa cliente simple.
        
        Args:
            dictionary_path: Ruta al diccionario (opcional)
        """
        # Cargar diccionario directamente
        if not dictionary_path:
            project_root = Path(__file__).parent.parent.parent
            dictionary_path = str(project_root / "data" / "diccionario_base.json")
        
        self.dictionary_path = Path(dictionary_path)
        self.logger = logging.getLogger(__name__)
        self._load_dictionary()
    
    def _load_dictionary(self) -> None:
        """Carga el diccionario localmente."""
        try:
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                self.dictionary = json.load(f)
            self.logger.info(f"✓ Diccionario cargado: {len(self.dictionary.get('terminos', {}))} términos")
        except Exception as e:
            self.logger.error(f"Error cargando diccionario: {e}")
            self.dictionary = {"terminos": {}, "categorias": {}}
    
    def buscar_termino(self, palabra: str) -> Dict[str, Any]:
        """Busca término en diccionario local."""
        terminos = self.dictionary.get('terminos', {})
        
        for key, value in terminos.items():
            if key.lower() == palabra.lower():
                return {
                    "encontrado": True,
                    "termino": key,
                    "datos": value
                }
        
        return {
            "encontrado": False,
            "mensaje": f"Término '{palabra}' no encontrado"
        }
    
    def buscar_similar(self, palabra: str, max_resultados: int = 5) -> Dict[str, Any]:
        """Encuentra términos similares."""
        from difflib import get_close_matches
        
        terminos = self.dictionary.get('terminos', {})
        similares = get_close_matches(
            palabra,
            list(terminos.keys()),
            n=max_resultados,
            cutoff=0.6
        )
        
        resultados = [{"termino": s, "datos": terminos[s]} for s in similares]
        
        return {
            "palabra_buscada": palabra,
            "encontrados": len(resultados),
            "resultados": resultados
        }
    
    def validar_uso(self, palabra: str, contexto: str) -> Dict[str, Any]:
        """
        Valida uso de término.
        
        IMPORTANTE: Este diccionario es TÉCNICO. Si una palabra no está,
        NO significa que sea incorrecta - simplemente no es un término técnico.
        
        Retorna:
        - valido=True, mantener_original=True: Regionalismo/término protegido, NO corregir
        - valido=True: Término técnico válido
        - valido=None (neutral): Palabra común no en diccionario, usar criterio del LLM
        """
        resultado = self.buscar_termino(palabra)
        
        if not resultado.get("encontrado"):
            # CAMBIO: No encontrado = NEUTRAL, no inválido
            # El diccionario es técnico, no contiene todas las palabras del español
            return {
                "valido": None,  # Neutral - dejar al LLM decidir
                "en_diccionario": False,
                "razon": "Palabra común (no es término técnico)",
                "sugerencias": []  # No sugerimos nada, es palabra normal
            }
        
        termino_info = resultado["datos"]
        
        # Verificar si debe mantener original (regionalismo/término protegido)
        if termino_info.get('mantener_original') or termino_info.get('no_corregir'):
            return {
                "valido": True,
                "en_diccionario": True,
                "es_regionalismo": True,
                "mantener_original": True,
                "mensaje": f"'{palabra}' es un término protegido, debe mantenerse tal cual"
            }
        
        # Término técnico válido
        return {
            "valido": True,
            "en_diccionario": True,
            "forma_correcta": termino_info.get('forma_correcta', palabra),
            "tipo": termino_info.get('tipo'),
            "categoria": termino_info.get('categoria')
        }
    
    def verificar_correccion(self, original: str, corregido: str) -> Dict[str, Any]:
        """Verifica corrección propuesta."""
        terminos = self.dictionary.get('terminos', {})
        cambios = []
        
        # Detectar cambios simples
        palabras_orig = original.lower().split()
        palabras_corr = corregido.split()
        
        for orig, corr in zip(palabras_orig, palabras_corr):
            if orig != corr.lower():
                # Verificar si es corrección válida
                for termino, info in terminos.items():
                    variantes = info.get('variantes_incorrectas', [])
                    forma_correcta = info.get('forma_correcta', termino)
                    
                    if orig in [v.lower() for v in variantes] and corr == forma_correcta:
                        cambios.append({
                            "original": orig,
                            "corregido": corr,
                            "valido": True,
                            "razon": f"Corrección a '{forma_correcta}'"
                        })
        
        return {
            "cambios_validados": len(cambios),
            "cambios": cambios,
            "confianza": 1.0 if cambios else 0.7
        }
