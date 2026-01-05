"""
Servidor MCP (Model Context Protocol) para diccionario de términos.

Expone recursos y herramientas que el modelo LLM puede consultar
durante la verificación de transcripciones.

Recursos disponibles:
- diccionario://terminos/lista - Lista todos los términos
- diccionario://termino/{palabra} - Definición de un término específico
- diccionario://categoria/{nombre} - Términos de una categoría

Herramientas disponibles:
- buscar_termino - Busca un término exacto
- buscar_similar - Encuentra términos similares
- validar_uso - Valida el uso de un término en contexto
- obtener_categoria - Obtiene la categoría de un término
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from difflib import get_close_matches
import logging

# MCP Protocol
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("ERROR: mcp package not installed. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)


class DictionaryMCPServer:
    """Servidor MCP para diccionario de términos."""
    
    def __init__(self, dictionary_path: str):
        """
        Inicializa el servidor MCP.
        
        Args:
            dictionary_path: Ruta al archivo JSON del diccionario
        """
        self.dictionary_path = Path(dictionary_path)
        self.dictionary: Dict[str, Any] = {}
        self.server = Server("dictionary-mcp-server")
        
        # Configurar logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s'
        )
        
        # Cargar diccionario
        self._load_dictionary()
        
        # Registrar handlers
        self._register_handlers()
    
    def _load_dictionary(self) -> None:
        """Carga el diccionario desde el archivo JSON."""
        try:
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                self.dictionary = json.load(f)
            self.logger.info(f"✓ Diccionario cargado: {len(self.dictionary.get('terminos', {}))} términos")
        except FileNotFoundError:
            self.logger.error(f"❌ Archivo no encontrado: {self.dictionary_path}")
            self.dictionary = {"terminos": {}, "categorias": {}}
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ Error parseando JSON: {e}")
            self.dictionary = {"terminos": {}, "categorias": {}}
    
    def _register_handlers(self) -> None:
        """Registra los handlers de recursos y herramientas."""
        
        # ==================== RECURSOS ====================
        
        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """Lista todos los recursos disponibles."""
            resources = [
                Resource(
                    uri="diccionario://terminos/lista",
                    name="Lista de Términos",
                    description="Lista completa de todos los términos en el diccionario",
                    mimeType="application/json"
                ),
                Resource(
                    uri="diccionario://categorias/lista",
                    name="Lista de Categorías",
                    description="Lista de todas las categorías disponibles",
                    mimeType="application/json"
                )
            ]
            
            # Agregar recurso por cada término
            terminos = self.dictionary.get('terminos', {})
            for palabra in list(terminos.keys())[:50]:  # Limitar a 50 para performance
                resources.append(Resource(
                    uri=f"diccionario://termino/{palabra}",
                    name=f"Término: {palabra}",
                    description=f"Definición y detalles del término '{palabra}'",
                    mimeType="application/json"
                ))
            
            return resources
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Lee el contenido de un recurso."""
            if uri == "diccionario://terminos/lista":
                terminos = self.dictionary.get('terminos', {})
                return json.dumps({
                    "total": len(terminos),
                    "terminos": list(terminos.keys())
                }, ensure_ascii=False, indent=2)
            
            elif uri == "diccionario://categorias/lista":
                categorias = self.dictionary.get('categorias', {})
                return json.dumps(categorias, ensure_ascii=False, indent=2)
            
            elif uri.startswith("diccionario://termino/"):
                palabra = uri.split("/")[-1]
                terminos = self.dictionary.get('terminos', {})
                
                if palabra in terminos:
                    return json.dumps(terminos[palabra], ensure_ascii=False, indent=2)
                else:
                    return json.dumps({"error": f"Término '{palabra}' no encontrado"})
            
            elif uri.startswith("diccionario://categoria/"):
                categoria = uri.split("/")[-1]
                categorias = self.dictionary.get('categorias', {})
                
                if categoria in categorias:
                    return json.dumps({
                        "categoria": categoria,
                        "terminos": categorias[categoria]
                    }, ensure_ascii=False, indent=2)
                else:
                    return json.dumps({"error": f"Categoría '{categoria}' no encontrada"})
            
            return json.dumps({"error": "Recurso no encontrado"})
        
        # ==================== HERRAMIENTAS ====================
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """Lista todas las herramientas disponibles."""
            return [
                Tool(
                    name="buscar_termino",
                    description="Busca un término exacto en el diccionario",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "palabra": {
                                "type": "string",
                                "description": "Palabra a buscar"
                            }
                        },
                        "required": ["palabra"]
                    }
                ),
                Tool(
                    name="buscar_similar",
                    description="Encuentra términos similares (útil para correcciones)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "palabra": {
                                "type": "string",
                                "description": "Palabra a buscar"
                            },
                            "max_resultados": {
                                "type": "number",
                                "description": "Número máximo de resultados (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["palabra"]
                    }
                ),
                Tool(
                    name="validar_uso",
                    description="Valida si un término es correcto en un contexto dado",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "palabra": {
                                "type": "string",
                                "description": "Palabra a validar"
                            },
                            "contexto": {
                                "type": "string",
                                "description": "Frase o contexto donde aparece la palabra"
                            }
                        },
                        "required": ["palabra", "contexto"]
                    }
                ),
                Tool(
                    name="obtener_categoria",
                    description="Obtiene la categoría de un término (marca, acrónimo, regionalismo)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "palabra": {
                                "type": "string",
                                "description": "Palabra a consultar"
                            }
                        },
                        "required": ["palabra"]
                    }
                ),
                Tool(
                    name="verificar_correccion",
                    description="Verifica si una corrección propuesta es válida según el diccionario",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "original": {
                                "type": "string",
                                "description": "Texto original"
                            },
                            "corregido": {
                                "type": "string",
                                "description": "Texto corregido propuesto"
                            }
                        },
                        "required": ["original", "corregido"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Ejecuta una herramienta."""
            
            if name == "buscar_termino":
                result = self._buscar_termino(arguments.get("palabra", ""))
                return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
            
            elif name == "buscar_similar":
                result = self._buscar_similar(
                    arguments.get("palabra", ""),
                    arguments.get("max_resultados", 5)
                )
                return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
            
            elif name == "validar_uso":
                result = self._validar_uso(
                    arguments.get("palabra", ""),
                    arguments.get("contexto", "")
                )
                return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
            
            elif name == "obtener_categoria":
                result = self._obtener_categoria(arguments.get("palabra", ""))
                return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
            
            elif name == "verificar_correccion":
                result = self._verificar_correccion(
                    arguments.get("original", ""),
                    arguments.get("corregido", "")
                )
                return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
            
            return [TextContent(type="text", text=json.dumps({"error": "Herramienta no encontrada"}))]
    
    # ==================== IMPLEMENTACIÓN DE HERRAMIENTAS ====================
    
    def _buscar_termino(self, palabra: str) -> Dict[str, Any]:
        """Busca un término exacto en el diccionario."""
        terminos = self.dictionary.get('terminos', {})
        
        # Búsqueda case-insensitive
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
    
    def _buscar_similar(self, palabra: str, max_resultados: int = 5) -> Dict[str, Any]:
        """Encuentra términos similares."""
        terminos = self.dictionary.get('terminos', {})
        todas_palabras = list(terminos.keys())
        
        # Obtener coincidencias cercanas
        similares = get_close_matches(
            palabra,
            todas_palabras,
            n=max_resultados,
            cutoff=0.6
        )
        
        resultados = []
        for sim in similares:
            resultados.append({
                "termino": sim,
                "datos": terminos[sim]
            })
        
        return {
            "palabra_buscada": palabra,
            "encontrados": len(resultados),
            "resultados": resultados
        }
    
    def _validar_uso(self, palabra: str, contexto: str) -> Dict[str, Any]:
        """Valida si un término es correcto en un contexto."""
        terminos = self.dictionary.get('terminos', {})
        
        # Buscar término
        termino_info = None
        for key, value in terminos.items():
            if key.lower() == palabra.lower():
                termino_info = value
                break
        
        if not termino_info:
            return {
                "valido": False,
                "razon": "Término no encontrado en diccionario",
                "sugerencias": self._buscar_similar(palabra, 3)["resultados"]
            }
        
        # Verificar si debe mantener original (regionalismos)
        if termino_info.get('mantener_original') or termino_info.get('no_corregir'):
            return {
                "valido": True,
                "es_regionalismo": True,
                "mantener_original": True,
                "mensaje": f"'{palabra}' es un {termino_info.get('tipo', 'término')} que debe mantenerse tal cual"
            }
        
        # Verificar contextos válidos
        contextos_validos = termino_info.get('contextos_validos', [])
        if contextos_validos:
            contexto_lower = contexto.lower()
            tiene_contexto_valido = any(c.lower() in contexto_lower for c in contextos_validos)
            
            if not tiene_contexto_valido:
                return {
                    "valido": True,
                    "advertencia": f"Contexto inusual para '{palabra}'",
                    "contextos_esperados": contextos_validos
                }
        
        return {
            "valido": True,
            "forma_correcta": termino_info.get('forma_correcta', palabra),
            "tipo": termino_info.get('tipo'),
            "categoria": termino_info.get('categoria')
        }
    
    def _obtener_categoria(self, palabra: str) -> Dict[str, Any]:
        """Obtiene la categoría de un término."""
        terminos = self.dictionary.get('terminos', {})
        
        for key, value in terminos.items():
            if key.lower() == palabra.lower():
                return {
                    "termino": key,
                    "tipo": value.get('tipo'),
                    "categoria": value.get('categoria'),
                    "definicion": value.get('definicion')
                }
        
        return {
            "encontrado": False,
            "mensaje": f"Término '{palabra}' no encontrado"
        }
    
    def _verificar_correccion(self, original: str, corregido: str) -> Dict[str, Any]:
        """Verifica si una corrección propuesta es válida."""
        terminos = self.dictionary.get('terminos', {})
        
        # Detectar cambios
        cambios_detectados = []
        palabras_originales = original.lower().split()
        palabras_corregidas = corregido.split()
        
        for idx, (orig, corr) in enumerate(zip(palabras_originales, palabras_corregidas)):
            if orig != corr.lower():
                # Verificar si es una corrección válida
                for termino, info in terminos.items():
                    variantes = info.get('variantes_incorrectas', [])
                    forma_correcta = info.get('forma_correcta', termino)
                    
                    if orig in [v.lower() for v in variantes] and corr == forma_correcta:
                        cambios_detectados.append({
                            "posicion": idx,
                            "original": orig,
                            "corregido": corr,
                            "valido": True,
                            "razon": f"Corrección de variante incorrecta a '{forma_correcta}'"
                        })
                        break
        
        return {
            "original": original,
            "corregido": corregido,
            "cambios_validados": len(cambios_detectados),
            "cambios": cambios_detectados,
            "confianza": 1.0 if cambios_detectados else 0.5
        }
    
    async def run(self):
        """Ejecuta el servidor MCP."""
        async with stdio_server() as (read_stream, write_stream):
            self.logger.info("🚀 Servidor MCP iniciado")
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())


def main():
    """Punto de entrada del servidor."""
    # Detectar ruta del diccionario
    if len(sys.argv) > 1:
        dictionary_path = sys.argv[1]
    else:
        # Ruta por defecto
        project_root = Path(__file__).parent.parent.parent
        dictionary_path = project_root / "data" / "diccionario_base.json"
    
    # Crear y ejecutar servidor
    server = DictionaryMCPServer(str(dictionary_path))
    
    import asyncio
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
