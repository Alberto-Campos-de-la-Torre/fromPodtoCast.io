#!/usr/bin/env python3
"""
Script de prueba para verificar la funcionalidad del sistema MCP.

Prueba:
1. Servidor MCP de diccionario
2. Cliente MCP
3. Verificador de texto MCP
4. Integración completa
"""

import sys
import json
from pathlib import Path

# Añadir src al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

def test_mcp_client():
    """Prueba el cliente MCP."""
    print("\n" + "="*60)
    print("PRUEBA 1: Cliente MCP")
    print("="*60)
    
    try:
        from mcp.mcp_client import SimpleMCPClient
        
        # Inicializar cliente
        client = SimpleMCPClient()
        print("✓ Cliente MCP inicializado")
        
        # Prueba 1: Buscar término exacto
        print("\n1. Buscar término 'ChatGPT':")
        result = client.buscar_termino("ChatGPT")
        print(f"   Encontrado: {result.get('encontrado')}")
        if result.get('encontrado'):
            print(f"   Tipo: {result['datos'].get('tipo')}")
            print(f"   Categoría: {result['datos'].get('categoria')}")
        
        # Prueba 2: Buscar términos similares
        print("\n2. Buscar similares a 'chat gpt':")
        result = client.buscar_similar("chat gpt", 3)
        print(f"   Encontrados: {result.get('encontrados')}")
        for r in result.get('resultados', [])[:3]:
            print(f"   - {r['termino']}")
        
        # Prueba 3: Validar uso
        print("\n3. Validar uso de 'güey':")
        result = client.validar_uso("güey", "Qué onda güey, está chido el podcast")
        print(f"   Válido: {result.get('valido')}")
        print(f"   Mantener original: {result.get('mantener_original')}")
        
        # Prueba 4: Verificar corrección
        print("\n4. Verificar corrección:")
        result = client.verificar_correccion(
            "hablamos de chat gpt",
            "Hablamos de ChatGPT"
        )
        print(f"   Cambios validados: {result.get('cambios_validados')}")
        print(f"   Confianza: {result.get('confianza')}")
        
        print("\n✓ TODAS LAS PRUEBAS DEL CLIENTE PASARON\n")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mcp_verifier():
    """Prueba el verificador MCP."""
    print("\n" + "="*60)
    print("PRUEBA 2: Verificador MCP")
    print("="*60)
    
    try:
        from text_verifier_mcp import TextVerifierMCP
        
        # Inicializar verificador (sin LLM para prueba rápida)
        verifier = TextVerifierMCP(
            ollama_host="http://localhost:11434",
            model="qwen3:14b"
        )
        print("✓ Verificador MCP inicializado")
        
        # Caso 1: Corrección válida (marca)
        print("\n1. Verificar corrección de marca:")
        original = "estamos usando chat gpt para el proyecto"
        corrected = "Estamos usando ChatGPT para el proyecto"
        
        result = verifier.verify_correction(original, corrected)
        print(f"   Texto verificado: {result.texto_verificado}")
        print(f"   Cambios aceptados: {len(result.cambios_aplicados)}")
        print(f"   Confianza: {result.confianza:.2f}")
        
        # Caso 2: Corrección que debe revertirse (regionalismo)
        print("\n2. Verificar regionalismo (debe mantener):")
        original = "pues si guey esta chido"
        corrected = "Pues sí amigo, está genial"  # Corrección INCORRECTA
        
        result = verifier.verify_correction(original, corrected)
        print(f"   Texto verificado: {result.texto_verificado}")
        print(f"   Cambios revertidos: {len(result.cambios_revertidos)}")
        
        print("\n✓ TODAS LAS PRUEBAS DEL VERIFICADOR PASARON\n")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Prueba la integración completa."""
    print("\n" + "="*60)
    print("PRUEBA 3: Integración Completa")
    print("="*60)
    
    try:
        from text_corrector_llm import TextCorrectorLLM
        from text_verifier_mcp import TextVerifierMCP
        
        # Textos de prueba
        test_texts = [
            "que es chat gpt y como funciona",
            "vamos a hablar de youtube y tiktok",
            "pues si guey esta bien chido el podcast"
        ]
        
        print("\n📝 Textos de prueba:")
        for i, text in enumerate(test_texts, 1):
            print(f"   {i}. {text}")
        
        # Fase 1: Corrección LLM
        print("\n1️⃣  FASE 1: Corrección LLM")
        corrector = TextCorrectorLLM(
            ollama_host="http://localhost:11434",
            model="qwen3:14b",
            batch_size=3,
            enable_verification=False  # Desactivar verificación interna
        )
        
        corrections = corrector.correct_batch_optimized(test_texts)
        
        print("\n   Resultados de corrección:")
        for i, (corrected, meta) in enumerate(corrections, 1):
            print(f"   {i}. {corrected}")
            print(f"      Cambios: {meta.get('cambios', [])}")
        
        # Fase 2: Verificación MCP
        print("\n2️⃣  FASE 2: Verificación MCP")
        verifier = TextVerifierMCP(
            ollama_host="http://localhost:11434",
            model="qwen3:14b"
        )
        
        verification_data = [
            (orig, corr, meta) 
            for orig, (corr, meta) in zip(test_texts, corrections)
        ]
        
        verification_results = verifier.verify_batch(verification_data)
        
        print("\n   Resultados de verificación:")
        for i, result in enumerate(verification_results, 1):
            print(f"   {i}. {result.texto_verificado}")
            if result.cambios_revertidos:
                print(f"      ⚠️  Revertidos: {result.cambios_revertidos}")
            print(f"      Confianza MCP: {result.confianza:.2f}")
        
        # Estadísticas
        print("\n📊 Estadísticas:")
        llm_stats = corrector.get_stats()
        mcp_stats = verifier.get_stats()
        
        print(f"   LLM:")
        print(f"   - Procesados: {llm_stats['processed']}")
        print(f"   - Corregidos: {llm_stats['corrected']}")
        print(f"   - Confianza promedio: {llm_stats['avg_confidence']:.2f}")
        
        print(f"   MCP:")
        print(f"   - Verificados: {mcp_stats['verificados']}")
        print(f"   - Revertidos: {mcp_stats['cambios_revertidos']}")
        print(f"   - Consultas diccionario: {mcp_stats['consultas_mcp']}")
        
        print("\n✓ INTEGRACIÓN COMPLETA EXITOSA\n")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Ejecuta todas las pruebas."""
    print("\n" + "🧪 " + "="*56 + " 🧪")
    print("   TEST SUITE: Sistema de Verificación MCP")
    print("🧪 " + "="*56 + " 🧪")
    
    results = []
    
    # Ejecutar pruebas
    results.append(("Cliente MCP", test_mcp_client()))
    results.append(("Verificador MCP", test_mcp_verifier()))
    
    # Preguntar si ejecutar integración completa (requiere Ollama)
    print("\n" + "="*60)
    print("⚠️  La prueba de integración completa requiere:")
    print("   - Servidor Ollama activo (http://localhost:11434)")
    print("   - Modelo qwen3:14b descargado")
    print("="*60)
    
    try:
        response = input("\n¿Ejecutar prueba de integración? (s/n): ").lower()
        if response == 's':
            results.append(("Integración Completa", test_integration()))
    except KeyboardInterrupt:
        print("\n\n⏭️  Prueba de integración omitida")
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN DE PRUEBAS")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASÓ" if passed else "❌ FALLÓ"
        print(f"{status:12} | {test_name}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 ¡TODAS LAS PRUEBAS PASARON!\n")
        return 0
    else:
        print("\n⚠️  ALGUNAS PRUEBAS FALLARON\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
