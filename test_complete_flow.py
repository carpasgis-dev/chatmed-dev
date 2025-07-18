#!/usr/bin/env python3
"""
Test completo del flujo SQL Agent para diagnosticar problemas
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatmed_v2_flexible.core.orchestrator_v2 import IntelligentOrchestratorV3

async def test_complete_flow():
    """Test completo del flujo"""
    
    print("🧪 Iniciando test completo del flujo...")
    
    # Crear instancia del orquestador
    orchestrator = IntelligentOrchestratorV3()
    
    # Verificar agentes disponibles
    print(f"📊 Agentes disponibles: {list(orchestrator.agents.keys())}")
    
    # Test query de hemoglobina
    test_query = "¿Cuál es la media de hemoglobina glicosilada (HbA₁c) de pacientes con diabetes tipo 2 en el último año?"
    print(f"\n🔍 Procesando consulta: {test_query}")
    
    # Función de callback para ver los pasos
    def stream_callback(message):
        print(f"   📡 STREAM: {message}")
    
    try:
        # Procesar la consulta completa
        result = await orchestrator.process_query_optimized(test_query, stream_callback=stream_callback)
        
        print(f"\n✅ RESULTADO FINAL:")
        print(f"   Éxito: {result.get('success', False)}")
        print(f"   Mensaje: {result.get('message', 'Sin mensaje')}")
        if result.get('data'):
            print(f"   Datos: {len(result['data'])} filas")
        if result.get('sql_query'):
            print(f"   SQL: {result['sql_query'][:100]}...")
        if result.get('error'):
            print(f"   Error: {result['error']}")
            
    except Exception as e:
        print(f"❌ Error en el test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Test completado!")

if __name__ == "__main__":
    asyncio.run(test_complete_flow())
