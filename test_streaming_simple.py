#!/usr/bin/env python3
"""
Test básico de streaming para verificar que funciona correctamente
"""
import asyncio
import sys
import os
from pathlib import Path

# Configurar el entorno
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(project_root))

from core.orchestrator_v2 import FlexibleOrchestrator

async def test_streaming():
    """Test básico de streaming"""
    try:
        # Crear el orquestador
        orchestrator = FlexibleOrchestrator()
        
        # Callback para mostrar progreso
        def stream_callback(message: str):
            print(f"   📡 STREAMING: {message}")
        
        # Consulta de prueba
        query = "constantes vitales para ana garcia"
        
        print("🔄 Iniciando test de streaming...")
        print(f"📝 Query: {query}")
        print("=" * 50)
        
        result = await orchestrator.process_query_optimized(
            query, 
            stream_callback=stream_callback
        )
        
        print("=" * 50)
        print(f"🩺 Resultado final: {result}")
        
    except Exception as e:
        print(f"❌ Error en el test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_streaming())
