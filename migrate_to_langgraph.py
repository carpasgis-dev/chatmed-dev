#!/usr/bin/env python3
"""
Script de migración a LangGraph
===============================

Migra el sistema ChatMed del orquestador actual al nuevo basado en LangGraph.
"""

import asyncio
import sys
import os
from pathlib import Path

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.langgraph_orchestrator import LangGraphOrchestrator

async def test_migration():
    """Prueba la migración a LangGraph"""
    print("🚀 Iniciando migración a LangGraph...")
    
    try:
        # Crear orquestador LangGraph
        orchestrator = LangGraphOrchestrator()
        
        print("✅ Orquestador LangGraph creado")
        
        # Pruebas de diferentes tipos de consultas
        test_queries = [
            "Hola, ¿cómo estás?",
            "¿Cuántos pacientes hay con diabetes?",
            "quiero crear un paciente llamado Juan Martínez",
            "efectos secundarios de la azitromicina"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Probando consulta: {query}")
            response = await orchestrator.process_query(query)
            print(f"📝 Respuesta: {response[:200]}...")
        
        print("\n✅ Migración completada exitosamente")
        
    except Exception as e:
        print(f"❌ Error en migración: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_migration()) 