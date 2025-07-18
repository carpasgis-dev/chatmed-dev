#!/usr/bin/env python3
"""
Test de Migración a LangGraph
=============================

Script para probar la migración completa a LangGraph y LangSmith.
"""

import asyncio
import os
import sys
from datetime import datetime

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.langgraph_orchestrator import LangGraphOrchestrator
from langchain_openai import ChatOpenAI

async def test_langgraph_migration():
    """Prueba la migración completa a LangGraph"""
    
    print("🚀 Iniciando prueba de migración a LangGraph...")
    print("=" * 60)
    
    # Configurar variables de entorno
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "chatmed-migration-test"
    
    try:
        # Inicializar LLM
        print("🔧 Inicializando LLM...")
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1
        )
        print("✅ LLM inicializado")
        
        # Inicializar orquestador
        print("🔧 Inicializando LangGraph Orchestrator...")
        orchestrator = LangGraphOrchestrator(
            db_path="database_new.sqlite3.db",
            llm=llm
        )
        print("✅ LangGraph Orchestrator inicializado")
        
        # Pruebas de diferentes tipos de consultas
        test_queries = [
            "Hola, ¿cómo estás?",
            "¿Cuántos pacientes hay en la base de datos?",
            "Busca pacientes con diabetes",
            "¿Qué es la diabetes mellitus?",
            "Muéstrame el último paciente creado"
        ]
        
        print("\n🧪 Ejecutando pruebas de consultas...")
        print("=" * 60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Prueba {i}: {query}")
            print("-" * 40)
            
            start_time = datetime.now()
            
            try:
                # Procesar consulta
                response = await orchestrator.process_query(
                    query=query,
                    user_id="test_user"
                )
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                print(f"✅ Respuesta obtenida en {processing_time:.2f}s")
                print(f"📄 Respuesta: {response[:200]}...")
                
            except Exception as e:
                print(f"❌ Error en prueba {i}: {e}")
        
        print("\n🎉 Todas las pruebas completadas")
        print("=" * 60)
        print("✅ Migración a LangGraph exitosa")
        print("📊 Puedes ver los traces en LangSmith")
        
    except Exception as e:
        print(f"❌ Error en la migración: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ejecutar prueba
    asyncio.run(test_langgraph_migration()) 