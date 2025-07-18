#!/usr/bin/env python3
"""
ChatMed Simple - Sistema Principal con Orquestador Simple
=======================================================

Sistema principal usando el orquestador simple sin LangGraph.
"""

import asyncio
import os
import sys
from datetime import datetime

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.orchestrator_v2 import IntelligentOrchestratorV3
from langchain_openai import ChatOpenAI

async def main():
    """Función principal del sistema"""
    
    print("🏥 ChatMed Simple - Sistema de IA Médica")
    print("=" * 50)
    
    # Configurar variables de entorno
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "chatmed-simple"
    
    try:
        # Inicializar LLM
        print("🔧 Inicializando LLM...")
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1
        )
        print("✅ LLM inicializado")
        
        # Inicializar orquestador inteligente V3
        print("🔧 Inicializando Intelligent Orchestrator V3...")
        orchestrator = IntelligentOrchestratorV3(
            db_path="database_new.sqlite3.db",
            llm=llm
        )
        print("✅ Sistema inicializado correctamente")
        
        print("\n💬 ChatMed está listo. Escribe 'salir' para terminar.")
        print("=" * 50)
        
        # Bucle principal de chat
        while True:
            try:
                # Obtener consulta del usuario
                query = input("\n👤 Tú: ").strip()
                
                if query.lower() in ['salir', 'exit', 'quit']:
                    print("👋 ¡Hasta luego!")
                    break
                
                if not query:
                    continue
                
                # Procesar consulta
                print("🤖 ChatMed: Procesando...")
                start_time = datetime.now()
                
                response = await orchestrator.process_query_optimized(
                    query=query
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Extraer el mensaje de la respuesta estructurada
                if isinstance(response, dict):
                    message = response.get('message', str(response))
                else:
                    message = str(response)
                
                print(f"🤖 ChatMed ({processing_time:.2f}s): {message}")
                
            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("🤖 ChatMed: Lo siento, hubo un error. ¿Puedes intentar de nuevo?")
        
    except Exception as e:
        print(f"❌ Error inicializando el sistema: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ejecutar sistema principal
    asyncio.run(main()) 