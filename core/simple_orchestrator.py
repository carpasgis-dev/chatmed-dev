#!/usr/bin/env python3
"""
Simple Orchestrator - Orquestador Simple con LangSmith
=====================================================

Orquestador simplificado que evita los problemas del checkpointer pero
mantiene trazabilidad completa con LangSmith.
"""

import asyncio
import logging
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langsmith import Client
from langchain_core.runnables import RunnableConfig

# Importar agentes
from agents.sql_agent_flexible_enhanced import SQLAgentIntelligentEnhanced
from agents.fhir_agent_complete import FHIRMedicalAgent
from agents.medgemma_clinical_agent import MedGemmaClinicalAgent
from agents.greeting_agent import IntelligentGreetingAgent

logger = logging.getLogger(__name__)

# Configurar LangSmith
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "pr-unnatural-propane-82"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "pr-unnatural-propane-82"

class SimpleOrchestrator:
    """
    Orquestador simple con LangSmith para trazabilidad completa
    """
    
    def __init__(self, db_path: str = "database_new.sqlite3.db", llm=None):
        self.db_path = db_path
        self.llm = llm
        
        # Configurar LangSmith
        self.langsmith_client = Client()
        
        # Inicializar agentes
        self._initialize_agents()
        
        logger.info("🚀 Simple Orchestrator con LangSmith inicializado")
    
    def _initialize_agents(self):
        """Inicializa todos los agentes"""
        logger.info("🔧 Inicializando agentes...")
        
        try:
            # Agente SQL
            self.sql_agent = SQLAgentIntelligentEnhanced(
                db_path=self.db_path,
                llm=self.llm
            )
            logger.info("✅ Agente SQL inicializado")
            
            # Agente FHIR
            self.fhir_agent = FHIRMedicalAgent(
                db_path=self.db_path,
                llm=self.llm,
                sql_agent=self.sql_agent  # Pasar el SQL agent creado
            )
            logger.info("✅ Agente FHIR inicializado")
            
            # Agente MedGemma
            self.medgemma_agent = MedGemmaClinicalAgent(llm=self.llm)
            logger.info("✅ Agente MedGemma inicializado")
            
            # Agente Greeting
            if self.llm:
                self.greeting_agent = IntelligentGreetingAgent(llm=self.llm)
                logger.info("✅ Agente Greeting inicializado")
            else:
                self.greeting_agent = None
                logger.warning("⚠️ Agente Greeting no disponible")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando agentes: {e}")
            raise
    
    async def _classify_query_with_langsmith(self, query: str) -> Dict[str, Any]:
        """Clasifica la consulta usando LLM con LangSmith"""
        if not self.llm:
            return {"agent": "greeting", "confidence": 0.5}
        
        try:
            prompt = f"""Clasifica esta consulta médica:

CONSULTA: "{query}"

TIPOS:
- GREETING: Saludos, preguntas sobre capacidades del sistema
- SQL: Consultas de datos existentes, búsquedas de pacientes, estadísticas, "último paciente", "mostrar pacientes"
- FHIR: Registros médicos nuevos, notas clínicas, historias clínicas, datos de pacientes para insertar, información médica estructurada
- MEDGEMMA: Análisis clínico, conceptos médicos, preguntas sobre enfermedades, tratamientos, diagnósticos

INDICADORES CLAVE:
- Si contiene datos estructurados de paciente (nombre, edad, síntomas, diagnósticos) → FHIR
- Si pregunta por datos existentes ("mostrar", "buscar", "último") → SQL
- Si pregunta sobre conceptos médicos generales → MEDGEMMA
- Si es saludo o pregunta sobre el sistema → GREETING

Responde SOLO con JSON:
{{"agent": "greeting|sql|fhir|medgemma", "confidence": 0.0-1.0}}"""

            # Configurar LangSmith para esta llamada
            config = RunnableConfig(
                tags=["query_classification"],
                metadata={
                    "query": query,
                    "step": "classification"
                }
            )
            
            response = await self.llm.ainvoke(prompt, config=config)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extraer JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group(0))
                return result
            else:
                return {"agent": "greeting", "confidence": 0.3}
                
        except Exception as e:
            logger.error(f"Error en clasificación: {e}")
            return {"agent": "greeting", "confidence": 0.3}
    
    def _log_result_to_langsmith(self, agent_type: str, query: str, result: Dict[str, Any]):
        """Registra resultado en LangSmith con SQL completa - UNA SOLA ENTRADA"""
        try:
            # Extraer SQL si está disponible
            sql_query = result.get("sql", "")
            data_count = result.get("count", 0)
            
            # Crear UNA SOLA entrada para LangSmith con toda la información
            run_data = {
                "name": f"chatmed_query_{agent_type}",
                "run_type": "chain",
                "inputs": {
                    "query": query,
                    "agent_type": agent_type
                },
                "outputs": {
                    "result": result,
                    "sql_query": sql_query,
                    "data_count": data_count,
                    "success": result.get("success", False),
                    "message": result.get("message", "")
                },
                "tags": [f"agent_{agent_type}", "chatmed", "query"],
                "metadata": {
                    "agent_type": agent_type,
                    "success": result.get("success", False),
                    "sql_generated": bool(sql_query),
                    "data_count": data_count,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Enviar a LangSmith
            self.langsmith_client.create_run(**run_data)
            
        except Exception as e:
            logger.warning(f"No se pudo registrar resultado en LangSmith: {e}")
    
    def _log_error_to_langsmith(self, agent_type: str, query: str, error: str):
        """Registra error en LangSmith - UNA SOLA ENTRADA"""
        try:
            # Crear UNA SOLA entrada de error para LangSmith
            run_data = {
                "name": f"chatmed_error_{agent_type}",
                "run_type": "chain",
                "inputs": {
                    "query": query,
                    "agent_type": agent_type
                },
                "outputs": {
                    "error": error,
                    "error_type": "agent_execution_error"
                },
                "tags": [f"agent_{agent_type}", "error", "chatmed"],
                "metadata": {
                    "agent_type": agent_type,
                    "error_type": "agent_execution_error",
                    "has_sql_context": "sql" in error.lower(),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Enviar a LangSmith
            self.langsmith_client.create_run(**run_data)
            
        except Exception as e:
            logger.warning(f"No se pudo registrar error en LangSmith: {e}")
    

    
    async def _execute_agent_with_langsmith(self, agent_type: str, query: str) -> Dict[str, Any]:
        """Ejecuta agente con LangSmith para trazabilidad completa - UNA SOLA ENTRADA"""
        try:
            # Configurar LangSmith para la ejecución del agente
            config = RunnableConfig(
                tags=[f"agent_execution_{agent_type}"],
                metadata={
                    "query": query,
                    "agent_type": agent_type,
                    "step": "agent_execution"
                }
            )
            
            if agent_type == "sql":
                logger.info("📊 Ejecutando agente SQL...")
                result = await self.sql_agent.process_query(query)
                
                # Log de SQL generada para debugging
                if result.get("success") and "sql" in result:
                    sql_generated = result.get("sql", "")
                    if sql_generated:
                        logger.info(f"🔍 SQL generada: {sql_generated}")
                
            elif agent_type == "fhir":
                logger.info("🏥 Ejecutando agente FHIR...")
                result = await self.fhir_agent.process_query(query)
                
            elif agent_type == "medgemma":
                logger.info("🧠 Ejecutando agente MedGemma...")
                result = await self.medgemma_agent.process_query(query)
                
            else:  # greeting
                logger.info("👋 Ejecutando agente Greeting...")
                if self.greeting_agent:
                    result = await self.greeting_agent.process_query(query=query)
                else:
                    result = {
                        "success": True,
                        "message": "¡Hola! Soy ChatMed, un sistema de IA médica. ¿En qué puedo ayudarte?"
                    }
            
            # Registrar UNA SOLA entrada en LangSmith
            self._log_result_to_langsmith(agent_type, query, result)
            
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "message": f"Error en agente {agent_type}: {str(e)}"
            }
            
            # Registrar UNA SOLA entrada de error en LangSmith
            self._log_error_to_langsmith(agent_type, query, str(e))
            
            logger.error(f"❌ Error en agente {agent_type}: {e}")
            return error_result
    
    async def process_query(self, query: str, user_id: str = "default") -> str:
        """
        Procesa consulta con LangSmith para trazabilidad completa
        """
        logger.info(f"🚀 Procesando consulta: {query}")
        start_time = datetime.now()
        
        try:
            # Clasificar consulta con LangSmith
            classification = await self._classify_query_with_langsmith(query)
            agent_type = classification.get("agent", "greeting")
            confidence = classification.get("confidence", 0.3)
            
            logger.info(f"🧠 Clasificado como '{agent_type}' (confianza: {confidence:.2f})")
            
            # Ejecutar agente correspondiente con LangSmith
            if confidence > 0.6:
                result = await self._execute_agent_with_langsmith(agent_type, query)
            else:
                # Fallback a greeting si confianza baja
                result = await self._execute_agent_with_langsmith("greeting", query)
            
            # Extraer respuesta
            if result.get("success"):
                response = result.get("message", result.get("response", "Procesamiento completado"))
            else:
                response = result.get("message", "Error en el procesamiento")
            
            # Calcular tiempo
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"📊 Procesamiento completado en {processing_time:.2f}s")
            
            # Registrar resultado final en LangSmith
            self._log_final_result_to_langsmith(query, response, processing_time, agent_type)
            
            return response
            
        except Exception as e:
            error_msg = f"Error procesando consulta: {str(e)}"
            logger.error(error_msg)
            
            # Registrar error final en LangSmith
            self._log_error_to_langsmith("orchestrator", query, str(e))
            
            return error_msg
    
    def _log_final_result_to_langsmith(self, query: str, response: str, processing_time: float, agent_type: str):
        """Registra resultado final en LangSmith - UNA SOLA ENTRADA"""
        try:
            run_data = {
                "name": "chatmed_final_result",
                "run_type": "chain",
                "inputs": {"query": query},
                "outputs": {"response": response},
                "tags": ["orchestrator", "final_result", "chatmed"],
                "metadata": {
                    "processing_time": processing_time,
                    "agent_type": agent_type,
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            self.langsmith_client.create_run(**run_data)
            
        except Exception as e:
            logger.warning(f"No se pudo registrar resultado final en LangSmith: {e}")

# Función de utilidad para usar directamente
async def process_query_simple(query: str, db_path: str = "database_new.sqlite3.db", llm=None) -> str:
    """Procesa una consulta usando el orquestador simple con LangSmith"""
    orchestrator = SimpleOrchestrator(db_path=db_path, llm=llm)
    return await orchestrator.process_query(query) 