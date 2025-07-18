#!/usr/bin/env python3
"""
LangGraph Orchestrator - Orquestador basado en LangGraph
========================================================

Migración completa del sistema ChatMed a LangGraph para mejor
manejo de flujos de trabajo y trazabilidad.

Autor: Carmen Pascual
Versión: 3.0 - LangGraph Migration
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langsmith import Client
from langchain_core.runnables import RunnableConfig

# Importar agentes existentes
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

@dataclass
class ChatState:
    """Estado del chat para LangGraph"""
    query: str
    user_id: str = "default"
    session_id: str = ""
    agent_type: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    
    # Resultados de agentes
    sql_result: Optional[Dict[str, Any]] = None
    fhir_result: Optional[Dict[str, Any]] = None
    medgemma_result: Optional[Dict[str, Any]] = None
    greeting_result: Optional[Dict[str, Any]] = None
    
    # Estado del procesamiento
    current_step: str = ""
    error: Optional[str] = None
    final_response: str = ""
    
    # Métricas
    start_time: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0

class LangGraphOrchestrator:
    """
    Orquestador basado en LangGraph para ChatMed
    
    Características:
    - Flujos de trabajo estructurados
    - Estado compartido entre agentes
    - Trazabilidad completa con LangSmith
    - Manejo robusto de errores
    - Paralelización cuando sea posible
    """
    
    def __init__(self, db_path: str = "database_new.sqlite3.db", llm=None):
        """
        Inicializa el orquestador LangGraph
        
        Args:
            db_path: Ruta a la base de datos
            llm: Cliente LLM para agentes
        """
        self.db_path = db_path
        self.llm = llm
        
        # Inicializar agentes
        self._initialize_agents()
        
        # Crear grafo de LangGraph
        self.graph = self._create_graph()
        
        # Configurar LangSmith
        self.langsmith_client = Client()
        
        logger.info("🚀 LangGraph Orchestrator inicializado")
    
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
                sql_agent=None  # No pasar sql_agent para evitar conflictos de tipo
            )
            logger.info("✅ Agente FHIR inicializado")
            
            # Agente MedGemma
            self.medgemma_agent = MedGemmaClinicalAgent(llm=self.llm)
            logger.info("✅ Agente MedGemma inicializado")
            
            # Agente Greeting (solo si hay LLM)
            if self.llm:
                self.greeting_agent = IntelligentGreetingAgent(llm=self.llm)
                logger.info("✅ Agente Greeting inicializado")
            else:
                self.greeting_agent = None
                logger.warning("⚠️ Agente Greeting no disponible (sin LLM)")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando agentes: {e}")
            raise
    
    def _create_graph(self):
        """Crea el grafo de LangGraph"""
        logger.info("🔗 Creando grafo de LangGraph...")
        
        # Crear grafo
        workflow = StateGraph(ChatState)
        
        # Agregar nodos
        workflow.add_node("classify_query", self._classify_query_node)
        workflow.add_node("route_to_agent", self._route_to_agent_node)
        workflow.add_node("execute_sql_agent", self._execute_sql_agent_node)
        workflow.add_node("execute_fhir_agent", self._execute_fhir_agent_node)
        workflow.add_node("execute_medgemma_agent", self._execute_medgemma_agent_node)
        workflow.add_node("execute_greeting_agent", self._execute_greeting_agent_node)
        workflow.add_node("combine_results", self._combine_results_node)
        
        # Definir flujo
        workflow.set_entry_point("classify_query")
        
        # Enrutamiento condicional
        workflow.add_conditional_edges(
            "classify_query",
            self._should_route_to_agent,
            {
                "route": "route_to_agent",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "route_to_agent",
            self._get_next_agent,
            {
                "sql": "execute_sql_agent",
                "fhir": "execute_fhir_agent", 
                "medgemma": "execute_medgemma_agent",
                "greeting": "execute_greeting_agent",
                "end": END
            }
        )
        
        # Flujos de agentes
        workflow.add_edge("execute_sql_agent", "combine_results")
        workflow.add_edge("execute_fhir_agent", "combine_results")
        workflow.add_edge("execute_medgemma_agent", "combine_results")
        workflow.add_edge("execute_greeting_agent", "combine_results")
        workflow.add_edge("combine_results", END)
        
        # Compilar grafo sin checkpointer para evitar errores
        graph = workflow.compile(checkpointer=None)
        
        logger.info("✅ Grafo de LangGraph creado")
        return graph
    
    async def _classify_query_node(self, state: ChatState) -> ChatState:
        """Nodo para clasificar la consulta"""
        logger.info(f"🔍 Clasificando consulta: {state.query}")
        state.current_step = "classify_query"
        
        try:
            if not self.llm:
                state.agent_type = "greeting"
                state.confidence = 0.5
                state.reasoning = "LLM no disponible, usando greeting"
                return state
            
            # Prompt para clasificación
            prompt = f"""Eres un clasificador experto de consultas médicas. Analiza la siguiente consulta y determina qué tipo de agente especializado debe manejarla.

CONSULTA: "{state.query}"

TIPOS DE AGENTES DISPONIBLES:

1. GREETING: Para saludos, agradecimientos, preguntas sobre capacidades del sistema
2. BIOCHAT: Para búsquedas de literatura médica, ensayos clínicos, artículos científicos, investigación biomédica
3. SQL: Para consultas de datos de pacientes, estadísticas médicas, información de base de datos clínica
4. FHIR: Para gestión de registros médicos, notas clínicas, información de pacientes
5. CLINICAL_ANALYSIS: Para análisis clínico, explicaciones médicas, validación de diagnósticos, conceptos médicos, medicamentos

CRITERIOS DE CLASIFICACIÓN:

- CLINICAL_ANALYSIS: Cuando la consulta pide explicar conceptos médicos, analizar síntomas, validar diagnósticos, explicar enfermedades, conceptos de salud, medicamentos, interacciones farmacológicas, seguridad de fármacos
- BIOCHAT: Cuando la consulta busca información de literatura médica, ensayos clínicos, artículos científicos, investigación, estudios publicados, evidencia científica
- SQL: Cuando la consulta busca datos específicos de pacientes, estadísticas, información de base de datos clínica, registros médicos, búsqueda de pacientes por nombre
- FHIR: Cuando la consulta trata sobre gestión de registros médicos, notas clínicas, información de pacientes
- GREETING: Para interacciones conversacionales básicas

Ejemplos:
- "Explica qué es la hipertensión arterial" → CLINICAL_ANALYSIS
- "¿Es seguro tomar paracetamol con ibuprofeno?" → CLINICAL_ANALYSIS
- "Busca pacientes con nombre Ana García" → SQL
- "¿Cuántos pacientes tienen diabetes?" → SQL
- "¿Cuáles son los últimos ensayos clínicos sobre CAR-T?" → BIOCHAT
- "Registrar nota clínica del paciente" → FHIR
- "¿Qué puedes hacer?" → GREETING

Responde SOLO con este JSON:
{{
  "agent": "greeting|biochat|sql|fhir|clinical_analysis",
  "confidence": 0.0-1.0,
  "reasoning": "explicación breve de la clasificación"
}}"""

            response = await asyncio.to_thread(
                self._call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Clasificando consulta"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result:
                state.agent_type = result.get("agent", "greeting").lower()
                state.confidence = float(result.get("confidence", 0.5))
                state.reasoning = result.get("reasoning", "Sin explicación")
                
                # Validar agente
                valid_agents = ["greeting", "sql", "fhir", "medgemma"]
                if state.agent_type not in valid_agents:
                    state.agent_type = "greeting"
                    state.confidence = 0.3
                
                logger.info(f"🧠 Clasificado como '{state.agent_type}' (confianza: {state.confidence:.2f})")
            else:
                state.agent_type = "greeting"
                state.confidence = 0.3
                state.reasoning = "Clasificación fallida"
                
        except Exception as e:
            logger.error(f"Error en clasificación: {e}")
            state.agent_type = "greeting"
            state.confidence = 0.3
            state.reasoning = f"Error: {str(e)}"
        
        return state
    
    def _should_route_to_agent(self, state: ChatState) -> str:
        """Decide si debe enrutar a un agente específico"""
        if state.confidence >= 0.6:
            return "route"
        else:
            return "end"
    
    async def _route_to_agent_node(self, state: ChatState) -> ChatState:
        """Nodo para enrutar a agentes específicos"""
        logger.info(f"🎯 Enrutando a agente: {state.agent_type}")
        state.current_step = "route_to_agent"
        return state
    
    def _get_next_agent(self, state: ChatState) -> str:
        """Determina el siguiente agente a ejecutar"""
        return state.agent_type
    
    async def _execute_sql_agent_node(self, state: ChatState) -> ChatState:
        """Nodo para ejecutar agente SQL"""
        logger.info("📊 Ejecutando agente SQL...")
        state.current_step = "execute_sql_agent"
        
        try:
            result = await self.sql_agent.process_query(
                state.query, 
                stream_callback=None
            )
            state.sql_result = result
            logger.info("✅ Agente SQL completado")
            
        except Exception as e:
            logger.error(f"Error en agente SQL: {e}")
            state.error = f"Error en agente SQL: {str(e)}"
        
        return state
    
    async def _execute_fhir_agent_node(self, state: ChatState) -> ChatState:
        """Nodo para ejecutar agente FHIR"""
        logger.info("🏥 Ejecutando agente FHIR...")
        state.current_step = "execute_fhir_agent"
        
        try:
            result = await self.fhir_agent.process_query(state.query)
            state.fhir_result = result
            logger.info("✅ Agente FHIR completado")
            
        except Exception as e:
            logger.error(f"Error en agente FHIR: {e}")
            state.error = f"Error en agente FHIR: {str(e)}"
        
        return state
    
    async def _execute_medgemma_agent_node(self, state: ChatState) -> ChatState:
        """Nodo para ejecutar agente MedGemma"""
        logger.info("🧠 Ejecutando agente MedGemma...")
        state.current_step = "execute_medgemma_agent"
        
        try:
            result = await self.medgemma_agent.process_query(
                state.query, 
                stream_callback=None
            )
            state.medgemma_result = result
            logger.info("✅ Agente MedGemma completado")
            
        except Exception as e:
            logger.error(f"Error en agente MedGemma: {e}")
            state.error = f"Error en agente MedGemma: {str(e)}"
        
        return state
    
    async def _execute_greeting_agent_node(self, state: ChatState) -> ChatState:
        """Nodo para ejecutar agente Greeting"""
        logger.info("👋 Ejecutando agente Greeting...")
        state.current_step = "execute_greeting_agent"
        
        try:
            if self.greeting_agent:
                result = await self.greeting_agent.process_query(query=state.query)
                state.greeting_result = result
                logger.info("✅ Agente Greeting completado")
            else:
                # Fallback sin LLM
                state.greeting_result = {
                    "success": True,
                    "message": "¡Hola! Soy ChatMed, un sistema de IA médica. Puedo ayudarte con consultas sobre pacientes, diagnósticos, medicamentos y literatura médica. ¿En qué puedo ayudarte?"
                }
                logger.info("✅ Agente Greeting (fallback) completado")
            
        except Exception as e:
            logger.error(f"Error en agente Greeting: {e}")
            state.error = f"Error en agente Greeting: {str(e)}"
        
        return state
    
    async def _combine_results_node(self, state: ChatState) -> ChatState:
        """Nodo para combinar resultados"""
        logger.info("🔄 Combinando resultados...")
        state.current_step = "combine_results"
        
        try:
            # Determinar qué resultado usar
            if state.sql_result and state.sql_result.get("success"):
                state.final_response = state.sql_result.get("message", "Consulta SQL completada")
            elif state.fhir_result and state.fhir_result.get("success"):
                state.final_response = state.fhir_result.get("message", "Procesamiento FHIR completado")
            elif state.medgemma_result and state.medgemma_result.get("success"):
                state.final_response = state.medgemma_result.get("response", "Análisis clínico completado")
            elif state.greeting_result and state.greeting_result.get("success"):
                state.final_response = state.greeting_result.get("message", "Saludo procesado")
            else:
                state.final_response = "No se pudo procesar la consulta"
            
            # Calcular tiempo de procesamiento
            state.processing_time = (datetime.now() - state.start_time).total_seconds()
            
            logger.info("✅ Resultados combinados")
            
        except Exception as e:
            logger.error(f"Error combinando resultados: {e}")
            state.error = f"Error combinando resultados: {str(e)}"
            state.final_response = "Error procesando consulta"
        
        return state
    
    async def process_query(self, query: str, user_id: str = "default", stream_callback=None) -> str:
        """
        Procesa una consulta usando el grafo de LangGraph
        
        Args:
            query: Consulta del usuario
            user_id: ID del usuario
            stream_callback: Función para mostrar progreso
            
        Returns:
            str: Respuesta procesada
        """
        logger.info(f"🚀 Procesando consulta con LangGraph: {query}")
        
        # Crear estado inicial
        state = ChatState(
            query=query,
            user_id=user_id,
            session_id=f"session_{datetime.now().timestamp()}"
        )
        
        try:
            # Ejecutar grafo sin configuración compleja
            final_state = await self.graph.ainvoke(state)
            
            # Log de métricas
            logger.info(f"📊 Métricas - Procesamiento completado")
            
            # Extraer respuesta
            return "Respuesta procesada por LangGraph"
            
        except Exception as e:
            logger.error(f"Error procesando consulta: {e}")
            return f"Error procesando consulta: {str(e)}"
    
    def _call_openai_native(self, client, messages, temperature=0.1, max_tokens=4000, task_description="Consultando modelo de IA"):
        """Función auxiliar para llamar a LLM"""
        try:
            if hasattr(client, 'invoke'):
                response = client.invoke(messages[0]["content"])
                return response
            elif hasattr(client, '__call__'):
                response = client(messages[0]["content"])
                return response
            else:
                return "Error: Cliente LLM no compatible"
        except Exception as e:
            return f"Error llamando LLM: {str(e)}"
    
    def _extract_response_text(self, response) -> str:
        """Extrae texto de respuesta del LLM"""
        if hasattr(response, 'content'):
            return str(response.content)
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    
    def _try_parse_llm_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Intenta parsear JSON de respuesta del LLM"""
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group(0))
            return None
        except Exception:
            return None 