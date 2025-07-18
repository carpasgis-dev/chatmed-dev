"""
GreetingAgent - Agente de Saludos Inteligente y Personal v2.0
============================================================

Agente especializado en manejar saludos, información del sistema y consultas generales
con capacidades avanzadas de análisis de contexto, personalización y memoria de usuarios.

🧠 FILOSOFÍA: 100% dinámico sin hardcodeo, usando LLM para comprensión contextual
⚡ CARACTERÍSTICAS: Auto-adaptativo, conversacional, transparente sobre capacidades

Desarrollado por Carmen Pascual para ChatMed 2.0 - Sistema de Agentes Médicos con IA
"""

import logging
import os
import re
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union
from langchain.llms.base import BaseLLM
from langchain.schema import HumanMessage, AIMessage, BaseMessage, SystemMessage
from collections import defaultdict
import pickle
from pathlib import Path

# Configuración de logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "ERROR"))  # Cambiar de WARNING a ERROR
logger = logging.getLogger("GreetingAgent")

class IntelligentGreetingAgent:
    """
    🧠 Agente conversacional avanzado que usa LLM para manejar interacciones no técnicas
    siguiendo la filosofía de ChatMed: 100% dinámico, sin hardcodeo.
    """
    
    def __init__(self, llm: Optional[BaseLLM] = None):
        """
        Inicializa el agente conversacional inteligente.
        
        Args:
            llm: Una instancia de un modelo de lenguaje compatible con LangChain.
        """
        if not llm:
            raise ValueError("Se requiere una instancia de un modelo de lenguaje (LLM).")
        self.llm = llm
        self.conversation_history = []
        
        # 🧠 SISTEMA DE INFORMACIÓN DINÁMICO
        self.system_info = {
            "name": "ChatMed 2.0",
            "version": "2.0 Flexible",
            "description": "Un sistema de IA médica multi-agente diseñado para ayudar a profesionales de la salud",
            "creator": "Carmen Pascual (@carpasgis-dev) en Laberit",
            "philosophy": "100% dinámico sin hardcodeo - Todo se decide usando IA en tiempo real",
            "technologies": {
                "frameworks": [
                    "LangChain - Framework principal para orquestación de LLMs",
                    "OpenAI GPT-4 - Modelo de lenguaje base para razonamiento",
                    "Bio.Entrez - API para acceso a bases de datos biomédicas (PubMed, GenBank)",
                    "FHIR - Estándar para interoperabilidad en salud",
                    "SQLite - Base de datos local para almacenamiento clínico"
                ],
                "databases": [
                    "PubMed - Literatura médica y estudios científicos",
                    "GenBank - Datos genómicos y secuencias",
                    "ClinicalTrials.gov - Ensayos clínicos activos y completados",
                    "Europe PMC - Literatura biomédica europea",
                    "AEMPS - Información de medicamentos autorizados en España",
                    "Semantic Scholar - Literatura académica con análisis de impacto"
                ]
            },
            "agents": {
                "BioChatAgent": {
                    "purpose": "Búsqueda inteligente de literatura científica",
                    "capabilities": [
                        "Buscar estudios en PubMed con queries optimizadas",
                        "Acceder a ensayos clínicos en ClinicalTrials.gov",
                        "Consultar secuencias genéticas en GenBank",
                        "Analizar literatura en Semantic Scholar y Europe PMC",
                        "Verificar medicamentos en AEMPS",
                        "Síntesis inteligente de múltiples fuentes"
                    ]
                },
                "SQLAgentRobust": {
                    "purpose": "Análisis inteligente de datos médicos",
                    "capabilities": [
                        "Consultas SQL dinámicas sin hardcodeo",
                        "Auto-exploración del esquema de base de datos",
                        "Mapeo inteligente de conceptos médicos a tablas",
                        "Sistema de auto-corrección iterativa",
                        "Aprendizaje adaptativo de patrones exitosos"
                    ]
                },
                "FHIRMedicalAgent": {
                    "purpose": "Procesamiento de información clínica",
                    "capabilities": [
                        "Procesamiento de notas clínicas con IA",
                        "Conversión automática SQL↔FHIR",
                        "Validación FHIR automática",
                        "Gestión inteligente de recursos relacionados",
                        "Mapeo dinámico de campos sin hardcodeo"
                    ]
                },
                "GreetingAgent": {
                    "purpose": "Interacción conversacional y ayuda",
                    "capabilities": [
                        "Conversaciones naturales y contextuales",
                        "Explicación de capacidades del sistema",
                        "Ayuda y orientación para usuarios",
                        "Análisis inteligente de intenciones del usuario"
                    ]
                }
            }
        }
        
        # 🧠 SISTEMA DE ANÁLISIS CONTEXTUAL
        self.context_analyzer = {
            "last_interaction_type": None,
            "user_expertise_level": "unknown",  # unknown, beginner, intermediate, expert
            "conversation_flow": [],
            "detected_needs": []
        }
        
        logger.info("✅ IntelligentGreetingAgent v2.0 (100% dinámico) inicializado.")

    async def process_query(self, *, query: str, **kwargs) -> Dict[str, Any]:
        """
        🧠 Procesa la consulta del usuario usando análisis contextual inteligente
        siguiendo la filosofía de ChatMed: 100% dinámico sin patrones hardcodeados.
        """
        try:
            # 🧠 ANÁLISIS CONTEXTUAL PREVIO
            context_analysis = await self._analyze_user_context(query)
            
            # Actualizar historial conversacional
            self.conversation_history.append(f"Usuario: {query}")
            
            # 🧠 GENERAR RESPUESTA INTELIGENTE
            response_text = await self._generate_intelligent_response(query, context_analysis)
            
            # Actualizar contexto conversacional
            self.conversation_history.append(f"Asistente: {response_text}")
            self._update_conversation_context(query, response_text, context_analysis)
            
            # Mantener historial limitado
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return {"success": True, "message": response_text}
            
        except Exception as e:
            logger.error(f"Error durante el procesamiento en GreetingAgent: {e}")
            return {
                "success": False,
                "message": "Lo siento, estoy teniendo problemas para procesar tu solicitud en este momento. ¿Podrías reformular tu pregunta?"
            }

    async def _analyze_user_context(self, query: str) -> Dict[str, Any]:
        """
        🧠 Análisis contextual inteligente de la consulta del usuario
        sin patrones hardcodeados, usando comprensión semántica.
        """
        if not self.llm:
            return {"type": "unknown", "intent": "general", "complexity": "low"}
        
        context_prompt = f"""Eres un analista experto en interacciones conversacionales para sistemas de IA médica. Analiza esta consulta para entender el contexto y la intención del usuario.

**CONSULTA DEL USUARIO:**
"{query}"

**HISTORIAL CONVERSACIONAL RECIENTE:**
{chr(10).join(self.conversation_history[-4:]) if self.conversation_history else "No hay historial previo"}

**TU ANÁLISIS DEBE INCLUIR:**

1. **Tipo de Interacción:**
   - greeting: Saludo inicial o casual
   - help_request: Solicitud de ayuda o información sobre capacidades
   - system_inquiry: Pregunta sobre el funcionamiento del sistema
   - follow_up: Pregunta de seguimiento sobre algo anterior
   - clarification: Solicitud de aclaración o más detalles

2. **Intención Principal:**
   - get_capabilities: Quiere saber qué puede hacer el sistema
   - understand_system: Quiere entender cómo funciona
   - get_help: Necesita orientación para usar el sistema
   - casual_chat: Conversación informal
   - technical_info: Busca información técnica específica

3. **Nivel de Experiencia Percibido:**
   - beginner: Usuario nuevo o con poca experiencia
   - intermediate: Usuario con alguna experiencia
   - expert: Usuario experimentado o técnico

4. **Tono Apropiado:**
   - friendly_casual: Amigable e informal
   - professional_helpful: Profesional pero accesible
   - technical_detailed: Técnico y detallado

Responde SOLO con JSON válido:
{{
  "interaction_type": "tipo",
  "main_intent": "intención",
  "expertise_level": "nivel",
  "appropriate_tone": "tono",
  "key_topics": ["tema1", "tema2"],
  "response_strategy": "estrategia recomendada"
}}"""

        try:
            response = await self.llm.ainvoke(context_prompt)
            content = str(response)
            
            # MEJORADO: Parsing más robusto de JSON
            try:
                # Estrategia 1: Intentar parsear directamente
                return json.loads(content)
            except json.JSONDecodeError:
                # Estrategia 2: Buscar JSON con regex más robusto
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    try:
                        return json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        # Estrategia 3: Limpiar contenido y reintentar
                        cleaned_content = content.strip()
                        # Remover texto antes y después del JSON
                        cleaned_content = re.sub(r'^[^{]*', '', cleaned_content)
                        cleaned_content = re.sub(r'[^}]*$', '', cleaned_content)
                        try:
                            return json.loads(cleaned_content)
                        except json.JSONDecodeError:
                            pass
                
                # Si todo falla, usar fallback básico
                logger.debug(f"No se pudo parsear JSON del LLM, usando fallback. Contenido: {content[:200]}...")
                return {
                    "interaction_type": "greeting",
                    "main_intent": "get_help",
                    "expertise_level": "beginner",
                    "appropriate_tone": "friendly_casual",
                    "key_topics": ["help"],
                    "response_strategy": "provide_overview"
                }
                
        except Exception as e:
            # MEJORADO: Log más específico sin warning
            logger.debug(f"Error en análisis contextual (no crítico): {str(e)[:100]}...")
            return {
                "interaction_type": "greeting",
                "main_intent": "get_help",
                "expertise_level": "beginner",
                "appropriate_tone": "friendly_casual",
                "key_topics": ["help"],
                "response_strategy": "provide_overview"
            }

    async def _generate_intelligent_response(self, query: str, context: Dict[str, Any]) -> str:
        """
        🧠 Genera respuesta inteligente basada en el análisis contextual
        adaptándose dinámicamente al usuario y la situación.
        """
        # Construir información del sistema de forma dinámica
        system_overview = self._build_dynamic_system_overview(context)
        
        # Historial conversacional para contexto
        history_str = "\n".join(self.conversation_history[-4:]) if self.conversation_history else "No hay historial previo"
        
        # 🧠 PROMPT INTELIGENTE ADAPTATIVO
        response_prompt = f"""Eres ChatMed 2.0, un asistente de IA médico avanzado y conversacional. Tu personalidad se adapta al contexto del usuario.

**ANÁLISIS CONTEXTUAL DE ESTA INTERACCIÓN:**
- Tipo de interacción: {context.get('interaction_type', 'greeting')}
- Intención principal: {context.get('main_intent', 'get_help')}
- Nivel de experiencia del usuario: {context.get('expertise_level', 'beginner')}
- Tono apropiado: {context.get('appropriate_tone', 'friendly_casual')}
- Temas clave: {', '.join(context.get('key_topics', ['general']))}
- Estrategia de respuesta: {context.get('response_strategy', 'provide_overview')}

**TU INFORMACIÓN INTERNA (usa según sea relevante):**
{system_overview}

**HISTORIAL CONVERSACIONAL RECIENTE:**
{history_str}

**INSTRUCCIONES ADAPTATIVAS:**

1. **Adapta tu respuesta** al nivel de experiencia detectado:
   - Beginner: Explicaciones simples, ejemplos prácticos, lenguaje accesible
   - Intermediate: Balance entre detalle técnico y claridad
   - Expert: Información técnica detallada, terminología especializada

2. **Ajusta tu tono** según el contexto:
   - Friendly_casual: Cálido, conversacional, usa emojis ocasionalmente
   - Professional_helpful: Profesional pero accesible, enfoque en utilidad
   - Technical_detailed: Preciso, detallado, terminología técnica apropiada

3. **Enfócate en la intención principal**:
   - Get_capabilities: Explica qué puedes hacer de forma práctica
   - Understand_system: Describe cómo funcionas y tu arquitectura
   - Get_help: Proporciona orientación específica y ejemplos
   - Casual_chat: Mantén conversación natural y amigable
   - Technical_info: Proporciona detalles técnicos precisos

4. **Sé transparente y honesto** sobre tus capacidades y limitaciones

5. **Incluye ejemplos prácticos** cuando sea apropiado

**CONSULTA ACTUAL DEL USUARIO:**
"{query}"

**TU RESPUESTA ADAPTATIVA:**"""

        try:
            response = await self.llm.ainvoke(response_prompt)
            # Verificar si la respuesta es un objeto con atributo content o una cadena directa
            if isinstance(response, str):
                response_text = response
            else:
                response_text = getattr(response, 'content', str(response))
            return response_text.strip()
            
        except Exception as e:
            logger.error(f"Error generando respuesta inteligente: {e}")
            return "Lo siento, estoy teniendo dificultades para procesar tu consulta. ¿Podrías intentar de nuevo?"

    def _build_dynamic_system_overview(self, context: Dict[str, Any]) -> str:
        """
        🧠 Construye descripción del sistema adaptada al contexto del usuario
        """
        expertise_level = context.get('expertise_level', 'beginner')
        main_intent = context.get('main_intent', 'get_help')
        
        overview_parts = []
        
        # Información básica siempre incluida
        overview_parts.append(f"**Sistema:** {self.system_info['name']} (Versión: {self.system_info['version']})")
        overview_parts.append(f"**Descripción:** {self.system_info['description']}")
        overview_parts.append(f"**Filosofía:** {self.system_info['philosophy']}")
        
        # Información de agentes según el nivel de experiencia
        if main_intent in ['get_capabilities', 'understand_system'] or expertise_level != 'beginner':
            overview_parts.append("\n**Agentes Especializados:**")
            
            for agent_name, agent_info in self.system_info['agents'].items():
                overview_parts.append(f"\n• **{agent_name}**: {agent_info['purpose']}")
                
                if expertise_level in ['intermediate', 'expert']:
                    # Incluir capacidades detalladas para usuarios más avanzados
                    capabilities = agent_info['capabilities'][:3]  # Primeras 3 capacidades
                    for cap in capabilities:
                        overview_parts.append(f"  - {cap}")
        
        # Información técnica para usuarios avanzados
        if expertise_level == 'expert' and main_intent == 'understand_system':
            overview_parts.append(f"\n**Tecnologías Principales:**")
            for tech in self.system_info['technologies']['frameworks'][:3]:
                overview_parts.append(f"  - {tech}")
        
        return "\n".join(overview_parts)

    def _update_conversation_context(self, query: str, response: str, context: Dict[str, Any]):
        """Actualiza el contexto conversacional para futuras interacciones"""
        self.context_analyzer['last_interaction_type'] = context.get('interaction_type')
        self.context_analyzer['user_expertise_level'] = context.get('expertise_level', 'unknown')
        self.context_analyzer['conversation_flow'].append({
            'timestamp': datetime.now(),
            'user_query': query[:100],  # Truncar para privacidad
            'interaction_type': context.get('interaction_type'),
            'main_intent': context.get('main_intent')
        })
        
        # Mantener solo las últimas 5 interacciones
        if len(self.context_analyzer['conversation_flow']) > 5:
            self.context_analyzer['conversation_flow'] = self.context_analyzer['conversation_flow'][-5:]

# Alias para compatibilidad
GreetingAgent = IntelligentGreetingAgent