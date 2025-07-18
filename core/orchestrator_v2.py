"""
ChatMed Flexible v2.0 - Orquestador Inteligente Ultra-Optimizado
=================================================================

Orquestador de nueva generación con:
- ⚡ Cache inteligente multinivel
- 🚀 Clasificación ultra-rápida con fallback optimizado
- 🔄 Arquitectura flexible y modular
- 📊 Monitoreo de rendimiento en tiempo real
- 🎯 Rutas optimizadas basadas en patrones de uso

Autor: Carmen Pascual
Versión: 2.0 - Flexible
"""

import logging
import os
import asyncio
import warnings
import time
from typing import Dict, Any, Tuple, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import re
from .memory_manager import load_memory, save_memory, UserMemory

# Corregido: Importar SecretStr desde la ruta correcta para evitar conflictos de pydantic.
from pydantic.v1.types import SecretStr

# NUEVO: Importar tipos de mensajes de OpenAI para compatibilidad
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

# --- Dependencias de LangChain ---
try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain no disponible. Funcionará con clasificación local.")

# --- Agentes Médicos Especializados ---
# Intentar importar agentes v2 disponibles
AGENTS_V2_AVAILABLE = False
try:
    # Imports absolutos para evitar problemas con imports relativos
    import sys
    import os
    current_dir = os.path.dirname(__file__)
    agents_dir = os.path.join(current_dir, '..', 'agents')
    sys.path.insert(0, agents_dir)
    
    # Importar agentes usando imports absolutos
    from agents.biochat_agent import BioChatAgent  # type: ignore
    from agents.greeting_agent import IntelligentGreetingAgent  # type: ignore
    from agents.pubmed_query_generator import PubMedQueryGenerator  # type: ignore
    from agents.sql_agent_flexible_enhanced import SQLAgentIntelligentEnhanced  # type: ignore  # SQL Agent original que funciona
    from agents.medgemma_clinical_agent import MedGemmaClinicalAgent  # type: ignore  # NUEVO: Agente MedGemma para análisis clínico
    from agents.fhir_agent_complete import FHIRMedicalAgent  # type: ignore  # Agente FHIR completo
    from agents.fhir_persistence_agent_old import FHIRPersistenceAgent  # type: ignore  # Agente de persistencia FHIR (versión antigua)
    AGENTS_V2_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Error importando agentes v2: {e}")
    AGENTS_V2_AVAILABLE = False

# Fallback a agentes v1 - DESHABILITADO para evitar conflictos
AGENTS_V1_AVAILABLE = False

# --- Configuración General ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings("ignore")
logger = logging.getLogger("ChatMedFlexibleV2")

class QueryType(Enum):
    """Tipos de consulta para clasificación rápida"""
    GREETING = "greeting"
    SQL_QUERY = "sql"
    FHIR_PROCESSING = "fhir"
    BIOCHAT_SEARCH = "biochat"
    CLINICAL_ANALYSIS = "clinical_analysis"  # NUEVO: Análisis clínico con MedGemma
    FOLLOW_UP = "follow_up"
    CLINICAL_NOTE = "clinical_note"
    UNKNOWN = "unknown"

@dataclass
class CacheEntry:
    """Entrada de cache con metadatos"""
    query_hash: str
    agent_type: str
    confidence: float
    result: Any
    timestamp: datetime
    hit_count: int = 0
    avg_response_time: float = 0.0
    
    def is_expired(self, ttl_minutes: int = 60) -> bool:
        """Verifica si la entrada ha expirado"""
        return datetime.now() - self.timestamp > timedelta(minutes=ttl_minutes)

@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento del orquestador"""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_response_time: float = 0.0
    agent_usage: Dict[str, int] = field(default_factory=dict)
    classification_time: float = 0.0
    execution_time: float = 0.0
    llm_calls: int = 0
    local_classifications: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        """Tasa de acierto del cache"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

class Colors:
    """Colores ANSI para la consola"""
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

class FlexibleOrchestrator:
    """
    🎯 Orquestador Flexible v2.0 - Ultra-Optimizado
    
    Características principales:
    - Cache inteligente multinivel con TTL
    - Clasificación ultra-rápida con patrones optimizados
    - Arquitectura modular y flexible
    - Monitoreo de rendimiento en tiempo real
    - Rutas optimizadas basadas en uso
    - Fallback inteligente sin dependencias externas
    """
    
    @staticmethod
    def _call_openai_native(client, messages, temperature=0.1, max_tokens=4000):
        """
        Función de compatibilidad para llamar a OpenAI nativo con streaming y logging.
        Ahora es un método estático para ser heredado correctamente.
        """
        try:
            # Logging más conciso
            from openai import OpenAI
            native_client = OpenAI()

            if isinstance(messages, list):
                openai_messages: List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]] = []
                for msg in messages:
                    role = "user"
                    content = ""
                    if hasattr(msg, 'content'):
                        content = str(msg.content)
                        if isinstance(msg, SystemMessage):
                            role = "system"
                    elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                        role = str(msg["role"])
                        content = str(msg["content"])
                    else:
                        content = str(msg)
                    
                    if role == "system":
                        openai_messages.append(ChatCompletionSystemMessageParam(role="system", content=content))
                    else:
                        openai_messages.append(ChatCompletionUserMessageParam(role="user", content=content))
            else:
                content = messages.content if hasattr(messages, 'content') else str(messages)
                openai_messages = [ChatCompletionUserMessageParam(role="user", content=str(content))]

            # Sin streaming para el orquestador
            response = native_client.chat.completions.create(
                model='gpt-4o',
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content or ""

            if not content.strip():
                content = '{"success": false, "message": "Error: Respuesta vacía del LLM"}'

            return content

        except Exception as e:
            error_msg = f"Error en llamada OpenAI del Orquestador: {str(e)}"
            logger.error(f"Error en _call_openai_native (Orquestador): {e}", exc_info=True)
            return '{"success": false, "message": "Error crítico en la llamada al LLM."}'
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 enable_cache: bool = True,
                 enable_performance_monitoring: bool = True,
                 cache_ttl_minutes: int = 60):
        """
        Inicializa el Orquestador Flexible v2.0
        """
        logger.info("🚀 Inicializando ChatMed Flexible Orchestrator v2.0...")
        
        # Configuración básica
        self.config_path = config_path
        self.enable_cache = enable_cache
        self.enable_performance_monitoring = enable_performance_monitoring
        self.cache_ttl_minutes = cache_ttl_minutes
        
        # Cargar configuración flexible
        self._load_flexible_config()
        
        # Inicializar LLM para clasificación inteligente
        self.llm_classifier = None
        self._initialize_llm_classifier()
        
        # Inicializar agentes v2
        self.agents = {}
        self._initialize_agents_v2()
        
        # Cache y métricas
        self.query_cache = {} if enable_cache else None
        self.classification_cache = {} if enable_cache else None
        self.metrics = PerformanceMetrics() if enable_performance_monitoring else None
        
        # Configuración de clasificación inteligente
        self.use_intelligent_classification = True
        self.classification_confidence_threshold = 0.6
        
        # Memoria de conversación
        self.conversation_history = []
        self.current_user_id = "default"
        
        # Cargar memoria de usuario
        self.user_memories = load_memory()
        
        logger.info("✅ Orquestador Flexible v2.0 inicializado")
        self._log_initialization_summary()
    
    def _load_flexible_config(self):
        try:
            config_paths = [
                "chatmed_v2_flexible/config/orchestrator_config.yaml",
                "../config/orchestrator_config.yaml"
            ]
            
            # Solo agregar config_path si no es None
            if self.config_path is not None:
                config_paths.insert(0, self.config_path)
            
            for path in config_paths:
                if path is not None and os.path.exists(path):
                    self.flexible_config = {
                        'cache_enabled': True,
                        'llm_model': 'gpt-4o',
                        'fallback_confidence_threshold': 0.6,
                        'max_conversation_history': 20,
                        'enable_smart_routing': True,
                        'enable_learning': True
                    }
                    logger.info(f"✅ Configuración flexible cargada desde: {path}")
                    return
            
            self.flexible_config = {
                'cache_enabled': True,
                'llm_model': 'gpt-4o',
                'fallback_confidence_threshold': 0.6,
                'max_conversation_history': 20,
                'enable_smart_routing': True,
                'enable_learning': True
            }
            logger.info("⚠️ Usando configuración por defecto")
            
        except Exception as e:
            logger.error(f"❌ Error cargando configuración: {e}")
            self.flexible_config = {}
    
    def _initialize_llm_classifier(self):
        if not LANGCHAIN_AVAILABLE:
            logger.warning("⚠️ LangChain no disponible. Usando clasificación local.")
            return
            
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("⚠️ OPENAI_API_KEY no encontrada. Clasificación local activada.")
                return
            
            model = self.flexible_config.get('llm_model', 'gpt-4o')
            self.llm_classifier = ChatOpenAI()
            logger.info(f"✅ Clasificador LLM ({model}) inicializado.")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando LLM: {e}")
    
    def _initialize_agents_v2(self):
        logger.info("🔧 Inicializando agentes v2.0...")
        
        db_path = self._find_database()
        if not db_path:
            logger.error("❌ Base de datos no encontrada")
            return
        
        llm_instance = self.llm_classifier
        
        if not llm_instance:
            logger.warning("⚠️ No se puede inicializar agentes que requieren LLM. ¿Falta la API Key de OpenAI?")
            logger.warning("⚠️ Algunas funciones estarán limitadas.")

        try:
            if AGENTS_V2_AVAILABLE:
                agents_initialized = []
                logger.info(f"[DEBUG] AGENTS_V2_AVAILABLE: {AGENTS_V2_AVAILABLE}")
                # Crear cliente nativo de OpenAI para agentes que lo requieren
                try:
                    from openai import OpenAI
                    openai_client = OpenAI()
                except Exception as e:
                    logger.warning(f"⚠️ No se pudo crear cliente OpenAI nativo: {e}")
                    openai_client = None
                
                if llm_instance:
                    try:
                        self.agents['greeting'] = IntelligentGreetingAgent(llm=llm_instance)  # type: ignore
                        agents_initialized.append('greeting')
                        logger.info("[DEBUG] Agente 'greeting' inicializado")
                    except Exception as e:
                        logger.warning(f"⚠️ Error inicializando agente de saludos: {e}")
                
                if llm_instance:
                    try:
                        self.agents['biochat'] = BioChatAgent(llm=llm_instance, medgemma_agent=None)  # type: ignore
                        agents_initialized.append('biochat')
                        logger.info("[DEBUG] Agente 'biochat' inicializado")
                    except Exception as e:
                        logger.warning(f"⚠️ Error inicializando agente BioChat: {e}")
                
                if openai_client:
                    try:
                        self.agents['pubmed'] = PubMedQueryGenerator(openai_client=openai_client)
                        agents_initialized.append('pubmed')
                        logger.info("[DEBUG] Agente 'pubmed' inicializado")
                    except Exception as e:
                        logger.warning(f"⚠️ Error inicializando agente PubMed: {e}")
                
                if llm_instance and db_path:
                    try:
                        medgemma_agent = None
                        try:
                            medgemma_agent = MedGemmaClinicalAgent()
                            logger.info("✅ Agente MedGemma creado para análisis clínico")
                        except Exception as e:
                            logger.warning(f"⚠️ Error creando agente MedGemma: {e}")
                        
                        sql_agent_instance = SQLAgentIntelligentEnhanced(
                            db_path=db_path, 
                            llm=llm_instance,
                            medgemma_agent=None  # Se asignará después si está disponible
                        )
                        self.agents['sql'] = sql_agent_instance
                        agents_initialized.append('sql')
                        logger.info("[DEBUG] Agente 'sql' inicializado")
                        
                        try:
                            self.agents['fhir'] = FHIRMedicalAgent(db_path=db_path, llm=llm_instance, sql_agent=sql_agent_instance, medgemma_agent=None)  # type: ignore
                            agents_initialized.append('fhir')
                            logger.info("[DEBUG] Agente 'fhir' inicializado")
                        except Exception as e:
                            logger.warning(f"⚠️ Error inicializando agente FHIR: {e}")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Error inicializando agente SQL Enhanced: {e}")
                
                if llm_instance and db_path:
                    try:
                        self.agents['fhir_persistence'] = FHIRPersistenceAgent(db_path=db_path, llm_client=llm_instance)
                        agents_initialized.append('fhir_persistence')
                        logger.info("[DEBUG] Agente 'fhir_persistence' inicializado")
                    except Exception as e:
                        logger.warning(f"⚠️ Error inicializando agente de persistencia: {e}")
                
                # Inicializar agente MedGemma Clínico
                try:
                    medgemma_agent_instance = MedGemmaClinicalAgent(llm=llm_instance)
                    self.agents['clinical_analysis'] = medgemma_agent_instance
                    agents_initialized.append('clinical_analysis')
                    logger.info(f"✅ Agente MedGemma Clínico inicializado: {medgemma_agent_instance}")
                    logger.info(f"[DEBUG] self.agents['clinical_analysis']: {self.agents.get('clinical_analysis')}")
                    # Pasar MedGemma a los agentes para análisis clínico integrado
                    if 'sql' in self.agents:
                        self.agents['sql'].medgemma_agent = medgemma_agent_instance
                        logger.info("✅ MedGemma integrado con agente SQL")
                    if 'biochat' in self.agents:
                        self.agents['biochat'].medgemma_agent = medgemma_agent_instance
                        logger.info("✅ MedGemma integrado con agente BioChat")
                    if 'fhir' in self.agents:
                        self.agents['fhir'].medgemma_agent = medgemma_agent_instance
                        logger.info("✅ MedGemma integrado con agente FHIR")
                except Exception as e:
                    logger.warning(f"⚠️ Error inicializando agente MedGemma: {e}")
                
                if agents_initialized:
                    logger.info(f"✅ {len(agents_initialized)} agentes v2.0 inicializados: {', '.join(agents_initialized)}")
                    logger.info(f"[DEBUG] Agentes disponibles tras inicialización: {list(self.agents.keys())}")
                else:
                    logger.warning("⚠️ No se pudo inicializar ningún agente v2.0")
            else:
                logger.warning("⚠️ Agentes v2.0 no disponibles - sistema limitado.")
        except Exception as e:
            logger.error(f"❌ Error crítico inicializando agentes: {e}")
        logger.info(f"📊 Agentes disponibles: {list(self.agents.keys())}")

    def _find_database(self) -> Optional[str]:
        # FORZAR uso de database_new.sqlite3.db específicamente
        target_db = 'database_new.sqlite3.db'
        
        if os.path.exists(target_db):
            logger.info(f"✅ Base de datos encontrada: {target_db}")
            return target_db
        
        # Fallback a rutas relativas si no existe en el directorio actual
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', '..', 'database_new.sqlite3.db'),
            os.path.join(os.getcwd(), 'database_new.sqlite3.db'),
            './database_new.sqlite3.db',
            '../database_new.sqlite3.db',
            '../../database_new.sqlite3.db'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"✅ Base de datos encontrada (fallback): {path}")
                return path
        
        logger.error("❌ Base de datos no encontrada")
        return None
    
    async def classify_query_ultra_fast(self, query: str) -> Tuple[str, float]:
        start_time = time.time()
        
        query_clean = query.strip().lower()
        
        # 1. Detección rápida de saludos básicos (sin LLM)
        simple_greetings = {
            "hola", "hi", "hello", "hey", "buenos días", "buenas tardes", 
            "buenas noches", "good morning", "good afternoon", "good evening",
            "¿qué tal?", "que tal", "cómo estás", "como estas", "¿cómo está?",
            "como esta", "saludos", "gracias", "thank you", "thanks", "ok", "vale",
            "que puedes hacer", "qué puedes hacer por mi", "que puedes hacer por mi",
            "qué puedes hacer por mi", "que sabes hacer", "qué sabes hacer"
        }
        
        # Normalizar la consulta para comparación
        query_normalized = query_clean.replace("?", "").replace("¿", "").strip()
        
        if query_normalized in simple_greetings:
            return "greeting", 0.99
        
        # 2. Clasificación inteligente con LLM
        if self.llm_classifier:
            try:
                agent, confidence = await self._classify_with_intelligent_llm(query)
                if confidence >= 0.6:  # Umbral de confianza
                    if self.metrics:
                        self.metrics.llm_calls += 1
                    return agent, confidence
            except Exception as e:
                logger.warning(f"⚠️ Error en clasificación LLM: {e}")
        
        # 3. Fallback inteligente sin patrones hardcodeados
        agent, confidence = await self._smart_fallback_classification(query)
        
        if self.metrics:
            self.metrics.classification_time += time.time() - start_time
            self.metrics.local_classifications += 1
        
        return agent, confidence

    async def _classify_with_intelligent_llm(self, query: str) -> Tuple[str, float]:
        """Clasificación inteligente usando LLM con prompt específico"""
        
        if not self.llm_classifier:
            logger.warning("⚠️ LLM no disponible para clasificación inteligente")
            return "greeting", 0.3
        
        prompt = f'''
Analiza esta consulta médica y determina qué tipo de agente especializado debe manejarla.

CONSULTA: "{query}"

TIPOS DE AGENTES DISPONIBLES:

1. GREETING: Para saludos, agradecimientos, preguntas sobre capacidades del sistema
2. BIOCHAT: Para búsquedas de literatura médica, ensayos clínicos, artículos científicos, investigación biomédica, estudios recientes, papers científicos
3. SQL: Para consultas de datos de pacientes, estadísticas médicas, información de base de datos clínica, búsqueda de pacientes
4. FHIR: Para gestión de registros médicos, notas clínicas, información de pacientes
5. CLINICAL_ANALYSIS: Para análisis clínico básico, explicaciones médicas simples, conceptos médicos, medicamentos, diagnósticos

CRITERIOS DE CLASIFICACIÓN DETALLADOS:

- CLINICAL_ANALYSIS: 
  * Explicaciones básicas de enfermedades ("¿Qué es la diabetes?")
  * Conceptos médicos simples ("¿Qué es la hipertensión?")
  * Información de medicamentos básica ("¿Para qué sirve la metformina?")
  * Síntomas y diagnósticos simples ("¿Cuáles son los síntomas de la gripe?")
  * Interacciones farmacológicas básicas ("¿Puedo tomar paracetamol con ibuprofeno?")

- BIOCHAT: 
  * Búsquedas de investigación ("¿Qué estudios recientes hay sobre...?")
  * Ensayos clínicos ("¿Cuáles son los últimos ensayos sobre...?")
  * Literatura científica ("¿Qué papers hay sobre...?")
  * Investigación biomédica ("¿Qué investigaciones recientes...?")
  * Evidencia científica ("¿Qué evidencia hay sobre...?")
  * Mecanismos moleculares ("¿Cuáles son los mecanismos moleculares...?")
  * Estudios avanzados ("¿Cuál es el estado actual de la investigación...?")

- SQL: 
  * Datos de pacientes ("Busca pacientes con...")
  * Estadísticas ("¿Cuántos pacientes...?")
  * Información de base de datos ("Muéstrame todos los...")
  * Búsquedas específicas ("¿Cuál es el último paciente?")

- FHIR: 
  * Gestión de registros ("Registrar paciente...")
  * Notas clínicas ("Crear nota clínica...")
  * Información estructurada de pacientes

- GREETING: 
  * Saludos ("Hola", "¿Qué puedes hacer?")
  * Preguntas sobre el sistema

Ejemplos específicos:
- "¿Qué es la diabetes?" → CLINICAL_ANALYSIS
- "¿Para qué sirve la metformina?" → CLINICAL_ANALYSIS
- "¿Cuáles son los síntomas de la hipertensión?" → CLINICAL_ANALYSIS
- "¿Qué investigaciones recientes hay sobre diabetes?" → BIOCHAT
- "¿Cuáles son los últimos ensayos clínicos sobre CAR-T?" → BIOCHAT
- "¿Qué estudios hay sobre nuevos tratamientos para diabetes?" → BIOCHAT
- "Busca pacientes con diabetes" → SQL
- "¿Cuántos pacientes hay?" → SQL
- "¿Cuál es el último paciente?" → SQL
- "Registrar paciente Juan Pérez" → FHIR
- "Hola" → GREETING

Responde SOLO con este JSON:
{{
  "agent": "greeting|biochat|sql|fhir|clinical_analysis",
  "confidence": 0.0-1.0,
  "reasoning": "explicación breve de la clasificación"
}}
'''
        
        try:
            response = await self.llm_classifier.ainvoke(prompt)
            content = str(response.content)
            
            # Extraer JSON de la respuesta
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group(0))
                agent = result.get("agent", "greeting").lower()
                confidence = float(result.get("confidence", 0.5))
                reasoning = result.get("reasoning", "")
                
                # Validar agente
                valid_agents = ["greeting", "sql", "fhir", "biochat"]
                if agent not in valid_agents:
                    agent = "greeting"
                    confidence = 0.3
                
                logger.info(f"🧠 LLM clasificó como '{agent}' (confianza: {confidence:.2f}) - Razón: {reasoning}")
                return agent, confidence
            else:
                logger.warning("⚠️ No se encontró JSON válido en respuesta del LLM")
                return "greeting", 0.3
                
        except Exception as e:
            logger.error(f"❌ Error en clasificación LLM: {e}")
            return "greeting", 0.3

    async def _smart_fallback_classification(self, query: str) -> Tuple[str, float]:
        """Clasificación inteligente usando LLM con prompt específico para fallback"""
        
        if not self.llm_classifier:
            logger.warning("⚠️ LLM no disponible para clasificación de fallback")
            return "greeting", 0.3
        
        prompt = f'''
Eres un clasificador inteligente de consultas médicas. Analiza la siguiente consulta y determina qué tipo de agente especializado debe manejarla.

CONSULTA: "{query}"

TIPOS DE AGENTES DISPONIBLES:

1. GREETING: Para saludos, agradecimientos, preguntas sobre capacidades del sistema
2. BIOCHAT: Para búsquedas de literatura médica, ensayos clínicos, artículos científicos, investigación biomédica, estudios recientes, papers científicos
3. SQL: Para consultas de datos de pacientes, estadísticas médicas, información de base de datos clínica, búsqueda de pacientes
4. FHIR: Para gestión de registros médicos, notas clínicas, información de pacientes
5. CLINICAL_ANALYSIS: Para análisis clínico básico, explicaciones médicas simples, conceptos médicos, medicamentos, diagnósticos

CRITERIOS DE CLASIFICACIÓN DETALLADOS:

- CLINICAL_ANALYSIS: 
  * Explicaciones básicas de enfermedades ("¿Qué es la diabetes?")
  * Conceptos médicos simples ("¿Qué es la hipertensión?")
  * Información de medicamentos básica ("¿Para qué sirve la metformina?")
  * Síntomas y diagnósticos simples ("¿Cuáles son los síntomas de la gripe?")
  * Interacciones farmacológicas básicas ("¿Puedo tomar paracetamol con ibuprofeno?")

- BIOCHAT: 
  * Búsquedas de investigación ("¿Qué estudios recientes hay sobre...?")
  * Ensayos clínicos ("¿Cuáles son los últimos ensayos sobre...?")
  * Literatura científica ("¿Qué papers hay sobre...?")
  * Investigación biomédica ("¿Qué investigaciones recientes...?")
  * Evidencia científica ("¿Qué evidencia hay sobre...?")
  * Mecanismos moleculares ("¿Cuáles son los mecanismos moleculares...?")
  * Estudios avanzados ("¿Cuál es el estado actual de la investigación...?")

- SQL: 
  * Datos de pacientes ("Busca pacientes con...")
  * Estadísticas ("¿Cuántos pacientes...?")
  * Información de base de datos ("Muéstrame todos los...")
  * Búsquedas específicas ("¿Cuál es el último paciente?")

- FHIR: 
  * Gestión de registros ("Registrar paciente...")
  * Notas clínicas ("Crear nota clínica...")
  * Información estructurada de pacientes

- GREETING: 
  * Saludos ("Hola", "¿Qué puedes hacer?")
  * Preguntas sobre el sistema

Ejemplos específicos:
- "¿Qué es la diabetes?" → CLINICAL_ANALYSIS
- "¿Para qué sirve la metformina?" → CLINICAL_ANALYSIS
- "¿Qué investigaciones recientes hay sobre diabetes?" → BIOCHAT
- "¿Cuáles son los últimos ensayos clínicos sobre CAR-T?" → BIOCHAT
- "Busca pacientes con diabetes" → SQL
- "¿Cuántos pacientes hay?" → SQL
- "Registrar paciente Juan Pérez" → FHIR
- "Hola" → GREETING

Responde SOLO con este JSON:
{{
  "agent": "greeting|biochat|sql|fhir|clinical_analysis",
  "confidence": 0.0-1.0,
  "reasoning": "explicación breve de la clasificación"
}}
'''
        
        try:
            response = await self.llm_classifier.ainvoke(prompt)
            content = str(response.content)
            
            # Extraer JSON de la respuesta
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group(0))
                agent = result.get("agent", "greeting").lower()
                confidence = float(result.get("confidence", 0.5))
                reasoning = result.get("reasoning", "")
                
                # Validar agente
                valid_agents = ["greeting", "sql", "fhir", "biochat"]
                if agent not in valid_agents:
                    agent = "greeting"
                    confidence = 0.3
                
                logger.info(f"🧠 LLM Fallback clasificó como '{agent}' (confianza: {confidence:.2f}) - Razón: {reasoning}")
                return agent, confidence
            else:
                logger.warning("⚠️ No se encontró JSON válido en respuesta del LLM de fallback")
                return "greeting", 0.3
                
        except Exception as e:
            logger.error(f"❌ Error en clasificación LLM de fallback: {e}")
            return "greeting", 0.3

    def _classify_with_basic_patterns(self, query: str) -> Tuple[str, float]:
        """ELIMINADO - Ya no usamos patrones hardcodeados"""
        # Este método se mantiene por compatibilidad pero ya no se usa
        return "greeting", 0.3
    
    async def process_query_optimized(self, query: str, stream_callback=None) -> Dict[str, Any]:
        start_time = time.time()
        
        if self.metrics:
            self.metrics.total_queries += 1
        
        if self.enable_cache and self.query_cache:
            query_hash = hash(query.lower().strip())
            if query_hash in self.query_cache:
                cached = self.query_cache[query_hash]
                if not cached.is_expired(self.cache_ttl_minutes):
                    return {
                        "success": True,
                        "agent_type": cached.agent_type,
                        "message": cached.result,
                        "confidence": cached.confidence,
                        "from_cache": True
                    }
        
        mem_context = self._get_memory_context()
        augmented_query = f"{mem_context}\n{query}" if mem_context else query

        agent_type, confidence = await self.classify_query_ultra_fast(augmented_query)
        
        if stream_callback:
            stream_callback(f"🎯 Derivando a: {agent_type}")
        
        try:
            if agent_type in self.agents and self.agents[agent_type] is not None:
                agent = self.agents[agent_type]
                
                context = self._get_user_memory_context()
                
                # Llamada al agente con manejo mejorado de errores
                try:
                    if asyncio.iscoroutinefunction(agent.process_query):
                        result = await agent.process_query(query=query, context=context)
                    else:
                        result = agent.process_query(query=query, context=context)
                    
                    # Validar que el resultado no sea None
                    if result is None:
                        result = {"success": False, "message": "El agente no devolvió una respuesta válida"}
                    
                    # Asegurar que el resultado sea un diccionario
                    if not isinstance(result, dict):
                        result = {"success": True, "message": str(result)}
                    
                except Exception as agent_error:
                    logger.error(f"❌ Error ejecutando agente {agent_type}: {agent_error}")
                    result = {
                        "success": False, 
                        "message": f"Error en el agente {agent_type}: {str(agent_error)}"
                    }

            else:
                result = {
                    "success": False,
                    "message": f"Agente '{agent_type}' no disponible."
                }
            
            # Extraer el mensaje limpio
            clean_message = self._extract_clean_message(result, agent_type)
            
            # Crear respuesta estructurada
            response_dict = {
                "success": result.get("success", True),
                "agent_type": agent_type,
                "confidence": confidence,
                "message": clean_message,
                "original_result": result,
                "processing_time": time.time() - start_time
            }
            
            if self.enable_cache and self.query_cache is not None:
                self.query_cache[query_hash] = CacheEntry(str(query_hash), agent_type, confidence, clean_message, datetime.now())
            
            self._update_conversation_memory_v2(query, agent_type, result)
            self._update_user_memory(result)
            
            return response_dict
                
        except Exception as e:
            logger.error(f"❌ Error crítico procesando consulta: {e}")
            return {
                "success": False,
                "agent_type": "unknown",
                "confidence": 0.0,
                "message": f"❌ Error procesando consulta: {str(e)}",
                "processing_time": time.time() - start_time
            }
    
    def _format_response_v2(self, result: Any, agent_type: str, confidence: float, query: str) -> str:
        clean_message = self._extract_clean_message(result, agent_type)
        # Preservar saltos de línea y aplicar colores
        formatted_message = self._apply_agent_colors(clean_message, agent_type)
        # Asegurar que los saltos de línea se mantengan
        return formatted_message.replace('\\n', '\n')
            
    def _extract_clean_message(self, result: Any, agent_type: str) -> str:
        """Extrae el mensaje limpio del resultado del agente"""
        if isinstance(result, dict):
            # PRIORIDAD 1: Usar formatted_data si está disponible (SQL Agent mejorado)
            if 'formatted_data' in result and result['formatted_data']:
                return str(result['formatted_data'])
            
            # PRIORIDAD 2: Buscar diferentes claves posibles para el mensaje
            message_keys = ['message', 'response_text', 'response', 'content', 'text']
            for key in message_keys:
                if key in result and result[key]:
                    return str(result[key])
            
            # PRIORIDAD 3: Si es SQL Agent y tiene datos, formatear básicamente
            if agent_type == 'sql' and 'data' in result and result['data']:
                return self._format_sql_data_basic(result['data'])
            
            # PRIORIDAD 4: Si es SQL Agent y tiene SQL pero no datos, mostrar mensaje informativo
            if agent_type == 'sql' and 'sql' in result:
                if result.get('success'):
                    return "✅ Consulta SQL ejecutada correctamente. Los resultados se han procesado."
                else:
                    return f"❌ Error en la consulta SQL: {result.get('error', 'Error desconocido')}"
            
            # PRIORIDAD 5: Si es análisis clínico con MedGemma
            if 'clinical_analysis' in result and result['clinical_analysis']:
                analysis = result['clinical_analysis']
                if isinstance(analysis, dict) and 'analysis' in analysis:
                    return f"🏥 Análisis clínico: {analysis['analysis']}"
            
            # Si no hay mensaje específico, usar todo el resultado
            return str(result)
        elif isinstance(result, str):
            return result
        else:
            return str(result)
    
    def _format_sql_data_basic(self, data: List[Dict[str, Any]]) -> str:
        """Formatea datos SQL de manera básica si no hay formatted_data"""
        if not data:
            return "📊 **No se encontraron resultados**"
        
        formatted = []
        formatted.append("📊 **RESULTADOS ENCONTRADOS**")
        formatted.append("")
        
        for i, item in enumerate(data[:10], 1):  # Mostrar máximo 10
            if isinstance(item, dict):
                item_str = " | ".join([f"{k}: {v}" for k, v in item.items()])
                formatted.append(f"   {i}. {item_str}")
            else:
                formatted.append(f"   {i}. {item}")
        
        if len(data) > 10:
            formatted.append("")
            formatted.append(f"📄 *... y {len(data) - 10} resultados más*")
        
        return "\n".join(formatted)
    
    def _apply_agent_colors(self, message: str, agent_type: str) -> str:
        colors = {'greeting': Colors.CYAN, 'sql': Colors.GREEN, 'fhir': Colors.BLUE, 'biochat': Colors.PURPLE, 'clinical_analysis': Colors.YELLOW}
        icons = {'greeting': '👋', 'sql': '🗄️', 'fhir': '🏥', 'biochat': '📚', 'clinical_analysis': '🏥'}
        color = colors.get(agent_type, Colors.END)
        icon = icons.get(agent_type, '🤖')
        return f"{color}{icon} {message}{Colors.END}"
    
    def _update_conversation_memory_v2(self, query: str, agent: str, result: Any):
        self.conversation_history.append({'query': query, 'agent': agent, 'result_summary': str(result)[:100]})
        if len(self.conversation_history) > self.flexible_config.get('max_conversation_history', 20):
            self.conversation_history.pop(0)
    
    def _extract_result_summary_v2(self, result: Any) -> str:
        if isinstance(result, dict) and 'summary' in result:
            return result['summary']
        return str(result)[:100]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        if not self.metrics:
            return {"error": "Métricas no habilitadas"}
        
        return {
            "total_queries": self.metrics.total_queries,
            "cache_hit_rate": f"{self.metrics.cache_hit_rate:.1%}",
            "avg_response_time": f"{self.metrics.avg_response_time:.3f}s",
            "agent_usage": self.metrics.agent_usage
        }

    def _log_initialization_summary(self):
        logger.info("📊 RESUMEN DE INICIALIZACIÓN:")
        logger.info(f"   ⚡ Cache: {'✅ Habilitado' if self.enable_cache else '❌ Deshabilitado'}")
        logger.info(f"   📈 Métricas: {'✅ Habilitadas' if self.enable_performance_monitoring else '❌ Deshabilitadas'}")
        logger.info(f"   🤖 LLM: {'✅ Disponible' if self.llm_classifier else '❌ No disponible'}")

    async def _looks_like_clinical_note(self, text: str) -> bool:
        """
        🧠 CLASIFICACIÓN SEMÁNTICA INTELIGENTE
        Usa LLM para determinar si es una nota clínica basándose en el significado, no en patrones
        """
        # Si el texto es muy corto, no es nota clínica
        if len(text.split()) < 10:
            return False
            
        # Si no tenemos LLM, usar heurística simple pero flexible
        if not self.llm_classifier:
            return self._simple_clinical_note_heuristic(text)
        
        try:
            # PROMPT INTELIGENTE para clasificación semántica
            prompt = f"""
Eres un clasificador experto de documentos médicos. Tu tarea es determinar si el siguiente texto es una NOTA CLÍNICA que debe procesarse con el agente FHIR.

DEFINICIÓN DE NOTA CLÍNICA:
- Documento que registra información médica de un paciente
- Contiene diagnósticos, tratamientos, medicamentos, signos vitales
- Puede tener formato estructurado (SOAP, H&P) o narrativo
- Es información para REGISTRAR, no para CONSULTAR

EJEMPLOS DE NOTAS CLÍNICAS:
✅ "S – Subjetivo: Paciente de 45 años con dolor lumbar..."
✅ "Control de paciente diabético. HbA1c 7.2%. Continuar metformina..."
✅ "Paciente presenta fiebre 38.5°C. Prescribir paracetamol..."
✅ "Motivo de consulta: Dolor torácico. Diagnóstico: Angina..."

EJEMPLOS DE NO NOTAS CLÍNICAS:
❌ "¿Qué medicamentos toma el paciente 1010?"
❌ "¿Cuántos pacientes con diabetes tenemos?"
❌ "Buscar información sobre el paciente Juan"
❌ "Mostrar las alergias del paciente 123"

TEXTO A ANALIZAR:
"{text}"

Responde SOLO con "SI" si es una nota clínica para registrar, o "NO" si no lo es.
"""
            
            response = await self.llm_classifier.ainvoke(prompt)
            result = str(response.content).strip().upper()
            
            is_clinical_note = "SI" in result or "YES" in result
            logger.info(f"🧠 Clasificación semántica: {'NOTA CLÍNICA' if is_clinical_note else 'NO ES NOTA CLÍNICA'}")
            
            return is_clinical_note
            
        except Exception as e:
            logger.error(f"Error en clasificación semántica: {e}")
            return self._simple_clinical_note_heuristic(text)
    
    def _simple_clinical_note_heuristic(self, text: str) -> bool:
        """
        Heurística simple de respaldo cuando no hay LLM disponible
        """
        text_lower = text.lower()
        
        # Detectar si es una pregunta
        question_indicators = ['¿', '?', 'cuántos', 'cuál', 'qué', 'dónde', 'cuándo', 'cómo']
        is_question = any(indicator in text_lower for indicator in question_indicators)
        
        if is_question:
            return False  # Las preguntas van a SQL, no a FHIR
        
        # Detectar información médica estructurada
        medical_indicators = [
            'diagnóstico', 'tratamiento', 'medicación', 'prescribir', 'mg/', 'mmhg',
            'signos vitales', 'exploración física', 'antecedentes', 'evolución'
        ]
        
        medical_matches = sum(1 for indicator in medical_indicators if indicator in text_lower)
        
        # Es nota clínica si tiene información médica Y NO es pregunta Y es texto largo
        return medical_matches >= 2 and len(text.split()) > 20

    def _get_memory_context(self) -> str:
        mem = self.user_memories.get(self.current_user_id)
        if not mem: return ""
        parts = []
        if mem.last_patient_id: parts.append(f"ultimo_paciente_id={mem.last_patient_id}")
        if mem.last_tables: parts.append(f"tablas_recientes={','.join(mem.last_tables[:3])}")
        return "MEMORIA_CONTEXTUAL: " + "; ".join(parts) if parts else ""

    def _update_user_memory(self, result: Any):
        if not isinstance(result, dict): return
        mem = self.user_memories.setdefault(self.current_user_id, UserMemory())
        pid = result.get('patient_id')
        if pid:
            mem.last_patient_id = str(pid)
        mem.updated_at = datetime.utcnow()
        save_memory(self.user_memories)

    def _get_user_memory_context(self) -> Dict[str, Any]:
        mem = self.user_memories.get(self.current_user_id)
        if mem and mem.last_patient_id:
            return {'last_patient_id': mem.last_patient_id}
        return {}
        
class IntelligentOrchestratorV3(FlexibleOrchestrator):
    """
    🧠 Orquestador Inteligente V3 - Enrutamiento Basado en Intención
    ================================================================
    Esta versión reemplaza la clasificación por patrones con un sistema de
    enrutamiento de dos etapas basado en análisis de intención y puntuación de agentes.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("🚀 Iniciando IntelligentOrchestratorV3 con enrutamiento avanzado.")
        # Limpiar patrones de fallback para forzar el nuevo sistema
        self.basic_fallback_patterns = {}
        self.use_intelligent_classification = True

    async def process_query_optimized(self, query: str, stream_callback=None) -> Union[Dict[str, Any], str]:
        """
        VERSIÓN SIMPLIFICADA: Procesamiento directo con clasificación LLM
        """
        start_time = time.time()
        
        # 0. DETECCIÓN RÁPIDA DE NOTA CLÍNICA (mejorada)
        if await self._looks_like_clinical_note(query):
            agent_name, best_score, justification = "fhir", 0.99, "Detectado formato de nota clínica"
            logger.info("📋 Nota clínica detectada, usando FHIR")
        else:
            # 1. CLASIFICACIÓN SIMPLE CON LLM
            logger.info("🧠 Clasificando consulta con LLM...")
            intent_analysis = await self._analyze_intent_and_entities(query)
            agent_name, best_score, justification = await self._score_and_select_agent(intent_analysis, query)
        
        # Mostrar agente seleccionado
        agent_icons = {
            'greeting': '👋',
            'sql': '🗄️', 
            'fhir': '📋',
            'biochat': '🔬',
            'clinical_analysis': '🧠'
        }
        
        agent_names = {
            'greeting': 'Saludo',
            'sql': 'Base de Datos',
            'fhir': 'Registros Médicos', 
            'biochat': 'Investigación Biomédica',
            'clinical_analysis': 'Análisis Clínico'
        }
        
        icon = agent_icons.get(agent_name, '🤖')
        name = agent_names.get(agent_name, 'Agente')
        
        if stream_callback:
            stream_callback(f"🎯 {icon} {name}")
        
        # 2. EJECUCIÓN DEL AGENTE
        agent = self.agents.get(agent_name)
        if not agent:
            return {
                'success': False,
                'agent_type': agent_name,
                'confidence': 0.0,
                'message': f"Agente '{agent_name}' no disponible. Agentes disponibles: {list(self.agents.keys())}",
                'data': [],
                'processing_time': time.time() - start_time
            }
            
        # Pasar contexto de memoria al agente si es necesario
        context = self._get_user_memory_context()
            
        # Ejecutar agente
        try:
            import inspect
            sig = inspect.signature(agent.process_query)
            
            call_kwargs: Dict[str, Any] = {'query': query}
            if 'stream_callback' in sig.parameters:
                call_kwargs['stream_callback'] = stream_callback
            if 'context' in sig.parameters:
                call_kwargs['context'] = context

            if asyncio.iscoroutinefunction(agent.process_query):
                result = await agent.process_query(**call_kwargs)
            else:
                result = agent.process_query(**call_kwargs)

        except Exception as e:
            logger.error(f"❌ Error ejecutando el agente '{agent_name}': {e}", exc_info=True)
            result = {
                'success': False,
                'message': f"Error al procesar la consulta con el agente {agent_name}: {str(e)}"
            }

        # Actualizar memoria
        self._update_conversation_memory_v2(query, agent_name, result)
        self._update_user_memory(result)
        
        # Extraer el mensaje limpio
        clean_message = self._extract_clean_message(result, agent_name)
        
        # Crear respuesta estructurada
        response_dict = {
            'success': result.get('success', True) if isinstance(result, dict) else True,
            'agent_type': agent_name,
            'confidence': best_score,
            'message': clean_message,
            'original_result': result,
            'processing_time': time.time() - start_time
        }
        
        return response_dict

    async def _analyze_intent_and_entities(self, query: str) -> Dict[str, Any]:
        """
        SISTEMA DE CLASIFICACIÓN CON MÚLTIPLES PROMPTS ESPECÍFICOS
        Cada prompt explica al LLM qué hace cada agente específicamente
        """
        if not self.llm_classifier:
            return {'intent': 'unknown', 'entities': {}, 'suggested_agent': 'greeting', 'justification': 'LLM no disponible'}

        # PROMPT 1: EXPLICACIÓN DE CADA AGENTE
        agent_explanation_prompt = f"""
Eres un clasificador experto de consultas médicas. Tu tarea es entender qué agente debe manejar cada consulta.

CONSULTA: "{query}"

EXPLICACIÓN DE CADA AGENTE:

1. **GREETING AGENT** - Para conversación general y saludos
   - Saludos: "hola", "buenos días", "gracias"
   - Preguntas sobre el sistema: "¿qué puedes hacer?", "ayuda"
   - Conversación casual: "¿cómo estás?", "ok", "vale"

2. **CLINICAL_ANALYSIS AGENT** - Para explicaciones médicas y medicamentos
   - Preguntas sobre medicamentos: "qué es la azitromicina", "para qué sirve el paracetamol"
   - Explicaciones médicas: "explica qué es la diabetes", "qué es la hipertensión"
   - Conceptos médicos: "qué significa anemia", "explica el concepto de diabetes"
   - Interacciones farmacológicas: "¿puedo tomar paracetamol con ibuprofeno?"

3. **BIOCHAT AGENT** - Para búsquedas de investigación científica
   - Estudios científicos: "estudios sobre diabetes", "investigaciones recientes"
   - Ensayos clínicos: "ensayos clínicos sobre cáncer", "últimos estudios"
   - Literatura médica: "papers sobre diabetes", "evidencia científica"
   - Investigación: "investigaciones sobre tratamientos", "artículos científicos"

4. **SQL AGENT** - Para consultas de base de datos de pacientes
   - Búsquedas de pacientes: "buscar Ana García", "datos de Juan"
   - Estadísticas: "cuántos pacientes", "mostrar pacientes"
   - Información de BD: "listar datos", "información de pacientes"

5. **FHIR AGENT** - Para registrar información médica
   - Notas clínicas: "procesar nota clínica", "registrar paciente"
   - Datos médicos: "crear paciente", "insertar diagnóstico"

Responde SOLO con el nombre del agente más apropiado: greeting|clinical_analysis|biochat|sql|fhir
"""

        try:
            # PRIMERA CLASIFICACIÓN
            response1 = await self.llm_classifier.ainvoke(agent_explanation_prompt)
            agent1 = self._extract_response_text(response1).strip().lower().replace('.', '').replace('"', '').replace("'", "")
            
            # Validar primera clasificación
            valid_agents = ["greeting", "clinical_analysis", "biochat", "sql", "fhir"]
            if agent1 not in valid_agents:
                agent1 = "greeting"
            
            logger.info(f"🧠 Primera clasificación: {agent1}")
            
            # PROMPT 2: CONFIRMACIÓN ESPECÍFICA PARA CONSULTAS AMBIGUAS
            if agent1 in ["clinical_analysis", "biochat"]:
                confirmation_prompt = f"""
CONSULTA: "{query}"
CLASIFICACIÓN PREVIA: {agent1}

Para confirmar la clasificación, analiza específicamente:

Si es CLINICAL_ANALYSIS:
- Preguntas sobre medicamentos específicos ("qué es la azitromicina")
- Explicaciones de conceptos médicos básicos ("qué es la diabetes")
- Información médica general para pacientes

Si es BIOCHAT:
- Búsquedas de estudios científicos ("estudios sobre diabetes")
- Investigaciones recientes ("últimas investigaciones")
- Literatura médica avanzada ("papers científicos")

¿La consulta busca información médica básica (clinical_analysis) o investigación científica (biochat)?

Responde SOLO: clinical_analysis|biochat
"""
                try:
                    response2 = await self.llm_classifier.ainvoke(confirmation_prompt)
                    agent2 = self._extract_response_text(response2).strip().lower().replace('.', '').replace('"', '').replace("'", "")
                    
                    if agent2 in ["clinical_analysis", "biochat"]:
                        final_agent = agent2
                        logger.info(f"🧠 Confirmación: {agent2}")
                    else:
                        final_agent = agent1
                        logger.info(f"🧠 Manteniendo clasificación original: {agent1}")
                except:
                    final_agent = agent1
            else:
                final_agent = agent1
            
            # PROMPT 3: DETECCIÓN ESPECIAL PARA NOTAS CLÍNICAS Y CASOS COMPLEJOS
            if len(query.split()) > 20:  # Textos largos pueden ser notas clínicas
                clinical_note_prompt = f"""
CONSULTA: "{query}"

Analiza si este texto es una NOTA CLÍNICA que debe procesarse con el agente FHIR.

Una NOTA CLÍNICA contiene:
- Información de paciente (nombre, edad, fecha)
- Diagnósticos médicos
- Tratamientos prescritos
- Signos vitales
- Observaciones médicas
- Formato médico estructurado

Ejemplos de NOTAS CLÍNICAS:
✅ "Paciente María González, 54 años. Fecha: 17/07/2025. Médico: Dr. Pérez. S: Dolor de garganta..."
✅ "Control de paciente diabético. HbA1c 7.2%. Continuar metformina..."
✅ "Sra. Ana García, 45 años. Diagnóstico: Hipertensión. Prescribir enalapril..."

¿Es una NOTA CLÍNICA para registrar (fhir) o una consulta normal?

Responde SOLO: fhir|normal
"""
                try:
                    response3 = await self.llm_classifier.ainvoke(clinical_note_prompt)
                    note_result = self._extract_response_text(response3).strip().lower()
                    
                    if "fhir" in note_result:
                        final_agent = "fhir"
                        logger.info(f"🧠 Detectado como nota clínica: FHIR")
                except:
                    pass  # Mantener clasificación anterior
            
            logger.info(f"🧠 Clasificación final: {final_agent}")
            
            return {
                'intent': f"Consulta clasificada como {final_agent}",
                'entities': {},
                'suggested_agent': final_agent,
                'justification': f"Clasificado por LLM como {final_agent}"
            }
                
        except Exception as e:
            logger.error(f"Error en clasificación LLM: {e}")
            return {'intent': 'error', 'entities': {}, 'suggested_agent': 'greeting', 'justification': str(e)}

    async def _score_and_select_agent(self, analysis: Dict[str, Any], original_query: str) -> Tuple[str, float, str]:
        """
        SISTEMA DE FALLBACK INTELIGENTE CON PROMPTS ESPECÍFICOS
        """
        suggested_agent = analysis.get('suggested_agent', 'greeting')
        justification = analysis.get('justification', 'Sin justificación.')
        
        logger.info(f"[DEBUG] Agente sugerido: {suggested_agent}, agentes disponibles: {list(self.agents.keys())}")
        
        # Verificar si el agente sugerido está disponible
        if suggested_agent in self.agents:
            return suggested_agent, 0.9, justification
        else:
            logger.warning(f"Agente '{suggested_agent}' no disponible. Agentes disponibles: {list(self.agents.keys())}")
            
            # FALLBACK INTELIGENTE CON PROMPT ESPECÍFICO
            if not self.llm_classifier:
                # Fallback simple sin LLM
                available_agents = [agent for agent in self.agents.keys() if agent != "greeting"]
                if available_agents:
                    fallback_agent = available_agents[0]
                    return fallback_agent, 0.5, f"Agente '{suggested_agent}' no disponible, usando {fallback_agent}"
                else:
                    return 'greeting', 0.3, "No hay agentes disponibles, usando greeting"
            
            # PROMPT DE FALLBACK INTELIGENTE
            fallback_prompt = f"""
CONSULTA ORIGINAL: "{original_query}"
AGENTE SUGERIDO (NO DISPONIBLE): {suggested_agent}
AGENTES DISPONIBLES: {list(self.agents.keys())}

Necesitas elegir el mejor agente alternativo disponible para manejar esta consulta.

EXPLICACIÓN DE AGENTES DISPONIBLES:

1. **GREETING**: Para saludos y conversación general
2. **CLINICAL_ANALYSIS**: Para explicaciones médicas y medicamentos
3. **BIOCHAT**: Para búsquedas de investigación científica
4. **SQL**: Para consultas de base de datos de pacientes
5. **FHIR**: Para registrar información médica

REGLAS DE FALLBACK:
- Si el agente sugerido era CLINICAL_ANALYSIS → usar BIOCHAT si está disponible
- Si el agente sugerido era BIOCHAT → usar CLINICAL_ANALYSIS si está disponible
- Si el agente sugerido era SQL → usar FHIR si está disponible
- Si el agente sugerido era FHIR → usar SQL si está disponible
- Si no hay alternativas específicas → usar el primer agente disponible que no sea GREETING

¿Cuál es el mejor agente alternativo para esta consulta?

Responde SOLO con el nombre del agente: greeting|clinical_analysis|biochat|sql|fhir
"""
            
            try:
                response = await self.llm_classifier.ainvoke(fallback_prompt)
                fallback_agent = self._extract_response_text(response).strip().lower().replace('.', '').replace('"', '').replace("'", "")
                
                # Validar que el agente de fallback esté disponible
                if fallback_agent in self.agents:
                    logger.info(f"🧠 Fallback inteligente: {suggested_agent} → {fallback_agent}")
                    return fallback_agent, 0.7, f"Agente '{suggested_agent}' no disponible, LLM sugirió {fallback_agent}"
                else:
                    # Último recurso: usar el primer agente disponible
                    available_agents = [agent for agent in self.agents.keys() if agent != "greeting"]
                    if available_agents:
                        final_fallback = available_agents[0]
                        return final_fallback, 0.5, f"Agente '{suggested_agent}' no disponible, usando {final_fallback}"
                    else:
                        return 'greeting', 0.3, "No hay agentes disponibles, usando greeting"
                        
            except Exception as e:
                logger.error(f"Error en fallback inteligente: {e}")
                # Fallback simple sin LLM
                available_agents = [agent for agent in self.agents.keys() if agent != "greeting"]
                if available_agents:
                    fallback_agent = available_agents[0]
                    return fallback_agent, 0.5, f"Agente '{suggested_agent}' no disponible, usando {fallback_agent}"
                else:
                    return 'greeting', 0.3, "No hay agentes disponibles, usando greeting"

    async def _smart_sql_vs_fhir_classification(self, query: str, intent: str, scores: Dict[str, float]):
        """
        ELIMINADA: Ya no se necesita esta función compleja
        """
        pass

    def _get_user_memory_context(self) -> Dict[str, Any]:
        """Recupera el contexto de la memoria del usuario actual."""
        mem = self.user_memories.get(self.current_user_id)
        if mem and mem.last_patient_id:
            return {'last_patient_id': mem.last_patient_id}
        return {}
    
    def _extract_response_text(self, response: Any) -> str:
        """Extrae texto de la respuesta del LLM (puede ser string o un objeto)."""
        if isinstance(response, str):
            return response
        if hasattr(response, 'content'):
            return str(response.content)
        return str(response)

# Función de utilidad para inicialización rápida
async def create_flexible_orchestrator(config_path: Optional[str] = None) -> "IntelligentOrchestratorV3":
    """
    Crea una instancia del orquestador inteligente V3 con configuración optimizada.
    """
    orchestrator = IntelligentOrchestratorV3(
        config_path=config_path,
        enable_cache=True,
        enable_performance_monitoring=True,
        cache_ttl_minutes=60
    )
    
    # Calentar el sistema con consultas comunes para asegurar que los modelos
    # y agentes estén listos.
    await orchestrator.process_query_optimized("Hola")
    await orchestrator.process_query_optimized("¿Cuántos pacientes hay?")
    
    return orchestrator


# Función principal para pruebas
async def main():
    """Función principal para pruebas del orquestador v2.0"""
    orchestrator = await create_flexible_orchestrator()
    
    # Pruebas de rendimiento
    test_queries = [
        "Hola, ¿cómo estás?",
        "¿Cuántos pacientes hay en la base de datos?",
        "Procesar nota: Paciente Juan Pérez, 45 años, hipertensión",
        "Buscar estudios sobre diabetes tipo 2",
        "Mostrar todos los procedimientos autorizados"
    ]
    
    print(f"{Colors.BOLD}🧪 PRUEBAS DE RENDIMIENTO{Colors.END}")
    
    for query in test_queries:
        start_time = time.time()
        result = await orchestrator.process_query_optimized(query)
        end_time = time.time()
        
        print(f"{Colors.GREEN}Consulta:{Colors.END} {query}")
        print(f"{Colors.BLUE}Tiempo:{Colors.END} {end_time - start_time:.3f}s")
        print(f"{Colors.CYAN}Resultado:{Colors.END} {str(result)[:100]}...")
        print("-" * 80)
    
    # Mostrar métricas
    metrics = orchestrator.get_performance_metrics()
    print(f"{Colors.BOLD}📊 MÉTRICAS FINALES{Colors.END}")
    for key, value in metrics.items():
        print(f"{Colors.YELLOW}{key}:{Colors.END} {value}")


if __name__ == "__main__":
    asyncio.run(main()) 