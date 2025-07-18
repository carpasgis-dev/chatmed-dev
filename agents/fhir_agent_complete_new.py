#!/usr/bin/env python3
"""
⚠️ ARCHIVO OBSOLETO - NO SE USA EN EL SISTEMA ⚠️
==================================================

Este archivo fue reemplazado por fhir_agent_complete.py (versión 4.0 unificada).
El orquestador usa fhir_agent_complete.py, no este archivo.

Para evitar confusión, este archivo debería ser eliminado o renombrado.

ESTADO: OBSOLETO - NO USAR
"""

import logging
import os
import uuid
import json
import sqlite3
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple, TYPE_CHECKING
from pathlib import Path
import threading
import asyncio
import concurrent.futures
import time
import sys
from fhir_persistence_agent_old import PersistenceResult

# LangChain imports
try:
    from langchain.llms.base import BaseLLM
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Import del sistema de mapeo empresarial
MAPPING_SYSTEM_AVAILABLE = False
try:
    # Intentar importar desde el sistema v1 (completo) con path absoluto
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    v1_mapping_dir = os.path.join(current_dir, '..', '..', 'chatmed_fhir_system', 'mapping')
    if os.path.exists(v1_mapping_dir):
        sys.path.insert(0, v1_mapping_dir)
        from fhir_sql_bridge import FHIRSQLBridge, ConversionDirection, ConversionResult  # type: ignore
        MAPPING_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: No se pudo importar FHIRSQLBridge: {e}")
    pass

# Import de componentes flexibles del sistema v2
FLEXIBLE_SYSTEM_AVAILABLE = False
try:
    from ..mapping.schema_introspector import SchemaIntrospector
    from ..mapping.flexible_engine import FlexibleEngine
    from ..mapping.dynamic_mapper import DynamicMapper
    FLEXIBLE_SYSTEM_AVAILABLE = True
    print("✅ Sistema flexible v2 importado correctamente")
except ImportError:
    try:
        # Fallback: import absoluto
        from chatmed_v2_flexible.mapping.schema_introspector import SchemaIntrospector
        from chatmed_v2_flexible.mapping.flexible_engine import FlexibleEngine  
        from chatmed_v2_flexible.mapping.dynamic_mapper import DynamicMapper
        FLEXIBLE_SYSTEM_AVAILABLE = True
        print("✅ Sistema flexible v2 importado (absoluto)")
    except ImportError as e:
        print(f"⚠️ Warning: No se pudo importar sistema flexible: {e}")

# Import del agente de persistencia FHIR
PERSISTENCE_AGENT_AVAILABLE = False

def get_persistence_agent(db_path: str, llm_client=None):
    """Obtiene una instancia del agente de persistencia FHIR"""
    try:
        # Importación dinámica
        import importlib
        persistence_module = importlib.import_module('chatmed_v2_flexible.agents.fhir_persistence_agent_old')
        PersistenceAgent = getattr(persistence_module, 'FHIRPersistenceAgent')
        agent = PersistenceAgent(db_path=db_path, llm_client=llm_client)
        global PERSISTENCE_AGENT_AVAILABLE
        PERSISTENCE_AGENT_AVAILABLE = True
        return agent
    except (ImportError, AttributeError) as e:
        print(f"❌ Error importando FHIRPersistenceAgent: {e}")
        
        # Clase de fallback cuando el agente no está disponible
        class FallbackPersistenceAgent:
            def __init__(self, db_path: str, llm_client=None):
                self.db_path = db_path
                self.llm_client = llm_client
                print("⚠️ Usando versión básica de FHIRPersistenceAgent")
                
            async def persist_fhir_resource(self, fhir_resource: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    'success': False,
                    'message': 'FHIRPersistenceAgent no está disponible',
                    'resource_type': fhir_resource.get('resourceType', 'Unknown'),
                    'resource_id': fhir_resource.get('id', 'unknown'),
                    'sql_queries': [],
                    'errors': ['Agente de persistencia no disponible']
                }
            
            def get_stats(self) -> Dict[str, Any]:
                return {
                    'total_resources_processed': 0,
                    'successful_persistences': 0,
                    'failed_persistences': 0
                }
        
        return FallbackPersistenceAgent(db_path=db_path, llm_client=llm_client)

# Configuración de logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"))
logger = logging.getLogger("FHIRAgent3.0")

class MockResponse:
    """Clase para compatibilidad con respuestas LLM"""
    def __init__(self, content: str):
        self.content = content

def _call_openai_native(client, messages, temperature=0.1, max_tokens=2000) -> MockResponse:
    """
    Función de compatibilidad para llamar a OpenAI nativo desde objetos LangChain
    ARREGLADO: Mejor manejo de respuestas JSON con logging detallado
    """
    try:
        print(f"   🚀 LLAMADA AL LLM:")
        print(f"   ├─ Temperatura: {temperature}")
        print(f"   ├─ Max tokens: {max_tokens}")
        
        # Crear cliente OpenAI nativo
        from openai import OpenAI
        native_client = OpenAI()
        
        # Convertir mensajes a formato OpenAI con tipos correctos
        if isinstance(messages, list):
            openai_messages = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    openai_messages.append({"role": "user", "content": str(msg.content)})
                elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                    openai_messages.append({"role": str(msg["role"]), "content": str(msg["content"])})
                else:
                    openai_messages.append({"role": "user", "content": str(msg)})
        else:
            content = messages.content if hasattr(messages, 'content') else str(messages)
            openai_messages = [{"role": "user", "content": str(content)}]
        
        # Mostrar información del prompt enviado
        total_chars = sum(len(msg["content"]) for msg in openai_messages)
        print(f"   ├─ Mensajes: {len(openai_messages)}")
        print(f"   ├─ Caracteres totales: {total_chars}")
        
        # Mostrar una vista previa del prompt (primeros 200 caracteres)
        if openai_messages and openai_messages[0]["content"]:
            preview = openai_messages[0]["content"][:200]
            print(f"   ├─ Vista previa prompt: {preview}...")
        
        print(f"   └─ Enviando a OpenAI GPT-4o...")
        
        if os.getenv("CHATMED_LLM_DEBUG") == "1":
            # --- STREAMING REAL ------------------------------------------------
            stream_buffer: List[str] = []
            print("   📡 Streaming de tokens activado (CHATMED_LLM_DEBUG=1)...")
            resp_stream = native_client.chat.completions.create(
                model="gpt-4o",
                messages=openai_messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            # Progreso rudimentario
            import itertools, sys
            spinner = itertools.cycle(["|", "/", "-", "\\"])
            for chunk in resp_stream:  # type: ignore
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    token = delta.content
                    stream_buffer.append(token)
                    # Imprimir token sin saltos de línea grandes (solo para debug)
                    sys.stdout.write(token)
                    sys.stdout.flush()
                else:
                    # animación spinner cada 0.25 s aprox
                    sys.stdout.write(next(spinner))
                    sys.stdout.flush()
                    sys.stdout.write("\b")
            sys.stdout.write("\n")
            content = "".join(stream_buffer)
        else:
            response = native_client.chat.completions.create(
                model='gpt-4o',
                messages=openai_messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens
            )
            # ARREGLO: Verificar que la respuesta no esté vacía
            content = response.choices[0].message.content or ""
        
        if not content.strip():
            content = "Error: Respuesta vacía del LLM"
            print(f"   ❌ RESPUESTA VACÍA DEL LLM")
        else:
            print(f"   📥 RESPUESTA RECIBIDA:")
            print(f"   ├─ Longitud: {len(content)} caracteres")
            
            # Solo mostrar tokens si no estamos en modo streaming (ya se mostraron)
            if os.getenv("CHATMED_LLM_DEBUG") != "1":
                print(f"   ├─ Tokens usados: {response.usage.total_tokens if response.usage else 'N/A'}")
            else:
                print(f"   ├─ Tokens usados: N/A (modo streaming)")
            
            # Mostrar vista previa de la respuesta
            response_preview = content[:300] + "..." if len(content) > 300 else content
            print(f"   ├─ Vista previa: {response_preview}")
            
            # Detectar si es JSON
            content_stripped = content.strip()
            if content_stripped.startswith('{') and content_stripped.endswith('}'):
                print(f"   └─ Formato detectado: JSON ✅")
            elif '```json' in content_stripped:
                print(f"   └─ Formato detectado: JSON en código ✅")
            else:
                print(f"   └─ Formato detectado: Texto plano ⚠️")
        
        return MockResponse(content)
            
    except Exception as e:
        error_msg = f"Error en llamada OpenAI: {str(e)}"
        print(f"   ❌ ERROR EN LLM: {error_msg}")
        logger.error(f"Error en _call_openai_native: {e}")
        return MockResponse(f"Error: {error_msg}")

if TYPE_CHECKING:
    from .sql_agent_clean import SQLAgentRobust

class TerminalColors:
    HEADER = '\033[96m\033[1m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    GREY = '\033[90m'
    CYAN = '\033[36m'
    WHITE = '\033[97m'


def format_terminal_response(summary: dict) -> str:
    """Formatea la respuesta final de la operación clínica en bloques visuales con emojis y colores."""
    c = TerminalColors
    lines = []
    # Encabezado
    lines.append(f"{c.HEADER}=============================================={c.ENDC}")
    lines.append(f"{c.BOLD}🏥 ChatMed - Resumen de Operación Clínica{c.ENDC}")
    lines.append(f"{c.HEADER}=============================================={c.ENDC}")
    # Paciente
    if 'patient_result' in summary:
        p = summary['patient_result']
        # MEJORADO: Mostrar tanto UUID como RowID para referencia completa
        patient_id_display = p.get('patient_id', 'N/A')
        patient_rowid = p.get('patient_rowid', '')
        
        # Si tenemos ambos IDs, mostrar información más completa
        if patient_rowid and patient_id_display != patient_rowid:
            id_info = f"UUID: {patient_id_display}, RowID: {patient_rowid}"
        else:
            id_info = f"ID: {patient_id_display}"
            
        lines.append(f"{c.OKGREEN}👤 Paciente: {p.get('patient_name','N/A')} ({id_info}){c.ENDC}")
        lines.append(f"   Edad: {p.get('patient_age','N/A')} | Acción: {p.get('action','N/A')}")
        
        # NUEVO: Agregar contexto para futuras consultas
        if p.get('action') == 'created' and patient_rowid:
            lines.append(f"   💡 Para consultas futuras puedes usar: \"paciente {patient_rowid}\" o \"paciente con ID {patient_id_display}\"")
    # Diagnósticos
    if 'created' in summary and summary['created']:
        dx = [x for x in summary['created'] if 'condition' in x]
        if dx:
            lines.append(f"{c.OKBLUE}🩺 Diagnósticos guardados: {len(dx)}{c.ENDC}")
    # Medicamentos
    if 'created' in summary and summary['created']:
        meds = [x for x in summary['created'] if 'medication' in x]
        if meds:
            lines.append(f"{c.OKBLUE}💊 Medicamentos guardados: {len(meds)}{c.ENDC}")
    # Observaciones
    if 'created' in summary and summary['created']:
        obs = [x for x in summary['created'] if 'observation' in x]
        if obs:
            lines.append(f"{c.CYAN}🔬 Observaciones guardadas: {len(obs)}{c.ENDC}")
    # Bundle FHIR
    if 'fhir_bundle' in summary:
        lines.append(f"{c.GREY}📦 Bundle FHIR generado: {summary['fhir_bundle'].get('id','N/A')[:8]}...{c.ENDC}")
    # Errores
    if 'errors' in summary and summary['errors']:
        lines.append(f"{c.FAIL}⚠️ Errores durante la operación:{c.ENDC}")
        for err in summary['errors']:
            lines.append(f"   {c.FAIL}- {err}{c.ENDC}")
    # Éxito
    if summary.get('success'):
        lines.append(f"{c.OKGREEN}✅ Operación completada exitosamente{c.ENDC}")
    else:
        lines.append(f"{c.FAIL}❌ Operación incompleta o con errores{c.ENDC}")
    lines.append(f"{c.HEADER}=============================================={c.ENDC}")
    return '\n'.join(lines)

class FHIRMedicalAgent:
    """
    🏥 Agente FHIR 3.0 - Procesamiento Empresarial Completo
    
    Características principales:
    - Procesamiento de notas clínicas con IA
    - Conversión automática SQL↔FHIR usando mapeo empresarial
    - Soporte para 236 tablas (50 detalladas + 186 genéricas) 
    - Validación FHIR automática
    - Gestión inteligente de recursos relacionados
    - ARREGLADO: Thread-safe SQLite connections
    """
    
    def __init__(self, 
                 db_path: str = "../database_new.sqlite3.db",  # ARREGLO: Ruta correcta a la BD
                 llm = None,
                 mapping_config: Optional[str] = None,
                 sql_agent: Optional['SQLAgentRobust'] = None):
        """Inicializa el agente FHIR con configuración avanzada."""
        
        # Configurar la base de datos primero
        self.db_path = db_path
        if not os.path.exists(db_path):
            logger.warning(f"⚠️ Base de datos no encontrada en: {db_path}")
            # Buscar en ubicaciones alternativas
            alt_paths = [
                "../database.sqlite3.db",
                "database.sqlite3.db",
                "database_new.sqlite3.db"
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    self.db_path = alt_path
                    logger.info(f"✅ Base de datos encontrada en ubicación alternativa: {alt_path}")
                    break
        
        # Inicializar estadísticas
        self.stats = {
            'clinical_notes_processed': 0,
            'fhir_resources_created': 0,
            'sql_records_updated': 0,
            'validation_errors': 0,
            'successful_conversions': 0
        }

        # Configuración del agente
        self.llm = llm
        self.sql_agent = sql_agent
        self._local = threading.local()
        
        # Intentar cargar el sistema de mapeo empresarial
        self.bridge = None
        if MAPPING_SYSTEM_AVAILABLE:
            try:
                config_path = mapping_config or self._find_mapping_config()
                self.bridge = FHIRSQLBridge(config_path)
                logger.info(f"✅ Sistema de mapeo FHIR cargado desde: {config_path}")
            except Exception as e:
                logger.error(f"❌ Error cargando sistema de mapeo FHIR: {e}")
        
        # Intentar cargar el sistema flexible v2
        self.flexible_engine = None
        if FLEXIBLE_SYSTEM_AVAILABLE:
            try:
                self.flexible_engine = FlexibleEngine(db_path=self.db_path)
                logger.info("✅ Sistema flexible v2 inicializado")
            except Exception as e:
                logger.error(f"❌ Error inicializando sistema flexible: {e}")
        
        # Intentar cargar el agente de persistencia
        self.persistence_agent = get_persistence_agent(self.db_path, llm)
        
        logger.info(f"""
🏥 FHIRAgent 3.0 inicializado:
├── Base de datos: {self.db_path}
├── Sistema de mapeo: {'✅' if self.bridge else '❌'}
├── Sistema flexible: {'✅' if self.flexible_engine else '❌'}
└── Persistencia FHIR: {'✅' if PERSISTENCE_AGENT_AVAILABLE else '❌'}
""")
        
        # Inicializar FHIRPersistenceAgent para mapeo inteligente
        try:
            from fhir_persistence_agent_old import FHIRPersistenceAgent
            self.persistence_agent = FHIRPersistenceAgent(db_path=db_path, llm_client=llm)
            logger.info("✅ FHIRPersistenceAgent inicializado")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo inicializar FHIRPersistenceAgent: {e}")
            self.persistence_agent = None
        
        # Cargar configuración de mapeo
        self.mappings = {}  # Por ahora vacío, se puede expandir después
        
        logger.info("✅ FHIR Medical Agent inicializado correctamente")
        
        if self.sql_agent:
            logger.info("✅ (DEBUG) FHIRAgent inicializado CON SQLAgent.")
        else:
            logger.warning("⚠️ (DEBUG) FHIRAgent inicializado SIN SQLAgent.")
        
        # Inicializar sistema flexible v2 si está disponible
        self.introspector = None
        self.dynamic_mapper = None
        
        if FLEXIBLE_SYSTEM_AVAILABLE:
            try:
                # Inicializar introspector de esquemas
                self.introspector = SchemaIntrospector(db_path, cache_ttl=3600)
                logger.info("✅ SchemaIntrospector inicializado")
                
                # Inicializar motor flexible
                config_dir = mapping_config or self._find_mapping_config()
                self.flexible_engine = FlexibleEngine(
                    db_path=db_path,
                    config_dir=config_dir,
                    enable_cache=True,
                    enable_validation=True
                )
                logger.info("✅ FlexibleEngine v2 inicializado")
                
                # Inicializar mapper dinámico si está disponible
                try:
                    self.dynamic_mapper = DynamicMapper(db_path)
                    logger.info("✅ DynamicMapper inicializado")
                except:
                    logger.info("⚠️ DynamicMapper no disponible")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error inicializando sistema flexible: {e}")
        
        # Inicializar sistema de mapeo empresarial V2 (FLEXIBLE) - Fallback
        if MAPPING_SYSTEM_AVAILABLE and not self.flexible_engine:
            mapping_dir = mapping_config or self._find_mapping_config()
            try:
                # Usar el bridge ya importado globalmente
                self.bridge = FHIRSQLBridge(
                    db_path=db_path,
                    llm=llm,
                    enable_cache=True,
                    validate_fhir=True,
                    mapping_dir=mapping_dir
                )
                logger.info("�� Sistema de mapeo V2 FLEXIBLE inicializado (fallback)")
                
                # Log de información del sistema V2
                stats = self.bridge.get_conversion_stats()
                system_mode = stats.get('system_mode', 'unknown')
                bridge_version = stats.get('bridge_version', 'unknown')
                logger.info(f"   📊 Versión Bridge: {bridge_version}")
                logger.info(f"   🎯 Modo de sistema: {system_mode}")
                logger.info(f"   🔧 Sistema flexible: {'✅ Activo' if system_mode == 'flexible' else '❌ Fallback'}")
                
            except Exception as e:
                logger.warning(f"⚠️ Bridge V2 usando fallback: {e}")  # Cambiar a warning, no es error crítico
                self.bridge = None
        else:
            self.bridge = None
            if not self.flexible_engine:
                logger.info("⚠️ Sistema de mapeo no disponible - continuando sin bridge")  # Info en lugar de error
        
        # Inicializar agente de persistencia FHIR
        if PERSISTENCE_AGENT_AVAILABLE:
            try:
                self.persistence_agent = FHIRPersistenceAgent(db_path, llm)
                logger.info("🏥 Agente de persistencia FHIR inicializado")
            except Exception as e:
                logger.warning(f"⚠️ Error inicializando agente de persistencia: {e}")
                self.persistence_agent = None
        else:
            logger.warning("⚠️ Agente de persistencia FHIR no disponible")
            self.persistence_agent = None
        
        # Estado del agente
        self.conversation_state = {}
        self.pending_operations = []
        
        logger.info("🚀 FHIRAgent 3.0 inicializado con Sistema Completamente Flexible")
        logger.info("   ✅ Sin hardcodeo - 100% dinámico con LLM")
        logger.info("   ⚡ Auto-introspección de esquemas en tiempo real")
        logger.info("   🎯 Compatibilidad API 100% mantenida")

    def _get_db_connection(self) -> Optional[sqlite3.Connection]:
        """
        ARREGLADO: Obtiene conexión thread-safe a la base de datos
        """
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            try:
                self._local.conn = sqlite3.connect(
                    self.db_path, 
                    check_same_thread=False,  # Permitir uso en múltiples threads
                    timeout=30.0  # Timeout de 30 segundos
                )
                self._local.conn.row_factory = sqlite3.Row  # Para acceso por nombre de columna
                logger.debug(f"✅ Nueva conexión thread-safe creada para thread {threading.current_thread().ident}")
            except Exception as e:
                logger.error(f"❌ Error creando conexión thread-safe: {e}")
                self._local.conn = None
        
        return self._local.conn

    def _find_mapping_config(self) -> str:
        """Busca la configuración de mapeo en ubicaciones estándar"""
        possible_paths = [
            "chatmed_fhir_system/config",  # Ruta principal donde está el archivo
            "config",
            os.path.join(os.path.dirname(__file__), "..", "..", "chatmed_fhir_system", "config"),  # Ruta desde v2 a v1
            os.path.join(os.path.dirname(__file__), "..", "config"),
            os.path.join(os.path.dirname(__file__), "..", "mapping"),
            "old/legacy_system/new/mapping"
        ]
        
        for path in possible_paths:
            if os.path.exists(os.path.join(path, "mapping_consolidado.json")):
                return path
        
        logger.warning("⚠️ mapping_consolidado.json no encontrado, usando defaults")
        return "config"

    def _generate_user_response(self, result: Dict[str, Any]) -> str:
        """
        Genera respuesta formateada para el usuario basada en el resultado del procesamiento
        """
        if not result.get('success', False):
            return f"❌ **Error en procesamiento FHIR:** {result.get('error', 'Error desconocido')}"
        
        response_type = result.get('type', 'unknown')
        
        if response_type == 'clinical_note_processing' and result.get('success'):
             # Para notas clínicas, el resumen ya está bien formateado en 'summary'
            return result.get('summary', 'Nota clínica procesada exitosamente.')
        elif response_type == 'clinical_note_processing' and not result.get('success'):
            return f"❌ **Error procesando nota clínica:** {result.get('error', 'Error desconocido')}"
        elif response_type == 'fhir_query':
            return self._format_fhir_query_response(result)
        elif response_type == 'conversion':
            return self._format_conversion_response(result)
        else:
            return self._format_general_response(result)
    
    def _format_clinical_note_response(self, result: Dict[str, Any]) -> str:
        """Formatea respuesta para procesamiento de nota clínica. Esta función es ahora un fallback."""
        response = "🏥 PROCESAMIENTO DE NOTA CLÍNICA FHIR\n\n"
        
        # Si hubo error, mostrar mensaje apropiado
        if not result.get('success', False):
            response += f"❌ Error: {result.get('error', 'Error desconocido')}\n"
            return response
        
        # Información del paciente
        extracted_data = result.get('extracted_data', {})
        if not extracted_data:
            response += "⚠️ No se pudieron extraer datos de la nota clínica\n"
            return response
            
        patient_data = extracted_data.get('patient', {})
        
        if patient_data:
            patient_name = f"{patient_data.get('name', 'N/A')} {patient_data.get('surname', '')}".strip()
            response += f"👤 Paciente: {patient_name}\n"
            
            if patient_data.get('age'):
                response += f"📅 Edad: {patient_data['age']} años\n"
            
            if patient_data.get('gender'):
                gender_map = {'male': 'Masculino', 'female': 'Femenino', 'unknown': 'No especificado'}
                response += f"⚧ Género: {gender_map.get(patient_data['gender'], patient_data['gender'])}\n"
        
        response += "\n"
        
        # Recursos procesados
        conditions = extracted_data.get('conditions', [])
        medications = extracted_data.get('medications', [])
        observations = extracted_data.get('observations', [])
        
        if conditions:
            response += f"🏥 Diagnósticos/Condiciones ({len(conditions)}):\n"
            for i, condition in enumerate(conditions[:3], 1):  # Mostrar máximo 3
                response += f"  {i}. {condition.get('description', 'Sin descripción')}\n"
            if len(conditions) > 3:
                response += f"  ... y {len(conditions) - 3} más\n"
            response += "\n"
        
        if medications:
            response += f"💊 Medicamentos ({len(medications)}):\n"
            for i, med in enumerate(medications[:3], 1):  # Mostrar máximo 3
                med_text = med.get('name', 'Sin nombre')
                if med.get('dosage'):
                    med_text += f" - {med['dosage']}"
                response += f"  {i}. {med_text}\n"
            if len(medications) > 3:
                response += f"  ... y {len(medications) - 3} más\n"
            response += "\n"
        
        if observations:
            response += f"🔬 Observaciones ({len(observations)}):\n"
            for i, obs in enumerate(observations[:3], 1):  # Mostrar máximo 3
                obs_text = obs.get('type', 'Observación')
                if obs.get('value'):
                    obs_text += f": {obs['value']}"
                    if obs.get('unit'):
                        obs_text += f" {obs['unit']}"
                response += f"  {i}. {obs_text}\n"
            if len(observations) > 3:
                response += f"  ... y {len(observations) - 3} más\n"
            response += "\n"
        
        # Recursos FHIR generados
        additional_resources = result.get('additional_resources', 0)
        total_resources = 1 + additional_resources if result.get('patient_result') else additional_resources
        
        response += f"📋 Recursos FHIR generados: {total_resources}\n"
        
        # Bundle FHIR
        if result.get('fhir_bundle'):
            bundle_id = result['fhir_bundle'].get('id', 'N/A')
            response += f"📦 Bundle FHIR: {bundle_id[:8]}...\n"
        
        # Estadísticas
        stats = result.get('stats', {})
        if stats:
            response += f"\n📊 Estadísticas de sesión:\n"
            response += f"  • Notas procesadas: {stats.get('clinical_notes_processed', 0)}\n"
            response += f"  • Recursos FHIR creados: {stats.get('fhir_resources_created', 0)}\n"
            response += f"  • Conversiones exitosas: {stats.get('successful_conversions', 0)}\n"
        
        # Resumen generado
        if result.get('summary'):
            response += f"\n{result['summary']}"
        
        response += f"\n✅ Procesamiento completado exitosamente"
        
        return response
    
    def _format_fhir_query_response(self, result: Dict[str, Any]) -> str:
        """Formatea respuesta para consultas FHIR"""
        return f"📋 Consulta FHIR procesada\n\n{result.get('message', 'Consulta completada')}"
    
    def _format_conversion_response(self, result: Dict[str, Any]) -> str:
        """Formatea respuesta para conversiones SQL↔FHIR"""
        return f"🔄 Conversión SQL↔FHIR\n\n{result.get('message', 'Conversión completada')}"
    
    def _format_general_response(self, result: Dict[str, Any]) -> str:
        """Formatea respuesta general"""
        return f"🏥 Operación FHIR\n\n{result.get('message', 'Operación completada')}"

    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Punto de entrada principal para el orquestador
        ARREGLADO: Ahora incluye response_text formateada
        
        Args:
            query: Consulta del usuario (nota clínica o comando FHIR)
            
        Returns:
            Dict con resultado del procesamiento y respuesta formateada
        """
        logger.info(f"🔄 FHIRAgent procesando: '{query[:100]}...'")
        
        # Determinar tipo de operación
        operation_type = self._classify_fhir_operation(query)
        
        if operation_type == "clinical_note":
            result = await self.process_clinical_note(query)
        elif operation_type == "fhir_query":
            result = self._process_fhir_query(query)
        elif operation_type == "conversion_request":
            result = self._process_conversion_request(query)
        else:
            result = self._process_general_fhir_request(query)
        
        # ARREGLO: Generar respuesta formateada para el usuario
        response_text = self._generate_user_response(result)
        
        # Agregar response_text al resultado
        result['response_text'] = response_text
        
        return result

    def _classify_fhir_operation(self, query: str) -> str:
        """Clasifica el tipo de operación FHIR usando IA y patrones"""
        
        # Primero, intentar clasificación con patrones mejorados
        query_lower = query.lower()
        
        # Patrones para notas clínicas (más específicos)
        clinical_patterns = [
            r"paciente\s+\w+.*\d+\s*años",  # "paciente X, Y años"
            r"paciente\s+\w+.*años",        # "paciente X años"
            r"control\s+ambulatorio",        # "control ambulatorio"
            r"diagnóstico.*:",               # "diagnóstico:"
            r"tratamiento\s+con",            # "tratamiento con"
            r"prescribir|prescribe",         # prescripciones
            r"mg/dl|mmhg|°c|kg",            # unidades médicas
            r"glicemia|presión\s+arterial|temperatura",  # parámetros vitales
            r"evolución\s+favorable",        # evolución
            r"alta\s+hospitalaria",          # alta médica
            r"cuadro\s+febril",             # síntomas
            r"examen\s+físico",             # exploración
        ]
        
        # Verificar patrones de nota clínica
        clinical_matches = 0
        for pattern in clinical_patterns:
            if re.search(pattern, query_lower):
                clinical_matches += 1
        
        # Si tiene múltiples patrones clínicos, es nota clínica
        if clinical_matches >= 2 or len(query) > 200: # Asumir que textos largos son notas
            return "clinical_note"
        
        # Patrones para consultas FHIR específicas
        if any(keyword in query_lower for keyword in ["fhir", "resource", "bundle", "validar recurso", "formato fhir"]):
            return "fhir_query"
        
        # Patrones para conversiones
        if any(keyword in query_lower for keyword in ["convertir", "conversión", "sql", "tabla", "base de datos"]):
            return "conversion_request"
        
        # Si no se puede clasificar con patrones, usar LLM si está disponible
        if self.llm and LANGCHAIN_AVAILABLE:
            try:
                prompt = f"""
                Clasifica esta consulta médica en una de estas categorías:
                - clinical_note: Para notas clínicas que describen pacientes, síntomas, diagnósticos, tratamientos
                - fhir_query: Para consultas específicas sobre recursos FHIR, validación, formatos
                - conversion_request: Para solicitudes de conversión SQL↔FHIR
                - general_fhir: Para otras consultas relacionadas con FHIR
                
                Consulta: "{query}"
                
                Responde SOLO con la categoría (clinical_note, fhir_query, conversion_request, o general_fhir).
                """
                
                response = _call_openai_native(self.llm, [HumanMessage(content=prompt)])
                classification = response.content.strip().lower()
                
                valid_types = ["clinical_note", "fhir_query", "conversion_request", "general_fhir"]
                if classification in valid_types:
                    return classification
                    
            except Exception as e:
                logger.error(f"Error clasificando operación FHIR: {e}")
        
        # Fallback: si menciona paciente, probablemente es nota clínica
        if "paciente" in query_lower:
            return "clinical_note"
        
        return "general_fhir"

    async def process_clinical_note(self, note: str) -> Dict[str, Any]:
        """Procesa una nota clínica y extrae información estructurada"""
        try:
            # Extraer datos de la nota
            extracted_data = await self._extract_clinical_data(note)
            if not extracted_data:
                return self._create_error_response("No se pudo extraer información de la nota")

            # Guardar en la base de datos
            save_result = await self._save_to_database(extracted_data)
            if not save_result.get('success'):
                return self._create_error_response(save_result.get('error', 'Error al guardar datos'))

            # Generar resumen
            patient_id = save_result.get('patient_id')
            if patient_id:
                bundle = await self._generate_fhir_bundle(patient_id, extracted_data)
                if bundle:
                    save_result['fhir_bundle'] = bundle

            # Actualizar estadísticas
            self.stats['clinical_notes_processed'] += 1
            if save_result.get('success'):
                self.stats['successful_conversions'] += 1

            return {
                'success': True,
                'message': 'Nota clínica procesada correctamente',
                'type': 'clinical_note',
                'data': save_result,
                'stats': self.stats
            }

        except Exception as e:
            logger.error(f"Error procesando nota clínica: {e}")
            self.stats['validation_errors'] += 1
            return self._create_error_response(f"Error procesando nota: {str(e)}")

    def _process_fhir_query(self, query: str) -> Dict[str, Any]:
        """Procesa consultas sobre recursos FHIR existentes"""
        logger.info("📋 Procesando consulta FHIR específica")
        
        # Intentar buscar recursos FHIR relacionados
        try:
            if self.bridge:
                # Usar estadísticas del bridge
                bridge_stats = self.bridge.get_conversion_stats()
                total_tables = bridge_stats.get('total_tables_mapped', 0)
                
                message = f"Consulta FHIR procesada.\n• Sistema de mapeo: Operativo\n• Tablas disponibles: {total_tables}\n• Validación FHIR: Habilitada"
            else:
                message = "Consulta FHIR procesada usando sistema básico."
            
            self.stats['successful_conversions'] += 1
            
            return {
                'success': True,
                'message': message,
                'type': 'fhir_query',
                'query': query,
                'stats': self.stats.copy()
            }
            
        except Exception as e:
            logger.error(f"Error procesando consulta FHIR: {e}")
            return self._create_error_response(f"Error en consulta FHIR: {str(e)}")

    def _process_conversion_request(self, query: str) -> Dict[str, Any]:
        """Procesa solicitudes de conversión SQL↔FHIR"""
        logger.info("🔄 Procesando solicitud de conversión SQL↔FHIR")
        
        try:
            conversion_info = []
            
            # Detectar si es conversión SQL→FHIR o FHIR→SQL
            if any(keyword in query.lower() for keyword in ['sql', 'tabla', 'base de datos', 'bd']):
                conversion_type = "SQL → FHIR"
                conversion_info.append("• Tipo: Conversión de datos SQL a recursos FHIR")
                
                if self.bridge:
                    # Usar estadísticas del bridge
                    bridge_stats = self.bridge.get_conversion_stats()
                    detailed_mappings = bridge_stats.get('detailed_mappings', 0)
                    conversion_info.append(f"• Mapeos detallados: {detailed_mappings} tablas")
                
            else:
                conversion_type = "FHIR → SQL"
                conversion_info.append("• Tipo: Conversión de recursos FHIR a formato SQL")
            
            conversion_info.append(f"• Sistema de mapeo: {'Disponible' if self.bridge else 'No disponible'}")
            conversion_info.append(f"• Validación FHIR: {'Habilitada' if self.bridge else 'Deshabilitada'}")
            
            message = f"Conversión {conversion_type} preparada:\n" + "\n".join(conversion_info)
            
            self.stats['successful_conversions'] += 1
            
            return {
                'success': True,
                'message': message,
                'type': 'conversion',
                'conversion_type': conversion_type,
                'query': query,
                'stats': self.stats.copy()
            }
            
        except Exception as e:
            logger.error(f"Error en conversión: {e}")
            return self._create_error_response(f"Error en conversión: {str(e)}")

    def _process_general_fhir_request(self, query: str) -> Dict[str, Any]:
        """Procesa otras operaciones FHIR generales"""
        logger.info("🏥 Procesando operación FHIR general")
        
        try:
            operation_info = []
            
            # Detectar el tipo de operación general
            query_lower = query.lower()
            
            if any(keyword in query_lower for keyword in ['validar', 'validación', 'verificar']):
                operation_type = "Validación FHIR"
                operation_info.append("• Operación: Validación de recursos FHIR")
                operation_info.append(f"• Validador: {'Disponible' if self.bridge else 'No disponible'}")
                
            elif any(keyword in query_lower for keyword in ['crear', 'generar', 'nuevo']):
                operation_type = "Creación de recursos"
                operation_info.append("• Operación: Creación de nuevos recursos FHIR")
                operation_info.append("• Recursos soportados: Patient, Condition, MedicationRequest, Observation")
                
            elif any(keyword in query_lower for keyword in ['buscar', 'encontrar', 'consultar']):
                operation_type = "Búsqueda FHIR"
                operation_info.append("• Operación: Búsqueda en recursos FHIR")
                if self.bridge:
                    bridge_stats = self.bridge.get_conversion_stats()
                    total_tables = bridge_stats.get('total_tables_mapped', 0)
                    operation_info.append(f"• Índice: {total_tables} tablas indexadas")
                
            else:
                operation_type = "Operación general"
                operation_info.append("• Operación: Procesamiento FHIR general")
            
            operation_info.append(f"• Sistema de mapeo: {'Operativo' if self.bridge else 'No disponible'}")
            operation_info.append(f"• Base de datos: {'Conectada' if self._get_db_connection() else 'No disponible'}")
            
            message = f"{operation_type} procesada:\n" + "\n".join(operation_info)
            
            self.stats['successful_conversions'] += 1
            
            return {
                'success': True,
                'message': message,
                'type': 'general_fhir',
                'operation_type': operation_type,
                'query': query,
                'stats': self.stats.copy()
            }
            
        except Exception as e:
            logger.error(f"Error en operación general: {e}")
            return self._create_error_response(f"Error en operación FHIR: {str(e)}")

    def can_handle(self, query: str) -> bool:
        """Determina si este agente puede manejar la consulta"""
        fhir_keywords = [
            "procesar", "nota", "diagnóstico", "paciente con",
            "fhir", "recurso", "bundle", "convertir",
            "historial clínico", "registrar paciente"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in fhir_keywords)

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del agente"""
        bridge_stats = self.bridge.get_conversion_stats() if self.bridge else {}
        
        return {
            **self.stats,
            'bridge_stats': bridge_stats,
            'mapping_system_available': MAPPING_SYSTEM_AVAILABLE,
            'database_connected': self._get_db_connection() is not None,
            'langchain_available': LANGCHAIN_AVAILABLE
        }

    def reset_stats(self):
        """Reinicia las estadísticas"""
        self.stats = {
            'clinical_notes_processed': 0,
            'fhir_resources_created': 0,
            'sql_records_updated': 0,
            'validation_errors': 0,
            'successful_conversions': 0
        }
        
        if self.bridge:
            self.bridge.clear_cache()

    def __del__(self):
        """Cleanup al destruir el agente"""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()

    def _try_parse_llm_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Analiza de forma segura el JSON de la salida del LLM con recuperación automática de errores."""
        try:
            # Estrategia 1: Buscar un bloque de código JSON explícito (más fiable)
            json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Estrategia 2: Si no hay bloque, buscar el primer '{' y el último '}'
                start_index = content.find('{')
                end_index = content.rfind('}')
                if start_index != -1 and end_index > start_index:
                    json_str = content[start_index : end_index + 1]
                else:
                    # Si todo falla, registrar el error y devolver None
                    logger.error(f"❌ No se pudo encontrar un objeto JSON válido en la respuesta: {content[:200]}...")
                    return None
            
            if not json_str or not json_str.strip():
                logger.warning("⚠️ Respuesta LLM vacía")
                return None
                
            json_str = json_str.strip()
            
            # Intentar parsing directo primero
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ JSON incompleto detectado, intentando auto-reparación...")
                
                # Auto-reparación para JSON incompleto
                repaired_json = self._repair_incomplete_json(json_str)
                if repaired_json:
                    try:
                        result = json.loads(repaired_json)
                        logger.info("✅ JSON reparado exitosamente")
                        return result
                    except json.JSONDecodeError:
                        logger.warning("⚠️ Reparación automática falló")
                
                # Si la reparación falla, intentar extraer datos parciales
                partial_data = self._extract_partial_json_data(json_str)
                if partial_data:
                    logger.info("✅ Datos parciales extraídos exitosamente")
                    return partial_data
                
                logger.error(f"❌ Error decodificando JSON de respuesta LLM: {content[:200]}...")
                return None

        except Exception as e:
            logger.error(f"❌ Error crítico en parsing JSON: {e}")
            return None

    def _repair_incomplete_json(self, json_str: str) -> Optional[str]:
        """Intenta reparar JSON incompleto añadiendo llaves/corchetes faltantes"""
        try:
            # Contar llaves y corchetes
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            open_brackets = json_str.count('[')
            close_brackets = json_str.count(']')
            
            # Añadir llaves faltantes
            missing_braces = open_braces - close_braces
            missing_brackets = open_brackets - close_brackets
            
            repaired = json_str
            
            # Cerrar strings incompletos si es necesario
            if repaired.count('"') % 2 != 0:
                repaired += '"'
            
            # Añadir comas si la última línea no termina correctamente
            repaired = repaired.rstrip()
            if repaired and not repaired.endswith((',', '{', '[', '}', ']')):
                repaired += '"'  # Cerrar string si está abierto
            
            # Cerrar estructuras abiertas
            repaired += '}' * missing_braces
            repaired += ']' * missing_brackets
            
            return repaired
            
        except Exception:
            return None

    def _extract_partial_json_data(self, json_str: str) -> Optional[Dict[str, Any]]:
        """Extrae datos utilizables de JSON parcialmente válido"""
        try:
            result = {}
            
            # Buscar patrones de datos comunes
            patterns = {
                'patient': r'"patient"\s*:\s*{([^}]*)}',
                'name': r'"name"\s*:\s*"([^"]*)"',
                'surname': r'"surname"\s*:\s*"([^"]*)"',
                'age': r'"age"\s*:\s*"?(\d+)"?',
                'gender': r'"gender"\s*:\s*"([^"]*)"',
                'conditions': r'"conditions"\s*:\s*\[([^\]]*)\]',
                'description': r'"description"\s*:\s*"([^"]*)"'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, json_str, re.IGNORECASE)
                if match:
                    if key in ['age']:
                        try:
                            result[key] = int(match.group(1))
                        except ValueError:
                            result[key] = match.group(1)
                    elif key == 'patient':
                        # Extraer datos del paciente
                        patient_data = {}
                        patient_content = match.group(1)
                        for sub_key, sub_pattern in patterns.items():
                            if sub_key != 'patient':
                                sub_match = re.search(sub_pattern, patient_content)
                                if sub_match:
                                    patient_data[sub_key] = sub_match.group(1)
                        if patient_data:
                            result[key] = patient_data
                    elif key == 'conditions':
                        # Extraer condiciones básicas
                        conditions_content = match.group(1)
                        desc_match = re.search(r'"description"\s*:\s*"([^"]*)"', conditions_content)
                        if desc_match:
                            result[key] = [{'description': desc_match.group(1)}]
                    else:
                        result[key] = match.group(1)
            
            return result if result else None
            
        except Exception:
            return None

    async def process_patient_data(self, patient_data: Dict[str, Any], conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        """Procesa datos de paciente con manejo robusto de errores"""
        try:
            # 1. Validar / solicitar datos mínimos requeridos (nombre y edad)
            required_fields = ['name', 'age']

            # Valores que indican nombre desconocido
            invalid_name_tokens = {"unknown", "desconocido", "sin nombre", "n/a", "na", ""}

            # Solicitar nombre si falta o es 'unknown'
            if (not patient_data.get('name')) or str(patient_data.get('name')).strip().lower() in invalid_name_tokens:
                try:
                    patient_data['name'] = input("🔎 No se detectó el nombre del paciente en la nota clínica. Introduce el nombre del paciente: ").strip()
                except Exception:
                    patient_data['name'] = "Paciente SinNombre"

            # Solicitar apellidos si faltan o son inválidos
            if (not patient_data.get('surname')) or str(patient_data.get('surname')).strip().lower() in invalid_name_tokens:
                try:
                    patient_data['surname'] = input("🔎 Introduce los apellidos del paciente (opcional, Enter para omitir): ").strip()
                except Exception:
                    patient_data['surname'] = ""

            # Solicitar edad si falta
            if not patient_data.get('age'):
                try:
                    age_input = input("🔎 Introduce la edad del paciente (en años): ").strip()
                    if age_input.isdigit():
                        patient_data['age'] = int(age_input)
                except Exception:
                    patient_data['age'] = None

            missing_fields = [f for f in required_fields if not patient_data.get(f)]
            if missing_fields:
                logger.warning(f"⚠️ Faltan campos requeridos incluso tras la solicitud: {missing_fields}")
                return self._create_error_response("Datos de paciente incompletos")

            # 2. Si tenemos introspector, usarlo para detectar la tabla de pacientes
            patient_table = None
            if self.introspector:
                try:
                    # Introspección completa del esquema para encontrar tabla de pacientes
                    introspection_result = await self.introspector.introspect_full_schema(max_tables=10)
                    
                    # Buscar tabla que parezca ser de pacientes
                    for table_name, mapping in introspection_result.fhir_mappings.items():
                        if mapping == 'Patient':
                            patient_table = table_name
                            logger.info(f"✅ Tabla de pacientes detectada: {patient_table}")
                            break
                except Exception as e:
                    logger.warning(f"⚠️ Error en introspección: {e}")

            # 3. Buscar paciente existente
            existing_patient = self._search_existing_patient(patient_data, conn)
            
            # Si el usuario canceló, propagar el resultado de cancelación
            if existing_patient and existing_patient.get('action') == 'cancel':
                return {'success': False, 'error': 'Operación cancelada por el usuario', 'action': 'cancel'}

            # 4. Crear o actualizar paciente
            if existing_patient and existing_patient.get('id'):
                logger.info(f"👤 Actualizando paciente existente: {existing_patient['id']}")
                return await self._update_existing_patient(existing_patient['id'], patient_data, conn)
            else:
                logger.info("👤 Creando nuevo paciente")
                return await self._create_patient(patient_data, conn)

        except Exception as e:
            logger.error(f"❌ Error procesando paciente: {e}")
            return self._create_error_response(str(e))

    def _search_existing_patient(self, patient_data: Dict[str, Any], conn: Optional[sqlite3.Connection] = None) -> Optional[Dict[str, Any]]:
        """Busca paciente existente usando múltiples criterios inteligentes."""
        try:
            if not self.llm or not self.sql_agent:
                logger.warning("⚠️ LLM o SQL Agent no disponible para búsqueda dinámica")
                return None

            print("🔍 Iniciando búsqueda inteligente de pacientes similares...")

            # 1. Identificar la tabla de pacientes
            print("   📋 Identificando tabla de pacientes...")
            patient_table_name = None
            candidate_tables = [tbl for tbl in self.sql_agent.column_metadata if 'pati' in tbl.lower()]
            
            if 'PATI_PATIENTS' in candidate_tables:
                patient_table_name = 'PATI_PATIENTS'
            elif candidate_tables:
                patient_table_name = candidate_tables[0]

            if not patient_table_name:
                print("   ❌ No se pudo determinar la tabla de pacientes para la búsqueda.")
                logger.warning("⚠️ No se pudo determinar la tabla de pacientes para la búsqueda.")
                return None

            print(f"   ✅ Tabla de pacientes identificada: {patient_table_name}")
            print(f"🔍 Buscando pacientes similares a: {patient_data.get('name', '')} {patient_data.get('surname', '')}, {patient_data.get('age', 'N/A')} años")
            
            available_columns = [col['name'] for col in self.sql_agent.column_metadata[patient_table_name]['columns']]
            print(f"   📊 Columnas disponibles: {len(available_columns)}")

            # 2. Búsqueda inteligente con múltiples criterios
            print("   🤖 Generando consulta de búsqueda con LLM...")
            
            # Filtrar solo columnas relevantes para la búsqueda
            name_surname_cols = [col for col in available_columns if any(keyword in col.lower() for keyword in ['name', 'nombre', 'surname', 'apellido'])][:5]
            other_cols = [col for col in available_columns if any(keyword in col.lower() for keyword in ['birth', 'id', 'gender'])][:5]
            relevant_columns = list(set(name_surname_cols + other_cols))

            prompt = f"""
            Busca pacientes similares en la tabla '{patient_table_name}'.

            NUEVO PACIENTE: {patient_data.get('name', '')} {patient_data.get('surname', '')}, {patient_data.get('age', '')} años

            COLUMNAS RELEVANTES: {', '.join(relevant_columns)}

            INSTRUCCIONES DE BÚSQUEDA (SQLite):
            1. BÚSQUEDA PRECISA: Si se proporcionan nombre Y apellido, la consulta debe buscar registros que coincidan con AMBOS (usando `AND`). Si solo se proporciona uno, busca por ese.
            2. BÚSQUEDA FLEXIBLE: Usa `LOWER()` y `LIKE '%valor%'` para el texto.
            3. CÁLCULO DE EDAD: Calcula la edad usando `(strftime('%Y','now') - strftime('%Y', PATI_BIRTH_DATE))` y busca en un rango de ±2 años.
            4. RESPUESTA: Devuelve SOLO el JSON.

            JSON:
            {{
                "query": "SELECT * FROM {patient_table_name} WHERE (LOWER(PATI_NAME) LIKE ? AND LOWER(PATI_SURNAME_1) LIKE ?) AND ... LIMIT 5",
                "params": ["%mario%", "%lopez%"]
            }}
            """
            
            print("   ⚡ Enviando consulta al LLM...")
            response = _call_openai_native(self.llm, [HumanMessage(content=prompt)])
            print("   ✅ Respuesta del LLM recibida")
            
            print("   🔧 Parseando respuesta JSON...")
            sql_info = self._try_parse_llm_json(response.content)

            if not sql_info or 'query' not in sql_info or 'params' not in sql_info:
                print("   ❌ Error en respuesta LLM para búsqueda de paciente")
                logger.error("❌ Error en respuesta LLM para búsqueda de paciente")
                return None

            print("   💾 Ejecutando consulta en base de datos...")
            # ARREGLO: Usar la conexión pasada como parámetro en lugar de crear una nueva
            if not conn:
                print("   ❌ No se proporcionó conexión a la base de datos")
                return None

            cursor = conn.cursor()
            cursor.execute(sql_info['query'], sql_info['params'])
            results = cursor.fetchall()
            print(f"   📊 Consulta ejecutada: {len(results)} resultados encontrados")

            if results:
                columns = [desc[0] for desc in cursor.description]
                pk_col = next((c['name'] for c in self.sql_agent.column_metadata[patient_table_name]['columns'] if c.get('pk')), columns[0])
                
                # Analizar todos los resultados y mostrar al usuario
                print(f"\n🚨 ATENCIÓN: Se encontraron {len(results)} pacientes similares:")
                
                for i, result in enumerate(results[:5], 1):  # Mostrar máximo 5
                    data = dict(zip(columns, result))
                    patient_id = data.get(pk_col)
                    
                    # Extraer información relevante para mostrar
                    name_cols = [col for col in columns if 'name' in col.lower() or 'nombre' in col.lower()]
                    surname_cols = [col for col in columns if 'surname' in col.lower() or 'apellido' in col.lower()]
                    birth_date_col = next((col for col in columns if 'birth' in col.lower()), None)
                    
                    display_info = []
                    
                    # Nombre y Apellido
                    name_val = data.get(name_cols[0]) if name_cols else 'N/A'
                    surname_val = data.get(surname_cols[0]) if surname_cols else ''
                    display_info.append(f"Nombre: {name_val} {surname_val}".strip())
                    
                    # Edad (calculada)
                    if birth_date_col and data.get(birth_date_col):
                        try:
                            age = self._calc_age(data[birth_date_col])
                            display_info.append(f"Edad: {age}")
                        except:
                            display_info.append(f"Fecha Nac: {data[birth_date_col]}") # Fallback
                    
                    info_str = " | ".join(display_info) if display_info else "Información limitada"
                    print(f"   {i}. ID: {patient_id} - {info_str}")
                
                # Preguntar al usuario qué hacer
                print(f"\n❓ ¿Qué deseas hacer?")
                print(f"   1. Usar el primer paciente similar (ID: {dict(zip(columns, results[0])).get(pk_col)})")
                print(f"   2. Crear un nuevo paciente de todas formas")
                print(f"   3. Cancelar la operación actual")

                while True:
                    choice = input("   Selecciona una opción (1, 2, 3): ").strip()
                    if choice == '1':
                        first_result = results[0]
                        data = dict(zip(columns, first_result))
                        patient_id = data.get(pk_col)
                        print(f"🔄 Usando paciente existente con ID: {patient_id}")
                        return {'id': patient_id, 'data': data}
                    elif choice == '2':
                        print("👤 Entendido. Se creará un nuevo paciente.")
                        return None # Retornar None indica que se debe crear un nuevo paciente
                    elif choice == '3':
                        print("❌ Operación cancelada por el usuario.")
                        return {'id': None, 'action': 'cancel'} # Indicar cancelación
                    else:
                        print("⚠️ Opción no válida. Por favor, introduce 1, 2 o 3.")
            
            print("✅ No se encontraron pacientes duplicados. Procediendo a crear nuevo paciente.")
            return None
            
        except Exception as e:
            print(f"   ❌ Error en búsqueda de pacientes: {str(e)}")
            logger.error(f"❌ Error buscando paciente: {e}", exc_info=True)
            return None

    async def _update_existing_patient(self, patient_id: str, patient_data: Dict[str, Any], conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        """Actualiza un paciente existente delegando toda la lógica SQL al SQLAgent."""
        try:
            if not self.sql_agent:
                return self._create_error_response("SQL Agent no disponible para actualizar paciente.")

            print(f"📝 Actualizando paciente existente (vía SQLAgent): ID {patient_id}")

            # Añadir el ID a los datos para que el SQLAgent lo use en la cláusula WHERE
            patient_data['id'] = patient_id

            # ARREGLO: Manejo correcto de event loops async con streaming
            print("   🤖 Preparando consulta SQL de actualización...")
            async def update_patient_async():
                if not self.sql_agent:
                    return {'success': False, 'message': 'SQL Agent no disponible'}
                print("   📝 Generando SQL para actualización de paciente...")
                result = await self.sql_agent.process_data_manipulation(
                    operation="UPDATE",
                    data=patient_data,
                    context={"table_hint": "PATI_PATIENTS", "conn": conn}
                )
                print("   ✅ SQL de actualización generado")
                return result
            
            # ARREGLO: Mejor manejo de event loops con progreso
            print("   ⚡ Ejecutando operación de actualización...")
            try:
                # Verificar si ya hay un loop corriendo
                loop = asyncio.get_running_loop()
                # Si hay un loop, usar create_task para evitar conflictos
                print("   🔄 Usando event loop existente...")
                task = loop.create_task(update_patient_async())
                # Simplemente esperamos a la tarea. El logging ahora está en el SQLAgent.
                dml_result = await task
                print("   ✅ Operación de actualización completada.")

            except RuntimeError:
                # No hay loop corriendo, usar asyncio.run
                print("   🔄 Creando nuevo event loop para la actualización del paciente...")
                dml_result = asyncio.run(update_patient_async())
                print("   ✅ Operación de actualización completada.")
                
            except Exception as e:
                logger.error(f"Error en gestión de event loop durante actualización de paciente: {e}", exc_info=True)
                print(f"   ⚠️ Error en event loop, usando fallback con ThreadPoolExecutor...")
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, update_patient_async())
                    dml_result = future.result(timeout=60) # Timeout de 60s
                print("   ✅ Operación de actualización completada vía fallback.")

            print("   📊 Procesando resultado de la actualización...")
            if not dml_result.get('success'):
                error_msg = dml_result.get('message', 'Error desconocido del SQLAgent al actualizar paciente.')
                print(f"   ❌ Error del SQL Agent: {error_msg}")
                logger.error(f"Error del SQLAgent: {error_msg}")
                return self._create_error_response(error_msg)

            print(f"✅ Paciente ID {patient_id} actualizado exitosamente por SQLAgent.")
            print(f"   📝 SQL ejecutado: {dml_result.get('sql_query', 'N/A')[:60]}...")
            
            # Crear recurso FHIR actualizado (esto se puede mejorar más adelante)
            print("   🏥 Generando recurso FHIR actualizado...")
            fhir_resource = self._create_patient_fhir_resource_dynamic(patient_data, patient_id)
            
            # ARREGLO: Incluir nombre y edad del paciente en el resultado
            patient_name = f"{patient_data.get('name', '')} {patient_data.get('surname', '')}".strip()
            patient_age = patient_data.get('age', 'unknown')
            
            # NUEVO: Capturar el rowid para referencia futura (aunque en UPDATE podría no cambiar)
            patient_rowid = dml_result.get('last_insert_id') or dml_result.get('patient_rowid') or patient_id
            
            return {
                'success': True,
                'patient_id': patient_id,
                'patient_rowid': patient_rowid,  # NUEVO: Incluir rowid de SQLite
                'patient_name': patient_name,  # NUEVO: Incluir nombre del paciente
                'patient_age': patient_age,     # NUEVO: Incluir edad del paciente
                'fhir_resource': fhir_resource,
                'action': 'updated',
                'sql': dml_result.get('sql_query', '')
            }
            
        except Exception as e:
            print(f"   ❌ Error técnico en actualización: {str(e)}")
            logger.error(f"❌ Error en FHIRAgent._update_existing_patient: {e}", exc_info=True)
            return self._create_error_response(f"Error técnico actualizando paciente: {str(e)}")

    async def _get_next_patient_id(self, conn: Optional[sqlite3.Connection] = None) -> str:
        """Genera un nuevo ID de paciente UUID."""
        import uuid
        return str(uuid.uuid4())

    async def _create_patient(self, patient_data: Dict[str, Any], conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        """Crea un nuevo paciente delegando toda la lógica SQL al SQLAgent."""
        try:
            if not self.sql_agent:
                return self._create_error_response("SQL Agent no disponible para crear paciente.")

            print(f"👤 Creando nuevo paciente (vía SQLAgent): {patient_data.get('name', '')} {patient_data.get('surname', '')}")

            # Generar un ID único para el paciente
            print("   🔢 Generando ID único para el paciente...")
            new_patient_id = await self._get_next_patient_id(conn) # Usar la función corregida
            patient_data['id'] = new_patient_id
            print(f"   ✅ ID generado: {new_patient_id}")

            # ARREGLO: Manejo correcto de event loops async con streaming
            print("   🤖 Preparando consulta SQL con LLM...")
            async def create_patient_async():
                if not self.sql_agent:
                    return {'success': False, 'message': 'SQL Agent no disponible'}
                print("   📝 Generando SQL para inserción de paciente...")
                result = await self.sql_agent.process_data_manipulation(
                    operation="INSERT",
                    data=patient_data,
                    context={"table_hint": "PATI_PATIENTS", "conn": conn}
                )
                print("   ✅ SQL generado por el agente")
                return result
            
            # ARREGLO: Mejor manejo de event loops con progreso
            print("   ⚡ Ejecutando operación asíncrona...")
            try:
                # Verificar si ya hay un loop corriendo
                loop = asyncio.get_running_loop()
                print("   🔄 Usando event loop existente...")
                task = loop.create_task(create_patient_async())
                # Simplemente esperamos a la tarea. El logging ahora está en el SQLAgent.
                dml_result = await task
                print("   ✅ Operación LLM (creación de paciente) completada.")

            except RuntimeError:
                # No hay loop corriendo, usar asyncio.run
                print("   🔄 Creando nuevo event loop para la creación del paciente...")
                dml_result = asyncio.run(create_patient_async())
                print("   ✅ Operación LLM (creación de paciente) completada.")
                
            except Exception as e:
                logger.error(f"Error en gestión de event loop durante creación de paciente: {e}", exc_info=True)
                print(f"   ⚠️ Error en event loop, usando fallback con ThreadPoolExecutor...")
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, create_patient_async())
                    dml_result = future.result(timeout=60) # Timeout de 60s
                print("   ✅ Operación LLM (creación de paciente) completada vía fallback.")

            print("   📊 Procesando resultado del SQL Agent...")
            if not dml_result or not dml_result.get('success'):
                error_msg = dml_result.get('message', 'Error desconocido del SQLAgent al crear paciente.') if dml_result else 'No se obtuvo respuesta del SQLAgent'
                print(f"   ❌ Error del SQL Agent: {error_msg}")
                logger.error(f"Error del SQLAgent: {error_msg}")
                return self._create_error_response(error_msg)

            print(f"✅ Nuevo paciente creado exitosamente por SQLAgent.")
            print(f"   🆔 ID asignado: {new_patient_id}")
            print(f"   📝 SQL ejecutado: {dml_result.get('sql_query', 'N/A')[:60]}...")
            
            # ARREGLO: Incluir nombre y edad del paciente en el resultado
            patient_name = f"{patient_data.get('name', '')} {patient_data.get('surname', '')}".strip()
            patient_age = patient_data.get('age', 'unknown')
            
            # NUEVO: Capturar el rowid para referencia futura
            patient_rowid = dml_result.get('last_insert_id') or dml_result.get('patient_rowid')
            
            return {
                "success": True, 
                "patient_id": new_patient_id,
                "patient_rowid": patient_rowid,  # NUEVO: Incluir rowid de SQLite
                "patient_name": patient_name,  # NUEVO: Incluir nombre del paciente
                "patient_age": patient_age,     # NUEVO: Incluir edad del paciente
                "sql": dml_result.get('sql_query', ''),
                "saved": True,
                "action": "created"
            }

        except Exception as e:
            print(f"   ❌ Error técnico en creación: {str(e)}")
            logger.error(f"❌ Error en FHIRAgent._create_patient: {e}", exc_info=True)
            return self._create_error_response(f"Error técnico creando paciente: {str(e)}")

    def _validate_patient_data_dynamic(self, sql_data: Dict[str, Any], validations: List[str]) -> bool:
        """Valida datos de paciente basándose en validaciones sugeridas por LLM"""
        try:
            # Por ahora, validación básica
            # TODO: Implementar validaciones más complejas basadas en las sugerencias del LLM
            return True
        except Exception:
            return False

    def _create_patient_fhir_resource_dynamic(self, sql_data: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
        """Crea recurso FHIR de paciente usando LLM para mapeo dinámico"""
        try:
            if not self.llm:
                # Fallback básico si no hay LLM
                return {
            'resourceType': 'Patient',
                    'id': patient_id,
            'meta': {
                        'lastUpdated': datetime.now().isoformat()
                    }
                }
            
            # Usar LLM para generar recurso FHIR
            prompt = f"""
            Convierte estos datos SQL de paciente a un recurso FHIR Patient válido:
            
            Datos SQL: {json.dumps(sql_data, indent=2)}
            ID del paciente: {patient_id}
            
            Genera un recurso FHIR Patient completo siguiendo el estándar HL7 FHIR R4.
            
            Incluye:
            - resourceType: "Patient"
            - id
            - meta con lastUpdated
            - identifier con sistema y valor
            - name (convierte apropiadamente)
            - birthDate si hay información de edad o fecha
            - gender si está disponible
            - telecom si hay teléfono/email
            - address si hay dirección
            
            Responde SOLO con el JSON del recurso FHIR.
            """
            
            response = _call_openai_native(self.llm, [HumanMessage(content=prompt)])
            
            try:
                fhir_resource = self._try_parse_llm_json(response.content)
                if not fhir_resource:
                    raise json.JSONDecodeError("LLM response is not valid JSON", "", 0)
                
                # Asegurar campos mínimos
                fhir_resource['resourceType'] = 'Patient'
                fhir_resource['id'] = patient_id
                
                if 'meta' not in fhir_resource:
                    fhir_resource['meta'] = {}
                fhir_resource['meta']['lastUpdated'] = datetime.now().isoformat()
                
                return fhir_resource
                
            except json.JSONDecodeError:
                logger.error("❌ Error decodificando FHIR del LLM, usando fallback")
                return {
                    'resourceType': 'Patient',
                    'id': patient_id,
                    'meta': {
                        'lastUpdated': datetime.now().isoformat()
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Error creando recurso FHIR: {e}")
            return {
                'resourceType': 'Patient',
                'id': patient_id,
                'meta': {
                    'lastUpdated': datetime.now().isoformat()
                }
            }

    async def _save_to_database(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Guarda los datos extraídos usando el sistema de transacciones mejorado."""
        # Usar el nuevo método transaccional
        return await self._save_all_data_transactional(extracted_data)
    
    async def _save_all_data_transactional(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Orquesta la transacción completa: procesa al paciente y luego los datos clínicos."""
        results = {'created': [], 'updated': [], 'errors': [], 'success': False}
        
        # ARREGLO: Usar una sola conexión con timeout más largo
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=60.0)  # Timeout de 60 segundos
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE TRANSACTION")  # Bloqueo inmediato para evitar conflictos
            
            # Parte 1: Procesar paciente
            patient_result = await self.process_patient_data(extracted_data.get('patient', {}), conn)
            if not patient_result.get('success'):
                results['errors'].append(patient_result.get('error', 'Error procesando paciente.'))
                conn.rollback()
                return results

            patient_id = patient_result.get('patient_id')
            if patient_result.get('action') == 'created': 
                results['created'].append('Patient')
            if patient_result.get('action') == 'updated': 
                results['updated'].append('Patient')
            
            # NUEVO: Guardar el resultado del paciente completo
            results['patient_result'] = patient_result

            # Parte 2: Guardar datos clínicos adicionales (solo si tenemos un ID de paciente)
            if patient_id:
                clinical_results = await self._save_clinical_data_within_transaction(patient_id, extracted_data, conn)
                results['created'].extend(clinical_results.get('created', []))
                results['errors'].extend(clinical_results.get('errors', []))

            # Parte 3: Generar bundle FHIR si tenemos un paciente (incluso con errores en datos clínicos)
            fhir_bundle = None
            if patient_id:
                try:
                    print(f"   📦 Generando bundle FHIR...")
                    fhir_bundle = await self._generate_fhir_bundle(patient_id, extracted_data)
                    if fhir_bundle:
                        results['fhir_bundle'] = fhir_bundle
                        print(f"   ✅ Bundle FHIR generado: {fhir_bundle.get('id', 'N/A')[:8]}...")
                except Exception as e:
                    logger.warning(f"Error generando bundle FHIR: {e}")

            # NUEVO: Lógica de rollback más inteligente
            # Solo hacer rollback si falló la creación del paciente
            patient_errors = [e for e in results['errors'] if 'patient' in e.lower() or 'paciente' in e.lower()]
            clinical_errors = [e for e in results['errors'] if e not in patient_errors]
            
            if patient_errors:
                print(f"   ❌ Rollback por errores críticos del paciente: {len(patient_errors)} errores")
                conn.rollback()
            elif clinical_errors:
                print(f"   ⚠️ Commit con errores en datos clínicos: {len(clinical_errors)} errores (paciente guardado)")
                results['success'] = True
                results['warnings'] = clinical_errors
                conn.commit()
                print(f"   ✅ Transacción completada con advertencias")
            else:
                results['success'] = True
                conn.commit()
                print(f"   ✅ Transacción completada exitosamente")

            return results
            
        except Exception as e:
            logger.error(f"Error en la transacción principal: {e}", exc_info=True)
            if conn:
                try: 
                    conn.rollback()
                    print(f"   🔄 Rollback ejecutado por error: {str(e)}")
                except Exception as rb_e: 
                    logger.error(f"Error en rollback: {rb_e}")
            results['errors'].append(str(e))
            return results
        finally:
            if conn:
                conn.close()
                self._local.conn = None
    
    async def _save_clinical_data_within_transaction(self, patient_id: str, extracted_data: Dict[str, Any], conn: sqlite3.Connection) -> Dict[str, Any]:
        """Guarda los datos clínicos usando el sistema dinámico del SQL Agent"""
        results = {'created': [], 'errors': []}
        
        if not self.sql_agent:
            results['errors'].append("SQLAgent no disponible para guardado dinámico")
            return results
            
        print(f"   ⚡ Guardando datos clínicos para paciente {patient_id} (sistema dinámico)...")
        
        # Procesar cada tipo de dato clínico
        data_types = {
            "conditions": "diagnóstico/condición médica",
            "medications": "medicamento/tratamiento", 
            "observations": "observación/signo vital",
            "procedures": "procedimiento/intervención",
            "allergies": "alergia"
        }
        
        total_items = 0
        for data_type, description in data_types.items():
            items = extracted_data.get(data_type, [])
            if items:
                print(f"   📊 Procesando {len(items)} {description}(es)...")
                total_items += len(items)
                
                for i, item_data in enumerate(items, 1):
                    try:
                        # Agregar información del paciente al item
                        item_data['patient_id'] = patient_id
                        item_data['recorded_date'] = datetime.now().isoformat()
                        
                        # Usar el sistema dinámico del SQL Agent
                        result = await self.sql_agent.process_data_manipulation(
                            operation="INSERT",
                            data=item_data,
                            context={"intent": f"create_{data_type[:-1]}", "conn": conn}
                        )
                        
                        if result.get('success'):
                            results['created'].append(f"{data_type}_{i}")
                            table_used = result.get('table_used', 'desconocida')
                            print(f"      ✅ {description.capitalize()} #{i}: Guardado en '{table_used}'")
                        else:
                            error_msg = result.get('message', f"Error desconocido al guardar {description} #{i}")
                            results['errors'].append(error_msg)
                            print(f"      ❌ {description.capitalize()} #{i}: {error_msg}")
                            
                    except Exception as e:
                        error_msg = f"Error crítico guardando {description} #{i}: {str(e)}"
                        results['errors'].append(error_msg)
                        print(f"      ⚠️ Continuando con siguiente item después de error...")
        
        print(f"\n   ✅ Proceso de guardado dinámico finalizado.")
        print(f"   📈 Total procesado: {total_items} elementos")
        if results['errors']:
            print(f"   ⚠️ {len(results['errors'])} errores durante el guardado (pero continuó procesando).")
        else:
            print(f"   ✅ Todos los datos clínicos guardados exitosamente.")
        
        return results
    
    async def _generate_fhir_bundle(self, patient_id: str, extracted_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Genera un bundle FHIR con todos los recursos creados y los persiste usando FHIRPersistenceAgent."""
        try:
            bundle_id = str(uuid.uuid4())
            bundle = {
                "resourceType": "Bundle",
                "id": bundle_id,
                "meta": {
                    "lastUpdated": datetime.now().isoformat()
                },
                "type": "collection",
                "timestamp": datetime.now().isoformat(),
                "entry": []
            }
            
            # Lista para almacenar resultados de persistencia
            persistence_results = []
            
            # 1. Añadir recurso Patient
            patient_resource = self._create_patient_fhir_resource_dynamic(extracted_data.get('patient', {}), patient_id)
            bundle["entry"].append({
                "fullUrl": f"Patient/{patient_id}",
                "resource": patient_resource
            })
            
            # Persistir el recurso Patient si tenemos persistence_agent
            if self.persistence_agent:
                try:
                    result: Union[PersistenceResult, Dict[str, Any]] = await self.persistence_agent.persist_fhir_resource(patient_resource)
                    persistence_results.append(result)
                    success = result.success if isinstance(result, PersistenceResult) else result.get('success', False)
                    sql_queries = result.sql_queries if isinstance(result, PersistenceResult) else result.get('sql_queries', [])
                    if success:
                        logger.info(f"✅ Recurso Patient persistido: {len(sql_queries)} queries SQL generadas")
                except Exception as e:
                    logger.warning(f"⚠️ Error persistiendo Patient: {e}")
            
            # 2. Añadir recursos Condition (diagnósticos)
            for i, condition in enumerate(extracted_data.get('conditions', [])):
                condition_resource = {
                    "resourceType": "Condition",
                    "id": f"condition-{patient_id}-{i+1}",
                    "clinicalStatus": {
                        "coding": [{
                            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                            "code": "active"
                        }]
                    },
                    "code": {
                        "text": condition.get('description', 'Condición médica')
                    },
                    "subject": {
                        "reference": f"Patient/{patient_id}"
                    },
                    "recordedDate": datetime.now().isoformat()
                }
                bundle["entry"].append({
                    "fullUrl": f"Condition/condition-{patient_id}-{i+1}",
                    "resource": condition_resource
                })
                
                # Persistir el recurso Condition
                if self.persistence_agent:
                    try:
                        result = await self.persistence_agent.persist_fhir_resource(condition_resource)
                        persistence_results.append(result)
                    except Exception as e:
                        logger.warning(f"⚠️ Error persistiendo Condition {i+1}: {e}")
            
            # 3. Añadir recursos MedicationRequest (medicamentos)
            for i, medication in enumerate(extracted_data.get('medications', [])):
                med_resource = {
                    "resourceType": "MedicationRequest",
                    "id": f"medication-{patient_id}-{i+1}",
                    "status": "active",
                    "intent": "order",
                    "medicationCodeableConcept": {
                        "text": medication.get('name', 'Medicamento')
                    },
                    "subject": {
                        "reference": f"Patient/{patient_id}"
                    },
                    "dosageInstruction": [{
                        "text": medication.get('dosage', 'Según indicación médica')
                    }],
                    "authoredOn": datetime.now().isoformat()
                }
                bundle["entry"].append({
                    "fullUrl": f"MedicationRequest/medication-{patient_id}-{i+1}",
                    "resource": med_resource
                })
                
                # Persistir el recurso MedicationRequest
                if self.persistence_agent:
                    try:
                        result = await self.persistence_agent.persist_fhir_resource(med_resource)
                        persistence_results.append(result)
                    except Exception as e:
                        logger.warning(f"⚠️ Error persistiendo MedicationRequest {i+1}: {e}")
            
            # 4. Añadir recursos Observation (observaciones)
            for i, observation in enumerate(extracted_data.get('observations', [])):
                obs_resource = {
                    "resourceType": "Observation",
                    "id": f"observation-{patient_id}-{i+1}",
                    "status": "final",
                    "code": {
                        "text": observation.get('type', 'Observación')
                    },
                    "subject": {
                        "reference": f"Patient/{patient_id}"
                    },
                    "effectiveDateTime": datetime.now().isoformat(),
                    "valueString": observation.get('value', '')
                }
                bundle["entry"].append({
                    "fullUrl": f"Observation/observation-{patient_id}-{i+1}",
                    "resource": obs_resource
                })
                
                # Persistir el recurso Observation
                if self.persistence_agent:
                    try:
                        result = await self.persistence_agent.persist_fhir_resource(obs_resource)
                        persistence_results.append(result)
                    except Exception as e:
                        logger.warning(f"⚠️ Error persistiendo Observation {i+1}: {e}")
            
            # Agregar resumen de persistencia al bundle
            if persistence_results:
                total_records = sum(r.records_created for r in persistence_results)
                total_errors = sum(len(r.errors) for r in persistence_results)
                logger.info(f"📊 Resumen de persistencia FHIR: {total_records} registros creados, {total_errors} errores")
                
                bundle["meta"]["extension"] = [{
                    "url": "http://chatmed.ai/fhir/extensions/persistence-summary",
                    "valueString": f"Registros SQL creados: {total_records}, Errores: {total_errors}"
                }]
            
            return bundle
            
        except Exception as e:
            logger.error(f"❌ Error guardando en base de datos: {e}")
            return {
                'created': [],
                'updated': [],
                'errors': [str(e)],
                'success': False
            }

    async def _prepare_condition_sql(self, condition: Dict[str, Any], patient_id: str) -> Optional[Dict[str, Any]]:
        """Prepara datos SQL para una condición usando LLM"""
        if not self.llm:
            return None
            
        prompt = f"""
        Prepara datos SQL para guardar esta condición médica:
        Condición: {json.dumps(condition)}
        ID del paciente: {patient_id}
        
        Genera un diccionario con los campos SQL apropiados.
        Incluye ID único, referencia al paciente, descripción, código si está disponible, fecha, etc.
        
        Responde SOLO con el diccionario JSON.
        """
        
        response = _call_openai_native(self.llm, [HumanMessage(content=prompt)])
        try:
            return json.loads(response.content)
        except:
            return None

    async def _prepare_medication_sql(self, medication: Dict[str, Any], patient_id: str) -> Optional[Dict[str, Any]]:
        """Prepara datos SQL para un medicamento usando LLM"""
        if not self.llm:
            return None
            
        prompt = f"""
        Prepara datos SQL para guardar este medicamento:
        Medicamento: {json.dumps(medication)}
        ID del paciente: {patient_id}
        
        Genera un diccionario con los campos SQL apropiados.
        Incluye ID único, referencia al paciente, nombre del medicamento, dosis, frecuencia, etc.
        
        Responde SOLO con el diccionario JSON.
        """
        
        response = _call_openai_native(self.llm, [HumanMessage(content=prompt)])
        try:
            return json.loads(response.content)
        except:
            return None

    async def _prepare_observation_sql(self, observation: Dict[str, Any], patient_id: str) -> Optional[Dict[str, Any]]:
        """Prepara datos SQL para una observación usando LLM"""
        if not self.llm:
            return None
            
        prompt = f"""
        Prepara datos SQL para guardar esta observación médica:
        Observación: {json.dumps(observation)}
        ID del paciente: {patient_id}
        
        Genera un diccionario con los campos SQL apropiados.
        Incluye ID único, referencia al paciente, tipo, valor, unidad, fecha, etc.
        
        Responde SOLO con el diccionario JSON.
        """
        
        response = _call_openai_native(self.llm, [HumanMessage(content=prompt)])
        try:
            return json.loads(response.content)
        except:
                return None
            
    async def _save_with_llm_generated_sql(self, extracted_data: Dict[str, Any], 
                                          patient_id: str, results: Dict[str, Any]) -> None:
        """Guarda datos usando SQL generado completamente por LLM"""
        if not self.llm:
            return
            
        # Generar SQL para cada tipo de recurso
        resources_to_save = {
            'conditions': extracted_data.get('conditions', []),
            'medications': extracted_data.get('medications', []),
            'observations': extracted_data.get('observations', [])
        }
        
        for resource_type, resources in resources_to_save.items():
            for resource in resources:
                prompt = f"""
                Genera el SQL completo para guardar este recurso médico en la base de datos:
                
                Tipo: {resource_type}
                Datos: {json.dumps(resource)}
                ID del paciente: {patient_id}
                
                La base de datos tiene tablas médicas estándar.
                
                Genera:
                1. La query INSERT completa con placeholders (?)
                2. Los valores en orden como lista
                3. El nombre de la tabla apropiada
                
                Formato JSON:
                {{
                    "query": "INSERT INTO tabla (...) VALUES (?...)",
                    "values": [...],
                    "table": "nombre_tabla"
                }}
                """
                
                response = _call_openai_native(self.llm, [HumanMessage(content=prompt)])
                
                try:
                    sql_info = json.loads(response.content)
                    
                    conn = self._get_db_connection()
                    if conn:
                        cursor = conn.cursor()
                        cursor.execute(sql_info['query'], sql_info['values'])
                        conn.commit()
                        results['created'].append(resource_type[:-1].capitalize())  # Remove 's' del plural
                        
                except Exception as e:
                    results['errors'].append(f"Error en {resource_type}: {str(e)}")

    async def _extract_clinical_data(self, note: str) -> Dict[str, Any]:
        """Extrae información clínica de forma ultra exhaustiva, global y genérica usando LLM"""
        try:
            if not self.llm:
                return {'error': 'LLM no disponible'}

            prompt = f"""
Eres un experto en extracción de datos clínicos. Analiza la siguiente nota clínica y extrae absolutamente TODA la información estructurada posible, sin omitir nada, aunque esté repetida, dispersa o en diferentes formatos.

NOTA CLÍNICA:
{note}

INSTRUCCIONES ULTRA EXHAUSTIVAS Y FLEXIBLES:
- Extrae TODOS los datos del paciente: nombre, apellidos, sexo, edad, fecha de nacimiento, número de historia clínica, dirección, contacto, etc.
- Extrae TODAS las condiciones/diagnósticos (crónicos y agudos): nombre, código, severidad, fecha de diagnóstico, estado, notas, etc.
- Extrae TODOS los medicamentos: nombre, principio activo, dosis, frecuencia, vía, indicación, fechas de inicio/fin, estado, notas, etc.
- Extrae TODAS las alergias: sustancia, tipo, reacción, severidad, fecha, notas.
- Extrae TODAS las observaciones: tipo, valor, unidad, fecha, contexto, notas (ej: signos vitales, resultados de laboratorio, síntomas, hallazgos físicos, escalas, etc.)
- Extrae TODOS los procedimientos/intervenciones: nombre, tipo, fecha, resultado, notas.
- Extrae TODO el plan de tratamiento: cambios, recomendaciones, seguimientos, objetivos.
- Extrae cualquier otro dato clínico relevante aunque no esté en los ejemplos.
- Si hay listas anidadas (ej: varios diagnósticos, varios medicamentos), inclúyelas todas.
- Si algún campo no está presente, déjalo vacío o null, pero incluye la clave.
- NO omitas ningún campo relevante aunque no esté en los ejemplos.
- El JSON debe ser lo más completo y flexible posible, permitiendo campos adicionales si aparecen.

ESTRUCTURA JSON GLOBAL Y FLEXIBLE (EJEMPLO):
{
  "patient": {
    "name": "",
    "surname": "",
    "age": "",
    "gender": "",
    "birth_date": "",
    "clinical_history_id": "",
    "address": "",
    "contact": ""
  },
  "conditions": [
    {
      "description": "",
      "code": "",
      "severity": "",
      "diagnosis_date": "",
      "status": "",
      "notes": ""
    }
  ],
  "medications": [
    {
      "name": "",
      "active_ingredient": "",
      "dosage": "",
      "frequency": "",
      "route": "",
      "indication": "",
      "start_date": "",
      "end_date": "",
      "status": "",
      "notes": ""
    }
  ],
  "allergies": [
    {
      "substance": "",
      "type": "",
      "reaction": "",
      "severity": "",
      "date": "",
      "notes": ""
    }
  ],
  "observations": [
    {
      "type": "",
      "value": "",
      "unit": "",
      "date": "",
      "context": "",
      "notes": ""
    }
  ],
  "procedures": [
    {
      "name": "",
      "type": "",
      "date": "",
      "result": "",
      "notes": ""
    }
  ],
  "plan": [
    {
      "action": "",
      "recommendation": "",
      "followup": "",
      "goal": "",
      "notes": ""
    }
  ],
  "other": [
    {
      "field": "",
      "value": "",
      "notes": ""
    }
  ]
}

IMPORTANTE: El JSON debe ser válido, global, flexible y contener absolutamente toda la información posible. Si algún campo no aplica, inclúyelo vacío. Si aparecen campos nuevos, inclúyelos también.

Responde SOLO con el JSON válido, sin explicaciones.
            """

            response = _call_openai_native(self.llm, [HumanMessage(content=prompt)])
            retry_resp = None
            extracted_data = self._try_parse_llm_json(response.content)

            # Validación exhaustiva: si alguna lista relevante está vacía, reintenta
            listas_relevantes = ["conditions", "medications", "observations", "allergies", "procedures", "plan"]
            needs_retry = False
            if not extracted_data:
                needs_retry = True
            else:
                for key in listas_relevantes:
                    if key not in extracted_data or not isinstance(extracted_data[key], list) or len(extracted_data[key]) == 0:
                        needs_retry = True
                        break

            if needs_retry:
                retry_prompt = f"""
Repite la extracción de la nota clínica de forma aún más exhaustiva. NO omitas ningún diagnóstico, medicamento, observación, alergia, procedimiento ni plan. Si algún campo no está presente, inclúyelo vacío, pero asegúrate de que todas las listas tengan al menos un elemento (aunque sea vacío). Si aparecen campos nuevos, inclúyelos también. Responde SOLO con el JSON global y flexible.

NOTA CLÍNICA:
{note}
                """
                retry_resp = _call_openai_native(self.llm, [HumanMessage(content=retry_prompt)])
                extracted_data = self._try_parse_llm_json(retry_resp.content)

            if not extracted_data:
                logger.error("❌ Error decodificando JSON de respuesta LLM tras reintento")
                return {
                    'error': 'Respuesta LLM no es JSON válido',
                    'raw_response': response.content,
                    'retry_response': retry_resp.content if retry_resp else 'No hubo reintento'
                }

            # Log de conteo de entidades extraídas
            logger.info(f"✅ Datos clínicos extraídos: " + ", ".join([f"{k}: {len(extracted_data.get(k, []))}" for k in listas_relevantes]))
            return extracted_data

        except Exception as e:
            logger.error(f"❌ Error extrayendo datos clínicos: {e}")
            return {'error': str(e)}

    def _calc_age(self, birthdate: str) -> int:
        """Calcula edad aproximada a partir de una fecha AAAA-MM-DD."""
        try:
            birth = datetime.fromisoformat(birthdate).date()
            today = datetime.today().date()
            return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        except Exception:
            return 0

    async def _save_clinical_data_transactional(self, patient_id: int, extracted_data: Dict, conn: sqlite3.Connection) -> Dict:
        """
        Guarda todos los datos clínicos (condiciones, medicamentos, etc.)
        dentro de una transacción existente.
        """
        summary = {
            'conditions_saved': 0,
            'medications_saved': 0,
            'observations_saved': 0,
            'errors': []
        }

        if not self.sql_agent:
            summary['errors'].append("El Agente SQL no está disponible para guardar datos clínicos.")
            return summary

        # Guardar Condiciones - SISTEMA FLEXIBLE
        for condition in extracted_data.get('conditions', []):
            try:
                # ✅ NUEVO: Usar el agente SQL para mapeo inteligente
                condition_data = {
                    "condition_description": condition.get('description'),
                    "condition_code": condition.get('code'),
                    "patient_id": patient_id,
                    "created_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # ✅ FLEXIBLE: Dejar que el agente SQL determine la tabla y mapeo
                result = await self.sql_agent.process_data_manipulation(
                    operation="INSERT",
                    data=condition_data,
                    context={"intent": "create_condition", "conn": conn}
                )
                
                if result.get('success'):
                    summary['conditions_saved'] += 1
                    print(f"   ✅ Condición guardada: {condition.get('description')} en tabla {result.get('table_used', 'desconocida')}")
                else:
                    summary['errors'].append(f"Condición '{condition.get('description')}': {result.get('message')}")
            except Exception as e:
                summary['errors'].append(f"Error guardando condición: {e}")
        
        # Guardar Medicamentos - SISTEMA FLEXIBLE
        for medication in extracted_data.get('medications', []):
            try:
                # ✅ NUEVO: Usar el agente SQL para mapeo inteligente
                # El agente SQL detectará automáticamente la tabla correcta y mapeará los campos
                med_data = {
                    "medication_name": medication.get('name'),
                    "dosage": medication.get('dosage'),
                    "patient_id": patient_id,
                    "created_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # ✅ FLEXIBLE: Dejar que el agente SQL determine la tabla y mapeo
                result = await self.sql_agent.process_data_manipulation(
                    operation="INSERT",
                    data=med_data,
                    context={"intent": "create_medication", "conn": conn}
                )
                
                if result.get('success'):
                    summary['medications_saved'] += 1
                    print(f"   ✅ Medicamento guardado: {medication.get('name')} en tabla {result.get('table_used', 'desconocida')}")
                else:
                    summary['errors'].append(f"Medicamento '{medication.get('name')}': {result.get('message')}")
            except Exception as e:
                summary['errors'].append(f"Error guardando medicamento: {e}")

        # Guardar Observaciones - SISTEMA FLEXIBLE
        for observation in extracted_data.get('observations', []):
            try:
                # ✅ NUEVO: Usar el agente SQL para mapeo inteligente
                obs_data = {
                    "observation_type": observation.get('type'),
                    "observation_value": observation.get('value'),
                    "observation_unit": observation.get('unit'),
                    "patient_id": patient_id,
                    "created_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # ✅ FLEXIBLE: Dejar que el agente SQL determine la tabla y mapeo
                result = await self.sql_agent.process_data_manipulation(
                    operation="INSERT",
                    data=obs_data,
                    context={"intent": "create_observation", "conn": conn}
                )
                
                if result.get('success'):
                    summary['observations_saved'] += 1
                    print(f"   ✅ Observación guardada: {observation.get('type')} en tabla {result.get('table_used', 'desconocida')}")
                else:
                    summary['errors'].append(f"Observación '{observation.get('type')}': {result.get('message')}")
            except Exception as e:
                summary['errors'].append(f"Error guardando observación: {e}")

        print(f"   -> Resumen de guardado: {summary['conditions_saved']} condiciones, {summary['medications_saved']} medicamentos, {summary['observations_saved']} observaciones.")
        return summary

    def _create_error_response(self, error: str) -> Dict[str, Any]:
        """Crea respuesta de error estandarizada"""
        return {
            'success': False,
            'message': f"❌ Error: {error}",
            'error': error
        }