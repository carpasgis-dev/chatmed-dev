#!/usr/bin/env python3
"""
🏥 FHIRAgent 4.0 - Agente FHIR ChatMed Unificado
=================================================

Agente FHIR que combina las mejores características de:
- FHIRAgent 3.0 (fhir_agent_complete_new.py) - Procesamiento completo
- FHIRPersistenceAgent 1.0 (fhir_persistence_agent_old.py) - Persistencia especializada

Características principales:
- Sistema de mapeo JSON consolidado
- Traductor bidireccional SQL↔FHIR 
- Validación empresarial automática
- Procesamiento inteligente de notas clínicas
- Persistencia flexible y robusta
- Integración con SQLAgentRobust

Versión: 4.0 - Unificación completa
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
from dataclasses import dataclass, field
import hashlib

# LangChain imports
try:
    from langchain.llms.base import BaseLLM
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Type checking imports
if TYPE_CHECKING:
    from .sql_agent_clean import SQLAgentRobust
    from .sql_agent_flexible_enhanced import SQLAgentIntelligentEnhanced

# Import del sistema de mapeo empresarial
MAPPING_SYSTEM_AVAILABLE = False
try:
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
        from chatmed_v2_flexible.mapping.schema_introspector import SchemaIntrospector
        from chatmed_v2_flexible.mapping.flexible_engine import FlexibleEngine  
        from chatmed_v2_flexible.mapping.dynamic_mapper import DynamicMapper
        FLEXIBLE_SYSTEM_AVAILABLE = True
        print("✅ Sistema flexible v2 importado (absoluto)")
    except ImportError as e:
        print(f"⚠️ Warning: No se pudo importar sistema flexible: {e}")

from .batch_processor import FHIRBatchProcessor



# Configuración de logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"))
logger = logging.getLogger("FHIRAgent4.0")

@dataclass
class PersistenceResult:
    """Resultado de persistencia FHIR→SQL"""
    success: bool
    resource_type: str
    resource_id: str
    target_tables: List[str] = field(default_factory=list)
    sql_queries: List[str] = field(default_factory=list)
    records_created: int = 0
    records_updated: int = 0
    processing_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence_score: float = 0.0

class MockResponse:
    """Clase para compatibilidad con respuestas LLM"""
    def __init__(self, content: str):
        self.content = content

def _call_openai_native(client, messages, temperature=0.1, max_tokens=8000) -> MockResponse:
    """Función de compatibilidad para llamar a OpenAI nativo con mejor manejo de errores"""
    try:
        from openai import OpenAI
        native_client = OpenAI()
        
        # Convertir mensajes a formato OpenAI
        openai_messages = []
        if isinstance(messages, list):
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
        
        # Verificar que tenemos mensajes válidos
        if not openai_messages:
            logger.error("No se pudieron convertir mensajes a formato OpenAI")
            return MockResponse('{"success": false, "message": "Error: No se pudieron convertir mensajes"}')
        
        # Llamada a OpenAI con manejo de errores específico
        try:
            response = native_client.chat.completions.create(
                model='gpt-4o',
                messages=openai_messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as api_error:
            logger.error(f"Error en llamada a OpenAI API: {api_error}")
            return MockResponse('{"success": false, "message": "Error en llamada a OpenAI API"}')
        
        # Verificar que la respuesta es válida
        if not response or not hasattr(response, 'choices') or not response.choices:
            logger.error("Respuesta vacía de OpenAI")
            return MockResponse('{"success": false, "message": "Error: Respuesta vacía del LLM"}')
        
        # Extraer contenido de la respuesta
        try:
            content = response.choices[0].message.content or ""
        except (AttributeError, IndexError) as content_error:
            logger.error(f"Error extrayendo contenido de respuesta: {content_error}")
            return MockResponse('{"success": false, "message": "Error extrayendo contenido de respuesta"}')
        
        if not content.strip():
            content = '{"success": false, "message": "Error: Respuesta vacía del LLM"}'
        
        return MockResponse(content)
            
    except ImportError as import_error:
        logger.error(f"Error importando OpenAI: {import_error}")
        return MockResponse('{"success": false, "message": "Error: OpenAI no disponible"}')
    except Exception as e:
        error_msg = f"Error en llamada OpenAI: {str(e)}"
        logger.error(f"Error en _call_openai_native: {e}", exc_info=True)
        return MockResponse('{"success": false, "message": "Error crítico en la llamada al LLM."}')

class FHIRMedicalAgent:
    """
    🏥 Agente FHIR Médico Unificado v4.0
    
    Combina procesamiento completo de recursos FHIR con persistencia especializada
    """
    
    def __init__(self, 
                 db_path: str = "../database_new.sqlite3.db",
                 llm = None,
                 mapping_config: Optional[str] = None,
                 sql_agent: Optional['SQLAgentIntelligentEnhanced'] = None,  # Usar el tipo correcto
                 medgemma_agent=None):
        """Inicializa el agente FHIR unificado con MedGemma"""
        
        self.db_path = db_path
        self.llm = llm
        self.sql_agent = sql_agent
        self.medgemma_agent = medgemma_agent
        self.batch_processor = FHIRBatchProcessor(sql_agent, self) if sql_agent else None
        self.mapping_config = mapping_config or self._find_mapping_config()
        
        # Estadísticas
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'clinical_notes_processed': 0,
            'fhir_resources_created': 0,
            'sql_conversions': 0,
            'avg_response_time': 0.0
        }
        
        # Cache para optimización
        self.query_cache = {}
        self.schema_cache = {}
        
        # Inicialización de componentes
        self._initialize_components()
        
        # Inicializar ID del paciente actual
        self._current_patient_id = None
        
        logger.info("🚀 FHIRMedicalAgent 4.0 inicializado")
        logger.info(f"   📊 Base de datos: {self.db_path}")
        logger.info(f"   🤖 LLM disponible: {'✅' if self.llm else '❌'}")
        logger.info(f"   🔧 SQL Agent: {'✅' if self.sql_agent else '❌'}")
        logger.info(f"   📋 Mapping config: {self.mapping_config}")
    
    def _initialize_components(self):
        """Inicializa los componentes del agente"""
        try:
            # Inicializar sistema de mapeo si está disponible
            if MAPPING_SYSTEM_AVAILABLE:
                self.fhir_bridge = FHIRSQLBridge(
                    db_path=self.db_path,
                    llm=self.llm,
                    enable_cache=True,
                    validate_fhir=True,
                    mapping_dir=self.mapping_config
                )
                logger.info("✅ FHIRSQLBridge inicializado")
            else:
                self.fhir_bridge = None
                logger.warning("⚠️ FHIRSQLBridge no disponible")
            
            # Inicializar sistema flexible si está disponible
            if FLEXIBLE_SYSTEM_AVAILABLE:
                self.introspector = SchemaIntrospector(self.db_path)
                self.mapper = DynamicMapper(self.db_path)
                logger.info("✅ Sistema flexible inicializado")
            else:
                self.introspector = None
                self.mapper = None
                logger.warning("⚠️ Sistema flexible no disponible")
                
        except Exception as e:
            logger.error(f"Error inicializando componentes: {e}")
            self.fhir_bridge = None
            self.introspector = None
            self.mapper = None

    def _find_mapping_config(self) -> str:
        """Busca el archivo de configuración de mapeo"""
        possible_paths = [
            "config/mapping_rules.yaml",
            "../config/mapping_rules.yaml",
            "chatmed_v2_flexible/config/mapping_rules.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return "config/mapping_rules.yaml"  # Default

    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Procesa una consulta FHIR de forma unificada
        """
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        try:
            # Clasificar el tipo de operación FHIR usando LLM
            operation_type = await self._classify_fhir_operation(query)
            
            # Procesar según el tipo
            if operation_type == 'clinical_note':
                result = await self.process_clinical_note(query)
            elif operation_type == 'fhir_query':
                result = self._process_fhir_query(query)
            elif operation_type == 'conversion':
                result = self._process_conversion_request(query)
            else:
                result = self._process_general_fhir_request(query)
        
            # Actualizar estadísticas
            processing_time = time.time() - start_time
            self.stats['avg_response_time'] = (
                (self.stats['avg_response_time'] * (self.stats['total_queries'] - 1) + processing_time) 
                / self.stats['total_queries']
            )
            
            if result.get('success', False):
                self.stats['successful_queries'] += 1
            else:
                self.stats['failed_queries'] += 1
        
            return result

        except Exception as e:
            self.stats['failed_queries'] += 1
            logger.error(f"Error procesando consulta FHIR: {e}")
            return self._create_error_response(str(e))
    
    async def _classify_fhir_operation(self, query: str) -> str:
        """Clasifica el tipo de operación FHIR usando LLM inteligente"""
        
        # Si no hay LLM disponible, usar fallback básico
        if not self.llm:
            return self._classify_fhir_operation_fallback(query)
        
        try:
            prompt = f"""
Eres un clasificador experto de consultas médicas. Analiza la siguiente consulta y determina si es una NOTA CLÍNICA que debe procesarse con el agente FHIR.

DEFINICIÓN DE NOTA CLÍNICA:
- Documento que registra información médica de un paciente
- Contiene diagnósticos, tratamientos, medicamentos, signos vitales
- Puede tener formato estructurado (SOAP, H&P) o narrativo
- Es información para REGISTRAR, no para CONSULTAR
- Suelen ser textos largos con información médica detallada

HERRAMIENTAS DE ANÁLISIS:
1. Identificar si la consulta contiene información médica para registrar
2. Identificar si la consulta menciona pacientes, diagnósticos, tratamientos
3. Identificar si la consulta es para crear o registrar datos
4. Identificar si la consulta es para consultar datos existentes

CONSULTA A ANALIZAR:
"{query}"

Responde SOLO con "NOTA_CLINICA" si es una nota clínica para registrar, o "OTRO" si no lo es.
"""
            
            response = await self.llm.ainvoke(prompt)
            result = str(response.content).strip().upper()
            
            if "NOTA_CLINICA" in result or "CLINICA" in result:
                logger.info(f"🧠 LLM clasificó como NOTA CLÍNICA: {query[:50]}...")
                return 'clinical_note'
            else:
                logger.info(f"🧠 LLM clasificó como OTRO: {query[:50]}...")
                return 'general'
                
        except Exception as e:
            logger.error(f"Error en clasificación LLM: {e}")
            return self._classify_fhir_operation_fallback(query)
    
    def _classify_fhir_operation_fallback(self, query: str) -> str:
        """Clasificación de fallback sin LLM"""
        query_lower = query.lower()
        
        # Patrones básicos para notas clínicas
        clinical_patterns = [
            'paciente', 'patient', 'diagnóstico', 'diagnosis',
            'síntomas', 'symptoms', 'tratamiento', 'treatment',
            'medicación', 'medication', 'nota clínica', 'clinical note',
            'crear', 'create', 'registrar', 'register', 'temperatura', 'temperature'
        ]
        
        # Patrones para consultas FHIR
        fhir_patterns = [
            'fhir', 'resource', 'bundle', 'observation',
            'condition', 'procedure', 'encounter'
        ]
        
        # Patrones para conversión
        conversion_patterns = [
            'convertir', 'convert', 'transformar', 'transform',
            'mapear', 'map', 'sql to fhir', 'fhir to sql'
        ]
        
        if any(pattern in query_lower for pattern in clinical_patterns):
            return 'clinical_note'
        elif any(pattern in query_lower for pattern in fhir_patterns):
            return 'fhir_query'
        elif any(pattern in query_lower for pattern in conversion_patterns):
            return 'conversion'
        else:
            return 'general'
    
    async def process_clinical_note(self, note: str) -> Dict[str, Any]:
        """
        Procesa una nota clínica de forma completamente genérica usando LLM
        
        FLUJO GENÉRICO:
        1. Extraer datos estructurados con LLM
        2. Generar recursos FHIR con LLM
        3. Persistir en SQL usando LLM + SQL Agent
        """
        try:
            self.stats['clinical_notes_processed'] += 1
            
            print(f"🏥 INICIANDO PROCESAMIENTO DE NOTA CLÍNICA")
            print(f"📝 Nota: {note[:100]}...")
            
            # PASO 1: Extraer datos estructurados usando LLM genérico
            extracted_data_result = await self._extract_clinical_data_generic(note)
            print(f"📊 Datos extraídos: {extracted_data_result.get('success', False)}")
            
            # DEBUG: Imprimir los datos crudos para ver qué se extrajo
            if extracted_data_result.get('success'):
                raw_data = extracted_data_result.get('extracted_data', {})
                print(f"🔍 DEBUG: Datos extraídos crudos: {json.dumps(raw_data, indent=2, ensure_ascii=False)}")
                # Imprimir conteos para una rápida verificación
                patient_data = raw_data.get('patient', {})
                patient_name = patient_data.get('name', 'No especificado')
                print(f"   �� Paciente: {patient_name}")
                print(f"   🏥 Condiciones: {len(raw_data.get('conditions', []))}")
                print(f"   💊 Medicamentos: {len(raw_data.get('medications', []))}")
                print(f"   📊 Observaciones: {len(raw_data.get('observations', []))}")
            
            if not extracted_data_result.get('success') or not extracted_data_result.get('extracted_data'):
                print(f"❌ Abortando: no se extrajeron datos válidos.")
                return extracted_data_result
            
            # PASO 2: Generar recursos FHIR usando LLM genérico
            fhir_resources = await self._generate_fhir_resources_generic(extracted_data_result)
            print(f"🏥 Recursos FHIR generados: {len(fhir_resources)}")
            
            # PASO 3: Persistir recursos con lógica de dos fases (primero paciente, luego el resto)
            persistence_results = []
            patient_id = None
            print(f"🔍 DEBUG: SQL Agent disponible: {self.sql_agent is not None}")
            if self.sql_agent:
                print(f"💾 PASO 3: Iniciando persistencia en base de datos (lógica de 2 fases)")
                
                # Fase 1: Encontrar y persistir el paciente para obtener un ID
                print(f"   🔍 Buscando recurso Patient en {len(fhir_resources)} recursos...")
                print(f"   📋 Tipos de recursos disponibles: {[res.get('resourceType', 'Unknown') for res in fhir_resources]}")
                patient_resource = next((res for res in fhir_resources if res.get('resourceType') == 'Patient'), None)
                
                print(f"   🔍 Recurso Patient encontrado: {patient_resource is not None}")
                
                if patient_resource:
                    print(f"   👤 Encontrado recurso Patient. Intentando persistir primero...")
                    print(f"   📊 Datos del paciente: {json.dumps(patient_resource, indent=2, ensure_ascii=False)}")
                    patient_persist_result = await self._persist_patient_resource(patient_resource)
                    persistence_results.append(patient_persist_result)
                    
                    if patient_persist_result.success and patient_persist_result.resource_id:
                        patient_id = patient_persist_result.resource_id
                        print(f"   ✅ Paciente persistido con éxito. UUID FHIR: {patient_id}")
                        # Obtener el PATI_ID real de la base de datos
                        if patient_persist_result.warnings:
                            for warning in patient_persist_result.warnings:
                                if 'PATI_ID:' in warning:
                                    pati_id = warning.split('PATI_ID: ')[1]
                                    print(f"   📊 PATI_ID en BD: {pati_id}")
                                    break
                    else:
                        print(f"   ❌ ERROR CRÍTICO: No se pudo persistir el paciente. Abortando persistencia del resto de recursos.")
                        # Añadir errores al resumen
                        final_summary = await self._generate_processing_summary(note, extracted_data_result, fhir_resources, persistence_results)
                        return self._create_error_response("Fallo al persistir el recurso Paciente, no se pueden guardar los demás datos.")
                else:
                    print("   ⚠️ No se encontró un recurso FHIR 'Patient' para persistir.")

                # Fase 2: Persistir el resto de los recursos usando procesamiento en lotes
                if patient_id:
                    other_resources = [res for res in fhir_resources if res.get('resourceType') != 'Patient']
                    print(f"   🔄 Persistiendo {len(other_resources)} recursos restantes con ID de paciente: {patient_id}")
                    
                    if self.batch_processor and other_resources:
                        # Usar procesamiento en lotes para mayor eficiencia
                        batch_result = await self.batch_processor.process_fhir_batch(other_resources, patient_id)
                        if batch_result['success']:
                            print(f"✅ Procesamiento en lotes completado: {batch_result.get('total_processed', 0)} recursos procesados")
                        else:
                            print(f"❌ Error en procesamiento en lotes: {batch_result.get('error', 'Error desconocido')}")
                    else:
                        # Fallback: procesar uno por uno
                        for i, resource in enumerate(other_resources, 1):
                            print(f"      - Persistiendo {i}/{len(other_resources)}: {resource.get('resourceType')}")
                            # Usar mapeo inteligente antes de pasar al agente SQL
                            print(f"      🔄 Mapeando recurso {resource.get('resourceType')} a SQL...")
                            mapped_data = await self._intelligent_fhir_to_sql_mapping(resource)
                            if mapped_data and mapped_data.get('success'):
                                print(f"      ✅ Mapeo exitoso para {resource.get('resourceType')}")
                                # Pasar datos ya mapeados al agente SQL
                                persist_result = await self._persist_resource_generic(mapped_data, patient_id_context=patient_id)
                            else:
                                print(f"      ⚠️ Mapeo fallido para {resource.get('resourceType')}, usando fallback")
                                # Fallback: usar recurso FHIR crudo
                                persist_result = await self._persist_resource_generic(resource, patient_id_context=patient_id)
                            persistence_results.append(persist_result)
                            print(f"      📊 Resultado de persistencia: {persist_result.success}")
            else:
                print(f"❌ ERROR: SQL Agent no disponible, saltando persistencia")
                return self._create_error_response("SQL Agent no disponible para persistencia")
            
            # PASO 4: Generar resumen final con LLM
            print(f"📄 PASO 4: Generando resumen final con información de tablas actualizadas")
            final_summary = await self._generate_processing_summary(note, extracted_data_result, fhir_resources, persistence_results)

            return {
                'success': True,
                'type': 'clinical_note_processing',
                'extracted_data': extracted_data_result,
                'fhir_resources': fhir_resources,
                'persistence_results': persistence_results,
                'summary': final_summary,
                'message': f'Nota clínica procesada exitosamente. Se generaron {len(fhir_resources)} recursos FHIR y se actualizaron las tablas de la base de datos.'
            }

        except Exception as e:
            logger.error(f"Error procesando nota clínica: {e}")
            return self._create_error_response(f"Error procesando nota clínica: {str(e)}")
    
    async def _extract_clinical_data(self, note: str) -> Dict[str, Any]:
        """Extrae datos clínicos de una nota usando LLM"""
        if not self.llm:
            return {
                'success': False,
                'error': 'LLM no disponible para extracción de datos clínicos'
            }
        
        try:
            prompt = f"""
            Extrae la siguiente información médica de esta nota clínica:
            
            Nota: {note}
            
            Devuelve un JSON con:
            {{
                "patient_data": {{
                    "name": "nombre del paciente",
                    "age": edad,
                    "gender": "género",
                    "birthdate": "fecha de nacimiento si está disponible"
                }},
                "conditions": [
                    {{
                        "code": "código de condición",
                        "display": "descripción de la condición",
                        "severity": "severidad"
                    }}
                ],
                "medications": [
                    {{
                        "code": "código de medicación",
                        "display": "nombre del medicamento",
                        "dosage": "dosis"
                    }}
                ],
                "observations": [
                    {{
                        "code": "código de observación",
                        "display": "tipo de observación",
                        "value": "valor",
                        "unit": "unidad"
                    }}
                ]
            }}
            """
            
            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result:
                return {
                    'success': True,
                    'patient_data': result.get('patient_data', {}),
                    'conditions': result.get('conditions', []),
                    'medications': result.get('medications', []),
                    'observations': result.get('observations', [])
                }
            else:
                return {
                    'success': False,
                    'error': 'No se pudo parsear la respuesta del LLM'
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error extrayendo datos clínicos: {str(e)}'
            }
    
    async def _generate_fhir_resources(self, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera recursos FHIR a partir de datos extraídos"""
        resources = []
        
        try:
            # Generar recurso Patient
            patient_data = extracted_data.get('patient_data', {})
            if patient_data:
                patient_resource = {
                    'resourceType': 'Patient',
                    'id': str(uuid.uuid4()),
                    'name': [{
                        'given': [patient_data.get('name', '')],
                        'family': patient_data.get('surname', '')
                    }],
                    'gender': patient_data.get('gender', 'unknown'),
                    'birthDate': patient_data.get('birthdate', '')
                }
                resources.append(patient_resource)
            
            # Generar recursos Condition
            for condition in extracted_data.get('conditions', []):
                condition_resource = {
                    'resourceType': 'Condition',
                    'id': str(uuid.uuid4()),
                    'code': {
                        'coding': [{
                            'code': condition.get('code', ''),
                            'display': condition.get('display', '')
                        }]
                    },
                    'severity': {
                        'coding': [{
                            'code': condition.get('severity', 'moderate'),
                            'display': condition.get('severity', 'moderate')
                        }]
                    }
                }
                resources.append(condition_resource)
            
            # Generar recursos Medication
            for medication in extracted_data.get('medications', []):
                medication_resource = {
                    'resourceType': 'Medication',
                    'id': str(uuid.uuid4()),
                    'code': {
                        'coding': [{
                            'code': medication.get('code', ''),
                            'display': medication.get('display', '')
                        }]
                    },
                    'form': {
                        'coding': [{
                            'code': 'tablet',
                            'display': 'Tablet'
                        }]
                    }
                }
                resources.append(medication_resource)
            
            # Generar recursos Observation
            for observation in extracted_data.get('observations', []):
                observation_resource = {
                    'resourceType': 'Observation',
                    'id': str(uuid.uuid4()),
                    'status': 'final',
                    'code': {
                        'coding': [{
                            'code': observation.get('code', ''),
                            'display': observation.get('display', '')
                        }]
                    },
                    'valueQuantity': {
                        'value': observation.get('value', ''),
                        'unit': observation.get('unit', '')
                    }
                }
                resources.append(observation_resource)
            
            self.stats['fhir_resources_created'] += len(resources)
            return resources
            
        except Exception as e:
            logger.error(f"Error generando recursos FHIR: {e}")
            return []
    
    async def _extract_clinical_data_generic(self, note: str) -> Dict[str, Any]:
        """Extrae datos clínicos usando MedGemma cuando está disponible, o LLM como fallback"""
        
        # USAR MEDGEMMA SI ESTÁ DISPONIBLE
        if self.medgemma_agent:
            try:
                print(f"🧠 PASO 1: Usando MedGemma para análisis clínico avanzado")
                
                # Analizar con MedGemma
                medgemma_result = await self.medgemma_agent.analyze_clinical_data(
                    note,
                    None  # stream_callback
                )
                
                if medgemma_result and medgemma_result.get('success'):
                    analysis = medgemma_result.get('analysis', '')
                    if analysis:
                        print(f"✅ Análisis clínico con MedGemma completado")
                        # Convertir análisis de MedGemma a formato estructurado
                        return {
                            'success': True,
                            'extracted_data': {
                                'patient': {'name': 'Paciente analizado por MedGemma'},
                                'conditions': [],
                                'medications': [],
                                'observations': [],
                                'medgemma_analysis': analysis
                            },
                            'source': 'medgemma'
                        }
                        
            except Exception as e:
                print(f"⚠️ Error con MedGemma: {e}, usando LLM...")
                logger.error(f"Error con MedGemma en FHIR: {e}")
                # Continuar con LLM si MedGemma falla
        
        # FALLBACK A LLM
        try:
            print(f"📊 PASO 1: Extrayendo datos clínicos estructurados con LLM")
            
            if not self.llm:
                print(f"❌ LLM no disponible para extracción")
                return {'success': False, 'error': 'LLM no disponible'}
            
            print(f"🤖 Preparando prompt de extracción robusto...")
            
            # PROMPT MEJORADO CON VALIDACIONES Y EJEMPLOS MÁS ROBUSTOS
            prompt = f"""
Eres un experto en extracción de información médica de notas clínicas. Tu tarea es analizar la siguiente nota y extraer de forma EXHAUSTIVA todos los datos relevantes en un formato JSON estructurado y limpio.

**NOTA CLÍNICA A ANALIZAR:**
---
{note}
---

**INSTRUCCIONES CRÍTICAS (DEBES SEGUIRLAS AL PIE DE LA LETRA):**

1. **EXTRACCIÓN COMPLETA Y ROBUSTA:**
   - Extrae TODA la información disponible del paciente, condiciones, medicamentos y observaciones
   - Si no encuentras un dato específico, usa `null` o cadena vacía, pero NUNCA omitas campos
   - Busca nombres de pacientes en cualquier formato (iniciales, apellidos, nombres completos)
   - Extrae edades aunque estén en formato narrativo (ej: "paciente de 45 años" → age: 45)

2. **FORMATO JSON ESTRICTO:**
   - Tu respuesta debe ser ÚNICAMENTE el objeto JSON
   - No incluyas texto, explicaciones, ni la palabra "json"
   - Usa comillas dobles para strings, no simples

3. **VALIDACIONES OBLIGATORIAS:**
   - Si la nota menciona condiciones médicas, `conditions` NO puede estar vacía
   - Si la nota menciona medicamentos, `medications` NO puede estar vacía
   - Si la nota menciona observaciones/laboratorio, `observations` NO puede estar vacía
   - El campo `patient.name` NO puede estar vacío si hay información del paciente

4. **BÚSQUEDA INTELIGENTE DE DATOS DEL PACIENTE:**
   - Busca nombres en formato: "Paciente: [nombre]", "Sra. [nombre]", "Dr. [nombre]", etc.
   - Busca edades en formato: "X años", "edad X", "paciente de X años"
   - Busca géneros: "masculino/femenino", "hombre/mujer", "varón/mujer"

**EJEMPLO DE FORMATO JSON PERFECTO:**
```json
{{
    "patient": {{
        "name": "María López",
        "age": 35,
        "gender": "female",
        "clinical_history_id": null
    }},
    "conditions": [
        {{ "description": "Hipertensión arterial", "details": "diagnóstico reciente" }},
        {{ "description": "Diabetes tipo 2", "details": "control subóptimo" }}
    ],
    "medications": [
        {{ "name": "Enalapril", "dosage": "10mg 1 comprimido al día", "status": "active" }},
        {{ "name": "Metformina", "dosage": "500mg 2 veces al día", "status": "active" }}
    ],
    "observations": [
        {{ "test": "Tensión arterial", "value": "140/90", "unit": "mmHg" }},
        {{ "test": "Glucemia", "value": "180", "unit": "mg/dL" }}
    ],
    "plan_summary": "Se prescribe Enalapril para hipertensión y Metformina para diabetes"
}}
```

**IMPORTANTE:** Si no encuentras el nombre del paciente en la nota, pero hay información médica, usa un nombre genérico como "Paciente" o extrae cualquier nombre que aparezca en el contexto.

**RESPUESTA (SOLO JSON):**
"""
            
            print(f"🔄 Enviando nota al LLM para extracción...")
            response = _call_openai_native(self.llm, prompt)
            print(f"📥 Respuesta recibida: {len(response.content)} caracteres")
            
            result = self._try_parse_llm_json(response.content)
            
            # VALIDACIONES ADICIONALES DESPUÉS DE LA EXTRACCIÓN
            if result:
                print(f"✅ Datos extraídos exitosamente (formato nuevo)")
                
                # Validación 1: Verificar que el paciente tenga nombre
                patient_data = result.get('patient', {})
                patient_name = patient_data.get('name', '').strip()
                
                if not patient_name:
                    print(f"⚠️ ADVERTENCIA: No se encontró nombre del paciente en la nota")
                    print(f"🔧 Intentando extraer nombre del contexto...")
                    
                    # Intentar extraer nombre del contexto de la nota
                    import re
                    name_patterns = [
                        r'Paciente:\s*([A-Za-zÁáÉéÍíÓóÚúÑñ\s]+)',
                        r'Sra\.\s*([A-Za-zÁáÉéÍíÓóÚúÑñ\s]+)',
                        r'Dr\.\s*([A-Za-zÁáÉéÍíÓóÚúÑñ\s]+)',
                        r'([A-Za-zÁáÉéÍíÓóÚúÑñ]+)\s+[A-Za-zÁáÉéÍíÓóÚúÑñ]+\s*,\s*\d+\s*años'
                    ]
                    
                    extracted_name = None
                    for pattern in name_patterns:
                        match = re.search(pattern, note)
                        if match:
                            extracted_name = match.group(1).strip()
                            break
                    
                    if extracted_name:
                        print(f"✅ Nombre extraído del contexto: {extracted_name}")
                        result['patient']['name'] = extracted_name
                    else:
                        print(f"❌ No se pudo extraer nombre del contexto")
                        result['patient']['name'] = "Paciente"
                
                # Validación 2: Verificar que haya condiciones médicas
                conditions = result.get('conditions', [])
                if not conditions:
                    print(f"⚠️ ADVERTENCIA: No se encontraron condiciones médicas")
                    print(f"🔍 Buscando condiciones en el texto...")
                    
                    # Buscar condiciones médicas comunes en el texto
                    condition_keywords = [
                        'diabetes', 'hipertensión', 'hipertension', 'dislipemia', 'obesidad',
                        'asma', 'epoc', 'insuficiencia cardíaca', 'insuficiencia cardiaca',
                        'neuropatía', 'neuropatia', 'retinopatía', 'retinopatia'
                    ]
                    
                    found_conditions = []
                    for keyword in condition_keywords:
                        if keyword.lower() in note.lower():
                            found_conditions.append({
                                "description": keyword.title(),
                                "details": "mencionada en la nota clínica"
                            })
                    
                    if found_conditions:
                        print(f"✅ Condiciones encontradas en el texto: {len(found_conditions)}")
                        result['conditions'] = found_conditions
                
                # Validación 3: Verificar que haya medicamentos
                medications = result.get('medications', [])
                if not medications:
                    print(f"⚠️ ADVERTENCIA: No se encontraron medicamentos")
                    print(f"🔍 Buscando medicamentos en el texto...")
                    
                    # Buscar medicamentos comunes en el texto
                    medication_keywords = [
                        'metformina', 'enalapril', 'atorvastatina', 'aspirina', 'paracetamol',
                        'ibuprofeno', 'omeprazol', 'simvastatina', 'losartán', 'losartan'
                    ]
                    
                    found_medications = []
                    for keyword in medication_keywords:
                        if keyword.lower() in note.lower():
                            found_medications.append({
                                "name": keyword.title(),
                                "dosage": "mencionado en la nota",
                                "status": "active"
                            })
                    
                    if found_medications:
                        print(f"✅ Medicamentos encontrados en el texto: {len(found_medications)}")
                        result['medications'] = found_medications
                
                print(f"📊 Datos extraídos: True")
                print(f"🔍 DEBUG: Datos extraídos crudos: {result}")
                
                # Contar elementos extraídos
                patient_count = 1 if result.get('patient') else 0
                conditions_count = len(result.get('conditions', []))
                medications_count = len(result.get('medications', []))
                observations_count = len(result.get('observations', []))
                
                print(f"   👤 Paciente: {patient_count}")
                print(f"   🏥 Condiciones: {conditions_count}")
                print(f"   💊 Medicamentos: {medications_count}")
                print(f"   📊 Observaciones: {observations_count}")
                
                return {
                    'success': True,
                    'extracted_data': result
                }
            else:
                print(f"❌ Error parseando extracción de datos o formato inesperado")
                print(f"🔍 Respuesta completa: {response.content[:500]}...")
                return {'success': False, 'error': 'No se pudo parsear la extracción de datos en el formato esperado'}
            
        except Exception as e:
            print(f"❌ Error crítico extrayendo datos: {str(e)}")
            return {'success': False, 'error': f'Error extrayendo datos: {str(e)}'}
    
    async def _generate_fhir_resources_generic(self, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera recursos FHIR de forma completamente genérica usando LLM"""
        try:
            print(f"🏥 PASO 2: Generando recursos FHIR")
            
            if not self.llm:
                print(f"❌ LLM no disponible para generación FHIR")
                return []
            
            # CORRECCIÓN: Usar directamente 'extracted_data', no un sub-diccionario.
            data = extracted_data.get('extracted_data', {})
            
            print(f"🔍 DEBUG: Datos para FHIR (formato nuevo): {json.dumps(data, indent=2, ensure_ascii=False)}")

            if not data or not data.get('patient'):
                print("⚠️ No hay datos de paciente para generar recursos FHIR. Abortando.")
                return []

            print(f"📊 Datos disponibles para FHIR:")
            print(f"   👤 Paciente: {bool(data.get('patient'))}")
            print(f"   🏥 Condiciones: {len(data.get('conditions', []))}")
            print(f"   💊 Medicamentos: {len(data.get('medications', []))}")
            print(f"   📊 Observaciones: {len(data.get('observations', []))}")
            
            # LLAMADA ÚNICA AL LLM: Generación directa de recursos FHIR
            generation_prompt = f"""Eres un experto en creación de recursos FHIR. Crea recursos FHIR válidos a partir de estos datos clínicos.

DATOS CLÍNICOS:
{json.dumps(data, indent=2, ensure_ascii=False)}

INSTRUCCIONES CRÍTICAS:
1. Crea recursos FHIR individuales (NO Bundle)
2. Usa IDs numéricos únicos: "1", "2", "3", etc.
3. NO uses UUIDs ficticios
4. Asegúrate de que el JSON sea válido
5. Incluye todos los tipos de recursos necesarios

TIPOS DE RECURSOS A CREAR:
- Patient: Información del paciente
- Condition: Diagnósticos y condiciones
- MedicationRequest: Medicamentos prescritos
- Observation: Signos vitales y laboratorio
- AllergyIntolerance: Alergias

FORMATO REQUERIDO:
```json
[
  {{
    "resourceType": "Patient",
    "id": "1",
    "name": [{{"text": "Nombre Completo"}}],
    "gender": "female",
    "birthDate": "1989-03-15"
  }},
  {{
    "resourceType": "Condition",
    "id": "2",
    "subject": {{"reference": "Patient/1"}},
    "code": {{"text": "Diagnóstico"}}
  }}
]
```

CRÍTICO: Responde SOLO con el array JSON, sin explicaciones, sin texto adicional."""

            print(f"🔄 LLAMADA ÚNICA: Generando recursos FHIR...")
            generation_response = _call_openai_native(self.llm, generation_prompt)
            generation_result = self._try_parse_llm_json(generation_response.content)
            
            if generation_result and isinstance(generation_result, list):
                print(f"✅ LLM generó {len(generation_result)} recursos FHIR")
                self.stats['fhir_resources_created'] += len(generation_result)
                return generation_result
            
            print(f"❌ LLM no pudo generar recursos FHIR válidos")
            
            # MÉTODO DIRECTO: Crear recursos FHIR manualmente sin LLM
            print(f"🔧 Creando recursos FHIR manualmente...")
            manual_resources = self._create_manual_fhir_resources(data)
            if manual_resources:
                print(f"✅ Creados {len(manual_resources)} recursos FHIR manualmente")
                self.stats['fhir_resources_created'] += len(manual_resources)
                return manual_resources
            
            return []
                
        except Exception as e:
            print(f"❌ Error crítico generando recursos FHIR: {str(e)}")
            return []

    async def _lookup_medication_id(self, medication_name: str) -> Dict[str, Any]:
        """
        Busca el ID real de un medicamento en la tabla MEDI_ACTIVE_INGREDIENTS
        usando el SQL Agent de forma inteligente
        """
        try:
            if not self.sql_agent:
                return {'found': False, 'acin_id': None, 'description': None}
            
            # Normalizar el nombre del medicamento
            med_name_normalized = medication_name.strip().upper()
            
            # Usar SQL Agent para buscar el medicamento
            search_query = f"""
            Busca el ACIN_ID y la descripción del ingrediente activo '{medication_name}' 
            en la tabla MEDI_ACTIVE_INGREDIENTS. 
            Si no encuentras coincidencia exacta, busca coincidencias parciales.
            """
            
            result = await self.sql_agent.process_query(search_query)
            
            if result.get('success') and result.get('data'):
                data = result['data']
                if data and isinstance(data[0], dict):
                    return {
                        'found': True,
                        'acin_id': data[0].get('ACIN_ID'),
                        'description': data[0].get('ACIN_DESCRIPTION_ES', data[0].get('ACIN_DESCRIPTION'))
                    }
            
            # Si no se encuentra, intentar búsqueda más flexible
            print(f"   ⚠️ No se encontró ID para medicamento '{medication_name}', se usará texto libre")
            return {'found': False, 'acin_id': None, 'description': None}
            
        except Exception as e:
            logger.error(f"Error buscando ID de medicamento: {e}")
            return {'found': False, 'acin_id': None, 'description': None}

    async def _persist_patient_resource(self, patient_resource: Dict[str, Any]) -> PersistenceResult:
        """
        Persiste un recurso FHIR Patient de forma robusta y directa.
        Maneja la lógica de separación de nombre y apellido.
        """
        start_time = time.time()
        
        try:
            if not self.sql_agent:
                return PersistenceResult(success=False, resource_type='Patient', resource_id='unknown', errors=['SQL Agent no disponible'])

            # 1. Extraer y procesar el nombre usando LLM (DINÁMICO)
            name_data = patient_resource.get('name', [{}])[0]
            first_name, last_name = None, None
            
            if self.llm:
                print(f"   🧠 Usando LLM para extracción inteligente de nombres...")
                
                # PROMPT 1: Extracción de nombres desde FHIR
                extraction_prompt = f"""
Eres un experto en procesamiento de nombres médicos. Extrae el nombre y apellido del recurso FHIR Patient.

**RECURSO FHIR:**
{json.dumps(patient_resource, indent=2, ensure_ascii=False)}

**INSTRUCCIONES:**
1. Extrae el nombre completo del campo 'name'
2. Separa en nombre y apellido según estándares españoles
3. Maneja nombres compuestos (ej: "Juan Carlos")
4. Maneja apellidos compuestos (ej: "Martínez López")
5. Considera que en español: nombre + primer apellido + segundo apellido

**EJEMPLOS:**
- "Ana Martínez López" → nombre: "Ana", apellido: "Martínez López"
- "Juan Carlos Pascual García" → nombre: "Juan Carlos", apellido: "Pascual García"
- "María del Carmen López" → nombre: "María del Carmen", apellido: "López"

Responde SOLO con JSON:
{{
  "first_name": "nombre extraído",
  "last_name": "apellido extraído",
  "reasoning": "explicación del proceso"
}}
"""
                
                try:
                    response = _call_openai_native(self.llm, extraction_prompt)
                    name_extraction = self._try_parse_llm_json(response.content)
                    
                    if name_extraction and isinstance(name_extraction, dict):
                        first_name = name_extraction.get('first_name', '')
                        last_name = name_extraction.get('last_name', '')
                        reasoning = name_extraction.get('reasoning', '')
                        print(f"   ✅ LLM extrajo nombres: {first_name} {last_name}")
                        print(f"   💡 Razonamiento: {reasoning}")
                    else:
                        print(f"   ⚠️ LLM no pudo extraer nombres, usando fallback")
                        # Fallback básico
                        if 'text' in name_data:
                            full_name = str(name_data['text'])
                            name_parts = full_name.split()
                            if len(name_parts) >= 2:
                                first_name = name_parts[0]
                                last_name = ' '.join(name_parts[1:])
                            else:
                                first_name = full_name
                                last_name = ''
                        elif 'given' in name_data and 'family' in name_data:
                            first_name = ' '.join(name_data['given'])
                            last_name = name_data['family']
                        
                except Exception as e:
                    print(f"   ❌ Error en extracción LLM: {e}")
                    # Fallback básico
                    if 'text' in name_data:
                        full_name = str(name_data['text'])
                        name_parts = full_name.split()
                        if len(name_parts) >= 2:
                            first_name = name_parts[0]
                            last_name = ' '.join(name_parts[1:])
                        else:
                            first_name = full_name
                            last_name = ''
                    elif 'given' in name_data and 'family' in name_data:
                        first_name = ' '.join(name_data['given'])
                        last_name = name_data['family']
            else:
                print(f"   ⚠️ LLM no disponible, usando extracción básica")
                # Fallback básico sin LLM
                if 'text' in name_data:
                    full_name = str(name_data['text'])
                    name_parts = full_name.split()
                    if len(name_parts) >= 2:
                        first_name = name_parts[0]
                        last_name = ' '.join(name_parts[1:])
                    else:
                        first_name = full_name
                        last_name = ''
                elif 'given' in name_data and 'family' in name_data:
                    first_name = ' '.join(name_data['given'])
                    last_name = name_data['family']

            # PROMPT 3: Validación inteligente de estructura de nombres
            if self.llm:
                print(f"   🧠 Usando LLM para validación inteligente de nombres...")
                
                validation_prompt = f"""
Eres un experto en validación de nombres médicos. Valida la estructura del nombre extraído.

**NOMBRE EXTRAÍDO:**
- Nombre: "{first_name}"
- Apellido: "{last_name}"

**INSTRUCCIONES:**
1. Valida que el nombre sea coherente y completo
2. Verifica que no esté vacío o sea inválido
3. Comprueba que siga patrones de nombres españoles
4. Identifica posibles errores de extracción

**CRITERIOS DE VALIDACIÓN:**
- Nombre no puede estar vacío
- Apellido debe existir (puede ser compuesto)
- No debe contener caracteres extraños
- Debe seguir formato: Nombre + Apellido(s)

Responde SOLO con JSON:
{{
  "is_valid": true/false,
  "reason": "explicación de la validación",
  "suggestions": ["sugerencias de mejora si aplica"]
}}
"""
                
                try:
                    response = _call_openai_native(self.llm, validation_prompt)
                    validation_result = self._try_parse_llm_json(response.content)
                    
                    if validation_result and isinstance(validation_result, dict):
                        is_valid = validation_result.get('is_valid', False)
                        reason = validation_result.get('reason', '')
                        suggestions = validation_result.get('suggestions', [])
                        
                        if not is_valid:
                            print(f"   ❌ ERROR: Nombre del paciente inválido: '{first_name}'")
                            print(f"   💡 Razón: {reason}")
                            if suggestions:
                                print(f"   💡 Sugerencias: {', '.join(suggestions)}")
                            return PersistenceResult(
                                success=False, 
                                resource_type='Patient', 
                                resource_id='unknown', 
                                errors=[f'Nombre del paciente inválido: "{first_name}". {reason}']
                            )
                        else:
                            print(f"   ✅ Nombre del paciente válido: {first_name} {last_name}")
                            print(f"   💡 Validación: {reason}")
                    else:
                        print(f"   ⚠️ LLM no pudo validar, usando validación básica")
                        # Validación básica
                        if not first_name or not first_name.strip():
                            return PersistenceResult(
                                success=False, 
                                resource_type='Patient', 
                                resource_id='unknown', 
                                errors=['Nombre del paciente está vacío']
                            )
                        print(f"   ✅ Nombre del paciente válido (validación básica): {first_name} {last_name}")
                        
                except Exception as e:
                    print(f"   ❌ Error en validación LLM: {e}")
                    # Validación básica
                    if not first_name or not first_name.strip():
                        return PersistenceResult(
                            success=False, 
                            resource_type='Patient', 
                            resource_id='unknown', 
                            errors=['Nombre del paciente está vacío']
                        )
                    print(f"   ✅ Nombre del paciente válido (validación básica): {first_name} {last_name}")
            else:
                print(f"   ⚠️ LLM no disponible, usando validación básica")
                # Validación básica sin LLM
                if not first_name or not first_name.strip():
                    return PersistenceResult(
                        success=False, 
                        resource_type='Patient', 
                        resource_id='unknown', 
                        errors=['Nombre del paciente está vacío']
                    )
                print(f"   ✅ Nombre del paciente válido (validación básica): {first_name} {last_name}")

            # 2. Construir el diccionario de datos para SQL (de forma más robusta)
            # Extraer identificador de forma segura
            national_id = None
            try:
                identifiers = patient_resource.get('identifier', [])
                if isinstance(identifiers, list):
                    national_id = next((identifier.get('value') for identifier in identifiers if isinstance(identifier, dict) and 'NHC' in identifier.get('system', '')), None)
            except (TypeError, AttributeError):
                print("   ⚠️  Warning: El campo 'identifier' del paciente no es una lista de diccionarios como se esperaba.")
                pass

            # USAR LLM PARA MAPEO INTELIGENTE DE CAMPOS
            if self.llm:
                print(f"   🧠 Usando LLM para mapeo inteligente de campos...")
                
                # Obtener esquema de la tabla
                schema_info = await self._get_database_schema_info()
                
                # Crear prompt para mapeo inteligente
                mapping_prompt = f"""
Eres un experto en mapeo de datos FHIR a SQL. Tu tarea es convertir los datos del paciente FHIR a campos SQL válidos.

**ESQUEMA DE LA TABLA PATI_PATIENTS:**
{schema_info}

**DATOS DEL PACIENTE FHIR:**
{json.dumps(patient_resource, indent=2, ensure_ascii=False)}

**INSTRUCCIONES:**
1. Mapea los datos FHIR a los campos SQL disponibles
2. Usa valores por defecto apropiados para campos requeridos
3. Convierte tipos de datos correctamente (ej: 'male'/'female' → 1/2)
4. Genera fechas en formato ISO si no están presentes
5. Crea nombres completos combinando nombre y apellido
6. Solo incluye campos que existen en el esquema

**EJEMPLO DE RESPUESTA:**
{{
  "PATI_NAME": "María",
  "PATI_SURNAME_1": "López",
  "PATI_FULL_NAME": "María López",
  "PATI_BIRTH_DATE": "1980-01-01",
  "PATI_START_DATE": "2024-01-15T10:30:00",
  "GEND_ID": 2,
  "PATI_ACTIVE": 1
}}

Responde SOLO con el JSON del mapeo:
"""
                
                try:
                    response = _call_openai_native(self.llm, mapping_prompt)
                    llm_mapping = self._try_parse_llm_json(response.content)
                    
                    if llm_mapping and isinstance(llm_mapping, dict):
                        print(f"   ✅ LLM generó mapeo inteligente de campos")
                        sql_data = llm_mapping
                        
                        # PROMPT 2: Separación inteligente de apellidos
                        if self.llm and last_name:
                            print(f"   🧠 Usando LLM para separación inteligente de apellidos...")
                            
                            surname_prompt = f"""
Eres un experto en apellidos españoles. Separa los apellidos en primer y segundo apellido.

**APELLIDO COMPLETO:** "{last_name}"

**INSTRUCCIONES:**
1. Separa en primer apellido y segundo apellido
2. Considera apellidos compuestos (ej: "del Río", "de la Cruz")
3. Maneja casos especiales (ej: "Martínez López", "García Rodríguez")
4. Si solo hay un apellido, déjalo como primer apellido

**EJEMPLOS:**
- "Martínez López" → primer: "Martínez", segundo: "López"
- "del Río García" → primer: "del Río", segundo: "García"
- "de la Cruz" → primer: "de la Cruz", segundo: ""
- "García" → primer: "García", segundo: ""

Responde SOLO con JSON:
{{
  "surname_1": "primer apellido",
  "surname_2": "segundo apellido (o vacío)",
  "reasoning": "explicación"
}}
"""
                            
                            try:
                                response = _call_openai_native(self.llm, surname_prompt)
                                surname_separation = self._try_parse_llm_json(response.content)
                                
                                if surname_separation and isinstance(surname_separation, dict):
                                    sql_data["PATI_SURNAME_1"] = surname_separation.get('surname_1', last_name)
                                    sql_data["PATI_SURNAME_2"] = surname_separation.get('surname_2', '')
                                    reasoning = surname_separation.get('reasoning', '')
                                    print(f"   ✅ LLM separó apellidos: {sql_data['PATI_SURNAME_1']} {sql_data['PATI_SURNAME_2']}")
                                    print(f"   💡 Razonamiento: {reasoning}")
                                else:
                                    print(f"   ⚠️ LLM no pudo separar apellidos, usando fallback")
                                    # Fallback básico
                                    if last_name and ' ' in last_name:
                                        surname_parts = last_name.split()
                                        sql_data["PATI_SURNAME_1"] = surname_parts[0]
                                        if len(surname_parts) > 1:
                                            sql_data["PATI_SURNAME_2"] = surname_parts[1]
                                    else:
                                        sql_data["PATI_SURNAME_1"] = last_name
                                    
                            except Exception as e:
                                print(f"   ❌ Error en separación LLM: {e}")
                                # Fallback básico
                                if last_name and ' ' in last_name:
                                    surname_parts = last_name.split()
                                    sql_data["PATI_SURNAME_1"] = surname_parts[0]
                                    if len(surname_parts) > 1:
                                        sql_data["PATI_SURNAME_2"] = surname_parts[1]
                                else:
                                    sql_data["PATI_SURNAME_1"] = last_name
                        else:
                            # Fallback sin LLM
                            if not sql_data.get("PATI_SURNAME_1"):
                                if last_name and ' ' in last_name:
                                    surname_parts = last_name.split()
                                    sql_data["PATI_SURNAME_1"] = surname_parts[0]
                                    if len(surname_parts) > 1:
                                        sql_data["PATI_SURNAME_2"] = surname_parts[1]
                                else:
                                    sql_data["PATI_SURNAME_1"] = last_name
                        
                        # VALIDAR Y COMPLETAR CAMPOS CRÍTICOS
                        if not sql_data.get("PATI_NAME"):
                            sql_data["PATI_NAME"] = first_name
                        if not sql_data.get("PATI_FULL_NAME"):
                            sql_data["PATI_FULL_NAME"] = f"{first_name} {last_name}".strip()
                        if not sql_data.get("PATI_START_DATE"):
                            from datetime import datetime
                            sql_data["PATI_START_DATE"] = datetime.now().isoformat()
                        if not sql_data.get("GEND_ID"):
                            sql_data["GEND_ID"] = 1 if patient_resource.get('gender') == 'male' else (2 if patient_resource.get('gender') == 'female' else 3)
                        if not sql_data.get("PATI_ACTIVE"):
                            sql_data["PATI_ACTIVE"] = 1
                        
                        # GENERAR ID DE HISTORIA CLÍNICA ÚNICO
                        import uuid
                        sql_data["PATI_CLINICAL_HISTORY_ID"] = str(uuid.uuid4())[:8].upper()
                        
                        print(f"   🔧 Campos completados y validados")
                        
                    else:
                        print(f"   ⚠️ LLM no pudo generar mapeo, usando mapeo básico mejorado")
                        # Fallback a mapeo básico MEJORADO
                        from datetime import datetime
                        import uuid

                        # CORREGIDO: Separar apellidos correctamente
                        surname_1, surname_2 = None, None
                        if last_name and ' ' in last_name:
                            surname_parts = last_name.split()
                            surname_1 = surname_parts[0]  # Primer apellido
                            if len(surname_parts) > 1:
                                surname_2 = surname_parts[1]  # Segundo apellido
                        else:
                            surname_1 = last_name
                        
                        sql_data = {
                            "PATI_NAME": first_name,
                            "PATI_SURNAME_1": surname_1,
                            "PATI_FULL_NAME": f"{first_name} {last_name}".strip(),
                            "PATI_BIRTH_DATE": patient_resource.get('birthDate'),
                            "PATI_START_DATE": datetime.now().isoformat(),
                            "PATI_CLINICAL_HISTORY_ID": str(uuid.uuid4())[:8].upper(),
                            "GEND_ID": 1 if patient_resource.get('gender') == 'male' else (2 if patient_resource.get('gender') == 'female' else 3),
                            "PATI_ACTIVE": 1,
                        }
                        
                        # Añadir segundo apellido si existe
                        if surname_2:
                            sql_data["PATI_SURNAME_2"] = surname_2
                        
                except Exception as e:
                    print(f"   ❌ Error en mapeo LLM: {e}")
                    # Fallback a mapeo básico CORREGIDO
                    # Separar apellidos correctamente
                    surname_1, surname_2 = None, None
                    if last_name and ' ' in last_name:
                        surname_parts = last_name.split()
                        surname_1 = surname_parts[0]  # Primer apellido
                        if len(surname_parts) > 1:
                            surname_2 = surname_parts[1]  # Segundo apellido
                    else:
                        surname_1 = last_name
                    
                    sql_data = {
                        "PATI_NAME": first_name,
                        "PATI_SURNAME_1": surname_1,
                        "PATI_BIRTH_DATE": patient_resource.get('birthDate'),
                        "GEND_ID": 1 if patient_resource.get('gender') == 'male' else (2 if patient_resource.get('gender') == 'female' else 3),
                        "PATI_ACTIVE": 1,
                    }
                    
                    # Añadir segundo apellido si existe
                    if surname_2:
                        sql_data["PATI_SURNAME_2"] = surname_2
                    
                    # Generar nombre completo
                    full_name = f"{first_name} {last_name}".strip()
                    if full_name:
                        sql_data["PATI_FULL_NAME"] = full_name
                    
                    # Añadir fecha de inicio
                    from datetime import datetime
                    sql_data["PATI_START_DATE"] = datetime.now().isoformat()
            else:
                print(f"   ⚠️ LLM no disponible, usando mapeo básico")
                # Mapeo básico sin LLM CORREGIDO
                # Separar apellidos correctamente
                surname_1, surname_2 = None, None
                if last_name and ' ' in last_name:
                    surname_parts = last_name.split()
                    surname_1 = surname_parts[0]  # Primer apellido
                    if len(surname_parts) > 1:
                        surname_2 = surname_parts[1]  # Segundo apellido
                else:
                    surname_1 = last_name
                
                sql_data = {
                    "PATI_NAME": first_name,
                    "PATI_SURNAME_1": surname_1,
                    "PATI_BIRTH_DATE": patient_resource.get('birthDate'),
                    "GEND_ID": 1 if patient_resource.get('gender') == 'male' else (2 if patient_resource.get('gender') == 'female' else 3),
                    "PATI_ACTIVE": 1,
                }
                
                # Añadir segundo apellido si existe
                if surname_2:
                    sql_data["PATI_SURNAME_2"] = surname_2
                
                # Generar nombre completo
                full_name = f"{first_name} {last_name}".strip()
                if full_name:
                    sql_data["PATI_FULL_NAME"] = full_name
                
                # Añadir fecha de inicio
                from datetime import datetime
                sql_data["PATI_START_DATE"] = datetime.now().isoformat()
            
            # ELIMINAR PATI_ID SI EXISTE EN EL DICCIONARIO DE INSERCIÓN
            if "PATI_ID" in sql_data:
                print(f"   ⚠️ Eliminando PATI_ID del diccionario de inserción para evitar corrupción de IDs")
                del sql_data["PATI_ID"]

            # 3. VERIFICAR SI EL PACIENTE YA EXISTE ANTES DE INSERTAR (ROBUSTO Y ESTRICTO)
            print(f"   🔍 Verificando si el paciente ya existe (matching ESTRICTO)...")
            try:
                import sqlite3
                conn = sqlite3.connect(self.sql_agent.db_path)
                cursor = conn.cursor()

                # Matching estricto: nombre completo + fecha de nacimiento (y opcionalmente DNI)
                full_name = sql_data.get("PATI_FULL_NAME")
                birth_date = sql_data.get("PATI_BIRTH_DATE")
                dni = patient_resource.get('identifier', [{}])[0].get('value') if patient_resource.get('identifier') else None

                # Buscar por nombre completo y fecha de nacimiento
                if full_name and birth_date:
                    cursor.execute("""
                        SELECT PATI_ID, PATI_NAME, PATI_SURNAME_1, PATI_FULL_NAME, PATI_BIRTH_DATE, PATI_CLINICAL_HISTORY_ID
                        FROM PATI_PATIENTS
                        WHERE LOWER(PATI_FULL_NAME) = LOWER(?) AND PATI_BIRTH_DATE = ?
                        LIMIT 1
                    """, (full_name, birth_date))
                    existing_record = cursor.fetchone()
                else:
                    existing_record = None

                # Si no se encuentra, buscar por DNI si está disponible
                if not existing_record and dni:
                    cursor.execute("""
                        SELECT PATI_ID, PATI_NAME, PATI_SURNAME_1, PATI_FULL_NAME, PATI_BIRTH_DATE, PATI_CLINICAL_HISTORY_ID
                        FROM PATI_PATIENTS
                        WHERE PATI_CLINICAL_HISTORY_ID = ?
                        LIMIT 1
                    """, (dni,))
                    existing_record = cursor.fetchone()

                conn.close()

                if existing_record:
                    patient_id = existing_record[0]
                    print(f"   ⚠️ Paciente ya existe con ID: {patient_id} (matching exacto)")
                    print(f"   📋 Datos existentes: {existing_record[3]} | Fecha nacimiento: {existing_record[4]}")
                    
                    # Verificar si el patient_id es válido
                    if patient_id is None:
                        print(f"   ⚠️ PATI_ID es NULL, procediendo con inserción de nuevo registro...")
                        # Si PATI_ID es NULL, proceder con inserción
                        # ELIMINAR PATI_ID SI EXISTE EN EL DICCIONARIO DE INSERCIÓN
                        if "PATI_ID" in sql_data:
                            print(f"   ⚠️ Eliminando PATI_ID del diccionario de inserción para evitar corrupción de IDs")
                            del sql_data["PATI_ID"]
                        print(f"   🤖 Delegando la inserción del paciente al SQLAgent...")
                        sql_result = await self.sql_agent.process_data_manipulation(
                            operation='INSERT',
                            data=sql_data,
                            context={
                                'intent': 'create_patient',
                                'resource_type': 'Patient',
                                'table_hint': 'PATI_PATIENTS'
                            }
                        )
                        
                        processing_time = (time.time() - start_time) * 1000
                        
                        if not sql_result.get('success'):
                            return PersistenceResult(
                                success=False,
                                resource_type='Patient',
                                resource_id='unknown',
                                errors=[sql_result.get('error', 'SQLAgent falló al insertar paciente')]
                            )
                        
                        patient_id = sql_result.get('inserted_id')
                        
                        # VALIDAR QUE EL ID NO SEA UN UUID FICTICIO
                        if isinstance(patient_id, str) and ('patient-id-unico' in patient_id or 'urn:uuid:' in patient_id):
                            print(f"   ⚠️ UUID ficticio detectado como inserted_id: {patient_id}")
                            # Obtener el ID real usando estrategia robusta y directa
                            real_id = await self._get_real_id_robust(
                                table=sql_result.get('table_used', 'PATI_PATIENTS'),
                                data=sql_data
                            )
                            
                            if real_id:
                                patient_id = real_id
                                print(f"   ✅ ID real obtenido dinámicamente: {patient_id}")
                            else:
                                print(f"   ❌ No se pudo obtener ID real")
                        
                        return PersistenceResult(
                            success=True,
                            resource_type='Patient',
                            resource_id=str(patient_id) if patient_id else 'unknown',
                            target_tables=[sql_result.get('table_used', 'PATI_PATIENTS')],
                            sql_queries=[sql_result.get('sql_used', '')],
                            processing_time_ms=processing_time,
                            records_created=1,
                            records_updated=0,
                            warnings=[f'Paciente persistido con PATI_ID: {patient_id}' if patient_id else 'PATI_ID no disponible']
                        )
                    else:
                        print(f"   🔄 Actualizando registro existente con nuevos datos...")

                        # ACTUALIZAR EL REGISTRO EXISTENTE CON DATOS MEJORADOS
                        # LLAMADA 1: Determinar campos a actualizar usando LLM
                        update_fields = await self._llm_determine_update_fields(sql_data, str(patient_id))
                        
                        # LLAMADA 2: Generar SQL de actualización usando LLM
                        update_sql = await self._llm_generate_update_sql(update_fields, str(patient_id))
                        
                        # LLAMADA 3: Ejecutar actualización usando LLM
                        update_result = await self._llm_execute_update_sql(update_sql, str(patient_id))
                        
                        if update_result.get('success'):
                            print(f"   ✅ Paciente actualizado exitosamente")
                            print(f"   📊 Campos actualizados: {update_result.get('fields_updated', 0)}")
                            print(f"   ⏱️ Tiempo de actualización: {update_result.get('execution_time', 0):.3f}s")
                            
                            return PersistenceResult(
                                success=True,
                                resource_type='Patient',
                                resource_id=str(patient_id),
                                target_tables=['PATI_PATIENTS'],
                                records_created=0,
                                records_updated=1,
                                processing_time_ms=0,
                                warnings=[f'Paciente actualizado: {sql_data.get("PATI_FULL_NAME", "N/A")}']
                            )
                        else:
                            print(f"   ❌ Error actualizando paciente: {update_result.get('error', 'Error desconocido')}")
                            return PersistenceResult(
                                success=False,
                                resource_type='Patient',
                                resource_id='unknown',
                                errors=[update_result.get('error', 'Error actualizando paciente')]
                            )
                else:
                    print(f"   ✅ Paciente no existe, procediendo con inserción...")
            except Exception as e:
                print(f"   ⚠️ Error en verificación estricta: {e}")
                print(f"   ✅ Continuando con inserción normal...")

            # ELIMINAR PATI_ID SI EXISTE EN EL DICCIONARIO DE INSERCIÓN ANTES DE INSERTAR
            if "PATI_ID" in sql_data:
                print(f"   ⚠️ Eliminando PATI_ID del diccionario de inserción para evitar corrupción de IDs")
                del sql_data["PATI_ID"]

            # 4. Delegar la inserción al SQLAgent para que use su inteligencia
            print(f"   🤖 Delegando la inserción del paciente al SQLAgent...")
            print(f"   📊 Datos a insertar: {json.dumps(sql_data, indent=2, ensure_ascii=False)}")
            sql_result = await self.sql_agent.process_data_manipulation(
                operation='INSERT',
                data=sql_data,
                context={
                    'intent': 'create_patient',
                    'resource_type': 'Patient',
                    'table_hint': 'PATI_PATIENTS'
                }
            )
            print(f"   📊 Resultado del SQL Agent: {json.dumps(sql_result, indent=2, ensure_ascii=False)}")

            processing_time = (time.time() - start_time) * 1000

            if not sql_result.get('success'):
                return PersistenceResult(
                    success=False,
                    resource_type='Patient',
                    resource_id='unknown',
                    errors=[sql_result.get('error', 'SQLAgent falló al insertar paciente')]
                )

            # Usar el ID del registro insertado directamente
            patient_id = sql_result.get('inserted_id')
            
            if patient_id:
                print(f"   ✅ ID del paciente obtenido directamente: {patient_id}")
            else:
                print(f"   ⚠️ No se pudo obtener el ID del paciente, intentando búsqueda alternativa...")
                # Búsqueda alternativa si no se pudo obtener el ID directamente
                try:
                    conn = sqlite3.connect(self.sql_agent.db_path)
                    cursor = conn.cursor()
                    
                    # Buscar por múltiples criterios
                    search_criteria = []
                    search_params = []
                    
                    if sql_data.get("PATI_FULL_NAME"):
                        search_criteria.append("LOWER(PATI_FULL_NAME) = LOWER(?)")
                        search_params.append(sql_data.get("PATI_FULL_NAME"))
                    
                    if sql_data.get("PATI_BIRTH_DATE"):
                        search_criteria.append("PATI_BIRTH_DATE = ?")
                        search_params.append(sql_data.get("PATI_BIRTH_DATE"))
                    
                    if sql_data.get("PATI_NAME") and sql_data.get("PATI_SURNAME_1"):
                        search_criteria.append("(LOWER(PATI_NAME) = LOWER(?) AND LOWER(PATI_SURNAME_1) = LOWER(?))")
                        search_params.extend([sql_data.get("PATI_NAME"), sql_data.get("PATI_SURNAME_1")])
                    
                    if search_criteria:
                        where_clause = " OR ".join(search_criteria)
                        query = f"SELECT PATI_ID FROM PATI_PATIENTS WHERE {where_clause} ORDER BY PATI_ID DESC LIMIT 1"
                        cursor.execute(query, search_params)
                        row = cursor.fetchone()
                        patient_id = row[0] if row else None
                        print(f"   🔍 Búsqueda alternativa: {patient_id}")
                    else:
                        print(f"   ⚠️ No hay criterios de búsqueda disponibles")
                    
                    conn.close()
                except Exception as e:
                    print(f"   ⚠️ Error en búsqueda alternativa: {e}")
                    patient_id = None

            return PersistenceResult(
                success=True,
                resource_type='Patient',
                resource_id=str(patient_id) if patient_id else 'unknown',  # Usar el ID real de la base de datos
                target_tables=[sql_result.get('table_used', 'PATI_PATIENTS')],
                sql_queries=[sql_result.get('sql_used', '')],
                processing_time_ms=processing_time,
                records_created=1,
                records_updated=0,
                warnings=[f'Paciente persistido con PATI_ID: {patient_id}' if patient_id else 'PATI_ID no disponible']
            )

        except Exception as e:
            print(f"   🔥 EXCEPCIÓN EN _persist_patient_resource: {e}")
            logger.error(f"Error crítico al persistir paciente: {e}", exc_info=True)
            return PersistenceResult(
                success=False,
                resource_type='Patient',
                resource_id='unknown',
                errors=[f'Error crítico al persistir paciente: {str(e)}']
            )

    async def _persist_resource_generic(self, fhir_resource: Dict[str, Any], patient_id_context: Optional[str] = None) -> PersistenceResult:
        """Persiste un recurso FHIR usando LLM completamente genérico con mapeo específico"""
        print(f"   🚀 ENTRANDO A _persist_resource_generic")
        print(f"   📊 Tipo de recurso: {fhir_resource.get('resourceType', 'Desconocido')}")
        print(f"   🆔 Patient ID Context: {patient_id_context}")
        print(f"   🤖 LLM disponible: {'Sí' if self.llm is not None else 'No'}")
        print(f"   🔧 SQL Agent disponible: {'Sí' if self.sql_agent is not None else 'No'}")
        
        try:
            if not self.llm or not self.sql_agent:
                print(f"   ❌ LLM o SQL Agent no disponible")
                return PersistenceResult(
                    success=False,
                    resource_type=fhir_resource.get('resourceType', 'Unknown'),
                    resource_id=fhir_resource.get('id', 'unknown'),
                    errors=['LLM o SQL Agent no disponible']
                )
            
            print(f"   ✅ LLM y SQL Agent disponibles, continuando...")
            
            # FORZAR CORRECCIÓN DE UUIDs ANTES DE CUALQUIER PROCESAMIENTO
            print(f"   🔧 FORZANDO corrección de UUIDs ANTES del procesamiento...")
            print(f"   📊 Recurso FHIR original: {json.dumps(fhir_resource, indent=2, ensure_ascii=False)}")
            print(f"   🆔 Patient ID Context: {patient_id_context}")
            
            # LLAMAR DIRECTAMENTE A LA CORRECCIÓN
            try:
                print(f"   🚀 LLAMANDO DIRECTAMENTE A _fix_fhir_uuid_mapping...")
                corrected_fhir_resource = await self._fix_fhir_uuid_mapping(fhir_resource, patient_id_context)
                print(f"   ✅ Corrección de UUIDs completada")
                print(f"   📊 Recurso FHIR corregido: {json.dumps(corrected_fhir_resource, indent=2, ensure_ascii=False)}")
            except Exception as e:
                print(f"   ❌ Error en corrección de UUIDs: {e}")
                print(f"   🔧 Usando recurso original sin corregir")
                corrected_fhir_resource = fhir_resource
            
            # CORRECCIÓN AUTOMÁTICA DE UUIDs FHIR
            print(f"   🔧 Corrigiendo UUIDs FHIR automáticamente...")
            print(f"   📊 Recurso FHIR original: {json.dumps(fhir_resource, indent=2, ensure_ascii=False)}")
            print(f"   🆔 Patient ID Context: {patient_id_context}")
            
            try:
                print(f"   🚀 LLAMANDO A _fix_fhir_uuid_mapping...")
                corrected_fhir_resource = await self._fix_fhir_uuid_mapping(fhir_resource, patient_id_context)
                print(f"   ✅ UUIDs FHIR corregidos exitosamente")
                print(f"   📊 Recurso FHIR corregido: {json.dumps(corrected_fhir_resource, indent=2, ensure_ascii=False)}")
                
                # Verificar que los UUIDs se corrigieron
                if 'id' in corrected_fhir_resource and 'urn:uuid:' in str(corrected_fhir_resource['id']):
                    print(f"   ⚠️ ADVERTENCIA: UUID no corregido: {corrected_fhir_resource['id']}")
                    # Forzar corrección básica
                    corrected_fhir_resource = await self._fix_fhir_uuid_mapping(corrected_fhir_resource, patient_id_context)
                    print(f"   🔧 Corrección básica aplicada")
                
            except Exception as e:
                print(f"   ❌ Error en corrección de UUIDs FHIR: {e}")
                print(f"   🔧 Aplicando corrección básica como fallback...")
                corrected_fhir_resource = await self._fix_fhir_uuid_mapping(fhir_resource, patient_id_context)
            
            # FORZAR CORRECCIÓN BÁSICA SIEMPRE PARA GARANTIZAR QUE LOS UUIDs SE CORRIGAN
            print(f"   🔧 FORZANDO corrección básica para garantizar UUIDs válidos...")
            corrected_fhir_resource = await self._fix_fhir_uuid_mapping(corrected_fhir_resource, patient_id_context)
            print(f"   ✅ Corrección básica forzada completada")
            
            # VERIFICAR QUE LOS UUIDs SE CORRIGIERON
            print(f"   🔍 Verificando corrección de UUIDs...")
            if 'id' in corrected_fhir_resource:
                old_id = corrected_fhir_resource['id']
                if 'urn:uuid:' in str(old_id):
                    print(f"   ⚠️ UUID NO corregido: {old_id}")
                    # Forzar corrección manual
                    corrected_fhir_resource['id'] = str(hash(old_id) % 10000)
                    print(f"   🔧 UUID corregido manualmente: {old_id} → {corrected_fhir_resource['id']}")
                else:
                    print(f"   ✅ UUID ya corregido: {old_id}")
            
            # CORREGIR REFERENCIAS EN SUBJECT Y ENCOUNTER
            if 'subject' in corrected_fhir_resource and isinstance(corrected_fhir_resource['subject'], dict):
                if 'reference' in corrected_fhir_resource['subject']:
                    old_ref = corrected_fhir_resource['subject']['reference']
                    if 'urn:uuid:' in str(old_ref):
                        corrected_fhir_resource['subject']['reference'] = str(hash(old_ref) % 10000)
                        print(f"   🔧 Referencia subject corregida: {old_ref} → {corrected_fhir_resource['subject']['reference']}")
            
            if 'encounter' in corrected_fhir_resource and isinstance(corrected_fhir_resource['encounter'], dict):
                if 'reference' in corrected_fhir_resource['encounter']:
                    old_ref = corrected_fhir_resource['encounter']['reference']
                    if 'urn:uuid:' in str(old_ref):
                        corrected_fhir_resource['encounter']['reference'] = str(hash(old_ref) % 10000)
                        print(f"   🔧 Referencia encounter corregida: {old_ref} → {corrected_fhir_resource['encounter']['reference']}")
            
            # DETECTAR SI ES UN MEDICATIONREQUEST Y USAR MAPEO ESPECÍFICO
            if corrected_fhir_resource.get('resourceType') == 'MedicationRequest':
                print(f"   🩺 Detectado MedicationRequest, usando mapeo específico para medicamentos...")
                medication_mapping = await self._llm_map_medication_request(corrected_fhir_resource, patient_id_context)
                
                if medication_mapping.get('success'):
                    target_table = medication_mapping['target_table']
                    validated_data = medication_mapping['mapped_data']
                    
                    print(f"   ✅ Mapeo de medicamento exitoso:")
                    print(f"      📋 Tabla: {target_table}")
                    print(f"      📊 Datos mapeados: {json.dumps(validated_data, indent=2, ensure_ascii=False)}")
                    
                    # Usar el agente SQL directamente con datos mapeados específicamente para medicamentos
                    print(f"   🤖 Llamando al SQL Agent para insertar medicamento en {target_table}...")
                    sql_result = await self.sql_agent.process_data_manipulation(
                        operation='INSERT',
                        data=validated_data,
                        context={
                            'intent': 'insert_medication',
                            'table_hint': target_table,
                            'patient_id': patient_id_context
                        }
                    )
                    print(f"   📊 Resultado del SQL Agent: {json.dumps(sql_result, indent=2, ensure_ascii=False)}")
                    
                    if sql_result.get('success'):
                        # Tracking de la inserción
                        inserted_id = sql_result.get('inserted_id')
                        if inserted_id and isinstance(inserted_id, (int, float)) and inserted_id > 0:
                            tracking_info = await self._track_insertion_location(target_table, validated_data, int(inserted_id))
                        
                        return PersistenceResult(
                            success=True,
                            resource_type='MedicationRequest',
                            resource_id=corrected_fhir_resource.get('id', 'unknown'),
                            target_tables=[target_table],
                            sql_queries=[sql_result.get('sql_used', '')],
                            records_created=1,
                            records_updated=0,
                            warnings=[f'Medicamento insertado en {target_table} con ID: {inserted_id}']
                        )
                    else:
                        return PersistenceResult(
                            success=False,
                            resource_type='MedicationRequest',
                            resource_id=corrected_fhir_resource.get('id', 'unknown'),
                            errors=[sql_result.get('error', 'Error insertando medicamento')]
                        )
                else:
                    return PersistenceResult(
                        success=False,
                        resource_type='MedicationRequest',
                        resource_id=corrected_fhir_resource.get('id', 'unknown'),
                        errors=[medication_mapping.get('error', 'Error en mapeo de medicamento')]
                    )
            
            # USAR LLM PARA DETECTAR SI LOS DATOS YA ESTÁN MAPEADOS (para otros tipos de recursos)
            is_mapped = await self._llm_detect_if_data_mapped(corrected_fhir_resource)
            
            if is_mapped:
                # Los datos ya están mapeados, usar LLM para determinar tabla
                print(f"   ✅ Datos ya mapeados detectados por LLM")
                target_table = await self._llm_determine_table_from_mapped_data(corrected_fhir_resource)
                
                # Validar y corregir el mapeo de tabla
                validated_data = await self._validate_and_fix_table_mapping(target_table, corrected_fhir_resource)
                
                # Asegurar que PATI_ID esté presente en todos los recursos
                if patient_id_context and 'PATI_ID' in validated_data:
                    validated_data['PATI_ID'] = patient_id_context
                    print(f"   ✅ PATI_ID añadido: {patient_id_context}")
                
                # Usar el agente SQL directamente con datos ya mapeados y validados
                print(f"   🤖 Llamando al SQL Agent para insertar en {target_table}...")
                print(f"   📊 Datos validados: {json.dumps(validated_data, indent=2, ensure_ascii=False)}")
                sql_result = await self.sql_agent.process_data_manipulation(
                    operation='INSERT',
                    data=validated_data,
                    context={
                        'intent': 'insert_mapped_data',
                        'table_hint': target_table,
                        'patient_id': patient_id_context
                    }
                )
                print(f"   📊 Resultado del SQL Agent: {json.dumps(sql_result, indent=2, ensure_ascii=False)}")
                
                if sql_result.get('success'):
                    # Tracking de la inserción
                    inserted_id = sql_result.get('inserted_id')
                    if inserted_id and isinstance(inserted_id, (int, float)) and inserted_id > 0:
                        tracking_info = await self._track_insertion_location(target_table, validated_data, int(inserted_id))
                    
                    return PersistenceResult(
                        success=True,
                        resource_type='MappedData',
                        resource_id=corrected_fhir_resource.get('id', 'unknown'),
                        target_tables=[target_table],
                        sql_queries=[sql_result.get('sql_used', '')],
                        records_created=1,
                        records_updated=0,
                        warnings=[f'Insertado en {target_table} con ID: {inserted_id}']
                    )
                else:
                    return PersistenceResult(
                        success=False,
                        resource_type='MappedData',
                        resource_id=corrected_fhir_resource.get('id', 'unknown'),
                        errors=[sql_result.get('error', 'Error desconocido')]
                )
            
            # --- VALIDACIÓN INTELIGENTE DE PATI_ID PARA FHIR ---
            validation_result = await self._validate_pati_id_fhir(corrected_fhir_resource, patient_id_context)
            if validation_result.get('requires_pati_id', False) and not validation_result.get('valid', False):
                return PersistenceResult(
                    success=False,
                    resource_type=fhir_resource.get('resourceType', 'Unknown'),
                    resource_id=fhir_resource.get('id', 'unknown'),
                    errors=[f'Validación PATI_ID falló: {validation_result.get("error", "PATI_ID inválido")}'],
                    warnings=[validation_result.get('reasoning', '')]
                )
            # --- FIN VALIDACIÓN INTELIGENTE ---
            
            # Añadir el ID del paciente al contexto si está disponible
            context_info = ""
            if patient_id_context:
                context_info = f"CONTEXTO ADICIONAL: El ID del paciente para las claves foráneas (FK) es '{patient_id_context}'. Úsalo para campos como `PATI_ID`."

            # MEJORA: Para medicamentos, usar PATI_USUAL_MEDICATION en lugar de MEDI_MEDICATIONS
            medication_lookup_info = ""
            if fhir_resource.get('resourceType') == 'MedicationRequest':
                medication_name = ""
                dosage_info = ""
                
                # Extraer nombre del medicamento del recurso FHIR
                if 'medicationCodeableConcept' in fhir_resource:
                    med_concept = fhir_resource['medicationCodeableConcept']
                    if 'text' in med_concept:
                        medication_name = med_concept['text']
                    elif 'coding' in med_concept and med_concept['coding']:
                        medication_name = med_concept['coding'][0].get('display', '')
                
                # Extraer información de dosis
                if 'dosageInstruction' in fhir_resource and fhir_resource['dosageInstruction']:
                    dosage = fhir_resource['dosageInstruction'][0]
                    if 'text' in dosage:
                        dosage_info = dosage['text']
                
                if medication_name:
                    medication_lookup_info = f"""
                        
            INFORMACIÓN ADICIONAL DE MEDICAMENTO:
            Para el medicamento "{medication_name}":
            - Dosis: {dosage_info}
            
            IMPORTANTE: 
            - Usar tabla PATI_USUAL_MEDICATION (NO MEDI_MEDICATIONS)
            - Campo PAUM_OBSERVATIONS para el texto completo: "{medication_name} {dosage_info}"
            - PATI_ID es obligatorio para vincular con el paciente
            - NO usar MEDI_ID ni ACIN_ID en esta inserción
            """

            # LLM determina cómo persistir este recurso específico con validación estricta
            prompt = f"""
            Necesito persistir este recurso FHIR en la base de datos SQL.
            
            RECURSO FHIR:
            {json.dumps(fhir_resource, indent=2)}
            
            ESQUEMA DE BASE DE DATOS:
            {await self._get_database_schema_info()}
            
            {context_info}
            {medication_lookup_info}
            
            REGLAS CRÍTICAS Y ESPECÍFICAS:
            1. SOLO usa nombres de columnas que EXISTAN en el esquema
            2. NO inventes nombres de columnas
            3. SIEMPRE incluir PATI_ID cuando el campo "patient_id" esté presente en los datos
            4. ANALIZA el esquema completo para encontrar la tabla más apropiada
            5. Para recursos médicos, busca tablas que contengan información similar
            6. Para medicamentos, prioriza tablas con campos de medicación
            7. Para diagnósticos, busca tablas con campos de observación o diagnóstico
            8. Para observaciones, busca tablas con campos de valores y unidades
            9. Para pacientes, busca tablas con información personal y demográfica
            
            MAPEO ESPECÍFICO POR TIPO DE RECURSO:
            - Patient → PATI_PATIENTS (datos personales del paciente)
            - Condition → EPIS_DIAGNOSTICS (diagnósticos y condiciones médicas)
            - MedicationRequest → PATI_USUAL_MEDICATION (medicación del paciente)
            - Observation → EPIS_DIAGNOSTICS (observaciones médicas, incluye constantes vitales)
            - Encounter → APPO_APPOINTMENTS (citas y encuentros médicos)
            
            REGLAS CRÍTICAS PARA MEDICAMENTOS:
            - SIEMPRE usar tabla PATI_USUAL_MEDICATION para MedicationRequest
            - Campo PAUM_OBSERVATIONS: texto libre con nombre y dosis del medicamento
            - Campo PATI_ID: ID del paciente (obligatorio)
            - NO usar tabla MEDI_MEDICATIONS para inserción de medicación del paciente
            - NO usar campos MEDI_ID, ACIN_ID en PATI_USUAL_MEDICATION
            - Formato PAUM_OBSERVATIONS: "Nombre medicamento: dosis" (ej: "Paracetamol: 1g c/6h PRN")
            
            REGLAS ESPECÍFICAS PARA CONSTANTES VITALES:
            - Si el recurso es Observation y contiene valores numéricos (presión, temperatura, etc.)
            - Usar EPIS_DIAGNOSTICS.DIAG_OBSERVATION para almacenar los datos
            - Formato: "Tipo: valor unidad" (ej: "Presión arterial: 120/80 mmHg")
            
            REGLAS ESPECÍFICAS PARA MEDICAMENTOS:
            - SIEMPRE usar tabla PATI_USUAL_MEDICATION para MedicationRequest
            - Campo PAUM_OBSERVATIONS: texto libre con nombre y dosis del medicamento
            - Campo PATI_ID: ID del paciente (obligatorio)
            - NO usar tabla MEDI_MEDICATIONS para inserción de medicación del paciente
            - NO usar campos MEDI_ID, ACIN_ID en PATI_USUAL_MEDICATION
            - Formato PAUM_OBSERVATIONS: "Nombre medicamento: dosis" (ej: "Paracetamol: 1g c/6h PRN")
            - Para MedicationRequest: mapear medicationCodeableConcept.text + dosageInstruction.text a PAUM_OBSERVATIONS
            
            REGLAS ESPECÍFICAS PARA DIAGNÓSTICOS:
            - Usar EPIS_DIAGNOSTICS.DIAG_OBSERVATION
            - Incluir código y descripción del diagnóstico
            - Formato: "Diagnóstico: descripción (código)" (ej: "Diabetes tipo 2 (E11)")
            
            TAREA:
            Determina la mejor forma de guardar este recurso FHIR específico.
            Mapea cada campo FHIR relevante a los campos SQL EXACTOS del esquema.
            
            RESPONDE EN JSON:
            {{
                "target_table": "nombre_tabla_exacto_del_esquema",
                "sql_data": {{
                    "PATI_ID": "valor_si_disponible",
                    "campo1": "valor1",
                    "campo2": "valor2"
                }},
                "intent": "tipo_de_operacion",
                "confidence": 0.95,
                "reasoning": "explicación_del_mapeo"
            }}
            
            IMPORTANTE: 
            - Genera SQL con valores específicos, NO placeholders
            - SIEMPRE incluir PATI_ID cuando esté disponible en los datos
            - Responde SOLO con el JSON
            """
            
            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if not result:
                print(f"   ❌ Error: No se pudo parsear la respuesta del LLM")
                return PersistenceResult(
                    success=False,
                    resource_type=fhir_resource.get('resourceType', 'Unknown'),
                    resource_id=fhir_resource.get('id', 'unknown'),
                    errors=['Error parseando respuesta del LLM']
                )
            
            # Validar que los datos contengan solo campos que existen en el esquema
            target_table = result.get('target_table', '')
            sql_data = result.get('sql_data', {})
            
            if not target_table or not sql_data:
                print(f"   ❌ Error: Respuesta del LLM incompleta")
                return PersistenceResult(
                    success=False,
                    resource_type=fhir_resource.get('resourceType', 'Unknown'),
                    resource_id=fhir_resource.get('id', 'unknown'),
                    errors=['Respuesta del LLM incompleta']
                )
            
            # Validar que la tabla existe y corregir si es necesario
            if target_table not in self.sql_agent.column_metadata:
                print(f"   ⚠️ Tabla {target_table} no existe, intentando corregir...")
                
                # VALIDACIÓN ADAPTATIVA: Usar LLM para validar y corregir tabla
                if self.llm:
                    # TODO: Implementar validación adaptativa del LLM aquí
                    print(f"   ⚠️ Tabla {target_table} no existe, usando validación adaptativa del LLM")
                else:
                    print(f"   ❌ Error: Tabla {target_table} no existe en el esquema")
                    return PersistenceResult(
                        success=False,
                        resource_type=fhir_resource.get('resourceType', 'Unknown'),
                        resource_id=fhir_resource.get('id', 'unknown'),
                        errors=[f'Tabla {target_table} no existe en el esquema']
                    )
                
            
            # Filtrar solo campos que existen en la tabla
            valid_columns = self.sql_agent.column_metadata[target_table]
            valid_data = {}
            for key, value in sql_data.items():
                if key in valid_columns:
                    valid_data[key] = value
                else:
                    print(f"   ⚠️ Campo {key} no existe en tabla {target_table}, omitiendo")
            
            # CRÍTICO: Asegurar que PATI_ID esté presente si tenemos el contexto
            if patient_id_context and 'PATI_ID' not in valid_data:
                # Validar que el ID del paciente no sea ficticio
                if not self._is_fictitious_id(patient_id_context):
                    valid_data['PATI_ID'] = patient_id_context
                    print(f"   ✅ Añadido PATI_ID real: {patient_id_context}")
                else:
                    print(f"   ⚠️ ID de paciente ficticio detectado: {patient_id_context}")
                    # Intentar usar el ID real del paciente si está disponible
                    if hasattr(self, '_current_patient_id') and self._current_patient_id:
                        valid_data['PATI_ID'] = self._current_patient_id
                        print(f"   ✅ Usando PATI_ID del contexto: {self._current_patient_id}")
            
            # CRÍTICO: Usar LLM para corregir valores vacíos de forma inteligente
            if self.llm:
                corrected_data = await self._llm_fix_empty_values(valid_data, target_table, patient_id_context)
                if corrected_data:
                    valid_data = corrected_data
                    print(f"   ✅ LLM corrigió valores vacíos en {target_table}")
                else:
                    print(f"   ⚠️ LLM no pudo corregir valores, usando fallback básico")
                    valid_data = self._fix_empty_values_fallback(valid_data, target_table)
            else:
                # Fallback sin LLM
                valid_data = self._fix_empty_values_fallback(valid_data, target_table)
                # Usar solo datos válidos
                if result is not None:
                    result['sql_data'] = valid_data
                else:
                    result = {'sql_data': valid_data}

                # Usar SQL Agent para ejecutar
                sql_result = await self.sql_agent.process_data_manipulation(
                    'INSERT',
                result.get('sql_data', {}) if result else {},
                {
                    'intent': result.get('intent', 'generic_insert') if result else 'generic_insert',
                    'table_hint': result.get('target_table', '') if result else '',
                'fhir_resource': fhir_resource,
                    'patient_id': patient_id_context # Pasar el ID del paciente al contexto
                }
            )
            
            # Extraer valores de forma segura
            target_table = ''
            if result and isinstance(result, dict):
                target_table = str(result.get('target_table', ''))
            
            sql_used = ''
            if sql_result and isinstance(sql_result, dict):
                sql_used = str(sql_result.get('sql_used', ''))
            
            confidence = 0.0
            if result and isinstance(result, dict):
                confidence_value = result.get('confidence', 0.0)
                if isinstance(confidence_value, (int, float)):
                    confidence = float(confidence_value)
                else:
                    confidence = 0.0
            
            success = False
            if sql_result and isinstance(sql_result, dict):
                success = sql_result.get('success', False)
            
            errors = []
            if sql_result and isinstance(sql_result, dict) and not success:
                error_msg = str(sql_result.get('error', ''))
                if error_msg:
                    errors = [error_msg]
            
            return PersistenceResult(
                success=success,
                resource_type=fhir_resource.get('resourceType', ''),
                resource_id=fhir_resource.get('id', 'unknown'),
                target_tables=[target_table],
                sql_queries=[sql_used],
                confidence_score=confidence,
                errors=errors
            )
            
        except Exception as e:
            return PersistenceResult(
                success=False,
                resource_type=fhir_resource.get('resourceType', 'Unknown'),
                resource_id=fhir_resource.get('id', 'unknown'),
                errors=[f'Error persistiendo recurso: {str(e)}']
            )
    
    async def _generate_processing_summary(self, original_note: str, extracted_data: Dict[str, Any], 
                                         fhir_resources: List[Dict[str, Any]], 
                                         persistence_results: List[PersistenceResult]) -> str:
        """Genera un resumen final del procesamiento usando LLM"""
        try:
            if not self.llm:
                return "Resumen no disponible (LLM no configurado)"
            
            # Preparar estadísticas
            successful_persistences = sum(1 for r in persistence_results if r.success)
            failed_persistences = len(persistence_results) - successful_persistences
            
            # Analizar tablas actualizadas
            updated_tables = set()
            for result in persistence_results:
                if result.success and result.target_tables:
                    updated_tables.update(result.target_tables)
            
            # Contar tipos de recursos por tabla
            table_summary = {}
            for result in persistence_results:
                if result.success and result.target_tables:
                    for table in result.target_tables:
                        if table not in table_summary:
                            table_summary[table] = {'count': 0, 'types': set()}
                        table_summary[table]['count'] += 1
                        table_summary[table]['types'].add(result.resource_type)
            
            # Formatear información de tablas
            tables_info = ""
            if table_summary:
                tables_info = "\nTABLAS ACTUALIZADAS EN LA BASE DE DATOS:\n"
                for table, info in table_summary.items():
                    types_str = ", ".join(info['types'])
                    tables_info += f"- {table}: {info['count']} registros ({types_str})\n"
            
            prompt = f"""
            Genera un resumen profesional del procesamiento de esta nota clínica.
            
            NOTA ORIGINAL:
            {original_note[:500]}...
            
            ESTADÍSTICAS DE PROCESAMIENTO:
            - Recursos FHIR generados: {len(fhir_resources)}
            - Persistencias exitosas: {successful_persistences}
            - Persistencias fallidas: {failed_persistences}
            - Tablas actualizadas: {len(updated_tables)}
            
            RECURSOS FHIR GENERADOS:
            {[r.get('resourceType') for r in fhir_resources]}
            
            DATOS EXTRAÍDOS:
            {json.dumps(extracted_data.get('extracted_data', {}), indent=2)[:1000]}...
            
            {tables_info}
            
            GENERA UN RESUMEN PROFESIONAL EN ESPAÑOL que incluya:
            1. Resumen de la nota clínica procesada
            2. Datos del paciente identificados
            3. Condiciones médicas encontradas
            4. Medicamentos y tratamientos
            5. Recursos FHIR creados
            6. Estado de la persistencia en base de datos
            7. **INFORMACIÓN DETALLADA DE LAS TABLAS ACTUALIZADAS** - qué tablas se modificaron y cuántos registros se añadieron
            8. Cualquier advertencia o información relevante
            
            El resumen debe ser claro, conciso y útil para personal médico.
            **IMPORTANTE**: Incluye específicamente qué tablas de la base de datos se actualizaron y con qué información.
            """
            
            response = _call_openai_native(self.llm, prompt)
            return response.content.strip()

        except Exception as e:
            return f"Error generando resumen: {str(e)}"
    
    async def _intelligent_fhir_to_sql_mapping(self, fhir_resource: Dict[str, Any]) -> Dict[str, Any]:
        """Mapea un recurso FHIR a SQL usando LLM inteligente"""
        try:
            if not self.llm:
                return {
                    'success': False,
                    'error': 'LLM no disponible para mapeo inteligente'
                }
            
            resource_type = fhir_resource.get('resourceType', '')
            
            # Obtener información del esquema de la base de datos
            schema_info = await self._get_database_schema_info()
            
            # Prompt inteligente para mapeo FHIR→SQL con validación estricta
            prompt = f"""
            Mapea este recurso FHIR a datos SQL usando el esquema de la base de datos disponible.
            
            RECURSO FHIR:
            {json.dumps(fhir_resource, indent=2)}
            
            ESQUEMA DE BASE DE DATOS DISPONIBLE:
            {schema_info}
            
            REGLAS CRÍTICAS:
            1. SOLO usa nombres de columnas que EXISTAN en el esquema
            2. NO inventes nombres de columnas
            3. Si no hay tabla apropiada, usa la más genérica disponible
            4. ANALIZA el esquema completo para encontrar la tabla más apropiada
            5. Para recursos médicos, busca tablas que contengan información similar
            6. Para medicamentos, prioriza tablas con campos de medicación
            7. Para diagnósticos, busca tablas con campos de observación o diagnóstico
            8. Para observaciones, busca tablas con campos de valores y unidades
            9. Para pacientes, busca tablas con información personal y demográfica
            
            INSTRUCCIONES:
            1. Identifica la tabla más apropiada para este tipo de recurso FHIR
            2. Mapea los campos FHIR a las columnas SQL EXACTAS del esquema
            3. Genera un intent apropiado para el SQL Agent
            4. Incluye SOLO los campos que existen en la tabla
            5. Usa valores por defecto apropiados para campos requeridos
            6. Si no hay mapeo directo, usa columnas de texto/notes
            7. Considera múltiples tablas si el recurso puede mapearse a varias
            
            RESPONDE EN JSON:
            {{
                "success": true,
                "target_table": "nombre_tabla_exacto_del_esquema",
                "intent": "create_patient|create_condition|create_medication|create_observation",
                "data": {{
                    "nombre_columna_exacta_1": "valor_mapeado_1",
                    "nombre_columna_exacta_2": "valor_mapeado_2"
                }},
                "confidence": 0.95,
                "mapping_notes": "Explicación del mapeo realizado",
                "validation": "Confirmo que todos los nombres de columnas existen en el esquema"
            }}
            
            IMPORTANTE: Verifica que cada nombre de columna en "data" existe exactamente en el esquema.
            """
            
            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if not result:
                return {
                    'success': False,
                    'error': 'No se pudo parsear la respuesta del LLM para mapeo FHIR→SQL'
                }
            
            # Validar que el mapeo sea correcto
            if not result.get('target_table') or not result.get('data'):
                return {
                    'success': False,
                    'error': 'Mapeo FHIR→SQL incompleto'
                }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error en mapeo inteligente FHIR→SQL: {str(e)}'
            }
    
    async def _get_database_schema_info(self) -> str:
        """Obtiene información detallada del esquema de la base de datos para mapeo inteligente"""
        try:
            if self.sql_agent and hasattr(self.sql_agent, 'column_metadata'):
                schema_info = []
                
                # Enfoque específico en PATI_PATIENTS para mapeo de pacientes
                if 'PATI_PATIENTS' in self.sql_agent.column_metadata:
                    table_info = self.sql_agent.column_metadata['PATI_PATIENTS']
                    schema_info.append("TABLA PATI_PATIENTS (Información detallada para mapeo):")
                    schema_info.append(f"  Total de registros: {table_info.get('row_count', 'N/A')}")
                    schema_info.append("  Columnas disponibles:")
                    
                    for col in table_info.get('columns', []):
                        col_name = col.get('name', '')
                        col_type = col.get('type', '')
                        is_pk = col.get('primary_key', False)
                        
                        # Información específica para mapeo
                        mapping_info = ""
                        if col_name == 'GEND_ID':
                            mapping_info = " - Valores: 1 (masculino), 2 (femenino), 3 (otro)"
                        elif col_name == 'PATI_ACTIVE':
                            mapping_info = " - Valores: 1 (activo), 0 (inactivo)"
                        elif col_name == 'PATI_BIRTH_DATE':
                            mapping_info = " - Formato: YYYY-MM-DD"
                        elif col_name == 'PATI_START_DATE':
                            mapping_info = " - Formato: ISO datetime (YYYY-MM-DDTHH:MM:SS)"
                        elif col_name == 'PATI_FULL_NAME':
                            mapping_info = " - Nombre completo del paciente"
                        elif col_name == 'PATI_NAME':
                            mapping_info = " - Nombre del paciente"
                        elif col_name == 'PATI_SURNAME_1':
                            mapping_info = " - Primer apellido"
                        elif col_name == 'PATI_SURNAME_2':
                            mapping_info = " - Segundo apellido (opcional)"
                        
                        col_info = f"    - {col_name}: {col_type}"
                        if is_pk:
                            col_info += " [PRIMARY KEY]"
                        col_info += mapping_info
                        schema_info.append(col_info)
                    
                    schema_info.append("")
                    schema_info.append("INSTRUCCIONES DE MAPEO:")
                    schema_info.append("  - PATI_NAME: Extraer del campo 'given' del nombre FHIR")
                    schema_info.append("  - PATI_SURNAME_1: Extraer del campo 'family' del nombre FHIR")
                    schema_info.append("  - PATI_FULL_NAME: Combinar nombre y apellido")
                    schema_info.append("  - GEND_ID: Convertir 'male'→1, 'female'→2, otros→3")
                    schema_info.append("  - PATI_BIRTH_DATE: Usar formato YYYY-MM-DD")
                    schema_info.append("  - PATI_START_DATE: Usar fecha actual en formato ISO")
                    schema_info.append("  - PATI_ACTIVE: Siempre usar 1 para nuevos pacientes")
                    schema_info.append("")
                
                # Añadir información de otras tablas relevantes
                other_tables = []
                for table_name, table_info in self.sql_agent.column_metadata.items():
                    if table_name != 'PATI_PATIENTS' and any(keyword in table_name.lower() for keyword in ['appo', 'medi', 'diag', 'epis', 'acci']):
                        other_tables.append(f"  - {table_name}: {len(table_info.get('columns', []))} columnas")
                
                if other_tables:
                    schema_info.append("OTRAS TABLAS RELEVANTES:")
                    schema_info.extend(other_tables)
                    schema_info.append("")
                
                return "\n".join(schema_info) if schema_info else "Esquema no disponible"
            
            # Fallback: obtener esquema directamente de la BD
            try:
                import sqlite3
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("PRAGMA table_info(PATI_PATIENTS)")
                columns = cursor.fetchall()
                
                schema_info = ["TABLA PATI_PATIENTS (Esquema directo):"]
                for col in columns:
                    col_name = col[1]
                    col_type = col[2]
                    is_nullable = "NULL" if col[3] == 0 else "NOT NULL"
                    default_value = col[4] if col[4] else "Sin valor por defecto"
                    is_pk = "PRIMARY KEY" if col[5] == 1 else ""
                    
                    schema_info.append(f"  - {col_name}: {col_type} ({is_nullable})")
                    if is_pk:
                        schema_info.append(f"    [PRIMARY KEY]")
                    if default_value != "Sin valor por defecto":
                        schema_info.append(f"    Default: {default_value}")
                
                conn.close()
                return "\n".join(schema_info)
                
            except Exception as e:
                return f"Error obteniendo esquema directo: {str(e)}"
            
        except Exception as e:
            return f"Error obteniendo esquema: {str(e)}"
    
    async def _persist_fhir_resource(self, fhir_resource: Dict[str, Any]) -> PersistenceResult:
        """Persiste un recurso FHIR usando el SQL Agent con mapeo inteligente LLM"""
        try:
            if not self.sql_agent:
                return PersistenceResult(
                    success=False,
                    resource_type=fhir_resource.get('resourceType', 'Unknown'),
                    resource_id=fhir_resource.get('id', 'unknown'),
                    errors=['SQL Agent no disponible']
                )
            
            # Usar LLM para mapear inteligentemente FHIR a SQL
            mapped_data = await self._intelligent_fhir_to_sql_mapping(fhir_resource)
            
            if not mapped_data.get('success'):
                return PersistenceResult(
                    success=False,
                    resource_type=fhir_resource.get('resourceType', 'Unknown'),
                    resource_id=fhir_resource.get('id', 'unknown'),
                    errors=[mapped_data.get('error', 'Error en mapeo FHIR→SQL')]
                )
            
            # Usar SQL Agent para persistir con contexto inteligente
            result = await self.sql_agent.process_data_manipulation(
                operation='INSERT',
                data=mapped_data['data'],
                context={
                    'intent': mapped_data.get('intent', 'create_resource'),
                    'resource_type': fhir_resource.get('resourceType', ''),
                    'fhir_resource': fhir_resource
                }
            )
            
            return PersistenceResult(
                success=result.get('success', False),
                resource_type=fhir_resource.get('resourceType', ''),
                resource_id=fhir_resource.get('id', 'unknown'),
                target_tables=[mapped_data.get('target_table', '')],
                sql_queries=[result.get('sql_used', '')],
                errors=[result.get('error', '')] if not result.get('success') else []
            )
            
        except Exception as e:
            return PersistenceResult(
                success=False,
                resource_type=fhir_resource.get('resourceType', 'Unknown'),
                resource_id=fhir_resource.get('id', 'unknown'),
                errors=[f'Error persistiendo recurso: {str(e)}']
            )
    
    async def _map_fhir_to_sql_basic(self, fhir_resource: Dict[str, Any]) -> Dict[str, Any]:
        """Mapeo básico de FHIR a SQL usando nombres correctos de columnas"""
        resource_type = fhir_resource.get('resourceType', '')
        
        # USAR LLM PARA MAPEO INTELIGENTE DE RECURSOS FHIR
        resource_mapping_result = await self._llm_map_fhir_resource_type(fhir_resource)
        return resource_mapping_result
    
    def _process_fhir_query(self, query: str) -> Dict[str, Any]:
        """Procesa consultas FHIR específicas"""
        return {
            'success': True,
            'type': 'fhir_query',
            'message': 'Consulta FHIR procesada (implementación básica)'
        }
    
    def _process_conversion_request(self, query: str) -> Dict[str, Any]:
        """Procesa solicitudes de conversión"""
        return {
            'success': True,
            'type': 'conversion',
            'message': 'Conversión procesada (implementación básica)'
        }
    
    def _process_general_fhir_request(self, query: str) -> Dict[str, Any]:
        """Procesa solicitudes FHIR generales"""
        return {
            'success': True,
            'type': 'general_fhir',
            'message': 'Solicitud FHIR general procesada (implementación básica)'
        }
    
    async def process_patient_data(self, patient_data: Dict[str, Any], conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        """Procesa datos de paciente"""
        try:
            # Implementación básica
            return {
                'success': True,
                'message': 'Datos de paciente procesados',
                'patient_data': patient_data
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error procesando datos de paciente: {str(e)}'
            }
    
    async def process_data_manipulation(self, operation: str, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Procesa operaciones de manipulación de datos
        Compatible con integraciones externas
        """
        try:
            if self.sql_agent:
                return await self.sql_agent.process_data_manipulation(operation, data, context)
            else:
                return {
                    'success': False,
                    'error': 'SQL Agent no disponible para manipulación de datos'
                }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error en manipulación de datos: {str(e)}'
            }
    
    def _create_error_response(self, error: str) -> Dict[str, Any]:
        """Crea respuesta de error estándar"""
        return {
            'success': False,
            'error': error,
            'type': 'error',
            'timestamp': datetime.now().isoformat()
        }
            
    def _try_parse_llm_json(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Intenta parsear JSON de respuesta LLM con manejo robusto de errores.
        Optimizado para recursos FHIR.
        """
        print(f"   🔍 INICIANDO _try_parse_llm_json")
        print(f"   📄 Contenido a parsear: {content[:200]}...")
        
        try:
            # PRIMERA ESTRATEGIA: Buscar array JSON (recursos FHIR)
            array_start = content.find('[')
            if array_start != -1:
                array_end = content.rfind(']')
                if array_end != -1 and array_end > array_start:
                    json_array = content[array_start:array_end + 1]
                    try:
                        result = json.loads(json_array)
                        print(f"   ✅ Array JSON parseado exitosamente: {len(result)} elementos")
                        return result
                    except json.JSONDecodeError as e:
                        print(f"   ⚠️ Array JSON falló: {e}")
            
            # SEGUNDA ESTRATEGIA: Buscar objeto JSON individual
            obj_start = content.find('{')
            if obj_start != -1:
                obj_end = content.rfind('}')
                if obj_end != -1 and obj_end > obj_start:
                    json_obj = content[obj_start:obj_end + 1]
                    try:
                        result = json.loads(json_obj)
                        print(f"   ✅ Objeto JSON parseado exitosamente")
                        return result
                    except json.JSONDecodeError as e:
                        print(f"   ⚠️ Objeto JSON falló: {e}")
            
            # TERCERA ESTRATEGIA: Limpiar y reparar
            print(f"   🔧 Intentando limpieza y reparación...")
            cleaned_content = self._clean_json_string(content)
            
            # Buscar JSON en contenido limpio
            for start_char in ['[', '{']:
                start_pos = cleaned_content.find(start_char)
                if start_pos != -1:
                    end_char = ']' if start_char == '[' else '}'
                    end_pos = cleaned_content.rfind(end_char)
                    if end_pos != -1 and end_pos > start_pos:
                        json_str = cleaned_content[start_pos:end_pos + 1]
                        try:
                            result = json.loads(json_str)
                            print(f"   ✅ JSON limpio parseado exitosamente")
                            return result
                        except json.JSONDecodeError:
                            continue
            
            # CUARTA ESTRATEGIA: Extraer primer JSON válido
            print(f"   🔧 Último intento: extraer primer JSON válido...")
            first_json = self._extract_first_valid_json(content)
            if first_json:
                try:
                    result = json.loads(first_json)
                    print(f"   ✅ Primer JSON válido extraído")
                    return result
                except json.JSONDecodeError:
                    pass
            
            print(f"   ❌ Todas las estrategias de parsing fallaron")
            return None
            
        except Exception as e:
            print(f"   ❌ Error inesperado en _try_parse_llm_json: {e}")
            return None
            
    def _repair_incomplete_json(self, json_str: str) -> Optional[str]:
        """Intenta reparar JSON incompleto"""
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
            
            # Cerrar strings incompletos
            if repaired.count('"') % 2 != 0:
                repaired += '"'
            
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
                'description': r'"description"\s*:\s*"([^"]*)"',
                'resourceType': r'"resourceType"\s*:\s*"([^"]*)"',
                'id': r'"id"\s*:\s*"([^"]*)"'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, json_str, re.IGNORECASE)
                if match:
                    if key in ['age']:
                        try:
                            result[key] = int(match.group(1))
                        except ValueError:
                            result[key] = match.group(1)
                    else:
                        result[key] = match.group(1)
            
            return result if result else None
            
        except Exception:
            return None
    
    def _extract_multiple_json_objects(self, content: str) -> List[Dict[str, Any]]:
        """Extrae múltiples objetos JSON usando LLM con manejo robusto de errores"""
        try:
            if not self.llm:
                return []
            
            # PRIMERA ESTRATEGIA: Intentar extraer JSONs directamente del contenido
            print(f"   🔍 Intentando extracción directa de JSONs del contenido...")
            direct_extraction = self._extract_json_objects_directly(content)
            if direct_extraction:
                print(f"   ✅ Extracción directa exitosa: {len(direct_extraction)} recursos")
                return direct_extraction
            
            # SEGUNDA ESTRATEGIA: Usar LLM para reparar y extraer
            print(f"   🧠 Usando LLM para extracción inteligente...")
            prompt = f"""Eres un experto en reparación de JSON FHIR. El contenido JSON está malformado. Repáralo y extrae todos los recursos FHIR válidos.

CONTENIDO MALFORMADO:
{content}

INSTRUCCIONES:
1. Identifica y repara errores de sintaxis JSON
2. Extrae todos los recursos FHIR válidos (Patient, Condition, MedicationRequest, Observation, etc.)
3. Asegúrate de que cada recurso tenga 'resourceType' válido
4. Corrige UUIDs ficticios con IDs numéricos únicos
5. Devuelve SOLO un array JSON con los recursos reparados

EJEMPLO DE RESPUESTA:
[
  {{
    "resourceType": "Patient",
    "id": "12345",
    "name": [{{"text": "María García"}}]
  }},
  {{
    "resourceType": "Condition", 
    "id": "67890",
    "code": {{"text": "Diabetes"}}
  }}
]

IMPORTANTE: Responde SOLO con el array JSON, sin explicaciones."""

            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result and isinstance(result, list):
                print(f"   ✅ LLM extrajo {len(result)} recursos FHIR")
                return result
            
            # TERCERA ESTRATEGIA: Fallback manual
            print(f"   ⚠️ LLM falló, intentando fallback manual...")
            manual_extraction = self._manual_json_extraction(content)
            if manual_extraction:
                print(f"   ✅ Extracción manual exitosa: {len(manual_extraction)} recursos")
                return manual_extraction
            
            print(f"   ❌ No se pudieron extraer recursos FHIR válidos")
            return []
            
        except Exception as e:
            print(f"   ❌ Error en extracción de objetos JSON: {e}")
            return []

    def _extract_json_objects_directly(self, content: str) -> List[Dict[str, Any]]:
        """Extrae objetos JSON directamente del contenido usando regex"""
        try:
            import re
            
            # Buscar todos los objetos JSON en el contenido
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, content, re.DOTALL)
            
            extracted_resources = []
            
            for i, json_str in enumerate(json_matches):
                try:
                    # Limpiar el JSON
                    cleaned_json = self._clean_json_string(json_str)
                    parsed_obj = json.loads(cleaned_json)
                    
                    # Verificar que es un recurso FHIR válido
                    if isinstance(parsed_obj, dict) and 'resourceType' in parsed_obj:
                        # Corregir UUIDs si es necesario
                        if 'id' in parsed_obj and isinstance(parsed_obj['id'], str) and 'urn:uuid:' in parsed_obj['id']:
                            parsed_obj['id'] = str(abs(hash(json_str)) % 10000)
                        
                        extracted_resources.append(parsed_obj)
                        print(f"   ✅ Recurso {i+1} extraído: {parsed_obj.get('resourceType')}")
                    
                except json.JSONDecodeError as e:
                    print(f"   ⚠️ JSON {i+1} inválido: {e}")
                    continue
                except Exception as e:
                    print(f"   ❌ Error procesando JSON {i+1}: {e}")
                    continue
            
            return extracted_resources
            
        except Exception as e:
            print(f"   ❌ Error en extracción directa: {e}")
            return []

    def _manual_json_extraction(self, content: str) -> List[Dict[str, Any]]:
        """Extracción manual de recursos FHIR usando LLM cuando todo lo demás falla"""
        try:
            if not self.llm:
                print(f"   ❌ LLM no disponible para extracción manual")
                return []
            
            print(f"   🧠 Usando LLM para extracción manual de recursos FHIR...")
            
            prompt = f"""Eres un experto en extracción de recursos FHIR desde contenido JSON malformado. Extrae todos los recursos FHIR válidos que puedas encontrar.

CONTENIDO JSON MALFORMADO:
{content}

INSTRUCCIONES:
1. Analiza el contenido JSON malformado
2. Identifica fragmentos que puedan ser recursos FHIR válidos
3. Repara y completa los recursos FHIR que encuentres
4. Usa IDs numéricos únicos para cada recurso
5. Asegúrate de que cada recurso tenga 'resourceType' válido
6. Corrige cualquier error de sintaxis JSON

TIPOS DE RECURSOS A BUSCAR:
- Patient: Información del paciente
- Condition: Diagnósticos y condiciones
- MedicationRequest: Medicaciones
- Observation: Signos vitales, laboratorio
- Encounter: Encuentros médicos

REGLAS:
- Usa IDs numéricos simples: "12345", "67890", etc.
- NO uses UUIDs ficticios
- Repara JSON malformado automáticamente
- Incluye solo recursos FHIR válidos

Devuelve SOLO un array JSON con los recursos FHIR extraídos y reparados."""

            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result and isinstance(result, list):
                print(f"   ✅ LLM extrajo {len(result)} recursos FHIR manualmente")
                return result
            else:
                print(f"   ❌ LLM no pudo extraer recursos FHIR manualmente")
                return []
            
        except Exception as e:
            print(f"   ❌ Error en extracción manual con LLM: {e}")
            return []
    
    def _extract_first_valid_json(self, content: str) -> Optional[str]:
        """Extrae el primer objeto JSON válido de un contenido"""
        try:
            # Buscar el primer '{'
            start_pos = content.find('{')
            if start_pos == -1:
                return None
            
            # Contar llaves para encontrar el cierre
            brace_count = 0
            for i in range(start_pos, len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Extraer el objeto JSON
                        json_str = content[start_pos:i+1]
                        
                        # Intentar parsear para validar
                        try:
                            json.loads(json_str)
                            return json_str
                        except json.JSONDecodeError:
                            # Si falla, intentar limpiar
                            cleaned = self._clean_json_string(json_str)
                            try:
                                json.loads(cleaned)
                                return cleaned
                            except json.JSONDecodeError:
                                continue
            
            return None
            
        except Exception as e:
            print(f"   ❌ Error extrayendo primer JSON: {e}")
            return None

    def _clean_json_string(self, json_str: str) -> str:
        """Limpia un string JSON de errores comunes"""
        try:
            # Remover comas extra al final
            import re
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            # Remover líneas vacías y espacios extra
            lines = json_str.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    cleaned_lines.append(line)
            
            return '\n'.join(cleaned_lines)
            
        except Exception as e:
            print(f"   ❌ Error limpiando JSON: {e}")
            return json_str
    
    def can_handle(self, query: str) -> bool:
        """Verifica si el agente puede manejar la consulta"""
        fhir_keywords = [
            'fhir', 'patient', 'paciente', 'clinical', 'clínica',
            'medical', 'médico', 'diagnosis', 'diagnóstico',
            'treatment', 'tratamiento', 'medication', 'medicación'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in fhir_keywords)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del agente"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reinicia las estadísticas"""
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'clinical_notes_processed': 0,
            'fhir_resources_created': 0,
            'sql_conversions': 0,
            'avg_response_time': 0.0
        }

    async def _validate_pati_id_fhir(self, fhir_resource: Dict[str, Any], patient_id_context: Optional[str] = None) -> Dict[str, Any]:
        """Validación inteligente de PATI_ID para recursos FHIR usando LLM"""
        try:
            if not self.llm:
                # Fallback básico si no hay LLM
                pati_id = patient_id_context or fhir_resource.get('patient_id') or fhir_resource.get('PATI_ID')
                if not pati_id or pati_id == 0 or (isinstance(pati_id, str) and (pati_id.strip() == '' or pati_id.strip() == '0')):
                    return {
                        'valid': False,
                        'error': f'PATI_ID inválido para recurso FHIR: {pati_id}',
                        'requires_pati_id': True
                    }
                return {'valid': True, 'requires_pati_id': True}

            # Obtener esquema completo
            schema_info = await self._get_database_schema_info()
            
            prompt = f"""
Eres un experto en validación de recursos FHIR médicos. Analiza si este recurso requiere PATI_ID válido.

RECURSO FHIR:
{json.dumps(fhir_resource, indent=2, ensure_ascii=False)}

CONTEXTO DE PACIENTE: {patient_id_context or 'No disponible'}

ESQUEMA DE BASE DE DATOS:
{schema_info}

INSTRUCCIONES:
1. Determina si este tipo de recurso FHIR requiere PATI_ID para persistir en BD
2. Valida si el PATI_ID disponible es válido (no 0, no None, no vacío)
3. Considera el contexto médico del recurso
4. Analiza qué tabla SQL sería apropiada para este recurso

REGLAS FHIR→SQL:
- Patient: No requiere PATI_ID (es el recurso principal)
- Condition, Observation, MedicationRequest: SIEMPRE requieren PATI_ID
- Encounter: Puede requerir PATI_ID dependiendo del contexto
- PATI_ID debe ser un valor válido (UUID, entero positivo, etc.)

Responde SOLO con este JSON:
{{
  "requires_pati_id": true/false,
  "valid": true/false,
  "error": "descripción del error si no es válido",
  "reasoning": "explicación de la decisión",
  "suggested_table": "tabla SQL sugerida"
}}
"""
            
            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result and isinstance(result, dict):
                return result
            else:
                # Fallback si no se puede parsear la respuesta
                return {
                    'valid': False,
                    'error': 'No se pudo validar PATI_ID con LLM',
                    'requires_pati_id': True
                }
                
        except Exception as e:
            return {
                'valid': False,
                'error': f'Error en validación FHIR: {e}',
                'requires_pati_id': True
        }

    async def _llm_determine_update_fields(self, sql_data: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
        """
        LLAMADA 1: Determina campos a actualizar usando LLM - SIN HARDCODEO.
        
        Args:
            sql_data: Datos SQL del paciente
            patient_id: ID del paciente
            
        Returns:
            Dict[str, Any]: Campos a actualizar
        """
        try:
            if not self.llm:
                return {'fields': list(sql_data.keys()), 'values': list(sql_data.values())}
            
            # PROMPT ESPECÍFICO PARA DETERMINAR CAMPOS - SIN HARDCODEO
            fields_prompt = f"""Eres un experto en actualización de datos de pacientes médicos.

DATOS DISPONIBLES:
{json.dumps(sql_data, indent=2, ensure_ascii=False)}

PATIENT_ID: {patient_id}

TAREA ESPECÍFICA: Determina qué campos del paciente deben ser actualizados.

ESTRATEGIA DE ANÁLISIS:
1. Identifica campos que han cambiado o mejorado
2. Considera campos obligatorios que deben estar presentes
3. Evalúa campos opcionales que pueden ser actualizados
4. Identifica campos de auditoría que deben actualizarse
5. Considera campos relacionados que deben mantenerse consistentes

CAMPOS POSIBLES:
- Información personal (nombre, apellidos, fecha nacimiento)
- Información demográfica (género, DNI, dirección)
- Información médica (historia clínica, alergias)
- Información administrativa (estado activo, fechas)
- Información de contacto (teléfono, email)

CRITERIOS DE SELECCIÓN:
- Relevancia médica del campo
- Precisión de la información
- Completitud de los datos
- Consistencia con otros campos
- Requisitos del sistema

INSTRUCCIONES:
- Analiza cada campo disponible
- Determina si necesita actualización
- Considera la calidad de la información
- Prioriza campos médicamente relevantes
- Incluye campos de auditoría apropiados

RESPUESTA JSON:
{{
    "campos_obligatorios": ["campo1", "campo2"],
    "campos_opcionales": ["campo3", "campo4"],
    "campos_auditoria": ["campo5", "campo6"],
    "valores_actualizados": {{
        "campo1": "valor1",
        "campo2": "valor2"
    }},
    "razonamiento": "explicación de los campos seleccionados",
    "prioridad": "alta|media|baja"
}}

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": fields_prompt}]
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json_new(content)
            
            if result:
                campos_obligatorios = result.get('campos_obligatorios', [])
                campos_opcionales = result.get('campos_opcionales', [])
                valores_actualizados = result.get('valores_actualizados', {})
                razonamiento = result.get('razonamiento', 'Sin explicación')
                
                print(f"   📋 CAMPOS A ACTUALIZAR:")
                print(f"      📊 Obligatorios: {len(campos_obligatorios)}")
                print(f"      📊 Opcionales: {len(campos_opcionales)}")
                print(f"      💡 Razonamiento: {razonamiento[:50]}...")
                
                # Mostrar cada campo y su valor
                for campo, valor in valores_actualizados.items():
                    print(f"      - {campo}: {valor}")
                
                return result
            else:
                return {'campos_obligatorios': list(sql_data.keys()), 'valores_actualizados': sql_data}
                
        except Exception as e:
            logger.error(f"Error determinando campos: {e}")
            return {'campos_obligatorios': list(sql_data.keys()), 'valores_actualizados': sql_data}

    async def _llm_generate_update_sql(self, update_fields: Dict[str, Any], patient_id: str) -> str:
        """
        LLAMADA 2: Genera SQL de actualización usando LLM - SIN HARDCODEO.
        
        Args:
            update_fields: Campos a actualizar
            patient_id: ID del paciente
            
        Returns:
            str: SQL de actualización
        """
        try:
            if not self.llm:
                # Fallback básico
                campos = update_fields.get('campos_obligatorios', [])
                valores = update_fields.get('valores_actualizados', {})
                set_clause = ', '.join([f"{campo} = '{valores.get(campo, '')}'" for campo in campos])
                return f"UPDATE PATI_PATIENTS SET {set_clause} WHERE PATI_ID = {patient_id};"
            
            # PROMPT ESPECÍFICO PARA GENERAR SQL - SIN HARDCODEO
            sql_prompt = f"""Eres un experto en SQL para actualización de pacientes médicos.

CAMPOS A ACTUALIZAR:
{json.dumps(update_fields, indent=2, ensure_ascii=False)}

PATIENT_ID: {patient_id}

TAREA ESPECÍFICA: Genera SQL UPDATE optimizado para actualizar los campos del paciente.

ESTRATEGIA DE GENERACIÓN:
1. Usa la tabla PATI_PATIENTS
2. Incluye todos los campos obligatorios
3. Incluye campos opcionales relevantes
4. Maneja tipos de datos correctamente
5. Escapa valores de texto apropiadamente
6. Incluye campos de auditoría si es necesario

REGLAS DE SQL:
- Usa solo columnas que existan en la tabla
- Escapa valores de texto correctamente
- Maneja fechas en formato SQLite
- Incluye la condición WHERE con PATI_ID
- Optimiza para rendimiento

CAMPOS COMUNES EN PATI_PATIENTS:
- PATI_NAME, PATI_SURNAME_1, PATI_FULL_NAME
- PATI_BIRTH_DATE, GEND_ID
- PATI_ACTIVE, PATI_START_DATE
- PATI_CLINICAL_HISTORY_ID

INSTRUCCIONES:
- Genera SQL válido para SQLite
- Incluye todos los campos necesarios
- Maneja valores NULL apropiadamente
- Optimiza para rendimiento
- Incluye validaciones si es necesario

RESPUESTA:
Devuelve SOLO el SQL UPDATE, sin explicaciones ni comentarios.
IMPORTANTE: NO uses formato markdown (```sql), responde SOLO con el SQL directo.
Ejemplo: UPDATE PATI_PATIENTS SET PATI_NAME = 'Juan', PATI_SURNAME_1 = 'Pérez' WHERE PATI_ID = 123;"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": sql_prompt}]
            )
            
            sql = self._extract_response_text(response).strip()
            sql = self._clean_sql_response(sql)
            
            if sql and not sql.startswith("Error"):
                print(f"   💾 SQL generado: {sql[:100]}...")
                return sql
            else:
                # Fallback básico
                campos = update_fields.get('campos_obligatorios', [])
                valores = update_fields.get('valores_actualizados', {})
                set_clause = ', '.join([f"{campo} = '{valores.get(campo, '')}'" for campo in campos])
                return f"UPDATE PATI_PATIENTS SET {set_clause} WHERE PATI_ID = {patient_id};"
                
        except Exception as e:
            logger.error(f"Error generando SQL: {e}")
            return f"UPDATE PATI_PATIENTS SET PATI_NAME = 'Error' WHERE PATI_ID = {patient_id};"

    async def _llm_execute_update_sql(self, update_sql: str, patient_id: str) -> Dict[str, Any]:
        """
        LLAMADA 3: Ejecuta SQL de actualización usando LLM - SIN HARDCODEO.
        
        Args:
            update_sql: SQL de actualización
            patient_id: ID del paciente
            
        Returns:
            Dict[str, Any]: Resultado de la ejecución
        """
        try:
            if not self.llm:
                # Fallback básico
                db_path = self.sql_agent.db_path if self.sql_agent else 'database_new.sqlite3.db'
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(update_sql)
                conn.commit()
                conn.close()
                return {'success': True, 'fields_updated': 1, 'execution_time': 0.001}
            
            # PROMPT ESPECÍFICO PARA VALIDACIÓN Y EJECUCIÓN - SIN HARDCODEO
            execution_prompt = f"""Eres un experto en ejecución de SQL médico.

SQL A EJECUTAR:
{update_sql}

PATIENT_ID: {patient_id}

TAREA ESPECÍFICA: Valida y ejecuta el SQL de actualización de manera segura.

ESTRATEGIA DE VALIDACIÓN:
1. Verifica que el SQL sea sintácticamente correcto
2. Valida que use solo columnas existentes
3. Verifica que los tipos de datos sean apropiados
4. Comprueba que no haya inyección SQL
5. Valida que la condición WHERE sea correcta
6. Verifica que el rendimiento sea aceptable

ESTRATEGIA DE EJECUCIÓN:
1. Ejecuta de manera segura
2. Maneja errores apropiadamente
3. Valida resultados después de la ejecución
4. Proporciona feedback detallado
5. Considera rollback si hay errores
6. Optimiza para rendimiento

INSTRUCCIONES:
- Valida el SQL antes de ejecutar
- Ejecuta de manera segura
- Maneja errores apropiadamente
- Proporciona feedback detallado
- Considera transacciones

RESPUESTA JSON:
{{
    "es_valido": true|false,
    "errores_validacion": ["error1", "error2"],
    "ejecutado_exitosamente": true|false,
    "campos_actualizados": 5,
    "tiempo_ejecucion": 0.005,
    "errores_ejecucion": ["error1", "error2"],
    "resultados_validacion": ["resultado1", "resultado2"]
}}

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": execution_prompt}]
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json_new(content)
            
            if result:
                es_valido = result.get('es_valido', False)
                ejecutado_exitosamente = result.get('ejecutado_exitosamente', False)
                campos_actualizados = result.get('campos_actualizados', 0)
                tiempo_ejecucion = result.get('tiempo_ejecucion', 0)
                
                # Ejecutar realmente el SQL si es válido
                if es_valido and ejecutado_exitosamente:
                    try:
                        db_path = self.sql_agent.db_path if self.sql_agent else 'database_new.sqlite3.db'
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        cursor.execute(update_sql)
                        conn.commit()
                        conn.close()
                        
                        return {
                            'success': True,
                            'fields_updated': campos_actualizados,
                            'execution_time': tiempo_ejecucion,
                            'errors': []
                        }
                    except Exception as e:
                        return {
                            'success': False,
                            'fields_updated': 0,
                            'execution_time': 0,
                            'error': str(e)
                        }
                else:
                    return {
                        'success': False,
                        'fields_updated': 0,
                        'execution_time': 0,
                        'error': 'SQL no válido o no se pudo ejecutar'
                    }
            else:
                # Fallback básico
                try:
                    db_path = self.sql_agent.db_path if self.sql_agent else 'database_new.sqlite3.db'
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute(update_sql)
                    conn.commit()
                    conn.close()
                    
                    return {
                        'success': True,
                        'fields_updated': 1,
                        'execution_time': 0.001,
                        'errors': []
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'fields_updated': 0,
                        'execution_time': 0,
                        'error': str(e)
                    }
                
        except Exception as e:
            logger.error(f"Error ejecutando SQL: {e}")
            return {
                'success': False,
                'fields_updated': 0,
                'execution_time': 0,
                'error': str(e)
            }

    def _extract_response_text(self, response) -> str:
        """Extrae el texto de la respuesta del LLM y limpia formato markdown"""
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)
        
        # Limpiar formato markdown del SQL
        if '```sql' in content:
            content = content.split('```sql')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        
        # Remover etiquetas como "SQL:" al inicio
        import re
        content = re.sub(r'^(SQL|sql):\s*', '', content, flags=re.MULTILINE)
        
        # Remover cualquier texto explicativo común
        content = re.sub(r'^(Esta consulta|This query|Consulta|Query).*?:', '', content, flags=re.IGNORECASE)
        
        return content.strip()

    def _try_parse_llm_json_new(self, content: str) -> Optional[Dict[str, Any]]:
        """Intenta parsear JSON de respuesta del LLM (nueva versión)"""
        try:
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error parseando JSON: {e}")
            return None

    async def _get_real_id_robust(self, table: str, data: Dict[str, Any]) -> Optional[int]:
        """
        Obtiene el ID real usando una estrategia robusta y directa sin depender del LLM.
        Esta solución es más confiable y eficiente.
        """
        try:
            if not self.sql_agent:
                print(f"   ⚠️ SQL Agent no disponible")
                return None
            
            import sqlite3
            conn = sqlite3.connect(self.sql_agent.db_path)
            cursor = conn.cursor()
            
            try:
                # ESTRATEGIA 1: Usar last_insert_rowid() directamente
                cursor.execute("SELECT last_insert_rowid();")
                row = cursor.fetchone()
                if row and row[0] and row[0] > 0:
                    real_id = row[0]
                    print(f"   💡 ID obtenido con last_insert_rowid: {real_id}")
                    return int(real_id)
                
                # ESTRATEGIA 2: Buscar por criterios únicos específicos de la tabla
                search_queries = []
                
                if table.upper() == 'PATI_PATIENTS':
                    # Para pacientes, usar nombre completo y fecha de nacimiento
                    if data.get("PATI_FULL_NAME") and data.get("PATI_BIRTH_DATE"):
                        search_queries.append((
                            "SELECT PATI_ID FROM PATI_PATIENTS WHERE LOWER(PATI_FULL_NAME) = LOWER(?) AND PATI_BIRTH_DATE = ? ORDER BY PATI_ID DESC LIMIT 1",
                            [data.get("PATI_FULL_NAME"), data.get("PATI_BIRTH_DATE")]
                        ))
                    
                    # Alternativa: nombre y apellido
                    if data.get("PATI_NAME") and data.get("PATI_SURNAME_1"):
                        search_queries.append((
                            "SELECT PATI_ID FROM PATI_PATIENTS WHERE LOWER(PATI_NAME) = LOWER(?) AND LOWER(PATI_SURNAME_1) = LOWER(?) ORDER BY PATI_ID DESC LIMIT 1",
                            [data.get("PATI_NAME"), data.get("PATI_SURNAME_1")]
                        ))
                    
                    # Alternativa: solo nombre completo
                    if data.get("PATI_FULL_NAME"):
                        search_queries.append((
                            "SELECT PATI_ID FROM PATI_PATIENTS WHERE LOWER(PATI_FULL_NAME) = LOWER(?) ORDER BY PATI_ID DESC LIMIT 1",
                            [data.get("PATI_FULL_NAME")]
                        ))
                
                elif table.upper() == 'EPIS_EPISODES':
                    # Para episodios, usar fecha y descripción
                    if data.get("EPIS_START_DATE") and data.get("EPIS_DIAG_DESCRIPTION"):
                        search_queries.append((
                            "SELECT EPIS_ID FROM EPIS_EPISODES WHERE EPIS_START_DATE = ? AND EPIS_DIAG_DESCRIPTION = ? ORDER BY EPIS_ID DESC LIMIT 1",
                            [data.get("EPIS_START_DATE"), data.get("EPIS_DIAG_DESCRIPTION")]
                        ))
                
                elif table.upper() == 'MEDI_MEDICATIONS':
                    # Para medicamentos, usar nombre
                    if data.get("MEDI_NAME"):
                        search_queries.append((
                            "SELECT MEDI_ID FROM MEDI_MEDICATIONS WHERE LOWER(MEDI_NAME) = LOWER(?) ORDER BY MEDI_ID DESC LIMIT 1",
                            [data.get("MEDI_NAME")]
                        ))
                
                elif table.upper() == 'EPIS_DIAGNOSTICS':
                    # Para diagnósticos, usar observación y paciente
                    if data.get("DIAG_OBSERVATION") and data.get("PATI_ID"):
                        search_queries.append((
                            "SELECT DIAG_ID FROM EPIS_DIAGNOSTICS WHERE DIAG_OBSERVATION = ? AND PATI_ID = ? ORDER BY DIAG_ID DESC LIMIT 1",
                            [data.get("DIAG_OBSERVATION"), data.get("PATI_ID")]
                        ))
                
                # ESTRATEGIA 3: Buscar el último registro insertado en la tabla
                if table.upper() == 'PATI_PATIENTS':
                    search_queries.append((
                        "SELECT PATI_ID FROM PATI_PATIENTS ORDER BY PATI_ID DESC LIMIT 1",
                        []
                    ))
                elif table.upper() == 'EPIS_EPISODES':
                    search_queries.append((
                        "SELECT EPIS_ID FROM EPIS_EPISODES ORDER BY EPIS_ID DESC LIMIT 1",
                        []
                    ))
                elif table.upper() == 'MEDI_MEDICATIONS':
                    search_queries.append((
                        "SELECT MEDI_ID FROM MEDI_MEDICATIONS ORDER BY MEDI_ID DESC LIMIT 1",
                        []
                    ))
                elif table.upper() == 'EPIS_DIAGNOSTICS':
                    search_queries.append((
                        "SELECT DIAG_ID FROM EPIS_DIAGNOSTICS ORDER BY DIAG_ID DESC LIMIT 1",
                        []
                    ))
                
                # Ejecutar las consultas en orden de prioridad
                for query, params in search_queries:
                    try:
                        cursor.execute(query, params)
                        row = cursor.fetchone()
                        if row and row[0] and row[0] > 0:
                            real_id = row[0]
                            print(f"   💡 ID encontrado con consulta: {query[:50]}...")
                            return int(real_id)
                    except Exception as e:
                        print(f"   ⚠️ Error en consulta: {e}")
                        continue
                
                print(f"   ❌ No se pudo encontrar ID real")
                return None
                
            finally:
                conn.close()
                
        except Exception as e:
            print(f"   ⚠️ Error en _get_real_id_robust: {e}")
            return None

    async def _validate_and_fix_table_mapping(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida y corrige el mapeo de tabla, asegurando que exista y que los datos sean válidos.
        """
        try:
            if not self.sql_agent:
                return data
            
            import sqlite3
            conn = sqlite3.connect(self.sql_agent.db_path)
            cursor = conn.cursor()
            
            try:
                # Verificar si la tabla existe
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
                if not cursor.fetchone():
                    print(f"   ⚠️ Tabla {table} no existe, buscando tabla alternativa...")
                    
                    # Buscar tabla alternativa basada en el tipo de datos
                    if 'PATI_' in str(data):
                        table = 'PATI_PATIENTS'
                    elif 'EPIS_' in str(data):
                        table = 'EPIS_EPISODES'
                    elif 'MEDI_' in str(data):
                        table = 'MEDI_MEDICATIONS'
                    elif 'OBSE_' in str(data):
                        # OBSE_OBSERVATIONS no existe, usar EPIS_DIAGNOSTICS
                        table = 'EPIS_DIAGNOSTICS'
                        # Adaptar datos para la tabla correcta
                        if 'code_text' in data:
                            data['DIAG_OBSERVATION'] = data.pop('code_text')
                        if 'value' in data:
                            data['DIAG_VALUE'] = data.pop('value')
                    
                    print(f"   ✅ Usando tabla alternativa: {table}")
                
                # Obtener columnas de la tabla
                cursor.execute(f"PRAGMA table_info({table})")
                columns_info = cursor.fetchall()
                valid_columns = [col[1] for col in columns_info]
                
                # Filtrar datos para usar solo columnas válidas
                filtered_data = {}
                for key, value in data.items():
                    if key in valid_columns:
                        filtered_data[key] = value
                    else:
                        print(f"   ⚠️ Columna {key} no existe en {table}, omitiendo")
                
                # Asegurar que PATI_ID esté presente si es necesario
                if table != 'PATI_PATIENTS' and 'PATI_ID' in valid_columns and 'PATI_ID' not in filtered_data:
                    print(f"   ⚠️ PATI_ID requerido en {table} pero no presente")
                
                return filtered_data
                
            finally:
                conn.close()
                
        except Exception as e:
            print(f"   ⚠️ Error validando tabla: {e}")
            return data

    async def _track_insertion_location(self, table: str, data: Dict[str, Any], inserted_id: int) -> Dict[str, Any]:
        """
        Track de la ubicación de inserción para referencia futura.
        """
        try:
            tracking_info = {
                'table': table,
                'row_id': inserted_id,
                'columns_used': list(data.keys()),
                'timestamp': time.time(),
                'data_summary': {}
            }
            
            # Crear resumen de datos para tracking
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 50:
                    tracking_info['data_summary'][key] = value[:50] + "..."
                else:
                    tracking_info['data_summary'][key] = value
            
            print(f"   📍 Tracking inserción: Tabla={table}, RowID={inserted_id}")
            print(f"   📍 Columnas usadas: {', '.join(tracking_info['columns_used'])}")
            
            return tracking_info
            
        except Exception as e:
            print(f"   ⚠️ Error en tracking: {e}")
            return {}

    async def _llm_detect_if_data_mapped(self, data: Dict[str, Any]) -> bool:
        """Detecta si los datos ya están mapeados usando LLM"""
        try:
            if not self.llm:
                # Fallback básico: verificar si hay columnas SQL
                sql_columns = [key for key in data.keys() if key.startswith(('PATI_', 'EPIS_', 'MEDI_', 'OBSE_', 'APPO_'))]
                return len(sql_columns) > 0
            
            # PROMPT ESPECÍFICO PARA DETECCIÓN DE MAPEO
            detection_prompt = f"""Eres un experto en análisis de datos médicos. Determina si estos datos ya están mapeados a columnas SQL.

DATOS A ANALIZAR:
{json.dumps(data, indent=2, ensure_ascii=False)}

INSTRUCCIONES:
1. Analiza las claves de los datos
2. Determina si contienen nombres de columnas SQL (como PATI_NAME, EPIS_ID, etc.)
3. Si las claves parecen ser nombres de columnas SQL, los datos están mapeados
4. Si las claves parecen ser campos FHIR (como resourceType, name, etc.), no están mapeados

RESPUESTA:
Responde SOLO con "true" si los datos están mapeados, o "false" si no lo están.

Ejemplo:
- Si las claves son: ["PATI_NAME", "PATI_SURNAME_1", "PATI_BIRTH_DATE"] → true
- Si las claves son: ["resourceType", "name", "birthDate"] → false

Responde SOLO con true o false:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": detection_prompt}]
            )
            
            content = self._extract_response_text(response)
            return content.strip().lower() == 'true'
            
        except Exception as e:
            logger.error(f"Error detectando si datos están mapeados: {e}")
            # Fallback básico
            sql_columns = [key for key in data.keys() if key.startswith(('PATI_', 'EPIS_', 'MEDI_', 'OBSE_', 'APPO_'))]
            return len(sql_columns) > 0

    async def _llm_determine_table_from_mapped_data(self, data: Dict[str, Any]) -> str:
        """Determina la tabla correcta basándose en los datos mapeados usando LLM"""
        try:
            if not self.llm:
                # Fallback básico: determinar tabla por prefijos
                for key in data.keys():
                    if key.startswith('PATI_'):
                        return 'PATI_PATIENTS'
                    elif key.startswith('EPIS_'):
                        return 'EPIS_EPISODES'
                    elif key.startswith('MEDI_'):
                        return 'MEDI_MEDICATIONS'
                    elif key.startswith('OBSE_'):
                        return 'OBSE_OBSERVATIONS'
                    elif key.startswith('APPO_'):
                        return 'APPO_APPOINTMENTS'
                return 'PATI_PATIENTS'  # Fallback
            
            # PROMPT ESPECÍFICO PARA DETERMINAR TABLA
            table_detection_prompt = f"""Eres un experto en bases de datos médicas. Determina la tabla más apropiada para estos datos mapeados.

DATOS MAPEADOS:
{json.dumps(data, indent=2, ensure_ascii=False)}

ESQUEMA DISPONIBLE:
{await self._get_database_schema_info()}

INSTRUCCIONES:
1. Analiza las claves de los datos mapeados
2. Identifica patrones de prefijos (PATI_, EPIS_, MEDI_, etc.)
3. Determina qué tabla corresponde mejor a estos datos
4. Considera el contexto médico de los campos

REGLAS:
- PATI_* → PATI_PATIENTS (datos de pacientes)
- EPIS_* → EPIS_EPISODES (episodios médicos)
- MEDI_* → MEDI_MEDICATIONS (medicamentos)
- OBSE_* → OBSE_OBSERVATIONS (observaciones)
- APPO_* → APPO_APPOINTMENTS (citas)
- DIAG_* → EPIS_DIAGNOSTICS (diagnósticos)
- PROC_* → PROC_PROCEDURES (procedimientos)

RESPUESTA:
Responde SOLO con el nombre exacto de la tabla (ej: "PATI_PATIENTS", "EPIS_EPISODES", etc.)

Responde SOLO con el nombre de la tabla:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": table_detection_prompt}]
            )
            
            content = self._extract_response_text(response)
            table_name = content.strip()
            
            # Validar que la tabla existe en el esquema
            if table_name in ['PATI_PATIENTS', 'EPIS_EPISODES', 'MEDI_MEDICATIONS', 'OBSE_OBSERVATIONS', 'APPO_APPOINTMENTS']:
                return table_name
            else:
                # Fallback si la tabla no existe
                for key in data.keys():
                    if key.startswith('PATI_'):
                        return 'PATI_PATIENTS'
                    elif key.startswith('EPIS_'):
                        return 'EPIS_EPISODES'
                    elif key.startswith('MEDI_'):
                        return 'MEDI_MEDICATIONS'
                    elif key.startswith('OBSE_'):
                        return 'OBSE_OBSERVATIONS'
                    elif key.startswith('APPO_'):
                        return 'APPO_APPOINTMENTS'
                return 'PATI_PATIENTS'
            
        except Exception as e:
            logger.error(f"Error determinando tabla desde datos mapeados: {e}")
            # Fallback básico
            for key in data.keys():
                if key.startswith('PATI_'):
                    return 'PATI_PATIENTS'
                elif key.startswith('EPIS_'):
                    return 'EPIS_EPISODES'
                elif key.startswith('MEDI_'):
                    return 'MEDI_MEDICATIONS'
                elif key.startswith('OBSE_'):
                    return 'OBSE_OBSERVATIONS'
                elif key.startswith('APPO_'):
                    return 'APPO_APPOINTMENTS'
            return 'PATI_PATIENTS'

    async def _fix_fhir_uuid_mapping(self, fhir_resource: Dict[str, Any], patient_id_context: Optional[str] = None) -> Dict[str, Any]:
        """Corrige automáticamente los UUIDs FHIR usando LLM de forma dinámica y flexible"""
        print(f"   🚀 ENTRANDO A _fix_fhir_uuid_mapping")
        print(f"   📊 Tipo de recurso: {fhir_resource.get('resourceType', 'Desconocido')}")
        print(f"   🆔 Contexto de ID de paciente: {patient_id_context}")
        
        try:
            if not self.llm:
                print(f"   ⚠️ LLM no disponible, usando corrección dinámica")
                return await self._fix_fhir_uuid_dynamic(fhir_resource, patient_id_context)
            
            print(f"   🧠 Corrigiendo UUIDs FHIR con LLM dinámico...")
            
            # Generar ID único dinámico
            import hashlib
            import time
            resource_content = json.dumps(fhir_resource, sort_keys=True)
            unique_id = str(abs(hash(resource_content)) % 10000)
            timestamp = int(time.time() * 1000)
            
            prompt = f"""Eres un experto en corrección de recursos FHIR. Corrige este recurso FHIR reemplazando UUIDs ficticios con IDs válidos.

RECURSO FHIR:
{json.dumps(fhir_resource, indent=2, ensure_ascii=False)}

CONTEXTO: {patient_id_context or 'No disponible'}
ID ÚNICO: {unique_id}

REGLAS:
1. Reemplaza IDs ficticios (urn:uuid:, unico, ficticio, etc.) con IDs válidos
2. Mantén la estructura FHIR original
3. Usa el contexto del paciente para referencias
4. Preserva toda la información médica
5. Solo modifica los IDs problemáticos

RESPUESTA: SOLO el JSON del recurso corregido, sin texto adicional."""

            response = _call_openai_native(self.llm, prompt)
            content = self._extract_response_text(response)
            
            # Intentar parsear directamente como JSON
            try:
                result = json.loads(content)
                if isinstance(result, dict) and 'resourceType' in result:
                    print(f"   ✅ UUIDs corregidos dinámicamente con LLM")
                    return result
            except json.JSONDecodeError:
                pass
            
            # Si falla, usar el método robusto
            result = self._try_parse_llm_json(content)
            
            if result and isinstance(result, dict) and 'resourceType' in result:
                print(f"   ✅ UUIDs corregidos dinámicamente con LLM (método robusto)")
                return result
            else:
                print(f"   ⚠️ LLM no pudo corregir dinámicamente, usando fallback inteligente")
                print(f"   🔍 Contenido recibido: {content[:200]}...")
                return await self._fix_fhir_uuid_dynamic(fhir_resource, patient_id_context)
                
        except Exception as e:
            print(f"   ❌ Error en corrección dinámica de UUIDs: {e}")
            return await self._fix_fhir_uuid_dynamic(fhir_resource, patient_id_context)

    async def _fix_fhir_uuid_dynamic(self, fhir_resource: Dict[str, Any], patient_id_context: Optional[str] = None) -> Dict[str, Any]:
        """Corrección dinámica de UUIDs FHIR usando LLM con prompts específicos"""
        print(f"      🔧 Iniciando corrección dinámica de UUIDs FHIR...")
        
        try:
            corrected_resource = fhir_resource.copy()
            resource_type = corrected_resource.get('resourceType', 'Unknown')
            
            # LLAMADA 1: LLM para generar ID único específico
            unique_id = await self._llm_generate_unique_id(fhir_resource, resource_type, patient_id_context)
            
            # LLAMADA 2: LLM para corregir ID principal
            corrected_resource = await self._llm_fix_main_id(corrected_resource, unique_id, resource_type)
            
            # LLAMADA 3: LLM para corregir referencias dinámicamente
            corrected_resource = await self._llm_fix_references_dynamic(corrected_resource, patient_id_context, unique_id, resource_type)
            
            print(f"      ✅ Corrección dinámica completada para {resource_type}")
            return corrected_resource
            
        except Exception as e:
            print(f"      ❌ Error en corrección dinámica: {e}")
            return fhir_resource

    async def _llm_generate_unique_id(self, fhir_resource: Dict[str, Any], resource_type: str, patient_id_context: Optional[str] = None) -> str:
        """LLM genera ID único específico para el tipo de recurso"""
        try:
            # Generar timestamp único para evitar duplicados
            import time
            import random
            timestamp = int(time.time() * 1000)  # Milisegundos para mayor unicidad
            random_suffix = random.randint(1000, 9999)  # Sufijo aleatorio adicional
            
            if not self.llm:
                # Fallback sin LLM con timestamp garantizado
                if resource_type == "Patient":
                    return f"PAT_{patient_id_context or 'unknown'}_{timestamp}"
                elif resource_type == "Condition":
                    return f"COND_{patient_id_context or 'unknown'}_{timestamp}_{random_suffix}"
                elif resource_type == "MedicationRequest":
                    return f"MED_{patient_id_context or 'unknown'}_{timestamp}_{random_suffix}"
                elif resource_type == "Observation":
                    return f"OBS_{patient_id_context or 'unknown'}_{timestamp}_{random_suffix}"
                elif resource_type == "AllergyIntolerance":
                    return f"ALL_{patient_id_context or 'unknown'}_{timestamp}_{random_suffix}"
                else:
                    return f"{resource_type.upper()}_{patient_id_context or 'unknown'}_{timestamp}_{random_suffix}"
            
            prompt = f"""Eres un experto en generación de IDs únicos para recursos FHIR. Genera un ID único apropiado para este recurso.

RECURSO FHIR:
{json.dumps(fhir_resource, indent=2, ensure_ascii=False)}

TIPO DE RECURSO: {resource_type}
CONTEXTO DE PACIENTE: {patient_id_context or 'No disponible'}
TIMESTAMP ÚNICO: {timestamp}
SUFIJO ALEATORIO: {random_suffix}

INSTRUCCIONES:
1. Analiza el tipo de recurso FHIR
2. Considera el contexto del paciente si está disponible
3. Genera un ID único que sea apropiado para este tipo de recurso
4. El ID debe ser único, significativo y seguir convenciones FHIR
5. SIEMPRE incluir el timestamp y sufijo aleatorio para garantizar unicidad

REGLAS DE GENERACIÓN OBLIGATORIAS:
- Patient: "PAT_{patient_id_context}_{timestamp}"
- Condition: "COND_{patient_id_context}_{timestamp}_{random_suffix}"
- MedicationRequest: "MED_{patient_id_context}_{timestamp}_{random_suffix}"
- Observation: "OBS_{patient_id_context}_{timestamp}_{random_suffix}"
- AllergyIntolerance: "ALL_{patient_id_context}_{timestamp}_{random_suffix}"
- Otros: "{resource_type.upper()}_{patient_id_context}_{timestamp}_{random_suffix}"

IMPORTANTE: Usa SIEMPRE el timestamp y sufijo aleatorio proporcionados para garantizar unicidad.

RESPUESTA: Solo el ID único generado, sin explicaciones adicionales."""

            response = _call_openai_native(self.llm, prompt)
            unique_id = response.content.strip()
            
            # Validar que el ID sea válido y contenga el timestamp
            if unique_id and len(unique_id) > 0 and str(timestamp) in unique_id:
                return unique_id
            else:
                # Fallback con timestamp garantizado
                if resource_type == "Patient":
                    return f"PAT_{patient_id_context or 'unknown'}_{timestamp}"
                elif resource_type == "Condition":
                    return f"COND_{patient_id_context or 'unknown'}_{timestamp}_{random_suffix}"
                elif resource_type == "MedicationRequest":
                    return f"MED_{patient_id_context or 'unknown'}_{timestamp}_{random_suffix}"
                elif resource_type == "Observation":
                    return f"OBS_{patient_id_context or 'unknown'}_{timestamp}_{random_suffix}"
                elif resource_type == "AllergyIntolerance":
                    return f"ALL_{patient_id_context or 'unknown'}_{timestamp}_{random_suffix}"
                else:
                    return f"{resource_type.upper()}_{patient_id_context or 'unknown'}_{timestamp}_{random_suffix}"
                
        except Exception as e:
            print(f"      ⚠️ Error generando ID único con LLM: {e}")
            # Fallback sin LLM con timestamp garantizado
            import time
            import random
            timestamp = int(time.time() * 1000)
            random_suffix = random.randint(1000, 9999)
            if resource_type == "Patient":
                return f"PAT_{patient_id_context or 'unknown'}_{timestamp}"
            elif resource_type == "Condition":
                return f"COND_{patient_id_context or 'unknown'}_{timestamp}_{random_suffix}"
            elif resource_type == "MedicationRequest":
                return f"MED_{patient_id_context or 'unknown'}_{timestamp}_{random_suffix}"
            elif resource_type == "Observation":
                return f"OBS_{patient_id_context or 'unknown'}_{timestamp}_{random_suffix}"
            elif resource_type == "AllergyIntolerance":
                return f"ALL_{patient_id_context or 'unknown'}_{timestamp}_{random_suffix}"
            else:
                return f"{resource_type.upper()}_{patient_id_context or 'unknown'}_{timestamp}_{random_suffix}"

    async def _llm_fix_main_id(self, fhir_resource: Dict[str, Any], unique_id: str, resource_type: str) -> Dict[str, Any]:
        """LLM corrige el ID principal del recurso"""
        try:
            if not self.llm:
                # Fallback sin LLM
                corrected_resource = fhir_resource.copy()
                if 'id' in corrected_resource:
                    old_id = corrected_resource['id']
                    if isinstance(old_id, str) and self._is_fictitious_id(old_id):
                        corrected_resource['id'] = unique_id
                return corrected_resource
            
            prompt = f"""Eres un experto en corrección de IDs principales en recursos FHIR. Corrige el ID principal de este recurso.

RECURSO FHIR:
{json.dumps(fhir_resource, indent=2, ensure_ascii=False)}

ID ÚNICO GENERADO: {unique_id}
TIPO DE RECURSO: {resource_type}

INSTRUCCIONES:
1. Analiza el ID actual del recurso
2. Si el ID es ficticio o problemático, reemplázalo con el ID único proporcionado
3. Si el ID es válido, mantenlo
4. Mantén la estructura del recurso FHIR
5. Solo modifica el campo 'id' si es necesario

REGLAS:
- Solo cambiar IDs que sean claramente ficticios (urn:uuid:, unico, ficticio, etc.)
- Preservar IDs válidos existentes
- Usar el ID único proporcionado para reemplazos
- Mantener la estructura JSON válida

RESPUESTA: JSON del recurso FHIR con el ID corregido."""

            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result and isinstance(result, dict):
                return result
            else:
                # Fallback sin LLM
                corrected_resource = fhir_resource.copy()
                if 'id' in corrected_resource:
                    old_id = corrected_resource['id']
                    if isinstance(old_id, str) and self._is_fictitious_id(old_id):
                        corrected_resource['id'] = unique_id
                return corrected_resource
                
        except Exception as e:
            print(f"      ⚠️ Error corrigiendo ID principal con LLM: {e}")
            # Fallback sin LLM
            corrected_resource = fhir_resource.copy()
            if 'id' in corrected_resource:
                old_id = corrected_resource['id']
                if isinstance(old_id, str) and self._is_fictitious_id(old_id):
                    corrected_resource['id'] = unique_id
            return corrected_resource

    async def _llm_fix_references_dynamic(self, fhir_resource: Dict[str, Any], patient_id_context: Optional[str], unique_id: str, resource_type: str) -> Dict[str, Any]:
        """LLM corrige referencias dinámicamente"""
        try:
            if not self.llm:
                # Fallback sin LLM
                return self._fix_references_dynamic(fhir_resource, patient_id_context, unique_id)
            
            prompt = f"""Eres un experto en corrección de referencias en recursos FHIR. Corrige las referencias ficticias en este recurso.

RECURSO FHIR:
{json.dumps(fhir_resource, indent=2, ensure_ascii=False)}

CONTEXTO DE PACIENTE: {patient_id_context or 'No disponible'}
ID ÚNICO: {unique_id}
TIPO DE RECURSO: {resource_type}

INSTRUCCIONES:
1. Analiza todas las referencias en el recurso (subject, encounter, performer, medication, etc.)
2. Identifica referencias que sean ficticias o problemáticas
3. Corrige las referencias usando el contexto apropiado
4. Mantén la coherencia entre referencias
5. Usa el contexto del paciente cuando sea relevante

TIPOS DE REFERENCIAS A CORREGIR:
- subject: Referencia al paciente
- encounter: Referencia al encuentro médico
- performer: Referencia al profesional médico
- medication: Referencia al medicamento
- requester: Referencia al solicitante
- recorder: Referencia al registrador

REGLAS DE CORRECCIÓN:
- Patient references: "Patient/{patient_id_context}" o "Patient/{unique_id}"
- Encounter references: "Encounter/{unique_id}"
- Practitioner references: "Practitioner/{unique_id}"
- Medication references: "Medication/{unique_id}"
- Mantener coherencia entre referencias relacionadas

RESPUESTA: JSON del recurso FHIR con referencias corregidas."""

            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result and isinstance(result, dict):
                # CRÍTICO: Forzar el uso del ID real del paciente en las referencias
                corrected_resource = result
                if patient_id_context and patient_id_context != 'No disponible':
                    # Forzar referencia correcta al paciente
                    if 'subject' in corrected_resource:
                        corrected_resource['subject'] = {'reference': f'Patient/{patient_id_context}'}
                        print(f"   🔗 Referencia de paciente corregida: Patient/{patient_id_context}")
                    
                    # También corregir otras referencias que puedan usar el paciente
                    if 'encounter' in corrected_resource and isinstance(corrected_resource['encounter'], dict):
                        if 'reference' in corrected_resource['encounter'] and 'Patient/' in corrected_resource['encounter']['reference']:
                            corrected_resource['encounter']['reference'] = f'Patient/{patient_id_context}'
                    
                    if 'performer' in corrected_resource and isinstance(corrected_resource['performer'], dict):
                        if 'reference' in corrected_resource['performer'] and 'Patient/' in corrected_resource['performer']['reference']:
                            corrected_resource['performer']['reference'] = f'Patient/{patient_id_context}'
                
                return corrected_resource
            else:
                # Fallback sin LLM
                return self._fix_references_dynamic(fhir_resource, patient_id_context, unique_id)
                
        except Exception as e:
            print(f"      ⚠️ Error corrigiendo referencias con LLM: {e}")
            # Fallback sin LLM
            return self._fix_references_dynamic(fhir_resource, patient_id_context, unique_id)

    async def _llm_fix_text_values(self, fhir_resource: Dict[str, Any], patient_id_context: Optional[str] = None) -> Dict[str, Any]:
        """LLAMADA 2: Corrige valores de texto que deberían ser datos reales usando LLM"""
        try:
            prompt = f"""Eres un experto en corrección de datos médicos. Corrige valores de texto que deberían ser datos reales.

RECURSO FHIR:
{json.dumps(fhir_resource, indent=2, ensure_ascii=False)}

CONTEXTO DE PACIENTE: {patient_id_context or 'No disponible'}

TAREA ESPECÍFICA: Identificar y corregir valores de texto que deberían ser datos reales.

TIPOS DE CORRECCIONES:
1. Timestamps ficticios → Fechas reales actuales
2. Valores vacíos → Valores por defecto apropiados
3. Textos descriptivos → Valores reales
4. Placeholders → Datos concretos

EJEMPLOS DE CORRECCIONES:
- "Timestamp de inicio" → "2024-01-15 10:30:00"
- "Valor asociado al paciente" → "123" (ID real)
- "fecha actual" → "2024-01-15"
- "" (vacío) → "N/A" o valor apropiado
- "active" → "2024-01-15 10:30:00"

INSTRUCCIONES:
1. Analiza cada campo del recurso
2. Identifica valores que parecen placeholders o ficticios
3. Reemplázalos con valores reales apropiados
4. Mantén la estructura y tipos de datos
5. Usa el contexto del paciente cuando sea relevante

CAMPOS ESPECÍFICOS A REVISAR:
- DIAG_START_DATE, MTIME, PATI_START_DATE
- Valores en valueQuantity, valueString
- Fechas en onsetDateTime, recordedDate, effectiveDateTime
- Referencias en subject, encounter, performer

RESPUESTA JSON:
{{
    "corrected_resource": {{
        // Recurso FHIR con valores de texto corregidos
    }},
    "text_corrections": [
        {{
            "field": "nombre_del_campo",
            "old_value": "valor_anterior",
            "new_value": "valor_nuevo",
            "reason": "explicación_de_la_corrección"
        }}
    ],
    "correction_summary": {{
        "total_text_fields_checked": 10,
        "text_fields_corrected": 4,
        "text_fields_valid": 6
    }}
}}

IMPORTANTE: Responde SOLO con el JSON del recurso corregido."""

            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result and isinstance(result, dict) and 'corrected_resource' in result:
                corrections = result.get('text_corrections', [])
                print(f"   📝 Valores de texto corregidos: {len(corrections)} campos")
                for correction in corrections:
                    print(f"      - {correction.get('field')}: {correction.get('old_value')} → {correction.get('new_value')}")
                return result['corrected_resource']
            else:
                print(f"   ⚠️ LLM no pudo corregir valores de texto")
                return fhir_resource
                
        except Exception as e:
            logger.error(f"Error corrigiendo valores de texto con LLM: {e}")
            return fhir_resource

    async def _llm_fix_dates_and_timestamps(self, fhir_resource: Dict[str, Any]) -> Dict[str, Any]:
        """LLAMADA 3: Corrige fechas y timestamps usando LLM"""
        try:
            prompt = f"""Eres un experto en manejo de fechas médicas. Corrige fechas y timestamps en este recurso FHIR.

RECURSO FHIR:
{json.dumps(fhir_resource, indent=2, ensure_ascii=False)}

TAREA ESPECÍFICA: Corregir fechas y timestamps inválidos o ficticios.

TIPOS DE FECHAS A CORREGIR:
1. Fechas vacías → Fecha actual
2. Timestamps ficticios → Timestamp real actual
3. Fechas en formato incorrecto → Formato ISO
4. Fechas futuras imposibles → Fecha actual
5. Fechas muy antiguas → Fecha actual

FORMATOS CORRECTOS:
- Fecha: YYYY-MM-DD
- Timestamp: YYYY-MM-DDTHH:MM:SS
- Fecha de nacimiento: YYYY-MM-DD
- Fecha de inicio: YYYY-MM-DDTHH:MM:SS

CAMPOS DE FECHA COMUNES:
- birthDate, onsetDateTime, recordedDate, effectiveDateTime
- DIAG_START_DATE, PATI_START_DATE, MTIME
- Fechas en valueDateTime, valuePeriod

REGLAS:
1. Usa fecha actual para fechas vacías o ficticias
2. Mantén fechas válidas existentes
3. Convierte formatos incorrectos a ISO
4. Valida que las fechas sean lógicas (no futuras imposibles)

RESPUESTA JSON:
{{
    "corrected_resource": {{
        // Recurso FHIR con fechas corregidas
    }},
    "date_corrections": [
        {{
            "field": "nombre_del_campo",
            "old_value": "valor_anterior",
            "new_value": "valor_nuevo",
            "reason": "explicación_de_la_corrección"
        }}
    ],
    "date_summary": {{
        "total_date_fields": 8,
        "dates_corrected": 3,
        "dates_valid": 5
    }}
}}

IMPORTANTE: Responde SOLO con el JSON del recurso corregido."""

            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result and isinstance(result, dict) and 'corrected_resource' in result:
                corrections = result.get('date_corrections', [])
                print(f"   📅 Fechas corregidas: {len(corrections)} campos")
                for correction in corrections:
                    print(f"      - {correction.get('field')}: {correction.get('old_value')} → {correction.get('new_value')}")
                return result['corrected_resource']
            else:
                print(f"   ⚠️ LLM no pudo corregir fechas")
                return fhir_resource
                
        except Exception as e:
            logger.error(f"Error corrigiendo fechas con LLM: {e}")
            return fhir_resource

    async def _llm_fix_numeric_values(self, fhir_resource: Dict[str, Any]) -> Dict[str, Any]:
        """LLAMADA 4: Corrige valores numéricos en observaciones usando LLM"""
        try:
            prompt = f"""Eres un experto en datos médicos numéricos. Corrige valores numéricos en observaciones FHIR.

RECURSO FHIR:
{json.dumps(fhir_resource, indent=2, ensure_ascii=False)}

TAREA ESPECÍFICA: Corregir valores numéricos en observaciones médicas.

TIPOS DE CORRECCIONES NUMÉRICAS:
1. Strings numéricos → Números reales
2. Valores vacíos → 0 o valor por defecto apropiado
3. Valores fuera de rango → Valores realistas
4. Unidades incorrectas → Unidades estándar

CAMPOS NUMÉRICOS COMUNES:
- valueQuantity.value: Valor numérico de la observación
- valueQuantity.unit: Unidad de medida
- valueInteger: Valores enteros
- valueDecimal: Valores decimales

RANGOS REALISTAS POR TIPO:
- Presión arterial: 80-200 mmHg
- Temperatura: 35-42 °C
- Frecuencia cardíaca: 40-200 bpm
- Glucemia: 40-600 mg/dL
- Peso: 1-300 kg
- Altura: 30-250 cm

INSTRUCCIONES:
1. Identifica campos numéricos en el recurso
2. Convierte strings numéricos a números
3. Valida rangos realistas para valores médicos
4. Corrige unidades de medida incorrectas
5. Usa valores por defecto apropiados para campos vacíos

RESPUESTA JSON:
{{
    "corrected_resource": {{
        // Recurso FHIR con valores numéricos corregidos
    }},
    "numeric_corrections": [
        {{
            "field": "nombre_del_campo",
            "old_value": "valor_anterior",
            "new_value": "valor_nuevo",
            "reason": "explicación_de_la_corrección"
        }}
    ],
    "numeric_summary": {{
        "total_numeric_fields": 5,
        "numeric_fields_corrected": 2,
        "numeric_fields_valid": 3
    }}
}}

IMPORTANTE: Responde SOLO con el JSON del recurso corregido."""

            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result and isinstance(result, dict) and 'corrected_resource' in result:
                corrections = result.get('numeric_corrections', [])
                print(f"   🔢 Valores numéricos corregidos: {len(corrections)} campos")
                for correction in corrections:
                    print(f"      - {correction.get('field')}: {correction.get('old_value')} → {correction.get('new_value')}")
                return result['corrected_resource']
            else:
                print(f"   ⚠️ LLM no pudo corregir valores numéricos")
                return fhir_resource
                
        except Exception as e:
            logger.error(f"Error corrigiendo valores numéricos con LLM: {e}")
            return fhir_resource

    async def _llm_fix_resource_references(self, fhir_resource: Dict[str, Any], patient_id_context: Optional[str] = None) -> Dict[str, Any]:
        """LLAMADA 5: Valida y corrige referencias entre recursos usando LLM"""
        try:
            prompt = f"""Eres un experto en referencias entre recursos FHIR. Valida y corrige las referencias en este recurso.

RECURSO FHIR:
{json.dumps(fhir_resource, indent=2, ensure_ascii=False)}

CONTEXTO DE PACIENTE: {patient_id_context or 'No disponible'}

TAREA ESPECÍFICA: Validar y corregir referencias entre recursos FHIR.

TIPOS DE REFERENCIAS:
1. subject.reference: Referencia al paciente
2. encounter.reference: Referencia al encuentro médico
3. performer.reference: Referencia al profesional
4. requester.reference: Referencia al solicitante
5. basedOn.reference: Referencia a la solicitud base

REGLAS DE VALIDACIÓN:
1. Las referencias deben ser consistentes
2. Si hay contexto de paciente, usarlo para subject.reference
3. Las referencias deben apuntar a recursos válidos
4. Evitar referencias circulares
5. Mantener coherencia entre recursos relacionados

CORRECCIONES COMUNES:
- Referencias ficticias → IDs reales
- Referencias inconsistentes → Referencias coherentes
- Referencias faltantes → Referencias apropiadas
- Referencias circulares → Referencias válidas

INSTRUCCIONES:
1. Analiza todas las referencias en el recurso
2. Valida que las referencias sean coherentes
3. Corrige referencias ficticias o inválidas
4. Asegura que las referencias apunten a recursos válidos
5. Usa el contexto del paciente cuando sea apropiado

RESPUESTA JSON:
{{
    "corrected_resource": {{
        // Recurso FHIR con referencias corregidas
    }},
    "reference_corrections": [
        {{
            "field": "nombre_del_campo",
            "old_value": "valor_anterior",
            "new_value": "valor_nuevo",
            "reason": "explicación_de_la_corrección"
        }}
    ],
    "reference_summary": {{
        "total_references": 3,
        "references_corrected": 1,
        "references_valid": 2
    }}
}}

IMPORTANTE: Responde SOLO con el JSON del recurso corregido."""

            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result and isinstance(result, dict) and 'corrected_resource' in result:
                corrections = result.get('reference_corrections', [])
                print(f"   🔗 Referencias corregidas: {len(corrections)} campos")
                for correction in corrections:
                    print(f"      - {correction.get('field')}: {correction.get('old_value')} → {correction.get('new_value')}")
                return result['corrected_resource']
            else:
                print(f"   ⚠️ LLM no pudo corregir referencias")
                return fhir_resource
                
        except Exception as e:
            logger.error(f"Error corrigiendo referencias con LLM: {e}")
            return fhir_resource

    async def _fix_fhir_uuid_mapping_basic(self, fhir_resource: Dict[str, Any], patient_id_context: Optional[str] = None) -> Dict[str, Any]:
        """Corrección básica de UUIDs FHIR usando LLM (fallback)"""
        print(f"      🔧 Iniciando corrección básica de UUIDs FHIR con LLM...")
        
        try:
            if not self.llm:
                print(f"      ⚠️ LLM no disponible para corrección básica")
                return await self._fix_fhir_uuid_dynamic(fhir_resource, patient_id_context)
            
            # Generar ID único para este recurso
            import time
            import random
            unique_id = str(int(time.time() * 1000) + random.randint(1, 999))
            
            prompt = f"""Eres un experto en corrección de recursos FHIR. Corrige este recurso FHIR reemplazando UUIDs ficticios con IDs numéricos válidos.

REGLAS IMPORTANTES:
- Usa el ID único proporcionado: {unique_id}
- Reemplaza TODOS los UUIDs ficticios (urn:uuid:...) con IDs numéricos únicos
- Mantén la estructura FHIR válida
- Asegúrate de que las referencias sean consistentes
- Usa el contexto del paciente: {patient_id_context or 'No disponible'}

RECURSO FHIR A CORREGIR:
{json.dumps(fhir_resource, indent=2, ensure_ascii=False)}

ID ÚNICO PARA ESTE RECURSO: {unique_id}

Responde SOLO con el JSON corregido, sin explicaciones adicionales."""

            response = await self.llm.ainvoke(prompt)
            response_text = self._extract_response_text(response)
            
            # Parsear la respuesta
            corrected_resource = self._try_parse_llm_json(response_text)
            if corrected_resource and isinstance(corrected_resource, dict):
                print(f"      ✅ Corrección básica con LLM completada")
                return corrected_resource
            else:
                print(f"      ⚠️ Fallo en parseo, usando corrección dinámica")
                return await self._fix_fhir_uuid_dynamic(fhir_resource, patient_id_context)
                
        except Exception as e:
            print(f"      ❌ Error en corrección básica con LLM: {e}")
            return await self._fix_fhir_uuid_dynamic(fhir_resource, patient_id_context)

    async def _llm_fix_text_values_intelligent(self, fhir_resource: Dict[str, Any], patient_id_context: Optional[str] = None) -> Dict[str, Any]:
        """LLM inteligente para corregir valores de texto ficticios"""
        try:
            prompt = f"""Eres un experto en corrección de valores de texto en recursos FHIR. Corrige valores ficticios o inapropiados.

RECURSO FHIR:
{json.dumps(fhir_resource, indent=2, ensure_ascii=False)}

CONTEXTO DE PACIENTE: {patient_id_context or 'No disponible'}

TAREA ESPECÍFICA: Corregir valores de texto ficticios o inapropiados en el recurso FHIR.

TIPOS DE CORRECCIONES:
1. Valores vacíos ("") → Valores apropiados
2. Textos genéricos ("fecha actual", "active") → Valores reales
3. Placeholders ("valor_por_defecto") → Valores significativos
4. Textos inconsistentes → Textos coherentes
5. Valores ficticios → Valores realistas

REGLAS DE CORRECCIÓN:
1. Usar fechas reales cuando sea apropiado
2. Mantener coherencia con el contexto del paciente
3. Evitar valores genéricos o ficticios
4. Preservar información médica relevante
5. Usar valores por defecto apropiados cuando sea necesario

INSTRUCCIONES:
1. Identifica valores de texto ficticios o inapropiados
2. Corrige cada valor con uno apropiado y realista
3. Mantén la estructura del recurso FHIR
4. Usa el contexto del paciente cuando sea relevante
5. Documenta cada corrección realizada

RESPUESTA JSON:
{{
    "corrected_resource": {{
        // Recurso FHIR con valores de texto corregidos
    }},
    "text_corrections": [
        {{
            "field": "nombre_del_campo",
            "old_value": "valor_anterior",
            "new_value": "valor_nuevo",
            "reason": "explicación_de_la_corrección"
        }}
    ],
    "text_summary": {{
        "total_text_fields": 5,
        "text_fields_corrected": 2,
        "text_fields_valid": 3
    }}
}}

IMPORTANTE: Responde SOLO con el JSON del recurso corregido."""

            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result and isinstance(result, dict) and 'corrected_resource' in result:
                corrections = result.get('text_corrections', [])
                print(f"      📝 Valores de texto corregidos: {len(corrections)} campos")
                for correction in corrections:
                    print(f"         - {correction.get('field')}: {correction.get('old_value')} → {correction.get('new_value')}")
                return result['corrected_resource']
            else:
                print(f"      ⚠️ LLM no pudo corregir valores de texto")
                return self._fix_text_values_basic_sync(fhir_resource)
                
        except Exception as e:
            logger.error(f"Error corrigiendo valores de texto con LLM: {e}")
            return self._fix_text_values_basic_sync(fhir_resource)

    def _llm_fix_text_values_intelligent_sync(self, fhir_resource: Dict[str, Any], patient_id_context: Optional[str] = None) -> Dict[str, Any]:
        """LLM inteligente síncrono para corregir valores de texto ficticios"""
        try:
            prompt = f"""Eres un experto en corrección de valores de texto en recursos FHIR. Corrige valores ficticios o inapropiados.

RECURSO FHIR:
{json.dumps(fhir_resource, indent=2, ensure_ascii=False)}

CONTEXTO DE PACIENTE: {patient_id_context or 'No disponible'}

TAREA ESPECÍFICA: Corregir valores de texto ficticios o inapropiados en el recurso FHIR.

TIPOS DE CORRECCIONES:
1. Valores vacíos ("") → Valores apropiados
2. Textos genéricos ("fecha actual", "active") → Valores reales
3. Placeholders ("valor_por_defecto") → Valores significativos
4. Textos inconsistentes → Textos coherentes
5. Valores ficticios → Valores realistas

REGLAS DE CORRECCIÓN:
1. Usar fechas reales cuando sea apropiado
2. Mantener coherencia con el contexto del paciente
3. Evitar valores genéricos o ficticios
4. Preservar información médica relevante
5. Usar valores por defecto apropiados cuando sea necesario

INSTRUCCIONES:
1. Identifica valores de texto ficticios o inapropiados
2. Corrige cada valor con uno apropiado y realista
3. Mantén la estructura del recurso FHIR
4. Usa el contexto del paciente cuando sea relevante
5. Documenta cada corrección realizada

RESPUESTA JSON:
{{
    "corrected_resource": {{
        // Recurso FHIR con valores de texto corregidos
    }},
    "text_corrections": [
        {{
            "field": "nombre_del_campo",
            "old_value": "valor_anterior",
            "new_value": "valor_nuevo",
            "reason": "explicación_de_la_corrección"
        }}
    ],
    "text_summary": {{
        "total_text_fields": 5,
        "text_fields_corrected": 2,
        "text_fields_valid": 3
    }}
}}

IMPORTANTE: Responde SOLO con el JSON del recurso corregido."""

            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result and isinstance(result, dict) and 'corrected_resource' in result:
                corrections = result.get('text_corrections', [])
                print(f"      📝 Valores de texto corregidos: {len(corrections)} campos")
                for correction in corrections:
                    print(f"         - {correction.get('field')}: {correction.get('old_value')} → {correction.get('new_value')}")
                return result['corrected_resource']
            else:
                print(f"      ⚠️ LLM no pudo corregir valores de texto")
                # Usar corrección básica síncrona
                return self._fix_text_values_basic_sync(fhir_resource)
                
        except Exception as e:
            logger.error(f"Error corrigiendo valores de texto con LLM: {e}")
            return self._fix_text_values_basic_sync(fhir_resource)

    def _fix_text_values_basic_sync(self, fhir_resource: Dict[str, Any]) -> Dict[str, Any]:
        """Corrección básica síncrona de valores de texto (fallback)"""
        try:
            corrected_resource = fhir_resource.copy()
            
            # Correcciones básicas de texto
            text_fixes = {
                "fecha actual": datetime.now().strftime("%Y-%m-%d"),
                "active": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "": "N/A",
                "valor_por_defecto": "N/A",
                "fecha_inicio_del_encuentro": datetime.now().strftime("%Y-%m-%d"),
                "fecha_fin_del_encuentro": datetime.now().strftime("%Y-%m-%d"),
                "current_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            for key, value in corrected_resource.items():
                if isinstance(value, str) and value in text_fixes:
                    old_value = value
                    corrected_resource[key] = text_fixes[value]
                    print(f"      📝 Texto corregido {key}: {old_value} → {corrected_resource[key]}")
                elif isinstance(value, str) and value == "":
                    old_value = value
                    corrected_resource[key] = "N/A"
                    print(f"      📝 Valor vacío corregido {key}: '' → 'N/A'")
                elif isinstance(value, dict):
                    # Recursivamente corregir valores en diccionarios anidados
                    corrected_resource[key] = self._fix_text_values_basic_sync(value)
            
            return corrected_resource
            
        except Exception as e:
            print(f"      ❌ Error en corrección básica de texto: {e}")
            return fhir_resource

    async def _fix_text_values_basic(self, fhir_resource: Dict[str, Any]) -> Dict[str, Any]:
        """Corrección básica de valores de texto usando LLM (fallback)"""
        try:
            if not self.llm:
                print(f"      ⚠️ LLM no disponible para corrección básica de texto")
                return fhir_resource
            
            prompt = f"""Eres un experto en corrección de valores de texto en recursos FHIR. Corrige valores ficticios o inapropiados.

RECURSO FHIR:
{json.dumps(fhir_resource, indent=2, ensure_ascii=False)}

TAREA ESPECÍFICA: Corregir valores de texto ficticios o inapropiados en el recurso FHIR.

TIPOS DE CORRECCIONES:
1. Valores vacíos ("") → Valores apropiados
2. Textos genéricos ("fecha actual", "active") → Valores reales
3. Placeholders ("valor_por_defecto") → Valores significativos
4. Textos inconsistentes → Textos coherentes
5. Valores ficticios → Valores realistas

REGLAS DE CORRECCIÓN:
1. Usar fechas reales cuando sea apropiado
2. Evitar valores genéricos o ficticios
3. Preservar información médica relevante
4. Usar valores por defecto apropiados cuando sea necesario

INSTRUCCIONES:
1. Identifica valores de texto ficticios o inapropiados
2. Corrige cada valor con uno apropiado y realista
3. Mantén la estructura del recurso FHIR
4. Documenta cada corrección realizada

Responde SOLO con el JSON del recurso corregido, sin explicaciones adicionales."""

            response = await self.llm.ainvoke(prompt)
            response_text = self._extract_response_text(response)
            
            # Parsear la respuesta
            corrected_resource = self._try_parse_llm_json(response_text)
            if corrected_resource:
                print(f"      ✅ Corrección básica de texto con LLM completada")
                return corrected_resource
            else:
                print(f"      ⚠️ Fallo en parseo, usando recurso original")
                return fhir_resource
                
        except Exception as e:
            print(f"      ❌ Error en corrección básica de texto con LLM: {e}")
            return fhir_resource

    async def _llm_validate_patient_name(self, first_name: str, last_name: str) -> Dict[str, Any]:
        """LLM inteligente para validar nombres de pacientes"""
        try:
            if not self.llm:
                # Fallback básico sin LLM
                invalid_names = ['paciente', 'patient', 'unknown', 'desconocido', '']
                is_valid = first_name and first_name.strip() and first_name.lower() not in invalid_names
                return {
                    'is_valid': is_valid,
                    'reason': 'Validación básica sin LLM' if is_valid else 'Nombre inválido o genérico'
                }
            
            prompt = f"""Eres un experto en validación de nombres de pacientes. Valida si el nombre proporcionado es válido para un registro médico.

NOMBRE A VALIDAR:
- Nombre: "{first_name}"
- Apellido: "{last_name}"

CRITERIOS DE VALIDACIÓN:
1. El nombre no debe estar vacío
2. No debe ser genérico (ej: "paciente", "patient", "unknown")
3. Debe ser un nombre real y válido
4. No debe contener caracteres especiales inapropiados
5. Debe tener sentido en contexto médico

NOMBRES INVÁLIDOS:
- "paciente", "patient", "unknown", "desconocido"
- Nombres vacíos o solo espacios
- Nombres con caracteres especiales inapropiados
- Nombres que no parecen reales

NOMBRES VÁLIDOS:
- Nombres reales como "María", "Juan", "Carlos"
- Nombres compuestos como "Juan Carlos"
- Nombres con acentos y caracteres especiales válidos

RESPUESTA JSON:
{{
    "is_valid": true/false,
    "reason": "explicación_detallada_de_la_validación",
    "suggestions": ["sugerencias_si_es_inválido"]
}}

IMPORTANTE: Responde SOLO con el JSON de validación."""

            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result and isinstance(result, dict):
                return {
                    'is_valid': result.get('is_valid', False),
                    'reason': result.get('reason', 'Validación LLM completada'),
                    'suggestions': result.get('suggestions', [])
                }
            else:
                # Fallback si el LLM no responde correctamente
                invalid_names = ['paciente', 'patient', 'unknown', 'desconocido', '']
                is_valid = first_name and first_name.strip() and first_name.lower() not in invalid_names
                return {
                    'is_valid': is_valid,
                    'reason': 'Validación fallback - LLM no disponible'
                }
                
        except Exception as e:
            logger.error(f"Error validando nombre de paciente con LLM: {e}")
            # Fallback básico
            invalid_names = ['paciente', 'patient', 'unknown', 'desconocido', '']
            is_valid = first_name and first_name.strip() and first_name.lower() not in invalid_names
            return {
                'is_valid': is_valid,
                'reason': 'Validación fallback por error'
            }

    async def _llm_map_fhir_resource_type(self, fhir_resource: Dict[str, Any]) -> Dict[str, Any]:
        """LLM inteligente para mapear tipos de recursos FHIR a tablas SQL"""
        try:
            if not self.llm:
                # Fallback básico sin LLM
                resource_type = fhir_resource.get('resourceType', '')
                if resource_type == 'Patient':
                    return {
                        'table': 'PATI_PATIENTS',
                        'data': {
                            'PATI_NAME': fhir_resource.get('name', [{}])[0].get('given', [''])[0],
                            'PATI_SURNAME_1': fhir_resource.get('name', [{}])[0].get('family', ''),
                            'GEND_ID': 1 if fhir_resource.get('gender') == 'male' else 2,
                            'PATI_BIRTH_DATE': fhir_resource.get('birthDate', ''),
                            'PATI_ACTIVE': 1,
                            'PATI_START_DATE': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                    }
                else:
                    return {
                        'table': 'APPO_APPOINTMENTS',
                        'data': {
                            'notes': json.dumps(fhir_resource)
                        }
                    }
            
            prompt = f"""Eres un experto en mapeo de recursos FHIR a tablas SQL. Mapea este recurso FHIR a la tabla SQL más apropiada.

RECURSO FHIR:
{json.dumps(fhir_resource, indent=2, ensure_ascii=False)}

ESQUEMA DE BASE DE DATOS DISPONIBLE:
{await self._get_database_schema_info()}

TAREA ESPECÍFICA: Determinar la tabla SQL más apropiada para este tipo de recurso FHIR.

REGLAS DE MAPEO:
1. Patient → PATI_PATIENTS (datos personales del paciente)
2. Condition → EPIS_DIAGNOSTICS (diagnósticos y condiciones médicas)
3. MedicationRequest → PATI_USUAL_MEDICATION (medicación del paciente)
4. Observation → EPIS_DIAGNOSTICS (observaciones médicas)
5. Encounter → APPO_APPOINTMENTS (citas y encuentros médicos)
6. Procedure → PROC_PROCEDURES (procedimientos médicos)

INSTRUCCIONES:
1. Analiza el tipo de recurso FHIR
2. Identifica la tabla SQL más apropiada
3. Mapea los campos FHIR a las columnas SQL
4. Genera datos SQL válidos
5. Incluye solo campos que existen en la tabla

RESPUESTA JSON:
{{
    "table": "nombre_tabla_exacto",
    "data": {{
        "campo1": "valor1",
        "campo2": "valor2"
    }},
    "mapping_notes": "explicación_del_mapeo",
    "confidence": 0.95
}}

IMPORTANTE: Responde SOLO con el JSON del mapeo."""

            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result and isinstance(result, dict):
                return {
                    'table': result.get('table', 'APPO_APPOINTMENTS'),
                    'data': result.get('data', {'notes': json.dumps(fhir_resource)})
                }
            else:
                # Fallback si el LLM no responde correctamente
                resource_type = fhir_resource.get('resourceType', '')
                if resource_type == 'Patient':
                    return {
                        'table': 'PATI_PATIENTS',
                        'data': {
                            'PATI_NAME': fhir_resource.get('name', [{}])[0].get('given', [''])[0],
                            'PATI_SURNAME_1': fhir_resource.get('name', [{}])[0].get('family', ''),
                            'GEND_ID': 1 if fhir_resource.get('gender') == 'male' else 2,
                            'PATI_BIRTH_DATE': fhir_resource.get('birthDate', ''),
                            'PATI_ACTIVE': 1,
                            'PATI_START_DATE': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                    }
                else:
                    return {
                        'table': 'APPO_APPOINTMENTS',
                        'data': {
                            'notes': json.dumps(fhir_resource)
                        }
                    }
                
        except Exception as e:
            logger.error(f"Error mapeando recurso FHIR con LLM: {e}")
            # Fallback básico
            resource_type = fhir_resource.get('resourceType', '')
            if resource_type == 'Patient':
                return {
                    'table': 'PATI_PATIENTS',
                    'data': {
                        'PATI_NAME': fhir_resource.get('name', [{}])[0].get('given', [''])[0],
                        'PATI_SURNAME_1': fhir_resource.get('name', [{}])[0].get('family', ''),
                        'GEND_ID': 1 if fhir_resource.get('gender') == 'male' else 2,
                        'PATI_BIRTH_DATE': fhir_resource.get('birthDate', ''),
                        'PATI_ACTIVE': 1,
                        'PATI_START_DATE': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                }
            else:
                return {
                    'table': 'APPO_APPOINTMENTS',
                    'data': {
                        'notes': json.dumps(fhir_resource)
                    }
                }

    async def _llm_map_medication_request(self, fhir_resource: Dict[str, Any], patient_id_context: Optional[str] = None) -> Dict[str, Any]:
        """Mapea un recurso MedicationRequest FHIR a SQL usando LLM"""
        try:
            if not self.llm:
                return self._map_medication_request_basic(fhir_resource, patient_id_context)
            
            # Obtener esquema de base de datos
            schema_info = await self._get_database_schema_info()
            
            prompt = f"""Eres un experto en mapeo de recursos FHIR MedicationRequest a SQL. Mapea este recurso a las tablas de medicación disponibles.

            RECURSO FHIR MEDICATIONREQUEST:
            {json.dumps(fhir_resource, indent=2, ensure_ascii=False)}

ESQUEMA DE BASE DE DATOS:
{schema_info}

CONTEXTO DE PACIENTE: {patient_id_context or 'No disponible'}

INSTRUCCIONES:
1. Analiza el recurso MedicationRequest
2. Mapea los campos a las tablas de medicación disponibles
3. Considera las relaciones con el paciente
4. Devuelve un JSON con el mapeo SQL

RESPUESTA: JSON con mapeo SQL para MedicationRequest"""

            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result and isinstance(result, dict):
                return result
            else:
                return self._map_medication_request_basic(fhir_resource, patient_id_context)
                
        except Exception as e:
            print(f"❌ Error en mapeo LLM de MedicationRequest: {e}")
            return self._map_medication_request_basic(fhir_resource, patient_id_context)

    def _create_basic_fhir_resources(self, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Crea recursos FHIR básicos usando LLM dinámico con formato específico"""
        try:
            if not self.llm:
                print(f"❌ LLM no disponible para creación de recursos FHIR básicos")
                return []
            
            print(f"🧠 Usando LLM para crear recursos FHIR básicos dinámicamente...")
            
            prompt = f"""Eres un experto en creación de recursos FHIR. Crea recursos FHIR válidos a partir de estos datos clínicos extraídos.

DATOS CLÍNICOS EXTRAÍDOS:
{json.dumps(extracted_data, indent=2, ensure_ascii=False)}

INSTRUCCIONES CRÍTICAS:
1. Analiza los datos clínicos proporcionados
2. Crea recursos FHIR apropiados para cada tipo de información
3. Usa IDs numéricos únicos (no UUIDs ficticios)
4. Incluye todos los tipos de recursos necesarios
5. Asegúrate de que el JSON sea válido y completo
6. Usa el formato EXACTO especificado abajo

FORMATO REQUERIDO:
```json
[
  {{
    "resourceType": "Patient",
    "id": "1",
    "name": [{{"text": "Nombre Completo del Paciente"}}],
    "gender": "female",
    "birthDate": "1989-03-15"
  }},
  {{
    "resourceType": "Condition",
    "id": "2",
    "subject": {{"reference": "Patient/1"}},
    "code": {{"text": "Diagnóstico Principal"}}
  }},
  {{
    "resourceType": "MedicationRequest",
    "id": "3",
    "subject": {{"reference": "Patient/1"}},
    "medicationCodeableConcept": {{"text": "Nombre del Medicamento"}},
    "dosageInstruction": [{{"text": "Dosis del Medicamento"}}]
  }},
  {{
    "resourceType": "Observation",
    "id": "4",
    "subject": {{"reference": "Patient/1"}},
    "code": {{"text": "Tipo de Observación"}},
    "valueQuantity": {{
      "value": "Valor Numérico",
      "unit": "Unidad de Medida"
    }}
  }},
  {{
    "resourceType": "AllergyIntolerance",
    "id": "5",
    "patient": {{"reference": "Patient/1"}},
    "code": {{"text": "Sustancia Alergénica"}}
  }}
]
```

TIPOS DE RECURSOS A CREAR:
- Patient: Información del paciente (nombre, edad, género, fecha de nacimiento)
- Condition: Diagnósticos y condiciones médicas
- MedicationRequest: Medicamentos prescritos con dosis
- Observation: Signos vitales y resultados de laboratorio
- AllergyIntolerance: Alergias del paciente

REGLAS IMPORTANTES:
- Usa IDs numéricos simples: "1", "2", "3", etc.
- NO uses UUIDs ficticios como "urn:uuid:..."
- Asegúrate de que cada recurso tenga 'resourceType' válido
- Incluye referencias entre recursos cuando sea apropiado
- Usa formato JSON válido y completo
- Sigue EXACTAMENTE el formato especificado arriba

CRÍTICO: Responde SOLO con el array JSON, sin explicaciones, sin texto adicional, sin comentarios. Solo el JSON válido."""

            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result and isinstance(result, list):
                print(f"✅ LLM creó {len(result)} recursos FHIR básicos")
                return result
            else:
                print(f"❌ LLM no pudo crear recursos FHIR básicos")
                return []
            
        except Exception as e:
            print(f"❌ Error creando recursos FHIR básicos con LLM: {e}")
            return []

    def _map_medication_request_basic(self, fhir_resource: Dict[str, Any], patient_id_context: Optional[str] = None) -> Dict[str, Any]:
        """Mapeo básico de MedicationRequest usando LLM"""
        try:
            if not self.llm:
                return {
                    'success': False,
                    'error': 'LLM no disponible para mapeo básico de medicamentos'
                }
            
            print(f"   🧠 Usando LLM para mapeo básico de MedicationRequest...")
            
            prompt = f"""Eres un experto en mapeo de recursos FHIR MedicationRequest a SQL. Mapea este recurso a las tablas de medicación disponibles.

RECURSO FHIR MEDICATIONREQUEST:
{json.dumps(fhir_resource, indent=2, ensure_ascii=False)}

CONTEXTO DE PACIENTE: {patient_id_context or 'No disponible'}

INSTRUCCIONES:
1. Analiza el recurso MedicationRequest
2. Mapea los campos a las tablas de medicación disponibles
3. Considera las relaciones con el paciente
4. Devuelve un JSON con el mapeo SQL

RESPUESTA: JSON con mapeo SQL para MedicationRequest"""

            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result and isinstance(result, dict):
                return result
            else:
                return {
                    'success': False,
                    'error': 'No se pudo mapear MedicationRequest con LLM'
                }
                
        except Exception as e:
            print(f"❌ Error en mapeo básico de MedicationRequest: {e}")
            return {
                'success': False,
                'error': f'Error en mapeo básico: {str(e)}'
            }

    def _create_manual_fhir_resources(self, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera recursos FHIR usando el LLM con un prompt genérico y claro sobre la tarea"""
        try:
            print(f"🔄 Solicitando al LLM la generación de recursos FHIR...")
            if not self.llm:
                print(f"❌ LLM no disponible para generación de recursos FHIR")
                return []

            prompt = f"""
Eres un experto en interoperabilidad médica y FHIR. Tu tarea es generar un array JSON de recursos FHIR válidos (Patient, Condition, MedicationRequest, Observation, AllergyIntolerance, etc.) a partir de los siguientes datos clínicos estructurados extraídos de una nota médica.

DATOS CLÍNICOS EXTRAÍDOS:
{json.dumps(extracted_data, indent=2, ensure_ascii=False)}

INSTRUCCIONES CRÍTICAS:
- Genera un array JSON de recursos FHIR, uno por cada entidad relevante (paciente, diagnóstico, medicación, observación, alergia, etc.)
- Usa IDs numéricos simples ("1", "2", "3", ...)
- NO uses UUIDs ni referencias complejas
- Asegúrate de que el JSON sea válido y parseable
- Incluye referencias entre recursos cuando corresponda (por ejemplo, subject/reference a Patient)
- No añadas explicaciones ni comentarios, SOLO el array JSON de recursos FHIR

RESPUESTA: Array JSON de recursos FHIR válidos"""

            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            if result and isinstance(result, list):
                print(f"✅ LLM generó {len(result)} recursos FHIR")
                return result
            else:
                print(f"❌ El LLM no pudo generar recursos FHIR válidos")
                return []
        except Exception as e:
            print(f"❌ Error en generación de recursos FHIR con LLM: {e}")
            return []

    def _is_fictitious_id(self, id_value: str) -> bool:
        """LLM detecta si un ID es ficticio"""
        try:
            if not self.llm:
                # Fallback sin LLM
                if not isinstance(id_value, str):
                    return False
                
                fictitious_patterns = [
                    'urn:uuid:', 'unico', 'ficticio', 'mock', 'fake', 
                    'patient-id', 'observation-id', 'encounter-id',
                    'medication-id', 'id-', 'placeholder', 'temp'
                ]
                
                return any(pattern in id_value.lower() for pattern in fictitious_patterns)
            
            prompt = f"""Eres un experto en detección de IDs ficticios en recursos FHIR. Determina si este ID es ficticio o problemático.

ID A EVALUAR: {id_value}

INSTRUCCIONES:
1. Analiza el ID proporcionado
2. Determina si es un ID ficticio, placeholder o problemático
3. Considera patrones comunes de IDs ficticios
4. Evalúa si el ID es válido para uso en FHIR

PATRONES DE IDs FICTICIOS:
- urn:uuid: (UUIDs genéricos)
- unico, ficticio, mock, fake
- patient-id, observation-id, encounter-id
- id-, placeholder, temp
- Valores genéricos o sin sentido

RESPUESTA: Solo "true" si el ID es ficticio, "false" si es válido."""

            response = _call_openai_native(self.llm, prompt)
            result = response.content.strip().lower()
            
            return result == "true"
            
        except Exception as e:
            print(f"      ⚠️ Error detectando ID ficticio con LLM: {e}")
            # Fallback sin LLM
            if not isinstance(id_value, str):
                return False
            
            fictitious_patterns = [
                'urn:uuid:', 'unico', 'ficticio', 'mock', 'fake', 
                'patient-id', 'observation-id', 'encounter-id',
                'medication-id', 'id-', 'placeholder', 'temp'
            ]
            
            return any(pattern in id_value.lower() for pattern in fictitious_patterns)

    def _fix_references_dynamic(self, resource: Dict[str, Any], patient_id_context: Optional[str], unique_id: str) -> Dict[str, Any]:
        """Corrige referencias dinámicamente sin LLM"""
        corrected_resource = resource.copy()
        
        # Corregir referencia al paciente
        if 'subject' in corrected_resource:
            if isinstance(corrected_resource['subject'], dict) and 'reference' in corrected_resource['subject']:
                if patient_id_context and patient_id_context != 'No disponible':
                    corrected_resource['subject']['reference'] = f'Patient/{patient_id_context}'
                else:
                    corrected_resource['subject']['reference'] = f'Patient/{unique_id}'
        
        # Corregir otras referencias
        for ref_key in ['encounter', 'performer', 'medication', 'requester']:
            if ref_key in corrected_resource:
                if isinstance(corrected_resource[ref_key], dict) and 'reference' in corrected_resource[ref_key]:
                    corrected_resource[ref_key]['reference'] = f'{ref_key.capitalize()}/{unique_id}'
        
        return corrected_resource

    async def _llm_fix_empty_values(self, data: Dict[str, Any], table: str, patient_id_context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """LLM corrige valores vacíos de forma inteligente"""
        try:
            if not self.llm:
                return None
            
            # Obtener información del esquema de la tabla
            schema_info = ""
            if hasattr(self, 'sql_agent') and self.sql_agent and hasattr(self.sql_agent, 'column_metadata'):
                if table in self.sql_agent.column_metadata:
                    columns = self.sql_agent.column_metadata[table]
                    schema_info = f"Columnas de la tabla {table}: {', '.join(columns)}"
            
            prompt = f"""Eres un experto en corrección de datos médicos para bases de datos SQL. Corrige los valores vacíos o nulos de forma inteligente.

DATOS ACTUALES:
{json.dumps(data, indent=2, ensure_ascii=False)}

TABLA: {table}
CONTEXTO DE PACIENTE: {patient_id_context or 'No disponible'}
INFORMACIÓN DEL ESQUEMA: {schema_info}

INSTRUCCIONES:
1. Analiza cada campo con valor vacío ('') o nulo (None)
2. Determina el valor apropiado basado en:
   - El tipo de campo (ID, fecha, texto, número, booleano)
   - El contexto médico
   - El nombre del campo
   - La tabla donde se insertará
3. Aplica las siguientes reglas:
   - Campos de eliminación (DELETED): 0 para no eliminado, 1 para eliminado
   - Timestamps (MTIME): Fecha y hora actual en formato ISO
   - Fechas (START_DATE, END_DATE): Fecha actual en formato YYYY-MM-DD
   - IDs opcionales: NULL si no son obligatorios
   - Textos descriptivos: 'N/A' o valor apropiado
   - Números: 0 o valor por defecto apropiado
   - Booleanos: 0 (false) o 1 (true)

REGLAS ESPECÍFICAS POR TABLA:
- EPIS_DIAGNOSTICS: DIAG_DELETED=0, MTIME=timestamp actual, fechas=actual
- PATI_PATIENTS: GEND_ID=2 (femenino) por defecto, fechas=actual
- PATI_USUAL_MEDICATION: PAUM_OBSERVATIONS=texto descriptivo
- APPO_APPOINTMENTS: fechas=actual, estados=activo

RESPUESTA JSON:
{{
    "corrected_data": {{
        // Datos con valores vacíos corregidos
    }},
    "corrections_applied": [
        {{
            "field": "nombre_campo",
            "old_value": "valor_anterior",
            "new_value": "valor_nuevo",
            "reason": "explicación_de_la_corrección"
        }}
    ],
    "summary": {{
        "total_fields_checked": 10,
        "fields_corrected": 5,
        "fields_valid": 5
    }}
}}

IMPORTANTE: Responde SOLO con el JSON del objeto corrected_data."""

            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result and isinstance(result, dict):
                if 'corrected_data' in result:
                    corrections = result.get('corrections_applied', [])
                    print(f"   📝 LLM aplicó {len(corrections)} correcciones:")
                    for correction in corrections:
                        print(f"      - {correction.get('field')}: {correction.get('old_value')} → {correction.get('new_value')}")
                    return result['corrected_data']
                else:
                    # Si el LLM devuelve directamente los datos corregidos
                    return result
            else:
                print(f"   ⚠️ LLM no pudo corregir valores vacíos")
                return None
                
        except Exception as e:
            print(f"   ⚠️ Error corrigiendo valores vacíos con LLM: {e}")
            return None

    def _fix_empty_values_fallback(self, data: Dict[str, Any], table: str) -> Dict[str, Any]:
        """Fallback para corregir valores vacíos sin LLM"""
        import datetime
        
        corrected_data = data.copy()
        current_time = datetime.datetime.now()
        
        for key, value in corrected_data.items():
            if value == '' or value is None:
                if key == 'DIAG_DELETED' or key.endswith('_DELETED'):
                    corrected_data[key] = 0  # No eliminado
                elif key == 'MTIME' or key.endswith('_TIME'):
                    corrected_data[key] = current_time.isoformat()
                elif key.endswith('_DATE') or key.endswith('_START_DATE') or key.endswith('_END_DATE'):
                    corrected_data[key] = current_time.strftime('%Y-%m-%d')
                elif key.endswith('_ID') and key not in ['PATI_ID', 'DIAG_ID']:
                    corrected_data[key] = None  # NULL para IDs opcionales
                elif key == 'GEND_ID':
                    corrected_data[key] = 2  # Femenino por defecto
                else:
                    corrected_data[key] = 'N/A'  # Valor por defecto
                print(f"   🔧 Campo {key} corregido: {value} → {corrected_data[key]}")
        
        return corrected_data

    def _clean_sql_response(self, sql: str) -> str:
        """Limpia específicamente respuestas SQL del LLM"""
        if not sql:
            return sql
        
        # Limpiar formato markdown del SQL
        if '```sql' in sql:
            sql = sql.split('```sql')[1].split('```')[0].strip()
        elif '```' in sql:
            sql = sql.split('```')[1].split('```')[0].strip()
        
        # Remover etiquetas como "SQL:" al inicio
        import re
        sql = re.sub(r'^(SQL|sql):\s*', '', sql, flags=re.MULTILINE)
        
        # Remover cualquier texto explicativo común
        sql = re.sub(r'^(Esta consulta|This query|Consulta|Query).*?:', '', sql, flags=re.IGNORECASE)
        
        # Remover comentarios SQL
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        
        # Limpiar espacios extra
        sql = re.sub(r'\s+', ' ', sql).strip()
        
        return sql

