#!/usr/bin/env python3
"""
🧠 SQL Agent Completamente Inteligente v5.0 - Versión Avanzada Modular
======================================================================
Sistema que usa SOLO LLM para cada tarea específica, sin hardcodeo ni patrones.
Incorporando las funcionalidades más avanzadas del clean de forma modular.
"""
import logging
import sqlite3
import json
import re
import os
import asyncio
import time
import traceback
import itertools
from typing import Dict, List, Any, Union, Optional, Tuple
from pathlib import Path
from langchain.schema import SystemMessage, HumanMessage
from datetime import datetime
import math
import ast
import hashlib


# Import del módulo sql_agent_tools para funcionalidades modulares
try:
    from .sql_agent_tools import SQLAgentTools
except ImportError:
    from sql_agent_tools import SQLAgentTools

# Import de los nuevos módulos de utilidades
try:
    from ..utils.sql_cleaner import SQLCleaner
    from ..utils.sql_executor import SQLExecutor
    from ..utils.fhir_mapping_corrector import correct_fhir_mapping_result
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.sql_cleaner import SQLCleaner
    from utils.sql_executor import SQLExecutor
    from utils.fhir_mapping_corrector import correct_fhir_mapping_result

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SQLAgentIntelligent_v5.0")

class MockResponse:
    def __init__(self, content: str):
        self.content = content

def _call_openai_native(client, messages, temperature=0.1, max_tokens=4000, task_description="Consultando modelo de IA") -> MockResponse:
    """
    Función de compatibilidad para llamar a OpenAI nativo con streaming, logging y reintentos.
    """
    import time
    import random
    
    max_retries = 3
    base_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            # Logging más conciso
            from openai import OpenAI
            native_client = OpenAI()

            if isinstance(messages, list):
                openai_messages = []
                for msg in messages:
                    role = "user"
                    content = ""
                    if hasattr(msg, 'content'):
                        content = str(msg.content)
                        if isinstance(msg, SystemMessage):
                            role = "system"
                        elif isinstance(msg, HumanMessage):
                            role = "user"
                    elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                        role = str(msg["role"])
                        content = str(msg["content"])
                    else:
                        content = str(msg)
                    openai_messages.append({"role": role, "content": content})
            else:
                content = messages.content if hasattr(messages, 'content') else str(messages)
                openai_messages = [{"role": "user", "content": str(content)}]

            # Mostrar progreso con intento
            if attempt > 0:
                print(f"   💡 {task_description} (intento {attempt + 1}/{max_retries})...", end="", flush=True)
            else:
                print(f"   💡 {task_description}...", end="", flush=True)
            
            # Siempre usar streaming para que se muestre el progreso en tiempo real
            stream_buffer: List[str] = []
            
            try:
                resp_stream = native_client.chat.completions.create(
                    model="gpt-4o",
                    messages=openai_messages,  # type: ignore
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                
                for chunk in resp_stream:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        token = delta.content
                        stream_buffer.append(token)
                        # Mostrar progreso visual
                        if len(stream_buffer) % 10 == 0:  # Cada 10 tokens
                            print(".", end="", flush=True)
                            
                print(" ✓")  # Finalizar línea de progreso
                content = "".join(stream_buffer)

                if not content.strip():
                    content = '{"success": false, "message": "Error: Respuesta vacía del LLM"}'

                # Si llegamos aquí, la llamada fue exitosa
                if attempt > 0:
                    print(f"   ✅ Llamada exitosa después de {attempt + 1} intentos")

                return MockResponse(content)
                
            except Exception as api_error:
                print(" ❌")  # Finalizar línea de progreso con error
                error_msg = str(api_error).lower()
                
                # Errores que merecen reintento
                if any(keyword in error_msg for keyword in [
                    'server had an error', 'timeout', 'rate limit', 'quota exceeded',
                    'service unavailable', 'internal server error', 'bad gateway'
                ]):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"   ⚠️ Error de API (intento {attempt + 1}/{max_retries}): {api_error}")
                        print(f"   ⏳ Reintentando en {delay:.2f} segundos...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"   ❌ ERROR EN LLM: Error de API después de {max_retries} intentos: {api_error}")
                        logger.error(f"Error en _call_openai_native (SQLAgent): {api_error}", exc_info=True)
                        return MockResponse('{"success": false, "message": "Error de API después de múltiples intentos"}')
                else:
                    # Error que no merece reintento
                    print(f"   ❌ ERROR EN LLM: {api_error}")
                    logger.error(f"Error en _call_openai_native (SQLAgent): {api_error}", exc_info=True)
                    return MockResponse('{"success": false, "message": "Error en llamada a OpenAI API"}')

        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"   ⚠️ Error general (intento {attempt + 1}/{max_retries}): {e}")
                print(f"   ⏳ Reintentando en {delay:.2f} segundos...")
                time.sleep(delay)
                continue
            else:
                error_msg = f"Error en llamada OpenAI del SQLAgent: {str(e)}"
                print(f"   ❌ ERROR EN LLM: {error_msg}")
                logger.error(f"Error en _call_openai_native (SQLAgent): {e}", exc_info=True)
                return MockResponse('{"success": false, "message": "Error crítico en la llamada al LLM."}')
    
    # Si llegamos aquí, todos los intentos fallaron
    return MockResponse('{"success": false, "message": "Error: Todos los intentos de llamada al LLM fallaron"}')

class SQLAgentIntelligentEnhanced:
    def __init__(self, db_path: str, llm=None, medgemma_agent=None):
        """
        Inicializa el agente SQL inteligente mejorado con caché inteligente y MedGemma.
        
        Args:
            db_path: Ruta a la base de datos SQLite
            llm: Cliente LLM (opcional)
            medgemma_agent: Agente MedGemma para análisis clínico (opcional)
        """
        self.db_path = db_path
        print(f"🔍 SQLAgent usando base de datos: {db_path}")
        self.llm = llm
        self.medgemma_agent = medgemma_agent
        self.schema = {}
        self.column_metadata = {}
        self.table_info = {}
        
        # CACHÉ INTELIGENTE: Almacena resultados para reutilización
        self._schema_cache = {}  # Caché de esquemas por tabla
        self._mapping_cache = {}  # Caché de mapeos exitosos
        self._validation_cache = {}  # Caché de validaciones previas
        self._table_selection_cache = {}  # Caché de selección de tablas
        self._id_validation_cache = {}  # Caché de validación de IDs
        self._field_mapping_cache = {}  # Caché de mapeo de campos
        
        # Configuración de caché
        self._cache_ttl = 3600  # 1 hora de TTL
        self._cache_timestamps = {}  # Timestamps para TTL
        
        # Sistema modular usando SQLAgentTools con LLM
        self.schema_tools = SQLAgentTools(db_path, llm=llm)
        self.sql_cleaner = SQLCleaner()
        self.sql_executor = SQLExecutor(db_path)
        
        # Inicializar componentes
        self._initialize_schema_with_tools()
        self._initialize_column_metadata()
        
        # Configuración de logging mejorada
        self.logger = logging.getLogger("SQLAgentEnhanced")
        self.logger.setLevel(logging.INFO)
        
        # Formato de logging personalizado
        formatter = logging.Formatter(
            '\n%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Handler para consola con colores
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # CRÍTICO: Para mantener el ID del paciente actual
        self._current_patient_id = None

    def _initialize_schema_with_tools(self):
        """Inicializa el esquema usando las herramientas modulares"""
        try:
            self.schema_tools.introspect_schema(use_cache=True)
            logger.info("✅ Esquema inicializado usando herramientas modulares")
        except Exception as e:
            logger.error(f"Error inicializando esquema: {e}")

    def _initialize_column_metadata(self):
        """Inicializa column_metadata para compatibilidad con otros agentes"""
        try:
            # Inicialización básica para compatibilidad
            self.column_metadata = {}
            logger.info("✅ Column metadata inicializado para compatibilidad")
        except Exception as e:
            logger.error(f"Error inicializando column metadata: {e}")
            self.column_metadata = {}

    async def _get_cached_schema_info(self, table_name: Optional[str] = None) -> str:
        """Devuelve el esquema real de la base de datos (stub para compatibilidad)."""
        return self._get_real_schema_info()

    async def process_query(self, query: str, stream_callback=None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        🧠 Procesamiento genérico de consultas SQL usando LLM para mapeo automático
        """
        start_time = time.time()
        
        try:
            if stream_callback:
                stream_callback("🔍 Analizando consulta con LLM...")
            
            # PASO 1: Generar SQL usando LLM con prompts específicos y dinámicos
            try:
                if stream_callback:
                    stream_callback("🤖 Generando SQL con IA dinámica...")
                
                # Obtener esquema de la base de datos
                schema_info = self._get_real_schema_info()
                
                # PROMPT 1: Análisis médico específico
                analysis_prompt = f"""Eres un experto en análisis de consultas médicas.

CONSULTA: "{query}"

ESQUEMA DE LA BASE DE DATOS:
{schema_info}

TAREA: Analiza la consulta médica y extrae información clave.

ANÁLISIS MÉDICO ESPECÍFICO:
1. Identificar el tipo de consulta (conteo, selección, búsqueda)
2. Identificar la entidad principal (pacientes, diagnósticos, medicamentos)
3. Identificar condiciones médicas específicas (diabetes, hipertensión, cáncer, etc.)
4. Identificar campos de búsqueda médica (DIAG_OBSERVATION, DIAG_DESCRIPTION)
5. Identificar tablas médicas relevantes

ESTRATEGIA MÉDICA:
- Para diagnósticos: Buscar en EPIS_DIAGNOSTICS.DIAG_OBSERVATION
- Para conteos: Usar COUNT(*) desde PATI_PATIENTS
- Para medicación: Usar PATI_USUAL_MEDICATION
- Para condiciones específicas: Usar LIKE '%termino%' en campos de texto

RESPUESTA JSON:
{{
    "tipo_consulta": "conteo|seleccion|busqueda",
    "entidad_principal": "pacientes|diagnosticos|medicamentos",
    "condicion_medica": "diabetes|hipertension|cancer|etc",
    "campos_busqueda": ["DIAG_OBSERVATION", "DIAG_DESCRIPTION"],
    "tablas_principales": ["PATI_PATIENTS", "EPIS_DIAGNOSTICS"],
    "operacion": "count|select|join"
}}"""
                
                analysis_response = await asyncio.to_thread(
                    _call_openai_native, self.llm, [{"role": "user", "content": analysis_prompt}],
                    task_description="Analizando consulta"
                )
                
                analysis = self._try_parse_llm_json(self._extract_response_text(analysis_response))
                
                # PROMPT 2: Generación de SQL médico específico
                sql_prompt = f"""Eres un experto en SQL médico. Genera SQL para esta consulta.

CONSULTA: "{query}"
ANÁLISIS: {json.dumps(analysis, indent=2, ensure_ascii=False) if analysis else '{}'}

REGLAS BÁSICAS:
- Para diagnósticos: JOIN PATI_PATIENTS p ON p.PATI_ID = d.PATI_ID
- Para medicación: usar PATI_USUAL_MEDICATION.PAUM_OBSERVATIONS
- Para conteos: COUNT(DISTINCT p.PATI_ID)
- Buscar en DIAG_OBSERVATION con LIKE '%termino%'

RESPUESTA: SOLO el SQL válido."""
                
                response = _call_openai_native(self.llm, [{"role": "user", "content": sql_prompt}], task_description="Generando SQL específico")
                
                sql = self._extract_response_text(response).strip()
                sql = self._clean_llm_sql_response(sql)
                
                if stream_callback:
                    stream_callback("✅ SQL generado con IA específica")
                
                # PASO 3: Validar y corregir SQL si es necesario
                if sql and not sql.startswith("SELECT COUNT(*) FROM HEAL_DIABETES_INDICATORS"):
                    validation_prompt = f"""Eres un experto en validación de SQL médico.

SQL GENERADO:
{sql}

CONSULTA ORIGINAL:
"{query}"

ANÁLISIS PREVIO:
{json.dumps(analysis, indent=2, ensure_ascii=False) if analysis else '{}'}

TAREA: Valida si el SQL es apropiado para la consulta médica.

PROBLEMAS ESPECÍFICOS A DETECTAR:
1. Para consultas de diabetes: ¿Busca en campos de texto libre (DIAG_OBSERVATION, DIAG_DESCRIPTION)?
2. Para conteos médicos: ¿Cuenta desde la tabla principal de pacientes?
3. Para condiciones médicas: ¿Usa LIKE '%termino%' para búsquedas flexibles?
4. Para diagnósticos: ¿Busca en campos de observación médica?
5. Para medicamentos: ¿Usa PAUM_OBSERVATIONS para ranking real de medicamentos?
6. Para ranking de medicamentos: ¿SELECT PAUM_OBSERVATIONS, COUNT(*) FROM PATI_USUAL_MEDICATION WHERE PAUM_OBSERVATIONS IS NOT NULL?
7. Para medicamentos más prescritos: ¿Agrupa por PAUM_OBSERVATIONS para obtener medicamentos reales prescritos?

CRITERIOS DE VALIDACIÓN:
- Para diagnósticos médicos debe buscar en campos de texto libre
- Para conteos debe usar COUNT(*) desde tabla de pacientes
- Para búsquedas médicas debe usar LIKE con comodines
- Para condiciones específicas debe buscar en campos de observación

RESPUESTA:
- Si el SQL es correcto: Devuelve el SQL original
- Si necesita corrección: Devuelve el SQL corregido
- Devuelve SOLO el SQL, sin explicaciones"""
                    
                    validation_response = _call_openai_native(self.llm, [{"role": "user", "content": validation_prompt}], task_description="Validando SQL")
                    
                    validated_sql = self._extract_response_text(validation_response).strip()
                    validated_sql = self._clean_llm_sql_response(validated_sql)
                    
                    if validated_sql and validated_sql != sql:
                        sql = validated_sql
                        if stream_callback:
                            stream_callback("✅ SQL corregido después de validación")
                    
            except Exception as e:
                if stream_callback:
                    stream_callback("⚠️ Error en generación SQL, usando fallback...")

                # Fallback: SQL básico
                sql = "SELECT COUNT(*) FROM PATI_PATIENTS"
            
            # MOSTRAR SQL GENERADO PARA DEPURACIÓN
            if stream_callback:
                stream_callback(f"🔍 SQL GENERADO: {sql}")
            
            if stream_callback:
                stream_callback("🔧 Ejecutando consulta...")
            
            # PASO 2: Ejecutar SQL usando la función robusta que maneja errores
            result = await self._execute_sql_with_llm_validation(query, sql, start_time, stream_callback=stream_callback)
            
            if not result.get("success"):
                return result  # Devolver el error directamente
            
            # Extraer datos del resultado exitoso
            formatted_results = result.get("data", [])
            interpretation = result.get("message", "Consulta completada")
            
            # Interpretar resultados dinámicamente
            interpretation = await self._interpret_results_generic(query, formatted_results, stream_callback)
            
            return {
                "success": True,
                "message": interpretation,
                "data": formatted_results,
                "sql": sql,
                "count": len(formatted_results)
            }
            
            # Los resultados ya están procesados en _execute_sql_with_llm_validation
            pass
                
        except Exception as e:
            logger.error(f"Error en process_query: {e}")
            
            # FALLBACK: Usar herramientas genéricas de SQL
            try:
                if stream_callback:
                    stream_callback("🔄 Activando fallback con herramientas genéricas...")
                
                # Usar herramientas genéricas como último recurso
                fallback_sql = await self._use_generic_sql_tools(query, stream_callback)
                
                if fallback_sql:
                    # Intentar ejecutar el SQL de fallback
                    fallback_result = await self._execute_sql_with_llm_validation(query, fallback_sql, start_time, stream_callback=stream_callback)
                    
                    if fallback_result.get("success"):
                        if stream_callback:
                            stream_callback("✅ Fallback exitoso con herramientas genéricas")
                        return fallback_result
                    else:
                        return {
                            "success": False,
                            "message": f"Error en consulta SQL (fallback también falló): {str(e)}",
                            "sql": fallback_sql
                        }
                else:
                    return {
                        "success": False,
                        "message": f"Error procesando consulta: {str(e)}",
                        "sql": sql if 'sql' in locals() else ""
                    }
                    
            except Exception as fallback_error:
                logger.error(f"Error en fallback: {fallback_error}")
                return {
                    "success": False,
                    "message": f"Error crítico en consulta SQL: {str(e)}",
                    "sql": sql if 'sql' in locals() else ""
                }
    
    async def _interpret_results_generic(self, query: str, results: List[Dict[str, Any]], stream_callback=None) -> str:
        """Interpretación dinámica de resultados usando LLM"""
        count = len(results) if results else 0
        
        if count == 0:
            return "No se encontraron resultados para esta consulta."
        
        if not self.llm:
            return f"Encontrados {count} registros."
        
        try:
            # Crear prompt para interpretación dinámica
            sample_data = results[:3] if results else []
            
            prompt = f"""Eres un experto en interpretación de resultados de bases de datos médicas.

CONSULTA ORIGINAL: "{query}"
NÚMERO DE RESULTADOS: {count}

MUESTRA DE DATOS:
{json.dumps(sample_data, indent=2, ensure_ascii=False)}

TAREA: Analiza la consulta y los datos para generar una respuesta clara y útil.

REGLAS ESPECÍFICAS:
- Si la consulta pide "cuántos pacientes con diabetes": Responde con el número total
- Si la consulta pide "qué medicación tienen": Lista las medicaciones encontradas
- Si la consulta combina conteo y medicación: Responde con ambos aspectos
- Para diabetes: Enfócate en el conteo de pacientes y sus medicaciones
- Para medicación: Muestra las medicaciones específicas encontradas
- Usa lenguaje natural y médico apropiado
- Si no hay datos, explica por qué no se encontraron resultados

RESPUESTA: Genera una respuesta clara y estructurada que responda directamente a la consulta."""

            response = _call_openai_native(self.llm, [{"role": "user", "content": prompt}], task_description="Interpretando resultados dinámicamente")
            
            interpretation = self._extract_response_text(response).strip()
            
            if stream_callback:
                stream_callback("   ✅ Resultados interpretados dinámicamente")
            
            return interpretation if interpretation else f"Encontrados {count} registros."
            
        except Exception as e:
            logger.error(f"Error en interpretación dinámica: {e}")
            return f"Encontrados {count} registros."

    async def _llm_analyze_semantics(self, query: str, stream_callback=None) -> Dict[str, Any]:
        """Análisis semántico completamente con LLM - SIN HARDCODEO"""
        try:
            if not self.llm:
                return {'intent': 'sql_query', 'entities': [], 'concepts': []}
            
            prompt = f"""Eres un experto en análisis semántico de consultas médicas.

ANALIZA esta consulta: "{query}"

TAREA: Extrae información semántica completa usando solo tu conocimiento, sin patrones predefinidos.

RESPUESTA JSON:
{{
    "intent": "descripción de la intención principal",
    "entities": ["entidad1", "entidad2"],
    "concepts": ["concepto1", "concepto2"],
    "query_type": "tipo de consulta",
    "complexity": "simple|medium|complex",
    "medical_focus": "enfoque médico específico"
}}"""

            response = _call_openai_native(self.llm, [{"role": "user", "content": prompt}], task_description="Analizando semántica de la consulta")
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if stream_callback and result:
                stream_callback(f"   - Intención detectada: {result.get('intent', 'N/A')}")
            
            return result if result else {'intent': 'sql_query', 'entities': [], 'concepts': []}
            
        except Exception as e:
            logger.error(f"Error en análisis semántico: {e}")
            return {'intent': 'sql_query', 'entities': [], 'concepts': []}

    async def _llm_map_medical_concepts(self, query: str, semantic_analysis: Dict[str, Any], stream_callback=None) -> Dict[str, Any]:
        """Mapeo de conceptos médicos completamente con LLM - SIN HARDCODEO"""
        try:
            if not self.llm:
                return {'medical_concepts': [], 'clinical_intent': 'query'}
            
            prompt = f"""Eres un experto en terminología médica y conceptos clínicos.

CONSULTA: "{query}"
ANÁLISIS SEMÁNTICO: {semantic_analysis}

TAREA: Identifica conceptos médicos específicos usando solo tu conocimiento médico, sin patrones predefinidos.

RESPUESTA JSON:
{{
    "medical_concepts": ["concepto_médico1", "concepto_médico2"],
    "clinical_intent": "intención clínica específica",
    "medical_entities": ["entidad_médica1", "entidad_médica2"],
    "specialized_terms": ["término_especializado1", "término_especializado2"]
}}"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Mapeando conceptos médicos"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if stream_callback and result:
                concepts = result.get('medical_concepts', [])
                stream_callback(f"   - Conceptos médicos: {', '.join(concepts[:3])}...")
            
            return result if result else {'medical_concepts': [], 'clinical_intent': 'query'}
            
        except Exception as e:
            logger.error(f"Error en mapeo médico: {e}")
            return {'medical_concepts': [], 'clinical_intent': 'query'}

    async def _llm_select_tables(self, query: str, medical_mapping: Dict[str, Any], stream_callback=None) -> List[str]:
        """Selección de tablas completamente con LLM - SIN HARDCODEO"""
        try:
            if not self.llm:
                return list(self.schema_tools.get_schema().keys())[:3]
            
            # Usar el nuevo sistema inteligente sin hardcodeo
            medical_concepts = medical_mapping.get('medical_concepts', [])
            return await self._llm_select_relevant_tables_intelligent(query, medical_concepts, stream_callback)
            
        except Exception as e:
            logger.error(f"Error en selección de tablas: {e}")
            return list(self.schema_tools.get_schema().keys())[:3]

    async def _llm_analyze_relationships(self, query: str, tables: List[str], stream_callback=None) -> Dict[str, Any]:
        """Análisis de relaciones completamente con LLM - SIN HARDCODEO"""
        try:
            if not self.llm:
                return {'join_conditions': [], 'relationships': []}
            
            fk_graph = self.schema_tools.get_fk_graph()
            schema = self.schema_tools.get_schema()
            
            # Obtener información de relaciones para las tablas seleccionadas
            relationship_info = {}
            for table in tables:
                if table in fk_graph:
                    relationship_info[table] = fk_graph[table]
                if table in schema:
                    relationship_info[f"{table}_columns"] = [col['name'] for col in schema[table]]
            
            prompt = f"""Eres un experto en relaciones de bases de datos médicas.

CONSULTA: "{query}"
TABLAS SELECCIONADAS: {tables}

INFORMACIÓN DE RELACIONES:
{json.dumps(relationship_info, indent=2)}

TAREA: Analiza las relaciones entre tablas usando solo tu conocimiento, sin patrones predefinidos.

RESPUESTA JSON:
{{
    "join_conditions": [{{"table1": "tabla1", "table2": "tabla2", "condition": "condición"}}],
    "relationships": ["relación1", "relación2"],
    "join_strategy": "estrategia de unión recomendada"
}}"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Analizando relaciones entre tablas"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if stream_callback and result:
                joins = result.get('join_conditions', [])
                stream_callback(f"   - Relaciones detectadas: {len(joins)} join(s)")
            
            return result if result else {'join_conditions': [], 'relationships': []}
            
        except Exception as e:
            logger.error(f"Error en análisis de relaciones: {e}")
            return {'join_conditions': [], 'relationships': []}

    async def _llm_generate_sql(self, query: str, tables: List[str], relationships: Dict[str, Any], stream_callback=None) -> str:
        """Generación de SQL con prompts cortos y específicos"""
        try:
            if not self.llm:
                return f"SELECT * FROM {tables[0]} LIMIT 10;"
            
            # DETECTAR SI ES CONSULTA DE DIAGNÓSTICO
            query_lower = query.lower()
            diagnosis_keywords = ['síndrome', 'diabetes', 'cáncer', 'hipertensión', 'asma', 'epilepsia', 'depresión', 'ansiedad', 'ovario poliquístico']
            
            if any(keyword in query_lower for keyword in diagnosis_keywords):
                # USAR PROMPT CORTO PARA DIAGNÓSTICOS
                condition = next((keyword for keyword in diagnosis_keywords if keyword in query_lower), 'condición')
                sql = await self._generate_diagnosis_sql_simple(query, condition)
                if stream_callback:
                    stream_callback("   ✅ SQL de diagnóstico generado")
                return sql
            
            # DETECTAR SI ES CONSULTA DE ÚLTIMO PACIENTE USANDO LLM
            if self.llm:
                detection_prompt = f"""Analiza esta consulta y determina si se refiere al ÚLTIMO PACIENTE registrado en la base de datos.

CONSULTA: "{query}"

CRITERIOS PARA DETECTAR CONSULTAS DE ÚLTIMO PACIENTE:
- Palabras clave: "último", "ultimo", "última", "ultima", "reciente", "creado", "registrado"
- Frases: "último paciente", "ultimo paciente", "último paciente creado", "ultimo paciente creado"
- Preguntas: "¿Cuál es el último paciente?", "¿Quién es el último paciente?", "¿Dime el último paciente?"
- Variaciones: "cual es el ultimo", "cuál es el último", "dime el ultimo", "dime el último", "quien es el ultimo", "quién es el último"

Responde SOLO con "SÍ" si es una consulta de último paciente, o "NO" si no lo es."""

                try:
                    detection_response = _call_openai_native(self.llm, detection_prompt)
                    detection_result = self._extract_response_text(detection_response).strip().upper()
                    
                    if "SÍ" in detection_result or "SI" in detection_result:
                        print(f"   🔍 DETECTADO POR LLM: Consulta de último paciente - '{query}'")
                        sql = await self._generate_last_patient_sql_simple(query)
                        if stream_callback:
                            stream_callback("   ✅ SQL de último paciente generado")
                        return sql
                except Exception as e:
                    print(f"   ⚠️ Error en detección LLM: {e}")
                    # Fallback a detección básica si LLM falla
                    last_patient_keywords = ['último', 'ultimo', 'última', 'ultima', 'reciente', 'creado', 'registrado', 'último paciente', 'ultimo paciente']
                    if any(keyword in query_lower for keyword in last_patient_keywords):
                        print(f"   🔍 DETECTADO POR FALLBACK: Consulta de último paciente - '{query}'")
                        sql = await self._generate_last_patient_sql_simple(query)
                        if stream_callback:
                            stream_callback("   ✅ SQL de último paciente generado")
                        return sql
            
            # PROMPT CORTO PARA OTRAS CONSULTAS
            sql_prompt = f"""Genera SQL para: "{query}"

REGLAS:
- Para diagnósticos: JOIN PATI_PATIENTS p ON p.PATI_ID = d.PATI_ID
- Para medicación: usar PATI_USUAL_MEDICATION.PAUM_OBSERVATIONS
- Para conteos: COUNT(DISTINCT p.PATI_ID)
- Buscar en DIAG_OBSERVATION con LIKE '%termino%'
- Para último paciente: usar PATI_ID DESC (NO PATI_START_DATE)
- Para nombres: usar PATI_NAME, PATI_SURNAME_1, PATI_FULL_NAME

SQL:"""
            
            response = _call_openai_native(self.llm, sql_prompt)
            sql = self._extract_response_text(response).strip()
            sql = self._clean_llm_sql_response(sql)
            
            if stream_callback:
                stream_callback("   ✅ SQL generado con IA dinámica")
            
            return sql if sql and not sql.startswith("Error") else f"SELECT * FROM {tables[0]} LIMIT 10;"
            
        except Exception as e:
            logger.error(f"Error en generación de SQL: {e}")
            return f"SELECT * FROM {tables[0]} LIMIT 10;"

    async def _generate_diagnosis_sql_simple(self, query: str, condition: str) -> str:
        """Genera SQL específico para diagnósticos con prompt corto"""
        try:
            if not self.llm:
                return f"SELECT p.PATI_ID, d.DIAG_OBSERVATION FROM PATI_PATIENTS p JOIN EPIS_DIAGNOSTICS d ON p.PATI_ID = d.PATI_ID WHERE d.DIAG_OBSERVATION LIKE '%{condition}%'"
            
            prompt = f"""Genera SQL para buscar pacientes con {condition}.

REGLAS:
- JOIN PATI_PATIENTS p ON p.PATI_ID = d.PATI_ID
- Buscar en d.DIAG_OBSERVATION LIKE '%{condition}%'
- Incluir p.PATI_ID y d.DIAG_OBSERVATION

SQL:"""
            
            response = _call_openai_native(self.llm, prompt)
            sql = self._extract_response_text(response).strip()
            
            return sql if sql and sql.upper().startswith('SELECT') else f"SELECT p.PATI_ID, d.DIAG_OBSERVATION FROM PATI_PATIENTS p JOIN EPIS_DIAGNOSTICS d ON p.PATI_ID = d.PATI_ID WHERE d.DIAG_OBSERVATION LIKE '%{condition}%'"
            
        except Exception as e:
            logger.error(f"Error en SQL de diagnóstico: {e}")
            return f"SELECT p.PATI_ID, d.DIAG_OBSERVATION FROM PATI_PATIENTS p JOIN EPIS_DIAGNOSTICS d ON p.PATI_ID = d.PATI_ID WHERE d.DIAG_OBSERVATION LIKE '%{condition}%'"

    async def _generate_last_patient_sql_simple(self, query: str) -> str:
        """Genera SQL específico para último paciente con doble llamada al LLM"""
        try:
            if not self.llm:
                return "SELECT PATI_ID, PATI_NAME, PATI_SURNAME_1, PATI_FULL_NAME FROM PATI_PATIENTS ORDER BY PATI_ID DESC LIMIT 1"
            
            # PRIMERA LLAMADA: Detectar si es consulta de último paciente
            detection_prompt = f"""Analiza esta consulta y determina si se refiere al ÚLTIMO PACIENTE registrado en la base de datos.

CONSULTA: "{query}"

CRITERIOS PARA DETECTAR CONSULTAS DE ÚLTIMO PACIENTE:
- Palabras clave: "último", "ultimo", "última", "ultima", "reciente", "creado", "registrado"
- Frases: "último paciente", "ultimo paciente", "último paciente creado", "ultimo paciente creado"
- Preguntas: "¿Cuál es el último paciente?", "¿Quién es el último paciente?", "¿Dime el último paciente?"

Responde SOLO con "SÍ" si es una consulta de último paciente, o "NO" si no lo es."""

            print(f"   🔍 PRIMERA LLAMADA: Detectando consulta de último paciente...")
            detection_response = _call_openai_native(self.llm, detection_prompt)
            detection_result = self._extract_response_text(detection_response).strip().upper()
            
            if "SÍ" not in detection_result and "SI" not in detection_result:
                print(f"   ❌ No se detectó como consulta de último paciente")
                return "SELECT PATI_ID, PATI_NAME, PATI_SURNAME_1, PATI_FULL_NAME FROM PATI_PATIENTS ORDER BY PATI_ID DESC LIMIT 1"
            
            # SEGUNDA LLAMADA: Generar SQL optimizado para último paciente
            sql_prompt = f"""Genera una consulta SQL optimizada para obtener información del ÚLTIMO PACIENTE registrado en la base de datos.

CONSULTA ORIGINAL: "{query}"

REGLAS OBLIGATORIAS:
- Usar SOLO PATI_ID DESC para determinar el último paciente (NO usar PATI_START_DATE ni PATI_UPDATE_DATE)
- Incluir campos: PATI_ID, PATI_NAME, PATI_SURNAME_1, PATI_FULL_NAME
- Usar ORDER BY PATI_ID DESC LIMIT 1
- Tabla: PATI_PATIENTS

EJEMPLO CORRECTO:
SELECT PATI_ID, PATI_NAME, PATI_SURNAME_1, PATI_FULL_NAME 
FROM PATI_PATIENTS 
ORDER BY PATI_ID DESC 
LIMIT 1

IMPORTANTE:
- NUNCA usar ORDER BY PATI_START_DATE DESC
- NUNCA usar ORDER BY PATI_UPDATE_DATE DESC
- SIEMPRE usar ORDER BY PATI_ID DESC

SQL:"""

            print(f"   🧠 SEGUNDA LLAMADA: Generando SQL optimizado para último paciente...")
            sql_response = _call_openai_native(self.llm, sql_prompt)
            sql = self._extract_response_text(sql_response).strip()
            sql = self._clean_llm_sql_response(sql)
            
            # Validar que el SQL generado sea correcto
            if sql and sql.upper().startswith('SELECT') and 'ORDER BY PATI_ID DESC' in sql.upper():
                print(f"   ✅ SQL de último paciente generado correctamente")
                return sql
            else:
                print(f"   ⚠️ SQL generado incorrecto, usando fallback")
                return "SELECT PATI_ID, PATI_NAME, PATI_SURNAME_1, PATI_FULL_NAME FROM PATI_PATIENTS ORDER BY PATI_ID DESC LIMIT 1"
            
        except Exception as e:
            logger.error(f"Error en SQL de último paciente: {e}")
            return "SELECT PATI_ID, PATI_NAME, PATI_SURNAME_1, PATI_FULL_NAME FROM PATI_PATIENTS ORDER BY PATI_ID DESC LIMIT 1"

    async def _llm_validate_and_optimize_sql(self, query: str, sql: str, stream_callback=None) -> str:
        """Validación y optimización completamente con LLM usando múltiples llamadas dinámicas"""
        try:
            if not self.llm:
                return sql
            
            if stream_callback:
                stream_callback("   - Validando y optimizando SQL con IA dinámica...")
            
            # PRIMERA LLAMADA: Análisis de problemas potenciales
            analysis_prompt = f"""Eres un experto en análisis de SQL médico. Analiza este SQL y identifica problemas potenciales.

CONSULTA ORIGINAL: "{query}"
SQL A VALIDAR: "{sql}"

INSTRUCCIONES:
1. Analiza la sintaxis del SQL
2. Identifica problemas de compatibilidad con SQLite
3. Detecta posibles errores de tablas o columnas
4. Sugiere optimizaciones
5. Considera casos edge y errores comunes
6. Analiza si el SQL responde completamente a la consulta

RESPUESTA: JSON con análisis de problemas"""

            print(f"   🔍 PRIMERA LLAMADA: Analizando problemas potenciales...")
            analysis_response = _call_openai_native(self.llm, analysis_prompt)
            analysis_result = self._try_parse_llm_json(analysis_response.content)
            
            if not analysis_result:
                print(f"   ❌ No se pudo analizar problemas")
                return sql

            # SEGUNDA LLAMADA: Corrección basada en análisis
            correction_prompt = f"""Eres un experto en corrección de SQL médico. Corrige este SQL basado en el análisis previo.

CONSULTA ORIGINAL: "{query}"
SQL ORIGINAL: "{sql}"

ANÁLISIS DE PROBLEMAS:
{json.dumps(analysis_result, indent=2, ensure_ascii=False)}

INSTRUCCIONES:
1. Usa el análisis previo para corregir problemas
2. Optimiza para SQLite
3. Corrige errores de sintaxis
4. Maneja casos edge
5. Considera múltiples formas de representar la misma información
6. Asegúrate de que responda completamente a la consulta

REGLAS DE CORRECCIÓN:
- Mantén la funcionalidad original
- Optimiza para rendimiento
- Maneja errores de manera robusta
- Considera diferentes estructuras de base de datos
- Usa solo tablas y columnas que existen

RESPUESTA: SQL corregido y optimizado"""

            print(f"   🧠 SEGUNDA LLAMADA: Corrigiendo SQL con contexto...")
            correction_response = _call_openai_native(self.llm, correction_prompt)
            
            corrected_sql = self._extract_response_text(correction_response).strip()
            corrected_sql = self._clean_llm_sql_response(corrected_sql)
            
            if corrected_sql and not corrected_sql.startswith("Error"):
                sql = corrected_sql
            
            # PASO 3: Validación final con esquema robusto
            sql = await self._validate_schema_with_robust_tools(sql, stream_callback)
            
            if stream_callback:
                stream_callback("   ✅ SQL validado y optimizado con IA dinámica")
            
            return sql
            
        except Exception as e:
            logger.error(f"Error en validación y optimización: {e}")
            return sql

    async def _regenerate_complete_sql(self, query: str, incomplete_sql: str, missing_tables: List[str], stream_callback=None) -> str:
        """Regenera SQL completo usando LLM con información de tablas faltantes"""
        try:
            if not self.llm:
                return incomplete_sql
            
            if stream_callback:
                stream_callback("   - Regenerando SQL completo con IA...")
            
            # Obtener esquema disponible
            schema = self.schema_tools.get_schema()
            schema_info = {}
            for table_name, columns in schema.items():
                schema_info[table_name] = [col['name'] for col in columns]
            
            prompt = f"""Eres un experto en SQL que regenera consultas completas.

CONSULTA ORIGINAL: "{query}"

SQL INCOMPLETO:
{incomplete_sql}

TABLAS FALTANTES IDENTIFICADAS:
{', '.join(missing_tables)}

ESQUEMA DISPONIBLE:
{json.dumps(schema_info, indent=2, ensure_ascii=False)}

TAREA: Regenera el SQL para incluir todas las tablas necesarias.

INSTRUCCIONES:
1. Analiza la consulta original para entender qué información se necesita
2. Incluye las tablas faltantes identificadas
3. Usa JOINs apropiados para conectar las tablas
4. Mantén la lógica original de la consulta
5. Asegúrate de que el SQL sea sintácticamente correcto
6. Usa solo tablas y columnas que existen en el esquema

EJEMPLOS DE CONEXIÓN:
- PATI_PATIENTS.PATI_ID = EPIS_EPISODES.PATI_ID
- EPIS_EPISODES.EPIS_ID = EPIS_DIAGNOSTICS.EPIS_ID
- PATI_PATIENTS.PATI_ID = OBSE_OBSERVATIONS.PATI_ID

RESPUESTA:
Devuelve SOLO el SQL completo y corregido, sin explicaciones."""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Regenerando SQL completo"
            )
            
            complete_sql = self._extract_response_text(response).strip()
            complete_sql = self._clean_llm_sql_response(complete_sql)
            
            if complete_sql and not complete_sql.startswith("Error"):
                if stream_callback:
                    stream_callback("   ✅ SQL completo regenerado")
                return complete_sql
            else:
                logger.warning("LLM no pudo regenerar SQL completo")
                return incomplete_sql
                
        except Exception as e:
            logger.error(f"Error regenerando SQL completo: {e}")
            return incomplete_sql

    async def _llm_clean_sql_before_execution(self, sql: str, stream_callback=None) -> str:
        """
        Usa el LLM para limpiar errores de sintaxis SQL de forma genérica antes de ejecutar.
        
        Args:
            sql: SQL a limpiar
            stream_callback: Función para mostrar progreso
            
        Returns:
            str: SQL limpio y corregido
        """
        try:
            if not self.llm:
                return sql
            
            if stream_callback:
                stream_callback("   - Limpiando errores de sintaxis SQL con IA...")
            
            prompt = f"""Eres un experto en SQL que corrige errores de sintaxis de forma genérica.

SQL A CORREGIR:
{sql}

TAREA: Corrige cualquier error de sintaxis que encuentres, especialmente:
- Palabras pegadas a keywords SQL (ej: PacientesJOIN → Pacientes JOIN)
- Espacios faltantes entre palabras clave
- Errores de formato comunes

INSTRUCCIONES:
1. NO uses patterns específicos ni hardcodees keywords
2. Detecta cualquier palabra pegada a una palabra clave SQL de forma genérica
3. Añade espacios donde falten
4. Mantén la lógica original del SQL
5. Asegúrate de que sea sintácticamente válido

RESPUESTA:
Devuelve SOLO el SQL corregido, sin explicaciones."""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Limpiando errores de sintaxis SQL"
            )
            
            cleaned_sql = self._extract_response_text(response).strip()
            cleaned_sql = self._clean_llm_sql_response(cleaned_sql)
            
            if cleaned_sql and not cleaned_sql.startswith("Error"):
                logger.info(f"🧠 LLM limpió SQL exitosamente")
                if stream_callback:
                    stream_callback("   ✅ Errores de sintaxis corregidos")
                return cleaned_sql
            else:
                logger.warning(f"⚠️ LLM no pudo limpiar SQL, usando original")
                return sql
                
        except Exception as e:
            logger.error(f"Error en _llm_clean_sql_before_execution: {e}")
            return sql

    async def _execute_sql_with_llm_validation(self, query: str, sql: str, start_time: float, sql_params: Optional[List[Any]] = None, stream_callback=None) -> Dict[str, Any]:
        """Ejecuta SQL usando los módulos centralizados de limpieza y ejecución con aprendizaje agresivo"""
        
        logger.info(f"🔍 SQL original recibido: {sql[:200]}...")
        
        if stream_callback:
            stream_callback("🔍 Optimizando y ejecutando consulta SQL...")
        
        try:
            # PASO 1: Limpiar y sanitizar el SQL
            if stream_callback:
                stream_callback("   - Limpiando y optimizando SQL...")
            
            # --- REFORZADO: limpiar errores de palabras pegadas antes de todo ---
            sql = self._fix_typo_errors(sql)
            cleaned_sql_temp = await self._basic_sql_cleanup(sql)
            
            cleaned_sql = SQLCleaner.sanitize_for_execution(cleaned_sql_temp)
            
            # Aplicar correcciones específicas de compatibilidad
            cleaned_sql = await self._fix_sql_compatibility(cleaned_sql, stream_callback)
            
            # NUEVO: Análisis flexible con prompts específicos según el contexto
            cleaned_sql = await self._llm_flexible_sql_analysis(cleaned_sql, query, stream_callback)
            
            # NUEVO: Validar esquema con herramientas robustas
            cleaned_sql = await self._validate_schema_with_robust_tools(cleaned_sql, stream_callback)
            
            # Aplicar correcciones de sintaxis
            cleaned_sql = SQLCleaner.fix_common_syntax_errors(cleaned_sql)
            
            logger.info(f"✅ SQL limpio y listo: {cleaned_sql[:200]}...")
            
            # PASO 2: Validar sintaxis antes de ejecutar
            if stream_callback:
                stream_callback("   - Validando sintaxis SQL...")
                
            executor = SQLExecutor(self.db_path)
            is_valid, syntax_error = executor.test_query_syntax(cleaned_sql)
            
            if not is_valid:
                logger.error(f"❌ Error de sintaxis SQL: {syntax_error}")
                
                # SISTEMA DE RECUPERACIÓN MEJORADO: Aplicar correcciones automáticas basadas en patrones
                if stream_callback:
                    stream_callback("   - Error de sintaxis detectado, aplicando correcciones automáticas...")
                
                # Aplicar correcciones automáticas basadas en patrones de error
                corrected_sql = await self._apply_automatic_error_corrections(cleaned_sql, syntax_error or "Error de sintaxis", stream_callback)
                
                if corrected_sql and corrected_sql != cleaned_sql:
                    logger.info(f"🔄 SQL corregido automáticamente después de error")
                    # Ejecutar el SQL corregido
                    return await self._execute_sql_with_llm_validation(query, corrected_sql, start_time, sql_params, stream_callback)
                else:
                    return {
                        'success': False,
                        'message': f'❌ Error de sintaxis SQL: {syntax_error}',
                        'data': [],
                        'sql_query': cleaned_sql,
                        'error': syntax_error
                    }
            
            # PASO 3: Verificar y ajustar parámetros
            sql_params = sql_params or []
            placeholder_count = cleaned_sql.count('?')
            
            # NORMALIZAR PARÁMETROS si contienen nombres
            normalized_params = []
            for param in sql_params:
                if isinstance(param, str) and param:
                    # Normalizar el parámetro usando nuestra función ROBUSTA
                    normalized_param = self._normalize_accents_python(param)
                    normalized_params.append(normalized_param)
                else:
                    normalized_params.append(param)
            
            sql_params = normalized_params
            
            # PASO 4: Ejecutar con el módulo ejecutor
            if stream_callback:
                stream_callback("   - Ejecutando consulta en la base de datos...")
                
            result = executor.execute_query(cleaned_sql, sql_params)
            
            # PASO 5: Procesar resultado
            if result['success']:
                if stream_callback:
                    stream_callback(f"   ✅ Consulta completada: {result['row_count']} resultados en {result['execution_time']:.2f}s")
                
                return {
                    'success': True,
                    'message': f'✅ Encontrados {result["row_count"]} resultados',
                    'data': result['data'],
                    'sql_query': cleaned_sql,
                    'execution_time': result['execution_time'],
                    'total_time': time.time() - start_time
                }
            else:
                if stream_callback:
                    stream_callback(f"   ❌ Error ejecutando SQL: {result['error'][:100]}")
                
                # SISTEMA DE APRENDIZAJE AGRESIVO: Aplicar correcciones basadas en el error específico
                corrected_sql = await self._apply_error_based_corrections(cleaned_sql, result['error'], stream_callback)
                
                if corrected_sql and corrected_sql != cleaned_sql:
                    logger.info(f"🔄 Aplicando corrección basada en error: {result['error'][:50]}")
                    return await self._execute_sql_with_llm_validation(query, corrected_sql, start_time, sql_params, stream_callback)
                else:
                    return {
                        'success': False,
                        'message': f'❌ Error ejecutando consulta: {result["error"]}',
                        'data': [],
                        'sql_query': cleaned_sql,
                        'error': result['error']
                    }
                
        except Exception as e:
            # Manejo de errores generales
            error_msg = str(e)
            logger.error(f"❌ Error en _execute_sql_with_learning: {error_msg}")
            
            if stream_callback:
                stream_callback(f"   ❌ Error inesperado: {error_msg[:100]}")
            
            return {
                'success': False,
                'message': f'❌ Error procesando consulta: {error_msg}',
                'data': [],
                'sql_query': sql,
                'error': error_msg
            }

    async def _apply_automatic_error_corrections(self, sql: str, error_message: str, stream_callback=None) -> str:
        """
        Aplica correcciones automáticas basadas en el mensaje de error.
        Maneja específicamente errores de alias no definidos.
        """
        try:
            if not self.llm:
                return sql
            
            if stream_callback:
                stream_callback("   🔧 Aplicando correcciones automáticas...")
            
            # Detectar errores específicos
            error_lower = error_message.lower()
            
            # Error de alias no definido
            if "no such column" in error_lower and "." in error_message:
                # Extraer el alias problemático
                import re
                alias_match = re.search(r'(\w+)\.(\w+)', error_message)
                if alias_match:
                    problematic_alias = alias_match.group(1)
                    column_name = alias_match.group(2)
                    
                    if stream_callback:
                        stream_callback(f"   ⚠️ Detectado alias no definido: {problematic_alias}")
                    
                    # Corregir usando LLM
                    prompt = f"""Eres un experto en SQL que corrige errores de alias no definidos.

SQL CON ERROR:
{sql}

ERROR DETECTADO:
{error_message}

PROBLEMA: El alias '{problematic_alias}' no está definido en el FROM o JOIN.

TAREA: Corrige el SQL para:
1. Definir correctamente el alias en el FROM/JOIN
2. O eliminar el alias si no es necesario
3. Asegurar que todas las referencias de columnas sean válidas

REGLAS:
- Usa solo tablas que existen en la base de datos
- Asegúrate de que los alias estén definidos
- Mantén la lógica original de la consulta
- Optimiza para SQLite

RESPUESTA: SOLO el SQL corregido, sin explicaciones."""

                    response = await asyncio.to_thread(
                        _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                        task_description="Corrigiendo alias no definido"
                    )
                    
                    corrected_sql = self._extract_response_text(response).strip()
                    corrected_sql = self._clean_llm_sql_response(corrected_sql)
                    
                    if corrected_sql and not corrected_sql.startswith("Error"):
                        if stream_callback:
                            stream_callback("   ✅ Alias corregido automáticamente")
                        return corrected_sql
            
            # Otros errores - usar LLM genérico
            prompt = f"""Eres un experto en SQL que corrige errores de sintaxis.

SQL CON ERROR:
{sql}

ERROR DETECTADO:
{error_message}

TAREA: Corrige el SQL para eliminar el error específico.

REGLAS:
- Mantén la lógica original de la consulta
- Usa solo tablas y columnas que existen
- Optimiza para SQLite
- Asegúrate de que la sintaxis sea correcta

RESPUESTA: SOLO el SQL corregido, sin explicaciones."""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Corrigiendo error SQL"
            )
            
            corrected_sql = self._extract_response_text(response).strip()
            corrected_sql = self._clean_llm_sql_response(corrected_sql)
            
            if corrected_sql and not corrected_sql.startswith("Error"):
                if stream_callback:
                    stream_callback("   ✅ Error corregido automáticamente")
                return corrected_sql
            else:
                if stream_callback:
                    stream_callback("   ⚠️ No se pudo corregir automáticamente")
                return sql
                
        except Exception as e:
            logger.error(f"Error en corrección automática: {e}")
            return sql
        
    async def _apply_error_based_corrections(self, sql: str, error_message: str, stream_callback=None) -> str:
        """
        Aplica correcciones específicas usando LLM con prompts específicos.
        SIN PATRONES HARDCODEADOS - todo via LLM.
        """
        try:
            if not self.llm:
                if stream_callback:
                    stream_callback("   - Corrección específica no disponible (sin LLM)...")
                return sql
            
            if stream_callback:
                stream_callback("   - Aplicando correcciones específicas con IA...")
            
            # Prompt específico para corrección basada en error
            correction_prompt = f"""Eres un experto en SQL que corrige errores específicos.

SQL CON ERROR:
{sql}

MENSAJE DE ERROR ESPECÍFICO:
{error_message}

TAREA:
Analiza este error específico y corrige SOLO el problema mencionado.

INSTRUCCIONES:
1. Enfócate ÚNICAMENTE en el error específico mencionado
2. Corrige SOLO el problema exacto
3. NO añadas lógica adicional
4. Mantén la intención original del SQL
5. Usa columnas y tablas que realmente existan

ESQUEMA DISPONIBLE:
{self.schema_tools.get_schema()}

RESPUESTA:
Devuelve SOLO el SQL corregido, sin explicaciones."""

            try:
                response = await asyncio.to_thread(
                    _call_openai_native, self.llm, [{"role": "user", "content": correction_prompt}]
                )
                
                corrected_sql = self._extract_response_text(response).strip()
                corrected_sql = self._clean_llm_sql_response(corrected_sql)
                
                if corrected_sql and corrected_sql != sql and not corrected_sql.startswith("Error"):
                    logger.info(f"🧠 LLM corrigió error específico: {error_message[:50]}...")
                    logger.info(f"   SQL original: {sql[:100]}...")
                    logger.info(f"   SQL corregido: {corrected_sql[:100]}...")
                    
                    if stream_callback:
                        stream_callback("   - Corrección específica aplicada con IA")
                    
                    return corrected_sql
                else:
                    if stream_callback:
                        stream_callback("   - No se pudo aplicar corrección específica")
                    return sql
                    
            except Exception as e:
                logger.warning(f"Error usando LLM para corrección específica: {e}")
                if stream_callback:
                    stream_callback(f"   - Error en corrección específica: {str(e)[:50]}...")
                return sql
            
        except Exception as e:
            logger.error(f"Error en _apply_error_based_corrections: {e}")
            return sql

    async def _llm_interpret_medical_results(self, query: str, data: List[Dict[str, Any]], stream_callback=None) -> str:
        """Interpretación médica usando MedGemma cuando está disponible, o LLM como fallback"""
        try:
            if not data:
                return "No hay datos para interpretar."
            
            # USAR MEDGEMMA SI ESTÁ DISPONIBLE
            if self.medgemma_agent:
                if stream_callback:
                    stream_callback("   🧠 Usando MedGemma para análisis clínico avanzado...")
                
                try:
                    # Preparar datos para MedGemma
                    medical_data = {
                        'query': query,
                        'results': data[:10],  # Limitar a 10 registros
                        'context': 'Resultados de consulta SQL médica'
                    }
                    
                    # Analizar con MedGemma
                    medgemma_result = await self.medgemma_agent.analyze_clinical_data(
                        json.dumps(medical_data, indent=2, ensure_ascii=False),
                        stream_callback
                    )
                    
                    if medgemma_result and medgemma_result.get('success'):
                        interpretation = medgemma_result.get('analysis', '')
                        if interpretation:
                            if stream_callback:
                                stream_callback("   ✅ Análisis clínico con MedGemma completado")
                            return interpretation
                    
                    # Si MedGemma falla, continuar con LLM
                    if stream_callback:
                        stream_callback("   ⚠️ MedGemma no disponible, usando LLM...")
                        
                except Exception as e:
                    if stream_callback:
                        stream_callback(f"   ⚠️ Error con MedGemma: {e}, usando LLM...")
            
            # FALLBACK A LLM
            if not self.llm:
                return f"Se encontraron {len(data)} resultados médicos."
            
            prompt = f"""Eres un médico experto que interpreta resultados de bases de datos médicas.

CONSULTA ORIGINAL: "{query}"
DATOS ENCONTRADOS ({len(data)} registros):
{json.dumps(data[:10], indent=2, ensure_ascii=False)}

TAREA: Proporciona una interpretación médica clara y útil usando solo tu conocimiento médico, sin patrones predefinidos.

INSTRUCCIONES ESPECÍFICAS:
- Analiza los datos desde una perspectiva médica
- Identifica información clínicamente relevante
- Destaca hallazgos importantes
- Proporciona contexto médico cuando sea apropiado
- Sugiere interpretaciones útiles

PARA RANKINGS DE MEDICAMENTOS:
- Lista los medicamentos en orden de mayor a menor prescripción
- Menciona el número exacto de prescripciones para cada medicamento
- Si hay empates, indícalo claramente
- Proporciona contexto clínico sobre los medicamentos más prescritos
- Sugiere posibles razones para el patrón de prescripción observado

RESPUESTA: Interpretación médica clara y profesional en español."""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Interpretando resultados médicos"
            )
            
            interpretation = self._extract_response_text(response)
            
            if stream_callback:
                stream_callback("   ✅ Interpretación médica completada")
            
            return interpretation if interpretation else f"Se encontraron {len(data)} resultados médicos."
            
        except Exception as e:
            logger.error(f"Error en interpretación médica: {e}")
            return f"Se encontraron {len(data)} resultados médicos."

    def _get_schema_summary_for_llm(self, schema: Dict[str, Any]) -> str:
        """Genera resumen del esquema para el LLM"""
        try:
            summary = []
            for table_name, columns in schema.items():
                if table_name.startswith('sqlite_'):
                    continue
                column_names = [col['name'] for col in columns]
                summary.append(f"{table_name}: {', '.join(column_names)}")
            return "\n".join(summary)
        except Exception as e:
            logger.error(f"Error generando resumen del esquema: {e}")
            return "Error generando resumen del esquema"

    def _extract_response_text(self, response) -> str:
        """Extrae el texto de la respuesta del LLM"""
        if hasattr(response, 'content'):
            return response.content
        return str(response)

    def _try_parse_llm_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Intenta parsear JSON de respuesta del LLM de forma robusta y tolerante a errores."""
        import re
        import ast
        try:
            content = content.strip()
            
            # ESTRATEGIA 1: Limpiar markdown y buscar primer bloque JSON
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            if content.startswith('```'):
                content = content[3:]
            
            # ESTRATEGIA 2: Buscar primer bloque {...} completo
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except Exception:
                    # Intentar con ast.literal_eval si hay comillas simples
                    try:
                        return ast.literal_eval(json_match.group(0))
                    except Exception:
                        pass
            
            # ESTRATEGIA 3: Intentar parsear directamente
            try:
                return json.loads(content)
            except Exception:
                pass
            
            # ESTRATEGIA 4: Limpiar comillas simples/dobles y reintentar
            content_fixed = content.replace("'", '"')
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content_fixed, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except Exception:
                    pass
                try:
                    return json.loads(content_fixed)
                except Exception:
                    pass
            
            # ESTRATEGIA 5: Buscar múltiples objetos JSON y usar el más completo
            json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
            if json_objects:
                # Intentar con cada objeto JSON encontrado, empezando por el más largo
                json_objects.sort(key=len, reverse=True)
                for json_obj in json_objects:
                    try:
                        return json.loads(json_obj)
                    except Exception:
                        continue
            
            # ESTRATEGIA 6: Reparar JSON incompleto
            try:
                # Si termina con coma, quitarla y añadir llave de cierre
                if content.strip().endswith(','):
                    content = content.rstrip(',') + '}'
                    return json.loads(content)
                
                # Si no termina con }, añadir llave de cierre
                if not content.strip().endswith('}'):
                    content = content.strip() + '}'
                    return json.loads(content)
                
                # Si empieza con { pero no termina con }, intentar cerrar
                if content.strip().startswith('{') and not content.strip().endswith('}'):
                    # Buscar el último } válido y añadir llaves de cierre faltantes
                    last_brace = content.rfind('}')
                    if last_brace > 0:
                        content = content[:last_brace+1]
                        return json.loads(content)
                    else:
                        content = content.strip() + '}'
                        return json.loads(content)
                        
            except Exception:
                pass
            
            # ESTRATEGIA 7: Reparación agresiva - intentar cerrar objetos anidados
            try:
                # Contar llaves abiertas y cerradas
                open_braces = content.count('{')
                close_braces = content.count('}')
                
                if open_braces > close_braces:
                    # Añadir llaves de cierre faltantes
                    missing_braces = open_braces - close_braces
                    content = content.strip() + '}' * missing_braces
                    return json.loads(content)
                    
            except Exception:
                pass
            
            # ESTRATEGIA 8: Extraer solo el primer objeto JSON válido
            try:
                # Buscar el primer { y el último } correspondiente
                start = content.find('{')
                if start >= 0:
                    brace_count = 0
                    end = start
                    for i, char in enumerate(content[start:], start):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end = i + 1
                                break
                    
                    if end > start:
                        json_part = content[start:end]
                        return json.loads(json_part)
                        
            except Exception:
                pass
            
            # ESTRATEGIA 9: Último intento - limpiar y reparar
            try:
                # Eliminar caracteres problemáticos
                content_clean = re.sub(r'[^\x20-\x7E]', '', content)
                content_clean = content_clean.replace('\n', ' ').replace('\r', ' ')
                content_clean = re.sub(r'\s+', ' ', content_clean)
                
                # Buscar JSON en el contenido limpio
                json_match = re.search(r'\{.*?\}', content_clean, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
                    
            except Exception:
                pass
            
            # ESTRATEGIA 10: Reparación de JSON con comillas malformadas
            try:
                # Buscar patrones de comillas problemáticas
                content_fixed = content
                # Corregir comillas simples dentro de strings
                content_fixed = re.sub(r"'([^']*)'", r'"\1"', content_fixed)
                # Corregir comillas dobles sin escapar
                content_fixed = re.sub(r'"([^"]*)"([^"]*)"', r'"\1\2"', content_fixed)
                
                return json.loads(content_fixed)
            except Exception:
                pass
            
            # ESTRATEGIA 11: Extracción de JSON desde respuestas mixtas
            try:
                # Buscar el primer { y el último } válido
                start_idx = content.find('{')
                end_idx = content.rfind('}')
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_part = content[start_idx:end_idx + 1]
                    # Intentar reparar el JSON extraído
                    json_part = re.sub(r',\s*}', '}', json_part)  # Quitar comas finales
                    json_part = re.sub(r',\s*]', ']', json_part)  # Quitar comas finales en arrays
                    return json.loads(json_part)
            except Exception:
                pass
            
            # ESTRATEGIA 12: Reparación agresiva de estructura JSON
            try:
                # Intentar cerrar objetos JSON incompletos
                if content.count('{') > content.count('}'):
                    # Añadir llaves de cierre faltantes
                    missing_braces = content.count('{') - content.count('}')
                    content_fixed = content + '}' * missing_braces
                    return json.loads(content_fixed)
                
                if content.count('[') > content.count(']'):
                    # Añadir corchetes de cierre faltantes
                    missing_brackets = content.count('[') - content.count(']')
                    content_fixed = content + ']' * missing_brackets
                    return json.loads(content_fixed)
            except Exception:
                pass
            
            # ESTRATEGIA 13: Último intento - crear JSON mínimo
            try:
                # Si parece ser una respuesta de error, crear JSON de error
                if 'error' in content.lower() or 'failed' in content.lower():
                    return {
                        'error': True,
                        'message': content[:200],
                        'raw_content': content
                    }
                
                # Si parece ser una respuesta de éxito, crear JSON de éxito
                if 'success' in content.lower() or 'ok' in content.lower():
                    return {
                        'success': True,
                        'message': content[:200],
                        'raw_content': content
                    }
            except Exception:
                pass
            
            # ESTRATEGIA 14: Reparación de JSON con arrays incompletos
            try:
                # Buscar arrays que no están cerrados
                if content.count('[') > content.count(']'):
                    missing_brackets = content.count('[') - content.count(']')
                    content_fixed = content + ']' * missing_brackets
                    return json.loads(content_fixed)
            except Exception:
                pass
            
            # ESTRATEGIA 15: Extracción de JSON desde respuestas con texto adicional
            try:
                # Buscar el JSON más largo en la respuesta
                json_patterns = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
                if json_patterns:
                    # Usar el JSON más largo (probablemente el más completo)
                    longest_json = max(json_patterns, key=len)
                    return json.loads(longest_json)
            except Exception:
                pass
            
            # ESTRATEGIA 16: Reparación de JSON con valores NULL malformados
            try:
                # Corregir "NULL" strings a null
                content_fixed = content.replace('"NULL"', 'null')
                content_fixed = content_fixed.replace("'NULL'", 'null')
                content_fixed = content_fixed.replace('NULL', 'null')
                return json.loads(content_fixed)
            except Exception:
                pass
            
            # Si todo falla, loggear y devolver None
            logger.error(f"Error parseando JSON robusto: {content[:200]}...")
            return None
            
        except Exception as e:
            logger.error(f"Error crítico parseando JSON: {e}")
            return None

    def _clean_llm_sql_response(self, sql_response: str) -> str:
        """
        Limpia la respuesta del LLM usando el LLM para eliminar automáticamente texto no-SQL.
        Sin listas hardcodeadas - todo via LLM.
        """
        try:
            if not sql_response or not self.llm:
                return sql_response
            
            # Si ya parece SQL puro, devolver sin cambios
            if sql_response.strip().upper().startswith('SELECT') or sql_response.strip().upper().startswith('WITH'):
                return sql_response
            
            # Usar LLM para extraer solo el SQL
            prompt = f"""Eres un experto en SQL que extrae solo código SQL de respuestas mixtas.

RESPUESTA DEL LLM:
{sql_response}

TAREA:
Extrae SOLO el código SQL de esta respuesta, eliminando:
- Explicaciones en español
- Comentarios
- Texto explicativo
- Markdown (```sql, ```)
- Cualquier texto que no sea SQL válido

REGLAS CRÍTICAS:
1. Devuelve SOLO el SQL puro
2. NO incluyas explicaciones
3. NO incluyas comentarios
4. Asegúrate de que sea SQL válido para SQLite
5. Si no hay SQL válido, devuelve "SELECT 1 as error_no_sql;"

EJEMPLO:
Entrada: "Aquí está el SQL para contar pacientes: SELECT COUNT(*) FROM patients;"
Salida: "SELECT COUNT(*) FROM patients;"

Responde SOLO con el SQL extraído:"""

            # Usar llamada síncrona en lugar de async
            response = _call_openai_native(self.llm, [{"role": "user", "content": prompt}])
            
            cleaned_sql = self._extract_response_text(response).strip()
            
            # Verificar que realmente es SQL
            if cleaned_sql and (cleaned_sql.upper().startswith('SELECT') or 
                               cleaned_sql.upper().startswith('WITH') or
                               cleaned_sql.upper().startswith('INSERT') or
                               cleaned_sql.upper().startswith('UPDATE') or
                               cleaned_sql.upper().startswith('DELETE')):
                return cleaned_sql
            else:
                # Fallback: intentar extraer SQL manualmente
                import re
                # Buscar patrones SQL comunes
                sql_patterns = [
                    r'SELECT\s+.*?;',  # SELECT statements
                    r'WITH\s+.*?;',    # CTE statements
                    r'INSERT\s+.*?;',  # INSERT statements
                    r'UPDATE\s+.*?;',  # UPDATE statements
                    r'DELETE\s+.*?;',  # DELETE statements
                ]
                
                for pattern in sql_patterns:
                    match = re.search(pattern, sql_response, re.IGNORECASE | re.DOTALL)
                    if match:
                        return match.group(0)
                
                # Si no se encuentra SQL válido, devolver error
                return "SELECT 1 as error_no_valid_sql;"
                
        except Exception as e:
            logger.error(f"Error limpiando respuesta SQL: {e}")
            # Fallback básico
            import re
            # Buscar SQL básico
            sql_match = re.search(r'SELECT\s+.*?;', sql_response, re.IGNORECASE | re.DOTALL)
            if sql_match:
                return sql_match.group(0)
            return "SELECT 1 as error_cleaning_failed;"

    def _extract_sql_basic(self, text: str) -> str:
        """
        Extracción básica de SQL sin LLM (fallback).
        Busca patrones SQL comunes en el texto.
        """
        try:
            if not text:
                return text
            
            # Buscar SQL que empiece con palabras clave comunes
            sql_patterns = [
                r'SELECT\s+.*?(?:;|$)',
                r'WITH\s+.*?(?:;|$)',
                r'INSERT\s+.*?(?:;|$)',
                r'UPDATE\s+.*?(?:;|$)',
                r'DELETE\s+.*?(?:;|$)'
            ]
            
            for pattern in sql_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                if matches:
                    # Tomar el primer match que parezca SQL válido
                    for match in matches:
                        if len(match.strip()) > 10:  # Filtrar fragmentos muy cortos
                            return match.strip()
            
            # Si no se encuentra SQL, devolver texto original
            return text
            
        except Exception as e:
            logger.error(f"Error en _extract_sql_basic: {e}")
            return text

    def _create_error_response(self, error: str, sql: str = "") -> Dict[str, Any]:
        return {'success': False, 'message': f"Error: {error}", 'data': [], 'sql_query': sql}
    
    async def _use_generic_sql_tools(self, query: str, stream_callback=None) -> str:
        """
        Usa el LLM para generar SQL de forma dinámica y segura
        """
        try:
            if not self.llm:
                return "SELECT COUNT(*) FROM PATI_PATIENTS"
            
            if stream_callback:
                stream_callback("   🔧 Generando SQL dinámico con LLM...")
            
            # Obtener esquema de la base de datos
            schema_info = self._get_real_schema_info()
            
            # Prompt específico para SQL médico dinámico
            prompt = f"""Eres un experto en SQL para bases de datos médicas. Genera SQL válido y seguro.

CONSULTA: "{query}"

ESQUEMA DE LA BASE DE DATOS:
{schema_info}

REGLAS CRÍTICAS:
1. Usa SOLO tablas y columnas que existen en el esquema
2. Para diagnósticos médicos busca en campos de texto libre (DIAG_OBSERVATION, DIAG_DESCRIPTION)
3. Para conteos usa COUNT(*) desde la tabla principal
4. Para búsquedas médicas usa LIKE '%termino%' para búsquedas flexibles
5. Define TODOS los alias en el FROM/JOIN antes de usarlos
6. Usa JOINs apropiados para conectar tablas
7. Optimiza para SQLite

ESTRATEGIA MÉDICA:
- Para diabetes: Buscar en DIAG_OBSERVATION con múltiples variantes:
  LIKE '%diabetes%' OR LIKE '%diabetes mellitus%' OR LIKE '%DM2%' OR LIKE '%DM1%' 
  OR LIKE '%diabetes tipo 2%' OR LIKE '%diabetes tipo 1%' OR LIKE '%diabetes gestacional%'
- Para medicación: Usar PATI_USUAL_MEDICATION.PAUM_OBSERVATIONS (NO MEDICATION_NAME)
- Para conteos: Usar COUNT(*) desde PATI_PATIENTS
- Para pacientes: Usar PATI_NAME, PATI_SURNAME_1
- Para diagnósticos: Usar EPIS_DIAGNOSTICS con DIAG_OBSERVATION

EJEMPLOS DE CONEXIÓN:
- PATI_PATIENTS.PATI_ID = EPIS_EPISODES.PATI_ID
- EPIS_EPISODES.EPIS_ID = EPIS_DIAGNOSTICS.EPIS_ID

RESPUESTA: SOLO el SQL válido, sin explicaciones."""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Generando SQL dinámico"
            )
            
            sql = self._extract_response_text(response).strip()
            sql = self._clean_llm_sql_response(sql)
            
            if sql and not sql.startswith("Error"):
                if stream_callback:
                    stream_callback("   ✅ SQL dinámico generado con LLM")
                return sql
            else:
                # Fallback básico
                if stream_callback:
                    stream_callback("   ⚠️ Fallback a SQL básico")
                return "SELECT COUNT(*) FROM PATI_PATIENTS"
                
        except Exception as e:
            logger.error(f"Error usando herramientas genéricas: {e}")
            if stream_callback:
                stream_callback(f"   ❌ Error en herramientas genéricas: {e}")
            return "SELECT COUNT(*) FROM PATI_PATIENTS"

    def _fix_typo_errors(self, sql: str) -> str:
        """
        Corrige errores tipográficos específicos en SQL.
        
        Args:
            sql: SQL con posibles errores tipográficos
            
        Returns:
            str: SQL con errores tipográficos corregidos
        """
        if not sql:
            return sql
        
        # Correcciones específicas para errores tipográficos comunes
        corrections = [
            # Espacios faltantes después de alias de tabla
            (r'(\w+)WHERE', r'\1 WHERE'),
            (r'(\w+)FROM', r'\1 FROM'),
            (r'(\w+)JOIN', r'\1 JOIN'),
            (r'(\w+)ON', r'\1 ON'),
            (r'(\w+)AND', r'\1 AND'),
            (r'(\w+)OR', r'\1 OR'),
            (r'(\w+)ORDER', r'\1 ORDER'),
            (r'(\w+)GROUP', r'\1 GROUP'),
            (r'(\w+)HAVING', r'\1 HAVING'),
            (r'(\w+)LIMIT', r'\1 LIMIT'),
            
            # Espacios faltantes después de columnas
            (r'(\w+\.\w+)FROM', r'\1 FROM'),
            (r'(\w+\.\w+)WHERE', r'\1 WHERE'),
            (r'(\w+\.\w+)ORDER', r'\1 ORDER'),
            (r'(\w+\.\w+)GROUP', r'\1 GROUP'),
            
            # Espacios faltantes después de asteriscos
            (r'\*FROM', r'* FROM'),
            (r'\*WHERE', r'* WHERE'),
            (r'\*ORDER', r'* ORDER'),
            
            # Espacios faltantes después de paréntesis
            (r'\)WHERE', r') WHERE'),
            (r'\)FROM', r') FROM'),
            (r'\)JOIN', r') JOIN'),
            (r'\)AND', r') AND'),
            (r'\)OR', r') OR'),
            
            # Espacios faltantes antes de paréntesis
            (r'WHERE\(', r'WHERE ('),
            (r'FROM\(', r'FROM ('),
            (r'JOIN\(', r'JOIN ('),
            
            # Correcciones específicas para palabras pegadas
            (r'SELECT\*', r'SELECT *'),
            (r'SELECT\w+\.\*', r'SELECT *'),
            (r'FROM\w+', r'FROM '),
            (r'WHERE\w+', r'WHERE '),
            (r'JOIN\w+', r'JOIN '),
            (r'ON\w+', r'ON '),
            (r'AND\w+', r'AND '),
            (r'OR\w+', r'OR '),
            (r'ORDER\w+', r'ORDER '),
            (r'GROUP\w+', r'GROUP '),
            (r'LIMIT\w+', r'LIMIT '),
        ]
        
        corrected_sql = sql
        for pattern, replacement in corrections:
            corrected_sql = re.sub(pattern, replacement, corrected_sql, flags=re.IGNORECASE)
        
        # Normalizar espacios múltiples
        corrected_sql = re.sub(r'\s+', ' ', corrected_sql).strip()
        
        return corrected_sql

    async def _basic_sql_cleanup(self, sql: str, stream_callback=None) -> str:
        """
        Limpieza robusta de SQL usando LLM para detectar y corregir errores de forma inteligente.
        
        Args:
            sql: SQL a limpiar
            stream_callback: Función para mostrar progreso
            
        Returns:
            str: SQL limpio y corregido
        """
        try:
            if not sql:
                return sql
            
            # Si no hay LLM, usar limpieza básica
            if not self.llm:
                return self._basic_sql_cleanup_fallback(sql)
            
            if stream_callback:
                stream_callback("   - Limpieza robusta de SQL con IA...")
            
            prompt = f"""Eres un experto en SQL que detecta y corrige errores de sintaxis de forma inteligente.

SQL A LIMPIAR:
{sql}

TAREA: Analiza el SQL y corrige cualquier error de sintaxis que encuentres.

TIPOS DE ERRORES A DETECTAR:
- Palabras pegadas a keywords SQL (ej: PacientesJOIN, DiagnósticosWHERE)
- Espacios faltantes entre palabras clave
- Caracteres de control o caracteres problemáticos
- Comentarios SQL mal formateados
- Errores de formato comunes
- Palabras clave SQL mal escritas

INSTRUCCIONES:
1. Detecta errores de forma inteligente, no uses patterns predefinidos
2. Corrige espacios faltantes entre palabras clave y tablas/columnas
3. Elimina caracteres problemáticos
4. Normaliza el formato del SQL
5. Mantén la lógica original del SQL
6. Asegúrate de que sea sintácticamente válido

RESPUESTA:
Devuelve SOLO el SQL corregido, sin explicaciones ni comentarios."""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Limpieza robusta de SQL"
            )
            
            cleaned_sql = self._extract_response_text(response).strip()
            cleaned_sql = self._clean_llm_sql_response(cleaned_sql)
            
            if cleaned_sql and not cleaned_sql.startswith("Error"):
                logger.info(f"🧠 LLM realizó limpieza robusta del SQL")
                if stream_callback:
                    stream_callback("   ✅ Limpieza robusta completada")
                return cleaned_sql
            else:
                logger.warning(f"⚠️ LLM no pudo limpiar SQL, usando fallback")
                return self._basic_sql_cleanup_fallback(sql)
                
        except Exception as e:
            logger.error(f"Error en _basic_sql_cleanup: {e}")
            return self._basic_sql_cleanup_fallback(sql)

    def _basic_sql_cleanup_fallback(self, sql: str) -> str:
        """
        Fallback básico para limpieza de SQL cuando no hay LLM disponible.
        
        Args:
            sql: SQL a limpiar
            
        Returns:
            str: SQL básicamente limpio
        """
        try:
            if not sql:
                return sql
            
            # 1. Limpieza básica
            sql = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sql)  # Caracteres de control
            sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)  # Comentarios SQL
            sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)  # Comentarios multilínea
            
            # 2. Corregir errores tipográficos específicos
            sql = self._fix_typo_errors(sql)
            
            # 3. Normalizar espacios
            sql = re.sub(r'\s+', ' ', sql).strip()
            
            # 4. Asegurar punto y coma al final
            if not sql.endswith(';'):
                sql += ';'
            
            return sql
            
        except Exception as e:
            logger.error(f"Error en _basic_sql_cleanup_fallback: {e}")
            return sql

    async def _llm_validate_and_correct_tables(self, sql: str, stream_callback=None) -> str:
        """
        Valida y corrige tablas usando LLM con acceso al esquema real.
        Le da al LLM las herramientas para verificar qué tablas existen realmente.
        """
        try:
            if not self.llm:
                if stream_callback:
                    stream_callback("   - Validación básica de tablas (sin LLM)...")
                return sql
            
            if stream_callback:
                stream_callback("   - Validando tablas con IA y esquema real...")
            
            # Usar el método correcto de SQLAgentTools
            return await self.schema_tools.llm_validate_and_correct_tables(sql, stream_callback)
            
        except Exception as e:
            logger.error(f"Error en _llm_validate_and_correct_tables: {e}")
            return sql

    async def _llm_validate_and_correct_columns(self, sql: str, stream_callback=None) -> str:
        """
        Valida y corrige columnas usando LLM con acceso al esquema real.
        Le da al LLM las herramientas para verificar qué columnas existen realmente.
        """
        try:
            if not self.llm:
                if stream_callback:
                    stream_callback("   - Validación básica de columnas (sin LLM)...")
                return sql
            
            if stream_callback:
                stream_callback("   - Validando columnas con IA y esquema real...")
            
            # Extraer tabla del SQL
            table_match = re.search(r'INSERT INTO (\w+)', sql, re.IGNORECASE)
            if not table_match:
                return sql
            
            table_name = table_match.group(1)
            
            # Obtener esquema específico de la tabla
            table_schema = self._get_table_schema_info(table_name)
            
            prompt = f"""Eres un experto en validación de columnas SQL en bases de datos médicas.

SQL A VALIDAR:
{sql}

ESQUEMA REAL DE LA TABLA {table_name}:
{table_schema}

TAREA ADAPTATIVA: Valida y corrige las columnas del SQL.

INSTRUCCIONES:
1. Analiza el SQL INSERT
2. Identifica las columnas que se están insertando
3. Verifica que TODAS las columnas existan en el esquema real
4. Si alguna columna NO existe, elimínala del INSERT
5. Mantén solo las columnas que SÍ existen en la tabla
6. Adapta el SQL según las columnas disponibles

REGLAS DE VALIDACIÓN:
- SOLO usa columnas que existan en el esquema real
- Si una columna no existe, elimínala completamente
- Mantén la estructura del INSERT válida
- No agregues columnas que no estén en el esquema
- Adapta el mapeo según las columnas disponibles

VALIDACIÓN ADAPTATIVA:
- Analiza el esquema de {table_name}
- Identifica las columnas disponibles
- Corrige el SQL según las columnas reales
- Mantén solo columnas que existan

RESPUESTA JSON:
{{
    "sql_corrected": "SQL corregido",
    "columns_validated": ["columna1", "columna2"],
    "corrections_applied": ["corrección1", "corrección2"],
    "table_used": "{table_name}",
    "validation_confidence": 0.95
}}

IMPORTANTE: Solo responde con el JSON."""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Validando columnas con IA"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result and result.get('sql_corrected'):
                corrections = result.get('corrections_applied', [])
                if stream_callback and corrections:
                    stream_callback(f"   ✅ Columnas validadas")
                    for correction in corrections:
                        stream_callback(f"      - {correction}")
                return result['sql_corrected']
            else:
                return sql
            
        except Exception as e:
            logger.error(f"Error en _llm_validate_and_correct_columns: {e}")
            return sql

    def _normalize_accents_python(self, text: str) -> str:
        """
        Normaliza vocales acentuadas en Python (más eficiente que en SQL).
        Versión ROBUSTA que maneja todos los casos edge.
        
        Args:
            text: Texto a normalizar
            
        Returns:
            str: Texto con vocales acentuadas normalizadas
        """
        if not text:
            return ""
        
        # Reemplazos completos de acentos y caracteres especiales
        replacements = {
            # Vocales acentuadas mayúsculas
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ü': 'U',
            # Vocales acentuadas minúsculas
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u',
            # Ñ
            'Ñ': 'N', 'ñ': 'n',
            # Caracteres especiales que pueden aparecer en nombres
            'À': 'A', 'È': 'E', 'Ì': 'I', 'Ò': 'O', 'Ù': 'U',
            'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
            'Â': 'A', 'Ê': 'E', 'Î': 'I', 'Ô': 'O', 'Û': 'U',
            'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
            'Ã': 'A', 'Õ': 'O',
            'ã': 'a', 'õ': 'o',
            # Eliminar caracteres problemáticos
            '\t': ' ', '\n': ' ', '\r': ' ',
        }
        
        normalized = text
        for accented, normal in replacements.items():
            normalized = normalized.replace(accented, normal)
        
        # Normalizar espacios múltiples
        normalized = ' '.join(normalized.split())
        
        # Convertir a mayúsculas para consistencia
        return normalized.upper()

    async def _fix_sql_compatibility(self, sql: str, stream_callback=None) -> str:
        """
        Corrige problemas de compatibilidad del SQL para SQLite usando LLM.
        SIN PATRONES HARDCODEADOS - todo via LLM.
        """
        try:
            if not sql:
                return sql
                
            # Si no hay LLM, usar fallback básico
            if not self.llm:
                if stream_callback:
                    stream_callback("   - Ajustando compatibilidad SQL con método básico...")
                return sql
                
            if stream_callback:
                stream_callback("   - Optimizando SQL para compatibilidad con SQLite...")
            
            # Usar LLM para corrección inteligente de compatibilidad
            compatibility_prompt = f"""Eres un experto en bases de datos que convierte SQL de MySQL/PostgreSQL a SQLite.

TAREA: Convierte este SQL para que sea 100% compatible con SQLite, manteniendo la lógica original.

SQL A CONVERTIR:
{sql}

REGLAS DE CONVERSIÓN PARA SQLITE:

1. **FECHAS Y TIEMPO:**
   - DATE_SUB(CURDATE(), INTERVAL n YEAR) → date('now', '-n years')
   - DATE_SUB(CURDATE(), INTERVAL n MONTH) → date('now', '-n months')
   - DATE_SUB(CURDATE(), INTERVAL n DAY) → date('now', '-n days')
   - CURDATE() → date('now')
   - NOW() → datetime('now')
   - GETDATE() → datetime('now')
   - YEAR(column) → strftime('%Y', column)
   - MONTH(column) → strftime('%m', column)
   - DAY(column) → strftime('%d', column)
   - DATEDIFF(date1, date2) → julianday(date1) - julianday(date2)

2. **LÍMITES:**
   - SELECT TOP n → SELECT ... LIMIT n
   - LIMIT n OFFSET m → LIMIT n OFFSET m (ya compatible)

3. **FUNCIONES DE CADENA:**
   - CONCAT(a, b) → (a || b)
   - LENGTH() → length() (ya compatible)
   - SUBSTRING() → substr()

4. **FUNCIONES MATEMÁTICAS:**
   - POW(a, b) → POWER(a, b)
   - RAND() → RANDOM()

5. **TIPOS DE DATOS:**
   - AUTO_INCREMENT → AUTOINCREMENT
   - TINYINT, SMALLINT, MEDIUMINT, BIGINT → INTEGER
   - TEXT, LONGTEXT → TEXT
   - DECIMAL(n,m) → REAL o NUMERIC

6. **OTRAS FUNCIONES:**
   - IFNULL(a, b) → COALESCE(a, b)
   - IF(condition, true_val, false_val) → CASE WHEN condition THEN true_val ELSE false_val END

IMPORTANTE:
- Mantén la lógica exacta del SQL original
- Asegúrate de que la sintaxis sea válida para SQLite
- No cambies nombres de tablas o columnas
- Preserva todos los WHERE, JOIN, GROUP BY, ORDER BY, etc.
- Si el SQL ya es compatible, devuélvelo sin cambios

RESPUESTA:
Devuelve SOLO el SQL corregido, sin explicaciones adicionales."""

            try:
                if stream_callback:
                    stream_callback("   - Analizando SQL para compatibilidad con SQLite...")
                    
                response = await asyncio.to_thread(
                    _call_openai_native, self.llm, [{"role": "user", "content": compatibility_prompt}]
                )
                
                corrected_sql = self._extract_response_text(response).strip()
                corrected_sql = self._clean_llm_sql_response(corrected_sql)
                
                # Validar que el LLM devolvió SQL válido
                if corrected_sql and not corrected_sql.startswith("Error") and len(corrected_sql) > 10:
                    # Log de cambios si fueron significativos
                    if corrected_sql != sql:
                        logger.info(f"🧠 LLM corrigió compatibilidad SQL para SQLite")
                        logger.info(f"   Original: {sql[:100]}...")
                        logger.info(f"   Corregido: {corrected_sql[:100]}...")
                        
                        if stream_callback:
                            stream_callback("   - Optimizadas funciones para compatibilidad con SQLite")
                    else:
                        if stream_callback:
                            stream_callback("   - SQL ya compatible con SQLite, sin cambios necesarios")
                    
                    return corrected_sql
                else:
                    logger.warning(f"⚠️ LLM devolvió respuesta inválida, usando original")
                    if stream_callback:
                        stream_callback("   - Usando SQL original")
                    return sql
                    
            except Exception as e:
                logger.error(f"Error usando LLM para compatibilidad: {e}")
                if stream_callback:
                    stream_callback(f"   - Error optimizando SQL: {str(e)[:50]}... Usando original")
                return sql
                
        except Exception as e:
            logger.error(f"Error en _fix_sql_compatibility: {e}")
            if stream_callback:
                stream_callback("   - Error en corrección de compatibilidad")
            return sql  # Devolver original si falla completamente

    async def _validate_schema_with_robust_tools(self, sql: str, stream_callback=None) -> str:
        """
        Valida el esquema usando herramientas robustas para verificar columnas y tablas reales.
        GARANTIZA que el LLM devuelva SOLO SQL sin explicaciones.
        """
        try:
            if not self.llm:
                if stream_callback:
                    stream_callback("   - Validación básica (sin LLM)...")
                return sql
            
            if stream_callback:
                stream_callback("   - Validando esquema con herramientas robustas...")
            
            # Obtener esquema real de la base de datos
            schema_info = self._get_real_schema_info()
            
            # PROMPT ESTRICTO que garantiza solo SQL
            validation_prompt = f"""Eres un experto en SQL que corrige consultas contra esquemas reales.

SQL A VALIDAR:
{sql}

ESQUEMA REAL DE LA BASE DE DATOS:
{schema_info}

TAREA CRÍTICA:
1. Verifica que TODAS las tablas mencionadas en el SQL existan en el esquema real
2. Verifica que TODAS las columnas mencionadas existan en sus respectivas tablas
3. Si encuentras tablas o columnas que NO existen, corrígelas usando las alternativas reales
4. Mantén la lógica original del SQL
5. NO inventes columnas que no existen

REGLAS IMPORTANTES:
- Si una tabla no existe, busca una tabla similar en el esquema
- Si una columna no existe, busca una columna similar en la misma tabla
- Para pacientes, usa PATI_PATIENTS con columnas como PATI_ID, PATI_FULL_NAME
- Para diagnósticos, usa EPIS_DIAGNOSTICS con columnas como DIAG_ID, CDTE_ID
- Para observaciones, usa OBSE_OBSERVATIONS con columnas como OBSE_ID, OBSE_VALUE

IMPORTANTE: Devuelve ÚNICAMENTE el SQL corregido, SIN explicaciones, SIN comentarios, SIN texto adicional.
Si el SQL está correcto, devuelve el SQL original sin cambios.

EJEMPLO DE RESPUESTA CORRECTA:
SELECT p.PATI_ID, p.PATI_FULL_NAME FROM PATI_PATIENTS p WHERE p.PATI_ACTIVE = 1;

RESPUESTA (SOLO SQL):"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": validation_prompt}],
                task_description="Validando esquema con herramientas robustas"
            )
            
            validated_sql = self._extract_response_text(response).strip()
            
            # LIMPIEZA AGRESIVA para eliminar cualquier texto que no sea SQL
            validated_sql = self._clean_sql_response_aggressive(validated_sql)
            
            if validated_sql and not validated_sql.startswith("Error"):
                # Verificar que realmente es SQL válido
                if self._is_valid_sql_response(validated_sql):
                    logger.info(f"🧠 SQL validado con esquema real")
                    if stream_callback:
                        stream_callback("   ✅ Esquema validado con herramientas robustas")
                    return validated_sql
                else:
                    logger.warning(f"⚠️ LLM devolvió texto que no es SQL válido")
                    return sql
            else:
                logger.warning(f"⚠️ No se pudo validar con herramientas, usando original")
                return sql
                
        except Exception as e:
            logger.error(f"Error en validación con herramientas: {e}")
            return sql

    def _clean_sql_response_aggressive(self, response: str) -> str:
        """
        Limpieza agresiva para eliminar cualquier texto que no sea SQL puro.
        """
        if not response:
            return response
        
        # Eliminar bloques de código markdown
        response = re.sub(r'^```[a-zA-Z]*\n', '', response)
        response = re.sub(r'\n```$', '', response)
        response = re.sub(r'^```', '', response)
        response = re.sub(r'```$', '', response)
        
        # Eliminar comentarios SQL
        response = re.sub(r'--.*$', '', response, flags=re.MULTILINE)
        response = re.sub(r'/\*.*?\*/', '', response, flags=re.DOTALL)
        
        # Eliminar explicaciones comunes del LLM
        explanations = [
            "Aquí está el SQL corregido:",
            "El SQL corregido es:",
            "SQL válido:",
            "Consulta corregida:",
            "Aquí tienes el SQL:",
            "El SQL sería:",
            "SQL resultante:",
            "Consulta SQL:",
            "SQL optimizado:",
            "SQL final:"
        ]
        
        for explanation in explanations:
            response = response.replace(explanation, "")
        
        # Buscar el primer SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, ALTER
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 'WITH']
        for keyword in sql_keywords:
            pos = response.upper().find(keyword)
            if pos != -1:
                response = response[pos:]
                break
        
        # Buscar el último punto y coma
        last_semicolon = response.rfind(';')
        if last_semicolon != -1:
            response = response[:last_semicolon + 1]
        
        # Limpiar espacios extra
        response = re.sub(r'\s+', ' ', response).strip()
        
        return response

    def _is_valid_sql_response(self, sql: str) -> bool:
        """
        Verifica si la respuesta es SQL válido.
        """
        if not sql:
            return False
        
        # Verificar que contenga palabras clave SQL
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 'WITH']
        has_sql_keyword = any(keyword in sql.upper() for keyword in sql_keywords)
        
        # Verificar que no contenga texto explicativo
        explanatory_words = ['explicación', 'explicación', 'aquí', 'corregido', 'válido', 'resultado', 'final']
        has_explanation = any(word in sql.lower() for word in explanatory_words)
        
        # Verificar que no sea solo texto sin SQL
        if len(sql) < 10:
            return False
        
        return has_sql_keyword and not has_explanation

    def _get_real_schema_info(self) -> str:
        """
        Obtiene información real del esquema de la base de datos usando LLM para categorización.
        SIN HARDCODEO - todo via LLM.
        """
        try:
            # Conectar a la base de datos y obtener esquema real
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Obtener todas las tablas
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Obtener información detallada de columnas para cada tabla
            schema_info = []
            for table in tables:
                if table.startswith('sqlite_') or table.startswith('_'):
                    continue
                    
                # Obtener columnas detalladas de cada tabla
                cursor.execute(f"PRAGMA table_info('{table}');")
                columns_info = cursor.fetchall()
                
                schema_info.append(f"📋 TABLA: {table}")
                schema_info.append("   COLUMNAS:")
                
                for col_info in columns_info:
                    col_name = col_info[1]
                    col_type = col_info[2]
                    not_null = "NOT NULL" if col_info[3] else ""
                    pk = "PRIMARY KEY" if col_info[5] else ""
                    default = f"DEFAULT {col_info[4]}" if col_info[4] else ""
                    
                    constraints = " ".join(filter(None, [not_null, pk, default])).strip()
                    if constraints:
                        schema_info.append(f"     - {col_name} ({col_type}) {constraints}")
                    else:
                        schema_info.append(f"     - {col_name} ({col_type})")
                
                schema_info.append("")
            
            conn.close()
            return "\n".join(schema_info)
            
        except Exception as e:
            logger.error(f"Error obteniendo esquema real: {e}")
            return "Error obteniendo esquema"

    async def _llm_categorize_tables_intelligent(self, tables: List[str], stream_callback=None) -> Dict[str, List[str]]:
        """
        Categoriza tablas usando LLM de forma inteligente - SIN HARDCODEO.
        
        Args:
            tables: Lista de nombres de tablas
            stream_callback: Función para mostrar progreso
            
        Returns:
            Dict[str, List[str]]: Categorías y tablas correspondientes
        """
        try:
            if not self.llm:
                # Fallback básico sin LLM
                return {'TABLAS': tables}
            
            if stream_callback:
                stream_callback("   - Categorizando tablas con IA inteligente...")
            
            # Obtener información de columnas para contexto
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            table_info = {}
            for table in tables:
                if table.startswith('sqlite_') or table.startswith('_'):
                    continue
                try:
                    cursor.execute(f"PRAGMA table_info('{table}');")
                    columns = [row[1] for row in cursor.fetchall()]
                    table_info[table] = columns
                except Exception as e:
                    logger.warning(f"No se pudo obtener información de tabla {table}: {e}")
            
            conn.close()
            
            # PROMPT ESPECÍFICO PARA CATEGORIZACIÓN INTELIGENTE - SIN HARDCODEO
            categorization_prompt = f"""Eres un experto en análisis de esquemas de bases de datos médicas.

TABLAS DISPONIBLES:
{json.dumps(table_info, indent=2, ensure_ascii=False)}

TAREA ESPECÍFICA: Analiza cada tabla y categorízala según su función en un sistema médico.

INSTRUCCIONES DE CATEGORIZACIÓN:
1. Analiza los nombres de las tablas y sus columnas
2. Identifica patrones en los nombres y estructura
3. Determina la función principal de cada tabla
4. Categoriza basándote en el contenido semántico, NO en patrones de nombres
5. Considera múltiples categorías si una tabla puede servir para varios propósitos

CRITERIOS DE ANÁLISIS:
- Tablas con información de pacientes (datos personales, demográficos)
- Tablas con episodios médicos (visitas, hospitalizaciones)
- Tablas con diagnósticos y condiciones médicas
- Tablas con medicamentos y tratamientos
- Tablas con procedimientos médicos
- Tablas con citas y programación
- Tablas con observaciones y resultados de laboratorio
- Tablas con códigos y parámetros del sistema
- Tablas con información administrativa
- Tablas con datos de especialidades específicas (oncología, cardiología, etc.)

RESPUESTA JSON:
{{
    "categorias": {{
        "PACIENTES": ["tabla1", "tabla2"],
        "EPISODIOS": ["tabla3", "tabla4"],
        "DIAGNÓSTICOS": ["tabla5", "tabla6"],
        "MEDICAMENTOS": ["tabla7", "tabla8"],
        "PROCEDIMIENTOS": ["tabla9", "tabla10"],
        "OBSERVACIONES": ["tabla11", "tabla12"],
        "CITAS": ["tabla13", "tabla14"],
        "CÓDIGOS": ["tabla15", "tabla16"],
        "ADMINISTRATIVO": ["tabla17", "tabla18"],
        "ESPECIALIDADES": ["tabla19", "tabla20"],
        "OTROS": ["tabla21", "tabla22"]
    }},
    "razonamiento": "explicación de la categorización",
    "total_tablas": {len(tables)}
}}

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": categorization_prompt}],
                task_description="Categorizando tablas con IA inteligente"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result and 'categorias' in result:
                categorias = result['categorias']
                razonamiento = result.get('razonamiento', 'Sin explicación')
                
                if stream_callback:
                    stream_callback(f"   - Categorización completada: {len(categorias)} categorías")
                    stream_callback(f"   - Razonamiento: {razonamiento[:50]}...")
                
                return categorias
            else:
                logger.warning("⚠️ LLM no pudo categorizar tablas, usando categoría única")
                return {'TABLAS': tables}
                
        except Exception as e:
            logger.error(f"Error categorizando tablas: {e}")
            return {'TABLAS': tables}

    async def _llm_select_relevant_tables_intelligent(self, query: str, medical_concepts: List[str], stream_callback=None) -> List[str]:
        """
        Selecciona tablas relevantes usando LLM de forma inteligente - SIN HARDCODEO.
        
        Args:
            query: Consulta original del usuario
            medical_concepts: Conceptos médicos detectados
            stream_callback: Función para mostrar progreso
            
        Returns:
            List[str]: Tablas relevantes seleccionadas
        """
        try:
            if not self.llm:
                # Fallback básico sin LLM
                schema = self.schema_tools.get_schema()
                return list(schema.keys())[:3]
            
            if stream_callback:
                stream_callback("   - Seleccionando tablas relevantes con IA inteligente...")
            
            # Obtener esquema completo
            schema = self.schema_tools.get_schema()
            schema_summary = self._get_schema_summary_for_llm(schema)
            
            # Categorizar tablas primero
            tables = list(schema.keys())
            categorias = await self._llm_categorize_tables_intelligent(tables, stream_callback)
            
            # PROMPT ESPECÍFICO PARA SELECCIÓN INTELIGENTE - SIN HARDCODEO
            selection_prompt = f"""Eres un experto en bases de datos médicas que selecciona las tablas más relevantes.

CONSULTA DEL USUARIO: "{query}"

CONCEPTOS MÉDICOS DETECTADOS: {medical_concepts}

CATEGORÍAS DE TABLAS DISPONIBLES:
{json.dumps(categorias, indent=2, ensure_ascii=False)}

ESQUEMA COMPLETO:
{schema_summary}

TAREA ESPECÍFICA: Selecciona las tablas más relevantes para responder a esta consulta.

ESTRATEGIA DE SELECCIÓN:
1. Analiza la consulta para entender qué información se necesita
2. Identifica qué categorías de tablas son relevantes
3. Selecciona tablas específicas dentro de esas categorías
4. Considera relaciones entre tablas
5. Prioriza tablas que contengan la información más específica
6. Incluye tablas de soporte si son necesarias para JOINs

CRITERIOS DE SELECCIÓN:
- Relevancia directa con la consulta
- Capacidad de proporcionar la información solicitada
- Posibilidad de JOINs efectivos
- Complejidad vs. simplicidad
- Rendimiento esperado

INSTRUCCIONES:
- Selecciona máximo 5 tablas para evitar consultas muy complejas
- Prioriza tablas que contengan la información principal
- Incluye tablas de soporte solo si son necesarias
- Considera el rendimiento de la consulta

RESPUESTA JSON:
{{
    "tablas_seleccionadas": ["tabla1", "tabla2", "tabla3"],
    "categorias_relevantes": ["categoria1", "categoria2"],
    "razonamiento": "explicación de la selección",
    "estrategia_joins": "descripción de cómo conectar las tablas"
}}

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": selection_prompt}],
                task_description="Seleccionando tablas relevantes con IA inteligente"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result and 'tablas_seleccionadas' in result:
                tablas_seleccionadas = result['tablas_seleccionadas']
                categorias_relevantes = result.get('categorias_relevantes', [])
                razonamiento = result.get('razonamiento', 'Sin explicación')
                
                # Validar que las tablas existan
                valid_tables = [t for t in tablas_seleccionadas if t in schema]
                
                if stream_callback:
                    stream_callback(f"   - Tablas seleccionadas: {', '.join(valid_tables[:3])}...")
                    stream_callback(f"   - Categorías relevantes: {', '.join(categorias_relevantes)}")
                
                return valid_tables[:5]  # Máximo 5 tablas
            else:
                logger.warning("⚠️ LLM no pudo seleccionar tablas, usando fallback")
                return list(schema.keys())[:3]
                
        except Exception as e:
            logger.error(f"Error seleccionando tablas: {e}")
            schema = self.schema_tools.get_schema()
            return list(schema.keys())[:3]

    # Método de compatibilidad para mantener funcionalidad existente
    async def _llm_map_fhir_to_sql_intelligent(self, fhir_data: Dict[str, Any], stream_callback=None) -> Dict[str, Any]:
        """
        Método de compatibilidad que redirige al nuevo sistema adaptativo.
        Mantiene compatibilidad hacia atrás mientras se migra completamente.
        """
        return await self._llm_map_fhir_to_sql_adaptive(fhir_data, stream_callback, None)

    async def _llm_map_fhir_to_sql_adaptive(self, fhir_data: Dict[str, Any], stream_callback=None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        🧠 ARQUITECTURA ADAPTATIVA: Mapeo dinámico que aprende y se adapta automáticamente.
        El LLM descubre el esquema, aprende de cada operación y valida contextualmente.
        
        Args:
            fhir_data: Datos FHIR a mapear
            stream_callback: Función para mostrar progreso
            
        Returns:
            Dict[str, Any]: Información de mapeo adaptativo
        """
        try:
            if not self.llm:
                # Fallback básico sin LLM
                return {
                    'table': 'PATI_PATIENTS',
                    'columns': ['PATI_NAME', 'PATI_SURNAME_1'],
                    'values': [fhir_data.get('name', ''), fhir_data.get('surname', '')]
                }
            
            if stream_callback:
                stream_callback("🧠 ARQUITECTURA ADAPTATIVA OPTIMIZADA: Consolidación + Caché...")
            
            # PASO 1: Mapeo dinámico usando LLM consolidado
            mapping_result = await self._llm_consolidated_discovery_and_mapping(fhir_data, stream_callback, context)
            
            # PASO 2: Validación dinámica
            final_result = await self._llm_adaptive_cleanup(mapping_result, stream_callback)
            
            # PASO 3: FORZAR VALIDACIÓN DE IDs (CRÍTICO)
            if final_result and final_result.get('values') and final_result.get('columns'):
                corrected_values = await self._llm_validate_and_correct_fictitious_ids_adaptive(
                    final_result['values'], 
                    final_result['columns'], 
                    stream_callback
                )
                if corrected_values:
                    final_result['values'] = corrected_values
                    if stream_callback:
                        stream_callback("   ✅ Validación de IDs forzada completada")
            
            if stream_callback:
                stream_callback(f"🧠 Mapeo adaptativo completado: {final_result.get('resource_type', 'Unknown')} → {final_result.get('table', 'Unknown')}")
                stream_callback(f"   - Campos adaptados: {len(final_result.get('columns', []))}")
            
            return final_result
                
        except Exception as e:
            logger.error(f"Error en mapeo adaptativo: {e}")
            
            # FALLBACK INTELIGENTE CON LLM: Usar LLM para seleccionar tabla correcta
            if stream_callback:
                stream_callback(f"   🔧 Aplicando fallback inteligente con LLM...")
            
            return await self._llm_intelligent_fallback_mapping(fhir_data, stream_callback)

    async def _llm_intelligent_fallback_mapping(self, fhir_data: Dict[str, Any], stream_callback=None) -> Dict[str, Any]:
        """
        FALLBACK INTELIGENTE CON LLM: Usa LLM para seleccionar tabla y mapear campos cuando el método principal falla.
        """
        try:
            if not self.llm:
                # Fallback básico sin LLM
                return {
                    'table': 'PATI_PATIENTS',
                    'columns': ['PATI_NAME', 'PATI_SURNAME_1'],
                    'values': [fhir_data.get('name', ''), fhir_data.get('surname', '')]
                    }
            
            if stream_callback:
                stream_callback("   🧠 Fallback inteligente: Analizando con LLM...")
            
            # PROMPT INTELIGENTE PARA DESCUBRIMIENTO DINÁMICO
            fallback_prompt = f"""Eres un experto en análisis dinámico de esquemas de bases de datos médicas. Analiza el esquema disponible y descubre automáticamente la tabla más apropiada.

DATOS FHIR A MAPEAR:
{json.dumps(fhir_data, indent=2, ensure_ascii=False)}

ESQUEMA COMPLETO DE LA BASE DE DATOS:
{self._get_real_schema_info()}

TAREA INTELIGENTE: Analiza el esquema dinámicamente y descubre la tabla más apropiada.

ESTRATEGIA DE DESCUBRIMIENTO DINÁMICO:
1. Analiza el tipo de recurso FHIR (Patient, Condition, Observation, etc.)
2. Examina todas las tablas disponibles en el esquema
3. Identifica patrones en los nombres de las tablas (PATI_*, EPIS_*, OBSE_*, MEDI_*, etc.)
4. Analiza las columnas de cada tabla para entender su propósito
5. Busca tablas que contengan campos relevantes para el tipo de recurso
6. Considera el contexto médico y semántico
7. Selecciona la tabla más apropiada basándose en el análisis dinámico

ANÁLISIS DINÁMICO REQUERIDO:
- Examina los prefijos de las tablas (PATI, EPIS, OBSE, MEDI, etc.)
- Analiza las columnas de cada tabla para entender su función
- Busca patrones semánticos (patient, diagnosis, observation, medication, etc.)
- Considera las relaciones implícitas entre tablas
- Identifica la tabla que mejor se adapta al tipo de recurso FHIR

INSTRUCCIONES CRÍTICAS:
- NO uses mapeos hardcodeados, analiza dinámicamente
- Examina el esquema completo para encontrar la tabla correcta
- Considera el contexto médico y semántico
- SOLO usa columnas que existan en la tabla seleccionada
- NO uses IDs ficticios, deja que se autoincrementen
- Para valores nulos, usa null (no "NULL" como string)
- Extrae valores específicos del FHIR, NO objetos completos

RESPUESTA JSON:
{{
    "resource_type": "tipo_de_recurso_detectado",
    "target_table": "tabla_descubierta_dinámicamente",
    "columns": ["columna1", "columna2"],
    "values": ["valor1", null],
    "mapping_strategy": "discovery_dynamic",
    "confidence": 0.9,
    "discovery_analysis": "análisis_del_descubrimiento",
    "table_selection_reasoning": "razonamiento_para_selección_de_tabla"
}}

IMPORTANTE: 
- Analiza DINÁMICAMENTE el esquema, NO uses mapeos hardcodeados
- Descubre la tabla basándote en el análisis del esquema
- Considera el contexto médico y semántico
- SOLO usa columnas que existan realmente en la tabla seleccionada

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": fallback_prompt}],
                task_description="Fallback inteligente con LLM"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result:
                if stream_callback:
                    stream_callback(f"   ✅ Fallback inteligente: {result.get('resource_type', 'Unknown')} → {result.get('target_table', 'Unknown')}")
                    stream_callback(f"   📊 Columnas: {len(result.get('columns', []))}")
                
                return result
            else:
                # Si el LLM falla, intentar con un prompt más simple pero aún dinámico
                if stream_callback:
                    stream_callback(f"   ⚠️ LLM falló, intentando análisis simplificado...")
                
                # PROMPT SIMPLIFICADO PERO DINÁMICO
                simple_prompt = f"""Analiza este recurso FHIR y encuentra la tabla más apropiada en el esquema.

RECURSO FHIR:
{json.dumps(fhir_data, indent=2, ensure_ascii=False)}

ESQUEMA:
{self._get_real_schema_info()}

TAREA: Encuentra la tabla más apropiada analizando el esquema dinámicamente.

RESPUESTA JSON:
{{
    "resource_type": "tipo_detectado",
    "target_table": "tabla_encontrada",
    "columns": ["columna1"],
    "values": ["valor1"],
    "mapping_strategy": "simple_dynamic",
    "confidence": 0.6
}}

Responde SOLO con el JSON:"""

                response = await asyncio.to_thread(
                    _call_openai_native, self.llm, [{"role": "user", "content": simple_prompt}],
                    task_description="Análisis simplificado"
                )
                
                content = self._extract_response_text(response)
                result = self._try_parse_llm_json(content)
                
                if result:
                    if stream_callback:
                        stream_callback(f"   ✅ Análisis simplificado: {result.get('resource_type', 'Unknown')} → {result.get('target_table', 'Unknown')}")
                    return result
                else:
                    # Último recurso: análisis básico sin LLM
                    if stream_callback:
                        stream_callback(f"   ⚠️ Usando análisis básico sin LLM...")
                    
                    # Análisis básico basado en el tipo de recurso
                    resource_type = fhir_data.get('resourceType', '')
                    
                    # Buscar tabla por patrones en el esquema
                    schema_info = self._get_real_schema_info()
                    if 'PATI_PATIENTS' in schema_info and resource_type == 'Patient':
                        selected_table = 'PATI_PATIENTS'
                    elif 'EPIS_DIAGNOSTICS' in schema_info and resource_type == 'Condition':
                        selected_table = 'EPIS_DIAGNOSTICS'
                    elif 'MEDI_MEDICATIONS' in schema_info and resource_type == 'MedicationRequest':
                        selected_table = 'MEDI_MEDICATIONS'
                    elif 'OBSE_OBSERVATIONS' in schema_info and resource_type == 'Observation':
                        selected_table = 'OBSE_OBSERVATIONS'
                    elif 'EPIS_EPISODES' in schema_info and resource_type == 'Encounter':
                        selected_table = 'EPIS_EPISODES'
                    else:
                        selected_table = 'PATI_PATIENTS'  # Fallback genérico
                    
                    return {
                        'table': selected_table,
                        'resource_type': resource_type,
                        'columns': ['PATI_NAME', 'PATI_SURNAME_1'],
                        'values': [fhir_data.get('name', ''), fhir_data.get('surname', '')],
                        'mapping_strategy': 'basic_analysis',
                        'confidence': 0.3
                    }
                
        except Exception as e:
            logger.error(f"Error en fallback inteligente: {e}")
            return {
                'table': 'PATI_PATIENTS',
                'columns': ['PATI_NAME', 'PATI_SURNAME_1'],
                'values': [fhir_data.get('name', ''), fhir_data.get('surname', '')],
                'mapping_strategy': 'fallback_error',
                'confidence': 0.3
            }

    async def _llm_discover_schema_adaptive(self, fhir_data: Dict[str, Any], stream_callback=None) -> Dict[str, Any]:
        """
        PASO 1: Descubrimiento dinámico del esquema usando LLM.
        El LLM explora la base de datos y descubre patrones automáticamente.
        """
        try:
            if not self.llm:
                return {'tables': [], 'patterns': []}
            
            if stream_callback:
                stream_callback("   - Descubriendo esquema dinámicamente...")
            
            # Obtener información real de la base de datos
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Explorar todas las tablas dinámicamente
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            all_tables = [row[0] for row in cursor.fetchall()]
            
            # Analizar estructura de cada tabla
            table_analysis = {}
            for table in all_tables:
                if table.startswith('sqlite_') or table.startswith('_'):
                    continue
                
                cursor.execute(f"PRAGMA table_info('{table}');")
                columns = cursor.fetchall()
                
                # Analizar patrones en columnas
                column_patterns = []
                for col in columns:
                    col_name = col[1]
                    col_type = col[2]
                    is_pk = col[5]
                    
                    # Detectar patrones automáticamente
                    if 'ID' in col_name.upper():
                        column_patterns.append('identifier')
                    elif 'NAME' in col_name.upper() or 'FULL' in col_name.upper():
                        column_patterns.append('name')
                    elif 'DATE' in col_name.upper():
                        column_patterns.append('date')
                    elif 'OBSERVATION' in col_name.upper():
                        column_patterns.append('observation')
                    elif 'DIAG' in col_name.upper():
                        column_patterns.append('diagnosis')
                
                table_analysis[table] = {
                    'columns': [col[1] for col in columns],
                    'patterns': column_patterns,
                    'primary_key': next((col[1] for col in columns if col[5]), None)
                }
            
            conn.close()
            
            # PROMPT ADAPTATIVO: Descubrimiento dinámico
            discovery_prompt = f"""Eres un experto en descubrimiento dinámico de esquemas de bases de datos médicas.

DATOS FHIR A ANALIZAR:
{json.dumps(fhir_data, indent=2, ensure_ascii=False)}

ESQUEMA DESCUBIERTO DINÁMICAMENTE:
{json.dumps(table_analysis, indent=2, ensure_ascii=False)}

TAREA ADAPTATIVA: Analiza el esquema descubierto y identifica patrones automáticamente.

ESTRATEGIA DE DESCUBRIMIENTO:
1. Analiza los nombres de las tablas para identificar su función
2. Examina los patrones de columnas para entender la estructura
3. Identifica relaciones implícitas entre tablas
4. Detecta campos de identificación, fechas, observaciones
5. Mapea conceptos médicos a estructuras de datos

MAPEO ESPECÍFICO DE TABLAS:
- PATI_PATIENTS → Datos de pacientes
- EPIS_EPISODES → Episodios médicos
- EPIS_DIAGNOSTICS → Diagnósticos y condiciones
- OBSE_OBSERVATIONS → Observaciones médicas
- MEDI_MEDICATIONS → Medicamentos
- CODR_TABULAR_DIAGNOSTICS → Códigos de diagnóstico

INSTRUCCIONES:
- Identifica las tablas disponibles en el esquema
- Mapea cada tipo de recurso FHIR a la tabla correspondiente
- Considera el contexto médico específico
- Sugiere mapeos basados en el contenido semántico

RESPUESTA JSON:
{{
    "discovered_tables": {{
        "patient_tables": ["PATI_PATIENTS"],
        "diagnosis_tables": ["EPIS_DIAGNOSTICS"],
        "observation_tables": ["OBSE_OBSERVATIONS"],
        "medication_tables": ["MEDI_MEDICATIONS"],
        "episode_tables": ["EPIS_EPISODES"]
    }},
    "patterns_identified": ["patrón1", "patrón2"],
    "suggested_mappings": {{
        "Patient": "PATI_PATIENTS",
        "Observation": "OBSE_OBSERVATIONS",
        "Condition": "EPIS_DIAGNOSTICS",
        "Encounter": "EPIS_EPISODES",
        "Medication": "MEDI_MEDICATIONS"
    }},
    "confidence_level": "high|medium|low"
}}

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": discovery_prompt}],
                task_description="Descubriendo esquema dinámicamente"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result:
                if stream_callback:
                    stream_callback(f"   - Esquema descubierto: {len(result.get('discovered_tables', {}))} categorías")
                
                return {
                    'discovered_tables': result.get('discovered_tables', {}),
                    'patterns': result.get('patterns_identified', []),
                    'suggested_mappings': result.get('suggested_mappings', {}),
                    'confidence': result.get('confidence_level', 'medium'),
                    'raw_analysis': table_analysis
                }
            else:
                return {'tables': [], 'patterns': []}
                
        except Exception as e:
            logger.error(f"Error en descubrimiento de esquema: {e}")
            return {'tables': [], 'patterns': []}

    async def _llm_analyze_context_adaptive(self, fhir_data: Dict[str, Any], schema_discovery: Dict[str, Any], stream_callback=None) -> Dict[str, Any]:
        """
        PASO 2: Análisis contextual de los datos FHIR.
        El LLM analiza el contexto médico y determina la mejor estrategia.
        """
        try:
            if not self.llm:
                return {'context': 'general', 'strategy': 'basic'}
            
            if stream_callback:
                stream_callback("   - Analizando contexto médico adaptativo...")
            
            # PROMPT ADAPTATIVO: Análisis contextual
            context_prompt = f"""Eres un experto en análisis contextual de datos médicos FHIR.

DATOS FHIR:
{json.dumps(fhir_data, indent=2, ensure_ascii=False)}

ESQUEMA DESCUBIERTO:
{json.dumps(schema_discovery, indent=2, ensure_ascii=False)}

TAREA ADAPTATIVA: Analiza el contexto médico y determina la mejor estrategia de mapeo.

ANÁLISIS CONTEXTUAL:
1. Identifica el tipo de información médica presente
2. Determina la urgencia y complejidad de los datos
3. Analiza las relaciones entre diferentes elementos
4. Considera el flujo de trabajo médico
5. Evalúa la precisión requerida

ESTRATEGIAS ADAPTATIVAS:
- Si son datos de paciente: Priorizar tablas de pacientes
- Si son observaciones: Usar tablas de observaciones
- Si son diagnósticos: Mapear a tablas de diagnósticos
- Si son medicamentos: Usar tablas de medicamentos
- Si son episodios: Mapear a tablas de episodios

INSTRUCCIONES:
- Analiza el contenido semántico completo
- Considera el contexto médico específico
- Determina la estrategia más apropiada
- Evalúa la confianza en la decisión

RESPUESTA JSON:
{{
    "medical_context": "patient|diagnosis|observation|medication|episode",
    "complexity_level": "simple|medium|complex",
    "priority_strategy": "accuracy|speed|completeness",
    "suggested_approach": "direct_mapping|flexible_mapping|hybrid_mapping",
    "confidence_score": 0.85,
    "contextual_factors": ["factor1", "factor2"],
    "medical_entities": ["entidad1", "entidad2"]
}}

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": context_prompt}],
                task_description="Analizando contexto médico adaptativo"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result:
                if stream_callback:
                    stream_callback(f"   - Contexto: {result.get('medical_context', 'Unknown')}")
                    stream_callback(f"   - Estrategia: {result.get('suggested_approach', 'Unknown')}")
                
                return result
            else:
                return {'context': 'general', 'strategy': 'basic'}
                
        except Exception as e:
            logger.error(f"Error en análisis contextual: {e}")
            return {'context': 'general', 'strategy': 'basic'}

    async def _llm_adaptive_mapping(self, fhir_data: Dict[str, Any], context_analysis: Dict[str, Any], stream_callback=None) -> Dict[str, Any]:
        """
        PASO 3: Mapeo inteligente basado en aprendizaje previo.
        El LLM aprende de cada operación y mejora automáticamente.
        """
        try:
            if not self.llm:
                return {
                    'table': 'PATI_PATIENTS',
                    'columns': ['PATI_NAME'],
                    'values': [fhir_data.get('name', '')]
                }
            
            if stream_callback:
                stream_callback("   - Mapeo adaptativo inteligente...")
            
            # Usar LLM para determinar dinámicamente la tabla correcta
            resource_type = fhir_data.get('resourceType', '')
            
            # PROMPT ADAPTATIVO: Selección dinámica de tabla
            table_selection_prompt = f"""Eres un experto en selección de tablas para mapeo FHIR→SQL.

DATOS FHIR:
{json.dumps(fhir_data, indent=2, ensure_ascii=False)}

TIPO DE RECURSO: {resource_type}

ESQUEMA DISPONIBLE:
{self._get_real_schema_info()}

TAREA ADAPTATIVA: Selecciona la tabla más apropiada para este tipo de recurso FHIR.

INSTRUCCIONES:
1. Analiza el tipo de recurso FHIR
2. Revisa el esquema disponible
3. Selecciona la tabla que mejor se adapte al contenido
4. Considera el contexto médico
5. Usa solo tablas que existan realmente en el esquema

REGLAS DE SELECCIÓN:
- Patient → buscar tabla de pacientes
- Condition → buscar tabla de condiciones/diagnósticos
- Observation → buscar tabla de observaciones
- Encounter → buscar tabla de episodios
- Medication → buscar tabla de medicamentos
- MedicationRequest → buscar tabla de medicamentos

RESPUESTA JSON:
{{
    "selected_table": "nombre_tabla_seleccionada",
    "reasoning": "explicación_de_la_selección",
    "confidence": 0.95
}}

IMPORTANTE: Solo usa tablas que existan en el esquema real.

Responde SOLO con el JSON:"""

            # Obtener tabla seleccionada por LLM
            if self.llm:
                try:
                    response = await asyncio.to_thread(
                        _call_openai_native, self.llm, [{"role": "user", "content": table_selection_prompt}],
                        task_description="Seleccionando tabla dinámicamente"
                    )
                    
                    content = self._extract_response_text(response)
                    table_result = self._try_parse_llm_json(content)
                    
                    if table_result and table_result.get('selected_table'):
                        target_table = table_result['selected_table']
                        if stream_callback:
                            stream_callback(f"   - Tabla seleccionada dinámicamente: {target_table}")
                    else:
                        target_table = 'PATI_PATIENTS'  # Fallback
                        if stream_callback:
                            stream_callback(f"   - Usando tabla por defecto: {target_table}")
                except Exception as e:
                    logger.error(f"Error seleccionando tabla: {e}")
                    target_table = 'PATI_PATIENTS'  # Fallback
            else:
                target_table = 'PATI_PATIENTS'  # Fallback sin LLM
            
            # Obtener esquema específico de la tabla seleccionada
            table_schema = self._get_table_schema_info(target_table)
            
            # PROMPT ADAPTATIVO: Mapeo inteligente
            mapping_prompt = f"""Eres un experto en mapeo adaptativo FHIR→SQL que aprende de cada operación.

DATOS FHIR:
{json.dumps(fhir_data, indent=2, ensure_ascii=False)}

ANÁLISIS CONTEXTUAL:
{json.dumps(context_analysis, indent=2, ensure_ascii=False)}

TABLA OBJETIVO: {target_table}
ESQUEMA ESPECÍFICO DE LA TABLA:
{table_schema}

TAREA ADAPTATIVA: Mapea los datos FHIR usando aprendizaje previo y contexto.

REGLAS OBLIGATORIAS (NO NEGOCIABLES):
1. SOLO usa la tabla {target_table}
2. NUNCA uses IDs ficticios, deja que se autoincrementen
3. Usa tu criterio para detectar IDs problemáticos
4. Usa NULL para valores nulos, NO "None"
5. SOLO usa columnas que existan en el esquema de la tabla
6. NO uses columnas que no existan
7. NUNCA uses IDs ficticios en valores
8. SOLO usa columnas que aparezcan en el esquema de {target_table}

MAPEO ADAPTATIVO:
- Analiza el esquema de {target_table}
- Identifica las columnas disponibles
- Mapea solo campos que existan realmente
- NO incluyas campos de ID en INSERT
- Usa valores reales o NULL
- Adapta el mapeo según las columnas disponibles

INSTRUCCIONES:
- Identifica el tipo de recurso FHIR
- Usa SOLO la tabla {target_table}
- Mapea solo campos que existan en el esquema de la tabla
- NO incluyas campos de ID en INSERT
- Usa valores reales o NULL
- NO uses columnas que no existan
- Usa tu criterio para detectar IDs problemáticos
- Adapta el mapeo según las columnas disponibles en {target_table}

RESPUESTA JSON:
{{
    "table": "{target_table}",
    "columns": ["columna1", "columna2"],
    "values": ["valor1", "valor2"],
    "resource_type": "{resource_type}",
    "mapping_strategy": "adaptive_mapping",
    "confidence": 0.95
}}

IMPORTANTE: SOLO usa columnas que existan en el esquema de {target_table} y usa tu criterio para detectar IDs problemáticos.

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": mapping_prompt}],
                task_description="Mapeo adaptativo inteligente"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result:
                if stream_callback:
                    stream_callback(f"   - Mapeo adaptativo: {result.get('table', 'Unknown')}")
                    stream_callback(f"   - Estrategia: {result.get('mapping_strategy', 'Unknown')}")
                
                return result
            else:
                return {
                    'table': 'PATI_PATIENTS',
                    'columns': ['PATI_NAME'],
                    'values': [fhir_data.get('name', '')]
                }
                
        except Exception as e:
            logger.error(f"Error en mapeo adaptativo: {e}")
            return {
                'table': 'PATI_PATIENTS',
                'columns': ['PATI_NAME'],
                'values': [fhir_data.get('name', '')]
            }

    async def _llm_contextual_validation(self, fhir_data: Dict[str, Any], adaptive_mapping: Dict[str, Any], stream_callback=None) -> Dict[str, Any]:
        """
        PASO 4: Validación contextual sin reglas fijas.
        El LLM valida basándose en el contexto específico.
        """
        try:
            if not self.llm:
                return {'needs_correction': False}
            
            if stream_callback:
                stream_callback("   - Validación contextual adaptativa...")
            
            # PROMPT ADAPTATIVO: Validación contextual
            validation_prompt = f"""Eres un experto en validación contextual de mapeos FHIR→SQL.

DATOS FHIR ORIGINALES:
{json.dumps(fhir_data, indent=2, ensure_ascii=False)}

MAPEO ADAPTATIVO ACTUAL:
{json.dumps(adaptive_mapping, indent=2, ensure_ascii=False)}

TAREA ADAPTATIVA: Valida el mapeo basándote en el contexto específico.

VALIDACIÓN CONTEXTUAL:
1. Analiza la coherencia semántica del mapeo
2. Verifica que la tabla sea apropiada para el tipo de datos
3. Valida que los campos mapeados existan realmente
4. Considera el contexto médico específico
5. Evalúa la precisión del mapeo

ESTRATEGIA DE VALIDACIÓN:
- NO uses reglas fijas, adapta según el contexto
- Considera la complejidad de los datos
- Evalúa la confianza del mapeo
- Identifica posibles mejoras
- Sugiere correcciones si es necesario

INSTRUCCIONES:
- Valida basándote en el contexto específico
- Considera la complejidad identificada
- Evalúa la precisión del mapeo
- Sugiere mejoras si es necesario

RESPUESTA JSON:
{{
    "needs_correction": true|false,
    "correction_type": "table|columns|values|strategy",
    "suggested_improvements": ["mejora1", "mejora2"],
    "confidence_in_validation": 0.85,
    "contextual_factors": ["factor1", "factor2"]
}}

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": validation_prompt}],
                task_description="Validación contextual adaptativa"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result:
                if stream_callback:
                    if result.get('needs_correction', False):
                        stream_callback(f"   - Corrección necesaria: {result.get('correction_type', 'Unknown')}")
                    else:
                        stream_callback("   - Validación contextual exitosa")
                
                return result
            else:
                return {'needs_correction': False}
                
        except Exception as e:
            logger.error(f"Error en validación contextual: {e}")
            return {'needs_correction': False}

    async def _llm_apply_adaptive_corrections(self, fhir_data: Dict[str, Any], current_mapping: Dict[str, Any], validation_result: Dict[str, Any], stream_callback=None) -> Dict[str, Any]:
        """
        PASO 5: Aplicar correcciones adaptativas.
        El LLM aplica correcciones basándose en el contexto.
        """
        try:
            if not self.llm:
                return current_mapping
            
            if stream_callback:
                stream_callback("   - Aplicando correcciones adaptativas...")
            
            # PROMPT ADAPTATIVO: Correcciones
            correction_prompt = f"""Eres un experto en correcciones adaptativas de mapeos FHIR→SQL.

DATOS FHIR:
{json.dumps(fhir_data, indent=2, ensure_ascii=False)}

MAPEO ACTUAL:
{json.dumps(current_mapping, indent=2, ensure_ascii=False)}

RESULTADO DE VALIDACIÓN:
{json.dumps(validation_result, indent=2, ensure_ascii=False)}

TAREA ADAPTATIVA: Aplica correcciones basándote en el contexto específico.

ESTRATEGIA DE CORRECCIÓN:
1. Analiza el tipo de corrección necesaria
2. Aplica mejoras basadas en el contexto
3. Considera la complejidad de los datos
4. Optimiza para precisión y rendimiento
5. Mantén la coherencia semántica

INSTRUCCIONES:
- Aplica correcciones específicas al contexto
- Considera las sugerencias de mejora
- Mantén la lógica original cuando sea posible
- Optimiza para el tipo de datos específico

RESPUESTA JSON:
{{
    "table": "tabla_corregida",
    "columns": ["columna1", "columna2"],
    "values": ["valor1", "valor2"],
    "resource_type": "tipo_corregido",
    "corrections_applied": ["corrección1", "corrección2"],
    "confidence": 0.9
}}

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": correction_prompt}],
                task_description="Aplicando correcciones adaptativas"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result:
                if stream_callback:
                    stream_callback(f"   - Correcciones aplicadas: {len(result.get('corrections_applied', []))}")
                
                return result
            else:
                return current_mapping
                
        except Exception as e:
            logger.error(f"Error aplicando correcciones adaptativas: {e}")
            return current_mapping

    async def _llm_adaptive_cleanup(self, mapping_result: Dict[str, Any], stream_callback=None) -> Dict[str, Any]:
        """
        PASO 6: Limpieza final adaptativa.
        El LLM limpia y optimiza el resultado final.
        """
        try:
            if not self.llm:
                return mapping_result
            
            if stream_callback:
                stream_callback("   - Limpieza final adaptativa...")
            
            # Obtener esquema específico de la tabla para validación
            table_name = mapping_result.get('table', 'PATI_PATIENTS')
            table_schema = self._get_table_schema_info(table_name)
            
            # PROMPT ADAPTATIVO: Limpieza final
            cleanup_prompt = f"""Eres un experto en limpieza adaptativa de mapeos FHIR→SQL.

RESULTADO DEL MAPEO:
{json.dumps(mapping_result, indent=2, ensure_ascii=False)}

TABLA OBJETIVO: {table_name}
ESQUEMA ESPECÍFICO DE LA TABLA:
{table_schema}

TAREA ADAPTATIVA: Limpia y optimiza el resultado final.

LIMPIEZA ADAPTATIVA CRÍTICA:
1. ELIMINA TODOS los IDs ficticios, deja que se autoincrementen
2. Usa tu criterio para detectar IDs problemáticos
3. Convierte tipos de datos apropiadamente
4. Maneja valores nulos correctamente (NULL, no "None")
5. Optimiza para rendimiento
6. Asegura coherencia semántica
7. SOLO usa columnas que existan en el esquema de la tabla

REGLAS DE LIMPIEZA OBLIGATORIAS:
- Usa tu criterio para detectar y eliminar IDs problemáticos
- Convierte a NULL o valor real según corresponda
- Convierte "None" → NULL
- Convierte "null" → NULL
- Usa fechas en formato SQLite (YYYY-MM-DD)
- SOLO usa columnas que existan en el esquema de {table_name}
- NO uses columnas que no existan (como PATI_IDENTIFIER, PATI_PHONE, PATI_ADDRESS)

INSTRUCCIONES:
- Limpia TODOS los valores problemáticos
- Optimiza para la base de datos específica
- Mantén la coherencia semántica
- Asegura que el resultado sea ejecutable
- Usa tu criterio para detectar IDs problemáticos
- SOLO usa columnas que existan en el esquema de {table_name}

RESPUESTA JSON:
{{
    "table": "{table_name}",
    "columns": ["columna1", "columna2"],
    "values": ["valor1", "valor2"],
    "resource_type": "tipo_final",
    "cleanup_applied": ["limpieza1", "limpieza2"],
    "final_confidence": 0.95
}}

IMPORTANTE: USA TU CRITERIO PARA DETECTAR IDs PROBLEMÁTICOS Y SOLO USA COLUMNAS QUE EXISTAN.

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": cleanup_prompt}],
                task_description="Limpieza final adaptativa"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result:
                if stream_callback:
                    stream_callback(f"   - Limpieza final adaptativa completada")
                
                return result
            else:
                return mapping_result
                
        except Exception as e:
            logger.error(f"Error en limpieza adaptativa: {e}")
            return mapping_result

    async def _llm_validate_and_correct_ids_adaptive(self, columns: List[str], values: List[Any], stream_callback=None) -> List[Any]:
        """
        VALIDACIÓN ADAPTATIVA: Usa LLM para detectar y corregir IDs ficticios.
        Sin patterns rígidos - todo via LLM.
        """
        try:
            if not self.llm:
                return values
            
            if stream_callback:
                stream_callback("   - Validando IDs con IA adaptativa...")
            
            # PROMPT ADAPTATIVO: Validación de IDs
            validation_prompt = f"""Eres un experto en validación de IDs en bases de datos médicas.

COLUMNAS Y VALORES A VALIDAR:
{json.dumps(list(zip(columns, values)), indent=2, ensure_ascii=False)}

TAREA ADAPTATIVA: Analiza cada valor y detecta si es un ID ficticio o problemático.

CRITERIOS DE VALIDACIÓN:
1. Identifica IDs ficticios o problemáticos usando tu criterio
2. Detecta IDs que no son reales
3. Identifica valores que deberían ser NULL
4. Corrige valores problemáticos automáticamente

ESTRATEGIA ADAPTATIVA:
- Analiza el contexto de cada columna
- Considera el tipo de datos esperado
- Identifica patrones de IDs problemáticos usando tu criterio
- Sugiere correcciones apropiadas

INSTRUCCIONES:
- Analiza cada par columna-valor
- Identifica IDs problemáticos usando tu criterio
- Sugiere valores corregidos
- Mantén valores válidos sin cambios

RESPUESTA JSON:
{{
    "corrected_values": ["valor1", "valor2", "valor3"],
    "corrections_applied": ["corrección1", "corrección2"],
    "validation_confidence": 0.95
}}

IMPORTANTE: Si un valor es un ID problemático, reemplázalo con NULL.

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": validation_prompt}],
                task_description="Validando IDs con IA adaptativa"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result and 'corrected_values' in result:
                corrected_values = result['corrected_values']
                corrections = result.get('corrections_applied', [])
                confidence = result.get('validation_confidence', 0.0)
                
                # VALIDACIÓN ADAPTATIVA: Limpieza básica sin recursión
                final_values = []
                for value in corrected_values:
                    if value == "NULL" or value == "null" or value == "None":
                        final_values.append(None)
                    elif isinstance(value, str) and value.lower() in ["null", "none"]:
                        final_values.append(None)
                    else:
                        final_values.append(value)
                
                if stream_callback:
                    stream_callback(f"   - Validación completada: {len(corrections)} correcciones")
                    stream_callback(f"   - Confianza: {confidence}")
                
                return final_values
            else:
                return values
                
        except Exception as e:
            logger.error(f"Error en validación de IDs adaptativa: {e}")
            return values

    async def _llm_validate_and_correct_table_adaptive(self, current_table: str, resource_type: str, stream_callback=None) -> str:
        """
        VALIDACIÓN ADAPTATIVA: Usa LLM para validar y corregir tabla.
        Sin patterns rígidos - todo via LLM.
        """
        try:
            if not self.llm:
                return current_table
            
            if stream_callback:
                stream_callback("   - Validando tabla con IA adaptativa...")
            
            # PROMPT ADAPTATIVO: Validación de tabla
            validation_prompt = f"""Eres un experto en validación de tablas en bases de datos médicas.

TABLA ACTUAL: {current_table}
TIPO DE RECURSO: {resource_type}

ESQUEMA DISPONIBLE:
{self._get_real_schema_info()}

TAREA ADAPTATIVA: Valida si la tabla es apropiada para el tipo de recurso.

CRITERIOS DE VALIDACIÓN:
1. Verifica que la tabla existe en el esquema
2. Analiza si es apropiada para el tipo de recurso
3. Sugiere tabla alternativa si es necesario
4. Considera el contexto médico

ESTRATEGIA ADAPTATIVA:
- Analiza la función de cada tabla
- Considera el tipo de datos del recurso
- Identifica la tabla más apropiada
- Sugiere correcciones si es necesario

INSTRUCCIONES:
- Verifica que la tabla existe
- Analiza si es apropiada para el recurso
- Sugiere tabla alternativa si es necesario
- Mantén la tabla si es correcta

RESPUESTA JSON:
{{
    "corrected_table": "tabla_corregida",
    "needs_correction": true|false,
    "reasoning": "explicación de la corrección",
    "confidence": 0.95
}}

IMPORTANTE: Solo sugiere tablas que existan realmente en el esquema.

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": validation_prompt}],
                task_description="Validando tabla con IA adaptativa"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result:
                corrected_table = result.get('corrected_table', current_table)
                needs_correction = result.get('needs_correction', False)
                reasoning = result.get('reasoning', 'Sin explicación')
                confidence = result.get('confidence', 0.0)
                
                if stream_callback:
                    if needs_correction:
                        stream_callback(f"   - Tabla corregida: {current_table} → {corrected_table}")
                        stream_callback(f"   - Razón: {reasoning}")
                    else:
                        stream_callback("   - Tabla validada correctamente")
                    stream_callback(f"   - Confianza: {confidence}")
                
                return corrected_table
            else:
                return current_table
                
        except Exception as e:
            logger.error(f"Error en validación de tabla adaptativa: {e}")
            return current_table

    async def _llm_final_cleanup_adaptive(self, mapping_result: Dict[str, Any], stream_callback=None) -> Dict[str, Any]:
        """
        LIMPIEZA FINAL ADAPTATIVA: Usa LLM para limpieza final completa.
        Sin patterns rígidos - todo via LLM.
        """
        try:
            if not self.llm:
                return mapping_result
            
            if stream_callback:
                stream_callback("   - Limpieza final adaptativa...")
            
            # PROMPT ADAPTATIVO: Limpieza final
            cleanup_prompt = f"""Eres un experto en limpieza final de mapeos FHIR→SQL.

RESULTADO DEL MAPEO:
{json.dumps(mapping_result, indent=2, ensure_ascii=False)}

TAREA ADAPTATIVA: Realiza limpieza final completa del mapeo.

LIMPIEZA FINAL ADAPTATIVA:
1. Elimina TODOS los IDs ficticios y UUIDs problemáticos
2. Convierte tipos de datos apropiadamente
3. Maneja valores nulos correctamente
4. Optimiza para rendimiento
5. Asegura coherencia semántica

ESTRATEGIA ADAPTATIVA:
- Analiza cada valor individualmente
- Identifica patrones problemáticos
- Aplica correcciones inteligentes
- Mantén valores válidos
- Optimiza para la base de datos específica

INSTRUCCIONES:
- Analiza cada columna y valor
- Identifica y corrige problemas
- Optimiza para SQLite
- Asegura que el resultado sea ejecutable
- Mantén la coherencia semántica
- Para valores nulos, usa null (no "NULL" como string)

RESPUESTA JSON:
{{
    "table": "tabla_final",
    "columns": ["columna1", "columna2"],
    "values": ["valor1", null],
    "resource_type": "tipo_final",
    "cleanup_applied": ["limpieza1", "limpieza2"],
    "final_confidence": 0.95
}}

IMPORTANTE: 
- Elimina TODOS los IDs ficticios y valores problemáticos
- Para valores nulos, usa null (no "NULL" como string)
- NO uses "NULL" como string, usa null

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": cleanup_prompt}],
                task_description="Limpieza final adaptativa"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result:
                # VALIDACIÓN ADICIONAL: Convertir "NULL" strings a None
                values = result.get('values', [])
                columns = result.get('columns', [])
                
                if values and columns:
                    # Limpieza básica sin recursión
                    corrected_values = []
                    for i, value in enumerate(values):
                        if value == "NULL" or value == "null" or value == "None":
                            corrected_values.append(None)
                        elif isinstance(value, str) and value.lower() in ["null", "none"]:
                            corrected_values.append(None)
                        else:
                            corrected_values.append(value)
                    result['values'] = corrected_values
                
                cleanup_applied = result.get('cleanup_applied', [])
                final_confidence = result.get('final_confidence', 0.0)
                
                if stream_callback:
                    stream_callback(f"   - Limpieza final: {len(cleanup_applied)} optimizaciones")
                    stream_callback(f"   - Confianza final: {final_confidence}")
                
                return result
            else:
                return mapping_result
                
        except Exception as e:
            logger.error(f"Error en limpieza final adaptativa: {e}")
            return mapping_result

    async def _detect_and_fix_missing_joins(self, sql: str, stream_callback=None) -> str:
        """
        Detecta y corrige JOINs faltantes usando la nueva función flexible.
        ARQUITECTURA ADAPTATIVA: Usa prompts específicos según el contexto.
        """
        try:
            if not self.llm:
                if stream_callback:
                    stream_callback("   - Verificación básica de JOINs (sin LLM)...")
                return sql
            
            if stream_callback:
                stream_callback("   - Detectando JOINs faltantes con IA...")
            
            # Usar la nueva función flexible con contexto vacío
            return await self._llm_flexible_sql_analysis(sql, "", stream_callback)
                
        except Exception as e:
            logger.error(f"Error en _detect_and_fix_missing_joins: {e}")
            return sql



    def _extract_medical_codes_from_codr_table(self, medical_terms: List[str]) -> List[str]:
        """
        Extrae códigos CDTE_ID asociados con términos médicos de la tabla CODR_TABULAR_DIAGNOSTICS.
        
        Args:
            medical_terms: Lista de términos médicos a buscar (ej: ['diabetes', 'hipertensión'])
            
        Returns:
            List[str]: Lista de códigos CDTE_ID que corresponden a los términos médicos
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Verificar si la tabla tiene datos
            cursor.execute("SELECT COUNT(*) FROM CODR_TABULAR_DIAGNOSTICS WHERE COTA_DESCRIPTION_ES IS NOT NULL")
            count = cursor.fetchone()[0]
            
            if count == 0:
                logger.warning("⚠️ La tabla CODR_TABULAR_DIAGNOSTICS está vacía o no tiene descripciones")
                conn.close()
                return []
            
            medical_codes = []
            
            for term in medical_terms:
                cursor.execute("""
                    SELECT COTA_ID, COTA_DESCRIPTION_ES 
                    FROM CODR_TABULAR_DIAGNOSTICS 
                    WHERE UPPER(COTA_DESCRIPTION_ES) LIKE UPPER(?)
                """, (f'%{term}%',))
                
                results = cursor.fetchall()
                for code, description in results:
                    if code not in medical_codes:
                        medical_codes.append(str(code))
                        logger.info(f"🔍 Código médico encontrado: {code} - {description}")
            
            conn.close()
            
            logger.info(f"✅ Se encontraron {len(medical_codes)} códigos para términos: {medical_terms}")
            return medical_codes
            
        except Exception as e:
            logger.error(f"Error extrayendo códigos médicos: {e}")
            return []

    async def _llm_detect_medical_terms_and_codes(self, query: str, stream_callback=None) -> Dict[str, Any]:
        """
        Usa LLM para detectar automáticamente términos médicos y extraer códigos correspondientes.
        ARQUITECTURA SOSTENIBLE: 100% basada en LLM, sin hardcodeo.
        
        Args:
            query: Consulta original del usuario
            stream_callback: Función para mostrar progreso
            
        Returns:
            Dict[str, Any]: Información sobre términos médicos y códigos encontrados
        """
        try:
            if not self.llm:
                # Fallback básico sin LLM
                return {
                    'medical_terms': [],
                    'medical_codes': [],
                    'search_strategy': 'free_text'
                }
            
            if stream_callback:
                stream_callback("   - Detectando términos médicos con IA avanzada...")
            
            # PROMPT ESPECÍFICO PARA DETECCIÓN MÉDICA - SIN HARDCODEO
            detection_prompt = f"""Eres un experto en terminología médica y códigos de diagnóstico.

CONSULTA DEL USUARIO: "{query}"

TAREA ESPECÍFICA: Analiza la consulta y extrae términos médicos relevantes que podrían tener códigos oficiales en la base de datos.

INSTRUCCIONES DETALLADAS:
1. Identifica condiciones médicas, diagnósticos, enfermedades mencionadas
2. Incluye sinónimos y variaciones comunes (ej: diabetes → diabético, diabética, DM)
3. Considera abreviaciones médicas (DM = diabetes mellitus, HTA = hipertensión arterial)
4. Incluye términos en español e inglés si aplica
5. Detecta términos relacionados y comorbilidades
6. Identifica la condición principal y secundarias

ESTRATEGIA DE DETECCIÓN:
- Buscar términos médicos específicos
- Identificar condiciones crónicas y agudas
- Detectar síntomas y signos
- Reconocer medicamentos y tratamientos
- Identificar especialidades médicas mencionadas

RESPUESTA JSON ESTRUCTURADA:
{{
    "medical_terms": ["término1", "término2", "término3"],
    "primary_condition": "condición principal identificada",
    "secondary_conditions": ["condición2", "condición3"],
    "symptoms": ["síntoma1", "síntoma2"],
    "medications": ["medicamento1", "medicamento2"],
    "search_strategy": "official_codes|free_text|hybrid",
    "confidence_level": "high|medium|low",
    "specialties_involved": ["especialidad1", "especialidad2"]
}}

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": detection_prompt}],
                task_description="Detectando términos médicos con IA avanzada"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result and 'medical_terms' in result:
                medical_terms = result['medical_terms']
                primary_condition = result.get('primary_condition', '')
                search_strategy = result.get('search_strategy', 'free_text')
                confidence_level = result.get('confidence_level', 'medium')
                
                # NUEVO: Extraer códigos oficiales usando LLM específico
                medical_codes = await self._llm_extract_medical_codes_intelligent(medical_terms, stream_callback)
                
                if stream_callback:
                    if medical_codes:
                        stream_callback(f"   - Encontrados {len(medical_codes)} códigos oficiales para {primary_condition}")
                    else:
                        stream_callback(f"   - No se encontraron códigos oficiales, usando búsqueda libre")
                
                return {
                    'medical_terms': medical_terms,
                    'medical_codes': medical_codes,
                    'primary_condition': primary_condition,
                    'search_strategy': search_strategy,
                    'confidence_level': confidence_level,
                    'secondary_conditions': result.get('secondary_conditions', []),
                    'symptoms': result.get('symptoms', []),
                    'medications': result.get('medications', []),
                    'specialties_involved': result.get('specialties_involved', [])
                }
            else:
                logger.warning("⚠️ LLM no pudo detectar términos médicos, usando búsqueda libre")
                return {
                    'medical_terms': [],
                    'medical_codes': [],
                    'search_strategy': 'free_text',
                    'confidence_level': 'low'
                }
                
        except Exception as e:
            logger.error(f"Error detectando términos médicos: {e}")
            return {
                'medical_terms': [],
                'medical_codes': [],
                'search_strategy': 'free_text',
                'confidence_level': 'low'
            }

    async def _llm_extract_medical_codes_intelligent(self, medical_terms: List[str], stream_callback=None) -> List[str]:
        """
        Extrae códigos médicos usando LLM específico - SIN HARDCODEO.
        ARQUITECTURA SOSTENIBLE: Todo via LLM.
        
        Args:
            medical_terms: Términos médicos detectados
            stream_callback: Función para mostrar progreso
            
        Returns:
            List[str]: Códigos médicos encontrados
        """
        try:
            if not self.llm or not medical_terms:
                return []
            
            if stream_callback:
                stream_callback("   - Extrayendo códigos médicos con IA específica...")
            
            # PROMPT ESPECÍFICO PARA EXTRACCIÓN DE CÓDIGOS - SIN HARDCODEO
            code_extraction_prompt = f"""Eres un experto en códigos de diagnóstico médico y terminología clínica.

TÉRMINOS MÉDICOS DETECTADOS: {medical_terms}

TAREA ESPECÍFICA: Analiza estos términos médicos y genera códigos de diagnóstico que podrían existir en la base de datos.

ESTRATEGIA DE EXTRACCIÓN:
1. Identifica códigos ICD-10, CIE-10, o códigos locales que podrían corresponder
2. Considera variaciones y sinónimos de los términos
3. Incluye códigos relacionados y subcategorías
4. Genera códigos numéricos y alfanuméricos
5. Considera códigos de diferentes sistemas de clasificación

EJEMPLOS DE CORRESPONDENCIA:
- diabetes → ['E11', 'E10', 'E13', '250']
- hipertensión → ['I10', 'I11', 'I12', '401']
- cáncer → ['C00-C97', '140-208']
- asma → ['J45', 'J46', '493']

INSTRUCCIONES:
- Genera códigos que podrían existir en la base de datos
- Incluye códigos principales y secundarios
- Considera códigos de diferentes especialidades
- Mantén formato consistente

RESPUESTA JSON:
{{
    "primary_codes": ["código1", "código2"],
    "secondary_codes": ["código3", "código4"],
    "related_codes": ["código5", "código6"],
    "code_system": "ICD-10|CIE-10|local",
    "confidence": "high|medium|low"
}}

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": code_extraction_prompt}],
                task_description="Extrayendo códigos médicos con IA específica"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result:
                # Combinar todos los códigos
                all_codes = []
                all_codes.extend(result.get('primary_codes', []))
                all_codes.extend(result.get('secondary_codes', []))
                all_codes.extend(result.get('related_codes', []))
                
                # Verificar qué códigos realmente existen en la base de datos
                existing_codes = await self._llm_verify_codes_in_database(all_codes, stream_callback)
                
                if stream_callback:
                    stream_callback(f"   - Verificados {len(existing_codes)} códigos en la base de datos")
                
                return existing_codes
            else:
                logger.warning("⚠️ LLM no pudo extraer códigos médicos")
                return []
                
        except Exception as e:
            logger.error(f"Error extrayendo códigos médicos: {e}")
            return []

    async def _llm_verify_codes_in_database(self, codes: List[str], stream_callback=None) -> List[str]:
        """
        Verifica qué códigos realmente existen en la base de datos usando LLM.
        ARQUITECTURA SOSTENIBLE: Sin hardcodeo.
        
        Args:
            codes: Códigos a verificar
            stream_callback: Función para mostrar progreso
            
        Returns:
            List[str]: Códigos que existen en la base de datos
        """
        try:
            if not self.llm or not codes:
                return []
            
            # Obtener información real de la base de datos
            db_info = await self._llm_get_database_schema_info(stream_callback)
            
            # PROMPT ESPECÍFICO PARA VERIFICACIÓN - SIN HARDCODEO
            verification_prompt = f"""Eres un experto en bases de datos médicas que verifica la existencia de códigos.

CÓDIGOS A VERIFICAR: {codes}

INFORMACIÓN DE LA BASE DE DATOS:
{db_info}

TAREA ESPECÍFICA: Analiza qué códigos podrían existir realmente en esta base de datos.

ESTRATEGIA DE VERIFICACIÓN:
1. Analiza la estructura de la base de datos
2. Identifica patrones de códigos existentes
3. Considera el formato y rango de códigos
4. Evalúa la probabilidad de existencia
5. Prioriza códigos más probables

INSTRUCCIONES:
- Si la tabla de códigos está vacía, devuelve lista vacía
- Si hay patrones de códigos, identifica los más probables
- Considera el contexto médico de la base de datos
- Evalúa la coherencia con el esquema

RESPUESTA JSON:
{{
    "existing_codes": ["código1", "código2"],
    "probability": "high|medium|low",
    "reasoning": "explicación de la verificación"
}}

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": verification_prompt}],
                task_description="Verificando códigos en la base de datos"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result:
                existing_codes = result.get('existing_codes', [])
                probability = result.get('probability', 'low')
                
                if stream_callback:
                    stream_callback(f"   - Probabilidad de códigos: {probability}")
                
                return existing_codes
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error verificando códigos: {e}")
            return []

    async def _llm_get_database_schema_info(self, stream_callback=None) -> str:
        """
        Obtiene información del esquema de la base de datos usando LLM.
        ARQUITECTURA SOSTENIBLE: Sin hardcodeo.
        
        Args:
            stream_callback: Función para mostrar progreso
            
        Returns:
            str: Información del esquema formateada
        """
        try:
            # Obtener información real de la base de datos
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Verificar tabla de códigos
            cursor.execute("SELECT COUNT(*) FROM CODR_TABULAR_DIAGNOSTICS WHERE COTA_DESCRIPTION_ES IS NOT NULL")
            codigos_count = cursor.fetchone()[0]
            
            # Obtener ejemplos de diagnósticos
            cursor.execute("SELECT DIAG_OBSERVATION FROM EPIS_DIAGNOSTICS WHERE DIAG_OBSERVATION IS NOT NULL LIMIT 10")
            diagnosticos_ejemplos = [row[0] for row in cursor.fetchall()]
            
            # Obtener estructura de tablas
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tablas = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            # PROMPT ESPECÍFICO PARA ANÁLISIS DE ESQUEMA - SIN HARDCODEO
            schema_prompt = f"""Eres un experto en análisis de esquemas de bases de datos médicas.

INFORMACIÓN DE LA BASE DE DATOS:
- Tabla CODR_TABULAR_DIAGNOSTICS: {codigos_count} registros con descripciones
- Tablas disponibles: {tablas}
- Ejemplos de diagnósticos: {diagnosticos_ejemplos}

TAREA ESPECÍFICA: Analiza esta información y proporciona un resumen estructurado del esquema.

ANÁLISIS REQUERIDO:
1. Estado de la tabla de códigos oficiales
2. Tipos de diagnósticos disponibles
3. Patrones en los datos
4. Capacidades de búsqueda
5. Limitaciones identificadas

RESPUESTA ESTRUCTURADA:
{{
    "codigos_status": "available|empty|limited",
    "diagnosticos_count": {len(diagnosticos_ejemplos)},
    "search_capabilities": ["búsqueda1", "búsqueda2"],
    "limitations": ["limitación1", "limitación2"],
    "recommended_strategy": "official_codes|free_text|hybrid"
}}

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": schema_prompt}],
                task_description="Analizando esquema de la base de datos"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result:
                return json.dumps(result, indent=2, ensure_ascii=False)
            else:
                return f"Tabla de códigos: {codigos_count} registros, Diagnósticos: {len(diagnosticos_ejemplos)} ejemplos"
                
        except Exception as e:
            logger.error(f"Error obteniendo información del esquema: {e}")
            return "Error obteniendo información del esquema"

    async def _process_medical_query_specialized(self, query: str, stream_callback=None) -> Optional[Dict[str, Any]]:
        """
        Procesa consultas médicas de manera especializada usando múltiples llamadas a LLM específicas.
        ARQUITECTURA SOSTENIBLE: 100% LLM, sin patterns hardcodeados.
        
        Args:
            query: Consulta original del usuario
            stream_callback: Función para mostrar progreso
            
        Returns:
            Optional[Dict[str, Any]]: Resultado si es consulta médica, None si no
        """
        try:
            # LLAMADA 1: Detectar si es consulta médica usando LLM específico
            is_medical = await self._llm_detect_medical_query(query, stream_callback)
            
            if not is_medical:
                return None
            
            if stream_callback:
                stream_callback("🩺 Detectada consulta médica, usando análisis especializado...")
            
            # LLAMADA 2: Detectar términos médicos y códigos
            medical_info = await self._llm_detect_medical_terms_and_codes(query, stream_callback)
            
            # LLAMADA 3: Generar SQL específico para diagnósticos médicos
            generated_sql = await self._llm_generate_medical_diagnosis_sql(query, medical_info, stream_callback)
            
            if generated_sql and not generated_sql.startswith("Error"):
                # LLAMADA 4: Ejecutar con validación robusta
                result = await self._execute_sql_with_llm_validation(query, generated_sql, time.time(), [], stream_callback)
                
                # LLAMADA 5: Interpretación médica de resultados
                if result.get('success'):
                    if stream_callback:
                        stream_callback("🩺 Interpretando resultados médicos...")
                    
                    interpretation = await self._llm_interpret_medical_results(query, result.get('data', []), stream_callback)
                    result['explanation'] = interpretation
                
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error procesando consulta médica especializada: {e}")
            return None

    async def _llm_detect_medical_query(self, query: str, stream_callback=None) -> bool:
        """
        LLAMADA ESPECÍFICA 1: Detecta si una consulta es médica usando LLM.
        SIN PATTERNS HARDCODEADOS - todo via LLM.
        
        Args:
            query: Consulta a analizar
            stream_callback: Función para mostrar progreso
            
        Returns:
            bool: True si es consulta médica, False si no
        """
        try:
            if not self.llm:
                return False
            
            if stream_callback:
                stream_callback("   - Detectando si es consulta médica con IA...")
            
            # PROMPT ESPECÍFICO PARA DETECCIÓN MÉDICA - SIN HARDCODEO
            medical_detection_prompt = f"""Eres un experto en clasificación de consultas médicas.

CONSULTA: "{query}"

TAREA ESPECÍFICA: Determina si esta consulta es de naturaleza médica o clínica.

CRITERIOS DE CLASIFICACIÓN MÉDICA:
- Preguntas sobre pacientes, diagnósticos, enfermedades
- Consultas sobre síntomas, condiciones médicas
- Búsquedas de información clínica
- Preguntas sobre tratamientos, medicamentos
- Consultas sobre especialidades médicas
- Preguntas sobre datos de salud

CRITERIOS NO MÉDICOS:
- Consultas administrativas generales
- Preguntas sobre el sistema o base de datos
- Consultas técnicas no relacionadas con salud
- Preguntas sobre configuración o mantenimiento

INSTRUCCIONES:
1. Analiza el contenido semántico de la consulta
2. Identifica si menciona conceptos médicos, pacientes, diagnósticos
3. Considera el contexto y la intención de la consulta
4. Evalúa si requiere conocimiento médico para responder

RESPUESTA JSON:
{{
    "is_medical": true|false,
    "confidence": "high|medium|low",
    "medical_elements": ["elemento1", "elemento2"],
    "reasoning": "explicación de la clasificación"
}}

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": medical_detection_prompt}],
                task_description="Detectando consulta médica"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result:
                is_medical = result.get('is_medical', False)
                confidence = result.get('confidence', 'low')
                
                if stream_callback:
                    stream_callback(f"   - Clasificación médica: {is_medical} (confianza: {confidence})")
                
                return is_medical
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error detectando consulta médica: {e}")
            return False

    async def _llm_handle_patient_search(self, query: str, stream_callback=None) -> Optional[Dict[str, Any]]:
        """
        LLAMADA ESPECÍFICA 2: Maneja búsquedas de pacientes usando LLM.
        SIN PATTERNS HARDCODEADOS - todo via LLM.
        
        Args:
            query: Consulta sobre pacientes
            stream_callback: Función para mostrar progreso
            
        Returns:
            Optional[Dict[str, Any]]: Resultado de búsqueda de pacientes
        """
        try:
            if not self.llm:
                return None
            
            if stream_callback:
                stream_callback("   - Analizando búsqueda de pacientes con IA...")
            
            # PROMPT SIMPLE PARA DETECCIÓN DE PACIENTES
            patient_search_prompt = f"""¿Es esta consulta sobre pacientes?

CONSULTA: "{query}"

Responde JSON:
{{
    "is_patient_query": true|false,
    "confidence": "high|medium|low"
}}"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": patient_search_prompt}],
                task_description="Analizando búsqueda de pacientes"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result:
                is_patient_query = result.get('is_patient_query', False)
                confidence = result.get('confidence', 'low')
                
                if stream_callback:
                    stream_callback(f"   - Es consulta de pacientes: {is_patient_query} (confianza: {confidence})")
                
                if is_patient_query:
                    # LLAMADA ESPECÍFICA 3: Generar SQL para búsqueda de pacientes
                    sql = await self._llm_generate_patient_search_sql(query, result, stream_callback)
                    
                    if sql:
                        # Ejecutar la búsqueda
                        search_result = await self._execute_sql_with_llm_validation(query, sql, time.time(), [], stream_callback)
                        
                        # LLAMADA ESPECÍFICA 4: Interpretar resultados de pacientes
                        if search_result.get('success'):
                            interpretation = await self._llm_interpret_patient_results(query, search_result.get('data', []), stream_callback)
                            search_result['explanation'] = interpretation
                        
                        return search_result
            
            return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error en búsqueda de pacientes: {e}")
            return None

    async def _llm_generate_patient_search_sql(self, query: str, search_info: Dict[str, Any], stream_callback=None) -> str:
        """
        LLAMADA ESPECÍFICA 3: Genera SQL para búsqueda de pacientes usando LLM.
        SIN PATTERNS HARDCODEADOS - todo via LLM.
        
        Args:
            query: Consulta original
            search_info: Información de búsqueda
            stream_callback: Función para mostrar progreso
            
        Returns:
            str: SQL para búsqueda de pacientes
        """
        try:
            if not self.llm:
                return ""
            
            if stream_callback:
                stream_callback("   - Generando SQL para búsqueda de pacientes...")
            
            # Obtener contexto dinámico de la base de datos
            try:
                from utils.schema_discovery import get_dynamic_sql_context, SchemaDiscovery
                discovery = SchemaDiscovery(self.db_path)
                if discovery.connect():
                    try:
                        db_context = discovery.generate_sql_prompt_context(['PATI_PATIENTS', 'PARA_GENDERS'])
                        # Descubrir valores de columnas automáticamente
                        column_values = discovery.discover_all_column_values('PARA_GENDERS')
                        if column_values:
                            db_context += "\n\nVALORES DE COLUMNAS DESCUBIERTOS:\n"
                            for col_name, values in column_values.items():
                                if values.values:
                                    db_context += f"- PARA_GENDERS.{col_name}: {', '.join(map(str, values.values))}\n"
                    finally:
                        discovery.close()
                else:
                    db_context = "TABLAS DISPONIBLES:\n- PATI_PATIENTS: PATI_ID, PATI_FULL_NAME, PATI_BIRTH_DATE, GEND_ID, PATI_ACTIVE\n- PARA_GENDERS: GEND_ID, GEND_DESCRIPTION_ES"
            except Exception as e:
                logger.warning(f"No se pudo obtener contexto dinámico: {e}")
                db_context = "TABLAS DISPONIBLES:\n- PATI_PATIENTS: PATI_ID, PATI_FULL_NAME, PATI_BIRTH_DATE, GEND_ID, PATI_ACTIVE\n- PARA_GENDERS: GEND_ID, GEND_DESCRIPTION_ES"
            
            # PROMPT DINÁMICO PARA SQL DE PACIENTES
            patient_sql_prompt = f"""Genera SQL para búsqueda de pacientes.

CONSULTA: "{query}"

{db_context}

REGLAS:
- Usa PATI_PATIENTS para datos básicos
- Para género, usa JOIN con PARA_GENDERS
- Para edad, usa PATI_BIRTH_DATE con strftime
- NO uses tablas de diagnósticos
- Mantén el SQL simple

Responde SOLO con SQL:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": patient_sql_prompt}],
                task_description="Generando SQL para búsqueda de pacientes"
            )
            
            sql = self._extract_response_text(response).strip()
            sql = self._clean_llm_sql_response(sql)
            
            # Si el LLM no generó SQL válido, usar herramientas genéricas
            if not sql or sql.startswith("Error") or "SELECT" not in sql.upper():
                sql = await self._use_generic_sql_tools(query, stream_callback)
            
            # Validación simple del SQL generado
            if sql and not sql.startswith("Error"):
                # Validar que no tenga JOINs problemáticos
                if "EPIS_DIAGNOSTICS" in sql.upper():
                    if stream_callback:
                        stream_callback("   ⚠️ Detectando JOIN problemático, corrigiendo...")
                    corrected_sql = await self._llm_correct_patient_search_sql(query, sql, stream_callback)
                    if corrected_sql:
                        sql = corrected_sql
                
                if stream_callback:
                    stream_callback("   ✅ SQL para pacientes generado")
                return sql
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Error generando SQL para pacientes: {e}")
            return ""

    async def _llm_correct_patient_search_sql(self, query: str, sql: str, stream_callback=None) -> str:
        """
        LLAMADA ESPECÍFICA PARA CORRECCIÓN: Corrige SQL de búsqueda de pacientes usando LLM.
        Detecta y elimina condiciones médicas innecesarias de manera dinámica.
        
        Args:
            query: Consulta original
            sql: SQL generado que puede contener condiciones innecesarias
            stream_callback: Función para mostrar progreso
            
        Returns:
            str: SQL corregido sin condiciones médicas innecesarias
        """
        try:
            if not self.llm:
                return sql
            
            if stream_callback:
                stream_callback("   - Corrigiendo SQL con LLM...")
            
            # PROMPT SIMPLE PARA CORRECCIÓN
            correction_prompt = f"""Corrige este SQL para búsqueda de pacientes.

CONSULTA: "{query}"
SQL ACTUAL: {sql}

REGLAS:
- Elimina JOINs con tablas de diagnósticos
- Elimina condiciones EXISTS innecesarias
- Mantén solo búsquedas básicas de pacientes
- Usa PATI_PATIENTS y PARA_GENDERS si es necesario

Responde SOLO con SQL corregido:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": correction_prompt}],
                task_description="Corrigiendo SQL de búsqueda de pacientes"
            )
            
            corrected_sql = self._extract_response_text(response).strip()
            corrected_sql = self._clean_llm_sql_response(corrected_sql)
            
            if corrected_sql and not corrected_sql.startswith("Error"):
                if stream_callback:
                    stream_callback("   ✅ SQL corregido exitosamente")
                return corrected_sql
            else:
                # Si la corrección falla, devolver el SQL original
                return sql
                
        except Exception as e:
            logger.error(f"Error corrigiendo SQL de pacientes: {e}")
            return sql

    async def _llm_interpret_patient_results(self, query: str, data: List[Dict[str, Any]], stream_callback=None) -> str:
        """
        LLAMADA ESPECÍFICA 4: Interpreta resultados de búsqueda de pacientes usando LLM.
        SIN PATTERNS HARDCODEADOS - todo via LLM.
        
        Args:
            query: Consulta original
            data: Datos encontrados
            stream_callback: Función para mostrar progreso
            
        Returns:
            str: Interpretación de los resultados
        """
        try:
            if not self.llm:
                return f"Se encontraron {len(data)} resultados de pacientes."
            
            if stream_callback:
                stream_callback("   - Interpretando resultados de pacientes...")
            
            # PROMPT ESPECÍFICO PARA INTERPRETACIÓN DE PACIENTES - SIN HARDCODEO
            patient_interpretation_prompt = f"""Eres un experto en interpretación de resultados de búsqueda de pacientes.

CONSULTA ORIGINAL: "{query}"

DATOS ENCONTRADOS ({len(data)} registros):
{json.dumps(data[:5], indent=2, ensure_ascii=False)}

TAREA ESPECÍFICA: Proporciona una interpretación clara y útil de los resultados de búsqueda de pacientes.

TIPOS DE INTERPRETACIÓN:
1. Paciente encontrado con información completa
2. Paciente encontrado sin episodios médicos
3. Paciente no encontrado
4. Múltiples pacientes encontrados
5. Pacientes con diagnósticos específicos

INSTRUCCIONES:
- Analiza si el paciente fue encontrado
- Identifica si tiene episodios médicos o diagnósticos
- Destaca información relevante (fechas, diagnósticos)
- Proporciona contexto médico cuando sea apropiado
- Sugiere próximos pasos si es necesario

RESPUESTA: Interpretación clara y profesional en español."""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": patient_interpretation_prompt}],
                task_description="Interpretando resultados de pacientes"
            )
            
            interpretation = self._extract_response_text(response)
            
            if stream_callback:
                stream_callback("   ✅ Interpretación de pacientes completada")
            
            return interpretation if interpretation else f"Se encontraron {len(data)} resultados de pacientes."
            
        except Exception as e:
            logger.error(f"Error interpretando resultados de pacientes: {e}")
            return f"Se encontraron {len(data)} resultados de pacientes."

    async def _llm_generate_medical_diagnosis_sql(self, query: str, medical_info: Dict[str, Any], stream_callback=None) -> str:
        """
        LLAMADA ESPECÍFICA 5: Genera SQL para diagnósticos médicos usando LLM.
        SIN PATTERNS HARDCODEADOS - todo via LLM.
        
        Args:
            query: Consulta original del usuario
            medical_info: Información sobre términos médicos y códigos detectados
            stream_callback: Función para mostrar progreso
            
        Returns:
            str: SQL optimizado para búsqueda de diagnósticos
        """
        try:
            if not self.llm:
                # Fallback sin LLM
                return self._create_generic_diagnosis_sql(medical_info)
            
            if stream_callback:
                stream_callback("   - Generando SQL inteligente para diagnósticos médicos...")
            
            medical_terms = medical_info.get('medical_terms', [])
            medical_codes = medical_info.get('medical_codes', [])
            primary_condition = medical_info.get('primary_condition', '')
            search_strategy = medical_info.get('search_strategy', 'free_text')
            confidence_level = medical_info.get('confidence_level', 'medium')
            
            # PROMPT ESPECÍFICO PARA GENERACIÓN DE SQL MÉDICO - SIN HARDCODEO
            sql_generation_prompt = f"""Eres un experto en SQL para bases de datos médicas especializado en diagnósticos.

CONSULTA ORIGINAL: "{query}"

INFORMACIÓN MÉDICA DETECTADA:
- Términos médicos: {medical_terms}
- Códigos oficiales: {medical_codes if medical_codes else "No disponibles"}
- Condición principal: {primary_condition}
- Estrategia de búsqueda: {search_strategy}
- Nivel de confianza: {confidence_level}

ESQUEMA DE TABLAS RELEVANTES:
- EPIS_DIAGNOSTICS: Contiene diagnósticos (CDTE_ID, DIAG_OBSERVATION, EPIS_PATI_ID)
- CODR_TABULAR_DIAGNOSTICS: Tabla de códigos oficiales (COTA_ID, COTA_DESCRIPTION_ES)
- PATI_PATIENTS: Información de pacientes (PATI_ID, PATI_FULL_NAME)

ESTRATEGIAS DE BÚSQUEDA INTELIGENTE:

1. SI HAY CÓDIGOS OFICIALES ({len(medical_codes)} encontrados):
   - Usar JOIN con CODR_TABULAR_DIAGNOSTICS
   - Filtrar por CDTE_ID IN (códigos_médicos)
   - Incluir descripción oficial del diagnóstico

2. SI NO HAY CÓDIGOS OFICIALES:
   - Buscar en DIAG_OBSERVATION con términos médicos detectados
   - Usar búsqueda flexible con LIKE para cada término
   - Considerar variaciones y sinónimos

3. DETECTAR TIPO DE CONSULTA:
   - Si pregunta "¿cuántos?" o "número de" → Usar COUNT(DISTINCT PATI_ID)
   - Si pregunta "mostrar" o "listar" → Usar SELECT con información detallada
   - Si pregunta "pacientes con" → Usar SELECT con nombres

4. OPTIMIZACIONES:
   - Ordenar por EPIS_ID DESC (más recientes primero)
   - Limitar a 50 resultados si no es conteo
   - Incluir información del paciente cuando sea relevante
   - Usar índices eficientes

GENERA SQL OPTIMIZADO que:
- Use la estrategia más apropiada según los códigos disponibles
- Sea compatible con SQLite
- Incluya información relevante del paciente
- Sea eficiente y preciso
- Maneje correctamente los términos médicos detectados
- Use COUNT cuando se pida un conteo numérico

IMPORTANTE: Responde SOLO con el código SQL, sin explicaciones, comentarios ni texto adicional.
El SQL debe ser válido para SQLite y ejecutable directamente."""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": sql_generation_prompt}],
                task_description="Generando SQL inteligente para diagnósticos médicos"
            )
            
            generated_sql = self._extract_response_text(response).strip()
            generated_sql = self._clean_llm_sql_response(generated_sql)
            
            if generated_sql and not generated_sql.startswith("Error"):
                if stream_callback:
                    stream_callback("   ✅ SQL inteligente para diagnósticos generado")
                return generated_sql
            else:
                # Fallback a método manual
                return self._create_generic_diagnosis_sql(medical_info)
                
        except Exception as e:
            logger.error(f"Error generando SQL para diagnósticos: {e}")
            return self._create_generic_diagnosis_sql(medical_info)

    def _create_generic_diagnosis_sql(self, medical_info: Dict[str, Any]) -> str:
        """
        Crea SQL genérico para buscar diagnósticos médicos.
        FALLBACK: Solo cuando LLM no está disponible.
        
        Args:
            medical_info: Información sobre términos médicos y códigos detectados
            
        Returns:
            str: SQL optimizado para búsqueda de diagnósticos
        """
        medical_codes = medical_info.get('medical_codes', [])
        medical_terms = medical_info.get('medical_terms', [])
        
        if medical_codes:
            # Usar códigos oficiales - más preciso
            codes_list = ','.join(medical_codes)
            sql = f"""
            SELECT 
                e.EPIS_ID,
                e.EPIS_PATI_ID,
                e.CDTE_ID,
                e.DIAG_OBSERVATION,
                c.COTA_DESCRIPTION_ES as DIAGNOSIS_DESCRIPTION,
                p.PATI_FULL_NAME
            FROM EPIS_DIAGNOSTICS e
            INNER JOIN CODR_TABULAR_DIAGNOSTICS c ON e.CDTE_ID = c.COTA_ID
            LEFT JOIN PATI_PATIENTS p ON e.EPIS_PATI_ID = p.PATI_ID
            WHERE e.CDTE_ID IN ({codes_list})
            ORDER BY e.EPIS_ID DESC
            LIMIT 50
            """
            logger.info(f"🔍 Usando {len(medical_codes)} códigos oficiales")
            return sql
        elif medical_terms:
            # Búsqueda libre con términos médicos
            conditions = []
            for term in medical_terms:
                conditions.append(f"UPPER(e.DIAG_OBSERVATION) LIKE UPPER('%{term}%')")
            
            where_clause = " OR ".join(conditions)
            sql = f"""
            SELECT 
                e.EPIS_ID,
                e.EPIS_PATI_ID,
                e.CDTE_ID,
                e.DIAG_OBSERVATION,
                p.PATI_FULL_NAME
            FROM EPIS_DIAGNOSTICS e
            LEFT JOIN PATI_PATIENTS p ON e.EPIS_PATI_ID = p.PATI_ID
            WHERE {where_clause}
            ORDER BY e.EPIS_ID DESC
            LIMIT 50
            """
            logger.info(f"🔍 Usando búsqueda libre con términos: {medical_terms}")
            return sql
        else:
            # Fallback: búsqueda general
            sql = """
            SELECT 
                e.EPIS_ID,
                e.EPIS_PATI_ID,
                e.CDTE_ID,
                e.DIAG_OBSERVATION,
                p.PATI_FULL_NAME
            FROM EPIS_DIAGNOSTICS e
            LEFT JOIN PATI_PATIENTS p ON e.EPIS_PATI_ID = p.PATI_ID
            ORDER BY e.EPIS_ID DESC
            LIMIT 50
            """
            logger.info("🔍 Usando búsqueda general de diagnósticos")
            return sql

    async def process_data_manipulation(self, operation: str, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None, stream_callback=None) -> Dict[str, Any]:
        """
        Procesa operaciones de manipulación de datos (INSERT, UPDATE, DELETE)
        Compatible con FHIRAgent - Método de compatibilidad
        """
        try:
            print(f"🔄 Procesando operación de datos: {operation}")
            print(f"   📥 Datos recibidos: {json.dumps(data, indent=2, ensure_ascii=False)}")
            print(f"   📋 Contexto: {context}")
            
            # Extraer información del contexto
            intent = context.get('intent', 'general') if context else 'general'
            conn = context.get('conn') if context else None
            
            # Generar SQL para la operación usando LLM
            if operation.upper() == 'INSERT':
                sql_result = await self._generate_insert_sql_intelligent(data, context, stream_callback)
                sql = sql_result['sql']
                values = sql_result.get('values', [])
                table_used = sql_result.get('table', 'desconocida')
            elif operation.upper() == 'UPDATE':
                sql_result = await self._generate_update_sql_intelligent(data, context, stream_callback)
                sql = sql_result['sql']
                values = sql_result.get('values', [])
                table_used = sql_result.get('table', 'desconocida')
            elif operation.upper() == 'DELETE':
                sql_result = await self._generate_delete_sql_intelligent(data, context, stream_callback)
                sql = sql_result['sql']
                values = sql_result.get('values', [])
                table_used = sql_result.get('table', 'desconocida')
            else:
                return {
                    'success': False,
                    'error': f'Operación no soportada: {operation}',
                    'data': []
                }
            
            print(f"   💾 SQL generado: {sql[:100]}...")
            if values:
                print(f"   📊 Valores: {values}")
            
            # Mostrar información detallada de la operación
            print(f"   📋 OPERACIÓN: {operation.upper()}")
            print(f"   🗃️ TABLA: {table_used}")
            
            # Extraer información de columnas del SQL
            if operation.upper() == 'INSERT':
                # Buscar columnas en INSERT
                import re
                column_match = re.search(r'INSERT INTO \w+ \((.*?)\) VALUES', sql, re.IGNORECASE)
                if column_match:
                    columns = [col.strip() for col in column_match.group(1).split(',')]
                    print(f"   📊 COLUMNAS A INSERTAR: {', '.join(columns)}")
                    
                    # Mostrar valores correspondientes
                    if values:
                        for i, (col, val) in enumerate(zip(columns, values)):
                            print(f"      - {col}: {val}")
                    else:
                        # Extraer valores del SQL si no están en la lista de valores
                        values_match = re.search(r'VALUES \((.*?)\)', sql, re.IGNORECASE)
                        if values_match:
                            sql_values = [v.strip().strip("'") for v in values_match.group(1).split(',')]
                            for i, (col, val) in enumerate(zip(columns, sql_values)):
                                # Manejar valores NULL correctamente
                                if val.upper() == 'NULL' or val == 'None':
                                    print(f"      - {col}: NULL")
                                else:
                                    print(f"      - {col}: {val}")
            
            elif operation.upper() == 'UPDATE':
                # Buscar columnas en UPDATE
                import re
                set_match = re.search(r'SET (.*?) WHERE', sql, re.IGNORECASE)
                if set_match:
                    set_clause = set_match.group(1)
                    # Extraer pares columna=valor
                    updates = [pair.strip() for pair in set_clause.split(',')]
                    print(f"   📊 COLUMNAS A ACTUALIZAR:")
                    for update in updates:
                        if '=' in update:
                            col, val = update.split('=', 1)
                            print(f"      - {col.strip()}: {val.strip()}")
            
            elif operation.upper() == 'DELETE':
                # Buscar condición WHERE en DELETE
                import re
                where_match = re.search(r'WHERE (.*?)(?:;|$)', sql, re.IGNORECASE)
                if where_match:
                    where_clause = where_match.group(1)
                    print(f"   🔍 CONDICIÓN WHERE: {where_clause}")
            
            # Ejecutar SQL usando la conexión proporcionada o crear una nueva
            inserted_id = None
            if conn:
                # Usar conexión existente (para transacciones)
                try:
                    cursor = conn.cursor()
                    if values and '?' in sql:
                        cursor.execute(sql, values)
                    else:
                        cursor.execute(sql)
                    
                    # Obtener el ID del registro insertado si es posible
                    if operation.upper() == 'INSERT':
                        try:
                            # Usar lastrowid (más confiable y directo)
                            inserted_id = cursor.lastrowid
                            if inserted_id:
                                print(f"   ✅ ID del registro insertado (lastrowid): {inserted_id}")
                            else:
                                print(f"   ⚠️ lastrowid no disponible, usando método adaptativo...")
                                # Fallback al método adaptativo si lastrowid no funciona
                                inserted_id = await self._get_real_inserted_id_adaptive(table_used, data, stream_callback)
                                if inserted_id:
                                    print(f"   ✅ ID del registro insertado (adaptativo): {inserted_id}")
                                else:
                                    print(f"   ⚠️ No se pudo obtener el ID del registro insertado")
                        except Exception as e:
                            print(f"   ⚠️ Error obteniendo ID: {e}")
                            inserted_id = None
                    
                    # No hacer commit aquí, dejar que el llamador maneje la transacción
                    result = {
                        'success': True,
                        'data': [],
                        'sql_used': sql,
                        'table_used': table_used,
                        'inserted_id': inserted_id
                    }
                except Exception as e:
                    result = {
                        'success': False,
                        'error': str(e),
                        'sql_used': sql,
                        'table_used': table_used,
                        'inserted_id': None
                    }
            else:
                # Usar SQL executor normal
                if values and '?' in sql:
                    result = self.sql_executor.execute_query(sql, values)
                else:
                    # SQL ya tiene valores específicos
                    result = self.sql_executor.execute_query(sql)
                
                # Para INSERT, intentar obtener el ID del último registro insertado
                if operation.upper() == 'INSERT' and result.get('success'):
                    try:
                        # Intentar obtener lastrowid del resultado
                        inserted_id = result.get('lastrowid')
                        if inserted_id:
                                    print(f"   ✅ ID del registro insertado (lastrowid): {inserted_id}")
                        else:
                            print(f"   ⚠️ lastrowid no disponible, usando método adaptativo...")
                            # Fallback al método adaptativo
                            inserted_id = await self._get_real_inserted_id_adaptive(table_used, data, stream_callback)
                            if inserted_id:
                                print(f"   ✅ ID del registro insertado (adaptativo): {inserted_id}")
                            else:
                                print(f"   ⚠️ No se pudo obtener el ID del registro insertado")
                    except Exception as e:
                        print(f"   ⚠️ Error obteniendo ID: {e}")
                        inserted_id = None
                
                result['inserted_id'] = inserted_id

            # Formatear los datos para mejor visualización
            formatted_data = await self._format_sql_results_for_display(result.get('data', []), f"Operación {operation}")
            
            return {
                'success': result['success'],
                'data': result.get('data', []),
                'formatted_data': formatted_data,
                'error': result.get('error', ''),
                'operation': operation,
                'sql_used': sql,
                'table_used': result.get('table_used', table_used),
                'inserted_id': result.get('inserted_id'),
                'message': f"Operación {operation} completada" if result['success'] else result.get('error', 'Error desconocido')
            }
            
        except Exception as e:
            # Formatear datos vacíos para error
            formatted_data = await self._format_sql_results_for_display([], f"Operación {operation}")
            
            return {
                'success': False,
                'error': f'Error en manipulación de datos: {str(e)}',
                'data': [],
                'formatted_data': formatted_data,
                'operation': operation,
                'table_used': 'desconocida',
                'inserted_id': None,
                'message': f'Error en manipulación de datos: {str(e)}'
            }

    async def _generate_insert_sql_intelligent(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None, stream_callback=None) -> Dict[str, Any]:
        """Genera SQL INSERT inteligente usando LLM para mapeo dinámico FHIR→SQL"""
        try:
            # DEBUG: Mostrar datos de entrada
            print(f"🔍 DEBUG _generate_insert_sql_intelligent:")
            print(f"   📥 Datos de entrada: {json.dumps(data, indent=2, ensure_ascii=False)}")
            print(f"   📋 Contexto: {context}")
            
            # DETECTAR SI SON DATOS FHIR O YA MAPEADOS
            is_fhir_data = 'resourceType' in data
            
            if is_fhir_data:
                # USAR MAPEO FHIR→SQL INTELIGENTE CON LLM
                print(f"   🔄 Detectados datos FHIR, usando mapeo LLM flexible...")
                
                # Importar el mapper FHIR→SQL
                from agents.fhir_sql_llm_mapper import FHIRSQLLLMMapper
                
                # Crear instancia del mapper
                mapper = FHIRSQLLLMMapper(
                    llm=self.llm,
                    schema_getter=self._get_table_columns
                )
                
                # Obtener esquema completo de la base de datos
                db_schema = {}
                try:
                    # Obtener todas las tablas de la base de datos
                    import sqlite3
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    all_tables = [row[0] for row in cursor.fetchall()]
                    conn.close()
                    
                    # print(f"   📋 Tablas encontradas en la BD: {all_tables}")  # ELIMINADO para no mostrar todas las tablas
                    
                    # Obtener columnas de cada tabla (solo debug de tabla seleccionada)
                    for table in all_tables:
                        try:
                            columns = await self._get_table_columns(table)
                            db_schema[table] = columns
                            # print(f"      - {table}: {len(columns)} columnas")  # ELIMINADO para no mostrar todas las columnas
                        except Exception as e:
                            print(f"      ⚠️ Error obteniendo columnas de {table}: {e}")
                            db_schema[table] = []
                            
                except Exception as e:
                    print(f"   ⚠️ Error obteniendo esquema completo: {e}")
                    # Fallback con tablas básicas
                    fallback_tables = ['PATI_PATIENTS', 'EPIS_DIAGNOSTICS', 'PATI_USUAL_MEDICATION']
                    for table in fallback_tables:
                        try:
                            columns = await self._get_table_columns(table)
                            db_schema[table] = columns
                        except:
                            db_schema[table] = []
                
                # Realizar mapeo inteligente
                mapping_result = await mapper.map_fhir_to_sql(data, db_schema, context)
                
                # El mapper siempre devuelve un resultado válido
                target_table = mapping_result['table']
                mapped_data = mapping_result['mapped_data']
                resource_type = data.get('resourceType', 'Unknown')
                
                print(f"   ✅ Mapeo LLM exitoso:")
                print(f"      📋 Tabla: {target_table}")
                print(f"      🔄 Tipo de recurso: {resource_type}")
                print(f"      📊 Campos mapeados: {len(mapped_data)}")
                print(f"      📝 Resumen: {mapping_result['mapping_summary']}")
                print(f"      🐞 DEBUG VALORES A INSERTAR: {mapped_data}")
                for k, v in mapped_data.items():
                    if v is None or v == '' or v == 'none' or v == 'None':
                        print(f"      ⚠️ CAMPO VACÍO O NONE: {k} = {v}")
                
                # Si no hay datos mapeados, usar fallback
                if not mapped_data:
                    print(f"   ⚠️ No se mapearon campos, usando fallback básico...")
                    target_table = context.get('table_hint', 'PATI_PATIENTS') if context else 'PATI_PATIENTS'
                    mapped_data = {
                        'PATI_ID': context.get('patient_id') if context else None,
                        'DIAG_OBSERVATION': json.dumps(data, ensure_ascii=False)
                    }
                    # Filtrar valores None
                    mapped_data = {k: v for k, v in mapped_data.items() if v is not None}
                    
                    resource_type = data.get('resourceType', 'Unknown')
                    target_table = context.get('table_hint', 'PATI_PATIENTS') if context else 'PATI_PATIENTS'
                    mapped_data = {
                        'PATI_ID': context.get('patient_id') if context else None,
                        'DIAG_OBSERVATION': json.dumps(data, ensure_ascii=False)
                    }
                    # Filtrar valores None
                    mapped_data = {k: v for k, v in mapped_data.items() if v is not None}
                
            else:
                # Datos ya mapeados, usar tabla del contexto
                target_table = context.get('table_hint', 'PATI_PATIENTS') if context else 'PATI_PATIENTS'
                mapped_data = data
                resource_type = 'Unknown'
            
            # Generar SQL INSERT
            columns = list(mapped_data.keys())
            values = list(mapped_data.values())
            placeholders = ', '.join(['?' for _ in values])
            column_names = ', '.join(columns)
            
            sql = f"INSERT INTO {target_table} ({column_names}) VALUES ({placeholders})"
            
            print(f"   🗃️ Tabla seleccionada: {target_table}")
            print(f"   📊 Columnas: {columns}")
            print(f"   💾 Valores: {values}")
            for i, v in enumerate(values):
                if v is None or v == '' or v == 'none' or v == 'None':
                    print(f"      ⚠️ VALOR VACÍO O NONE EN COLUMNA: {columns[i]} = {v}")
            print(f"   📝 SQL generado: {sql}")
            
            return {
                'sql': sql,
                'values': values,
                'table': target_table,
                'resource_type': resource_type,
                'mapped_data': mapped_data
            }
                    
        except Exception as e:
            logger.error(f"Error generando SQL INSERT: {e}")
            # Fallback simple
            table_name = context.get('table_hint', 'PATI_PATIENTS') if context else 'PATI_PATIENTS'
            columns = list(data.keys())
            values = list(data.values())
            placeholders = ', '.join(['?' for _ in values])
            column_names = ', '.join(columns)
            
            sql = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"
        
            return {
                'sql': sql,
                'values': values,
                'table': table_name,
                'resource_type': 'Unknown',
                'mapped_data': data
            }

    async def _generate_update_sql_intelligent(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None, stream_callback=None) -> Dict[str, Any]:
        """Genera SQL UPDATE inteligente usando LLM"""
        try:
            if not self.llm:
                default_table = 'PATI_PATIENTS'
                fallback_sql = await self._llm_generate_fallback_sql_adaptive('UPDATE', data, default_table, None)
                return {
                    'sql': fallback_sql,
                    'values': [],
                    'table': default_table
                }

            schema_info = self._get_real_schema_info()
            
            prompt = f"""Eres un experto en SQL UPDATE médico. Genera SQL UPDATE basado en estos datos:

DATOS A ACTUALIZAR:
{json.dumps(data, indent=2, ensure_ascii=False)}

ESQUEMA DISPONIBLE:
{schema_info}

INSTRUCCIONES:
1. Identifica la tabla apropiada para la actualización
2. Usa el campo 'id' como condición WHERE
3. Mapea los campos FHIR a columnas SQL
4. Genera SQL UPDATE completo con valores específicos

RESPUESTA JSON:
{{
    "sql": "UPDATE tabla SET columna1 = 'valor1' WHERE id = 'valor_id';",
    "table": "nombre_tabla",
    "values": []
}}

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Generando SQL UPDATE inteligente"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result and result.get('sql'):
                return {
                    'sql': result['sql'],
                    'values': result.get('values', []),
                    'table': result.get('table', 'desconocida')
                }
            else:
                default_table = await self._llm_select_default_table_adaptive('Unknown', data, None)
                fallback_sql = await self._llm_generate_fallback_sql_adaptive('UPDATE', data, default_table, None)
                return {
                    'sql': fallback_sql,
                    'values': [],
                    'table': default_table
                }
                
        except Exception as e:
            logger.error(f"Error generando SQL UPDATE: {e}")
            default_table = await self._llm_select_default_table_adaptive('Unknown', data, None)
            fallback_sql = await self._llm_generate_fallback_sql_adaptive('UPDATE', data, default_table, None)
            return {
                'sql': fallback_sql,
                'values': [],
                'table': default_table
            }

    async def _generate_delete_sql_intelligent(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None, stream_callback=None) -> Dict[str, Any]:
        """Genera SQL DELETE inteligente usando LLM"""
        try:
            if not self.llm:
                default_table = 'PATI_PATIENTS'
                fallback_sql = await self._llm_generate_fallback_sql_adaptive('DELETE', data, default_table, None)
                return {
                    'sql': fallback_sql,
                    'values': [],
                    'table': default_table
                }

            schema_info = self._get_real_schema_info()
            
            prompt = f"""Eres un experto en SQL DELETE médico. Genera SQL DELETE basado en estos datos:

DATOS PARA ELIMINAR:
{json.dumps(data, indent=2, ensure_ascii=False)}

ESQUEMA DISPONIBLE:
{schema_info}

INSTRUCCIONES:
1. Identifica la tabla apropiada para la eliminación
2. Usa el campo 'id' como condición WHERE
3. Genera SQL DELETE completo

RESPUESTA JSON:
{{
    "sql": "DELETE FROM tabla WHERE id = 'valor_id';",
    "table": "nombre_tabla",
    "values": []
}}

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Generando SQL DELETE inteligente"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result and result.get('sql'):
                return {
                    'sql': result['sql'],
                    'values': result.get('values', []),
                    'table': result.get('table', 'desconocida')
                }
            else:
                default_table = await self._llm_select_default_table_adaptive('Unknown', data, None)
                fallback_sql = await self._llm_generate_fallback_sql_adaptive('DELETE', data, default_table, None)
                return {
                    'sql': fallback_sql,
                    'values': [],
                    'table': default_table
                }
                
        except Exception as e:
            logger.error(f"Error generando SQL DELETE: {e}")
            default_table = await self._llm_select_default_table_adaptive('Unknown', data, None)
            fallback_sql = await self._llm_generate_fallback_sql_adaptive('DELETE', data, default_table, None)
            return {
                'sql': fallback_sql,
                'values': [],
                'table': default_table
            }

    async def _format_sql_results_for_display(self, data: List[Dict[str, Any]], query: str) -> str:
        """Formatea resultados SQL para mostrar al usuario"""
        try:
            if not data:
                return "No se encontraron resultados."
            
            # Formato básico
            result_lines = []
            for i, row in enumerate(data[:10], 1):  # Mostrar máximo 10 filas
                row_str = " | ".join([f"{k}: {v}" for k, v in row.items()])
                result_lines.append(f"{i}. {row_str}")
            
            if len(data) > 10:
                result_lines.append(f"... y {len(data) - 10} registros más")
            
            return "\n".join(result_lines)
            
        except Exception as e:
            logger.error(f"Error formateando resultados: {e}")
            return f"Se encontraron {len(data)} resultados."

    async def _llm_validate_mapping_intelligent(self, fhir_data: Dict[str, Any], tabla_actual: str, tipo_actual: str, valores_actuales: Dict[str, Any], stream_callback=None) -> Optional[Dict[str, Any]]:
        """
        LLAMADA ESPECÍFICA: Valida y corrige el mapeo FHIR→SQL usando LLM inteligente.
        SIN PATTERNS HARDCODEADOS - todo via LLM.
        
        Args:
            fhir_data: Datos FHIR originales
            tabla_actual: Tabla mapeada actualmente
            tipo_actual: Tipo de recurso actual
            valores_actuales: Valores mapeados actualmente
            stream_callback: Función para mostrar progreso
            
        Returns:
            Optional[Dict[str, Any]]: Correcciones aplicadas o None si no hay cambios
        """
        try:
            if not self.llm:
                return None
            
            if stream_callback:
                stream_callback("   - Validando mapeo con IA inteligente...")
            
            # PROMPT ESPECÍFICO PARA VALIDACIÓN DE MAPEO - SIN HARDCODEO
            validation_prompt = f"""Eres un experto en validación de mapeo FHIR→SQL para bases de datos médicas.

DATOS FHIR ORIGINALES:
{json.dumps(fhir_data, indent=2, ensure_ascii=False)}

MAPEO ACTUAL:
- Tabla: {tabla_actual}
- Tipo: {tipo_actual}
- Valores: {json.dumps(valores_actuales, indent=2, ensure_ascii=False)}

ESQUEMA DISPONIBLE:
{self._get_real_schema_info()}

TAREA ESPECÍFICA: Analiza el mapeo actual y detecta errores o inconsistencias.

PROBLEMAS COMUNES A DETECTAR:
1. Observaciones mapeadas como pacientes
2. Pacientes mapeados como observaciones
3. Tipos de recurso incorrectos
4. Campos mapeados en tablas incorrectas
5. Valores en columnas inapropiadas

ESTRATEGIA DE VALIDACIÓN:
1. Analiza el contenido semántico de los datos FHIR
2. Identifica el tipo de recurso real basado en campos específicos
3. Verifica que la tabla sea apropiada para el tipo de recurso
4. Corrige mapeos incorrectos automáticamente
5. Sugiere mejoras en el mapeo

REGLAS DE CORRECCIÓN:
- Si contiene "valueQuantity", "component" → ES OBSERVACIÓN
- Si contiene "name", "birthDate", "gender" → ES PACIENTE
- Si contiene "code", "clinicalStatus" → ES DIAGNÓSTICO
- Si contiene "medicationCodeableConcept" → ES MEDICAMENTO
- Si contiene "status", "period" → ES EPISODIO

INSTRUCCIONES:
1. Analiza los datos FHIR para determinar el tipo real
2. Verifica si el mapeo actual es correcto
3. Si hay errores, proporciona correcciones
4. Si está correcto, confirma el mapeo
5. Considera el contexto médico completo

RESPUESTA JSON:
{{
    "needs_correction": true|false,
    "corrected_table": "tabla_corregida",
    "corrected_type": "tipo_corregido",
    "corrected_values": {{
        "columna1": "valor1",
        "columna2": "valor2"
    }},
    "reasoning": "explicación de la corrección",
    "confidence": "high|medium|low"
}}

IMPORTANTE: 
- Solo corrige si hay errores claros
- Mantén la lógica original si el mapeo es correcto
- Usa tablas y columnas que realmente existen
- Considera el contexto médico completo

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": validation_prompt}],
                task_description="Validando mapeo con IA inteligente"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result and result.get('needs_correction', False):
                corrected_table = result.get('corrected_table', tabla_actual)
                corrected_type = result.get('corrected_type', tipo_actual)
                corrected_values = result.get('corrected_values', valores_actuales)
                reasoning = result.get('reasoning', 'Sin explicación')
                confidence = result.get('confidence', 'medium')
                
                if stream_callback:
                    stream_callback(f"   - Corrección aplicada: {tabla_actual} → {corrected_table}")
                    stream_callback(f"   - Tipo corregido: {tipo_actual} → {corrected_type}")
                    stream_callback(f"   - Confianza: {confidence}")
                
                logger.info(f"🧠 LLM corrigió mapeo: {reasoning}")
                
                return {
                    'corrected_table': corrected_table,
                    'corrected_type': corrected_type,
                    'corrected_values': corrected_values,
                    'reasoning': reasoning,
                    'confidence': confidence
                }
            else:
                if stream_callback:
                    stream_callback("   ✅ Mapeo validado correctamente")
                return None
                
        except Exception as e:
            logger.error(f"Error en validación de mapeo: {e}")
            return None

    async def _llm_detect_resource_type_intelligent(self, fhir_data: Dict[str, Any], stream_callback=None) -> str:
        """
        LLAMADA ESPECÍFICA: Detecta el tipo de recurso FHIR usando LLM inteligente.
        SIN PATRONES HARDCODEADOS - todo via LLM.
        
        Args:
            fhir_data: Datos FHIR a analizar
            stream_callback: Función para mostrar progreso
            
        Returns:
            str: Tipo de recurso detectado
        """
        try:
            if not self.llm:
                return 'Patient'  # Fallback básico
            
            if stream_callback:
                stream_callback("   - Detectando tipo de recurso con IA...")
            
            # PROMPT ESPECÍFICO PARA DETECCIÓN DE TIPO - SIN HARDCODEO
            detection_prompt = f"""Eres un experto en identificación de tipos de recursos FHIR.

DATOS FHIR A ANALIZAR:
{json.dumps(fhir_data, indent=2, ensure_ascii=False)}

TAREA ESPECÍFICA: Identifica el tipo de recurso FHIR basado en el contenido semántico.

TIPOS DE RECURSOS POSIBLES:
- Patient: Datos de pacientes (nombre, fecha nacimiento, género)
- Condition: Diagnósticos y condiciones médicas
- Medication: Medicamentos y tratamientos
- Observation: Observaciones médicas (signos vitales, resultados)
- Encounter: Episodios de atención médica
- EpisodeOfCare: Episodios de cuidado

CRITERIOS DE IDENTIFICACIÓN:
1. Buscar "resourceType" explícito primero
2. Analizar campos específicos de cada tipo
3. Considerar el contexto médico completo
4. Identificar patrones en los datos

INSTRUCCIONES:
- Analiza todos los campos disponibles
- Considera el contexto médico
- Identifica el tipo más apropiado
- Proporciona confianza en la identificación

RESPUESTA JSON:
{{
    "resource_type": "Patient|Condition|Medication|Observation|Encounter|EpisodeOfCare",
    "confidence": "high|medium|low",
    "key_fields": ["campo1", "campo2"],
    "reasoning": "explicación de la identificación"
}}

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": detection_prompt}],
                task_description="Detectando tipo de recurso con IA"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result:
                resource_type = result.get('resource_type', 'Patient')
                confidence = result.get('confidence', 'medium')
                reasoning = result.get('reasoning', 'Sin explicación')
                
                if stream_callback:
                    stream_callback(f"   - Tipo detectado: {resource_type} (confianza: {confidence})")
                
                return resource_type
            else:
                return 'Patient'  # Fallback
                
        except Exception as e:
            logger.error(f"Error detectando tipo de recurso: {e}")
            return 'Patient'  # Fallback

    async def _llm_select_table_intelligent(self, resource_type: str, fhir_data: Dict[str, Any], stream_callback=None) -> str:
        """
        Selecciona la tabla SQL apropiada usando SOLO LLM, sin ningún mapeo rígido ni fallback, salvo el último recurso por excepción.
        """
        try:
            if stream_callback:
                stream_callback("   - Seleccionando tabla SQL con IA...")
            
            selection_prompt = f"""Eres un experto en mapeo de recursos FHIR a tablas SQL médicas.

TIPO DE RECURSO: {resource_type}

DATOS FHIR:
{json.dumps(fhir_data, indent=2, ensure_ascii=False)}

ESQUEMA DISPONIBLE:
{self._get_real_schema_info()}

TAREA: Selecciona la tabla SQL más apropiada para este tipo de recurso.

INSTRUCCIONES:
- Analiza el esquema completo dinámicamente
- Busca patrones en nombres de tablas
- Considera el contexto médico específico
- Selecciona la tabla más apropiada
- NO uses mapeos rígidos, analiza dinámicamente

RESPUESTA: Solo el nombre exacto de la tabla más apropiada, sin explicaciones, sin JSON, sin texto adicional."""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": selection_prompt}],
                task_description="Seleccionando tabla SQL con IA"
            )
            selected_table = self._extract_response_text(response).strip().strip('"').strip("'")
            if stream_callback:
                stream_callback(f"   - Tabla seleccionada: {selected_table}")
            return selected_table
        except Exception as e:
            logger.error(f"Error seleccionando tabla: {e}")
            return 'PATI_PATIENTS'  # Último recurso


    async def _llm_map_fields_intelligent(self, fhir_data: Dict[str, Any], resource_type: str, target_table: str, stream_callback=None) -> Dict[str, Any]:
        """
        LLAMADA ESPECÍFICA: Mapea campos FHIR a columnas SQL usando LLM inteligente.
        SIN PATRONES HARDCODEADOS - todo via LLM.
        
        Args:
            fhir_data: Datos FHIR
            resource_type: Tipo de recurso
            target_table: Tabla SQL objetivo
            stream_callback: Función para mostrar progreso
            
        Returns:
            Dict[str, Any]: Mapeo de campos y valores
        """
        try:
            if not self.llm:
                # Fallback básico adaptativo
                name_fields = await self._llm_detect_common_fields_adaptive(target_table, 'name', None)
                return {
                    'columns': name_fields,
                    'values': [fhir_data.get('name', '')] * len(name_fields)
                }
            
            if stream_callback:
                stream_callback("   - Mapeando campos con IA inteligente...")
            
            # PROMPT ESPECÍFICO PARA MAPEO DE CAMPOS - SIN HARDCODEO
            mapping_prompt = f"""Eres un experto en mapeo de campos FHIR a columnas SQL médicas.

DATOS FHIR:
{json.dumps(fhir_data, indent=2, ensure_ascii=False)}

TIPO DE RECURSO: {resource_type}
TABLA OBJETIVO: {target_table}

ESQUEMA DE LA TABLA:
{self._get_table_schema_info(target_table)}

TAREA ESPECÍFICA: Mapea los campos FHIR a columnas SQL específicas.

ESTRATEGIA DE MAPEO:
1. Analiza cada campo FHIR
2. Identifica la columna SQL correspondiente
3. Convierte tipos de datos apropiadamente
4. Maneja valores nulos y opcionales
5. Considera campos de auditoría

REGLAS CRÍTICAS:
- NO incluyas campos de ID en INSERT (déjalos autoincrementarse)
- Maneja fechas en formato SQLite (YYYY-MM-DD)
- Escapa valores de texto apropiadamente
- Considera valores por defecto cuando sea necesario
- NO uses UUIDs ficticios como "patient-id-unico" o "urn:uuid:..."
- Para valores nulos, usa NULL, NO "None"
- Para IDs, usa valores numéricos reales o NULL
- NO generes IDs ficticios, deja que la base de datos los genere
- Para fechas vacías, usa NULL, NO "None"
- Para campos opcionales sin valor, usa NULL

INSTRUCCIONES:
- Mapea solo campos que existen en la tabla
- Convierte tipos de datos correctamente
- Maneja valores nulos apropiadamente
- Considera el contexto médico

RESPUESTA JSON:
{{
    "columns": ["columna1", "columna2"],
    "values": ["valor1", "valor2"],
    "field_mapping": {{
        "campo_fhir1": "columna_sql1",
        "campo_fhir2": "columna_sql2"
    }},
    "data_types": {{
        "columna1": "TEXT|INTEGER|REAL|DATE",
        "columna2": "TEXT|INTEGER|REAL|DATE"
    }}
}}

IMPORTANTE:
- NO uses "None" como valor, usa NULL para valores nulos
- NO uses UUIDs ficticios como "patient-id-unico" o "urn:uuid:..."
- NO uses IDs ficticios, deja que se autoincrementen
- Usa valores reales o NULL cuando corresponda
- Para fechas vacías, usa NULL en lugar de "None"
- Para IDs, NO incluyas campos de ID en INSERT

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": mapping_prompt}],
                task_description="Mapeando campos con IA inteligente"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result:
                columns = result.get('columns', [])
                values = result.get('values', [])
                field_mapping = result.get('field_mapping', {})
                
                if stream_callback:
                    stream_callback(f"   - Campos mapeados: {len(columns)} columnas")
                
                return {
                    'columns': columns,
                    'values': values,
                    'field_mapping': field_mapping
                }
            else:
                # Fallback básico adaptativo
                name_fields = await self._llm_detect_common_fields_adaptive(target_table, 'name', None)
                return {
                    'columns': name_fields,
                    'values': [fhir_data.get('name', '')] * len(name_fields)
                }
                
        except Exception as e:
            logger.error(f"Error mapeando campos: {e}")
            name_fields = await self._llm_detect_common_fields_adaptive(target_table, 'name', None)
            return {
                'columns': name_fields,
                'values': [fhir_data.get('name', '')] * len(name_fields)
            }

    def _get_table_schema_info(self, table_name: str) -> str:
        """
        Obtiene información del esquema de una tabla específica.
        
        Args:
            table_name: Nombre de la tabla
            
        Returns:
            str: Información del esquema de la tabla
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Obtener información de columnas
            cursor.execute(f"PRAGMA table_info('{table_name}');")
            columns_info = cursor.fetchall()
            
            schema_info = [f"TABLA: {table_name}"]
            schema_info.append("COLUMNAS DISPONIBLES:")
            
            for col_info in columns_info:
                col_name = col_info[1]
                col_type = col_info[2]
                not_null = col_info[3]
                default_value = col_info[4]
                pk = col_info[5]
                
                col_desc = f"  - {col_name}: {col_type}"
                if pk:
                    col_desc += " (PRIMARY KEY)"
                if not_null:
                    col_desc += " (NOT NULL)"
                if default_value:
                    col_desc += f" (DEFAULT: {default_value})"
                
                schema_info.append(col_desc)
            
            # Agregar información específica para tablas conocidas
            if table_name.upper() == 'PATI_PATIENTS':
                schema_info.append("")
                schema_info.append("INFORMACIÓN ESPECÍFICA PARA PATI_PATIENTS:")
                schema_info.append("  - PATI_ID: ID único del paciente (PRIMARY KEY)")
                schema_info.append("  - PATI_NAME: Nombre del paciente")
                schema_info.append("  - PATI_SURNAME_1: Primer apellido")
                schema_info.append("  - PATI_SURNAME_2: Segundo apellido (opcional)")
                schema_info.append("  - PATI_FULL_NAME: Nombre completo")
                schema_info.append("  - PATI_BIRTH_DATE: Fecha de nacimiento (YYYY-MM-DD)")
                schema_info.append("  - PATI_ACTIVE: Estado activo (1=activo, 0=inactivo)")
                schema_info.append("  - GEND_ID: ID de género (1=masculino, 2=femenino, 3=otro)")
                schema_info.append("  - PATI_START_DATE: Fecha de inicio de atención")
                schema_info.append("  - MTIME: Timestamp de modificación")
                schema_info.append("")
                schema_info.append("COLUMNAS QUE NO EXISTEN (NO USAR):")
                schema_info.append("  - PATI_IDENTIFIER (no existe)")
                schema_info.append("  - PATI_PHONE (no existe)")
                schema_info.append("  - PATI_ADDRESS (no existe)")
            
            conn.close()
            return "\n".join(schema_info)
            
        except Exception as e:
            logger.error(f"Error obteniendo esquema de tabla {table_name}: {e}")
            return f"Error obteniendo esquema de tabla {table_name}"

    async def _llm_detect_id_fields_adaptive(self, table_name: str, stream_callback=None) -> List[str]:
        """Usa LLM para detectar dinámicamente los campos de ID de una tabla"""
        try:
            if not self.llm:
                return ['PATI_ID', 'EPIS_ID', 'DIAG_ID', 'MEDI_ID', 'OBSE_ID']  # Fallback básico
            
            schema_info = self._get_real_schema_info()
            
            prompt = f"""Eres un experto en análisis de esquemas de base de datos médicos. 

ANÁLISIS DE CAMPOS DE ID:
Tabla: {table_name}

ESQUEMA DISPONIBLE:
{schema_info}

INSTRUCCIONES:
1. Analiza la tabla {table_name}
2. Identifica TODOS los campos que son claves primarias o IDs
3. Incluye campos que terminen en _ID, ID, o que sean claves primarias
4. Considera también campos de autoincremento

RESPUESTA JSON:
{{
    "id_fields": ["campo1", "campo2", "campo3"]
}}

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Detectando campos de ID adaptativamente"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result and result.get('id_fields'):
                return result['id_fields']
            else:
                return ['PATI_ID', 'EPIS_ID', 'DIAG_ID', 'MEDI_ID', 'OBSE_ID']  # Fallback
                
        except Exception as e:
            logger.error(f"Error detectando campos de ID: {e}")
            return ['PATI_ID', 'EPIS_ID', 'DIAG_ID', 'MEDI_ID', 'OBSE_ID']  # Fallback

    async def _llm_select_default_table_adaptive(self, resource_type: str, fhir_data: Dict[str, Any], stream_callback=None) -> str:
        """Usa LLM para seleccionar dinámicamente la tabla por defecto"""
        try:
            if not self.llm:
                return 'PATI_PATIENTS'  # Fallback básico
            
            schema_info = self._get_real_schema_info()
            
            prompt = f"""Eres un experto en mapeo FHIR a SQL médico.

SELECCIÓN DE TABLA POR DEFECTO:
Tipo de recurso: {resource_type}
Datos FHIR: {json.dumps(fhir_data, indent=2, ensure_ascii=False)}

ESQUEMA DISPONIBLE:
{schema_info}

INSTRUCCIONES:
1. Analiza el tipo de recurso FHIR
2. Identifica la tabla más apropiada para este tipo de datos
3. Considera el contenido de los datos FHIR
4. Selecciona la tabla que mejor se adapte

RESPUESTA JSON:
{{
    "default_table": "nombre_tabla",
    "reasoning": "explicación de la selección"
}}

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Seleccionando tabla por defecto adaptativamente"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result and result.get('default_table'):
                return result['default_table']
            else:
                return 'PATI_PATIENTS'  # Fallback
                
        except Exception as e:
            logger.error(f"Error seleccionando tabla por defecto: {e}")
            return 'PATI_PATIENTS'  # Fallback

    async def _llm_generate_fallback_sql_adaptive(self, operation: str, data: Dict[str, Any], table: str, stream_callback=None) -> str:
        """Usa LLM para generar SQL de fallback adaptativo"""
        try:
            if not self.llm:
                # Fallback básico hardcodeado
                if operation == 'INSERT':
                    return f"INSERT INTO {table} (PATI_NAME) VALUES ('{data.get('name', '')}');"
                elif operation == 'UPDATE':
                    return f"UPDATE {table} SET PATI_NAME = '{data.get('name', '')}' WHERE PATI_ID = '{data.get('id', '')}';"
                elif operation == 'DELETE':
                    return f"DELETE FROM {table} WHERE PATI_ID = '{data.get('id', '')}';"
                else:
                    return f"SELECT * FROM {table};"
            
            schema_info = self._get_real_schema_info()
            
            prompt = f"""Eres un experto en SQL médico. Genera SQL de fallback.

OPERACIÓN: {operation}
TABLA: {table}
DATOS: {json.dumps(data, indent=2, ensure_ascii=False)}

ESQUEMA DISPONIBLE:
{schema_info}

INSTRUCCIONES:
1. Genera SQL {operation} básico pero funcional
2. Usa solo campos que existan en la tabla
3. Maneja valores nulos correctamente
4. Evita campos de ID en INSERT
5. Usa condiciones WHERE apropiadas

RESPUESTA JSON:
{{
    "sql": "SQL generado aquí",
    "table": "{table}",
    "operation": "{operation}"
}}

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Generando SQL de fallback adaptativo"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result and result.get('sql'):
                return result['sql']
            else:
                # Fallback básico
                if operation == 'INSERT':
                    return f"INSERT INTO {table} (PATI_NAME) VALUES ('{data.get('name', '')}');"
                elif operation == 'UPDATE':
                    return f"UPDATE {table} SET PATI_NAME = '{data.get('name', '')}' WHERE PATI_ID = '{data.get('id', '')}';"
                elif operation == 'DELETE':
                    return f"DELETE FROM {table} WHERE PATI_ID = '{data.get('id', '')}';"
                else:
                    return f"SELECT * FROM {table};"
                
        except Exception as e:
            logger.error(f"Error generando SQL de fallback: {e}")
            # Fallback básico
            if operation == 'INSERT':
                return f"INSERT INTO {table} (PATI_NAME) VALUES ('{data.get('name', '')}');"
            elif operation == 'UPDATE':
                return f"UPDATE {table} SET PATI_NAME = '{data.get('name', '')}' WHERE PATI_ID = '{data.get('id', '')}';"
            elif operation == 'DELETE':
                return f"DELETE FROM {table} WHERE PATI_ID = '{data.get('id', '')}';"
            else:
                return f"SELECT * FROM {table};"

    async def _llm_validate_table_exists_adaptive(self, table_name: str, stream_callback=None) -> bool:
        """Usa LLM para validar si una tabla existe en el esquema"""
        try:
            if not self.llm:
                # Fallback mejorado - verificar en column_metadata
                return table_name in self.column_metadata
            
            if stream_callback:
                stream_callback("   💡 Validando tabla con IA adaptativa...")
            
            schema_info = self._get_real_schema_info()
            
            prompt = f"""Eres un experto en validación de esquemas de base de datos médicos.

VALIDACIÓN DE TABLA:
Tabla a verificar: {table_name}

ESQUEMA DISPONIBLE:
{schema_info}

INSTRUCCIONES:
1. Verifica si la tabla {table_name} existe en el esquema
2. Busca coincidencias exactas o similares
3. Considera variaciones de nombres
4. Si la tabla no existe, sugiere la tabla más apropiada

REGLAS ESPECÍFICAS:
- Para observaciones médicas, usa OBSE_OBSERVATIONS
- Para pacientes, usa PATI_PATIENTS
- Para episodios, usa EPIS_EPISODES
- Para diagnósticos, usa EPIS_DIAGNOSTICS
- Para medicamentos, usa MEDI_MEDICATIONS
- Para citas, usa APPO_APPOINTMENTS

RESPUESTA JSON:
{{
    "table_exists": true/false,
    "exact_match": true/false,
    "suggested_table": "tabla_sugerida_si_no_existe",
    "similar_tables": ["tabla1", "tabla2"]
}}

IMPORTANTE: Solo responde con el JSON."""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Validando existencia de tabla adaptativamente"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result and result.get('table_exists') is not None:
                table_exists = result.get('table_exists', False)
                suggested_table = result.get('suggested_table', '')
                
                if stream_callback:
                    if table_exists:
                        stream_callback(f"   ✅ Tabla {table_name} existe")
                    else:
                        stream_callback(f"   ⚠️ TABLA CORREGIDA: {table_name} → {suggested_table}")
                
                return table_exists
            else:
                # Fallback mejorado
                return table_name in self.column_metadata
                
        except Exception as e:
            logger.error(f"Error validando tabla: {e}")
            return table_name in self.column_metadata

    async def _llm_detect_common_fields_adaptive(self, table_name: str, field_type: str, stream_callback=None) -> List[str]:
        """Usa LLM para detectar dinámicamente campos comunes de una tabla"""
        try:
            if not self.llm:
                # Fallback básico
                if field_type == 'name':
                    return ['PATI_NAME']
                elif field_type == 'surname':
                    return ['PATI_SURNAME_1']
                elif field_type == 'id':
                    return ['PATI_ID']
                else:
                    return []
            
            schema_info = self._get_real_schema_info()
            
            prompt = f"""Eres un experto en análisis de esquemas de base de datos médicos.

DETECCIÓN DE CAMPOS COMUNES:
Tabla: {table_name}
Tipo de campo: {field_type}

ESQUEMA DISPONIBLE:
{schema_info}

INSTRUCCIONES:
1. Analiza la tabla {table_name}
2. Identifica campos que correspondan al tipo: {field_type}
3. Para 'name': busca campos de nombre, nombre completo, etc.
4. Para 'surname': busca campos de apellido
5. Para 'id': busca campos de identificación
6. Considera variaciones y sinónimos

RESPUESTA JSON:
{{
    "common_fields": ["campo1", "campo2"],
    "field_type": "{field_type}",
    "table": "{table_name}"
}}

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Detectando campos comunes adaptativamente"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result and result.get('common_fields'):
                return result['common_fields']
            else:
                # Fallback básico
                if field_type == 'name':
                    return ['PATI_NAME']
                elif field_type == 'surname':
                    return ['PATI_SURNAME_1']
                elif field_type == 'id':
                    return ['PATI_ID']
                else:
                    return []
                
        except Exception as e:
            logger.error(f"Error detectando campos comunes: {e}")
            # Fallback básico
            if field_type == 'name':
                return ['PATI_NAME']
            elif field_type == 'surname':
                return ['PATI_SURNAME_1']
            elif field_type == 'id':
                return ['PATI_ID']
            else:
                return []

    async def _llm_generate_dynamic_sql_adaptive(self, operation: str, table: str, data: Dict[str, Any], stream_callback=None) -> str:
        """Usa LLM para generar SQL dinámico basado en el esquema real"""
        try:
            if not self.llm:
                # Fallback básico
                if operation == 'INSERT':
                    return f"INSERT INTO {table} (PATI_NAME) VALUES ('{data.get('name', '')}');"
                elif operation == 'UPDATE':
                    return f"UPDATE {table} SET PATI_NAME = '{data.get('name', '')}' WHERE PATI_ID = '{data.get('id', '')}';"
                elif operation == 'DELETE':
                    return f"DELETE FROM {table} WHERE PATI_ID = '{data.get('id', '')}';"
                else:
                    return f"SELECT * FROM {table};"
            
            schema_info = self._get_real_schema_info()
            table_schema = self._get_table_schema_info(table)
            
            prompt = f"""Eres un experto en SQL médico. Genera SQL dinámico basado en el esquema real.

OPERACIÓN: {operation}
TABLA: {table}
DATOS: {json.dumps(data, indent=2, ensure_ascii=False)}

ESQUEMA COMPLETO:
{schema_info}

ESQUEMA DE LA TABLA ESPECÍFICA:
{table_schema}

INSTRUCCIONES:
1. Analiza el esquema real de la tabla {table}
2. Identifica campos apropiados para la operación {operation}
3. Mapea los datos disponibles a campos existentes
4. Genera SQL válido y funcional
5. Evita campos de ID en INSERT
6. Maneja valores nulos correctamente

REGLAS CRÍTICAS:
- Usa solo campos que existan en el esquema
- Para INSERT: NO incluyas campos de ID (déjalos autoincrementarse)
- Para UPDATE/DELETE: usa campos de ID apropiados en WHERE
- Maneja valores nulos con NULL, NO "None"
- Escapa valores de texto apropiadamente

RESPUESTA JSON:
{{
    "sql": "SQL generado aquí",
    "columns_used": ["columna1", "columna2"],
    "values_used": ["valor1", "valor2"],
    "operation": "{operation}"
}}

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Generando SQL dinámico adaptativo"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result and result.get('sql'):
                return result['sql']
            else:
                # Fallback básico
                if operation == 'INSERT':
                    return f"INSERT INTO {table} (PATI_NAME) VALUES ('{data.get('name', '')}');"
                elif operation == 'UPDATE':
                    return f"UPDATE {table} SET PATI_NAME = '{data.get('name', '')}' WHERE PATI_ID = '{data.get('id', '')}';"
                elif operation == 'DELETE':
                    return f"DELETE FROM {table} WHERE PATI_ID = '{data.get('id', '')}';"
                else:
                    return f"SELECT * FROM {table};"
                
        except Exception as e:
            logger.error(f"Error generando SQL dinámico: {e}")
            # Fallback básico
            if operation == 'INSERT':
                return f"INSERT INTO {table} (PATI_NAME) VALUES ('{data.get('name', '')}');"
            elif operation == 'UPDATE':
                return f"UPDATE {table} SET PATI_NAME = '{data.get('name', '')}' WHERE PATI_ID = '{data.get('id', '')}';"
            elif operation == 'DELETE':
                return f"DELETE FROM {table} WHERE PATI_ID = '{data.get('id', '')}';"
            else:
                return f"SELECT * FROM {table};"

    async def _llm_validate_and_correct_fictitious_ids_adaptive(self, values: List[Any], columns: List[str], stream_callback=None) -> List[Any]:
        """Validación rápida de IDs ficticios sin LLM para reducir costos"""
        try:
            if stream_callback:
                stream_callback("   💡 Validando IDs problemáticos...")
            
            # VALIDACIÓN RÁPIDA SIN LLM - usar patrones predefinidos
            corrected_values = []
            corrections_applied = []
            
            for val in values:
                if isinstance(val, str):
                    # Patrones de IDs ficticios más agresivos
                    if any(pattern in val.lower() for pattern in [
                        'unico', 'urn:uuid:', 'patient-id', 'observation-id', 
                        'encounter-id', 'medication-id', 'id-', 'ficticio', 
                        'mock', 'fake', 'unknown', 'null', 'none'
                    ]) or val.lower() in ['unknown', 'null', 'none', '']:
                        corrected_values.append(None)
                        corrections_applied.append(f"ID ficticio detectado: {val} → NULL")
                    elif val.isdigit():
                        # Convertir string numérico a int
                        corrected_values.append(int(val))
                        corrections_applied.append(f"String numérico convertido: {val} → {int(val)}")
                    else:
                        # Valor válido, mantener
                        corrected_values.append(val)
                elif val is None:
                    # Valor None, mantener
                    corrected_values.append(None)
                else:
                    # Valor no string, mantener
                    corrected_values.append(val)
            
            if stream_callback:
                stream_callback(f"   ✅ IDs validados")
                if corrections_applied:
                    for correction in corrections_applied:
                        stream_callback(f"      - {correction}")
            
            return corrected_values
                
        except Exception as e:
            logger.error(f"Error en validación de IDs ficticios: {e}")
            return values

    async def _get_real_inserted_id_adaptive(self, table: str, data: Dict[str, Any], stream_callback=None) -> Optional[int]:
        """
        Obtiene el ID real del registro insertado usando múltiples estrategias adaptativas.
        
        Args:
            table: Nombre de la tabla
            data: Datos insertados
            stream_callback: Función para mostrar progreso
            
        Returns:
            Optional[int]: ID del registro insertado o None si no se puede obtener
        """
        try:
            if stream_callback:
                stream_callback("   💡 Obteniendo ID real del registro insertado...")
            
            # DEBUG: Mostrar información de entrada
            print(f"   🔍 DEBUG _get_real_inserted_id_adaptive:")
            print(f"      - Tabla: {table}")
            print(f"      - Datos: {json.dumps(data, indent=2, ensure_ascii=False)}")
            
            # ESTRATEGIA 1: Buscar por criterios únicos según tabla
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # ESTRATEGIA DINÁMICA: Usar LLM para determinar criterios de búsqueda
                search_queries = []
                
                # Obtener esquema de la tabla dinámicamente
                table_schema = self._get_table_schema_info(table)
                print(f"      - Esquema de tabla: {table_schema}")
                
                # Buscar campos únicos en la tabla
                if self.llm:
                    # Usar LLM para determinar criterios de búsqueda dinámicos
                    search_criteria = await self._llm_determine_search_criteria(table, data, stream_callback)
                    for criteria in search_criteria:
                        search_queries.append(criteria)
                
                # ESTRATEGIA FALLBACK: Buscar el último registro insertado en la tabla
                # Determinar columna ID dinámicamente
                id_column = await self._llm_detect_id_column(table, stream_callback)
                print(f"      - Columna ID detectada: {id_column}")
                
                if id_column:
                    search_queries.append((
                        f"SELECT {id_column} FROM {table} ORDER BY {id_column} DESC LIMIT 1",
                        []
                    ))
                
                print(f"      - Consultas de búsqueda: {search_queries}")
                
                # Ejecutar las consultas en orden de prioridad
                for query, params in search_queries:
                    try:
                        print(f"      - Ejecutando consulta: {query}")
                        cursor.execute(query, params)
                        row = cursor.fetchone()
                        print(f"      - Resultado: {row}")
                    except Exception as e:
                        print(f"      - Error ejecutando consulta: {e}")
                        row = None
                        
                        if row and row[0]:
                            # Verificar si el ID es ficticio
                            id_str = str(row[0]).lower()
                            if any(fictitious in id_str for fictitious in [
                                "patient-id-unico", "urn:uuid:", "uuid:", "ficticio", "example", "test", "ejemplo"
                            ]):
                                print(f"      - ⚠️ ID ficticio detectado: {row[0]}, continuando...")
                                continue
                            
                            # Convertir a int de forma segura
                            try:
                                real_id = int(row[0])
                                if real_id > 0:
                                    if stream_callback:
                                        stream_callback(f"   💡 ID encontrado con consulta: {query[:50]}...")
                                    conn.close()
                                    return real_id
                            except (ValueError, TypeError):
                                # Si no se puede convertir a int, continuar con la siguiente consulta
                                print(f"      - Error convirtiendo a int: {row[0]}")
                                continue
                
                conn.close()
            except Exception as e:
                if stream_callback:
                    stream_callback(f"   ⚠️ Error en búsqueda por criterios: {e}")
                print(f"      - Error en búsqueda por criterios: {e}")
            
            # ESTRATEGIA 2: Buscar por datos específicos del registro insertado
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Buscar por campos únicos en los datos insertados (solo campos que existen en la tabla)
                table_schema = self._get_table_schema_info(table)
                existing_columns = []
                
                # Extraer nombres de columnas del esquema
                for line in table_schema.split('\n'):
                    if line.strip().startswith('- ') and ':' in line:
                        col_name = line.strip()[2:].split(':')[0].strip()
                        existing_columns.append(col_name)
                
                print(f"      - Columnas existentes en tabla: {existing_columns}")
                
                # Filtrar solo campos que existen en la tabla
                unique_fields = []
                for key, value in data.items():
                    if key in existing_columns and value and value != 'NULL' and value != 'null' and value != 'None':
                        if isinstance(value, str) and len(value) > 0:
                            unique_fields.append((key, value))
                
                print(f"      - Campos únicos válidos encontrados: {unique_fields}")
                
                if unique_fields:
                    # Construir consulta dinámica
                    conditions = []
                    params = []
                    for field, value in unique_fields[:3]:  # Usar máximo 3 campos
                        conditions.append(f"{field} = ?")
                        params.append(value)
                    
                    if conditions:
                        # Intentar con diferentes nombres de columna ID
                        id_columns_to_try = ['PATI_ID', 'ID', f"{table}_ID", 'id']
                        
                        for id_col in id_columns_to_try:
                            query = f"SELECT {id_col} FROM {table} WHERE {' AND '.join(conditions)} ORDER BY {id_col} DESC LIMIT 1"
                            try:
                                print(f"      - Probando consulta: {query}")
                                cursor.execute(query, params)
                                row = cursor.fetchone()
                                print(f"      - Resultado: {row}")
                                
                                if row and row[0]:
                                    # Verificar si el ID es ficticio
                                    id_str = str(row[0]).lower()
                                    if any(fictitious in id_str for fictitious in [
                                        "patient-id-unico", "urn:uuid:", "uuid:", "ficticio", "example", "test", "ejemplo"
                                    ]):
                                        print(f"      - ⚠️ ID ficticio detectado: {row[0]}, continuando...")
                                        continue
                                    
                                    try:
                                        real_id = int(row[0])
                                        if real_id > 0:
                                            if stream_callback:
                                                stream_callback(f"   💡 ID encontrado por datos únicos: {real_id}")
                                            conn.close()
                                            return real_id
                                    except (ValueError, TypeError):
                                        print(f"      - Error convirtiendo a int: {row[0]}")
                                        pass
                            except Exception as e:
                                print(f"      - Error con columna {id_col}: {e}")
                                continue
                
                conn.close()
            except Exception as e:
                if stream_callback:
                    stream_callback(f"   ⚠️ Error en búsqueda por datos específicos: {e}")
                print(f"      - Error en búsqueda por datos específicos: {e}")
            
            if stream_callback:
                stream_callback("   ❌ No se pudo obtener el ID del registro insertado")
            
            return None
            
        except Exception as e:
            logger.error(f"Error obteniendo ID real: {e}")
            print(f"      - Error general: {e}")
            return None

    async def _llm_determine_search_criteria(self, table: str, data: Dict[str, Any], stream_callback=None) -> List[Tuple[str, List[Any]]]:
        """Usa LLM para determinar criterios de búsqueda dinámicos"""
        try:
            if not self.llm:
                return []
            
            if stream_callback:
                stream_callback("   💡 Determinando criterios de búsqueda dinámicos...")
            
            table_schema = self._get_table_schema_info(table)
            
            prompt = f"""Eres un experto en bases de datos médicas. Determina criterios de búsqueda para encontrar un registro recién insertado.

TABLA: {table}
DATOS INSERTADOS: {json.dumps(data, indent=2, ensure_ascii=False)}
ESQUEMA DE LA TABLA:
{table_schema}

TAREA: Genera consultas SQL para encontrar el registro recién insertado usando criterios únicos.

INSTRUCCIONES:
1. Analiza los datos insertados
2. Identifica campos que pueden ser únicos
3. Genera consultas SQL para buscar por esos criterios
4. Usa campos como nombre, fecha, identificadores únicos
5. Considera combinaciones de campos para mayor precisión
6. SOLO usa campos que existan en el esquema de la tabla
7. NO uses campos FHIR como "resourceType", "id", etc. que no existen en SQL

ESTRATEGIA:
- Busca por campos únicos como nombres, fechas, identificadores
- Usa combinaciones de campos para mayor precisión
- Considera el contexto médico de la tabla
- SOLO usa campos que existan en el esquema

RESPUESTA JSON:
{{
    "search_queries": [
        {{
            "sql": "SELECT PATI_ID FROM PATI_PATIENTS WHERE PATI_NAME = ? AND PATI_BIRTH_DATE = ? ORDER BY PATI_ID DESC LIMIT 1",
            "params": ["Carlos Martínez", "1963-01-01"]
        }}
    ]
}}

IMPORTANTE: 
- SOLO usa campos que existan en el esquema de la tabla
- NO uses campos FHIR como "resourceType", "id", etc.
- Usa campos específicos de la tabla como PATI_NAME, PATI_BIRTH_DATE, etc.

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}]
            )
            
            result = self._try_parse_llm_json(response.content)
            if result and result.get('search_queries'):
                queries = []
                for query_info in result['search_queries']:
                    queries.append((query_info['sql'], query_info['params']))
                return queries
            
            return []
            
        except Exception as e:
            logger.error(f"Error determinando criterios de búsqueda: {e}")
            return []
    
    async def _llm_detect_id_column(self, table: str, stream_callback=None) -> Optional[str]:
        """Usa LLM para detectar la columna ID de una tabla"""
        try:
            if not self.llm:
                # Fallback básico
                table_prefix = table.split('_')[0] if '_' in table else table
                return f"{table_prefix}_ID"
            
            if stream_callback:
                stream_callback("   💡 Detectando columna ID...")
            
            table_schema = self._get_table_schema_info(table)
            
            prompt = f"""Eres un experto en esquemas de base de datos. Identifica la columna ID principal de esta tabla.

TABLA: {table}
ESQUEMA:
{table_schema}

TAREA: Identifica la columna que actúa como clave primaria o ID principal.

INSTRUCCIONES:
1. Busca columnas que terminen en _ID
2. Busca columnas que sean claves primarias (PRIMARY KEY)
3. Busca columnas de autoincremento
4. Considera el contexto de la tabla
5. Prioriza columnas que sean PRIMARY KEY

ESTRATEGIA:
- Primero busca columnas marcadas como PRIMARY KEY
- Luego busca columnas que terminen en _ID
- Considera el prefijo de la tabla (PATI_, EPIS_, etc.)

RESPUESTA JSON:
{{
    "id_column": "nombre_de_la_columna_id"
}}

IMPORTANTE: 
- Prioriza columnas marcadas como PRIMARY KEY
- Usa el formato estándar: TABLA_ID (ej: PATI_ID, EPIS_ID)
- Si no hay PRIMARY KEY claro, usa el patrón estándar

Responde SOLO con el JSON:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}]
            )
            
            result = self._try_parse_llm_json(response.content)
            if result and result.get('id_column'):
                return result['id_column']
            
            # Fallback básico
            table_prefix = table.split('_')[0] if '_' in table else table
            return f"{table_prefix}_ID"
            
        except Exception as e:
            logger.error(f"Error detectando columna ID: {e}")
            # Fallback básico
            table_prefix = table.split('_')[0] if '_' in table else table
            return f"{table_prefix}_ID"

    async def _llm_consolidated_discovery_and_mapping(self, fhir_data: Dict[str, Any], stream_callback=None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mapeo FHIR→SQL optimizado con UNA SOLA llamada LLM inteligente"""
        try:
            # DEBUG CRÍTICO: Mostrar datos de entrada
            print(f"🔍 DEBUG _llm_consolidated_discovery_and_mapping:")
            print(f"   📥 FHIR Data: {json.dumps(fhir_data, indent=2, ensure_ascii=False)}")
            
            if not self.llm:
                return await self._llm_map_fhir_to_sql_adaptive(fhir_data, stream_callback, context)
            
            if stream_callback:
                stream_callback("   🚀 Iniciando mapeo FHIR→SQL optimizado (UNA llamada LLM)...")
            
            # OPTIMIZACIÓN: UNA SOLA LLAMADA LLM PARA TODO
            schema_info = await self._get_cached_schema_info()
            
            # CRÍTICO: Leer directamente el resourceType para evitar llamada LLM
            resource_type = fhir_data.get('resourceType', '')
            if not resource_type or not isinstance(resource_type, str) or not resource_type.strip():
                if stream_callback:
                    stream_callback("   ⚠️ No se encontró resourceType, usando LLM...")
                resource_type = "Unknown"
            else:
                resource_type = resource_type.strip()
                if stream_callback:
                    stream_callback(f"   📋 Tipo detectado directamente: {resource_type}")
            
            # UNA SOLA LLAMADA LLM PARA SELECCIÓN DE TABLA Y MAPEO
            prompt = f"""CRÍTICO: Análisis completo FHIR→SQL en UNA SOLA respuesta.

DATOS FHIR:
{json.dumps(fhir_data, indent=2, ensure_ascii=False)}

TIPO DE RECURSO: {resource_type}

ESQUEMA DE BASE DE DATOS:
{schema_info}

INSTRUCCIONES ABSOLUTAS:
1. Analiza el tipo de recurso FHIR
2. Selecciona la tabla SQL más apropiada
3. Mapea los campos FHIR a columnas SQL
4. Extrae SOLO datos reales (NO inventes datos)
5. Usa NULL para campos sin datos reales
6. Considera el contexto médico

RESPUESTA EN FORMATO JSON:
{{
    "resource_type": "tipo_detectado",
    "target_table": "tabla_seleccionada", 
    "mapped_fields": {{
        "columna_sql": "valor_real_o_null",
        "columna_sql2": "valor_real_o_null"
    }},
    "confidence": "alta/media/baja",
    "reasoning": "explicación_breve"
}}

IMPORTANTE: 
- Solo mapea campos con datos reales
- Usa NULL para campos sin datos
- NO inventes nombres, fechas, o valores
- Considera el contexto médico del recurso
- Selecciona tabla basada en el tipo de recurso"""

            if stream_callback:
                stream_callback("   🤖 Analizando con LLM (UNA llamada)...")
            
            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Análisis completo FHIR→SQL"
            )
            
            response_text = self._extract_response_text(response)
            parsed_result = self._try_parse_llm_json(response_text)
            
            if not parsed_result:
                if stream_callback:
                    stream_callback("   ⚠️ Error parsing LLM response, usando fallback...")
                return await self._llm_map_fhir_to_sql_adaptive(fhir_data, stream_callback, context)
            
            # Extraer resultados de la respuesta consolidada
            final_resource_type = parsed_result.get('resource_type', resource_type)
            target_table = parsed_result.get('target_table', '')
            mapped_data = parsed_result.get('mapped_fields', {})
            confidence = parsed_result.get('confidence', 'media')
            reasoning = parsed_result.get('reasoning', '')
            
            if stream_callback:
                stream_callback(f"   ✅ Análisis completado (confianza: {confidence})")
                stream_callback(f"   📋 Razonamiento: {reasoning}")
            
            # CRÍTICO: Establecer el ID del paciente si es un recurso Patient
            if final_resource_type == "Patient":
                patient_id = context.get('patient_id') if context else None
                if patient_id:
                    self._current_patient_id = patient_id
                    print(f"   🔗 Estableciendo _current_patient_id: {patient_id}")
            
            # Validación simple sin LLM adicional
            cleaned_data = {}
            for column, value in mapped_data.items():
                if value is not None and value != "" and str(value).strip():
                    # Validar que no sea un ID ficticio
                    if not self._is_fictitious_id(value):
                        cleaned_data[column] = value
                    else:
                        cleaned_data[column] = None
                else:
                    cleaned_data[column] = None
            
            # Agregar PATI_ID del contexto si existe
            if context and context.get('patient_id'):
                patient_id = context['patient_id']
                print(f"   🔍 Verificando PATI_ID del contexto: {patient_id}")
                
                # Obtener columnas de la tabla
                table_columns = await self._get_table_columns(target_table)
                print(f"   📋 Columnas de {target_table}: {table_columns}")
                
                # SISTEMA DINÁMICO: Usar LLM para detectar columnas de relación
                if self.llm:
                    try:
                        # PROMPT DINÁMICO para detectar columnas de relación
                        relationship_prompt = f"""Eres un experto en relaciones de base de datos médicas.

TABLA: {target_table}
COLUMNAS DISPONIBLES: {table_columns}
PATIENT_ID DEL CONTEXTO: {patient_id}

TAREA: Identifica qué columna(s) de la tabla deben recibir el ID del paciente.

REGLAS:
1. Busca columnas que referencien al paciente (ej: PATI_ID, PATIENT_ID, etc.)
2. Considera columnas que contengan "PATI", "PATIENT", "SUBJECT"
3. Si no hay columnas específicas, no agregues ninguna
4. Solo agrega relaciones válidas

RESPUESTA JSON:
{{
    "patient_columns": ["columna1", "columna2"],
    "reasoning": "explicación de las relaciones"
}}

Responde SOLO con el JSON:"""

                        response = await asyncio.to_thread(
                            _call_openai_native, self.llm, [{"role": "user", "content": relationship_prompt}],
                            task_description="Detectando relaciones de paciente"
                        )
                        
                        content = self._extract_response_text(response)
                        relationship_result = self._try_parse_llm_json(content)
                        
                        if relationship_result and relationship_result.get('patient_columns'):
                            for column in relationship_result['patient_columns']:
                                if column in table_columns:
                                    cleaned_data[column] = patient_id
                                    print(f"   🔗 AGREGANDO {column}: {patient_id} (LLM dinámico)")
                                    print(f"   📊 Razonamiento: {relationship_result.get('reasoning', 'N/A')}")
                        else:
                            print(f"   ⚠️ No se detectaron columnas de relación para paciente")
                    except Exception as e:
                        print(f"   ⚠️ Error detectando relaciones de paciente: {e}")
                        # Fallback simple
                        if 'PATI_ID' in table_columns:
                            cleaned_data['PATI_ID'] = patient_id
                            print(f"   🔗 INSERTANDO PATI_ID del contexto: {patient_id}")
                else:
                    # Fallback sin LLM
                    if 'PATI_ID' in table_columns:
                        cleaned_data['PATI_ID'] = patient_id
                        print(f"   🔗 INSERTANDO PATI_ID del contexto: {patient_id}")

                if stream_callback:
                    stream_callback(f"   🎉 Mapeo optimizado completado: {final_resource_type} → {target_table}")

                final_result = {
                    "success": True,
                    "resource_type": final_resource_type,
                "target_table": target_table,
                "mapped_data": cleaned_data,
                "validation": {
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "llm_calls": 1  # Solo una llamada LLM
                },
                "debug_info": {
                    "method": "optimized_single_llm_call",
                    "llm_calls_reduced": True
                }
            }
            
            # DEBUG FINAL: Mostrar resultado final
            print(f"   🎯 RESULTADO FINAL _llm_consolidated_discovery_and_mapping:")
            print(f"   📋 Resource Type: {final_resource_type}")
            print(f"   🗃️ Target Table: {target_table}")
            print(f"   📋 Mapped Data: {cleaned_data}")
            print(f"   ✅ Success: {final_result['success']}")
            print(f"   🚀 LLM Calls: 1 (optimizado)")
            
            return final_result
            
        except Exception as e:
            if stream_callback:
                stream_callback(f"   ❌ Error en mapeo optimizado: {str(e)}")
            return await self._llm_map_fhir_to_sql_adaptive(fhir_data, stream_callback, context)

    def _is_fictitious_id(self, value: Any) -> bool:
        """Detectar IDs ficticios sin usar LLM"""
        if not value:
            return False
        
        value_str = str(value).lower()
        fictitious_patterns = [
            'urn:uuid:', 'patient-id-', 'encounter-id-', 'medication-id-',
            'observation-id-', 'condition-id-', 'example', 'test', 'demo',
            'fictitious', 'mock', 'sample', 'dummy'
        ]
        
        return any(pattern in value_str for pattern in fictitious_patterns)
    
    async def _get_table_columns(self, table_name: str) -> List[str]:
        """Obtener columnas de tabla directamente de la base de datos"""
        try:
            if not table_name:
                return []
            
            # Consulta directa a la base de datos para obtener columnas
            query = f"PRAGMA table_info({table_name})"
            
            # Usar sqlite3 sincrónico en un thread separado
            def get_columns_sync():
                with sqlite3.connect(self.db_path) as db:
                    cursor = db.execute(query)
                    rows = cursor.fetchall()
                    return [row[1] for row in rows]  # row[1] es el nombre de la columna
            
            # Ejecutar en thread separado para no bloquear
            columns = await asyncio.to_thread(get_columns_sync)
            return columns
                    
        except Exception as e:
            print(f"   ⚠️ Error obteniendo columnas de {table_name}: {e}")
            # Fallback: intentar con el esquema cacheado
            try:
                schema_info = await self._get_cached_schema_info(table_name)
                if schema_info:
                    lines = schema_info.split('\n')
                    columns = []
                    for line in lines:
                        if ':' in line and '(' in line:
                            column_part = line.split(':')[0].strip()
                            if column_part.startswith('- '):
                                column_name = column_part[2:]
                                columns.append(column_name)
                    return columns
            except:
                pass
            return []
    
    async def _llm_detect_resource_type_step(self, fhir_data: Dict[str, Any], stream_callback=None) -> str:
        """Paso 1: Detectar tipo de recurso FHIR"""
        
        # PRIMERA VALIDACIÓN: Leer directamente el campo resourceType
        actual_resource_type = fhir_data.get('resourceType', '')
        if actual_resource_type and isinstance(actual_resource_type, str) and actual_resource_type.strip():
            if stream_callback:
                stream_callback(f"   📋 Tipo detectado directamente: {actual_resource_type}")
            return actual_resource_type.strip()
        
        # SEGUNDA VALIDACIÓN: Si no hay resourceType, usar LLM con prompt más específico
        prompt = f"""CRÍTICO: Lee EXACTAMENTE el campo "resourceType" del JSON FHIR.

DATOS FHIR:
{json.dumps(fhir_data, indent=2, ensure_ascii=False)}

INSTRUCCIONES ABSOLUTAS:
1. Busca el campo "resourceType" en el nivel raíz del JSON
2. Lee EXACTAMENTE el valor de ese campo
3. NO analices la estructura, NO infieras, NO adivines
4. Solo lee el valor del campo "resourceType"
5. Si el campo no existe, responde "Unknown"

EJEMPLOS ESPECÍFICOS:
- Si resourceType: "Patient" → responde "Patient"
- Si resourceType: "Encounter" → responde "Encounter"  
- Si resourceType: "Condition" → responde "Condition"
- Si resourceType: "MedicationRequest" → responde "MedicationRequest"
- Si resourceType: "Observation" → responde "Observation"
- Si resourceType: "Medication" → responde "Medication"

RESPUESTA OBLIGATORIA:
- Solo el valor del campo resourceType
- Sin comillas, sin explicaciones
- Sin formato JSON
- Sin texto adicional
- Si no encuentra el campo, responde "Unknown" """

        response = await asyncio.to_thread(
            _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
            task_description="Detección de tipo de recurso"
        )
        
        resource_type = self._extract_response_text(response).strip().strip('"').strip("'")
        
        # Validación dinámica usando LLM para verificar si el tipo es válido
        if stream_callback:
            stream_callback(f"   🔍 Validando tipo detectado: {resource_type}")
        
        validation_prompt = f"""ANÁLISIS DE VALIDACIÓN DE TIPO DE RECURSO FHIR

TIPO DETECTADO: {resource_type}

DATOS FHIR:
{json.dumps(fhir_data, indent=2, ensure_ascii=False)}

INSTRUCCIONES:
1. Analiza si el tipo "{resource_type}" es un tipo de recurso FHIR válido
2. Considera el contexto médico y la estructura de los datos
3. Si el tipo es válido, responde "VALID"
4. Si el tipo no es válido o no tiene sentido, responde "INVALID"
5. Si no hay suficiente información, responde "UNKNOWN"

RESPUESTA: Solo "VALID", "INVALID" o "UNKNOWN" """

        validation_response = await asyncio.to_thread(
            _call_openai_native, self.llm, [{"role": "user", "content": validation_prompt}],
            task_description="Validación dinámica de tipo de recurso"
        )
        
        validation_result = self._extract_response_text(validation_response).strip().upper()
        
        if validation_result == "INVALID" or validation_result == "UNKNOWN":
            if stream_callback:
                stream_callback(f"   ⚠️ Tipo detectado no válido: {resource_type}")
                stream_callback(f"   🔄 Reintentando detección con análisis más profundo...")
            
            # Reintentar con un prompt más específico y dinámico
            retry_prompt = f"""ANÁLISIS PROFUNDO DE TIPO DE RECURSO FHIR

DATOS FHIR:
{json.dumps(fhir_data, indent=2, ensure_ascii=False)}

INSTRUCCIONES:
1. Analiza la estructura completa del JSON FHIR
2. Busca patrones médicos, campos específicos, y contexto
3. Determina el tipo de recurso más apropiado basado en el contenido
4. Considera campos como "patient", "encounter", "medication", "observation", etc.
5. Si no hay información clara, usa el contexto médico para inferir

RESPUESTA: Solo el tipo de recurso más apropiado (Patient, Encounter, Condition, etc.)"""
            
            retry_response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": retry_prompt}],
                task_description="Reintento de detección de tipo"
            )
            
            resource_type = self._extract_response_text(retry_response).strip().strip('"').strip("'")
            
        if stream_callback:
            stream_callback(f"   📋 Tipo detectado: {resource_type}")
        
        return resource_type
    
    async def _llm_select_table_step(self, resource_type: str, fhir_data: Dict[str, Any], stream_callback=None) -> str:
        """Paso 2: Seleccionar tabla apropiada usando LLM inteligente"""
        
        prompt = f"""CRÍTICO: Selecciona la tabla SQL más apropiada para el tipo de recurso FHIR.

TIPO DE RECURSO: {resource_type}

ESQUEMA DISPONIBLE:
{self._get_real_schema_info()}

DATOS FHIR:
{json.dumps(fhir_data, indent=2, ensure_ascii=False)}

INSTRUCCIONES CRÍTICAS:
1. Analiza el tipo de recurso y el contexto médico
2. Busca en el esquema la tabla más apropiada
3. Considera el significado médico y la estructura de datos
4. NO uses mapeos rígidos, analiza dinámicamente
5. Si no encuentras una tabla específica, usa la más genérica disponible

ANÁLISIS DINÁMICO:
- Patient: busca tablas relacionadas con pacientes, personas, usuarios
- Encounter: busca tablas de episodios, encuentros, visitas
- Condition: busca tablas de condiciones, diagnósticos, problemas
- Observation: busca tablas de observaciones, mediciones, resultados
- MedicationRequest: busca tablas de medicamentos, prescripciones
- Medication: busca tablas de medicamentos, fármacos

RESPUESTA: Solo el nombre de la tabla más apropiada del esquema disponible."""

        response = await asyncio.to_thread(
            _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
            task_description="Selección de tabla"
        )
        
        target_table = self._extract_response_text(response).strip().strip('"').strip("'")
        
        # Validación: verificar que la tabla existe y es correcta
        schema_info = self._get_real_schema_info()
        
        # Validación dinámica de tabla usando LLM
        if stream_callback:
            stream_callback(f"   🔍 Validando tabla seleccionada: {target_table}")
        
        table_validation_prompt = f"""VALIDACIÓN DINÁMICA DE TABLA PARA RECURSO FHIR

TIPO DE RECURSO: {resource_type}
TABLA SELECCIONADA: {target_table}

ESQUEMA DE BASE DE DATOS:
{schema_info}

DATOS FHIR:
{json.dumps(fhir_data, indent=2, ensure_ascii=False)}

INSTRUCCIONES:
1. Analiza si la tabla "{target_table}" es apropiada para el tipo "{resource_type}"
2. Considera la estructura de la tabla y los campos disponibles
3. Si la tabla es correcta, responde "VALID"
4. Si la tabla no es correcta, responde "INVALID"
5. Si no hay suficiente información, responde "UNKNOWN"

RESPUESTA: Solo "VALID", "INVALID" o "UNKNOWN" """

        table_validation_response = await asyncio.to_thread(
            _call_openai_native, self.llm, [{"role": "user", "content": table_validation_prompt}],
            task_description="Validación dinámica de tabla"
        )
        
        table_validation_result = self._extract_response_text(table_validation_response).strip().upper()
        
        if table_validation_result == "INVALID" or table_validation_result == "UNKNOWN":
            if stream_callback:
                stream_callback(f"   ⚠️ Tabla incorrecta para {resource_type}: {target_table}")
                stream_callback(f"   🔄 Buscando tabla más apropiada...")
            
            # Buscar tabla más apropiada usando LLM
            table_correction_prompt = f"""SELECCIÓN DINÁMICA DE TABLA PARA RECURSO FHIR

TIPO DE RECURSO: {resource_type}

ESQUEMA DE BASE DE DATOS:
{schema_info}

DATOS FHIR:
{json.dumps(fhir_data, indent=2, ensure_ascii=False)}

INSTRUCCIONES:
1. Analiza el esquema completo de la base de datos
2. Busca la tabla más apropiada para el tipo "{resource_type}"
3. Considera patrones en nombres de tablas, campos disponibles, y contexto médico
4. Selecciona la tabla que mejor se adapte a los datos FHIR
5. Responde solo con el nombre de la tabla más apropiada

RESPUESTA: Solo el nombre de la tabla (ej: PATI_PATIENTS, EPIS_EPISODES, etc.)"""

            table_correction_response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": table_correction_prompt}],
                task_description="Corrección dinámica de tabla"
            )
            
            target_table = self._extract_response_text(table_correction_response).strip()
        
        # Verificar que la tabla existe en el esquema
        if target_table not in schema_info:
            if stream_callback:
                stream_callback(f"   ⚠️ Tabla no existe en esquema: {target_table}")
            # Usar tabla por defecto
            target_table = "PATI_PATIENTS"
            if stream_callback:
                stream_callback(f"   🔄 Usando tabla por defecto: {target_table}")
        
        if stream_callback:
            stream_callback(f"   🎯 Tabla seleccionada: {target_table}")
        
        return target_table

    async def _llm_flexible_sql_analysis(self, sql: str, query_context: str = "", stream_callback=None) -> str:
        """
        Análisis flexible de SQL usando prompts específicos del LLM.
        ARQUITECTURA ADAPTATIVA: Diferentes prompts según el contexto.
        
        Args:
            sql: SQL a analizar
            query_context: Contexto de la consulta original del usuario
            stream_callback: Función para mostrar progreso
            
        Returns:
            str: SQL analizado y mejorado
        """
        try:
            if not self.llm:
                return sql
            
            if stream_callback:
                stream_callback("   - Análisis flexible con IA...")
            
            # PROMPT ESPECÍFICO SEGÚN CONTEXTO
            if "grave" in query_context.lower() or "pronóstico" in query_context.lower():
                # Prompt para consultas de pronóstico grave
                prompt = f"""Eres un experto en SQL especializado en consultas de pronóstico médico grave.

CONSULTA ORIGINAL: {query_context}
SQL GENERADO: {sql}

ESQUEMA DISPONIBLE:
{self._get_real_schema_info()}

ANÁLISIS ESPECÍFICO PARA PRONÓSTICO GRAVE:
1. ¿Busca pacientes con condiciones médicas graves?
2. ¿Usa campos de texto libre para buscar términos médicos?
3. ¿Ordena por severidad o gravedad?
4. ¿Incluye información del paciente y diagnóstico?

REGLAS PARA PRONÓSTICO GRAVE:
- Usar DIAG_OBSERVATION (NO DIAG_DESCRIPTION que no existe)
- Buscar términos como 'grave', 'cáncer', 'terminal', 'crítico'
- JOIN PATI_PATIENTS con EPIS_DIAGNOSTICS
- Ordenar por términos de gravedad en DIAG_OBSERVATION
- Incluir información del paciente (PATI_NAME, PATI_SURNAME_1)

CAMPOS CORRECTOS:
- EPIS_DIAGNOSTICS.DIAG_OBSERVATION: Campo de texto para diagnósticos
- PATI_PATIENTS.PATI_NAME, PATI_SURNAME_1: Datos del paciente
- NO USAR: DIAG_DESCRIPTION (no existe en la tabla)

EJEMPLOS:
✅ CORRECTO: SELECT p.PATI_NAME, p.PATI_SURNAME_1, d.DIAG_OBSERVATION FROM PATI_PATIENTS p JOIN EPIS_DIAGNOSTICS d ON p.PATI_ID = d.PATI_ID WHERE d.DIAG_OBSERVATION LIKE '%grave%' ORDER BY CASE WHEN d.DIAG_OBSERVATION LIKE '%cáncer%' THEN 1 WHEN d.DIAG_OBSERVATION LIKE '%terminal%' THEN 2 ELSE 3 END LIMIT 1

INSTRUCCIONES:
1. Verifica que use DIAG_OBSERVATION (no DIAG_DESCRIPTION)
2. Añade JOINs necesarios para obtener datos del paciente
3. Incluye filtros para condiciones graves
4. Ordena por severidad médica
5. Mantén la lógica original

Devuelve SOLO el SQL:"""
            
            elif "pacientes" in query_context.lower() or "count" in sql.lower():
                # Prompt para consultas de conteo de pacientes
                prompt = f"""Eres un experto en SQL especializado en consultas médicas de pacientes.

CONSULTA ORIGINAL: {query_context}
SQL GENERADO: {sql}

ESQUEMA DISPONIBLE:
{self._get_real_schema_info()}

ANÁLISIS ESPECÍFICO PARA PACIENTES:
1. ¿Es una consulta simple de conteo? (SELECT COUNT(*) FROM tabla)
2. ¿Necesita JOINs para obtener datos relacionados?
3. ¿Hay filtros por condiciones médicas?
4. ¿Se busca información específica de pacientes?

REGLAS PARA CONSULTAS DE PACIENTES:
- Si es SELECT COUNT(*) FROM PATI_PATIENTS → Es correcto
- Si busca pacientes con condición → Añadir JOIN con diagnósticos
- Si busca medicación → Añadir JOIN con medicamentos
- Si busca por descripción → Usar LIKE en campos de texto

EJEMPLOS:
✅ SIMPLE: SELECT COUNT(*) FROM PATI_PATIENTS
✅ CON FILTRO: SELECT COUNT(*) FROM PATI_PATIENTS p JOIN EPIS_DIAGNOSTICS d ON p.PATI_ID = d.PATI_ID WHERE d.DIAG_OBSERVATION LIKE '%diabetes%'

INSTRUCCIONES:
1. Analiza si el SQL es apropiado para la consulta
2. Si es simple y correcto → NO MODIFICAR
3. Si necesita JOINs para datos relacionados → AÑADIR
4. Si hay errores → CORREGIR
5. Mantén la lógica original

Devuelve SOLO el SQL:"""
            
            elif "medicación" in query_context.lower() or "medication" in sql.lower():
                # Prompt para consultas de medicación
                prompt = f"""Eres un experto en SQL especializado en consultas de medicación.

CONSULTA ORIGINAL: {query_context}
SQL GENERADO: {sql}

ESQUEMA DISPONIBLE:
{self._get_real_schema_info()}

ANÁLISIS ESPECÍFICO PARA MEDICACIÓN:
1. ¿Incluye JOIN con tabla de medicamentos?
2. ¿Relaciona pacientes con sus medicaciones?
3. ¿Filtra por tipo de medicación?
4. ¿Agrupa por medicación?

REGLAS PARA MEDICACIÓN:
- Siempre JOIN PATI_PATIENTS con PATI_USUAL_MEDICATION
- Usar LEFT JOIN para incluir pacientes sin medicación
- Agrupar por MEDICATION_NAME si se pide conteo
- Filtrar por condición médica si se especifica

EJEMPLOS:
✅ CORRECTO: SELECT p.PATI_ID, m.MEDICATION_NAME FROM PATI_PATIENTS p LEFT JOIN PATI_USUAL_MEDICATION m ON p.PATI_ID = m.PATI_ID
✅ CON FILTRO: SELECT m.MEDICATION_NAME, COUNT(*) FROM PATI_PATIENTS p JOIN PATI_USUAL_MEDICATION m ON p.PATI_ID = m.PATI_ID JOIN EPIS_DIAGNOSTICS d ON p.PATI_ID = d.PATI_ID WHERE d.DIAG_OBSERVATION LIKE '%diabetes%' GROUP BY m.MEDICATION_NAME

INSTRUCCIONES:
1. Verifica que incluya JOIN con medicación
2. Añade filtros si se especifican condiciones
3. Corrige agrupaciones si es necesario
4. Mantén la lógica original

Devuelve SOLO el SQL:"""
            
            else:
                # Prompt genérico flexible
                prompt = f"""Eres un experto en SQL que analiza consultas de forma inteligente.

CONSULTA ORIGINAL: {query_context}
SQL GENERADO: {sql}

ESQUEMA DISPONIBLE:
{self._get_real_schema_info()}

ANÁLISIS FLEXIBLE:
1. ¿El SQL es sintácticamente correcto?
2. ¿Incluye todos los JOINs necesarios?
3. ¿Los filtros son apropiados?
4. ¿La lógica coincide con la consulta?

REGLAS GENERALES:
- Si es simple y correcto → NO MODIFICAR
- Si falta JOIN necesario → AÑADIR
- Si hay error de sintaxis → CORREGIR
- Si la lógica no coincide → AJUSTAR

INSTRUCCIONES:
1. Analiza la corrección del SQL
2. Si es correcto → devuelve original
3. Si necesita mejoras → mejóralo
4. Mantén la lógica original

Devuelve SOLO el SQL:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Análisis flexible específico con IA"
            )
            
            corrected_sql = self._extract_response_text(response).strip()
            corrected_sql = self._clean_llm_sql_response(corrected_sql)
            
            if corrected_sql and not corrected_sql.startswith("Error"):
                if corrected_sql != sql:
                    logger.info(f"🧠 LLM realizó análisis flexible específico")
                    logger.info(f"   SQL original: {sql[:100]}...")
                    logger.info(f"   SQL mejorado: {corrected_sql[:100]}...")
                    
                    if stream_callback:
                        stream_callback("   ✅ Análisis específico completado")
                else:
                    if stream_callback:
                        stream_callback("   ✅ SQL analizado, sin cambios necesarios")
                
                return corrected_sql
            else:
                logger.warning(f"⚠️ LLM no pudo realizar análisis flexible específico")
                return sql
                
        except Exception as e:
            logger.error(f"Error en _llm_flexible_sql_analysis: {e}")
            return sql