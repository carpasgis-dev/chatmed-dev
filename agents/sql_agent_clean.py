#!/usr/bin/env python3
"""
🧠 SQL Agent Completamente Robusto - Versión Avanzada
===================================================
Sistema que aprovecha todas las capacidades del SQL Agent original con múltiples correcciones LLM
"""
import logging
import sqlite3
import json
import re
import os
import asyncio
import time
import traceback
from typing import Dict, List, Any, Union, Optional, Tuple
from pathlib import Path
from langchain.schema import SystemMessage, HumanMessage
from datetime import datetime
import math

# Import de los módulos de utilidades
try:
    from ..utils.sql_cleaner import SQLCleaner
    from ..utils.sql_executor import SQLExecutor
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.sql_cleaner import SQLCleaner
    from utils.sql_executor import SQLExecutor

# --- Agentes Inteligentes Especializados ---
try:
    from .intelligent_name_detector import IntelligentNameDetector, IntelligentConceptDetector, create_intelligent_detectors
    INTELLIGENT_DETECTORS_AVAILABLE = True
except ImportError:
    INTELLIGENT_DETECTORS_AVAILABLE = False
    print("⚠️ Agentes inteligentes no disponibles. Usando detección básica.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SQLAgentRobust")

class MockResponse:
    def __init__(self, content: str):
        self.content = content

def _call_openai_native(client, messages, temperature=0.1, max_tokens=4000, task_description="Consultando modelo de IA") -> MockResponse:
    """Función de compatibilidad para llamar a OpenAI nativo con streaming y logging."""
    try:
        from openai import OpenAI
        native_client = OpenAI()

        # Convertir mensajes al formato correcto
        if isinstance(messages, list):
            openai_messages = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    role = "system" if isinstance(msg, SystemMessage) else "user"
                    openai_messages.append({"role": role, "content": str(msg.content)})
                elif isinstance(msg, dict):
                    openai_messages.append(msg)
                else:
                    openai_messages.append({"role": "user", "content": str(msg)})
        else:
            content = messages.content if hasattr(messages, 'content') else str(messages)
            openai_messages = [{"role": "user", "content": content}]

        # Streaming para mostrar progreso
        stream_buffer = []
        print(f"   💡 {task_description}...", end="", flush=True)
        
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
                if len(stream_buffer) % 10 == 0:
                    print(".", end="", flush=True)
                    
        print(" ✓")
        content = "".join(stream_buffer)

        if not content.strip():
            content = '{"success": false, "message": "Error: Respuesta vacía del LLM"}'

        return MockResponse(content)

    except Exception as e:
        error_msg = f"Error en llamada OpenAI: {str(e)}"
        print(f"   ❌ ERROR EN LLM: {error_msg}")
        logger.error(f"Error en _call_openai_native: {e}", exc_info=True)
        return MockResponse('{"success": false, "message": "Error crítico en la llamada al LLM."}')

class SQLAgentRobust:
    def __init__(self, db_path: str, llm=None):
        """Inicializa el agente SQL con todas las capacidades robustas del sistema original"""
        self.db_path = db_path
        self.llm = llm
        self.column_metadata = {}
        self.table_relationships = {}
        self.query_cache = {}
        
        # Sistema de aprendizaje y métricas avanzadas
        self.query_patterns = {}
        self.performance_metrics = {}
        self.medical_knowledge_base = {}
        self.adaptive_weights = {
            'temporal_relevance': 1.0,
            'severity_weight': 2.0,
            'recency_weight': 1.5,
            'complexity_bonus': 1.2
        }
        
        # Datos de esquema y estadísticas completas
        self.table_row_counts = {}
        self.sample_data = {}
        self.knowledge_gaps = {}
        self.learned_patterns = {}
        self.semantic_cache = {}
        
        # Configuración de logging avanzada
        self.logger = logging.getLogger("SQLAgentRobust")
        self.logger.setLevel(logging.INFO)
        
        # Inicializar componentes robustos
        self._initialize_schema_analysis()
        self._initialize_adaptive_learning()
        
        # Inicializar módulos de utilidades
        self.sql_cleaner = SQLCleaner()
        self.sql_executor = SQLExecutor(db_path)
        
        # Inicializar agentes inteligentes especializados
        if INTELLIGENT_DETECTORS_AVAILABLE and self.llm:
            self.name_detector, self.concept_detector = create_intelligent_detectors(self.llm)
            print("🧠 Agentes inteligentes especializados inicializados")
        else:
            self.name_detector = None
            self.concept_detector = None
            print("⚠️ Agentes inteligentes no disponibles, usando detección básica")

    def _initialize_schema_analysis(self):
        """Inicializa el análisis completo y robusto del esquema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Obtener todas las tablas
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Analizar cada tabla con detalle completo
            for table in tables:
                # Información de columnas
                cursor.execute(f"PRAGMA table_info({table});")
                columns = cursor.fetchall()
                
                # Contar filas
                cursor.execute(f"SELECT COUNT(*) FROM {table};")
                row_count = cursor.fetchone()[0]
                self.table_row_counts[table] = row_count
                
                # Obtener datos de muestra más amplios
                cursor.execute(f"SELECT * FROM {table} LIMIT 5;")
                sample_rows = cursor.fetchall()
                
                # Estructurar metadatos completos
                self.column_metadata[table] = {
                    'columns': [
                        {
                            'name': col[1],
                            'type': col[2],
                            'nullable': not col[3],
                            'primary_key': bool(col[5])
                        }
                        for col in columns
                    ],
                    'row_count': row_count,
                    'sample_data': sample_rows
                }
                
            # Análisis de relaciones automático
            self._analyze_table_relationships()
            
            conn.close()
            print(f"✅ Esquema analizado completamente: {len(tables)} tablas, {sum(self.table_row_counts.values())} registros totales")
            
        except Exception as e:
            self.logger.error(f"Error inicializando análisis de esquema: {e}")
            
    def _analyze_table_relationships(self):
        """Analiza las relaciones entre tablas de forma inteligente"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Obtener información de claves foráneas
            for table in self.column_metadata.keys():
                cursor.execute(f"PRAGMA foreign_key_list({table});")
                fk_info = cursor.fetchall()
                
                relationships = []
                for fk in fk_info:
                    relationships.append({
                        'from_column': fk[3],
                        'to_table': fk[2],
                        'to_column': fk[4]
                    })
                
                self.table_relationships[table] = relationships
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error analizando relaciones: {e}")
            
    def _initialize_adaptive_learning(self):
        """Inicializa el sistema de aprendizaje adaptativo robusto"""
        try:
            cache_dir = Path("cache")
            cache_dir.mkdir(exist_ok=True)
            
            # Cargar múltiples tipos de patrones aprendidos
            patterns_file = cache_dir / "learned_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    self.learned_patterns = json.load(f)
                    
            # Cargar conocimiento médico expandido
            medical_kb_file = cache_dir / "medical_knowledge.json"
            if medical_kb_file.exists():
                with open(medical_kb_file, 'r', encoding='utf-8') as f:
                    self.medical_knowledge_base = json.load(f)
                    
            # Cargar métricas de rendimiento
            metrics_file = cache_dir / "performance_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    self.performance_metrics = json.load(f)
                    
        except Exception as e:
            self.logger.error(f"Error inicializando aprendizaje adaptativo: {e}")

    async def process_query(self, query: str, stream_callback=None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Procesa una consulta médica usando el sistema completo robusto con múltiples correcciones LLM
        """
        start_time = time.time()
        
        # MEJORADO: Guardar la query original para uso en fallbacks
        self._current_query = query
        
        try:
            print(f"\n🔍 **ANALIZANDO TU CONSULTA MÉDICA**")
            print(f"📝 Consulta: '{query}'")
            print(f"⏱️ Iniciando procesamiento inteligente...")
            
            # Paso 1: Análisis médico inteligente multi-capa
            print(f"\n🧠 **PASO 1: Analizando conceptos médicos en tu consulta**")
            medical_analysis = await self._analyze_medical_intent_with_llm(query, stream_callback)
            concepts_count = len(medical_analysis.get('concepts', []))
            params_count = len(medical_analysis.get('parameters', []))
            print(f"✅ Encontrados {concepts_count} conceptos médicos y {params_count} parámetros clave")
            
            # Paso 2: Análisis semántico avanzado con contexto
            print(f"\n🔍 **PASO 2: Analizando el significado y contexto**")
            semantic_analysis = await self._enhanced_semantic_analysis(query, medical_analysis)
            entities_count = len(semantic_analysis.get('entities', []))
            qualifiers_count = len(semantic_analysis.get('qualifiers', []))
            print(f"✅ Identificadas {entities_count} entidades médicas y {qualifiers_count} calificadores")
            
            # Paso 3: Exploración activa del esquema para conceptos
            if medical_analysis.get('concepts'):
                print(f"\n🗂️ **PASO 3: Explorando la base de datos médica**")
                schema_exploration = await self._explore_schema_for_concepts(medical_analysis['concepts'])
                mapped_count = len(schema_exploration)
                print(f"✅ Mapeados {mapped_count} conceptos a tablas de la base de datos")
            
            # Paso 4: Mapeo inteligente de tablas con múltiples estrategias
            print(f"\n🎯 **PASO 4: Seleccionando las tablas más relevantes**")
            table_candidates = await self._intelligent_table_mapping(semantic_analysis, medical_analysis, stream_callback)
            print(f"✅ Seleccionadas {len(table_candidates)} tablas relevantes para tu consulta")
            
            # Paso 5: Generación de SQL con correcciones automáticas múltiples
            print(f"\n💾 **PASO 5: Generando consulta SQL inteligente**")
            print(f"🔍 DEBUG: Tablas candidatas: {table_candidates}")
            print(f"🔍 DEBUG: Parámetros médicos: {medical_analysis.get('parameters', [])}")
            sql_result = await self._generate_sql_with_multiple_corrections(query, table_candidates, medical_analysis, semantic_analysis, stream_callback)
            
            if sql_result.get('success', False):
                print(f"✅ Consulta SQL generada exitosamente")
                if sql_result.get('emergency_mode'):
                    print(f"⚠️ Usando modo de emergencia para consulta compleja")
            else:
                print(f"❌ Error generando consulta SQL")
                return sql_result
                
            # Paso 6: Ejecución con sistema robusto de reintentos
            print(f"\n🚀 **PASO 6: Ejecutando consulta en la base de datos**")
            print(f"📊 Consulta original: '{query}'")
            print(f"💾 Consulta SQL generada: {sql_result['sql'][:100]}...")
            print(f"🔍 DEBUG: SQL completo: {sql_result['sql']}")
            print(f"🔍 DEBUG: Éxito del SQL: {sql_result.get('success', False)}")
            
            # Obtener parámetros normalizados para la ejecución
            params = medical_analysis.get('parameters', [])
            if not params:
                params = medical_analysis.get('entities', {}).get('patient_names', [])
            execution_result = await self._execute_sql_with_learning(query, sql_result['sql'], start_time, params, stream_callback=stream_callback)
            
            # DESACTIVADO: No intentar recuperación automática, devolver el error directamente
            if not execution_result['success']:
                print(f"❌ Error ejecutando la consulta en la base de datos")
                print(f"🔍 Detalle del error: {execution_result.get('error', 'Error desconocido')}")
                return execution_result
            

                
            # Paso 7: Interpretación médica completa con LLM
            if execution_result['data']:
                print(f"\n🩺 **PASO 7: Interpretando resultados médicos**")
                interpretation = await self._llm_interpret_results(query, execution_result['data'], stream_callback)
                execution_result['interpretation'] = interpretation
                
                # Análisis de relevancia médica
                print(f"\n📊 **PASO 8: Analizando relevancia clínica**")
                relevance_analysis = await self._analyze_medical_relevance(query, execution_result['data'], stream_callback)
                execution_result['medical_relevance'] = relevance_analysis
            
            # Paso 8: Aprendizaje automático multi-dimensional
            print(f"\n🧠 **PASO 9: Aprendiendo de esta consulta para mejorar futuras búsquedas**")
            await self._learn_from_query_result(query, sql_result['sql'], len(execution_result.get('data', [])), time.time() - start_time)
            
            # Paso 9: Actualización de patrones y conocimiento
            await self._update_learned_patterns(query, sql_result['sql'], execution_result)
            
            print(f"\n✅ **PROCESAMIENTO COMPLETADO**")
            print(f"⏱️ Tiempo total: {time.time() - start_time:.2f} segundos")
            
            return execution_result
            
        except Exception as e:
            error_msg = f"Error crítico en process_query: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            await self._learn_from_error(query, error_msg)
            # Devuelve un diccionario de error en vez de llamar a un método inexistente
            return {
                'success': False,
                'error': error_msg
            }

    async def _generate_sql_with_multiple_corrections(self, query: str, table_candidates: List[str], medical_analysis: Dict[str, Any], semantic_analysis: Dict[str, Any], stream_callback=None) -> Dict[str, Any]:
        """
        VERSIÓN SIMPLIFICADA: Genera SQL usando solo el LLM sin correcciones automáticas
        """
        print(f"🔄 Generando consulta SQL inteligente para tu búsqueda...")
        
        try:
            # Preparar contexto completo para LLM
            # Extraer parámetros normalizados del análisis médico
            params = medical_analysis.get('parameters', [])
            if not params:
                params = medical_analysis.get('entities', {}).get('patient_names', [])
            
            execution_plan = {
                'tables': table_candidates,
                'medical_analysis': medical_analysis,
                'semantic_analysis': semantic_analysis,
                'schema': self.column_metadata,
                'relationships': self.table_relationships,
                'learned_patterns': self.learned_patterns,
                'params': params,  # Añadir parámetros normalizados
                'attempt': 1
            }
            
            if params:
                print(f"   🔧 Usando parámetros: {params}")
            
            # Generar SQL inicial con contexto completo
            print(f"   🧠 Creando consulta SQL personalizada...")
            sql = await self._llm_generate_smart_sql(query, execution_plan, stream_callback)
            
            # Verificación conceptual inteligente con LLM si está disponible
            if self.llm:
                print(f"   🔍 Verificando que la consulta sea médicamente correcta...")
                sql = await self._verify_medical_concept_mapping(query, sql, stream_callback)
            
            # Solo verificación básica de integridad
            if self._basic_sql_integrity_check(sql):
                print(f"✅ Consulta SQL creada exitosamente")
                return {
                    'success': True,
                    'sql': sql,
                    'attempt': 1,
                    'corrections_applied': False,
                    'mode': 'simplified'
                }
            else:
                print(f"❌ Error en la estructura de la consulta SQL")
                return {
                    'success': False,
                    'error': 'SQL generado falló verificación básica de integridad',
                    'sql': sql
                }
                
        except Exception as e:
            error_msg = f"Error en generación SQL simplificada: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'sql': ''
            }

    # Implementar métodos auxiliares básicos para evitar errores
    async def _analyze_medical_intent_with_llm(self, query: str, stream_callback=None) -> Dict[str, Any]:
        """Análisis médico básico con extracción de parámetros normalizados"""
        try:
            # Extraer parámetros de la consulta
            extracted_params = []
            
            # Buscar nombres de pacientes en la consulta
            import re
            name_patterns = [
                r"llamado\s+['\"]([A-Za-zÁáÉéÍíÓóÚúÑñ\s]+)['\"]",  # "llamado 'Ana García'"
                r"paciente\s+(?:llamad[ao]|que\s+se\s+llam[ae])\s+([A-Za-zÁáÉéÍíÓóÚúÑñ\s]+)",
                r"(?:de|del|para|sobre)\s+(?!de|del|para|sobre)([A-Z][a-záéíóúñ]+(?:\s+[A-Z][a-záéíóúñ]+)+)",
                r"(?:paciente|persona)\s+([A-Za-zÁáÉéÍíÓóÚúÑñ\s]+?)\s+(?:tiene|ha|con|ha sido|había)",
                r"\b([A-Z][a-záéíóúñ]+\s+[A-Z][a-záéíóúñ]+(?:\s+[A-Z][a-záéíóúñ]+)*)\b"
            ]
            
            for pattern in name_patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    for match in matches:
                        name = match.strip()
                        # Filtrar nombres muy cortos
                        if len(name) > 3:
                            # NUEVO: Usar LLM para validar si es un nombre válido
                            if self.llm:
                                is_valid_name = await self._validate_name_with_llm(name, query)
                                if is_valid_name:
                                    extracted_params.append(name)
                                    print(f"🔧 Parámetro extraído: '{name}' (validado por LLM)")
                                    break
                            else:
                                # Fallback sin LLM: solo verificar longitud
                                extracted_params.append(name)
                                print(f"🔧 Parámetro extraído: '{name}' (fallback)")
                            break
                    if extracted_params:
                        break
            
            # MEJORADO: Si no se encontró ningún nombre específico, usar LLM para extraer nombres
            if not extracted_params and self.llm:
                llm_extracted_name = await self._extract_name_with_llm(query)
                if llm_extracted_name:
                    extracted_params.append(llm_extracted_name)
                    print(f"🔧 Parámetro extraído por LLM: '{llm_extracted_name}'")
            
            return {
                'concepts': ['paciente', 'consulta'],
                'parameters': extracted_params if extracted_params else [query],
                'intent': 'búsqueda médica',
                'priority': 'media',
                'medical_domain': 'general',
                'original_query': query,
                'clinical_intent': 'búsqueda de pacientes',
                'medical_concepts': ['paciente', 'consulta'],
                'entities': {
                    'patient_names': extracted_params
                }
            }
        except Exception as e:
            print(f"❌ Error en análisis médico: {e}")
            return await self._create_fallback_concept_analysis(query)

    async def _enhanced_semantic_analysis(self, query: str, medical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis semántico avanzado usando LLM para extraer entidades y contexto"""
        try:
            if not self.llm:
                # Fallback básico si no hay LLM
                return {
                    'entities': [],
                    'qualifiers': [],
                    'complexity': 1,
                    'query_type': 'búsqueda'
                }
            
            prompt = f"""Analiza: "{query}". Responde SOLO con: {{"entities": [{{"type": "patient_name|vital_signs", "value": "valor", "confidence": 0.9}}], "query_type": "busqueda_paciente", "complexity": 3}}"""
            
            response = _call_openai_native(self.llm, prompt, task_description="Análisis semántico profundo")
            result = self._try_parse_llm_json(response.content)
            
            if result:
                entities = result.get('entities', [])
                qualifiers = result.get('qualifiers', [])
                
                print(f"   📊 Análisis semántico LLM: {len(entities)} entidades, {len(qualifiers)} calificadores")
                print(f"   🎯 Tipo de consulta: {result.get('query_type', 'desconocido')}")
                print(f"   🧩 Complejidad: {result.get('complexity', 1)}/10")
                
                return result
            
            # Fallback si el LLM falla
            return {
                'entities': [],
                'qualifiers': [],
                'complexity': 1,
                'query_type': 'búsqueda'
            }
            
        except Exception as e:
            print(f"   ❌ Error en análisis semántico: {e}")
        return {
            'entities': [],
            'qualifiers': [],
            'complexity': 1,
            'query_type': 'búsqueda'
        }

    async def _explore_schema_for_concepts(self, concepts: List[str]) -> Dict[str, Any]:
        """Explora el esquema usando LLM para mapear conceptos médicos a tablas de forma inteligente"""
        try:
            if not self.llm or not concepts:
                return {}
            
            # Obtener información del esquema
            schema_info = self._get_schema_context()
            
            prompt = f"""Mapea conceptos {concepts} a tablas. Responde SOLO con: {{"concept_mappings": [{{"concept": "paciente", "mapped_tables": [{{"table": "PATI_PATIENTS", "confidence": 1.0}}]}}], "global_confidence": 0.9}}"""
            
            response = _call_openai_native(self.llm, prompt, task_description="Mapeo inteligente de conceptos a esquema")
            result = self._try_parse_llm_json(response.content)
            
            if result:
                # Transformar el resultado al formato esperado
                schema_mapping = {}
                concept_mappings = result.get('concept_mappings', [])
                
                for mapping in concept_mappings:
                    concept = mapping.get('concept', '')
                    tables_info = mapping.get('mapped_tables', [])
                    
                    # Extraer solo los nombres de las tablas
                    tables = [t['table'] for t in tables_info if 'table' in t]
                    
                    # Calcular confianza promedio
                    confidences = [t.get('confidence', 0.5) for t in tables_info if 'confidence' in t]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
                    
                    if tables:
                        schema_mapping[concept] = {
                            'tables': tables,
                            'confidence': avg_confidence,
                            'reasoning': mapping.get('reasoning', '')
                        }
                
                print(f"   📊 Esquema explorado con LLM: {len(schema_mapping)} conceptos mapeados")
                print(f"   🎯 Confianza global: {result.get('global_confidence', 0)}")
                
                return schema_mapping
            
            return {}
            
        except Exception as e:
            print(f"   ❌ Error explorando esquema con LLM: {e}")
        return {}

    async def _intelligent_table_mapping(self, semantic_analysis: Dict[str, Any], medical_analysis: Dict[str, Any], stream_callback=None) -> List[str]:
        """Mapeo inteligente de tablas basado en análisis médico usando LLM"""
        
        print(f"🔍 Analizando qué tablas de la base de datos contienen la información que necesitas...")
        
        if not self.llm:
            print("⚠️ Usando selección automática de tablas (modo básico)")
            return list(self.column_metadata.keys())[:5]
        
        try:
            original_query = medical_analysis.get('original_query', '')
            
            # PASO 1: Análisis inteligente de la consulta para identificar conceptos médicos
            print(f"   🧠 Identificando conceptos médicos en tu consulta...")
            concept_analysis = await self._analyze_medical_concepts_in_query(original_query)
            
            # PASO 2: Mapeo inteligente de conceptos a tipos de datos
            print(f"   🔍 Mapeando conceptos a tipos de información en la base de datos...")
            data_type_mapping = await self._map_concepts_to_data_types(concept_analysis)
            
            # PASO 3: Selección inteligente de tablas basada en el mapeo
            print(f"   📋 Seleccionando las tablas más apropiadas...")
            selected_tables = await self._select_tables_by_data_types(data_type_mapping)
            
            print(f"✅ Encontradas {len(selected_tables)} tablas relevantes para tu consulta")
            return selected_tables
            
        except Exception as e:
            print(f"❌ Error en mapeo inteligente: {e}")
            return list(self.column_metadata.keys())[:5]

    async def _analyze_medical_concepts_in_query(self, query: str) -> Dict[str, Any]:
        """Analiza conceptos médicos en la consulta usando LLM inteligente"""
        if not self.llm:
            return {'concepts': [], 'medical_terms': [], 'context': 'LLM no disponible'}
        
        prompt = f"""
Eres un experto en terminología médica. Analiza esta consulta y identifica conceptos médicos.

CONSULTA: "{query}"

INSTRUCCIONES:
1. Identifica conceptos médicos en la consulta
2. Clasifica por tipo: signos_vitales, diagnostico, medicamento, procedimiento, temporal, datos_paciente
3. Para consultas como "qué datos tiene X", clasifica como "datos_paciente"
4. Responde SOLO con JSON válido, sin texto adicional

CONCEPTOS A IDENTIFICAR:
- Signos vitales: tensión, presión, frecuencia, temperatura, peso, talla, saturación
- Diagnósticos: diabetes, hipertensión, enfermedades
- Medicamentos: nombres de fármacos, dosis
- Datos de paciente: información personal, historial
- Conceptos temporales: último, reciente, actual

JSON DE RESPUESTA:
{{
  "conceptos_identificados": [
    {{
      "concepto": "nombre del concepto",
      "tipo": "signos_vitales|diagnostico|medicamento|datos_paciente|temporal",
      "terminos_relacionados": ["sinónimo1", "sinónimo2"],
      "contexto_clinico": "descripción breve"
    }}
  ],
  "contexto_principal": "tipo de consulta",
  "entidades_clave": ["entidad1", "entidad2"],
  "confianza_analisis": 0.8
}}
"""
        
        try:
            response = await self.llm.ainvoke(prompt)
            content = str(response.content)
            
            # Limpiar contenido antes de parsear
            content = content.strip()
            
            # Intentar parsear directamente primero
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # Si falla, extraer JSON con regex más robusto
                import re
                # Buscar JSON que empiece con { y termine con }
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                        return result
                    except json.JSONDecodeError as e:
                        print(f"   ⚠️ Error parseando JSON extraído: {e}")
                        # Intentar limpiar el JSON antes de parsear
                        try:
                            # Remover texto antes y después del JSON
                            cleaned_content = re.sub(r'^[^{]*', '', content)
                            cleaned_content = re.sub(r'[^}]*$', '', cleaned_content)
                            result = json.loads(cleaned_content)
                            return result
                        except:
                            print(f"   ⚠️ No se pudo limpiar el JSON")
                            return await self._create_fallback_concept_analysis(query)
                else:
                    print(f"   ⚠️ No se encontró JSON en la respuesta")
                    return await self._create_fallback_concept_analysis(query)
                
        except Exception as e:
            print(f"   ❌ Error en análisis de conceptos médicos: {e}")
            return await self._create_fallback_concept_analysis(query)
    
    async def _create_fallback_concept_analysis(self, query: str) -> Dict[str, Any]:
        """Crea un análisis de conceptos inteligente usando LLM cuando falla el análisis principal"""
        try:
            if not self.llm:
                return self._create_basic_fallback_analysis(query)
            
            prompt = f"""
Eres un experto en análisis de consultas médicas. Analiza esta consulta y identifica conceptos médicos de forma inteligente.

CONSULTA: "{query}"

INSTRUCCIONES:
1. Identifica TODOS los conceptos médicos y entidades relevantes
2. Clasifica por tipo: datos_paciente, signos_vitales, diagnostico, medicamento, temporal
3. Para consultas como "qué datos tiene X", identifica como "datos de paciente"
4. Para consultas sobre constantes vitales, identifica como "signos vitales"
5. Extrae nombres de pacientes, diagnósticos, medicamentos, etc.

CONCEPTOS A IDENTIFICAR:
- Datos de paciente: información personal, historial general
- Signos vitales: tensión, presión, frecuencia, temperatura, peso, talla, saturación
- Diagnósticos: diabetes, hipertensión, enfermedades, condiciones
- Medicamentos: nombres de fármacos, tratamientos, dosis
- Conceptos temporales: último, reciente, actual, histórico

Responde SOLO con este JSON válido:
{{
  "conceptos_identificados": [
    {{
      "concepto": "nombre del concepto médico",
      "tipo": "datos_paciente|signos_vitales|diagnostico|medicamento|temporal",
      "terminos_relacionados": ["sinónimo1", "sinónimo2"],
      "contexto_clinico": "descripción del contexto médico"
    }}
  ],
  "contexto_principal": "tipo de consulta médica",
  "entidades_clave": ["entidad1", "entidad2"],
  "confianza_analisis": 0.8
}}
"""
            
            response = await self.llm.ainvoke(prompt)
            content = str(response.content).strip()
            
            # Intentar parsear directamente
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # Si falla, extraer JSON con regex
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                        return result
                    except json.JSONDecodeError:
                        print(f"   ⚠️ Error parseando JSON del fallback LLM")
                        return self._create_basic_fallback_analysis(query)
                else:
                    print(f"   ⚠️ No se encontró JSON en respuesta del fallback LLM")
                    return self._create_basic_fallback_analysis(query)
                    
        except Exception as e:
            print(f"   ❌ Error en fallback LLM: {e}")
            return self._create_basic_fallback_analysis(query)
    
    async def _create_intelligent_fallback_analysis(self, query: str) -> Dict[str, Any]:
        """Análisis inteligente con LLM cuando falla el análisis principal"""
        try:
            if not self.llm:
                return self._create_basic_fallback_analysis(query)
            
            prompt = f"""
Eres un experto en análisis de consultas médicas. Analiza esta consulta y identifica conceptos médicos de forma inteligente.

CONSULTA: "{query}"

INSTRUCCIONES:
1. Identifica TODOS los conceptos médicos y entidades relevantes
2. Clasifica por tipo: datos_paciente, signos_vitales, diagnostico, medicamento, temporal
3. Para consultas como "qué datos tiene X", identifica como "datos de paciente"
4. Para consultas sobre constantes vitales, identifica como "signos vitales"
5. Para consultas sobre diagnósticos, identifica como "diagnostico"
6. Para consultas sobre medicamentos, identifica como "medicamento"
7. Extrae nombres de pacientes, diagnósticos, medicamentos, etc.

CONCEPTOS A IDENTIFICAR:
- Datos de paciente: información personal, historial general
- Signos vitales: tensión, presión, frecuencia, temperatura, peso, talla, saturación, constantes vitales
- Diagnósticos: diabetes, hipertensión, enfermedades, condiciones
- Medicamentos: nombres de fármacos, tratamientos, dosis
- Conceptos temporales: último, reciente, actual, histórico

TABLAS CORRECTAS PARA CADA CONCEPTO:
- Signos vitales: EPIS_DIAGNOSTICS (DIAG_OBSERVATION) 
- Diagnósticos: EPIS_DIAGNOSTICS (DIAG_OBSERVATION)
- Medicamentos: PATI_USUAL_MEDICATION (PAUM_OBSERVATIONS)
- Datos de paciente: PATI_PATIENTS

REGLAS ESPECÍFICAS PARA CONSTANTES VITALES:
- Buscar en EPIS_DIAGNOSTICS.DIAG_OBSERVATION
- Los datos se almacenan como texto descriptivo
- Formato típico: "Tipo: valor unidad" (ej: "Presión arterial: 120/80 mmHg")
- Buscar patrones como: presión, tensión, temperatura, frecuencia, peso, talla, saturación

Responde SOLO con este JSON válido:
{{
  "conceptos_identificados": [
    {{
      "concept": "tipo_de_concepto",
      "value": "valor_extraido",
      "confidence": 0.95
    }}
  ],
  "tablas_requeridas": ["lista", "de", "tablas"],
  "relaciones_criticas": ["PATI_ID"],
  "tipos_dominantes": ["tipo_principal"],
  "original_query": "{query}"
}}
"""
            
            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result:
                print(f"   ✅ Análisis inteligente exitoso")
                return result
            else:
                print(f"   ⚠️ Fallback a análisis básico")
                return self._create_basic_fallback_analysis(query)
                
        except Exception as e:
            print(f"   ❌ Error en análisis inteligente: {e}")
            return self._create_basic_fallback_analysis(query)
    
    def _create_basic_fallback_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback básico cuando no hay LLM disponible"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['qué datos', 'que datos', 'datos de', 'información de']):
            return {
                'conceptos_identificados': [
                    {
                        'concepto': 'datos de paciente',
                        'tipo': 'datos_paciente',
                        'terminos_relacionados': ['información', 'historial'],
                        'contexto_clinico': 'Consulta de información general del paciente',
                        'tabla_recomendada': 'PATI_PATIENTS'
                    }
                ],
                'contexto_principal': 'consulta_datos_paciente',
                'entidades_clave': ['paciente'],
                'confianza_analisis': 0.7
            }
        elif any(word in query_lower for word in ['constantes', 'vitales', 'signos', 'tensión', 'presión', 'frecuencia', 'temperatura', 'peso', 'talla', 'saturación']):
            return {
                'conceptos_identificados': [
                    {
                        'concepto': 'signos vitales',
                        'tipo': 'signos_vitales',
                        'terminos_relacionados': ['constantes', 'vitales', 'tensión', 'presión', 'frecuencia', 'temperatura', 'peso', 'talla', 'saturación'],
                        'contexto_clinico': 'Consulta de constantes vitales',
                        'tabla_recomendada': 'EPIS_DIAGNOSTICS'
                    }
                ],
                'contexto_principal': 'consulta_signos_vitales',
                'entidades_clave': ['signos vitales'],
                'confianza_analisis': 0.8
            }
        else:
            return {
                'conceptos_identificados': [],
                'contexto_principal': 'consulta_general',
                'entidades_clave': [],
                'confianza_analisis': 0.5
            }
    
    async def _map_concepts_to_data_types(self, concept_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Mapea los conceptos identificados a tipos de datos en la base de datos"""
        try:
            conceptos = concept_analysis.get('conceptos_identificados', [])
            tipos_dominantes = concept_analysis.get('tipos_dominantes', [])
            
            prompt = f"""Mapea conceptos {conceptos} a tablas. Responde SOLO con: {{"mapeo_conceptos": [{{"concepto": "paciente", "tipo_concepto": "datos_paciente", "tabla_principal": "PATI_PATIENTS"}}], "tablas_requeridas": ["PATI_PATIENTS"], "relaciones_criticas": ["PATI_ID"]}}"""
            
            # SOLUCIÓN LLM-FIRST: Usar LLM siempre que esté disponible
            if self.llm:
                try:
                    response = _call_openai_native(self.llm, prompt, task_description="Mapeo de conceptos a tipos de datos")
                    result = self._try_parse_llm_json(response.content)
                    
                    if result:
                        print(f"   🔗 Mapeo de conceptos completado")
                        print(f"   📋 Tablas requeridas: {result.get('tablas_requeridas', [])}")
                        return result
                    else:
                        print(f"   ⚠️ LLM no devolvió resultado válido, usando fallback inteligente")
                        return self._create_llm_fallback_mapping(concept_analysis)
                except Exception as e:
                    print(f"   ⚠️ Error con LLM: {e}")
                    return self._create_llm_fallback_mapping(concept_analysis)
            
            # Si no hay LLM disponible
            print(f"   🔧 LLM no disponible, usando fallback inteligente")
            return self._create_llm_fallback_mapping(concept_analysis)
            
        except Exception as e:
            print(f"❌ Error mapeando conceptos: {e}")
            return {"mapeo_conceptos": [], "tablas_requeridas": [], "relaciones_criticas": []}
    
    def _create_llm_fallback_mapping(self, concept_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        FALLBACK INTELIGENTE: Usa LLM con prompt simplificado cuando el principal falla
        """
        try:
            concepts = concept_analysis.get('concepts', [])
            query = concept_analysis.get('original_query', '')
            
            # Si no hay LLM, usar mapeo básico
            if not self.llm:
                return self._create_basic_mapping(concept_analysis)
            
            # FALLBACK CON LLM: Prompt simplificado y más robusto
            fallback_prompt = f"""
Analiza esta consulta médica y mapea los conceptos a tablas de base de datos.

CONSULTA: "{query}"
CONCEPTOS DETECTADOS: {[c.get('concept', '') for c in concepts]}

TABLAS DISPONIBLES:
- PATI_PATIENTS: Datos básicos de pacientes
- PATI_USUAL_MEDICATION: Medicación habitual
- MEDI_ACTIVE_INGREDIENTS: Ingredientes activos
- ACCI_PATIENT_CONDITIONS: Condiciones médicas
- EPIS_DIAGNOSTICS: Diagnósticos y observaciones (incluye constantes vitales)
- APPO_APPOINTMENTS: Citas y observaciones médicas
- PATI_PATIENT_ALLERGIES: Alergias

REGLAS SIMPLES:
- Si menciona "datos", "información", "paciente" → PATI_PATIENTS
- Si menciona "constantes vitales", "signos vitales", "tensión", "presión", "frecuencia" → EPIS_DIAGNOSTICS + APPO_APPOINTMENTS
- Si menciona "medicación", "medicamento" → PATI_USUAL_MEDICATION + MEDI_ACTIVE_INGREDIENTS
- Si menciona "diagnóstico", "condición" → EPIS_DIAGNOSTICS
- Si menciona "alergia" → PATI_PATIENT_ALLERGIES

IMPORTANTE: Para constantes vitales, buscar en EPIS_DIAGNOSTICS.DIAG_OBSERVATION y APPO_APPOINTMENTS.APPO_OBSERVATIONS

Responde SOLO con este JSON:
{{
  "tablas_requeridas": ["lista", "de", "tablas"],
  "relaciones_criticas": ["PATI_ID"],
  "tipos_dominantes": ["datos_paciente"]
}}
"""
            try:
                response = _call_openai_native(self.llm, fallback_prompt, task_description="Fallback LLM")
                result = self._try_parse_llm_json(response.content)
                
                if result:
                    print(f"   🔧 Fallback LLM exitoso")
                    return {
                        "mapeo_conceptos": concepts,
                        "tablas_requeridas": result.get('tablas_requeridas', ['PATI_PATIENTS']),
                        "relaciones_criticas": result.get('relaciones_criticas', ['PATI_ID']),
                        "tipos_dominantes": result.get('tipos_dominantes', ['datos_paciente']),
                        "original_query": query
                    }
            except Exception as e:
                print(f"   ⚠️ Fallback LLM también falló: {e}")
            
            # ÚLTIMO FALLBACK: Mapeo básico sin LLM
            return self._create_basic_mapping(concept_analysis)
            
        except Exception as e:
            print(f"❌ Error en fallback inteligente: {e}")
            return {
                "mapeo_conceptos": [],
                "tablas_requeridas": ["PATI_PATIENTS"],
                "relaciones_criticas": ["PATI_ID"],
                "tipos_dominantes": ["datos_paciente"],
                "original_query": query
            }
    
    def _create_basic_mapping(self, concept_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mapeo básico sin LLM como último recurso
        """
        try:
            concepts = concept_analysis.get('concepts', [])
            query = concept_analysis.get('original_query', '')
            
            # Mapeo básico basado en patrones
            tablas_requeridas = ['PATI_PATIENTS']  # Siempre incluir pacientes
            
            # Patrones básicos
            query_lower = query.lower()
            if any(word in query_lower for word in ['medicación', 'medicamento', 'medicina']):
                tablas_requeridas.extend(['PATI_USUAL_MEDICATION', 'MEDI_ACTIVE_INGREDIENTS'])
            if any(word in query_lower for word in ['diagnóstico', 'condición', 'enfermedad']):
                tablas_requeridas.extend(['ACCI_PATIENT_CONDITIONS', 'EPIS_DIAGNOSTICS'])
            if any(word in query_lower for word in ['alergia', 'alérgico']):
                tablas_requeridas.extend(['PATI_PATIENT_ALLERGIES'])
            
            # Validar que existan
            tablas_validas = [t for t in tablas_requeridas if t in self.column_metadata]
            
            print(f"   🔧 Mapeo básico: {tablas_validas}")
            
            return {
                "mapeo_conceptos": concepts,
                "tablas_requeridas": tablas_validas,
                "relaciones_criticas": ["PATI_ID"],
                "tipos_dominantes": ["datos_paciente"],
                "original_query": query
            }
            
        except Exception as e:
            print(f"❌ Error en mapeo básico: {e}")
            return {
                "mapeo_conceptos": [],
                "tablas_requeridas": ["PATI_PATIENTS"],
                "relaciones_criticas": ["PATI_ID"],
                "tipos_dominantes": ["datos_paciente"],
                "original_query": query
            }

    async def _select_tables_by_data_types(self, data_type_mapping: Dict[str, Any]) -> List[str]:
        """Selecciona las tablas finales basándose en el mapeo de tipos de datos"""
        try:
            tablas_requeridas = data_type_mapping.get('tablas_requeridas', [])
            
            # Validar que las tablas existan en el esquema real
            tablas_validas = [t for t in tablas_requeridas if t in self.column_metadata]
            
            if not tablas_validas:
                print(f"⚠️ No se encontraron tablas válidas, usando fallback inteligente")
                print(f"   🔍 DEBUG: Las tablas requeridas no existen en el esquema")
                
                # FALLBACK INTELIGENTE: Crear medical_analysis básico desde data_type_mapping
                # MEJORADO: Obtener la query original del contexto si está disponible
                original_query = data_type_mapping.get('original_query', '')
                if not original_query:
                    # Intentar obtener del contexto de la consulta actual
                    original_query = getattr(self, '_current_query', '')
                
                medical_analysis = {
                    'original_query': original_query,  # MEJORADO: Usar query original si está disponible
                    'conceptos_identificados': data_type_mapping.get('mapeo_conceptos', []),
                    'tipos_dominantes': data_type_mapping.get('tipos_dominantes', [])
                }
                fallback_tables = await self._get_intelligent_fallback_tables(medical_analysis)
                print(f"   🧠 Fallback inteligente seleccionó: {fallback_tables}")
                return fallback_tables
            
            # Asegurar que PATI_PATIENTS esté siempre incluida
            if 'PATI_PATIENTS' not in tablas_validas:
                tablas_validas.insert(0, 'PATI_PATIENTS')
            
            # Limitar a máximo 5 tablas
            tablas_finales = tablas_validas[:5]
            
            print(f"   ✅ Tablas seleccionadas: {', '.join(tablas_finales)}")
            return tablas_finales
            
        except Exception as e:
            print(f"❌ Error seleccionando tablas: {e}")
            return list(self.column_metadata.keys())[:5]

    async def _get_intelligent_fallback_tables(self, medical_analysis: Dict[str, Any]) -> List[str]:
        """Selección inteligente de tablas usando LLM cuando no hay mapeo específico"""
        try:
            original_query = medical_analysis.get('original_query', '')
            
            print(f"   🔍 Analizando consulta para fallback inteligente...")
            
            # Si no hay LLM disponible, usar fallback básico
            if not self.llm:
                return await self._get_basic_fallback_tables(original_query)
            
            # Usar LLM para análisis inteligente
            return await self._get_llm_fallback_tables(original_query)
            
        except Exception as e:
            print(f"❌ Error en fallback inteligente: {e}")
            return list(self.column_metadata.keys())[:5]
    
    async def _get_basic_fallback_tables(self, query: str) -> List[str]:
        """Análisis inteligente con LLM para determinar las tablas necesarias cuando no hay LLM disponible"""
        try:
            if not self.llm:
                # Fallback sin LLM: usar tablas básicas
                basic_tables = ['PATI_PATIENTS', 'EPIS_DIAGNOSTICS', 'PATI_USUAL_MEDICATION']
                valid_tables = [t for t in basic_tables if t in self.column_metadata]
                print(f"   🔧 Fallback sin LLM: usando tablas básicas: {valid_tables}")
                return valid_tables[:3]
            
            # ANÁLISIS INTELIGENTE CON LLM
            prompt = f"""
Eres un experto en bases de datos médicas. Analiza esta consulta y determina qué tablas son necesarias.

CONSULTA: "{query}"

TABLAS DISPONIBLES:
- PATI_PATIENTS: Datos básicos de pacientes
- PATI_USUAL_MEDICATION: Medicación habitual
- MEDI_ACTIVE_INGREDIENTS: Ingredientes activos
- EPIS_DIAGNOSTICS: Diagnósticos y observaciones (incluye constantes vitales)
- APPO_APPOINTMENTS: Citas y observaciones médicas
- PATI_PATIENT_ALLERGIES: Alergias

ANÁLISIS REQUERIDO:
1. Identifica el tipo de información solicitada
2. Selecciona las tablas más relevantes (máximo 5)
3. Siempre incluye PATI_PATIENTS si se mencionan datos de pacientes

REGLAS ESPECÍFICAS PARA CONSTANTES VITALES:
- Buscar en EPIS_DIAGNOSTICS.DIAG_OBSERVATION
- Los datos se almacenan como texto descriptivo
- Formato típico: "Tipo: valor unidad" (ej: "Presión arterial: 120/80 mmHg")
- Buscar patrones como: presión, tensión, temperatura, frecuencia, peso, talla, saturación

EJEMPLOS:
- "medicación de Ana" → PATI_PATIENTS, PATI_USUAL_MEDICATION
- "constantes vitales" → PATI_PATIENTS, EPIS_DIAGNOSTICS
- "diagnósticos" → PATI_PATIENTS, EPIS_DIAGNOSTICS
- "alergias" → PATI_PATIENTS, PATI_PATIENT_ALLERGIES

Responde SOLO con este JSON:
{{
  "tablas_seleccionadas": ["lista", "de", "tablas"],
  "tipo_consulta": "tipo de consulta detectado"
}}
"""
            
            try:
                response = _call_openai_native(self.llm, prompt, task_description="Análisis inteligente de tablas")
                result = self._try_parse_llm_json(response.content)
                
                if result and result.get('tablas_seleccionadas'):
                    tablas = result.get('tablas_seleccionadas', [])
                    tipo_consulta = result.get('tipo_consulta', 'desconocido')
                    
                    print(f"   🧠 LLM analizó: {tipo_consulta}")
                    
                    # Validar que las tablas existan
                    tablas_validas = [t for t in tablas if t in self.column_metadata]
                    
                    # Asegurar que PATI_PATIENTS esté incluida si hay datos de pacientes
                    if tablas_validas and 'PATI_PATIENTS' not in tablas_validas and any('PATI_' in t for t in tablas_validas):
                        tablas_validas.insert(0, 'PATI_PATIENTS')
                    
                    # Limitar a máximo 5 tablas
                    final_tables = tablas_validas[:5]
                    
                    print(f"   🧠 LLM seleccionó {len(final_tables)} tablas: {', '.join(final_tables)}")
                    return final_tables
                    
            except Exception as e:
                print(f"   ⚠️ Error con LLM: {e}")
            
            # Fallback si LLM falla
            fallback_tables = ['PATI_PATIENTS', 'EPIS_DIAGNOSTICS']
            valid_tables = [t for t in fallback_tables if t in self.column_metadata]
            print(f"   🔧 Fallback básico: {valid_tables}")
            return valid_tables
            
        except Exception as e:
            print(f"❌ Error en análisis de tablas: {e}")
            return ['PATI_PATIENTS']
    
    async def _get_llm_fallback_tables(self, query: str) -> List[str]:
        """Análisis inteligente con LLM para determinar las tablas necesarias"""
        try:
            # Obtener esquema disponible
            schema_info = self._get_schema_context()
            
            prompt = f"""
Eres un experto en bases de datos médicas. Analiza esta consulta y determina qué tablas de la base de datos son necesarias para responderla.

CONSULTA: "{query}"

ESQUEMA DISPONIBLE:
{schema_info}

TABLAS PRINCIPALES Y SU PROPÓSITO:
- PATI_PATIENTS: Información básica de pacientes (nombres, fechas, IDs)
- PATI_USUAL_MEDICATION: Medicación habitual de pacientes
- MEDI_ACTIVE_INGREDIENTS: Ingredientes activos de medicamentos
- ACCI_PATIENT_CONDITIONS: Condiciones médicas crónicas de pacientes
- EPIS_DIAGNOSTICS: Diagnósticos de episodios/visitas médicas
- PATI_PATIENT_ALLERGIES: Alergias de pacientes
- PATI_PATIENT_ADDRESSES: Direcciones de pacientes
- PATI_PATIENT_CONTACTS: Información de contacto de pacientes

ANÁLISIS REQUERIDO:
1. Identifica el tipo de información que se está solicitando
2. Determina qué tablas contienen esa información
3. Considera las relaciones entre tablas (siempre necesitas PATI_PATIENTS para datos de pacientes)
4. Selecciona solo las tablas más relevantes (máximo 5)

EJEMPLOS:
- "último paciente registrado" → PATI_PATIENTS (ordenar por fecha de registro)
- "medicación de Ana García" → PATI_PATIENTS + PATI_USUAL_MEDICATION
- "diagnósticos de diabetes" → PATI_PATIENTS + ACCI_PATIENT_CONDITIONS
- "alergias de pacientes" → PATI_PATIENT_ALLERGIES
- "historia clínica completa" → PATI_PATIENTS + PATI_USUAL_MEDICATION + ACCI_PATIENT_CONDITIONS

Responde SOLO con este JSON:
{{
  "tablas_seleccionadas": ["lista", "de", "tablas"],
  "razon": "explicación breve de por qué se seleccionaron estas tablas",
  "tipo_consulta": "tipo de consulta detectado (paciente, medicación, diagnóstico, etc.)"
}}
"""
            
            # SOLUCIÓN LLM-FIRST: Usar LLM siempre que esté disponible
            if self.llm:
                try:
                    response = _call_openai_native(self.llm, prompt, task_description="Análisis inteligente de tablas")
                    result = self._try_parse_llm_json(response.content)
                    
                    if result and result.get('tablas_seleccionadas'):
                        tablas = result.get('tablas_seleccionadas', [])
                        razon = result.get('razon', 'Sin razón especificada')
                        tipo_consulta = result.get('tipo_consulta', 'desconocido')
                        
                        print(f"   🧠 LLM analizó: {tipo_consulta} - Razón: {razon}")
                        
                        # Validar que las tablas existan en el esquema
                        tablas_validas = [t for t in tablas if t in self.column_metadata]
                        
                        # Asegurar que PATI_PATIENTS esté siempre incluida si hay datos de pacientes
                        if tablas_validas and 'PATI_PATIENTS' not in tablas_validas and any('PATI_' in t for t in tablas_validas):
                            tablas_validas.insert(0, 'PATI_PATIENTS')
                        
                        # Limitar a máximo 5 tablas
                        final_tables = tablas_validas[:5]
                        
                        print(f"   🧠 LLM seleccionó {len(final_tables)} tablas: {', '.join(final_tables)}")
                        
                        return final_tables
                    else:
                        print(f"   ⚠️ LLM no devolvió resultado válido, usando fallback inteligente")
                        return await self._get_llm_fallback_tables(query)
                except Exception as e:
                    print(f"   ⚠️ Error con LLM: {e}")
                    return await self._get_llm_fallback_tables(query)
            
            # Si no hay LLM disponible
            print(f"   🔧 LLM no disponible, usando fallback básico para: '{query}'")
            return await self._get_basic_fallback_tables(query)
            
        except Exception as e:
            print(f"❌ Error en análisis LLM: {e}")
            print(f"   🔍 DEBUG: Usando fallback básico para: '{query}'")
            return await self._get_basic_fallback_tables(query)

    async def _llm_generate_smart_sql(self, query: str, execution_plan: Dict[str, Any], stream_callback=None) -> str:
        """Generación genérica de SQL usando LLM con prompts ultra-cortos y reintentos inteligentes"""
        try:
            print(f"🧠 Generando SQL genérico para: '{query}'")
            
            if not self.llm:
                print("⚠️ LLM no disponible, usando SQL básico")
                return self._generate_basic_sql(query)
            
            # Obtener tablas disponibles
            available_tables = list(self.column_metadata.keys())
            print(f"📊 Tablas disponibles: {len(available_tables)} tablas")
            
            # PASO 1: Intento principal con prompt genérico
            sql = await self._generate_sql_attempt_1(query, available_tables)
            if sql and self._is_valid_sql(sql):
                # Validar dinámicamente que las tablas existan
                validated_sql = await self._validate_and_fix_tables(sql)
                if validated_sql != sql:
                    print(f"🔧 SQL corregido dinámicamente")
                    return validated_sql
                return sql
            
            # PASO 2: Reintento con contexto de error
            print(f"🔄 Reintentando con contexto de error...")
            sql = await self._generate_sql_attempt_2(query, available_tables, "SQL incompleto o inválido")
            if sql and self._is_valid_sql(sql):
                validated_sql = await self._validate_and_fix_tables(sql)
                if validated_sql != sql:
                    print(f"🔧 SQL corregido dinámicamente")
                    return validated_sql
                return sql
            
            # PASO 3: Intento ultra-restrictivo
            print(f"🔄 Intento ultra-restrictivo...")
            sql = await self._generate_sql_attempt_3(query, available_tables)
            if sql and self._is_valid_sql(sql):
                validated_sql = await self._validate_and_fix_tables(sql)
                if validated_sql != sql:
                    print(f"🔧 SQL corregido dinámicamente")
                    return validated_sql
                return sql
            
            # PASO 4: Último intento con validación estricta
            print(f"🔄 Último intento con validación estricta...")
            sql = await self._generate_sql_attempt_4(query, available_tables)
            if sql and self._is_valid_sql(sql):
                validated_sql = await self._validate_and_fix_tables(sql)
                if validated_sql != sql:
                    print(f"🔧 SQL corregido dinámicamente")
                    return validated_sql
                return sql
            
            # Si todos fallan, devolver error claro
            print(f"❌ Todos los intentos fallaron. Mostrando error claro.")
            return "-- ERROR: No se puede responder porque no existe ninguna tabla con la información solicitada en el esquema actual. --"
            
        except Exception as e:
            print(f"❌ Error en generación de SQL: {e}")
            return self._generate_basic_sql(query)
    
    async def _generate_sql_attempt_1(self, query: str, available_tables: List[str]) -> str:
        """Primer intento: prompt genérico y corto con validación estricta de esquema"""
        try:
            # Obtener esquema detallado de las tablas disponibles
            schema_info = self._get_detailed_schema_for_tables(available_tables)
            
            prompt = f"""Eres un experto en SQL para bases de datos médicas. Genera SQL válido para esta consulta.

CONSULTA: "{query}"

ESQUEMA EXACTO DE TABLAS DISPONIBLES:
{schema_info}

REGLAS CRÍTICAS:
1. SOLO usa tablas y columnas que aparecen en el esquema exacto
2. NO inventes nombres de columnas
3. NO inventes nombres de tablas
4. Usa los nombres EXACTOS de columnas del esquema
5. Responde SOLO con el SQL, sin explicaciones ni texto adicional
6. Si no puedes generar SQL válido, responde: -- ERROR: No se puede generar SQL válido --

EJEMPLOS DE NOMBRES CORRECTOS:
- PATI_PATIENTS.PATI_NAME (no "patient_name")
- DOCS_DOCUMENTS.DOCS_DOCUMENT_DATE (no "document_date")
- EPIS_DIAGNOSTICS.DIAG_OBSERVATION (no "observation")

SQL:"""
            
            response = _call_openai_native(self.llm, [{"role": "user", "content": prompt}])
            sql = self._clean_sql_response(response.content)
            
            # Validar que el SQL generado solo use columnas existentes
            validated_sql = await self._validate_and_correct_columns(sql, available_tables)
            
            return validated_sql
            
        except Exception as e:
            print(f"❌ Error en intento 1: {e}")
            return ""
    
    async def _validate_and_correct_columns(self, sql: str, available_tables: List[str]) -> str:
        """Valida y corrige columnas de forma más conservadora"""
        try:
            if not self.llm:
                return sql
                
            # Obtener esquema detallado de las tablas
            schema_info = self._get_detailed_schema_for_tables(available_tables)
            
            # Buscar errores específicos de columnas inexistentes
            import re
            column_pattern = r'([A-Z_]+)\.([A-Z_]+)'
            column_matches = re.findall(column_pattern, sql.upper())
            
            invalid_columns = []
            for table_name, column_name in column_matches:
                if table_name in self.column_metadata:
                    table_columns = [col['name'].upper() for col in self.column_metadata[table_name]['columns']]
                    if column_name not in table_columns:
                        invalid_columns.append(f"{table_name}.{column_name}")
            
            if invalid_columns:
                print(f"   ⚠️ Columnas inválidas detectadas: {invalid_columns}")
                
                prompt = f"""
Eres un experto en bases de datos médicas. Corrige SOLO las columnas que no existen.

SQL CON COLUMNAS INVÁLIDAS:
```sql
{sql}
```

COLUMNAS INVÁLIDAS DETECTADAS:
{invalid_columns}

ESQUEMA EXACTO DE TABLAS:
{schema_info}

INSTRUCCIONES CRÍTICAS:
1. SOLO corrige las columnas que aparecen en la lista de inválidas
2. NO cambies columnas que ya son correctas
3. Usa los nombres EXACTOS del esquema
4. Mantén la lógica del SQL intacta
5. Si no estás seguro, NO cambies nada

RESPUESTA:
```json
{{
  "columnas_corregidas": ["columna_invalida → columna_correcta"],
  "sql_corregido": "SQL con solo las columnas inválidas corregidas"
}}
```

Responde SOLO con el JSON.
                """
                
                response = _call_openai_native(self.llm, [{"role": "user", "content": prompt}])
                result = self._try_parse_llm_json(response.content)
                
                if result and result.get('sql_corregido'):
                    columnas_corregidas = result.get('columnas_corregidas', [])
                    if columnas_corregidas:
                        print(f"   🔧 Columnas corregidas: {columnas_corregidas}")
                    
                    return result['sql_corregido']
            
            return sql
            
        except Exception as e:
            print(f"   ❌ Error validando columnas: {e}")
            return sql
    
    async def _generate_sql_attempt_2(self, query: str, available_tables: List[str], error_context: str) -> str:
        """Segundo intento: con contexto de error y validación estricta"""
        try:
            # Obtener esquema detallado de las tablas disponibles
            schema_info = self._get_detailed_schema_for_tables(available_tables)
            
            prompt = f"""Error anterior: "{error_context}"

Corrige el SQL para: "{query}"

ESQUEMA EXACTO DE TABLAS DISPONIBLES:
{schema_info}

REGLAS CRÍTICAS PARA CORREGIR:
1. SOLO usa tablas y columnas que aparecen en el esquema exacto
2. NO inventes nombres de columnas
3. NO inventes nombres de tablas
4. Usa los nombres EXACTOS de columnas del esquema
5. Corrige el error específico mencionado
6. Responde SOLO con el SQL, sin explicaciones ni texto adicional
7. Si no puedes generar SQL válido, responde: -- ERROR: No se puede generar SQL válido --

EJEMPLOS DE NOMBRES CORRECTOS:
- PATI_PATIENTS.PATI_NAME (no "patient_name")
- DOCS_DOCUMENTS.DOCS_DOCUMENT_DATE (no "document_date")
- EPIS_DIAGNOSTICS.DIAG_OBSERVATION (no "observation")

SQL CORREGIDO:"""
            
            response = _call_openai_native(self.llm, [{"role": "user", "content": prompt}])
            sql = self._clean_sql_response(response.content)
            
            # Validar que el SQL generado solo use columnas existentes
            validated_sql = await self._validate_and_correct_columns(sql, available_tables)
            
            return validated_sql
            
        except Exception as e:
            print(f"❌ Error en intento 2: {e}")
            return ""
    
    async def _generate_sql_attempt_3(self, query: str, available_tables: List[str]) -> str:
        """Tercer intento: ultra-restrictivo"""
        try:
            prompt = f"""Genera SQL para: "{query}"

SOLO usa: {', '.join(available_tables)}

Si no puedes, responde: -- ERROR: No se puede responder porque no existe ninguna tabla con la información solicitada en el esquema actual. --

Solo SQL o mensaje de error."""
            response = _call_openai_native(self.llm, [{"role": "user", "content": prompt}])
            return self._clean_sql_response(response.content)
        except Exception as e:
            print(f"❌ Error en intento 3: {e}")
            return ""
    
    async def _generate_sql_attempt_4(self, query: str, available_tables: List[str]) -> str:
        """Cuarto intento: validación estricta"""
        try:
            prompt = f"""Genera SQL para: "{query}"

SOLO usa: {', '.join(available_tables)}

Si no es posible: -- ERROR: No se puede responder porque no existe ninguna tabla con la información solicitada en el esquema actual. --

Solo SQL o mensaje de error."""
            response = _call_openai_native(self.llm, [{"role": "user", "content": prompt}])
            return self._clean_sql_response(response.content)
        except Exception as e:
            print(f"❌ Error en intento 4: {e}")
            return ""
    
    def _clean_sql_response(self, content: str) -> str:
        """Limpia la respuesta del LLM para obtener SQL válido de forma estricta"""
        try:
            sql = content.strip()
            
            # Remover markdown
            if sql.startswith('```sql'):
                sql = sql[6:]
            elif sql.startswith('```'):
                sql = sql[3:]
            if sql.endswith('```'):
                sql = sql[:-3]
            
            # Remover texto antes del SELECT
            import re
            select_match = re.search(r'SELECT.*', sql, re.IGNORECASE | re.DOTALL)
            if select_match:
                sql = select_match.group(0)
            
            # Remover texto después del último punto y coma
            if ';' in sql:
                sql = sql[:sql.rfind(';')+1]
            
            # Remover cualquier texto explicativo o no-SQL
            # Buscar solo el SQL válido
            sql_patterns = [
                r'SELECT.*?;',  # SELECT hasta punto y coma
                r'UPDATE.*?;',  # UPDATE hasta punto y coma
                r'INSERT.*?;',  # INSERT hasta punto y coma
                r'DELETE.*?;',  # DELETE hasta punto y coma
            ]
            
            for pattern in sql_patterns:
                match = re.search(pattern, sql, re.IGNORECASE | re.DOTALL)
                if match:
                    sql = match.group(0)
                    break
            
            # Si no se encontró SQL válido, buscar solo SELECT
            if not re.search(r'SELECT.*', sql, re.IGNORECASE):
                select_match = re.search(r'SELECT.*', content, re.IGNORECASE | re.DOTALL)
                if select_match:
                    sql = select_match.group(0)
                    # Añadir punto y coma si no lo tiene
                    if not sql.endswith(';'):
                        sql += ';'
            
            # Limpiar espacios y saltos de línea
            sql = re.sub(r'\s+', ' ', sql).strip()
            
            # Asegurar punto y coma final
            if sql and not sql.endswith(';'):
                sql += ';'
            
            # Remover múltiples puntos y coma
            sql = re.sub(r';+', ';', sql)
            
            # Validar que sea SQL válido
            if not re.search(r'^(SELECT|UPDATE|INSERT|DELETE)', sql, re.IGNORECASE):
                return ""
            
            return sql.strip()
            
        except Exception as e:
            print(f"❌ Error limpiando SQL: {e}")
            return ""
            
    def _is_valid_sql(self, sql: str) -> bool:
        """Valida que el SQL sea válido y completo de forma estricta"""
        if not sql or len(sql) < 20:
            return False
        
        # Verificar que contenga SELECT, UPDATE, INSERT o DELETE
        if not re.search(r'^(SELECT|UPDATE|INSERT|DELETE)', sql, re.IGNORECASE):
            return False
        
        # Verificar que no sea solo un mensaje de error
        if sql.startswith('-- ERROR:') or sql.startswith('No se puede') or 'ERROR' in sql.upper():
            return False
        
        # Verificar que tenga FROM (para SELECT) o INTO/WHERE (para otros)
        if re.search(r'SELECT', sql, re.IGNORECASE) and not re.search(r'FROM', sql, re.IGNORECASE):
            return False
        
        # Verificar que no contenga texto explicativo
        if re.search(r'(explicación|explicar|nota|comentario|verificar|verifica)', sql, re.IGNORECASE):
            return False
        
        # Verificar que termine con punto y coma
        if not sql.strip().endswith(';'):
            return False
        
        return True
    
    def _generate_basic_sql(self, query: str) -> str:
        """Genera SQL básico sin LLM"""
        return "SELECT * FROM PATI_PATIENTS LIMIT 10;"

    async def _llm_explore_schema_intelligently(self, tables: List[str], query: str) -> Dict[str, Any]:
        """Usa LLM para explorar el esquema de manera inteligente y entender qué contiene cada tabla"""
        try:
            if not self.llm:
                return {}
                
            # Obtener información detallada de las tablas
            table_info = {}
            for table in tables:
                if table in self.column_metadata:
                    columns = [col['name'] for col in self.column_metadata[table]['columns'][:10]]
                    table_info[table] = columns
            
            prompt = f"""
Eres un experto en bases de datos médicas. Analiza este esquema para entender qué información contiene cada tabla.

CONSULTA: "{query}"

ESQUEMA DE TABLAS:
{table_info}

INSTRUCCIONES:
1. Analiza los nombres de las columnas para entender qué información contiene cada tabla
2. Busca patrones en los nombres (ej: "OBSERVATION" sugiere observaciones médicas)
3. Identifica qué tabla podría contener la información buscada en la consulta
4. Considera el contexto médico de la consulta

RESPUESTA:
```json
{{
  "analisis_tablas": {{
    "tabla1": "descripción de qué contiene",
    "tabla2": "descripción de qué contiene"
  }},
  "tabla_recomendada": "tabla que mejor se ajusta a la consulta",
  "razon": "explicación de por qué esta tabla es la mejor opción"
}}
```

Responde SOLO con el JSON.
            """
            
            response = _call_openai_native(self.llm, [{"role": "user", "content": prompt}])
            result = self._try_parse_llm_json(response.content)
            
            if result:
                analisis = result.get('analisis_tablas', {})
                tabla_recomendada = result.get('tabla_recomendada', '')
                razon = result.get('razon', '')
                
                if analisis:
                    print(f"   🧠 LLM analizó el esquema:")
                    for tabla, desc in analisis.items():
                        print(f"      - {tabla}: {desc}")
                
                if tabla_recomendada:
                    print(f"   💡 Tabla recomendada: {tabla_recomendada}")
                    print(f"   💭 Razón: {razon}")
                
                return result
            
            return {}
            
        except Exception as e:
            print(f"   ❌ Error explorando esquema: {e}")
            return {}

    async def _llm_validate_table_candidates(self, table_candidates: List[str], query: str) -> List[str]:
        """Usa LLM para validar dinámicamente las tablas candidatas"""
        try:
            if not self.llm:
                # Fallback: solo tablas que existen
                return [table for table in table_candidates if table in self.column_metadata]
                
            # Obtener lista de tablas disponibles
            available_tables = list(self.column_metadata.keys())
            
            prompt = f"""
Eres un experto en bases de datos médicas. Valida estas tablas candidatas para una consulta.

CONSULTA: "{query}"

TABLAS CANDIDATAS:
{table_candidates}

TABLAS DISPONIBLES EN LA BASE DE DATOS:
{len(available_tables)} tablas disponibles (lista completa en contexto interno)

INSTRUCCIONES:
1. Analiza la consulta para entender qué información necesita
2. Identifica qué tablas candidatas son válidas (existen en la base de datos)
3. Para tablas inválidas, sugiere la tabla correcta que contenga la información necesaria
4. Considera el contexto médico de la consulta
5. Busca patrones en los nombres de las tablas para entender su propósito
6. Si la consulta busca información médica específica, identifica qué tabla podría contenerla

RESPUESTA:
```json
{{
  "tablas_validas": ["tablas que existen y son relevantes"],
  "tablas_invalidas": ["tablas que no existen"],
  "sugerencias": ["tabla_invalida → tabla_correcta"],
  "razon": "explicación de las correcciones"
}}
```

Responde SOLO con el JSON.
            """
            
            response = _call_openai_native(self.llm, [{"role": "user", "content": prompt}])
            result = self._try_parse_llm_json(response.content)
            
            if result:
                tablas_validas = result.get('tablas_validas', [])
                tablas_invalidas = result.get('tablas_invalidas', [])
                sugerencias = result.get('sugerencias', [])
                
                if tablas_invalidas:
                    print(f"   ⚠️ LLM detectó tablas inválidas: {tablas_invalidas}")
                    print(f"   💡 Sugerencias: {sugerencias}")
                
                # Filtrar solo tablas que realmente existen
                final_tables = [table for table in tablas_validas if table in self.column_metadata]
                
                if not final_tables:
                    # Fallback: usar tablas básicas
                    print(f"   ⚠️ No se encontraron tablas válidas, usando fallback")
                    final_tables = ['PATI_PATIENTS']
                
                return final_tables
            
            # Fallback si no se puede parsear JSON
            return [table for table in table_candidates if table in self.column_metadata]
            
        except Exception as e:
            print(f"   ❌ Error en validación LLM de tablas candidatas: {e}")
            # Fallback: solo tablas que existen
            return [table for table in table_candidates if table in self.column_metadata]

    async def _llm_validate_tables(self, sql: str, original_query: str) -> str:
        """Usa LLM para validar y corregir tablas en SQL"""
        try:
            if not self.llm:
                return sql
                
            # Obtener lista de tablas disponibles
            available_tables = list(self.column_metadata.keys())
            
            prompt = f"""
Eres un experto en bases de datos médicas. Valida este SQL y corrige cualquier tabla incorrecta.

CONSULTA ORIGINAL: "{original_query}"

SQL A VALIDAR:
```sql
{sql}
```

TABLAS DISPONIBLES EN LA BASE DE DATOS:
{len(available_tables)} tablas disponibles (lista completa en contexto interno)

INSTRUCCIONES:
1. Identifica todas las tablas mencionadas en el SQL
2. Verifica que existan en la lista de tablas disponibles
3. Si hay tablas incorrectas, reemplázalas con las correctas
4. Para constantes vitales: usar EPIS_DIAGNOSTICS
5. Para pacientes: usar PATI_PATIENTS
6. Para medicamentos: usar PATI_USUAL_MEDICATION

RESPUESTA:
```json
{{
  "tablas_detectadas": ["lista de tablas en el SQL"],
  "tablas_incorrectas": ["tablas que no existen"],
  "correcciones": ["tabla_incorrecta → tabla_correcta"],
  "sql_corregido": "SQL con tablas corregidas"
}}
```

Responde SOLO con el JSON.
            """
            
            response = _call_openai_native(self.llm, [{"role": "user", "content": prompt}])
            result = self._try_parse_llm_json(response.content)
            
            if result and result.get('sql_corregido'):
                tablas_incorrectas = result.get('tablas_incorrectas', [])
                if tablas_incorrectas:
                    print(f"   ⚠️ LLM detectó tablas incorrectas: {tablas_incorrectas}")
                    print(f"   🔧 Correcciones aplicadas: {result.get('correcciones', [])}")
                
                return result['sql_corregido']
            
            return sql
            
        except Exception as e:
            print(f"   ❌ Error en validación LLM de tablas: {e}")
            return sql

    async def _llm_correct_invalid_tables(self, sql: str, invalid_tables: List[str]) -> str:
        """Usa LLM para corregir dinámicamente tablas inválidas de forma conservadora"""
        try:
            if not self.llm:
                return sql
                
            # Obtener lista de tablas disponibles
            available_tables = list(self.column_metadata.keys())
            
            prompt = f"""
Eres un experto en bases de datos médicas. Corrige SOLO las tablas que no existen.

SQL CON TABLAS INVÁLIDAS:
```sql
{sql}
```

TABLAS INVÁLIDAS DETECTADAS:
{invalid_tables}

TABLAS DISPONIBLES EN LA BASE DE DATOS:
{available_tables}

INSTRUCCIONES CRÍTICAS:
1. SOLO corrige las tablas que aparecen en la lista de inválidas
2. NO cambies tablas que ya son correctas
3. Usa los nombres EXACTOS de las tablas disponibles
4. Mantén la lógica del SQL intacta
5. Si no estás seguro, NO cambies nada
6. Para constantes vitales: usar EPIS_DIAGNOSTICS
7. Para pacientes: usar PATI_PATIENTS
8. Para medicamentos: usar PATI_USUAL_MEDICATION

RESPUESTA:
```json
{{
  "tablas_corregidas": ["tabla_invalida → tabla_correcta"],
  "razon": "explicación de las correcciones",
  "sql_corregido": "SQL con solo las tablas inválidas corregidas"
}}
```

Responde SOLO con el JSON.
            """
            
            response = _call_openai_native(self.llm, [{"role": "user", "content": prompt}])
            result = self._try_parse_llm_json(response.content)
            
            if result and result.get('sql_corregido'):
                tablas_corregidas = result.get('tablas_corregidas', [])
                if tablas_corregidas:
                    print(f"   🔧 Tablas corregidas: {tablas_corregidas}")
                
                return result['sql_corregido']
            
            return sql
            
        except Exception as e:
            print(f"   ❌ Error en corrección LLM de tablas: {e}")
            return sql

    async def _validate_and_fix_tables(self, sql: str) -> str:
        """Valida y corrige tablas que no existen en el esquema de forma más inteligente"""
        try:
            # Obtener lista de tablas disponibles
            available_tables = list(self.column_metadata.keys())
            
            # Buscar solo nombres de tablas reales en el SQL (no columnas)
            import re
            
            # Patrón más específico para detectar solo tablas, no columnas
            # Buscar FROM tabla, JOIN tabla, UPDATE tabla, etc.
            table_patterns = [
                r'FROM\s+([A-Z_]+)',           # FROM tabla
                r'JOIN\s+([A-Z_]+)',           # JOIN tabla  
                r'UPDATE\s+([A-Z_]+)',         # UPDATE tabla
                r'INSERT\s+INTO\s+([A-Z_]+)',  # INSERT INTO tabla
                r'DELETE\s+FROM\s+([A-Z_]+)',  # DELETE FROM tabla
            ]
            
            tables_in_sql = []
            for pattern in table_patterns:
                matches = re.findall(pattern, sql, re.IGNORECASE)
                for match in matches:
                    table_name = match.upper()
                    if table_name not in tables_in_sql:
                        tables_in_sql.append(table_name)
            
            # Verificar si las tablas encontradas existen
            invalid_tables = [table for table in tables_in_sql if table not in available_tables]
            
            if invalid_tables:
                print(f"   ⚠️ Tablas inválidas detectadas: {invalid_tables}")
                print(f"   🔧 Tablas disponibles: {len(available_tables)} tablas en total")
                
                # Corregir tablas inválidas usando LLM si está disponible
                corrected_sql = sql
                if self.llm:
                    print(f"   🔧 Usando LLM para corregir tablas inválidas...")
                    corrected_sql = await self._llm_correct_invalid_tables(sql, invalid_tables)
                    if corrected_sql != sql:
                        print(f"   ✅ LLM corrigió tablas inválidas")
                        return corrected_sql
                else:
                    # Fallback básico: usar tabla por defecto
                    for invalid_table in invalid_tables:
                        print(f"   ⚠️ Tabla inválida: {invalid_table} (sin LLM para corrección)")
                        # No hacer corrección automática sin LLM
                
                print(f"   ✅ SQL corregido: {corrected_sql}")
                return corrected_sql
            
            return sql
            
        except Exception as e:
            print(f"   ❌ Error validando tablas: {e}")
            return sql

    def _build_sql_generation_context(self, mapeo_conceptos: List[Dict[str, Any]], relaciones_criticas: List[str], schema_info: str) -> str:
        """Construye contexto específico para generación SQL basado en el mapeo de conceptos"""
        try:
            context_parts = []
            
            # Agrupar conceptos por tipo
            conceptos_por_tipo = {}
            for concepto in mapeo_conceptos:
                tipo = concepto.get('tipo_concepto', 'OTROS')
                if tipo not in conceptos_por_tipo:
                    conceptos_por_tipo[tipo] = []
                conceptos_por_tipo[tipo].append(concepto)
            
            # Construir contexto específico
            for tipo, conceptos in conceptos_por_tipo.items():
                context_parts.append(f"\n📋 {tipo}:")
                for concepto in conceptos:
                    termino = concepto.get('concepto', '')
                    tablas = concepto.get('tablas_relevantes', [])
                    campos = concepto.get('campos_busqueda', [])
                    
                    context_parts.append(f"  - '{termino}' → Tablas: {', '.join(tablas)}")
                    if campos:
                        context_parts.append(f"    Campos de búsqueda: {', '.join(campos)}")
            
            # Agregar relaciones críticas
            if relaciones_criticas:
                context_parts.append(f"\n🔗 RELACIONES CRÍTICAS:")
                for relacion in relaciones_criticas:
                    context_parts.append(f"  - {relacion}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            print(f"❌ Error construyendo contexto SQL: {e}")
            return "Contexto no disponible"

    async def _fix_common_column_errors(self, sql: str, stream_callback=None) -> str:
        """Corrección inteligente de errores comunes usando múltiples funciones especializadas"""
        try:
            print(f"🔧 Aplicando correcciones automáticas especializadas...")
            
            if not self.llm:
                print("⚠️ LLM no disponible, saltando corrección automática")
                return sql
            
            corrected_sql = sql
            
            # 1. Corrección de mapeo conceptual (condiciones vs medicamentos)
            corrected_sql = await self._fix_concept_mapping_errors(corrected_sql)
            
            # 2. Corrección de JOINs faltantes
            corrected_sql = await self._fix_missing_joins(corrected_sql)
            
            # 3. Corrección de nombres de campos
            corrected_sql = await self._fix_field_names(corrected_sql)
            
            # 4. Validación final con LLM
            corrected_sql = await self._validate_sql_logic(corrected_sql)
            
            if corrected_sql != sql:
                print(f"   ✅ SQL corregido con múltiples funciones especializadas")
            else:
                print(f"   ✅ No se requirieron correcciones")
                
            return corrected_sql
            
        except Exception as e:
            print(f"   ❌ Error en corrección automática: {e}")
            return sql

    async def _fix_concept_mapping_errors(self, sql: str) -> str:
        """Corrige errores de mapeo conceptual usando LLM"""
        try:
            prompt = f"""
Eres un experto en mapeo conceptual médico. Analiza este SQL y corrige errores de mapeo conceptual.

SQL:
```sql
{sql}
```

ERRORES A CORREGIR:
1. Condiciones médicas (diabetes, hipertensión, etc.) buscadas en tablas de medicamentos
2. Medicamentos (metformina, aspirina, etc.) buscados en tablas de condiciones

REGLAS:
- Condiciones médicas → ACCI_PATIENT_CONDITIONS
- Medicamentos → MEDI_ACTIVE_INGREDIENTS

RESPUESTA:
```json
{{
  "needs_correction": true/false,
  "corrected_sql": "SQL corregido o original"
}}
```
            """
            
            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result and result.get('needs_correction'):
                print(f"   🔄 Corrigiendo mapeo conceptual...")
                return result.get('corrected_sql', sql)
            
            return sql
            
        except Exception as e:
            print(f"   ❌ Error en corrección de mapeo conceptual: {e}")
            return sql

    async def _fix_missing_joins(self, sql: str) -> str:
        """Corrige JOINs faltantes usando LLM"""
        try:
            prompt = f"""
Eres un experto en SQL médico. Analiza este SQL y añade JOINs faltantes.

SQL:
```sql
{sql}
```

TABLAS PRINCIPALES:
- PATI_PATIENTS (p)
- ACCI_PATIENT_CONDITIONS (pc)
- PATI_USUAL_MEDICATION (um)
- MEDI_ACTIVE_INGREDIENTS (ai)

REGLAS DE JOIN:
- pc.PATI_ID = p.PATI_ID
- um.PATI_ID = p.PATI_ID
- ai.ACIN_ID = um.ACIN_ID

RESPUESTA:
```json
{{
  "needs_joins": true/false,
  "corrected_sql": "SQL con JOINs añadidos"
}}
```
            """
            
            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result and result.get('needs_joins'):
                print(f"   🔗 Añadiendo JOINs faltantes...")
                return result.get('corrected_sql', sql)
            
            return sql
            
        except Exception as e:
            print(f"   ❌ Error en corrección de JOINs: {e}")
            return sql

    async def _fix_field_names(self, sql: str) -> str:
        """Corrige nombres de campos usando LLM"""
        try:
            prompt = f"""
Eres un experto en esquemas de BD médicas. Corrige nombres de campos incorrectos.

SQL:
```sql
{sql}
```

CAMPOS CORRECTOS:
- PATI_PATIENTS: PATI_ID, PATI_NAME, PATI_ACTIVE
- ACCI_PATIENT_CONDITIONS: PATI_ID, CONDITION_DESCRIPTION
- MEDI_ACTIVE_INGREDIENTS: ACIN_ID, ACIN_DESCRIPTION_ES

RESPUESTA:
```json
{{
  "needs_field_fix": true/false,
  "corrected_sql": "SQL con campos corregidos"
}}
```
            """
            
            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result and result.get('needs_field_fix'):
                print(f"   📝 Corrigiendo nombres de campos...")
                return result.get('corrected_sql', sql)
            
            return sql
            
        except Exception as e:
            print(f"   ❌ Error en corrección de campos: {e}")
            return sql

    async def _validate_sql_logic(self, sql: str) -> str:
        """Validación final de la lógica SQL usando LLM"""
        try:
            prompt = f"""
Eres un experto en SQL médico. Valida la lógica final de esta consulta.

SQL:
```sql
{sql}
```

VERIFICAR:
1. Lógica SQL correcta
2. Sintaxis válida
3. Relaciones apropiadas
4. Filtros coherentes

RESPUESTA:
```json
{{
  "is_valid": true/false,
  "final_sql": "SQL final validado"
}}
```
            """
            
            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result and result.get('is_valid'):
                return result.get('final_sql', sql)
            
            return sql
            
        except Exception as e:
            print(f"   ❌ Error en validación final: {e}")
            return sql

    async def _llm_clean_and_fix_sql(self, sql: str, stream_callback=None) -> str:
        """Limpieza básica de SQL"""
        return sql
    async def _fix_sql_compatibility(self, sql: str, stream_callback=None) -> str:
        """Corrección de compatibilidad ROBUSTA para SQLite que no puede fallar"""
        try:
            import re
            
            # PASO 1: Limpiar el SQL de caracteres problemáticos
            sql_clean = sql.strip()
            
            # PASO 2: Eliminar múltiples puntos y coma y espacios extra
            sql_clean = re.sub(r';+\s*$', '', sql_clean)  # Remover puntos y coma al final
            sql_clean = re.sub(r'\s+', ' ', sql_clean)     # Normalizar espacios
            
            # PASO 3: Manejo robusto de TOP → LIMIT
            if 'TOP' in sql_clean.upper():
                # Extraer el número después de TOP
                top_match = re.search(r'SELECT\s+TOP\s+(\d+)', sql_clean, re.IGNORECASE)
                if top_match:
                    limit_num = top_match.group(1)
                    # Reemplazar SELECT TOP N con SELECT
                    sql_clean = re.sub(r'SELECT\s+TOP\s+\d+', 'SELECT', sql_clean, flags=re.IGNORECASE)
                    # Agregar LIMIT al final si no existe
                    if 'LIMIT' not in sql_clean.upper():
                        sql_clean += f' LIMIT {limit_num}'
                    print(f"   🔧 TOP convertido a LIMIT correctamente")
            
            # PASO 4: Limpiar múltiples LIMIT duplicados
            limit_matches = re.findall(r'LIMIT\s+\d+', sql_clean, re.IGNORECASE)
            if len(limit_matches) > 1:
                # Remover todos los LIMIT
                sql_clean = re.sub(r'LIMIT\s+\d+', '', sql_clean, flags=re.IGNORECASE)
                # Agregar solo uno al final
                sql_clean += ' LIMIT 1'
                print(f"   🔧 Múltiples LIMIT corregidos")
            
            # PASO 5: Corregir LIMIT mal posicionado (no después de SELECT)
            if re.search(r'SELECT\s+LIMIT', sql_clean, re.IGNORECASE):
                # Remover LIMIT después de SELECT
                sql_clean = re.sub(r'SELECT\s+LIMIT\s+\d+', 'SELECT', sql_clean, flags=re.IGNORECASE)
                # Agregar al final si no existe
                if 'LIMIT' not in sql_clean.upper():
                    sql_clean += ' LIMIT 1'
                print(f"   🔧 LIMIT reposicionado correctamente")
            
            # PASO 6: Asegurar que LIMIT esté después de ORDER BY si existe
            if 'ORDER BY' in sql_clean.upper() and 'LIMIT' in sql_clean.upper():
                # Verificar el orden correcto
                order_pos = sql_clean.upper().find('ORDER BY')
                limit_pos = sql_clean.upper().find('LIMIT')
                
                if limit_pos < order_pos:
                    # LIMIT está antes de ORDER BY, corregir
                    limit_match = re.search(r'LIMIT\s+(\d+)', sql_clean, re.IGNORECASE)
                    if limit_match:
                        limit_value = limit_match.group(1)
                        # Remover LIMIT mal posicionado
                        sql_clean = re.sub(r'LIMIT\s+\d+', '', sql_clean, flags=re.IGNORECASE)
                        # Agregar al final
                        sql_clean += f' LIMIT {limit_value}'
                        print(f"   🔧 LIMIT reordenado después de ORDER BY")
            
            # PASO 7: Otras correcciones comunes
            sql_clean = sql_clean.replace('ISNULL(', 'IFNULL(')
            sql_clean = sql_clean.replace('GETDATE()', 'datetime("now")')
            
            # PASO 8: Limpiar espacios finales y agregar punto y coma único
            sql_clean = sql_clean.strip()
            if not sql_clean.endswith(';'):
                sql_clean += ';'
            
            # PASO 9: Validación final básica
            if not sql_clean.upper().startswith('SELECT'):
                print(f"   ⚠️ SQL no comienza con SELECT, usando fallback")
                return "SELECT * FROM PATI_PATIENTS ORDER BY PATI_ID DESC LIMIT 1;"
            
            print(f"   ✅ SQL corregido y validado: {sql_clean}")
            return sql_clean
            
        except Exception as e:
            print(f"❌ Error en corrección robusta: {e}")
            # FALLBACK ABSOLUTO que siempre funciona
            return "SELECT * FROM PATI_PATIENTS ORDER BY PATI_ID DESC LIMIT 1;"

    async def _validate_sql_syntax(self, sql: str) -> Optional[str]:
        """Validación sintáctica usando SQLite para detectar errores"""
        try:
            import sqlite3
            # Crear conexión temporal para validar sintaxis
            temp_conn = sqlite3.connect(":memory:")
            cursor = temp_conn.cursor()
            
            # Intentar explicar la consulta (esto validará la sintaxis sin ejecutar)
            cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
            temp_conn.close()
            
            return None  # Sin errores
            
        except sqlite3.Error as e:
            return f"Error de sintaxis SQL: {str(e)}"
        except Exception as e:
            return f"Error validando sintaxis: {str(e)}"

    async def _regenerate_sql_with_error_context(self, query: str, sql: str, error: str, params: List[str], stream_callback=None) -> str:
        """Regeneración con contexto de error usando LLM"""
        try:
            if not self.llm:
                return sql
                
            # Obtener esquema real para corrección
            schema_info = self._get_schema_context()
            
            prompt = f"""
Eres un experto en SQL médico. Corrige este SQL que tiene errores de sintaxis.

CONSULTA ORIGINAL: "{query}"

SQL CON ERROR:
```sql
{sql}
```

ERROR DETECTADO:
{error}

ESQUEMA REAL:
{schema_info}

INSTRUCCIONES:
1. Identifica el error específico en el SQL
2. Corrige la sintaxis manteniendo la lógica original
3. Verifica que los nombres de tablas y columnas sean correctos
4. Usa el esquema real proporcionado

RESPUESTA:
```json
{{
  "error_analysis": "análisis del error",
  "corrected_sql": "SQL corregido"
}}
```
            """
            
            response = _call_openai_native(self.llm, prompt, task_description="Corrigiendo sintaxis SQL")
            result = self._try_parse_llm_json(response.content)
            
            if result and result.get('corrected_sql'):
                print(f"   🔧 SQL corregido por error de sintaxis")
                return result['corrected_sql']
            
            return sql
            
        except Exception as e:
            print(f"   ❌ Error en regeneración con contexto: {e}")
            return sql

    def _validate_columns_exist_in_schema(self, sql: str, tables_info: Dict[str, List[str]]) -> Optional[str]:
        """Validación de columnas contra esquema real"""
        try:
            import re
            
            # Extraer nombres de columnas del SQL
            # Buscar patrones como tabla.columna o columna sola
            column_patterns = [
                r'(\w+)\.(\w+)',  # tabla.columna
                r'SELECT\s+(.+?)\s+FROM',  # columnas en SELECT
                r'WHERE\s+(.+?)(?:\s+ORDER|\s+GROUP|\s+HAVING|$)',  # columnas en WHERE
                r'ORDER\s+BY\s+(.+?)(?:\s+LIMIT|$)',  # columnas en ORDER BY
            ]
            
            errors = []
            
            for pattern in column_patterns:
                matches = re.findall(pattern, sql, re.IGNORECASE | re.DOTALL)
                
                for match in matches:
                    if isinstance(match, tuple) and len(match) == 2:
                        # Patrón tabla.columna
                        table, column = match
                        if table in tables_info:
                            if column not in tables_info[table]:
                                errors.append(f"Columna '{column}' no existe en tabla '{table}'. Columnas disponibles: {', '.join(tables_info[table][:5])}")
                    else:
                        # Patrones de columnas en claúsulas
                        columns_text = match if isinstance(match, str) else str(match)
                        # Extraer nombres de columnas individuales
                        individual_columns = re.findall(r'\b(\w+)\.\w+|\b(\w+)\s*(?:,|\s+|$)', columns_text)
                        
                        for col_match in individual_columns:
                            table_col = col_match[0] if col_match[0] else col_match[1]
                            if table_col and table_col.upper() not in ['SELECT', 'FROM', 'WHERE', 'ORDER', 'BY', 'GROUP', 'HAVING', 'LIMIT', 'DISTINCT', 'AS']:
                                # Verificar si es una tabla conocida
                                if table_col in tables_info:
                                    continue
                                
                                # Buscar la columna en todas las tablas
                                found = False
                                for table_name, columns in tables_info.items():
                                    if table_col in columns:
                                        found = True
                                        break
                                
                                if not found and len(table_col) > 2:  # Evitar falsos positivos con palabras cortas
                                    errors.append(f"Columna '{table_col}' no encontrada en ninguna tabla del esquema")
            
            return "; ".join(errors) if errors else None
            
        except Exception as e:
            return f"Error validando columnas: {str(e)}"

    async def _fix_column_references(self, sql: str, error: str, stream_callback=None) -> str:
        """Corrección de referencias de columnas usando LLM"""
        try:
            if not self.llm:
                return sql
                
            # Obtener esquema completo para corrección
            schema_info = self._get_schema_context()
            
            prompt = f"""
Eres un experto en bases de datos médicas. Corrige las referencias de columnas incorrectas en este SQL.

SQL CON ERRORES:
```sql
{sql}
```

ERRORES DETECTADOS:
{error}

ESQUEMA REAL COMPLETO:
{schema_info}

INSTRUCCIONES:
1. Identifica las columnas incorrectas mencionadas en los errores
2. Busca las columnas correctas en el esquema real
3. Reemplaza las referencias incorrectas con las correctas
4. Mantén la lógica y estructura original del SQL
5. Asegúrate de usar los nombres exactos del esquema

RESPUESTA:
```json
{{
  "corrections_made": ["corrección1", "corrección2"],
  "corrected_sql": "SQL con columnas corregidas"
}}
```
            """
            
            response = _call_openai_native(self.llm, prompt, task_description="Corrigiendo referencias de columnas")
            result = self._try_parse_llm_json(response.content)
            
            if result and result.get('corrected_sql'):
                corrections = result.get('corrections_made', [])
                if corrections:
                    print(f"   🔧 Correcciones aplicadas: {', '.join(corrections)}")
                return result['corrected_sql']
            
            return sql
            
        except Exception as e:
            print(f"   ❌ Error corrigiendo referencias: {e}")
            return sql

    async def _llm_final_validation(self, sql: str, stream_callback=None) -> str:
        """Validación final completa usando LLM y verificaciones técnicas"""
        try:
            if not self.llm:
                return sql
                
            # 1. Validación sintáctica
            syntax_error = await self._validate_sql_syntax(sql)
            if syntax_error:
                print(f"   ⚠️ Error de sintaxis detectado: {syntax_error}")
                sql = await self._regenerate_sql_with_error_context("", sql, syntax_error, [])
            
            # 2. Validación de columnas
            available_tables = list(self.column_metadata.keys())
            tables_info = {t: [c['name'] for c in self.column_metadata[t]['columns']] for t in available_tables if t in self.column_metadata}
            
            column_error = self._validate_columns_exist_in_schema(sql, tables_info)
            if column_error:
                print(f"   ⚠️ Errores de columnas detectados: {column_error}")
                sql = await self._fix_column_references(sql, column_error)
            
            # 3. Validación lógica final con LLM
            schema_info = self._get_schema_context()
            
            prompt = f"""
Eres un experto en SQL médico. Realiza una validación final completa de este SQL.

SQL A VALIDAR:
```sql
{sql}
```

ESQUEMA REAL:
{schema_info}

VALIDACIONES CRÍTICAS:
1. ¿La sintaxis SQL es correcta?
2. ¿Todas las tablas y columnas existen en el esquema?
3. ¿Los JOINs están correctamente correlacionados?
4. ¿La lógica responde a la consulta médica?
5. ¿Se están usando los campos correctos para las búsquedas?

RESPUESTA:
```json
{{
  "is_valid": true/false,
  "validation_errors": ["error1", "error2"],
  "final_sql": "SQL final validado y corregido",
  "confidence": "alta/media/baja"
}}
```
            """
            
            response = _call_openai_native(self.llm, prompt, task_description="Validación final SQL")
            result = self._try_parse_llm_json(response.content)
            
            if result:
                is_valid = result.get('is_valid', False)
                errors = result.get('validation_errors', [])
                final_sql = result.get('final_sql', sql)
                confidence = result.get('confidence', 'media')
                
                if not is_valid and errors:
                    print(f"   ⚠️ Errores de validación final: {', '.join(errors)}")
                    if final_sql != sql:
                        print(f"   🔧 SQL corregido en validación final")
                        return final_sql
                else:
                    print(f"   ✅ Validación final exitosa (confianza: {confidence})")
                
                return final_sql
            
            return sql
            
        except Exception as e:
            print(f"   ❌ Error en validación final: {e}")
            return sql

    def _basic_sql_integrity_check(self, sql: str) -> bool:
        """Verificación de integridad robusta del SQL"""
        try:
            if not sql or not sql.strip():
                return False
            
            sql_upper = sql.upper()
            
            # 1. Verificar que tenga estructura SQL básica
            if 'SELECT' not in sql_upper:
                return False
            
            # 2. Verificar que tenga FROM (consultas SELECT deben tener FROM)
            if 'FROM' not in sql_upper:
                return False
            
            # 3. Verificar balance de paréntesis
            if sql.count('(') != sql.count(')'):
                return False
            
            # 4. Verificar que no tenga palabras clave SQL mal formadas
            malformed_patterns = [
                'SELECT SELECT',
                'FROM FROM',
                'WHERE WHERE',
                'JOIN JOIN',
                'AND AND',
                'OR OR'
            ]
            
            for pattern in malformed_patterns:
                if pattern in sql_upper:
                    return False
            
            # 5. Verificar que tenga al menos una tabla válida
            import re
            from_match = re.search(r'FROM\s+(\w+)', sql_upper)
            if not from_match:
                return False
            
            # 6. Verificar que las comillas estén balanceadas
            single_quotes = sql.count("'")
            double_quotes = sql.count('"')
            
            if single_quotes % 2 != 0 or double_quotes % 2 != 0:
                return False
            
            # 7. Verificar que no termine de forma abrupta
            sql_trimmed = sql.strip()
            if sql_trimmed.endswith(('AND', 'OR', 'WHERE', 'JOIN', 'ON')):
                return False
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error en verificación de integridad: {e}")
            return False

    async def _verify_medical_concept_mapping(self, query: str, sql: str, stream_callback=None) -> str:
        """Verifica y corrige errores de mapeo conceptual médico usando LLM"""
        try:
            if not self.llm:
                return sql
                
            # Obtener esquema para corrección
            schema_info = self._get_schema_context()
                
            prompt = f"""
Eres un experto en medicina y bases de datos clínicas. Analiza este SQL para detectar y CORREGIR errores conceptuales médicos.

CONSULTA ORIGINAL: "{query}"

SQL GENERADO:
```sql
{sql}
```

ESQUEMA DISPONIBLE:
{schema_info}

HERRAMIENTAS DE VERIFICACIÓN Y CORRECCIÓN:

1. **Clasificación de términos médicos**:
   - Identifica qué términos en la consulta son CONDICIONES MÉDICAS vs MEDICAMENTOS
   - Las condiciones médicas deben buscarse en tablas de condiciones/diagnósticos
   - Los medicamentos deben buscarse en tablas de medicamentos/ingredientes activos

2. **Corrección de mapeo tabla-concepto**:
   - Si una condición médica se busca en tabla de medicamentos, CORRÍGELO
   - Si un medicamento se busca en tabla de condiciones, CORRÍGELO
   - Asegúrate de que cada concepto se busque en la tabla correcta

3. **Corrección de lógica SQL**:
   - Si falta buscar un concepto médico, AÑÁDELO al WHERE
   - Si falta un JOIN necesario, AÑÁDELO
   - Asegúrate de que la consulta responda completamente a la pregunta

REGLAS CRÍTICAS IMPORTANTES:
- ACCI_PATIENT_CONDITIONS es SOLO un catálogo, NO tiene PATI_ID
- Para diagnósticos de pacientes usa EPIS_DIAGNOSTICS (tiene PATI_ID y DIAG_OBSERVATION)
- PATI_PATIENT_ALLERGIES es SOLO para alergias, NO para condiciones médicas generales
- EPIS_DIAGNOSTICS es para diagnósticos de pacientes (diabetes, hipertensión, etc.)

INSTRUCCIONES CRÍTICAS:
- CORRIGE el SQL para que busque TODOS los conceptos mencionados en la consulta
- Usa las tablas correctas para cada tipo de concepto médico
- Asegúrate de que el WHERE incluya condiciones para todos los conceptos
- **IMPORTANTE**: NO añadas filtros adicionales como PATI_ACTIVE, DIAG_RELEVANT o DIAG_DELETED
- **IMPORTANTE**: Mantén la consulta lo más inclusiva posible para no excluir registros válidos
- **IMPORTANTE**: Solo añade filtros que estén explícitamente solicitados en la consulta original

RESPUESTA:
```json
{{
  "tiene_errores_conceptuales": true/false,
  "errores_detectados": ["descripción específica del problema encontrado"],
  "sql_corregido": "SQL COMPLETAMENTE CORREGIDO que busca todos los conceptos en las tablas correctas"
}}
```

Responde SOLO con el JSON.
            """
            
            response = _call_openai_native(self.llm, prompt, task_description="Verificación conceptual médica")
            result = self._try_parse_llm_json(response.content)
            
            # CORREGIDO: Verificar que result sea un diccionario antes de usar .get()
            if result and isinstance(result, dict) and result.get('tiene_errores_conceptuales'):
                errores = result.get('errores_detectados', [])
                sql_corregido = result.get('sql_corregido', sql)
                
                print(f"   ⚠️ Errores conceptuales detectados: {len(errores)}")
                for error in errores[:2]:  # Mostrar máximo 2 errores
                    print(f"      - {error}")
                
                if sql_corregido and sql_corregido != sql:
                    print(f"   🔧 SQL corregido conceptualmente")
                    return sql_corregido
                else:
                    print(f"   ⚠️ Errores detectados pero no se pudo corregir - intentando regeneración")
                    # Si no se pudo corregir, intentar regenerar el SQL
                    return await self._regenerate_sql_with_conceptual_error(query, sql, errores)
            else:
                print(f"   ✅ Mapeo conceptual correcto")
            
            return sql
            
        except Exception as e:
            print(f"   ❌ Error en verificación conceptual: {e}")
            return sql

    async def _regenerate_sql_with_conceptual_error(self, query: str, original_sql: str, errors: List[str]) -> str:
        """Regenera SQL cuando hay errores conceptuales que no se pudieron corregir"""
        try:
            if not self.llm:
                return original_sql
                
            schema_info = self._get_schema_context()
            
            prompt = f"""
Eres un experto en SQL médico. El SQL anterior tiene errores conceptuales. Regenera el SQL correctamente.

CONSULTA ORIGINAL: "{query}"

SQL CON ERRORES:
```sql
{original_sql}
```

ERRORES DETECTADOS:
{chr(10).join(errors)}

ESQUEMA DISPONIBLE:
{schema_info}

REGLAS CRÍTICAS IMPORTANTES:
- ACCI_PATIENT_CONDITIONS es SOLO un catálogo, NO tiene PATI_ID
- Para diagnósticos de pacientes usa EPIS_DIAGNOSTICS (tiene PATI_ID y DIAG_OBSERVATION)
- PATI_PATIENT_ALLERGIES es SOLO para alergias, NO para condiciones médicas generales
- EPIS_DIAGNOSTICS es para diagnósticos de pacientes (diabetes, hipertensión, etc.)

INSTRUCCIONES:
1. Analiza la consulta original para identificar TODOS los conceptos médicos
2. Genera SQL que busque cada concepto en la tabla correcta
3. Asegúrate de que el WHERE incluya condiciones para todos los conceptos
4. Usa JOINs para conectar las tablas necesarias
5. **NO añadas filtros adicionales** como PATI_ACTIVE = 1, DIAG_RELEVANT = 1, DIAG_DELETED = 0
6. Mantén la consulta **inclusiva** para obtener todos los registros relevantes
7. Responde SOLO con el SQL corregido

Responde SOLO con el SQL.
            """
            
            response = _call_openai_native(self.llm, prompt, task_description="Regenerando SQL con corrección conceptual")
            corrected_sql = response.content.strip()
            
            # Limpiar el SQL de cualquier formato markdown
            if corrected_sql.startswith('```sql'):
                corrected_sql = corrected_sql[6:]
            if corrected_sql.startswith('```'):
                corrected_sql = corrected_sql[3:]
            if corrected_sql.endswith('```'):
                corrected_sql = corrected_sql[:-3]
            corrected_sql = corrected_sql.strip()
            
            print(f"   🔄 SQL regenerado por errores conceptuales")
            return corrected_sql
            
        except Exception as e:
            print(f"   ❌ Error regenerando SQL: {e}")
            return original_sql

    async def _find_correct_table_with_llm(self, sql: str, missing_table: str, original_query: str) -> str:
        """Encuentra la tabla correcta cuando una tabla no existe"""
        try:
            if not self.llm:
                return sql
                
            # Obtener lista de tablas reales disponibles
            available_tables = list(self.column_metadata.keys())
            
            # Filtrar tablas que podrían ser relevantes
            relevant_tables = []
            search_terms = missing_table.lower().split('_')
            
            for table in available_tables:
                table_lower = table.lower()
                # Buscar tablas que contengan términos similares
                if any(term in table_lower for term in search_terms if len(term) > 2):
                    relevant_tables.append(table)
            
            # Si no hay tablas relevantes, usar todas las que parezcan médicas
            if not relevant_tables:
                medical_keywords = ['patient', 'condition', 'diagnosis', 'medication', 'treatment', 'observation']
                relevant_tables = [t for t in available_tables if any(k in t.lower() for k in medical_keywords)]
            
            # Limitar a las 10 tablas más relevantes
            relevant_tables = relevant_tables[:10]
            
            # Obtener esquema de las tablas relevantes
            schema_info = ""
            for table in relevant_tables:
                if table in self.column_metadata:
                    columns = [col['name'] for col in self.column_metadata[table]['columns'][:5]]
                    schema_info += f"\n{table}: {', '.join(columns)}..."
            
            prompt = f"""
Eres un experto en bases de datos médicas. El SQL falló porque la tabla '{missing_table}' no existe.

CONSULTA ORIGINAL: "{original_query}"

SQL QUE FALLÓ:
```sql
{sql}
```

ERROR: La tabla '{missing_table}' no existe en la base de datos.

TABLAS DISPONIBLES RELEVANTES:
{schema_info}

INSTRUCCIONES:
1. Identifica qué información intentaba buscar en la tabla '{missing_table}'
2. Encuentra la tabla REAL que contiene esa información
3. Reemplaza '{missing_table}' con la tabla correcta
4. Ajusta los nombres de columnas si es necesario

RESPUESTA:
```json
{{
  "tabla_correcta": "nombre de la tabla real que debe usarse",
  "razon": "breve explicación de por qué esta tabla es la correcta",
  "sql_corregido": "SQL completo corregido con la tabla correcta"
}}
```

Responde SOLO con el JSON.
            """
            
            response = _call_openai_native(self.llm, prompt, task_description="Búsqueda de tabla correcta")
            result = self._try_parse_llm_json(response.content)
            
            if result and result.get('sql_corregido'):
                tabla_correcta = result.get('tabla_correcta', 'desconocida')
                razon = result.get('razon', '')
                
                print(f"   ✅ Tabla correcta encontrada: {tabla_correcta}")
                if razon:
                    print(f"   💡 Razón: {razon}")
                
                return result['sql_corregido']
            
            return sql
            
        except Exception as e:
            print(f"   ❌ Error buscando tabla correcta: {e}")
            return sql

    async def _generate_emergency_sql(self, query: str, medical_analysis: Dict[str, Any], stream_callback=None) -> str:
        """SQL de emergencia"""
        return "SELECT * FROM PATI_PATIENTS LIMIT 10;"

    async def _execute_sql_with_learning(self, query: str, sql: str, start_time: float, params: Optional[List[str]] = None, stream_callback=None) -> Dict[str, Any]:
        """Ejecución con aprendizaje y debugging detallado + corrección automática en caso de error"""
        try:
            print(f"   📊 Ejecutando consulta en la base de datos...")
            print(f"   📝 Tu consulta: '{query}'")
            print(f"   💾 Consulta SQL: {sql[:80]}...")
            if params:
                print(f"   🔧 Parámetros: {params}")
            
            # Corregir compatibilidad SQL antes de ejecutar
            sql_corrected = await self._fix_sql_compatibility(sql, stream_callback)
            
            # NUEVA VALIDACIÓN: Verificar tablas antes de ejecutar
            print(f"   🔍 Validando tablas en SQL antes de ejecutar...")
            validated_sql = await self._validate_and_fix_tables(sql_corrected)
            
            if validated_sql != sql_corrected:
                print(f"   🔧 SQL corregido automáticamente antes de ejecutar")
                sql_corrected = validated_sql
            
            # Ejecutar con parámetros SOLO si el SQL tiene placeholders '?'
            if params and '?' in sql_corrected:
                result = self.sql_executor.execute_query(sql_corrected, params)
            else:
                result = self.sql_executor.execute_query(sql_corrected)
            
            if result['success']:
                records_found = len(result.get('data', []))
                print(f"   ✅ Consulta ejecutada exitosamente")
                print(f"   📊 Registros encontrados: {records_found}")
            else:
                print(f"   ❌ Error ejecutando la consulta")
                print(f"   🔍 Detalle: {result.get('error', 'Error desconocido')}")
            
            # Corrección automática SOLO para errores de tabla inexistente
            if not result['success']:
                error_msg = result.get('error', '').lower()
                
                # Si es error de tabla inexistente o columna inexistente, intentar corrección inteligente
                if 'no such table' in error_msg or 'no such column' in error_msg:
                    print(f"\n   ⚠️ Detectado error en la estructura de la base de datos")
                    print(f"   🔧 Intentando corrección automática...")
                    
                    # Extraer el nombre de la tabla/columna que no existe
                    import re
                    table_match = re.search(r'no such table:\s*(\w+)', result.get('error', ''), re.IGNORECASE)
                    column_match = re.search(r'no such column:\s*(\w+)', result.get('error', ''), re.IGNORECASE)
                    
                    missing_table = table_match.group(1) if table_match else None
                    missing_column = column_match.group(1) if column_match else None
                    
                    if missing_table:
                        print(f"   📋 Tabla no encontrada: {missing_table}")
                        
                        # Usar LLM para encontrar la tabla correcta
                        corrected_sql = await self._find_correct_table_with_llm(sql, missing_table, query)
                        
                        if corrected_sql and corrected_sql != sql:
                            print(f"   🔄 Reintentando con tabla corregida...")
                            
                            # Reintentar con SQL corregido
                            retry_result = self.sql_executor.execute_query(corrected_sql)
                            
                            if retry_result['success']:
                                records_found = len(retry_result.get('data', []))
                                print(f"   ✅ ¡Corrección exitosa! Encontrados {records_found} registros")
                                # Formatear los datos para mejor visualización
                                formatted_data = await self._format_sql_results_for_display(retry_result.get('data', []), query)
                                
                                return {
                                    'success': True,
                                    'data': retry_result.get('data', []),
                                    'formatted_data': formatted_data,
                                    'error': '',
                                    'sql_used': corrected_sql,
                                    'execution_time': time.time() - start_time,
                                    'auto_corrected': True,
                                    'correction_type': 'missing_table'
                                }
                            else:
                                print(f"   ❌ La corrección no funcionó: {retry_result.get('error', '')}")
                                # Reintento con prompt alternativo
                                print(f"   🔄 Reintentando con prompt alternativo...")
                                retry_sql = await self._generate_sql_attempt_2(query, list(self.column_metadata.keys()), f"Tabla {missing_table} no existe")
                                if retry_sql and len(retry_sql) > 20:
                                    retry_result2 = self.sql_executor.execute_query(retry_sql)
                                    if retry_result2['success']:
                                        records_found = len(retry_result2.get('data', []))
                                        print(f"   ✅ ¡Reintento exitoso! Encontrados {records_found} registros")
                                        formatted_data = await self._format_sql_results_for_display(retry_result2.get('data', []), query)
                                        return {
                                            'success': True,
                                            'data': retry_result2.get('data', []),
                                            'formatted_data': formatted_data,
                                            'error': '',
                                            'sql_used': retry_sql,
                                            'execution_time': time.time() - start_time,
                                            'auto_corrected': True,
                                            'correction_type': 'retry_with_alternative_prompt'
                                        }
                    
                    elif missing_column:
                        print(f"   📋 Columna no encontrada: {missing_column}")
                        
                        # Usar LLM para corregir la columna incorrecta
                        corrected_sql = await self._fix_column_references(sql, result.get('error', ''), stream_callback)
                        
                        if corrected_sql and corrected_sql != sql:
                            print(f"   🔄 Reintentando con columna corregida...")
                            
                            # Reintentar con SQL corregido
                            retry_result = self.sql_executor.execute_query(corrected_sql)
                            
                            if retry_result['success']:
                                records_found = len(retry_result.get('data', []))
                                print(f"   ✅ ¡Corrección exitosa! Encontrados {records_found} registros")
                                # Formatear los datos para mejor visualización
                                formatted_data = await self._format_sql_results_for_display(retry_result.get('data', []), query)
                                
                                return {
                                    'success': True,
                                    'data': retry_result.get('data', []),
                                    'formatted_data': formatted_data,
                                    'error': '',
                                    'sql_used': corrected_sql,
                                    'execution_time': time.time() - start_time,
                                    'auto_corrected': True,
                                    'correction_type': 'missing_column'
                                }
                            else:
                                print(f"   ❌ La corrección no funcionó: {retry_result.get('error', '')}")
                                # Reintento con prompt alternativo
                                print(f"   🔄 Reintentando con prompt alternativo...")
                                retry_sql = await self._generate_sql_attempt_2(query, list(self.column_metadata.keys()), f"Columna {missing_column} no existe")
                                if retry_sql and len(retry_sql) > 20:
                                    retry_result2 = self.sql_executor.execute_query(retry_sql)
                                    if retry_result2['success']:
                                        records_found = len(retry_result2.get('data', []))
                                        print(f"   ✅ ¡Reintento exitoso! Encontrados {records_found} registros")
                                        formatted_data = await self._format_sql_results_for_display(retry_result2.get('data', []), query)
                                        return {
                                            'success': True,
                                            'data': retry_result2.get('data', []),
                                            'formatted_data': formatted_data,
                                            'error': '',
                                            'sql_used': retry_sql,
                                            'execution_time': time.time() - start_time,
                                            'auto_corrected': True,
                                            'correction_type': 'retry_with_alternative_prompt'
                                        }
                else:
                    print(f"\n   ❌ Error no corregible automáticamente")
                    print(f"   🔍 Detalle: {result.get('error', 'Error desconocido')}")
                    # Último intento con prompt ultra-restrictivo
                    print(f"   🔄 Último intento con prompt ultra-restrictivo...")
                    final_sql = await self._generate_sql_attempt_3(query, list(self.column_metadata.keys()))
                    if final_sql and len(final_sql) > 20 and not final_sql.startswith('-- ERROR:'):
                        final_result = self.sql_executor.execute_query(final_sql)
                        if final_result['success']:
                            records_found = len(final_result.get('data', []))
                            print(f"   ✅ ¡Último intento exitoso! Encontrados {records_found} registros")
                            formatted_data = await self._format_sql_results_for_display(final_result.get('data', []), query)
                            return {
                                'success': True,
                                'data': final_result.get('data', []),
                                'formatted_data': formatted_data,
                                'error': '',
                                'sql_used': final_sql,
                                'execution_time': time.time() - start_time,
                                'auto_corrected': True,
                                'correction_type': 'ultra_restrictive_prompt'
                            }
            
            # Formatear los datos para mejor visualización
            formatted_data = await self._format_sql_results_for_display(result.get('data', []), query)
            
            return {
                'success': result['success'],
                'data': result.get('data', []),
                'formatted_data': formatted_data,
                'error': result.get('error', ''),
                'sql_used': sql,
                'execution_time': time.time() - start_time
            }
        except Exception as e:
            print(f"❌ Error crítico en ejecución SQL: {e}")
            # Formatear datos vacíos para error
            formatted_data = await self._format_sql_results_for_display([], query)
            
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'formatted_data': formatted_data,
                'sql_used': sql,
                'execution_time': time.time() - start_time
            }

    async def _fix_sql_execution_error(self, sql: str, error: str, original_query: str) -> str:
        """Corrige errores de ejecución SQL usando LLM con contexto del error + validación lógica"""
        try:
            print(f"🔧 Analizando error SQL con LLM...")
            
            # Obtener información del esquema real
            schema_info = self._get_schema_context()
            
            prompt = f"""
Eres un experto en SQL médico. Corrige este SQL que falló en ejecución.

CONSULTA ORIGINAL: "{original_query}"

SQL QUE FALLÓ:
```sql
{sql}
```

ERROR DE EJECUCIÓN:
{error}

ESQUEMA REAL DE LA BASE DE DATOS:
{schema_info}

ERRORES COMUNES A CORREGIR:
1. Nombres de campos incorrectos (ejemplo: CONDITION_DESCRIPTION vs APCO_DESCRIPTION_ES)
2. Nombres de tablas incorrectos
3. JOINs con campos incorrectos
4. Alias de tabla incorrectos
5. EXISTS/subconsultas sin correlación correcta con tabla principal
6. Falta de relación entre paciente y sus condiciones/medicamentos

INSTRUCCIONES CRÍTICAS:
- Analiza el error y identifica el problema específico
- Corrige el SQL usando los nombres de campos y tablas REALES del esquema
- IMPORTANTE: Si hay condiciones médicas, SIEMPRE correlacionar con el paciente específico
- Usar JOINs explícitos en lugar de EXISTS cuando sea posible para mejor rendimiento
- Asegúrate de que cada JOIN tenga la correlación correcta (p.PATI_ID = apc.PATI_ID)

EJEMPLO DE CORRECCIÓN:
- MALO: EXISTS (SELECT 1 FROM ACCI_PATIENT_CONDITIONS WHERE condition...)
- BUENO: JOIN ACCI_PATIENT_CONDITIONS apc ON p.PATI_ID = apc.PATI_ID WHERE condition...

RESPUESTA:
```json
{{
  "error_analysis": "análisis detallado del error",
  "logical_issues": "problemas lógicos detectados",
  "corrected_sql": "SQL corregido con nombres reales y lógica correcta"
}}
```

RESPONDE SOLO CON EL JSON.
            """
            
            response = _call_openai_native(self.llm, prompt)
            result = self._try_parse_llm_json(response.content)
            
            if result:
                error_analysis = result.get('error_analysis', 'Análisis no disponible')
                logical_issues = result.get('logical_issues', 'No detectados')
                corrected_sql = result.get('corrected_sql', sql)
                
                print(f"   🔍 Análisis del error: {error_analysis}")
                if logical_issues != 'No detectados':
                    print(f"   ⚠️ Problemas lógicos: {logical_issues}")
                
                if corrected_sql != sql:
                    print(f"   ✅ SQL corregido por LLM")
                    
                    # Validación adicional de la corrección
                    validated_sql = await self._validate_sql_logic_post_correction(corrected_sql, original_query)
                    return validated_sql
                else:
                    print(f"   ⚠️ LLM no pudo corregir el error")
            
            return sql
            
        except Exception as e:
            print(f"   ❌ Error en corrección automática: {e}")
            return sql

    async def _validate_sql_logic_post_correction(self, sql: str, original_query: str) -> str:
        """Validación adicional de la lógica SQL después de la corrección"""
        try:
            print(f"🔍 Validando lógica del SQL corregido...")
            
            prompt = f"""
Eres un experto en validación de SQL médico. Revisa este SQL corregido para detectar errores lógicos.

CONSULTA ORIGINAL: "{original_query}"

SQL CORREGIDO:
```sql
{sql}
```

VALIDACIONES CRÍTICAS:
1. ¿Todas las tablas están correctamente correlacionadas con el paciente?
2. ¿Los JOINs tienen las relaciones correctas (p.PATI_ID = apc.PATI_ID)?
3. ¿Las subconsultas EXISTS están correlacionadas correctamente?
4. ¿La lógica responde realmente a la pregunta médica?

CONSULTA MÉDICA TÍPICA: "pacientes con diabetes que toman metformina"
- Diabetes: debe estar en tabla de condiciones Y correlacionada con el paciente
- Metformina: debe estar en tabla de medicamentos Y correlacionada con el paciente
- Ambas condiciones deben aplicar AL MISMO PACIENTE

RESPUESTA:
```json
{{
  "is_logically_correct": true/false,
  "issues_found": ["problema1", "problema2"],
  "final_sql": "SQL final validado o corregido",
  "validation_notes": "notas sobre la validación"
}}
```

RESPONDE SOLO CON EL JSON.
            """
            
            response = _call_openai_native(self.llm, prompt, task_description="Validación lógica SQL")
            result = self._try_parse_llm_json(response.content)
            
            if result:
                is_correct = result.get('is_logically_correct', False)
                issues = result.get('issues_found', [])
                final_sql = result.get('final_sql', sql)
                notes = result.get('validation_notes', '')
                
                if is_correct:
                    print(f"   ✅ Validación lógica: SQL es correcto")
                else:
                    print(f"   ⚠️ Problemas detectados: {', '.join(issues)}")
                    if final_sql != sql:
                        print(f"   🔧 SQL re-corregido por validación")
                        return final_sql
                
                if notes:
                    print(f"   📝 Notas: {notes}")
                
                return final_sql
            
            return sql
            
        except Exception as e:
            print(f"   ❌ Error en validación lógica: {e}")
            return sql

    async def _attempt_query_recovery(self, query: str, sql: str, error: str, medical_analysis: Dict[str, Any], stream_callback=None) -> Dict[str, Any]:
        """Intento de recuperación"""
        return {
            'success': False,
            'error': 'Recuperación no implementada',
            'data': []
        }



    async def _generate_insert_sql_fallback(self, data: Dict[str, Any]) -> str:
        """Genera SQL INSERT inteligente usando LLM para mapeo dinámico de datos a tablas"""
        try:
            if not self.llm:
                # Fallback básico si no hay LLM disponible
                # Detectar si es medicamento, condición u observación
                if 'patient_id' in data:
                    if 'medication_name' in data or 'dosage' in data:
                        return "INSERT INTO PATI_USUAL_MEDICATION (PATI_ID, PAUM_OBSERVATIONS) VALUES (?, ?);"
                    elif 'condition_description' in data:
                        return "INSERT INTO EPIS_DIAGNOSTICS (PATI_ID, DIAG_OBSERVATION) VALUES (?, ?);"
                    elif 'observation_type' in data:
                        return "INSERT INTO EPIS_DIAGNOSTICS (PATI_ID, DIAG_OBSERVATION) VALUES (?, ?);"
                    else:
                        return "INSERT INTO PATI_PATIENTS (PATI_NAME, PATI_SURNAME_1) VALUES (?, ?);"
                else:
                    return "INSERT INTO PATI_PATIENTS (PATI_NAME, PATI_SURNAME_1) VALUES (?, ?);"
            
            print(f"   🎯 Analizando qué tabla es la más apropiada para estos datos...")
            
            # Obtener esquema completo disponible
            schema_info = self._get_schema_context()
            
            prompt = f"""
Eres un experto en bases de datos médicas. Analiza estos datos y genera SQL INSERT con placeholders (?) para la tabla más apropiada.

DATOS A INSERTAR:
{json.dumps(data, indent=2, ensure_ascii=False)}

ESQUEMA COMPLETO DISPONIBLE:
{schema_info}

INSTRUCCIONES CRÍTICAS:
1. Analiza el tipo de datos y determina la tabla SQL más apropiada
2. Identifica las columnas relevantes en esa tabla
3. Genera SQL INSERT con placeholders (?) para los valores
4. Usa solo columnas que existan en el esquema real
5. Considera el contexto médico de los datos
6. ANALIZA el esquema completo para encontrar la tabla más apropiada
7. Para recursos médicos, busca tablas que contengan información similar
8. Para medicamentos, prioriza tablas con campos de medicación
9. Para diagnósticos, busca tablas con campos de observación o diagnóstico
10. Para observaciones, busca tablas con campos de valores y unidades
11. Para pacientes, busca tablas con información personal y demográfica

REGLAS DE MAPEO INTELIGENTE:
- ANALIZA el esquema completo para encontrar la tabla más apropiada
- Para recursos médicos, busca tablas que contengan información similar
- Para medicamentos, prioriza tablas con campos de medicación
- Para diagnósticos, busca tablas con campos de observación o diagnóstico
- Para observaciones, busca tablas con campos de valores y unidades
- Para pacientes, busca tablas con información personal y demográfica
- Considera múltiples tablas si el recurso puede mapearse a varias

REGLAS CRÍTICAS PARA PATI_ID:
- SIEMPRE incluir PATI_ID cuando el campo "patient_id" esté presente en los datos
- Para medicamentos: PATI_ID es OBLIGATORIO
- Para condiciones: PATI_ID es OBLIGATORIO  
- Para observaciones: PATI_ID es OBLIGATORIO
- Mapear "patient_id" → "PATI_ID" en el SQL
- IMPORTANTE: Los UUIDs deben ir entre comillas simples: 'f55cda1b-5796-4eac-80d3-c3d68c339708'

EJEMPLO DE SALIDA PARA MEDICAMENTOS:
```json
{{
  "tabla": "PATI_USUAL_MEDICATION",
  "sql": "INSERT INTO PATI_USUAL_MEDICATION (PATI_ID, PAUM_OBSERVATIONS) VALUES (?, ?);",
  "razon": "Datos de medicación mapeados a tabla PATI_USUAL_MEDICATION con PATI_ID obligatorio",
  "campos_mapeados": {{
    "patient_id": "PATI_ID",
    "dosage": "PAUM_OBSERVATIONS"
  }},
  "placeholders_count": 2
}}
```

IMPORTANTE: 
- Genera SQL con placeholders (?) para valores dinámicos
- Usa solo columnas que existan en el esquema real
- SIEMPRE incluir PATI_ID cuando esté disponible en los datos
- Responde SOLO con el JSON

Responde SOLO con el JSON.
            """
            
            response = _call_openai_native(self.llm, prompt, task_description="Analizando tabla más apropiada")
            result = self._try_parse_llm_json(response.content)
            
            if result and result.get('sql'):
                sql = result['sql'].strip()
                tabla = result.get('tabla', 'desconocida')
                razon = result.get('razon', '')
                campos_mapeados = result.get('campos_mapeados', {})
                placeholders_count = result.get('placeholders_count', 0)
                
                print(f"   ✅ Tabla seleccionada: {tabla}")
                if razon:
                    print(f"   💡 Razón: {razon[:80]}...")
                print(f"   📊 Campos a insertar: {len(campos_mapeados)}")
                print(f"   🔢 Valores dinámicos: {placeholders_count}")
                
                # Validar que el SQL sea válido
                validation_error = self._validate_insert_columns_exist(sql)
                if validation_error:
                    print(f"   ⚠️ Error en la estructura de la tabla: {validation_error}")
                    print(f"   🔧 Intentando corrección automática...")
                    # Intentar corrección automática
                    corrected_sql = await self._fix_insert_sql_with_llm(sql, data, schema_info, validation_error)
                    if corrected_sql:
                        print(f"   ✅ Corrección exitosa")
                        return corrected_sql
                    else:
                        # Fallback si la corrección falla
                        print(f"   ⚠️ Usando tabla por defecto")
                        return "INSERT INTO PATI_PATIENTS (PATI_NAME, PATI_SURNAME_1) VALUES (?, ?);"
                
                return sql
            else:
                print(f"   ❌ No se pudo determinar la tabla apropiada, usando tabla por defecto")
                return "INSERT INTO PATI_PATIENTS (PATI_NAME, PATI_SURNAME_1) VALUES (?, ?);"
                
        except Exception as e:
            print(f"   ❌ Error analizando datos: {e}")
            return "INSERT INTO PATI_PATIENTS (PATI_NAME, PATI_SURNAME_1) VALUES (?, ?);"

    async def process_data_manipulation(self, operation: str, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Procesa operaciones de manipulación de datos (INSERT, UPDATE, DELETE)
        Compatible con FHIRAgent
        """
        try:
            # --- VALIDACIÓN INTELIGENTE DE PATI_ID CON LLM ---
            table = (context or {}).get('table_hint', '').upper()
            if operation.upper() == 'INSERT' and table:
                validation_result = await self._validate_pati_id_intelligent(data, table, operation)
                if validation_result.get('requires_pati_id', False) and not validation_result.get('valid', False):
                    return {
                        'success': False,
                        'error': f'Validación PATI_ID falló: {validation_result.get("error", "PATI_ID inválido")}',
                        'reasoning': validation_result.get('reasoning', '')
                    }
            # --- FIN VALIDACIÓN INTELIGENTE ---

            print(f"🔄 Procesando operación de datos: {operation}")
            
            # Extraer información del contexto
            intent = context.get('intent', 'general') if context else 'general'
            conn = context.get('conn') if context else None
            
            # Generar SQL para la operación
            if operation.upper() == 'INSERT':
                sql_result = await self._generate_insert_sql(data)
                sql = sql_result['sql']
                values = sql_result.get('values', [])
                table_used = sql_result.get('table', 'desconocida')
            elif operation.upper() == 'UPDATE':
                sql = self._generate_update_sql(data)
                values = []
                table_used = self._extract_table_name(sql)
            elif operation.upper() == 'DELETE':
                sql = self._generate_delete_sql(data)
                values = []
                table_used = self._extract_table_name(sql)
            else:
                return {
                    'success': False,
                    'error': f'Operación no soportada: {operation}',
                    'data': []
                }
            
            print(f"   💾 SQL generado: {sql[:100]}...")
            if values:
                print(f"   📊 Valores: {values}")
            
            # Ejecutar SQL usando la conexión proporcionada o crear una nueva
            debug_info = {}
            if conn:
                # Usar conexión existente (para transacciones)
                try:
                    cursor = conn.cursor()
                    if values and '?' in sql:
                        cursor.execute(sql, values)
                    else:
                        cursor.execute(sql)
                    # No hacer commit aquí, dejar que el llamador maneje la transacción
                    result = {
                        'success': True,
                        'data': [],
                        'sql_used': sql,
                        'table_used': table_used
                    }
                except Exception as e:
                    result = {
                        'success': False,
                        'error': str(e),
                        'sql_used': sql,
                        'table_used': table_used
                    }
            else:
                # Usar SQL executor normal
                if values and '?' in sql:
                    result = self.sql_executor.execute_query(sql, values)
                else:
                    # SQL ya tiene valores específicos
                    result = self.sql_executor.execute_query(sql)

            # DEBUG: Log completo tras inserción de medicamentos
            if table_used == 'PATI_USUAL_MEDICATION' and operation.upper() == 'INSERT':
                pati_id = None
                if 'PATI_ID' in sql or 'patient_id' in str(data):
                    # Intentar extraer el PATI_ID del SQL o de los datos
                    import re
                    match = re.search(r"VALUES\s*\(\s*'([^']+)'", sql)
                    if match:
                        pati_id = match.group(1)
                    elif 'patient_id' in data:
                        pati_id = data.get('patient_id')
                print("\n🩺 [DEBUG MEDICAMENTO] -- RESUMEN DE INSERCIÓN")
                print(f"   📝 SQL ejecutado: {sql}")
                print(f"   📋 Tabla: {table_used}")
                print(f"   📦 Datos: {data}")
                print(f"   🆔 PATI_ID: {pati_id}")
                print(f"   ✅ Éxito: {result.get('success', False)}")
                if not result.get('success', True):
                    print(f"   ❌ Error: {result.get('error', '')}")
                # Consulta inmediata para verificar inserción
                if pati_id:
                    try:
                        check_sql = f"SELECT * FROM PATI_USUAL_MEDICATION WHERE PATI_ID = '{pati_id}' ORDER BY rowid DESC LIMIT 3;"
                        check_result = self.sql_executor.execute_query(check_sql)
                        print(f"   🔎 Verificación inmediata en BBDD (últimos 3 registros para PATI_ID={pati_id}):")
                        for row in check_result.get('data', []):
                            print(f"      - {row}")
                    except Exception as e:
                        print(f"   ⚠️ Error al verificar inserción: {e}")
                print("🩺 [FIN DEBUG MEDICAMENTO]\n")
            
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
                'message': f'Error en manipulación de datos: {str(e)}'
            }
    
    async def _generate_insert_sql(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Genera SQL INSERT inteligente usando LLM para mapeo dinámico FHIR→SQL con valores específicos"""
        try:
            if not self.llm:
                # Fallback básico si no hay LLM
                fallback_sql = await self._generate_insert_sql_fallback(data)
                return {
                    'sql': fallback_sql,
                    'values': [],
                    'table': 'PATI_PATIENTS'
                }

            # Obtener esquema completo disponible
            schema_info = self._get_schema_context()
            
            prompt = f"""
Eres un experto en mapeo FHIR→SQL médico. Analiza estos datos y genera SQL INSERT COMPLETO con valores específicos.

DATOS FHIR A INSERTAR:
{json.dumps(data, indent=2, ensure_ascii=False)}

ESQUEMA COMPLETO DISPONIBLE:
{schema_info}

INSTRUCCIONES CRÍTICAS:
1. Analiza el tipo de recurso FHIR (Patient, Condition, Medication, etc.)
2. Identifica la tabla SQL más apropiada según las reglas de mapeo
3. Mapea cada campo FHIR a una columna SQL específica
4. Genera valores SQL específicos (NO placeholders ?)
5. Usa solo columnas que existan en el esquema real
6. Para campos de texto, escapa correctamente las comillas
7. Para fechas, usa formato SQLite estándar
8. Para valores numéricos, no uses comillas

REGLAS DE MAPEO FHIR→SQL INTELIGENTE:
- ANALIZA el esquema completo para encontrar la tabla más apropiada
- Para recursos médicos, busca tablas que contengan información similar
- Para medicamentos, prioriza tablas con campos de medicación
- Para diagnósticos, busca tablas con campos de observación o diagnóstico
- Para observaciones, busca tablas con campos de valores y unidades
- Para pacientes, busca tablas con información personal y demográfica
- Considera múltiples tablas si el recurso puede mapearse a varias

REGLAS CRÍTICAS PARA PATI_ID:
- SIEMPRE incluir PATI_ID cuando el campo "patient_id" esté presente en los datos
- Para medicamentos: PATI_ID es OBLIGATORIO
- Para condiciones: PATI_ID es OBLIGATORIO  
- Para observaciones: PATI_ID es OBLIGATORIO
- Mapear "patient_id" → "PATI_ID" en el SQL
- IMPORTANTE: Los UUIDs deben ir entre comillas simples: 'f55cda1b-5796-4eac-80d3-c3d68c339708'

EJEMPLO DE SALIDA PARA MEDICAMENTOS:
```json
{{
  "tabla": "PATI_USUAL_MEDICATION",
  "sql": "INSERT INTO PATI_USUAL_MEDICATION (PATI_ID, PAUM_OBSERVATIONS) VALUES (?, ?);",
  "razon": "Datos de medicación mapeados a tabla PATI_USUAL_MEDICATION con PATI_ID obligatorio",
  "campos_mapeados": {{
    "patient_id": "PATI_ID",
    "dosage": "PAUM_OBSERVATIONS"
  }}
}}
```

IMPORTANTE: 
- Genera SQL COMPLETO con valores específicos, NO placeholders
- SIEMPRE incluir PATI_ID cuando esté disponible en los datos
- Responde SOLO con el JSON
            """
            
            response = _call_openai_native(self.llm, prompt, task_description="Generando SQL INSERT inteligente")
            result = self._try_parse_llm_json(response.content)
            
            if result and result.get('sql'):
                sql = result['sql'].strip()
                tabla = result.get('tabla', 'desconocida')
                razon = result.get('razon', '')
                campos_mapeados = result.get('campos_mapeados', {})
                
                print(f"   🎯 Tabla seleccionada: {tabla}")
                if razon:
                    print(f"   💡 Razón: {razon[:100]}...")
                print(f"   📊 Campos mapeados: {len(campos_mapeados)} campos")
                
                # Validar que el SQL sea válido
                validation_error = self._validate_insert_columns_exist(sql)
                if validation_error:
                    print(f"   ⚠️ Error de validación: {validation_error}")
                    # Intentar corrección automática
                    corrected_sql = await self._fix_insert_sql_with_llm(sql, data, schema_info, validation_error)
                    if corrected_sql:
                        return {
                            'sql': corrected_sql,
                            'values': [],  # SQL ya tiene valores específicos
                            'table': tabla
                        }
                
                # NUEVO: Validar que se incluya PATI_ID cuando esté disponible en los datos
                if 'patient_id' in data and 'PATI_ID' not in sql:
                    print(f"   ⚠️ PATI_ID faltante en SQL generado, corrigiendo...")
                    corrected_sql = await self._fix_missing_pati_id(sql, data, tabla)
                    if corrected_sql:
                        return {
                            'sql': corrected_sql,
                            'values': [],
                            'table': tabla
                        }
                
                return {
                    'sql': sql,
                    'values': [],  # SQL ya tiene valores específicos
                    'table': tabla
                }
            else:
                print(f"   ❌ LLM no pudo generar SQL válido, usando fallback")
                fallback_sql = await self._generate_insert_sql_fallback(data)
                return {
                    'sql': fallback_sql,
                    'values': [],
                    'table': 'PATI_PATIENTS'
                }
                
        except Exception as e:
            print(f"   ❌ Error en generación SQL inteligente: {e}")
            fallback_sql = await self._generate_insert_sql_fallback(data)
            return {
                'sql': fallback_sql,
                'values': [],
                'table': 'PATI_PATIENTS'
            }

    async def _fix_insert_sql_with_llm(self, sql: str, data: Dict[str, Any], schema_info: str, error: str) -> str:
        """Corrige SQL INSERT usando LLM cuando hay errores de validación"""
        try:
            prompt = f"""
El SQL INSERT generado tiene errores. Corrígelo usando el esquema real.

SQL CON ERROR:
{sql}

ERROR DETECTADO:
{error}

DATOS A INSERTAR:
{json.dumps(data, indent=2, ensure_ascii=False)}

ESQUEMA REAL:
{schema_info}

INSTRUCCIONES:
1. Corrige el SQL usando solo columnas que existan en el esquema
2. Mantén los valores específicos (NO placeholders)
3. Usa la tabla correcta según el tipo de datos
4. Escapa correctamente los valores de texto

Responde SOLO con el SQL corregido.
            """
            
            response = _call_openai_native(self.llm, prompt, task_description="Corrigiendo SQL INSERT")
            corrected_sql = response.content.strip()
            
            # Limpiar formato markdown si existe
            if corrected_sql.startswith('```sql'):
                corrected_sql = corrected_sql[6:]
            if corrected_sql.startswith('```'):
                corrected_sql = corrected_sql[3:]
            if corrected_sql.endswith('```'):
                corrected_sql = corrected_sql[:-3]
            
            corrected_sql = corrected_sql.strip()
            
            # Validar de nuevo
            new_error = self._validate_insert_columns_exist(corrected_sql)
            if new_error:
                print(f"   ❌ Corrección falló: {new_error}")
                return ""
            
            print(f"   ✅ SQL corregido exitosamente")
            return corrected_sql
            
        except Exception as e:
            print(f"   ❌ Error en corrección SQL: {e}")
            return ""

    async def _fix_missing_pati_id(self, sql: str, data: Dict[str, Any], table_name: str) -> str:
        """Corrige SQL que no incluye PATI_ID cuando debería incluirlo"""
        try:
            if not self.llm:
                return sql
            
            patient_id = data.get('patient_id')
            if not patient_id:
                return sql
            
            prompt = f"""
El SQL generado no incluye PATI_ID aunque está disponible en los datos. Corrígelo.

SQL ACTUAL:
{sql}

DATOS DISPONIBLES:
{json.dumps(data, indent=2, ensure_ascii=False)}

TABLA: {table_name}
PATIENT_ID DISPONIBLE: {patient_id}

INSTRUCCIONES:
1. Añade PATI_ID a la lista de columnas
2. Añade el valor del patient_id a la lista de valores
3. Mantén el resto del SQL igual
4. Escapa correctamente los valores de texto

EJEMPLO DE CORRECCIÓN:
- ANTES: INSERT INTO PATI_USUAL_MEDICATION (PAUM_OBSERVATIONS) VALUES ('Metformina');
- DESPUÉS: INSERT INTO PATI_USUAL_MEDICATION (PATI_ID, PAUM_OBSERVATIONS) VALUES (610, 'Metformina');

Responde SOLO con el SQL corregido.
            """
            
            response = _call_openai_native(self.llm, prompt, task_description="Corrigiendo PATI_ID faltante")
            corrected_sql = response.content.strip()
            
            # Limpiar formato markdown si existe
            if corrected_sql.startswith('```sql'):
                corrected_sql = corrected_sql[6:]
            if corrected_sql.startswith('```'):
                corrected_sql = corrected_sql[3:]
            if corrected_sql.endswith('```'):
                corrected_sql = corrected_sql[:-3]
            
            corrected_sql = corrected_sql.strip()
            
            # Verificar que ahora sí incluye PATI_ID
            if 'PATI_ID' in corrected_sql:
                print(f"   ✅ PATI_ID añadido correctamente")
                return corrected_sql
            else:
                print(f"   ❌ No se pudo añadir PATI_ID")
                return sql
            
        except Exception as e:
            print(f"   ❌ Error corrigiendo PATI_ID: {e}")
            return sql

    def _validate_insert_columns_exist(self, sql: str) -> str:
        """Valida que todas las columnas usadas en el SQL existen en el esquema. Devuelve string de error si hay alguna inválida."""
        try:
            import re
            # Extraer nombres de tabla y columnas del SQL
            matches = re.findall(r'INSERT INTO (\w+) \(([^)]+)\)', sql, re.IGNORECASE)
            if not matches:
                return "No se pudo extraer tabla y columnas del SQL."
            table, columns_str = matches[0]
            columns = [c.strip() for c in columns_str.split(',')]
            if table not in self.column_metadata:
                return f"La tabla '{table}' no existe en el esquema."
            real_columns = [col['name'] for col in self.column_metadata[table]['columns']]
            invalid = [c for c in columns if c not in real_columns]
            if invalid:
                return f"Columnas inválidas para la tabla {table}: {', '.join(invalid)}. Columnas válidas: {', '.join(real_columns)}"
            return ""
        except Exception as e:
            return f"Error validando columnas: {e}"
    
    def _generate_update_sql(self, data: Dict[str, Any]) -> str:
        """Genera SQL UPDATE usando LLM para mapeo flexible"""
        try:
            if not self.llm:
                return "UPDATE PATI_PATIENTS SET PATI_ACTIVE = 1 WHERE PATI_ID = 1;"
            
            # Obtener esquema disponible
            schema_info = self._get_schema_context()
            
            prompt = f"""
Eres un experto en bases de datos médicas. Genera SQL UPDATE para estos datos:

DATOS A ACTUALIZAR:
{json.dumps(data, indent=2, ensure_ascii=False)}

ESQUEMA DISPONIBLE:
{schema_info}

INSTRUCCIONES:
1. Analiza los datos y determina la tabla más apropiada
2. Mapea los campos de datos a columnas reales de la tabla
3. Genera SQL UPDATE válido con los campos correctos
4. Escapa correctamente los valores de texto
5. Usa solo columnas que existan en el esquema
6. NO actualices campos de ID (claves primarias)

RESPUESTA:
```json
{{
  "tabla": "nombre_de_la_tabla",
  "sql": "UPDATE tabla SET campo1 = 'valor1', campo2 = valor2 WHERE id = 1;",
  "razon": "explicación breve del mapeo"
}}
```

Responde SOLO con el JSON.
            """
            
            response = _call_openai_native(self.llm, prompt, task_description="Generando SQL UPDATE")
            result = self._try_parse_llm_json(response.content)
            
            if result and result.get('sql'):
                print(f"   ✅ SQL UPDATE generado por LLM: {result.get('tabla', 'N/A')}")
                return result['sql']
            
            # Fallback si LLM falla
            return "UPDATE PATI_PATIENTS SET PATI_ACTIVE = 1 WHERE PATI_ID = 1;"
            
        except Exception as e:
            return f"UPDATE PATI_PATIENTS SET PATI_ACTIVE = 1 WHERE PATI_ID = 1; -- Error: {str(e)}"
    
    def _generate_delete_sql(self, data: Dict[str, Any]) -> str:
        """Genera SQL DELETE usando LLM para mapeo flexible"""
        try:
            if not self.llm:
                return "DELETE FROM PATI_PATIENTS WHERE PATI_ID = 1;"
            
            # Obtener esquema disponible
            schema_info = self._get_schema_context()
            
            prompt = f"""
Eres un experto en bases de datos médicas. Genera SQL DELETE para estos datos:

DATOS PARA ELIMINAR:
{json.dumps(data, indent=2, ensure_ascii=False)}

ESQUEMA DISPONIBLE:
{schema_info}

INSTRUCCIONES:
1. Analiza los datos y determina la tabla más apropiada
2. Identifica el campo de ID o clave primaria para la condición WHERE
3. Genera SQL DELETE válido con la condición correcta
4. Usa solo columnas que existan en el esquema
5. Sé específico con la condición WHERE para evitar eliminaciones accidentales

RESPUESTA:
```json
{{
  "tabla": "nombre_de_la_tabla",
  "sql": "DELETE FROM tabla WHERE campo_id = valor;",
  "razon": "explicación breve del mapeo"
}}
```

Responde SOLO con el JSON.
            """
            
            response = _call_openai_native(self.llm, prompt, task_description="Generando SQL DELETE")
            result = self._try_parse_llm_json(response.content)
            
            if result and result.get('sql'):
                print(f"   ✅ SQL DELETE generado por LLM: {result.get('tabla', 'N/A')}")
                return result['sql']
            
            # Fallback si LLM falla
            return "DELETE FROM PATI_PATIENTS WHERE PATI_ID = 1;"
            
        except Exception as e:
            return f"DELETE FROM PATI_PATIENTS WHERE PATI_ID = 1; -- Error: {str(e)}"
    
    def _try_parse_llm_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Intenta parsear JSON de respuesta del LLM con manejo mejorado de errores"""
        try:
            # Importar utilidad de parseo JSON
            try:
                from ..utils.json_parser import robust_json_parse
            except ImportError:
                # Fallback si no se puede importar
                def robust_json_parse(content: str, fallback=None):
                    try:
                        return json.loads(content.strip())
                    except:
                        return fallback
            
            # Intentar parseo robusto
            result = robust_json_parse(content, fallback=None)
            
            if result:
                print(f"   ✅ JSON parseado exitosamente")
                return result
            
            # Si falla, intentar reparar JSON incompleto
            print(f"   ⚠️ JSON incompleto detectado, intentando reparar...")
            print(f"   🔍 DEBUG: Contenido recibido: {content[:300]}...")
            
            # Buscar JSON incompleto y completarlo
            repaired_content = self._repair_incomplete_json(content)
            if repaired_content:
                result = robust_json_parse(repaired_content, fallback=None)
                if result:
                    print(f"   ✅ JSON reparado y parseado exitosamente")
                    return result
            
            print(f"   ❌ No se pudo parsear JSON del contenido")
            return None
            
        except Exception as e:
            print(f"❌ Error general parseando JSON: {e}")
            print(f"   🔍 DEBUG: Contenido que causó error: {content[:100]}...")
            return None
    
    def _repair_incomplete_json(self, content: str) -> Optional[str]:
        """Repara JSON incompleto detectando patrones comunes"""
        try:
            # Limpiar contenido
            content = content.strip()
            
            # Remover markdown si está presente
            if content.startswith('```json'):
                content = content[7:]
            elif content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            
            # Buscar JSON incompleto con patrones comunes
            json_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # JSON completo
                r'\{.*"concept_mappings".*\}',       # JSON con concept_mappings
                r'\{.*"tablas_requeridas".*\}',      # JSON con tablas_requeridas
                r'\{.*"concepts".*\}',               # JSON con concepts
            ]
            
            for pattern in json_patterns:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    # Intentar completar JSON si está incompleto
                    if not json_str.endswith('}'):
                        # Buscar el último objeto completo
                        brace_count = 0
                        last_complete = 0
                        for i, char in enumerate(json_str):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    last_complete = i + 1
                        
                        if last_complete > 0:
                            json_str = json_str[:last_complete]
                    
                    # Validar que sea JSON válido
                    try:
                        json.loads(json_str)
                        return json_str
                    except json.JSONDecodeError:
                            continue
            
            return None
            
        except Exception as e:
            print(f"   ⚠️ Error reparando JSON: {e}")
            return None

    def _get_detailed_schema_for_tables(self, tables: List[str]) -> str:
        """Obtiene información detallada del esquema para las tablas seleccionadas"""
        try:
            schema_info = []
            
            for table_name in tables:
                if table_name in self.column_metadata:
                    table_info = self.column_metadata[table_name]
                    schema_info.append(f"\nTABLA: {table_name}")
                    schema_info.append(f"- Filas: {table_info.get('row_count', 'N/A')}")
                    schema_info.append("- Columnas:")
                    
                    # Mostrar todas las columnas para tablas seleccionadas
                    for col in table_info.get('columns', []):
                        col_name = col.get('name', '')
                        col_type = col.get('type', '')
                        is_pk = col.get('primary_key', False)
                        pk_indicator = " (PK)" if is_pk else ""
                        schema_info.append(f"  * {col_name} ({col_type}){pk_indicator}")
                    
                    # Mostrar datos de muestra si están disponibles
                    sample_data = table_info.get('sample_data', [])
                    if sample_data:
                        schema_info.append("- Ejemplo de datos:")
                        for i, row in enumerate(sample_data[:2], 1):
                            if len(row) > 2:  # Asegurar que hay datos
                                schema_info.append(f"  Registro {i}: {row[:5]}...")  # Primeras 5 columnas
            
            return "\n".join(schema_info) if schema_info else "Esquema no disponible"
            
        except Exception as e:
            return f"Error obteniendo esquema detallado: {e}"

    def _get_schema_context(self) -> str:
        """Obtiene información detallada del esquema completo para mapeo dinámico"""
        try:
            schema_info = []
            
            # Obtener TODAS las tablas disponibles
            all_tables = list(self.column_metadata.keys())
            
            # Ordenar tablas por relevancia médica
            medical_tables = [
                'PATI_PATIENTS',
                'EPIS_DIAGNOSTICS', 
                'PATI_USUAL_MEDICATION',
                'MEDI_ACTIVE_INGREDIENTS',
                'APPO_APPOINTMENTS',
                'PATI_PATIENT_ALLERGIES',
                'ACCI_PATIENT_CONDITIONS'
            ]
            
            # Agregar primero las tablas médicas más relevantes
            for table_name in medical_tables:
                if table_name in self.column_metadata:
                    table_info = self.column_metadata[table_name]
                    schema_info.append(f"\n🏥 TABLA MÉDICA: {table_name}")
                    schema_info.append(f"- Filas: {table_info.get('row_count', 'N/A')}")
                    schema_info.append("- Columnas REALES:")
                    
                    # Mostrar TODAS las columnas para mapeo preciso
                    for col in table_info.get('columns', []):
                        col_name = col.get('name', '')
                        col_type = col.get('type', '')
                        pk_indicator = " (PK)" if col.get('primary_key') else ""
                        schema_info.append(f"  * {col_name} ({col_type}){pk_indicator}")
                    
                    # Mostrar datos de muestra para entender el contenido
                    sample_data = table_info.get('sample_data', [])
                    if sample_data and len(sample_data) > 0:
                        schema_info.append("- Ejemplo de datos:")
                        column_names = [col['name'] for col in table_info.get('columns', [])]
                        for i, row in enumerate(sample_data[:2], 1):
                            if len(row) >= 3:
                                sample_display = []
                                for j, value in enumerate(row[:3]):
                                    col_name = column_names[j] if j < len(column_names) else f"col{j}"
                                    sample_display.append(f"{col_name}={value}")
                                schema_info.append(f"  Registro {i}: {', '.join(sample_display)}")
            
            # Agregar otras tablas disponibles (máximo 10 más)
            other_tables = [t for t in all_tables if t not in medical_tables][:10]
            if other_tables:
                schema_info.append(f"\n📋 OTRAS TABLAS DISPONIBLES ({len(other_tables)}):")
                for table_name in other_tables:
                    if table_name in self.column_metadata:
                        table_info = self.column_metadata[table_name]
                        row_count = table_info.get('row_count', 'N/A')
                        columns = [col['name'] for col in table_info.get('columns', [])]
                        schema_info.append(f"- {table_name}: {row_count} filas, {len(columns)} columnas")
                        # Mostrar solo las primeras 5 columnas para no sobrecargar
                        schema_info.append(f"  Columnas: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}")
            
            # Información sobre relaciones dinámicas
            schema_info.append("\n🔍 RELACIONES DETECTADAS DINÁMICAMENTE:")
            schema_info.append("- PATI_ID: Campo común para relacionar pacientes")
            schema_info.append("- EPIS_ID: Para episodios clínicos")
            schema_info.append("- MEDI_ID: Para medicamentos")
            schema_info.append("- APPO_ID: Para citas y observaciones")
            schema_info.append("- LLM analizará relaciones específicas según los datos")
            
            # IMPORTANTE: Lista de tablas que SÍ existen
            schema_info.append(f"\n✅ TABLAS DISPONIBLES REALES:")
            schema_info.append(f"- Solo usar estas tablas: {', '.join(all_tables)}")
            schema_info.append(f"- NO usar tablas que no estén en esta lista")
            
            return "\n".join(schema_info) if schema_info else "Esquema no disponible"
            
        except Exception as e:
            return f"Error obteniendo esquema: {e}"

    async def _llm_detect_description_field(self, table_name: str) -> Optional[str]:
        """Usa el LLM para detectar el campo de descripción más adecuado en una tabla dada"""
        if not self.llm or table_name not in self.column_metadata:
            return None
        columns = [col['name'] for col in self.column_metadata[table_name]['columns']]
        prompt = f"""
Dada la tabla '{table_name}' con las columnas: {columns},
¿cuál es el campo más adecuado para buscar descripciones clínicas, diagnósticos o información textual relevante?
Responde SOLO con el nombre exacto de la columna (sin explicaciones).
Si no hay un campo adecuado, responde SOLO con 'NINGUNO'.
"""
        response = _call_openai_native(self.llm, prompt)
        field = response.content.strip().split()[0]
        if field.upper() == 'NINGUNO':
            return None
        return field

    async def _llm_detect_join_columns(self, table1: str, table2: str) -> Optional[Tuple[str, str]]:
        """Usa el LLM para detectar las columnas de JOIN entre dos tablas"""
        if not self.llm or table1 not in self.column_metadata or table2 not in self.column_metadata:
            return None
        columns1 = [col['name'] for col in self.column_metadata[table1]['columns']]
        columns2 = [col['name'] for col in self.column_metadata[table2]['columns']]
        prompt = f"""
Dadas las tablas '{table1}' y '{table2}' con sus columnas:
{table1}: {columns1}
{table2}: {columns2}

¿Cuáles son las columnas que permiten hacer un JOIN entre estas tablas?
Busca columnas con nombres similares o que representen la misma entidad (ej: ID, PATI_ID, etc.)

Responde SOLO con el formato: columna1,columna2
Si no hay relación clara, responde SOLO con 'NINGUNA'.
"""
        response = _call_openai_native(self.llm, prompt, task_description=f"Detectando JOIN {table1}-{table2}")
        result = response.content.strip()
        if result.upper() == 'NINGUNA':
            return None
        parts = result.split(',')
        if len(parts) == 2:
            join_result = (parts[0].strip(), parts[1].strip())
            print(f"   🔗 JOIN detectado: {table1}.{join_result[0]} = {table2}.{join_result[1]}")
            return join_result
        return None
    
    async def _format_sql_results_for_display(self, data: List[Dict[str, Any]], query: str) -> str:
        """Formatea los resultados SQL usando LLM para análisis dinámico e inteligente"""
        try:
            if not data:
                return "📊 **No se encontraron resultados**"
            
            # Usar LLM para análisis dinámico si está disponible
            if self.llm:
                return await self._format_with_llm_analysis(data, query)
            else:
                # Fallback al método original
                return await self._format_with_fallback(data, query)
                
        except Exception as e:
            print(f"   ❌ Error en formateo con LLM: {e}")
            return await self._format_with_fallback(data, query)
    
    async def _format_with_llm_analysis(self, data: List[Dict[str, Any]], query: str) -> str:
        """Formatea resultados usando análisis dinámico con LLM"""
        try:
            # Verificar que data sea una lista de diccionarios
            if not data or not isinstance(data, list):
                return await self._format_with_fallback(data, query)
            
            # Verificar que el primer elemento sea un diccionario
            if not isinstance(data[0], dict):
                return await self._format_with_fallback(data, query)
            
            # Preparar datos para el LLM (solo primeros registros para evitar tokens excesivos)
            sample_data = data[:3] if len(data) > 3 else data
            
            prompt = f"""
Eres un experto en análisis médico de datos. Analiza estos resultados y formatea la respuesta de manera médicamente relevante y dinámica.

CONSULTA ORIGINAL: "{query}"

DATOS ENCONTRADOS:
{json.dumps(sample_data, indent=2, ensure_ascii=False)}

TOTAL DE REGISTROS: {len(data)}

ANÁLISIS DINÁMICO REQUERIDO:
1. **Identifica el tipo de consulta médica**:
   - Historia clínica
   - Medicación
   - Diagnósticos
   - Estadísticas/conteo
   - Búsqueda de paciente
   - Información temporal (último, reciente)
   - Otros

2. **Analiza los datos encontrados**:
   - ¿Qué información médica contienen?
   - ¿Son datos de pacientes, medicamentos, diagnósticos?
   - ¿Hay patrones o información relevante?
   - ¿Los datos están completos o faltan campos?

3. **Genera una respuesta formateada dinámica**:
   - Título descriptivo y relevante
   - Resumen de lo encontrado
   - Lista de resultados relevantes
   - Información médica destacada
   - Observaciones o recomendaciones si aplica

FORMATO DE RESPUESTA:
```json
{{
  "tipo_consulta": "historia_clinica|medicacion|diagnostico|estadistica|busqueda_paciente|temporal|otros",
  "titulo": "Título descriptivo y relevante",
  "resumen": "Resumen de lo que se encontró",
  "resultados_formateados": [
    "Formato de cada resultado relevante"
  ],
  "informacion_medica": "Información médica destacada o relevante",
  "observaciones": "Observaciones o recomendaciones médicas si aplica",
  "completitud_datos": "alta|media|baja"
}}
```

Responde SOLO con el JSON.
"""
            
            response = _call_openai_native(self.llm, prompt, task_description="Análisis dinámico de resultados")
            result = self._try_parse_llm_json(response.content)
            
            # Verificar que result sea un diccionario válido
            if result and isinstance(result, dict) and not isinstance(result, list):
                # Construir respuesta formateada
                formatted_response = []
                
                # Título
                titulo = result.get('titulo', '📊 RESULTADOS ENCONTRADOS')
                formatted_response.append(f"**{titulo}**")
                formatted_response.append("")
                
                # Resumen
                resumen = result.get('resumen', '')
                if resumen:
                    formatted_response.append(f"📋 **Resumen**: {resumen}")
                    formatted_response.append("")
                
                # Resultados formateados
                resultados = result.get('resultados_formateados', [])
                if isinstance(resultados, list):
                    for i, resultado in enumerate(resultados[:10], 1):  # Máximo 10
                        formatted_response.append(f"   {i}. {resultado}")
                    
                    if len(data) > 10:
                        formatted_response.append("")
                        formatted_response.append(f"📄 *... y {len(data) - 10} resultados más*")
                
                # Información médica
                info_medica = result.get('informacion_medica', '')
                if info_medica:
                    formatted_response.append("")
                    formatted_response.append(f"🏥 **Información Médica**: {info_medica}")
                
                # Observaciones
                observaciones = result.get('observaciones', '')
                if observaciones:
                    formatted_response.append("")
                    formatted_response.append(f"💡 **Observaciones**: {observaciones}")
                
                return "\n".join(formatted_response)
            
            # Fallback si el LLM falla
            return await self._format_with_fallback(data, query)
            
        except Exception as e:
            print(f"   ❌ Error en análisis dinámico: {e}")
            return await self._format_with_fallback(data, query)
    
    async def _format_with_fallback(self, data: List[Dict[str, Any]], query: str) -> str:
        """Formateo de fallback usando el método original"""
        query_lower = query.lower()
        
                # Si es una consulta de distribución por grupos de edad
        if any(word in query_lower for word in ['edad', 'grupo', 'distribución', 'distribucion']):
            return self._format_age_distribution_results(data)
        
        # Si es una consulta de conteo simple
        if any(word in query_lower for word in ['cuántos', 'cuantos', 'total', 'contar']):
            return self._format_count_results(data)
        
        # Si es una consulta de medicamentos
        if any(word in query_lower for word in ['medicamento', 'medicación', 'medicina', 'fármaco']):
            return self._format_medication_results(data)
        
        # Si es una consulta de diagnósticos
        if any(word in query_lower for word in ['diagnóstico', 'diagnostico', 'condición', 'condicion']):
            return self._format_diagnosis_results(data)
        
        # Formato genérico para otros tipos de consultas
        return await self._format_generic_results(data)
    
    async def _get_patient_basic_info(self, query: str) -> Optional[str]:
        """Obtiene información básica del paciente cuando no se encuentran resultados médicos"""
        try:
            # Extraer nombre del paciente de la consulta
            patient_name = await self._extract_patient_name_from_query(query)
            if not patient_name:
                return None
            
            # Buscar información básica del paciente
            sql = f"""
            SELECT PATI_NAME, PATI_SURNAME_1, PATI_SURNAME_2, PATI_FULL_NAME, 
                   PATI_CLINICAL_HISTORY_ID, PATI_START_DATE, PATI_ACTIVE
            FROM PATI_PATIENTS 
            WHERE PATI_NAME LIKE '%{patient_name}%' 
               OR PATI_SURNAME_1 LIKE '%{patient_name}%'
               OR PATI_FULL_NAME LIKE '%{patient_name}%'
            LIMIT 1
            """
            
            result = self.sql_executor.execute_query(sql)
            if result and result.get('success') and result.get('data'):
                patient_data = result['data'][0]
                
                formatted_info = []
                formatted_info.append("👤 **INFORMACIÓN DEL PACIENTE**")
                formatted_info.append("")
                
                # Nombre completo
                full_name = patient_data.get('PATI_FULL_NAME', '')
                if not full_name:
                    name = patient_data.get('PATI_NAME', '')
                    surname = patient_data.get('PATI_SURNAME_1', '')
                    full_name = f"{name} {surname}".strip()
                
                formatted_info.append(f"**Nombre:** {full_name}")
                
                # ID de historia clínica
                clinical_id = patient_data.get('PATI_CLINICAL_HISTORY_ID', '')
                if clinical_id:
                    formatted_info.append(f"**ID Historia Clínica:** {clinical_id}")
                
                # Fecha de registro
                start_date = patient_data.get('PATI_START_DATE', '')
                if start_date:
                    formatted_info.append(f"**Fecha de Registro:** {start_date}")
                
                # Estado
                active = patient_data.get('PATI_ACTIVE', '')
                if active is not None:
                    status = "Activo" if active == 1 else "Inactivo"
                    formatted_info.append(f"**Estado:** {status}")
                
                return "\n".join(formatted_info)
            
            return None
            
        except Exception as e:
            print(f"   ❌ Error obteniendo información del paciente: {e}")
            return None
    
    def _format_age_distribution_results(self, data: List[Dict[str, Any]]) -> str:
        """Formatea resultados de distribución por edad"""
        if not data:
            return "📊 **No se encontraron datos de distribución por edad**"
        
        formatted = []
        formatted.append("📊 **DISTRIBUCIÓN POR GRUPOS DE EDAD**")
        formatted.append("")
        
        total_patients = sum(item.get('patient_count', 0) for item in data)
        
        for item in data:
            age_group = item.get('age_group', 'N/A')
            patient_count = item.get('patient_count', 0)
            percentage = (patient_count / total_patients * 100) if total_patients > 0 else 0
            
            # Emoji basado en el grupo de edad
            if '0-12' in age_group or '13-19' in age_group:
                emoji = "👶"
            elif '20-35' in age_group or '36-50' in age_group:
                emoji = "👨‍⚕️"
            elif '51-65' in age_group:
                emoji = "👴"
            else:
                emoji = "👵"
            
            formatted.append(f"   {emoji} **{age_group} años:** {patient_count} pacientes ({percentage:.1f}%)")
        
        formatted.append("")
        formatted.append(f"📈 **Total:** {total_patients} pacientes")
        
        return "\n".join(formatted)
    
    def _format_count_results(self, data: List[Dict[str, Any]]) -> str:
        """Formatea resultados de conteo"""
        if not data:
            return "📊 **No se encontraron resultados**"
        
        # Si es un solo número
        if len(data) == 1 and len(data[0]) == 1:
            count = list(data[0].values())[0]
            return f"📊 **Total encontrado:** {count}"
        
        # Si son múltiples conteos
        formatted = []
        formatted.append("📊 **RESULTADOS DE CONTEO**")
        formatted.append("")
        
        for item in data:
            for key, value in item.items():
                formatted.append(f"   📋 **{key}:** {value}")
        
        return "\n".join(formatted)
    def _format_medication_results(self, data: List[Dict[str, Any]]) -> str:
        """Formatea resultados de medicamentos de forma inteligente"""
        if not data:
            return "💊 **No se encontraron medicamentos**"
        
        formatted = []
        formatted.append("💊 **MEDICAMENTOS ENCONTRADOS**")
        formatted.append("")
        
        # Detectar el tipo de resultado basado en las columnas disponibles
        if data and isinstance(data[0], dict):
            first_item = data[0]
            
            # Caso 1: Resultados de PAUM_OBSERVATIONS (texto libre)
            if 'Medicacion' in first_item or 'PAUM_OBSERVATIONS' in first_item:
                for i, item in enumerate(data, 1):
                    # Obtener el campo de medicación
                    med_text = (
                        item.get('Medicacion') or 
                        item.get('PAUM_OBSERVATIONS') or
                        item.get('Observaciones') or
                        'Sin información'
                    )
                    
                    # Si hay información adicional del ingrediente activo
                    active_ingredient = item.get('ACIN_DESCRIPTION_ES', '')
                    
                    if active_ingredient and active_ingredient != med_text:
                        formatted.append(f"   {i}. **{active_ingredient}**")
                        formatted.append(f"      Información adicional: {med_text}")
                    else:
                        formatted.append(f"   {i}. **{med_text}**")
                    
                    formatted.append("")
            
            # Caso 2: Conteo de medicamentos por paciente
            elif 'medication_name' in first_item and 'patient_count' in first_item:
                for i, item in enumerate(data, 1):
                    medication_name = item.get('medication_name', 'N/A')
                    patient_count = item.get('patient_count', 0)
                    formatted.append(f"   {i}. **{medication_name}** - {patient_count} pacientes")
                formatted.append("")
            
            # Caso 3: Formato genérico
            else:
                # Llamada al LLM para formateo genérico si está disponible
                if hasattr(self, 'llm') and self.llm:
                    try:
                        prompt = f"""
Eres un asistente médico. Formatea de manera clara y profesional la siguiente lista de resultados de medicamentos para mostrar a un usuario clínico. Incluye nombre del medicamento, dosis, frecuencia y cualquier información relevante. Si hay varios pacientes, agrupa por paciente.

Resultados:
{data}
"""
                        # Llamada síncrona o asíncrona según el LLM
                        if hasattr(self.llm, 'invoke'):
                            llm_response = self.llm.invoke(prompt)
                            formatted.append(str(llm_response))
                        elif hasattr(self.llm, 'ainvoke'):
                            import asyncio
                            llm_response = asyncio.run(self.llm.ainvoke(prompt))
                            formatted.append(str(llm_response))
                        else:
                            # Fallback a formato clásico
                            for i, item in enumerate(data, 1):
                                item_str = " | ".join([f"{k}: {v}" for k, v in item.items() if v])
                                formatted.append(f"   {i}. {item_str}")
                    except Exception as e:
                        formatted.append(f"❌ Error LLM: {e}")
                else:
                    for i, item in enumerate(data, 1):
                        # Buscar cualquier campo que parezca contener información de medicamento
                        med_info = None
                        for key in ['medication', 'medicamento', 'drug', 'medicine', 'treatment']:
                            if key in item or key.upper() in item:
                                med_info = item.get(key) or item.get(key.upper())
                                break
                        if med_info:
                            formatted.append(f"   {i}. **{med_info}**")
                        else:
                            # Mostrar todos los campos disponibles
                            item_str = " | ".join([f"{k}: {v}" for k, v in item.items() if v])
                            formatted.append(f"   {i}. {item_str}")
                        formatted.append("")
        
        # Agregar información del paciente si está disponible
        if data and any('PATI_' in str(key) for key in data[0].keys()):
            patient_info = []
            first_item = data[0]
            
            # Extraer información del paciente
            patient_name = ""
            if 'PATI_NAME' in first_item and 'PATI_SURNAME_1' in first_item:
                name = f"{first_item.get('PATI_NAME', '')} {first_item.get('PATI_SURNAME_1', '')}".strip()
                if name:
                    patient_name = name
            elif 'PATI_FULL_NAME' in first_item:
                patient_name = first_item.get('PATI_FULL_NAME', '')
            
            if patient_name:
                formatted.insert(2, f"👤 **Paciente**: {patient_name}")
                formatted.insert(3, "")
        
        return "\n".join(formatted)
    
    def _format_diagnosis_results(self, data: List[Dict[str, Any]]) -> str:
        """Formatea resultados de diagnósticos"""
        if not data:
            return "🩺 **No se encontraron diagnósticos**"
        
        formatted = []
        formatted.append("🩺 **DIAGNÓSTICOS ENCONTRADOS**")
        formatted.append("")
        
        for i, item in enumerate(data, 1):
            diagnosis = item.get('diagnosis', 'N/A')
            patient_count = item.get('patient_count', 0)
            formatted.append(f"   {i}. **{diagnosis}** - {patient_count} pacientes")
        
        return "\n".join(formatted)
    
    async def _format_generic_results(self, data: List[Dict[str, Any]]) -> str:
        """Formatea resultados genéricos de forma inteligente"""
        if not data:
            return "📊 **No se encontraron resultados**"
        
        # Detectar si es una consulta de historia clínica
        if await self._is_clinical_history_query(data):
            return await self._format_clinical_history_results(data)
        
        formatted = []
        formatted.append("📊 **RESULTADOS ENCONTRADOS**")
        formatted.append("")
        
        # Mostrar solo los primeros 10 resultados para evitar sobrecarga
        display_data = data[:10]
        
        for i, item in enumerate(display_data, 1):
            if isinstance(item, dict):
                item_str = " | ".join([f"{k}: {v}" for k, v in item.items()])
                formatted.append(f"   {i}. {item_str}")
            else:
                formatted.append(f"   {i}. {item}")
        
        if len(data) > 10:
            formatted.append(f"")
            formatted.append(f"📄 *... y {len(data) - 10} resultados más*")
        
        return "\n".join(formatted)
    
    async def _is_clinical_history_query(self, data: List[Dict[str, Any]]) -> bool:
        """Detecta si los datos corresponden a una consulta de historia clínica usando LLM"""
        if not data or not isinstance(data[0], dict):
            return False
        
        try:
            if not self.llm:
                # Fallback básico sin LLM
                first_item = data[0]
                clinical_fields = [
                    'Historia_Clinica', 'PATI_CLINICAL_HISTORY_ID', 'PATI_NAME', 
                    'PATI_SURNAME_1', 'PATI_BIRTH_DATE', 'Medicacion_Usual',
                    'Diagnosticos', 'Alergias'
                ]
                return any(field in first_item for field in clinical_fields)
            
            # Usar LLM para análisis inteligente
            prompt = f"""
Analiza estos datos médicos y determina si corresponden a una HISTORIA CLÍNICA completa de un paciente.

DATOS A ANALIZAR:
{json.dumps(data[:3], indent=2, ensure_ascii=False)}  # Primeros 3 registros para análisis

INSTRUCCIONES:
1. Busca información típica de historia clínica:
   - Datos del paciente (nombre, apellidos, fecha nacimiento)
   - Diagnósticos médicos
   - Medicamentos prescritos
   - Signos vitales
   - Alergias
   - Antecedentes médicos
   - Observaciones clínicas

2. Determina si los datos contienen información médica completa y estructurada

3. Considera que una historia clínica típicamente incluye:
   - Información personal del paciente
   - Múltiples registros médicos
   - Diferentes tipos de información clínica

Responde SOLO con "SI" si es historia clínica, o "NO" si no lo es.
"""
            
            response = _call_openai_native(self.llm, prompt, temperature=0.1)
            result = response.content.strip().upper()
            
            is_clinical_history = "SI" in result or "YES" in result
            print(f"   🧠 LLM detectó historia clínica: {'SÍ' if is_clinical_history else 'NO'}")
            
            return is_clinical_history
            
        except Exception as e:
            print(f"   ⚠️ Error analizando historia clínica con LLM: {e}")
            # Fallback básico
            first_item = data[0]
            clinical_fields = [
                'Historia_Clinica', 'PATI_CLINICAL_HISTORY_ID', 'PATI_NAME', 
                'PATI_SURNAME_1', 'PATI_BIRTH_DATE', 'Medicacion_Usual',
                'Diagnosticos', 'Alergias'
            ]
            return any(field in first_item for field in clinical_fields)
    
    async def _format_clinical_history_results(self, data: List[Dict[str, Any]]) -> str:
        """Formatea resultados de historia clínica usando LLM para clasificación inteligente"""
        if not data:
            return "📋 **No se encontró información de historia clínica**"
        
        try:
            if not self.llm:
                return await self._format_clinical_history_basic(data)
            
            # Usar LLM para clasificación inteligente
            prompt = f"""
Analiza estos datos médicos y clasifícalos inteligentemente para una historia clínica.

DATOS MÉDICOS:
{json.dumps(data[:5], indent=2, ensure_ascii=False)}  # Primeros 5 registros

INSTRUCCIONES:
1. Clasifica cada registro médico en las siguientes categorías:
   - DIAGNOSTICOS: Enfermedades, condiciones médicas, diagnósticos
   - SIGNOS_VITALES: Tensión, frecuencia cardíaca, temperatura, peso, talla, etc.
   - MEDICAMENTOS: Medicinas, fármacos, dosis, tratamientos
   - OTROS: Información clínica que no encaja en las categorías anteriores

2. Extrae información del paciente (nombre, apellidos)

3. Organiza la información de forma clara y profesional

Responde SOLO con JSON en este formato:
```json
{{
  "paciente": {{
    "nombre": "nombre completo del paciente"
  }},
  "diagnosticos": ["diagnóstico 1", "diagnóstico 2"],
  "signos_vitales": ["signo vital 1", "signo vital 2"],
  "medicamentos": ["medicamento 1", "medicamento 2"],
  "otros": ["otra información 1", "otra información 2"]
}}
```
"""
            
            response = _call_openai_native(self.llm, prompt, temperature=0.1)
            result = self._try_parse_llm_json(response.content)
            
            if result:
                return await self._format_clinical_history_with_llm(result)
            else:
                return await self._format_clinical_history_basic(data)
                
        except Exception as e:
            print(f"   ⚠️ Error en formateo inteligente de historia clínica: {e}")
            return await self._format_clinical_history_basic(data)
        
        # Construir respuesta formateada usando LLM
        return await self._format_clinical_history_with_llm(result)
    
    async def _format_clinical_history_basic(self, data: List[Dict[str, Any]]) -> str:
        """Formateo básico de historia clínica sin LLM (fallback)"""
        if not data:
            return "📋 **No se encontró información de historia clínica**"
        
        # Agrupar información por tipo usando patrones básicos
        patient_info = {}
        diagnoses = []
        vitals = []
        medications = []
        other_info = []
        
        for record in data:
            # Extraer información del paciente
            if 'PATI_NAME' in record and 'PATI_SURNAME_1' in record:
                patient_info['name'] = f"{record['PATI_NAME']} {record['PATI_SURNAME_1']}"
                if 'PATI_FULL_NAME' in record and record['PATI_FULL_NAME']:
                    patient_info['full_name'] = record['PATI_FULL_NAME']
            
            # Clasificar la información médica con patrones básicos
            condition = record.get('Condicion', '') or record.get('DIAG_OBSERVATION', '') or record.get('DIAG_OTHER_DIAGNOSTIC', '')
            
            if condition:
                condition_lower = condition.lower()
                
                # Diagnósticos principales
                if any(keyword in condition_lower for keyword in ['diabetes', 'hipertensión', 'hipertension', 'cáncer', 'cancer', 'tumor', 'apendicectomía', 'apendicectomia', 'mellitus']):
                    diagnoses.append(condition)
                
                # Signos vitales
                elif any(keyword in condition_lower for keyword in ['tensión', 'tension', 'ta:', 'fc:', 'fr:', 'temperatura', 'peso', 'talla', 'imc', 'lpm', 'rpm', 'mmhg']):
                    vitals.append(condition)
                
                # Medicamentos
                elif any(keyword in condition_lower for keyword in ['mg', 'ml', 'tableta', 'cápsula', 'capsula', 'inyección', 'inyeccion']):
                    medications.append(condition)
                
                # Otros
                else:
                    other_info.append(condition)
        
        # Construir respuesta formateada
        formatted = []
        formatted.append("🏥 **HISTORIA CLÍNICA DEL PACIENTE**")
        formatted.append("=" * 60)
        formatted.append("")
        
        if patient_info:
            formatted.append(f"👤 **PACIENTE:** {patient_info.get('full_name', patient_info.get('name', 'N/A'))}")
            formatted.append("")
        
        # Diagnósticos principales
        if diagnoses:
            formatted.append("🔍 **DIAGNÓSTICOS PRINCIPALES:**")
            for i, diagnosis in enumerate(set(diagnoses), 1):
                formatted.append(f"   {i}. {diagnosis}")
            formatted.append("")
        
        # Signos vitales
        if vitals:
            formatted.append("💓 **SIGNOS VITALES:**")
            for vital in set(vitals):
                formatted.append(f"   • {vital}")
            formatted.append("")
        
        # Medicamentos
        if medications:
            formatted.append("💊 **MEDICACIÓN:**")
            for med in set(medications):
                formatted.append(f"   • {med}")
            formatted.append("")
        
        # Otra información
        if other_info:
            formatted.append("📝 **OTRA INFORMACIÓN CLÍNICA:**")
            for info in set(other_info):
                formatted.append(f"   • {info}")
            formatted.append("")
        
        formatted.append("=" * 60)
        formatted.append(f"📊 Total de registros médicos: {len(data)}")
        
        return "\n".join(formatted)
    
    async def _format_clinical_history_with_llm(self, llm_result: Dict[str, Any]) -> str:
        """Formatea historia clínica usando el resultado del LLM"""
        formatted = []
        formatted.append("🏥 **HISTORIA CLÍNICA DEL PACIENTE**")
        formatted.append("=" * 60)
        formatted.append("")
        
        # Información del paciente
        paciente = llm_result.get('paciente', {})
        if paciente and paciente.get('nombre'):
            formatted.append(f"👤 **PACIENTE:** {paciente['nombre']}")
            formatted.append("")
        
        # Diagnósticos
        diagnosticos = llm_result.get('diagnosticos', [])
        if diagnosticos:
            formatted.append("🔍 **DIAGNÓSTICOS PRINCIPALES:**")
            for i, diagnosis in enumerate(diagnosticos, 1):
                formatted.append(f"   {i}. {diagnosis}")
            formatted.append("")
        
        # Signos vitales
        signos_vitales = llm_result.get('signos_vitales', [])
        if signos_vitales:
            formatted.append("💓 **SIGNOS VITALES:**")
            for vital in signos_vitales:
                formatted.append(f"   • {vital}")
            formatted.append("")
        
        # Medicamentos
        medicamentos = llm_result.get('medicamentos', [])
        if medicamentos:
            formatted.append("💊 **MEDICACIÓN:**")
            for med in medicamentos:
                formatted.append(f"   • {med}")
            formatted.append("")
        
        # Otra información
        otros = llm_result.get('otros', [])
        if otros:
            formatted.append("📝 **OTRA INFORMACIÓN CLÍNICA:**")
            for info in otros:
                formatted.append(f"   • {info}")
            formatted.append("")
        
        formatted.append("=" * 60)
        formatted.append(f"📊 Total de registros médicos: {len(diagnosticos) + len(signos_vitales) + len(medicamentos) + len(otros)}")
        
        return "\n".join(formatted)
    
    def _normalize_accents_python(self, text: str) -> str:
        """
        Normaliza vocales acentuadas en Python (más eficiente que en SQL).
        
        Args:
            text: Texto a normalizar
            
        Returns:
            str: Texto con vocales acentuadas normalizadas
        """
        replacements = {
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ñ': 'N',
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n'
        }
        
        normalized = text
        for accented, normal in replacements.items():
            normalized = normalized.replace(accented, normal)
        
        return normalized.upper()

    def _extract_table_name(self, sql: str) -> str:
        """Extrae el nombre de la tabla de una consulta SQL"""
        try:
            import re
            
            # Buscar patrones comunes de nombres de tabla
            patterns = [
                r'INSERT\s+INTO\s+(\w+)',  # INSERT INTO tabla
                r'UPDATE\s+(\w+)',          # UPDATE tabla
                r'DELETE\s+FROM\s+(\w+)',   # DELETE FROM tabla
                r'FROM\s+(\w+)',            # FROM tabla
                r'JOIN\s+(\w+)',            # JOIN tabla
            ]
            
            for pattern in patterns:
                match = re.search(pattern, sql, re.IGNORECASE)
                if match:
                    return match.group(1)
            
            return "desconocida"
            
        except Exception as e:
            return "desconocida"
    
    def _determine_table_for_insert(self, data: Dict[str, Any]) -> str:
        """Determina la tabla apropiada para INSERT usando LLM para mapeo completamente dinámico"""
        try:
            if not self.llm:
                return 'PATI_PATIENTS'  # Fallback si no hay LLM
            
            # Obtener esquema completo disponible
            schema_info = self._get_schema_context()
            
            prompt = f"""
Eres un experto en bases de datos médicas. Analiza estos datos y determina la tabla MÁS APROPIADA para insertarlos.

DATOS A INSERTAR:
{json.dumps(data, indent=2, ensure_ascii=False)}

ESQUEMA COMPLETO DISPONIBLE:
{schema_info}

INSTRUCCIONES:
1. Analiza los datos y determina qué tipo de información médica contienen
2. Revisa TODAS las tablas disponibles en el esquema
3. Encuentra la tabla que mejor se ajuste al tipo de datos
4. Considera la funcionalidad y propósito de cada tabla
5. NO uses patrones rígidos, analiza el contenido real de los datos
6. Si hay múltiples opciones, elige la más específica y apropiada

RESPUESTA:
```json
{{
  "tabla_recomendada": "nombre_exacto_de_la_tabla",
  "razon": "explicación detallada de por qué esta tabla es la más apropiada",
  "campos_mapeables": ["campo1", "campo2"],
  "confianza": "alta/media/baja"
}}
```

Responde SOLO con el JSON.
            """
            
            response = _call_openai_native(self.llm, prompt, task_description="Determinando tabla dinámicamente")
            result = self._try_parse_llm_json(response.content)
            
            if result and result.get('tabla_recomendada'):
                tabla = result.get('tabla_recomendada', 'PATI_PATIENTS')
                razon = result.get('razon', 'No especificada')
                confianza = result.get('confianza', 'media')
                
                print(f"   🎯 Tabla determinada dinámicamente: {tabla} (confianza: {confianza})")
                if razon:
                    print(f"   💡 Razón: {razon[:100]}...")
                
                return tabla
            
            # Fallback si LLM falla
            return 'PATI_PATIENTS'
            
        except Exception as e:
            print(f"   ❌ Error determinando tabla dinámicamente: {e}")
            return 'PATI_PATIENTS'
    
    def _get_field_mapping(self, data: Dict[str, Any], table_name: str) -> Dict[str, str]:
        """Obtiene el mapeo de campos de datos a campos de tabla usando LLM para mapeo completamente dinámico"""
        try:
            if not self.llm:
                return {}  # Fallback si no hay LLM
            
            # Obtener información detallada de la tabla específica
            table_info = ""
            if table_name in self.column_metadata:
                columns = [col['name'] for col in self.column_metadata[table_name]['columns']]
                table_info = f"Columnas disponibles en {table_name}: {', '.join(columns)}"
            else:
                table_info = f"Tabla {table_name} no encontrada en el esquema"
            
            prompt = f"""
Eres un experto en bases de datos médicas. Analiza estos datos y mapea los campos a las columnas de la tabla especificada.

DATOS A INSERTAR:
{json.dumps(data, indent=2, ensure_ascii=False)}

TABLA DESTINO: {table_name}

INFORMACIÓN DE LA TABLA:
{table_info}

INSTRUCCIONES:
1. Analiza cada campo de los datos
2. Encuentra la columna más apropiada en la tabla para cada campo
3. Considera el tipo de dato y el significado semántico
4. Mapea campos similares (ej: 'name' → 'PATI_NAME', 'description' → 'APCO_DESCRIPTION_ES')
5. Si no hay columna exacta, usa la más cercana en significado
6. NO uses patrones rígidos, analiza el contenido real

RESPUESTA:
```json
{{
  "mapeo_campos": {{
    "campo_datos": "columna_tabla",
    "campo_datos2": "columna_tabla2"
  }},
  "razon": "explicación del mapeo realizado",
  "campos_sin_mapear": ["campo1", "campo2"],
  "confianza": "alta/media/baja"
}}
```

Responde SOLO con el JSON.
            """
            
            response = _call_openai_native(self.llm, prompt, task_description="Mapeando campos dinámicamente")
            result = self._try_parse_llm_json(response.content)
            
            if result and result.get('mapeo_campos'):
                mapeo = result.get('mapeo_campos', {})
                razon = result.get('razon', 'No especificada')
                confianza = result.get('confianza', 'media')
                campos_sin_mapear = result.get('campos_sin_mapear', [])
                
                print(f"   🎯 Mapeo de campos dinámico: {len(mapeo)} campos mapeados (confianza: {confianza})")
                if razon:
                    print(f"   💡 Razón: {razon[:100]}...")
                if campos_sin_mapear:
                    print(f"   ⚠️ Campos sin mapear: {', '.join(campos_sin_mapear)}")
                
                return mapeo
            
            # Fallback si LLM falla
            return {}
            
        except Exception as e:
            print(f"   ❌ Error mapeando campos dinámicamente: {e}")
            return {}

    def _generate_simple_patient_sql(self, query: str, params: list) -> str:
        """Genera SQL simple para consultas básicas de pacientes"""
        try:
            query_lower = query.lower()
            
            # BÚSQUEDA DE ÚLTIMO PACIENTE (MEJORADO con filtros más estrictos)
            if any(keyword in query_lower for keyword in ['último', 'ultimo', 'última', 'ultima', 'reciente', 'creado']):
                return """
SELECT PATI_ID, PATI_NAME, PATI_SURNAME_1, PATI_FULL_NAME, PATI_CLINICAL_HISTORY_ID, PATI_START_DATE
FROM PATI_PATIENTS 
WHERE PATI_NAME IS NOT NULL 
  AND PATI_NAME != ''
  AND PATI_NAME != 'unknown'
  AND PATI_NAME != 'Unknown'
  AND PATI_NAME != 'UNKNOWN'
  AND PATI_NAME != 'GENERATED_UUID'
  AND PATI_NAME != 'None'
  AND PATI_NAME != 'No especificado'
  AND PATI_NAME != 'PRUEBAS'
  AND PATI_NAME != 'SINA'
  AND PATI_NAME != 'TEST'
  AND PATI_NAME != 'test'
  AND LENGTH(TRIM(PATI_NAME)) > 1
  AND PATI_SURNAME_1 IS NOT NULL
  AND PATI_SURNAME_1 != ''
  AND PATI_SURNAME_1 != 'unknown'
  AND PATI_SURNAME_1 != 'Unknown'
  AND PATI_SURNAME_1 != 'UNKNOWN'
  AND PATI_SURNAME_1 != 'GENERATED_UUID'
  AND PATI_SURNAME_1 != 'None'
  AND PATI_SURNAME_1 != 'No especificado'
  AND PATI_SURNAME_1 != 'PRUEBAS'
  AND PATI_SURNAME_1 != 'SINA'
  AND PATI_SURNAME_1 != 'TEST'
  AND PATI_SURNAME_1 != 'test'
  AND LENGTH(TRIM(PATI_SURNAME_1)) > 1
  AND PATI_FULL_NAME IS NOT NULL
  AND PATI_FULL_NAME != ''
  AND PATI_FULL_NAME NOT LIKE '%unknown%'
  AND PATI_FULL_NAME NOT LIKE '%GENERATED%'
  AND PATI_FULL_NAME NOT LIKE '%PRUEBAS%'
  AND PATI_FULL_NAME NOT LIKE '%SINA%'
  AND PATI_FULL_NAME NOT LIKE '%TEST%'
  AND LENGTH(TRIM(PATI_FULL_NAME)) > 3
  AND PATI_ID IS NOT NULL
  AND PATI_ID != ''
  AND PATI_ID != 'GENERATED_UUID'
ORDER BY 
  CASE 
    WHEN PATI_ID LIKE 'urn:uuid:%' THEN 1  -- Priorizar UUIDs (más recientes)
    ELSE 2
  END,
  PATI_ID DESC 
LIMIT 1;"""
            
            # BÚSQUEDA DE PACIENTES (MEJORADA para nombres completos)
            if any(keyword in query_lower for keyword in ['paciente', 'llamad', 'nombre', 'historial', 'historia']):
                if params and len(params) > 0:
                    param = params[0]
                    if isinstance(param, list) and len(param) > 0:
                        param = param[0]
                    
                    # MEJORADO: Búsqueda más flexible para nombres completos
                    # Dividir el nombre en partes para búsqueda más precisa
                    # Asegurar que param sea una cadena
                    param_str = str(param) if param is not None else ""
                    name_parts = param_str.split()
                    if len(name_parts) >= 2:
                        # Si tiene al menos nombre y apellido
                        first_name = name_parts[0]
                        last_name = name_parts[-1]
                        
                        return f"""
SELECT 
    p.PATI_ID, 
    p.PATI_NAME, 
    p.PATI_SURNAME_1, 
    p.PATI_SURNAME_2,
    p.PATI_FULL_NAME 
FROM PATI_PATIENTS p 
WHERE (
    (LOWER(p.PATI_NAME) LIKE LOWER('%{first_name}%') AND LOWER(p.PATI_SURNAME_1) LIKE LOWER('%{last_name}%'))
    OR LOWER(p.PATI_FULL_NAME) LIKE LOWER('%{param}%')
    OR (LOWER(p.PATI_NAME) LIKE LOWER('%{param}%') OR LOWER(p.PATI_SURNAME_1) LIKE LOWER('%{param}%'))
)
  AND p.PATI_NAME != 'unknown'
  AND p.PATI_NAME IS NOT NULL
  AND p.PATI_NAME != ''
ORDER BY p.PATI_ID 
LIMIT 10;"""
                    else:
                        # Búsqueda simple para nombres cortos
                        return f"""
SELECT 
    p.PATI_ID, 
    p.PATI_NAME, 
    p.PATI_SURNAME_1, 
    p.PATI_FULL_NAME 
FROM PATI_PATIENTS p 
WHERE (LOWER(p.PATI_FULL_NAME) LIKE LOWER('%{param}%')
   OR (LOWER(p.PATI_NAME) LIKE LOWER('%{param}%') OR LOWER(p.PATI_SURNAME_1) LIKE LOWER('%{param}%')))
  AND p.PATI_NAME != 'unknown'
  AND p.PATI_NAME IS NOT NULL
  AND p.PATI_NAME != ''
ORDER BY p.PATI_ID 
LIMIT 10;"""
                else:
                    return """
SELECT 
    p.PATI_ID, 
    p.PATI_NAME, 
    p.PATI_SURNAME_1, 
    p.PATI_FULL_NAME 
FROM PATI_PATIENTS p 
WHERE p.PATI_NAME != 'unknown'
  AND p.PATI_NAME IS NOT NULL
  AND p.PATI_NAME != ''
ORDER BY p.PATI_ID 
LIMIT 10;"""
            
            # SQL genérico por defecto (filtrar valores por defecto)
            return """
SELECT PATI_ID, PATI_NAME, PATI_SURNAME_1, PATI_FULL_NAME
FROM PATI_PATIENTS 
WHERE PATI_NAME != 'unknown' 
  AND PATI_NAME IS NOT NULL 
  AND PATI_NAME != ''
  AND PATI_ID != 'GENERATED_UUID'
  AND PATI_ID IS NOT NULL
ORDER BY PATI_ID 
LIMIT 10;"""
            
        except Exception as e:
            logger.error(f"Error en _generate_simple_patient_sql: {e}")
            return """
SELECT PATI_ID, PATI_NAME, PATI_SURNAME_1, PATI_FULL_NAME
FROM PATI_PATIENTS 
WHERE PATI_NAME != 'unknown' 
  AND PATI_NAME IS NOT NULL
ORDER BY PATI_ID 
LIMIT 10;"""
    
    async def _extract_patient_name_from_query(self, query: str) -> str:
        """Extrae el nombre del paciente de la consulta y busca por similitud si no hay coincidencia exacta"""
        try:
            # Extraer nombre con LLM
            name = await self._extract_name_with_llm(query)
            if not name:
                return ""
            
            # Buscar coincidencia exacta en la base de datos (case-insensitive)
            sql = f"SELECT PATI_FULL_NAME FROM PATI_PATIENTS WHERE LOWER(PATI_FULL_NAME) = LOWER(?) LIMIT 1"
            result = self.sql_executor.execute_query(sql, [name])
            if result and result.get('success') and result.get('data'):
                print(f"   ✅ Coincidencia exacta encontrada: {result['data'][0]['PATI_FULL_NAME']}")
                return result['data'][0]['PATI_FULL_NAME']
            
            # Si no hay coincidencia exacta, buscar por similitud (LIKE case-insensitive)
            name_parts = name.split()
            if len(name_parts) >= 2:
                # Buscar por nombre y primer apellido
                sql_like = f"SELECT PATI_FULL_NAME FROM PATI_PATIENTS WHERE LOWER(PATI_FULL_NAME) LIKE LOWER(?) LIMIT 3"
                search_pattern = f"%{name_parts[0]}%{name_parts[1]}%"
                result_like = self.sql_executor.execute_query(sql_like, [search_pattern])
                if result_like and result_like.get('success') and result_like.get('data'):
                    sugerencias = [row['PATI_FULL_NAME'] for row in result_like['data']]
                    print(f"   ⚠️ No se encontró coincidencia exacta. Sugerencias: {sugerencias}")
                    # Devolver la mejor sugerencia
                    return sugerencias[0]
            
            # Búsqueda más flexible por solo el primer nombre
            if name_parts:
                sql_flexible = f"SELECT PATI_FULL_NAME FROM PATI_PATIENTS WHERE LOWER(PATI_FULL_NAME) LIKE LOWER(?) LIMIT 3"
                result_flexible = self.sql_executor.execute_query(sql_flexible, [f"%{name_parts[0]}%"])
                if result_flexible and result_flexible.get('success') and result_flexible.get('data'):
                    sugerencias = [row['PATI_FULL_NAME'] for row in result_flexible['data']]
                    print(f"   ⚠️ Búsqueda flexible. Sugerencias: {sugerencias}")
                    return sugerencias[0]
            
            print(f"   ⚠️ No se encontró ningún paciente con el nombre: {name}")
            # Si no hay nada, devolver el nombre extraído
            return name
        except Exception as e:
            print(f"   ⚠️ Error en búsqueda flexible de nombre: {e}")
            return ""
    
    async def _basic_name_extraction(self, query: str) -> str:
        """Detección básica de nombres usando LLM como fallback"""
        try:
            if not self.llm:
                return ""
            
            prompt = f"""
Extrae el nombre del paciente de esta consulta médica.

CONSULTA: "{query}"

INSTRUCCIONES:
1. Busca nombres de personas (nombre y apellido)
2. Ignora términos médicos, verbos, o palabras comunes
3. Si no hay nombre específico, responde "NINGUNO"
4. Considera variaciones culturales y nombres con tildes

EJEMPLOS:
- "que medicacion toma jacinto benavente" → "jacinto benavente"
- "buscar paciente Ana García" → "Ana García"
- "cuántos pacientes hay" → "NINGUNO"
- "último paciente registrado" → "NINGUNO"

Responde SOLO con el nombre extraído o "NINGUNO".
"""
            
            try:
                response = _call_openai_native(self.llm, prompt, temperature=0.1)
                extracted_name = response.content.strip().strip('"').strip("'")
                
                if extracted_name.upper() == "NINGUNO":
                    return ""
                
                if len(extracted_name) >= 2:
                    print(f"   🧠 LLM extrajo nombre básico: '{extracted_name}'")
                    return extracted_name
                
                return ""
                
            except Exception as e:
                print(f"   ⚠️ Error en extracción básica con LLM: {e}")
                return ""
            
        except Exception as e:
            print(f"Error en extracción básica de nombre: {e}")
            return ""

    async def _validate_name_with_llm(self, name: str, query: str) -> bool:
        """Usa LLM para validar si un nombre extraído es realmente un nombre de persona"""
        try:
            prompt = f"""
Eres un experto en análisis de texto médico. Valida si el siguiente texto es un nombre de persona.

TEXTO A VALIDAR: "{name}"
CONSULTA COMPLETA: "{query}"

INSTRUCCIONES:
1. Determina si el texto es un nombre de persona (nombre y apellido)
2. Considera variaciones culturales y nombres con tildes
3. Ignora palabras que son términos médicos, verbos, o palabras comunes

EJEMPLOS:
- "Ana García" → VÁLIDO (nombre de persona)
- "Jacinto Benavete" → VÁLIDO (nombre de persona)
- "diabetes" → INVÁLIDO (enfermedad)
- "medicación" → INVÁLIDO (término médico)
- "toma" → INVÁLIDO (verbo)

Responde SOLO con "SI" si es un nombre válido, o "NO" si no lo es.
"""
            
            response = _call_openai_native(self.llm, prompt, task_description="Validando nombre")
            result = response.content.strip().upper()
            
            is_valid = "SI" in result or "YES" in result
            print(f"   🧠 LLM validó '{name}': {'VÁLIDO' if is_valid else 'INVÁLIDO'}")
            return is_valid
            
        except Exception as e:
            print(f"   ⚠️ Error validando nombre con LLM: {e}")
            # Fallback: considerar válido si tiene más de 3 caracteres y no es una palabra común
            return len(name) > 3 and not any(common in name.lower() for common in ['que', 'toma', 'tiene', 'medicacion', 'medicamento', 'paciente', 'diabetes'])

    async def _extract_name_with_llm(self, query: str) -> str:
        """Usa LLM para extraer nombres de personas de una consulta MEJORADO"""
        try:
            prompt = f"""
Eres un experto en análisis de consultas médicas. Tu tarea es extraer ÚNICAMENTE el nombre completo del paciente.

CONSULTA: "{query}"

REGLAS DE EXTRACCIÓN:
1. Extrae SOLO el nombre completo del paciente (nombre y apellido)
2. Ignora palabras como: paciente, buscar, dame, historia, clínica, datos, información, mostrar, encontrar
3. Si no hay nombre específico, responde "NINGUNO"
4. Considera variaciones culturales y nombres con tildes
5. Busca después de ":", "de", "llamado", "llamada"
6. IMPORTANTE: Mantén las minúsculas originales, NO conviertas a mayúsculas

EJEMPLOS MEJORADOS:
✅ "que medicacion toma jacinto benavente" → "jacinto benavente"
✅ "buscar paciente Ana García" → "Ana García"  
✅ "Dame la historia clínica de Lamina Yamala" → "Lamina Yamala"
✅ "Buscar paciente: María Fernández" → "María Fernández"
✅ "Historia clínica de Juan López Martín" → "Juan López Martín"
✅ "datos del paciente Carlos Mendoza" → "Carlos Mendoza"
✅ "hablame de paiente juan carlos pascual" → "juan carlos pascual"
❌ "cuántos pacientes hay" → "NINGUNO"
❌ "último paciente registrado" → "NINGUNO"
❌ "qué medicación toma el paciente" → "NINGUNO"

INSTRUCCIONES ESPECÍFICAS:
- Si ves "Buscar paciente: [NOMBRE]" → extrae [NOMBRE]
- Si ves "historia clínica de [NOMBRE]" → extrae [NOMBRE]  
- Si ves "datos de [NOMBRE]" → extrae [NOMBRE]
- Si ves "medicación de [NOMBRE]" → extrae [NOMBRE]
- Si ves "hablame de paiente [NOMBRE]" → extrae [NOMBRE]
- MANTÉN las minúsculas originales

Responde SOLO con el nombre extraído o "NINGUNO" si no hay nombre específico.
"""
            
            response = _call_openai_native(self.llm, prompt, task_description="Extrayendo nombre mejorado")
            extracted_name = response.content.strip().strip('"').strip("'")
            
            if extracted_name.upper() == "NINGUNO":
                return ""
            
            # Validaciones adicionales
            if not extracted_name:
                return ""
            
            # Eliminar palabras de consulta que puedan haber quedado
            invalid_words = ['buscar', 'paciente', 'historia', 'clínica', 'datos', 'información', 'dame', 'mostrar', 'encontrar', 'paiente']
            name_words = extracted_name.split()
            clean_words = [word for word in name_words if word.lower() not in invalid_words]
            
            if len(clean_words) < 1:
                return ""
            
            clean_name = ' '.join(clean_words)
            
            # Verificar que tenga al menos 2 caracteres y parezca un nombre
            if len(clean_name) >= 2 and not clean_name.lower() in ['el', 'la', 'de', 'del', 'que', 'con', 'por']:
                print(f"   🧠 LLM extrajo nombre mejorado: '{clean_name}'")
                return clean_name
            
            return ""
            
        except Exception as e:
            print(f"   ⚠️ Error extrayendo nombre con LLM: {e}")
            return ""

    async def _generate_flexible_search_sql(self, patient_name: str, search_type: str) -> str:
        """Usa LLM para generar SQL con búsqueda flexible que maneje errores ortográficos"""
        try:
            prompt = f"""
Eres un experto en SQL médico. Genera una consulta SQL flexible para buscar información de un paciente.

NOMBRE DEL PACIENTE: "{patient_name}"
TIPO DE BÚSQUEDA: {search_type}

ESQUEMA DISPONIBLE:
- PATI_PATIENTS: PATI_ID, PATI_NAME, PATI_SURNAME_1, PATI_SURNAME_2, PATI_FULL_NAME
- PATI_USUAL_MEDICATION: PATI_ID, PAUM_OBSERVATIONS
- MEDI_ACTIVE_INGREDIENTS: ACIN_ID, ACIN_DESCRIPTION_ES

REQUISITOS:
1. Maneja errores ortográficos en el nombre
2. Busca en múltiples campos (PATI_NAME, PATI_SURNAME_1, PATI_FULL_NAME)
3. Usa búsqueda flexible con LIKE
4. Para medicamentos, incluye JOIN con PATI_USUAL_MEDICATION
5. Considera variaciones de nombres (con/sin tildes)

EJEMPLO PARA MEDICAMENTOS:
```sql
SELECT 
    p.PATI_NAME, 
    p.PATI_SURNAME_1, 
    um.PAUM_OBSERVATIONS AS Medicacion
FROM PATI_PATIENTS p 
LEFT JOIN PATI_USUAL_MEDICATION um ON p.PATI_ID = um.PATI_ID
WHERE p.PATI_NAME LIKE '%{patient_name.split()[0]}%'
   OR p.PATI_SURNAME_1 LIKE '%{patient_name.split()[-1]}%'
   OR p.PATI_FULL_NAME LIKE '%{patient_name}%'
ORDER BY p.PATI_ID;
```

Genera SOLO el SQL sin explicaciones.
"""
            
            response = _call_openai_native(self.llm, prompt, task_description="Generando SQL flexible")
            sql = response.content.strip()
            
            # Limpiar el SQL de markdown si está presente
            if sql.startswith('```sql'):
                sql = sql[6:]
            if sql.endswith('```'):
                sql = sql[:-3]
            
            print(f"   🧠 LLM generó SQL flexible para '{patient_name}'")
            return sql
            
        except Exception as e:
            print(f"   ⚠️ Error generando SQL flexible: {e}")
            return ""

    async def _llm_interpret_results(self, query: str, data: List[Dict[str, Any]], stream_callback=None) -> str:
        """Interpretación médica experta de los resultados usando LLM"""
        try:
            if not self.llm or not data:
                return f"Se encontraron {len(data)} registros para la consulta '{query}'"
            
            print(f"🩺 Generando interpretación médica de resultados...")
            
            # Preparar muestra de datos para análisis
            sample_data = data[:5] if len(data) > 5 else data
            
            prompt = f"""
Eres un médico experto analizando resultados de una consulta clínica.

CONSULTA ORIGINAL: "{query}"
NÚMERO DE RESULTADOS: {len(data)}
MUESTRA DE DATOS: {json.dumps(sample_data, indent=2)}

COMO MÉDICO EXPERTO, PROPORCIONA:

1. **Interpretación Clínica**: ¿Qué significan estos resultados desde el punto de vista médico?

2. **Relevancia Clínica**: ¿Qué importancia tienen estos hallazgos para la práctica médica?

3. **Consideraciones Adicionales**: ¿Qué otros factores deberían considerarse?

4. **Recomendaciones**: ¿Qué acciones o investigaciones adicionales podrían ser útiles?

INSTRUCCIONES ESPECÍFICAS POR TIPO DE CONSULTA:

**SIGNOS VITALES Y CONSTANTES:**
- Identifica valores normales vs anormales
- Proporciona rangos de referencia cuando sea apropiado
- Destaca valores críticos que requieren atención inmediata
- Organiza por sistemas (cardiovascular, respiratorio, etc.)

**MEDICAMENTOS Y TRATAMIENTOS:**
- Lista medicamentos activos y sus indicaciones
- Identifica posibles interacciones medicamentosas
- Destaca medicamentos críticos o de alto riesgo
- Sugiere monitoreo de efectos secundarios

**DIAGNÓSTICOS Y CONDICIONES:**
- Clasifica diagnósticos por severidad
- Identifica condiciones crónicas vs agudas
- Destaca condiciones que requieren seguimiento
- Sugiere investigaciones adicionales si es necesario

**INFORMACIÓN DEMOGRÁFICA:**
- Proporciona contexto clínico relevante
- Identifica factores de riesgo basados en edad/género
- Sugiere screening apropiado para la edad

FORMATO DE RESPUESTA:
```json
{{
  "interpretacion_clinica": "explicación médica detallada",
  "relevancia_clinica": "importancia para la práctica médica",
  "consideraciones_adicionales": "factores a considerar",
  "recomendaciones": "acciones sugeridas",
  "resumen_ejecutivo": "resumen conciso para presentar al usuario",
  "valores_criticos": ["valor1", "valor2"],
  "sistemas_afectados": ["sistema1", "sistema2"]
}}
```

RESPONDE SOLO CON EL JSON.
            """
            
            response = _call_openai_native(self.llm, prompt, task_description="Interpretación médica de resultados")
            result = self._try_parse_llm_json(response.content)
            
            if result:
                # Formatear interpretación para el usuario con saltos de línea correctos
                interpretation = []
                interpretation.append("🩺 **INTERPRETACIÓN MÉDICA DE RESULTADOS**")
                interpretation.append("")
                
                if result.get('interpretacion_clinica'):
                    interpretation.append("📋 **Interpretación Clínica:**")
                    interpretation.append(result['interpretacion_clinica'])
                    interpretation.append("")
                
                if result.get('relevancia_clinica'):
                    interpretation.append("⚡ **Relevancia Clínica:**")
                    interpretation.append(result['relevancia_clinica'])
                    interpretation.append("")
                
                if result.get('consideraciones_adicionales'):
                    interpretation.append("🔍 **Consideraciones Adicionales:**")
                    interpretation.append(result['consideraciones_adicionales'])
                    interpretation.append("")
                
                if result.get('recomendaciones'):
                    interpretation.append("💡 **Recomendaciones:**")
                    interpretation.append(result['recomendaciones'])
                    interpretation.append("")
                
                return "\n".join(interpretation)
            
            # Fallback si no hay LLM o falla el parsing
            return f"📊 Se encontraron {len(data)} registros para la consulta '{query}'"
            
        except Exception as e:
            print(f"❌ Error en interpretación médica: {e}")
            return f"📊 Se encontraron {len(data)} registros para la consulta '{query}'"

    async def _analyze_medical_relevance(self, query: str, data: List[Dict[str, Any]], stream_callback=None) -> Dict[str, Any]:
        """Análisis de relevancia básico"""
        return {
            'clinical_relevance': 5,
            'completeness': 0.5,
            'urgency_level': 'normal'
        }

    async def _learn_from_query_result(self, query: str, sql: str, count: int, time: float):
        """Sistema de aprendizaje de consultas exitosas para mejorar futuras consultas"""
        try:
            print(f"   🧠 Guardando información para mejorar futuras consultas...")
            
            if not self.llm:
                return
            
            # Analizar la consulta exitosa
            success_analysis = await self._analyze_successful_query(query, sql, count, time)
            
            # Guardar en la base de conocimiento
            await self._save_success_learning(query, sql, count, time, success_analysis)
            
            # Actualizar patrones exitosos
            await self._update_success_patterns(query, sql, success_analysis)
            
            print(f"   ✅ Información guardada para futuras mejoras")
            
        except Exception as e:
            print(f"   ❌ Error guardando información: {e}")

    async def _analyze_successful_query(self, query: str, sql: str, count: int, time: float) -> Dict[str, Any]:
        """Analiza consultas exitosas para extraer patrones útiles"""
        try:
            prompt = f"""
Eres un experto en análisis de consultas SQL médicas exitosas. Analiza esta consulta para aprender patrones.

CONSULTA ORIGINAL: "{query}"
SQL GENERADO: {sql}
RESULTADOS ENCONTRADOS: {count}
TIEMPO DE EJECUCIÓN: {time:.2f} segundos

ANALIZA Y EXTRAE:

1. **Tipo de Consulta**: ¿Qué tipo de consulta médica es? (conteo, búsqueda, análisis, etc.)

2. **Patrón SQL**: ¿Qué patrón SQL se usó? (JOINs, agregaciones, filtros, etc.)

3. **Conceptos Médicos**: ¿Qué conceptos médicos se consultaron?

4. **Eficiencia**: ¿La consulta fue eficiente? (basado en tiempo y resultados)

5. **Reutilización**: ¿Este patrón puede reutilizarse para consultas similares?

FORMATO DE RESPUESTA:
```json
{{
  "tipo_consulta": "clasificación de la consulta",
  "patron_sql": "patrón SQL utilizado",
  "conceptos_medicos": ["concepto1", "concepto2"],
  "eficiencia": "alta/media/baja",
  "reutilizable": true/false,
  "palabras_clave": ["palabra1", "palabra2"],
  "tablas_utilizadas": ["tabla1", "tabla2"],
  "joins_pattern": "patrón de joins usado"
}}
```

RESPONDE SOLO CON EL JSON.
            """
            
            response = _call_openai_native(self.llm, prompt, task_description="Análisis de consulta exitosa")
            result = self._try_parse_llm_json(response.content)
            
            if result:
                tipo_consulta = result.get('tipo_consulta', 'No clasificado')
                eficiencia = result.get('eficiencia', 'No evaluada')
                reutilizable = result.get('reutilizable', 'No determinado')
                
                print(f"   📊 Tipo de consulta: {tipo_consulta}")
                print(f"   ⚡ Eficiencia: {eficiencia}")
                print(f"   🔄 Reutilizable: {reutilizable}")
                return result
            
            return {}
            
        except Exception as e:
            print(f"❌ Error analizando consulta exitosa: {e}")
            return {}

    async def _save_success_learning(self, query: str, sql: str, count: int, time: float, analysis: Dict[str, Any]):
        """Guarda el aprendizaje de consultas exitosas en cache persistente"""
        try:
            # Crear directorio de cache si no existe
            import os
            cache_dir = "cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            # Cargar éxitos existentes
            success_file = os.path.join(cache_dir, "successful_queries.json")
            successful_queries = []
            
            if os.path.exists(success_file):
                try:
                    with open(success_file, 'r', encoding='utf-8') as f:
                        successful_queries = json.load(f)
                except:
                    successful_queries = []
            
            # Agregar nueva consulta exitosa
            success_entry = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'sql': sql,
                'result_count': count,
                'execution_time': time,
                'analysis': analysis,
                'query_hash': hash(query.lower().strip())
            }
            
            successful_queries.append(success_entry)
            
            # Mantener solo las últimas 200 consultas exitosas
            if len(successful_queries) > 200:
                successful_queries = successful_queries[-200:]
            
            # Guardar
            with open(success_file, 'w', encoding='utf-8') as f:
                json.dump(successful_queries, f, indent=2, ensure_ascii=False)
            
            print(f"   💾 Información guardada en base de conocimiento")
            
        except Exception as e:
            print(f"❌ Error guardando éxito: {e}")

    async def _update_success_patterns(self, query: str, sql: str, analysis: Dict[str, Any]):
        """Actualiza patrones de consultas exitosas"""
        try:
            query_type = analysis.get('tipo_consulta', 'general')
            
            # Actualizar patrones en memoria
            if query_type not in self.query_patterns:
                self.query_patterns[query_type] = {
                    'common_errors': [],
                    'successful_patterns': [],
                    'keywords': set()
                }
            
            # Agregar patrón exitoso
            success_pattern = {
                'sql_pattern': analysis.get('patron_sql', ''),
                'joins_pattern': analysis.get('joins_pattern', ''),
                'tablas_utilizadas': analysis.get('tablas_utilizadas', []),
                'eficiencia': analysis.get('eficiencia', 'media'),
                'reutilizable': analysis.get('reutilizable', False),
                'count': 1
            }
            
            # Buscar si ya existe este patrón
            existing_pattern = None
            for pattern in self.query_patterns[query_type]['successful_patterns']:
                if pattern['sql_pattern'] == success_pattern['sql_pattern']:
                    existing_pattern = pattern
                    break
            
            if existing_pattern:
                existing_pattern['count'] += 1
            else:
                self.query_patterns[query_type]['successful_patterns'].append(success_pattern)
            
            # Actualizar keywords
            keywords = analysis.get('palabras_clave', [])
            self.query_patterns[query_type]['keywords'].update(keywords)
            
            print(f"   📈 Patrones actualizados para consultas similares")
            
        except Exception as e:
            print(f"❌ Error actualizando patrones de éxito: {e}")

    async def _update_learned_patterns(self, query: str, sql: str, result: Dict[str, Any]):
        """Actualización de patrones básica"""
        pass

    async def _learn_from_error(self, query: str, error: str):
        """Sistema de aprendizaje de errores para mejorar futuras consultas"""
        try:
            print(f"   🧠 Analizando error para evitar problemas futuros...")
            
            if not self.llm:
                return
            
            # Analizar el error y extraer patrones
            error_analysis = await self._analyze_error_pattern(query, error)
            
            # Guardar en la base de conocimiento
            await self._save_error_learning(query, error, error_analysis)
            
            # Actualizar patrones de consulta
            await self._update_query_patterns(query, error, error_analysis)
            
            print(f"   ✅ Información del error guardada para futuras mejoras")
            
        except Exception as e:
            print(f"   ❌ Error analizando problema: {e}")

    async def _analyze_error_pattern(self, query: str, error: str) -> Dict[str, Any]:
        """Analiza patrones de error usando LLM"""
        try:
            prompt = f"""
Eres un experto en análisis de errores SQL médicos. Analiza este error para aprender patrones.

CONSULTA ORIGINAL: "{query}"
ERROR: {error}

ANALIZA Y EXTRAE:

1. **Tipo de Error**: ¿Qué tipo de error es? (campo inexistente, tabla incorrecta, JOIN incorrecto, correlación faltante, etc.)

2. **Causa Raíz**: ¿Cuál es la causa fundamental del error?

3. **Patrón de Consulta**: ¿Qué tipo de consulta médica es? (conteo, búsqueda, análisis, etc.)

4. **Error Lógico**: ¿Hay problemas de lógica SQL? (subconsultas sin correlación, JOINs incorrectos, etc.)

5. **Corrección Sugerida**: ¿Cómo se debería corregir este tipo de error?

6. **Prevención Futura**: ¿Cómo evitar este error en futuras consultas similares?

FORMATO DE RESPUESTA:
```json
{{
  "tipo_error": "clasificación del error",
  "causa_raiz": "causa fundamental",
  "patron_consulta": "tipo de consulta médica",
  "error_logico": "problemas de lógica SQL detectados",
  "correccion_sugerida": "cómo corregir",
  "prevencion_futura": "cómo evitar en el futuro",
  "palabras_clave": ["palabra1", "palabra2"],
  "severidad": "alta/media/baja"
}}
```

RESPONDE SOLO CON EL JSON.
            """
            
            response = _call_openai_native(self.llm, prompt, task_description="Análisis de patrones de error")
            result = self._try_parse_llm_json(response.content)
            
            if result:
                tipo_error = result.get('tipo_error', 'No clasificado')
                causa_raiz = result.get('causa_raiz', 'No identificada')
                
                print(f"   📊 Tipo de error: {tipo_error}")
                print(f"   🔍 Causa raíz: {causa_raiz}")
                return result
            
            return {}
            
        except Exception as e:
            print(f"❌ Error analizando patrón: {e}")
            return {}

    async def _save_error_learning(self, query: str, error: str, analysis: Dict[str, Any]):
        """Guarda el aprendizaje de errores en cache persistente"""
        try:
            # Crear directorio de cache si no existe
            import os
            cache_dir = "cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            # Cargar errores existentes
            errors_file = os.path.join(cache_dir, "learned_errors.json")
            learned_errors = []
            
            if os.path.exists(errors_file):
                try:
                    with open(errors_file, 'r', encoding='utf-8') as f:
                        learned_errors = json.load(f)
                except:
                    learned_errors = []
            
            # Agregar nuevo error
            error_entry = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'error': error,
                'analysis': analysis,
                'query_hash': hash(query.lower().strip())
            }
            
            learned_errors.append(error_entry)
            
            # Mantener solo los últimos 100 errores
            if len(learned_errors) > 100:
                learned_errors = learned_errors[-100:]
            
            # Guardar
            with open(errors_file, 'w', encoding='utf-8') as f:
                json.dump(learned_errors, f, indent=2, ensure_ascii=False)
            
            print(f"   💾 Información del error guardada")
            
        except Exception as e:
            print(f"❌ Error guardando aprendizaje: {e}")

    async def _update_query_patterns(self, query: str, error: str, analysis: Dict[str, Any]):
        """Actualiza patrones de consulta basado en errores"""
        try:
            # Extraer palabras clave de la consulta
            keywords = analysis.get('palabras_clave', [])
            query_type = analysis.get('patron_consulta', 'general')
            
            # Actualizar patrones en memoria
            if query_type not in self.query_patterns:
                self.query_patterns[query_type] = {
                    'common_errors': [],
                    'successful_patterns': [],
                    'keywords': set()
                }
            
            # Agregar error común
            error_pattern = {
                'error_type': analysis.get('tipo_error', 'unknown'),
                'prevention': analysis.get('prevencion_futura', ''),
                'count': 1
            }
            
            # Buscar si ya existe este tipo de error
            existing_error = None
            for err in self.query_patterns[query_type]['common_errors']:
                if err['error_type'] == error_pattern['error_type']:
                    existing_error = err
                    break
            
            if existing_error:
                existing_error['count'] += 1
            else:
                self.query_patterns[query_type]['common_errors'].append(error_pattern)
            
            # Actualizar keywords
            self.query_patterns[query_type]['keywords'].update(keywords)
            
            print(f"   📈 Patrones actualizados para tipo: {query_type}")
            
        except Exception as e:
            print(f"❌ Error actualizando patrones: {e}")

    async def _detect_last_patient_query_with_llm(self, query: str) -> bool:
        """Detección inteligente de consultas de último paciente usando LLM con múltiples prompts"""
        try:
            if not self.llm:
                return self._basic_last_patient_detection(query)
            
            print("🔍 Analizando consulta para detectar si es sobre último paciente...")
            
            # Prompt 1: Análisis semántico
            semantic_prompt = f"""
            Analiza la siguiente consulta médica desde una perspectiva semántica:
            
            CONSULTA: "{query}"
            
            ANÁLISIS REQUERIDO:
            1. Identificar el tipo de consulta (búsqueda, conteo, información específica)
            2. Detectar referencias temporales (último, reciente, nuevo, etc.)
            3. Identificar el sujeto de la consulta (paciente, diagnóstico, medicación, etc.)
            4. Determinar si hay indicadores de "último" o "más reciente"
            
            Responde con un análisis estructurado de los elementos semánticos encontrados.
            """
            
            # Prompt 2: Clasificación de patrones
            pattern_prompt = f"""
            Basándote en el análisis semántico, clasifica la consulta según estos patrones:
            
            PATRONES DE ÚLTIMO PACIENTE:
            - "último paciente creado/registrado"
            - "paciente más reciente"
            - "último usuario añadido"
            - "paciente más nuevo"
            - "quién es el último"
            - "como se llama el ultimo paciente"
            - "nombre del último paciente"
            - "último paciente que se creó"
            
            PATRONES DE OTROS TIPOS:
            - Búsquedas por criterios específicos
            - Consultas de conteo o estadísticas
            - Búsquedas por nombre o ID específico
            - Consultas de diagnósticos o medicaciones
            
            Clasifica la consulta como "ÚLTIMO_PACIENTE" o "OTRO_TIPO".
            """
            
            # Prompt 3: Validación final
            validation_prompt = f"""
            Valida la clasificación anterior para la consulta:
            
            CONSULTA: "{query}"
            CLASIFICACIÓN: {{CLASSIFICATION_PLACEHOLDER}}
            
            CRITERIOS DE VALIDACIÓN:
            1. ✅ ¿Contiene palabras clave de "último" o "reciente"?
            2. ✅ ¿Se refiere a un paciente específico?
            3. ✅ ¿Busca información de identificación?
            4. ✅ ¿No es una consulta de búsqueda o estadísticas?
            
            Si cumple al menos 3 criterios, es una consulta de último paciente.
            Responde SOLO con "SÍ" o "NO".
            """
            
            try:
                # Paso 1: Análisis semántico
                print("   📋 Paso 1: Análisis semántico...")
                response1 = _call_openai_native(self.llm, semantic_prompt, temperature=0.1)
                semantic_analysis = response1.content.strip()
                print(f"      ✅ Análisis completado: {semantic_analysis[:60]}...")
                
                # Paso 2: Clasificación de patrones
                print("   🎯 Paso 2: Clasificación de patrones...")
                response2 = _call_openai_native(self.llm, pattern_prompt, temperature=0.1)
                classification = response2.content.strip()
                print(f"      ✅ Clasificación: {classification}")
                
                # Paso 3: Validación final
                print("   ✅ Paso 3: Validación final...")
                validation_prompt_with_class = validation_prompt.replace("{CLASSIFICATION_PLACEHOLDER}", classification)
                response3 = _call_openai_native(self.llm, validation_prompt_with_class, temperature=0.1)
                final_result = response3.content.strip().lower()
                
                is_last_patient = final_result in ['sí', 'si', 'yes', 'true', '1']
                print(f"      ✅ Resultado final: {'SÍ' if is_last_patient else 'NO'}")
                print("   🎉 Detección completada con múltiples prompts")
                
                return is_last_patient
                
            except Exception as e:
                print(f"   ⚠️ Error en detección con múltiples prompts: {e}")
                print("   🔄 Usando fallback con prompt único...")
                
                # Fallback con prompt único
                fallback_prompt = f"""
                Analiza la siguiente consulta y determina si se refiere al ÚLTIMO PACIENTE registrado en la base de datos.
                
                CONSULTA: "{query}"
                
                CRITERIOS PARA DETECTAR CONSULTAS DE ÚLTIMO PACIENTE:
                1. Palabras clave: "último", "más reciente", "reciente", "nuevo", "última"
                2. Contexto de pacientes: "paciente", "persona", "usuario"
                3. Referencias temporales: "creado", "registrado", "ingresado", "añadido"
                4. Consultas sobre identificación: "quién es", "cuál es", "dime"
                
                EJEMPLOS POSITIVOS:
                - "como se llama el ultimo paciente creado"
                - "quién es el paciente más reciente"
                - "dime el último paciente registrado"
                - "cuál es el paciente más nuevo"
                
                EJEMPLOS NEGATIVOS:
                - "busca pacientes con diabetes"
                - "cuántos pacientes hay"
                - "pacientes mayores de 50 años"
                
                Responde SOLO con "SÍ" si es una consulta de último paciente, o "NO" si no lo es.
                """
                
                response = _call_openai_native(self.llm, fallback_prompt, temperature=0.1)
                result = response.content.strip().lower()
                
                is_last_patient = result in ['sí', 'si', 'yes', 'true', '1']
                print(f"   🎯 Resultado con fallback: {'SÍ' if is_last_patient else 'NO'}")
                
                return is_last_patient
                
        except Exception as e:
            print(f"❌ Error detectando consulta de último paciente: {e}")
            return self._basic_last_patient_detection(query)

    def _basic_last_patient_detection(self, query: str) -> bool:
        """Detección básica de consultas de último paciente sin LLM"""
        query_lower = query.lower()
        
        # Patrones básicos para último paciente
        last_patient_patterns = [
            'último paciente',
            'último registrado',
            'paciente más reciente',
            'último ingreso',
            'último paciente ingresado',
            'mostrar el último paciente',
            'información del último paciente',
            'último que se registró',
            'cual es el id del ultimo paciente',
            'que patologia tiene',
            'diagnóstico del último paciente',
            'último paciente creado',
            'como se llama el ultimo paciente',
            'nombre del último paciente',
            'último paciente que se creó',
            'paciente más nuevo',
            'último en la base de datos'
        ]
        
        for pattern in last_patient_patterns:
            if pattern in query_lower:
                print(f"   🔍 Detección básica: patrón '{pattern}' encontrado")
                return True
        
        return False

    async def _generate_last_patient_sql_with_llm(self, query: str) -> str:
        """Genera SQL optimizado para consultas de último paciente usando LLM con múltiples prompts"""
        try:
            if not self.llm:
                return self._generate_basic_last_patient_sql()
            
            print("🧠 Generando SQL inteligente para último paciente con múltiples prompts...")
            
            # Prompt 1: Análisis de la consulta y contexto médico
            analysis_prompt = f"""
            Analiza la siguiente consulta médica para encontrar el último paciente:
            
            CONSULTA: "{query}"
            
            CONTEXTO MÉDICO:
            - Necesitamos encontrar el paciente más recientemente registrado
            - Los pacientes más recientes suelen tener PATI_ID con formato UUID
            - Debemos incluir información de diagnósticos y patologías
            - Los datos médicos pueden estar en múltiples tablas relacionadas
            
            ESTRUCTURA DE LA BASE DE DATOS:
            {self._get_schema_context()}
            
            TAREAS:
            1. Identificar las tablas principales para pacientes
            2. Identificar las tablas para diagnósticos y patologías
            3. Determinar las relaciones entre tablas
            4. Identificar campos de calidad de datos para filtros
            
            Responde con un análisis estructurado de las tablas y relaciones relevantes.
            """
            
            # Prompt 2: Generación de estrategia SQL
            strategy_prompt = f"""
            Basándote en el análisis anterior, genera una estrategia SQL para:
            
            OBJETIVO: Encontrar el último paciente con su información médica completa
            
            ESTRATEGIA REQUERIDA:
            1. Seleccionar el paciente más reciente (ordenar por PATI_ID DESC)
            2. Incluir información personal completa (nombre, apellidos, etc.)
            3. Incluir información médica (diagnósticos, patologías)
            4. Filtrar datos de calidad (no nulos, no valores de prueba)
            5. Usar JOINs apropiados para obtener información completa
            6. Priorizar pacientes con UUIDs (más recientes)
            
            ESTRUCTURA SQL ESPERADA:
            - SELECT con campos relevantes del paciente y diagnósticos
            - FROM con tabla principal de pacientes
            - LEFT JOIN con tablas de diagnósticos y episodios
            - WHERE con filtros de calidad de datos
            - ORDER BY para priorizar pacientes recientes
            - LIMIT 1
            
            Genera solo la estrategia SQL, sin implementar el código.
            """
            
            # Prompt 3: Implementación SQL
            implementation_prompt = f"""
            Implementa la siguiente estrategia SQL para encontrar el último paciente:
            
            ESTRATEGIA: {strategy_prompt}
            
            REQUISITOS TÉCNICOS:
            1. Compatibilidad con SQLite
            2. Filtros robustos para datos de calidad
            3. JOINs optimizados para información médica
            4. Ordenamiento inteligente (UUIDs primero)
            5. Manejo de valores nulos y de prueba
            
            FILTROS DE CALIDAD:
            - Excluir valores 'unknown', 'None', 'GENERATED_UUID'
            - Excluir valores de prueba ('PRUEBAS', 'TEST', 'SINA')
            - Verificar longitud mínima de nombres
            - Validar que los campos no estén vacíos
            
            Genera solo el SQL final, sin comentarios ni explicaciones.
            """
            
            # Prompt 4: Validación y optimización
            validation_prompt = f"""
            Valida y optimiza el siguiente SQL para encontrar el último paciente:
            
            SQL A VALIDAR: {{SQL_PLACEHOLDER}}
            
            CRITERIOS DE VALIDACIÓN:
            1. ✅ Sintaxis SQL correcta para SQLite
            2. ✅ Filtros apropiados para datos de calidad
            3. ✅ Ordenamiento correcto (más reciente primero)
            4. ✅ JOINs necesarios para información médica completa
            5. ✅ Campos relevantes seleccionados
            6. ✅ Manejo de valores nulos y de prueba
            
            Si el SQL no cumple estos criterios, genera uno nuevo optimizado.
            Genera solo el SQL final, sin explicaciones.
            """
            
            try:
                # Paso 1: Análisis
                print("   📋 Paso 1: Analizando consulta y contexto...")
                response1 = _call_openai_native(self.llm, analysis_prompt, temperature=0.1)
                analysis = response1.content.strip()
                print(f"      ✅ Análisis completado: {analysis[:80]}...")
                
                # Paso 2: Estrategia
                print("   🎯 Paso 2: Generando estrategia SQL...")
                response2 = _call_openai_native(self.llm, strategy_prompt, temperature=0.1)
                strategy = response2.content.strip()
                print(f"      ✅ Estrategia generada: {strategy[:80]}...")
                
                # Paso 3: Implementación
                print("   🔧 Paso 3: Implementando SQL...")
                response3 = _call_openai_native(self.llm, implementation_prompt, temperature=0.1)
                sql = response3.content.strip()
                sql = sql.replace('```sql', '').replace('```', '').strip()
                print(f"      ✅ SQL implementado: {sql[:80]}...")
                
                # Paso 4: Validación
                print("   ✅ Paso 4: Validando y optimizando...")
                validation_prompt_with_sql = validation_prompt.replace("{SQL_PLACEHOLDER}", sql)
                response4 = _call_openai_native(self.llm, validation_prompt_with_sql, temperature=0.1)
                final_sql = response4.content.strip()
                final_sql = final_sql.replace('```sql', '').replace('```', '').strip()
                
                print(f"      ✅ SQL final optimizado: {final_sql[:80]}...")
                print("   🎉 SQL generado exitosamente con múltiples prompts")
                
                return final_sql
                
            except Exception as e:
                print(f"   ⚠️ Error en generación con múltiples prompts: {e}")
                print("   🔄 Usando fallback con prompt único...")
                
                # Fallback con prompt único
                fallback_prompt = f"""
                Genera una consulta SQL optimizada para obtener información del ÚLTIMO PACIENTE registrado en la base de datos CON SU PATOLOGÍA.

                Consulta original: "{query}"

                ESTRUCTURA DE LA BASE DE DATOS:
                {self._get_schema_context()}

                Requisitos:
                1. Obtener el paciente con PATI_ID más alto (más reciente)
                2. Filtrar solo pacientes con datos válidos (no NULL, no 'unknown', no 'None')
                3. Incluir información básica del paciente Y su patología
                4. Considerar tanto diagnósticos directos como a través de episodios
                5. Ordenar por PATI_ID DESC y limitar a 1 resultado

                IMPORTANTE: 
                - Los pacientes más recientes tienen UUIDs y pueden tener diagnósticos directos sin episodios
                - Usa LEFT JOIN para considerar tanto diagnósticos directos como a través de episodios
                - Explora la estructura real de la base de datos para encontrar las tablas y campos correctos
                - Los pacientes con UUIDs pueden tener diagnósticos en EPIS_DIAGNOSTICS con PATI_ID directo

                Responde SOLO con la consulta SQL, sin explicaciones.
                """
                
                response = _call_openai_native(self.llm, fallback_prompt, temperature=0.1)
                sql = response.content.strip()
                sql = sql.replace('```sql', '').replace('```', '').strip()
                
                print(f"   🎯 SQL generado con fallback: {sql[:100]}...")
                return sql
                
        except Exception as e:
            print(f"❌ Error generando SQL para último paciente: {e}")
            return self._generate_basic_last_patient_sql()

    def _generate_basic_last_patient_sql(self) -> str:
        """Genera SQL básico para último paciente usando LLM con múltiples prompts"""
        try:
            # Prompt 1: Análisis de la consulta y contexto
            analysis_prompt = f"""
            Analiza la siguiente consulta médica y genera un SQL optimizado para encontrar el último paciente creado:
            
            CONSULTA: "como se llama el ultimo paciente creado"
            
            CONTEXTO DE LA BASE DE DATOS:
            - Tabla principal: PATI_PATIENTS
            - Campos disponibles: PATI_ID, PATI_NAME, PATI_SURNAME_1, PATI_SURNAME_2, PATI_FULL_NAME, PATI_BIRTH_DATE, PATI_START_DATE, GEND_ID
            - Tablas relacionadas: EPIS_EPISODES, EPIS_DIAGNOSTICS
            - Los pacientes más recientes suelen tener PATI_ID con formato UUID
            - Debes filtrar pacientes válidos (no nulos, no valores de prueba)
            
            REQUISITOS:
            1. Encontrar el paciente más recientemente creado
            2. Incluir información completa del paciente
            3. Filtrar datos de prueba y valores nulos
            4. Priorizar pacientes con UUIDs (más recientes)
            5. Incluir información de diagnósticos si está disponible
            
            Genera solo el SQL, sin explicaciones adicionales.
            """
            
            # Prompt 2: Generación del SQL principal
            sql_prompt = f"""
            Basándote en el análisis anterior, genera un SQL optimizado que:
            
            1. Seleccione el último paciente creado
            2. Incluya información completa: nombre, apellidos, fecha de nacimiento
            3. Filtre valores de prueba y nulos
            4. Priorice UUIDs sobre otros formatos de ID
            5. Incluya información de diagnósticos si está disponible
            6. Use JOINs apropiados para obtener información completa
            
            ESTRUCTURA ESPERADA:
            - SELECT con campos relevantes
            - FROM con tabla principal
            - LEFT JOIN con tablas relacionadas
            - WHERE con filtros de calidad de datos
            - ORDER BY para priorizar pacientes recientes
            - LIMIT 1
            
            Genera solo el SQL, sin comentarios ni explicaciones.
            """
            
            # Prompt 3: Validación y optimización
            validation_prompt = f"""
            Valida y optimiza el siguiente SQL para encontrar el último paciente:
            
            {self._get_schema_context()}
            
            CRITERIOS DE VALIDACIÓN:
            1. ✅ Sintaxis SQL correcta
            2. ✅ Filtros apropiados para datos de calidad
            3. ✅ Ordenamiento correcto (más reciente primero)
            4. ✅ JOINs necesarios para información completa
            5. ✅ Compatibilidad con SQLite
            
            Si el SQL anterior no cumple estos criterios, genera uno nuevo optimizado.
            Genera solo el SQL final, sin explicaciones.
            """
            
            # Ejecutar prompts en secuencia
            print("🔍 Generando SQL para último paciente con LLM...")
            
            # Paso 1: Análisis
            response1 = _call_openai_native(self.llm, analysis_prompt, temperature=0.1)
            analysis = response1.content.strip()
            print(f"   📋 Análisis completado: {analysis[:100]}...")
            
            # Paso 2: Generación SQL
            response2 = _call_openai_native(self.llm, sql_prompt, temperature=0.1)
            sql = response2.content.strip()
            sql = sql.replace('```sql', '').replace('```', '').strip()
            print(f"   🎯 SQL generado: {sql[:100]}...")
            
            # Paso 3: Validación
            validation_prompt_with_sql = validation_prompt.replace("{SQL_PLACEHOLDER}", sql)
            response3 = _call_openai_native(self.llm, validation_prompt_with_sql, temperature=0.1)
            final_sql = response3.content.strip()
            final_sql = final_sql.replace('```sql', '').replace('```', '').strip()
            
            print(f"   ✅ SQL validado y optimizado: {final_sql[:100]}...")
            
            return final_sql
            
        except Exception as e:
            print(f"❌ Error en generación LLM para último paciente: {e}")
            # Fallback al SQL básico original
            return """
            SELECT DISTINCT 
                p.PATI_ID,
                p.PATI_NAME,
                p.PATI_SURNAME_1,
                p.PATI_SURNAME_2,
                p.PATI_FULL_NAME,
                p.PATI_BIRTH_DATE,
                p.PATI_START_DATE,
                p.GEND_ID,
                d.DIAG_OBSERVATION
            FROM PATI_PATIENTS p
            LEFT JOIN EPIS_EPISODES e ON p.PATI_ID = e.PATI_ID
            LEFT JOIN EPIS_DIAGNOSTICS d ON (p.PATI_ID = d.PATI_ID OR e.EPIS_ID = d.EPIS_ID)
            WHERE p.PATI_NAME IS NOT NULL 
                AND p.PATI_NAME != ''
                AND p.PATI_NAME != 'unknown'
                AND p.PATI_NAME != 'Unknown'
                AND p.PATI_NAME != 'UNKNOWN'
                AND p.PATI_NAME != 'GENERATED_UUID'
                AND p.PATI_NAME != 'None'
                AND p.PATI_NAME != 'No especificado'
                AND p.PATI_NAME != 'PRUEBAS'
                AND p.PATI_NAME != 'SINA'
                AND p.PATI_NAME != 'TEST'
                AND p.PATI_NAME != 'test'
                AND LENGTH(TRIM(p.PATI_NAME)) > 1
                AND p.PATI_SURNAME_1 IS NOT NULL
                AND p.PATI_SURNAME_1 != ''
                AND p.PATI_SURNAME_1 != 'unknown'
                AND p.PATI_SURNAME_1 != 'Unknown'
                AND p.PATI_SURNAME_1 != 'UNKNOWN'
                AND p.PATI_SURNAME_1 != 'GENERATED_UUID'
                AND p.PATI_SURNAME_1 != 'None'
                AND p.PATI_SURNAME_1 != 'No especificado'
                AND p.PATI_SURNAME_1 != 'PRUEBAS'
                AND p.PATI_SURNAME_1 != 'SINA'
                AND p.PATI_SURNAME_1 != 'TEST'
                AND p.PATI_SURNAME_1 != 'test'
                AND LENGTH(TRIM(p.PATI_SURNAME_1)) > 1
                AND p.PATI_FULL_NAME IS NOT NULL
                AND p.PATI_FULL_NAME != ''
                AND p.PATI_FULL_NAME NOT LIKE '%unknown%'
                AND p.PATI_FULL_NAME NOT LIKE '%GENERATED%'
                AND p.PATI_FULL_NAME NOT LIKE '%PRUEBAS%'
                AND p.PATI_FULL_NAME NOT LIKE '%SINA%'
                AND p.PATI_FULL_NAME NOT LIKE '%TEST%'
                AND LENGTH(TRIM(p.PATI_FULL_NAME)) > 3
                AND p.PATI_ID IS NOT NULL
                AND p.PATI_ID != ''
                AND p.PATI_ID != 'GENERATED_UUID'
            ORDER BY 
              CASE 
                WHEN p.PATI_ID LIKE 'urn:uuid:%' THEN 1  -- Priorizar UUIDs (más recientes)
                ELSE 2
              END,
              p.PATI_ID DESC
            LIMIT 1;
            """
    
    async def _validate_pati_id_intelligent(self, data: Dict[str, Any], table: str, operation: str) -> Dict[str, Any]:
        """Validación inteligente de PATI_ID usando LLM para determinar si es requerido y válido"""
        try:
            if not self.llm:
                # Fallback básico si no hay LLM
                pati_id = data.get('PATI_ID') or data.get('pati_id') or data.get('patient_id')
                if pati_id is None or pati_id == 0 or (isinstance(pati_id, str) and (pati_id.strip() == '' or pati_id.strip() == '0')):
                    return {
                        'valid': False,
                        'error': f'PATI_ID inválido: {pati_id}',
                        'requires_pati_id': True
                    }
                return {'valid': True, 'requires_pati_id': True}

            # Obtener esquema de la tabla
            schema_info = self._get_detailed_schema_for_tables([table])
            
            prompt = f"""
Eres un experto en validación de bases de datos médicas. Analiza si esta operación requiere PATI_ID válido.

OPERACIÓN: {operation}
TABLA: {table}
DATOS A INSERTAR: {json.dumps(data, indent=2, ensure_ascii=False)}

ESQUEMA DE LA TABLA:
{schema_info}

INSTRUCCIONES:
1. Determina si esta tabla requiere PATI_ID (clave foránea a PATI_PATIENTS)
2. Valida si el PATI_ID en los datos es válido (no 0, no None, no vacío)
3. Considera el contexto médico de la tabla
4. Analiza si los datos contienen información de paciente

REGLAS MÉDICAS:
- Tablas de datos clínicos (diagnósticos, medicamentos, observaciones) SIEMPRE requieren PATI_ID
- Tablas de configuración o parámetros NO requieren PATI_ID
- PATI_ID debe ser un valor válido (UUID, entero positivo, etc.)

Responde SOLO con este JSON:
{{
  "requires_pati_id": true/false,
  "valid": true/false,
  "error": "descripción del error si no es válido",
  "reasoning": "explicación de la decisión"
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
                'error': f'Error en validación inteligente: {e}',
                'requires_pati_id': True
            }
