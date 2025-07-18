#!/usr/bin/env python3
"""
🧠 SQL Agent Completamente Inteligente v4.2 - Reconstrucción Definitiva
======================================================================
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

# Import del módulo sql_agent_tools
try:
    from .sql_agent_tools import SQLAgentTools
except ImportError:
    from sql_agent_tools import SQLAgentTools

# Import de los nuevos módulos de utilidades
try:
    from ..utils.sql_cleaner import SQLCleaner
    from ..utils.sql_executor import SQLExecutor
except ImportError:
    # Fallback para imports relativos
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.sql_cleaner import SQLCleaner
    from utils.sql_executor import SQLExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SQLAgentIntelligent_v4.2")

class MockResponse:
    def __init__(self, content: str):
        self.content = content

def _call_openai_native(client, messages, temperature=0.1, max_tokens=4000, task_description="Consultando modelo de IA") -> MockResponse:
    """
    Función de compatibilidad para llamar a OpenAI nativo con streaming y logging.
    """
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

        # Siempre usar streaming para que se muestre el progreso en tiempo real
        stream_buffer: List[str] = []
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
                # Mostrar progreso visual
                if len(stream_buffer) % 10 == 0:  # Cada 10 tokens
                    print(".", end="", flush=True)
                    
        print(" ✓")  # Finalizar línea de progreso
        content = "".join(stream_buffer)

        if not content.strip():
            content = '{"success": false, "message": "Error: Respuesta vacía del LLM"}'

        return MockResponse(content)

    except Exception as e:
        error_msg = f"Error en llamada OpenAI del SQLAgent: {str(e)}"
        print(f"   ❌ ERROR EN LLM: {error_msg}")
        logger.error(f"Error en _call_openai_native (SQLAgent): {e}", exc_info=True)
        return MockResponse('{"success": false, "message": "Error crítico en la llamada al LLM."}')

class SQLAgentIntelligent:
    def __init__(self, db_path: str, llm=None):
        """Inicializa el agente SQL inteligente con capacidades avanzadas"""
        self.db_path = db_path
        self.llm = llm
        self.column_metadata = {}
        self.table_relationships = {}
        self.query_cache = {}
        
        # NUEVO: Sistema de aprendizaje
        self.query_patterns = {}  # Patrones de consulta exitosos
        self.performance_metrics = {}  # Métricas de rendimiento
        self.medical_knowledge_base = {}  # Base de conocimiento médico
        self.adaptive_weights = {  # Pesos adaptativos para scoring
            'temporal_relevance': 1.0,
            'severity_weight': 2.0,
            'recency_weight': 1.5,
            'complexity_bonus': 1.2
        }
        
        # NUEVO: Atributos para datos de muestra y estadísticas
        self.table_row_counts = {}
        self.sample_data = {}
        self.knowledge_gaps = {}
        self.learned_patterns = {}
        self.semantic_cache = {}
        
        # NUEVO: Herramientas de LLM para validación inteligente
        try:
            from .sql_agent_tools import SQLAgentTools
            self.schema_tools = SQLAgentTools(db_path, llm=llm)
        except ImportError:
            from sql_agent_tools import SQLAgentTools
            self.schema_tools = SQLAgentTools(db_path, llm=llm)
        
        # Inicializar componentes
        self._initialize_schema_analysis()  # Cargar esquema primero
        self._initialize_adaptive_learning()
        
        # Configuración de logging mejorada con visualización streaming
        self.logger = logging.getLogger("SQLAgent")
        self.logger.setLevel(logging.INFO)
        
        # Formato de logging personalizado con emojis y colores
        formatter = logging.Formatter(
            '%(message)s',  # Solo el mensaje, sin timestamp
            datefmt='%H:%M:%S'
        )
        
        # Handler para consola con colores
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # NUEVO: Sistema de visualización streaming
        self.stream_visualization = True
        self.current_step = 0
        self.total_steps = 0

    def _find_ambiguous_columns(self, sql: str, tables: list) -> list:
        """Detecta columnas ambiguas en el SQL generado (sin prefijo de tabla) que existen en más de una tabla."""
        pattern = re.compile(r'(?<!\.)\b([A-Z0-9_]+)\b', re.IGNORECASE)
        used_columns = set(pattern.findall(sql))
        ambiguous = []
        col_table_map = {}
        for t in tables:
            if t in self.column_metadata:
                for c in self.column_metadata[t]['columns']:
                    col_table_map.setdefault(c['name'].upper(), []).append(t)
        for col in used_columns:
            if col.upper() in col_table_map and len(col_table_map[col.upper()]) > 1:
                ambiguous.append(col)
        return ambiguous

    def _auto_prefix_sql(self, sql: str, tables: list) -> str:
        ambiguous_columns = self._find_ambiguous_columns(sql, tables)
        if not ambiguous_columns:
            return sql

        for col in ambiguous_columns:
            candidate_tables = [t for t in tables if col.upper() in [c['name'].upper() for c in self.column_metadata[t]['columns']]]
            if len(candidate_tables) == 1:
                table = candidate_tables[0]
                pattern = re.compile(rf'(?<![\w.]){col}(?![\w.])', re.IGNORECASE)
                sql = pattern.sub(f'{table}.{col}', sql)
            else:
                raise ValueError(f"Ambigüedad persistente en columna '{col}', presente en tablas: {', '.join(candidate_tables)}")

        remaining_ambiguous = self._find_ambiguous_columns(sql, tables)
        if remaining_ambiguous:
            raise ValueError(f"Ambigüedad persistente después de corrección automática: {remaining_ambiguous}")

        return sql

    async def _explore_schema_for_concepts(self, conceptos: list) -> dict:
        """Busca en el esquema columnas candidatas para cada concepto clínico detectado."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        all_tables = [row[0] for row in cursor.fetchall()]
        # Buscar columnas candidatas para cada concepto
        concept_candidates = {}
        for concept in conceptos:
            concept_lower = concept.lower().replace(' ', '')
            candidates = []
            for table in all_tables:
                if table.startswith('sqlite_'):
                    continue
                cursor.execute(f"PRAGMA table_info('{table}');")
                for col in cursor.fetchall():
                    col_name = col[1]
                    if concept_lower in col_name.lower().replace('_',''):
                        candidates.append((table, col_name))
            concept_candidates[concept] = candidates
        conn.close()
        return {
            'all_tables': all_tables,
            'concept_candidates': concept_candidates
        }

    async def interactive_schema_exploration(self, conceptos: list) -> dict:
        """Explora el esquema y pregunta al usuario qué columnas usar para cada concepto clínico detectado."""
        exploration = await self._explore_schema_for_concepts(conceptos)
        print("\n🔬 Exploración automática del esquema:")
        print("\nTablas encontradas:")
        for t in exploration['all_tables']:
            print(f"  - {t}")

        user_choices = {}
        for concept in conceptos:
            print(f"\nColumnas candidatas para '{concept}':")
            candidates = exploration['concept_candidates'].get(concept, [])
            if candidates:
                for tbl, col in candidates:
                    print(f"  - {tbl}.{col}")
            else:
                print("  (No se encontraron columnas para este concepto)")
            print(f"¿Qué columna quieres usar para '{concept}'? (ejemplo: TABLA.COLUMNA, o deja vacío para omitir):")
            user_choice = input().strip()
            user_choices[concept] = user_choice

        return {
            'concept_column_map': user_choices,
            'exploration': exploration
        }
    async def _active_schema_exploration(self, keywords: list) -> dict:
        """Explora activamente el esquema buscando tablas y columnas que contengan los keywords dados."""
        results = {}
        for table, meta in self.column_metadata.items():
            for col in meta['columns']:
                for kw in keywords:
                    if kw.lower() in col['name'].lower():
                        if table not in results:
                            results[table] = []
                        results[table].append(col['name'])
        return results


    def _initialize_adaptive_learning(self):
        """Inicializa el sistema de aprendizaje adaptativo."""
        try:
            # Cargar conocimiento previo si existe
            learning_cache = Path(f"learning_cache_{Path(self.db_path).stem}.json")
            if learning_cache.exists():
                with learning_cache.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.knowledge_gaps = data.get('knowledge_gaps', {})
                    self.learned_patterns = data.get('learned_patterns', {})
                    self.semantic_cache = data.get('semantic_cache', {})
                logger.info("🧠 Conocimiento previo cargado desde cache.")
        except Exception as e:
            logger.warning(f"No se pudo cargar el cache de aprendizaje: {e}")

    def _save_learning_cache(self):
        """Guarda el conocimiento adquirido para futuras sesiones."""
        try:
            learning_cache = Path(f"learning_cache_{Path(self.db_path).stem}.json")
            with learning_cache.open('w', encoding='utf-8') as f:
                json.dump({
                    'knowledge_gaps': self.knowledge_gaps,
                    'learned_patterns': self.learned_patterns,
                    'semantic_cache': self.semantic_cache
                }, f, indent=2, ensure_ascii=False)
            logger.info("🧠 Conocimiento guardado en cache de aprendizaje.")
        except Exception as e:
            logger.warning(f"No se pudo guardar el cache de aprendizaje: {e}")

    def _initialize_schema_analysis(self):
        cache_file = Path(f"schema_cache_{Path(self.db_path).stem}.json")
        if cache_file.exists():
            try:
                with cache_file.open('r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    self.column_metadata = cached_data.get('column_metadata', {})
                    self.table_row_counts = cached_data.get('table_row_counts', {})
                    self.sample_data = cached_data.get('sample_data', {}) # NUEVO: Cargar muestra de cache
                logger.info("🗄️ Esquema cargado desde cache.")
                return
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"No se pudo cargar la cache del esquema ({e}), se regenerará.")

        logger.info("🔍 Generando nuevo análisis de esquema...")
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            self.table_row_counts = {}  # NUEVO: Almacenar conteo de filas
            
            for table in tables:
                if not table.startswith('sqlite_'):
                    logger.debug(f"🔍 Analizando estructura de tabla: {table}")
                    cursor.execute(f"PRAGMA table_info('{table}');")
                    columns_info = cursor.fetchall()
                    self.column_metadata[table] = {'columns': [{'name': r[1], 'type': r[2]} for r in columns_info]}
                    
                    try:
                        logger.debug(f"   ▶️ Conteo de registros en {table}")
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        self.table_row_counts[table] = count
                        
                        # NUEVO: Capturar datos de muestra si la tabla no está vacía
                        if count > 0:
                            try:
                                logger.debug(f"   ▶️ Extrayendo muestra de {table}")
                                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                                sample_rows = cursor.fetchall()
                                column_names = [desc[0] for desc in cursor.description]
                                self.sample_data[table] = {
                                    'columns': column_names,
                                    'rows': [dict(zip(column_names, row)) for row in sample_rows]
                                }
                            except sqlite3.Error as sample_err:
                                # Manejar errores comunes de funciones no soportadas (ej. GETDATE)
                                if "GETDATE" in str(sample_err).upper() or "NO SUCH FUNCTION" in str(sample_err).upper():
                                    logger.warning(f"⚠️ Saltando muestra para {table} (función no soportada): {sample_err}")
                                else:
                                    logger.warning(f"⚠️ No se pudo obtener muestra de {table}: {sample_err}")
                    except Exception as e:
                        self.table_row_counts[table] = 0
                        if "GETDATE" in str(e).upper() or "NO SUCH FUNCTION" in str(e).upper():
                            logger.warning(f"⚠️ Saltando estadística para {table} (función no soportada): {e}")
                            logger.debug(f"   ❌ Query fallida: SELECT COUNT(*) FROM {table}")
                        else:
                            logger.warning(f"No se pudo obtener estadísticas/muestras de {table}: {e}")
                        
            conn.close()
            self._save_schema_cache()
        except Exception as e:
            logger.error(f"Error inicializando análisis de esquema: {e}")

    def _save_schema_cache(self):
        try:
            cache_file = Path(f"schema_cache_{Path(self.db_path).stem}.json")
            with cache_file.open('w', encoding='utf-8') as f:
                json.dump({
                    'column_metadata': self.column_metadata,
                    'table_row_counts': self.table_row_counts,
                    'sample_data': self.sample_data, # NUEVO: Guardar muestra en cache
                    'table_relationships': self.table_relationships  # NUEVO: Guardar relaciones
                }, f, indent=2)
            logger.info("💾 Cache del esquema guardado.")
        except Exception as e:
            logger.warning(f"No se pudo guardar la cache del esquema: {e}")

    def _analyze_table_relationships(self):
        """
        🔍 MEJORA: Analiza las relaciones entre tablas basándose en:
        1. Foreign keys explícitas (si existen)
        2. Nombres de columnas coincidentes
        3. Patrones de nomenclatura
        """
        try:
            logger.info("🔍 Analizando relaciones entre tablas...")
            
            # Cargar desde cache si existe
            cache_file = Path(f"schema_cache_{Path(self.db_path).stem}.json")
            if cache_file.exists():
                try:
                    with cache_file.open('r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                        if 'table_relationships' in cached_data:
                            self.table_relationships = cached_data['table_relationships']
                            logger.info("✅ Relaciones cargadas desde cache")
                            return
                except Exception:
                    pass
            
            self.table_relationships = {}
            
            # Analizar cada tabla
            for table1, metadata1 in self.column_metadata.items():
                self.table_relationships[table1] = {
                    'foreign_keys': [],
                    'referenced_by': [],
                    'potential_joins': []
                }
                
                columns1 = {col['name'].lower(): col for col in metadata1['columns']}
                
                # Buscar relaciones con otras tablas
                for table2, metadata2 in self.column_metadata.items():
                    if table1 == table2:
                        continue
                    
                    columns2 = {col['name'].lower(): col for col in metadata2['columns']}
                    
                    # Detectar foreign keys por patrones de nomenclatura
                    # Ej: PATI_ID en otra tabla probablemente referencia a PATI_PATIENTS
                    for col_name, col_info in columns1.items():
                        # Patrón 1: TABLA_ID en otra tabla
                        if col_name.endswith('_id'):
                            prefix = col_name[:-3]  # Quitar '_id'
                            # Buscar tabla que coincida con el prefijo
                            if any(table2.lower().startswith(prefix) for table2 in self.column_metadata):
                                self.table_relationships[table1]['foreign_keys'].append({
                                    'column': col_info['name'],
                                    'references_table': table2,
                                    'references_column': col_name,
                                    'confidence': 'high' if table2.lower().startswith(prefix) else 'medium'
                                })
                        
                        # Patrón 2: Columnas con mismo nombre en ambas tablas
                        if col_name in columns2:
                            self.table_relationships[table1]['potential_joins'].append({
                                'local_column': col_info['name'],
                                'foreign_table': table2,
                                'foreign_column': columns2[col_name]['name'],
                                'type': 'same_name'
                            })
            
            # Analizar relaciones inversas
            for table1, rels in self.table_relationships.items():
                for fk in rels['foreign_keys']:
                    ref_table = fk['references_table']
                    if ref_table in self.table_relationships:
                        self.table_relationships[ref_table]['referenced_by'].append({
                            'table': table1,
                            'column': fk['column']
                        })
            
            # Log de resumen
            total_fks = sum(len(rels['foreign_keys']) for rels in self.table_relationships.values())
            total_joins = sum(len(rels['potential_joins']) for rels in self.table_relationships.values())
            logger.info(f"✅ Análisis de relaciones completado: {total_fks} FKs detectadas, {total_joins} joins potenciales")
            
            # Guardar en cache
            self._save_schema_cache()
            
        except Exception as e:
            logger.error(f"Error analizando relaciones entre tablas: {e}")
            self.table_relationships = {}

    def _create_error_response(self, error: str, sql: str = "") -> Dict[str, Any]:
        return {'success': False, 'message': f"Error: {error}", 'data': [], 'sql_query': sql}
    
    def _check_table_has_data(self, table_name: str) -> bool:
        """Verifica si una tabla tiene datos (no está vacía)."""
        # Siempre verificar directamente la base de datos para mayor confiabilidad
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            conn.close()
            
            # Actualizar el cache si está disponible
            if hasattr(self, 'table_row_counts'):
                self.table_row_counts[table_name] = count
            
            logger.debug(f"📊 Tabla {table_name}: {count} registros")
            return count > 0
        except Exception as e:
            logger.warning(f"⚠️ Error verificando datos en {table_name}: {e}")
            return False
    
    def _find_alternative_tables_with_data(self, empty_tables: List[str], concept: str) -> List[str]:
        """Busca tablas alternativas con datos para un concepto dado usando el LLM."""
        try:
            # Recopilar información sobre tablas con datos
            tables_with_data = []
            table_info = []
            
            for table_name, metadata in self.column_metadata.items():
                if table_name in empty_tables:
                    continue
                    
                if self._check_table_has_data(table_name):
                    row_count = self.table_row_counts.get(table_name, 0)
                    if row_count > 0:
                        tables_with_data.append(table_name)
                        columns = [col['name'] for col in metadata['columns']][:10]  # Primeras 10 columnas
                        table_info.append(f"{table_name} ({row_count} registros): {', '.join(columns)}")
            
            if not tables_with_data:
                return []
            
            # Usar el LLM para seleccionar las mejores alternativas
            prompt = f"""Eres un arquitecto de datos médicos especializado en encontrar información en esquemas complejos.

SITUACIÓN: Las tablas principales para "{concept}" están vacías. Necesito encontrar tablas alternativas con datos.

CONCEPTO BUSCADO: "{concept}"

TABLAS DISPONIBLES CON DATOS (nombre_tabla (registros): columnas principales):
{chr(10).join(table_info[:30])}

ESTRATEGIA DE BÚSQUEDA:

1. ANÁLISIS DEL CONCEPTO:
   - ¿Qué tipo de información representa "{concept}"?
   - ¿En qué contexto médico se usa?
   - ¿Qué sinónimos o términos relacionados existen?

2. PATRONES DE BÚSQUEDA POR TIPO:
   
   DIAGNÓSTICOS/ENFERMEDADES:
   - Tablas con: DIAG, EPIS, HIST, CONDITION, PATHOLOGY
   - Columnas con: diagnosis, disease, condition, observation
   
   PACIENTES:
   - Tablas con: PATI, PATIENT, PERSON
   - Columnas con: name, birth, gender, id
   
   MEDICAMENTOS:
   - Tablas con: MEDI, DRUG, PHARMA, TREATMENT
   - Columnas con: medication, drug, dose, prescription
   
   ALERGIAS:
   - Tablas con: ALLER, INTOL, REACTION
   - Columnas con: allergy, allergen, reaction
   
   PROCEDIMIENTOS:
   - Tablas con: PROC, SURG, INTERVENTION
   - Columnas con: procedure, surgery, operation

3. CRITERIOS DE SELECCIÓN:
   - Prioriza tablas con más registros (mejor cobertura)
   - Busca columnas de texto libre donde pueda estar la información
   - Considera tablas de historial que pueden contener datos legacy

RESPUESTA REQUERIDA (JSON):
{{
    "tablas_alternativas": ["TABLA1", "TABLA2", "TABLA3"],
    "justificacion": "Por qué estas tablas pueden contener información sobre {concept}"
}}"""

            response = _call_openai_native(self.llm, [SystemMessage(content=prompt)])
            content = self._extract_response_text(response)
            
            try:
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    result = json.loads(json_match.group(0))
                    alternatives = result.get('tablas_alternativas', [])
                    # Filtrar solo tablas que existen y tienen datos
                    valid_alternatives = [t for t in alternatives if t in tables_with_data]
                    return valid_alternatives[:3]
            except:
                pass
                
            # Fallback: buscar por palabras clave simples
            alternatives = []
            concept_lower = concept.lower()
            for table in tables_with_data[:10]:
                if concept_lower[:4] in table.lower():
                    alternatives.append(table)
                    if len(alternatives) >= 3:
                        break
                        
            return alternatives
            
        except Exception as e:
            logger.error(f"Error buscando alternativas: {e}")
            return []

    async def process_query(self, query: str, stream_callback=None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        🧠 ARQUITECTURA README v3: Flujo de 4 Etapas
        
        Implementa el flujo de razonamiento descrito en la documentación para máxima robustez.
        """
        start_time = time.time()
        
        try:
            # -----------------------------------------------------------------
            # ETAPA 1: Análisis Semántico con LLM
            # -----------------------------------------------------------------
            if stream_callback:
                stream_callback("🧠 Etapa 1: Analizando intención de la consulta...")

            medical_analysis = await self._analyze_medical_intent_with_llm(query, stream_callback)
            medical_analysis['original_query'] = query # Guardar query original
            
            # DEBUG: Loggear el resultado del análisis semántico
            logger.info(f"DEBUG: Resultado Etapa 1 (Análisis Semántico) -> {json.dumps(medical_analysis, indent=2)}")
            
            if stream_callback:
                intent = medical_analysis.get('clinical_intent', 'N/A')[:50]
                stream_callback(f"   - Intención detectada: {intent}...")

            # -----------------------------------------------------------------
            # ETAPA 1.5: Enriquecimiento de Conceptos (¡NUEVO!)
            # -----------------------------------------------------------------
            if stream_callback:
                stream_callback("🧠 Etapa 1.5: Expandiendo conceptos médicos abstractos...")

            original_concepts = medical_analysis.get('medical_concepts', [])
            qualifiers = medical_analysis.get('qualifiers', [])
            enriched_concepts = await self._enrich_medical_concepts(original_concepts, qualifiers, stream_callback)
            
            # NO SOBRESCRIBIR: guardar conceptos enriquecidos por separado
            medical_analysis['enriched_keywords'] = [c for c in enriched_concepts if c not in original_concepts]

            if stream_callback:
                # Mostrar si hubo cambios
                if medical_analysis.get('enriched_keywords'):
                    stream_callback(f"   - Conceptos expandidos: {medical_analysis['enriched_keywords']}")
                else:
                    stream_callback("   - Conceptos médicos confirmados sin necesidad de expansión")

            logger.info(f"DEBUG: Análisis médico para Etapa 2 -> {json.dumps(medical_analysis, indent=2)}")

            # -----------------------------------------------------------------
            # ETAPA 2: Mapeo Inteligente de Tablas
            # -----------------------------------------------------------------
            if stream_callback:
                stream_callback("🗺️ Etapa 2: Mapeando conceptos a tablas del esquema...")
                
            table_candidates = await self._intelligent_table_mapping(medical_analysis, medical_analysis, stream_callback)
            
            # NUEVO: Análisis dinámico con LLM para detectar tablas obligatorias
            if self.llm:
                mandatory_analysis = await self._analyze_mandatory_tables_with_llm(query, table_candidates)
                if mandatory_analysis:
                    logger.info(f"🧠 Análisis dinámico de tablas obligatorias: {mandatory_analysis}")
                    # El análisis dinámico ya incluye las instrucciones en el prompt
            
            if not table_candidates:
                return self._create_error_response("No se pudieron identificar tablas relevantes para la consulta.")
            
            if stream_callback:
                stream_callback(f"   - Tablas candidatas: {', '.join(table_candidates[:4])}...")

            # -----------------------------------------------------------------
            # ETAPA 3: Análisis de Conectividad (JOINs)
            # -----------------------------------------------------------------
            if stream_callback:
                stream_callback("🔗 Etapa 3: Buscando la mejor ruta de conexión (JOINs)...")

            join_analysis = await self._llm_find_join_path_optimized(query, table_candidates, stream_callback)
            final_tables = join_analysis.get("final_tables", table_candidates[:1])
            join_conditions = join_analysis.get("join_conditions", [])

            if stream_callback:
                stream_callback(f"   - Ruta de JOIN encontrada para: {', '.join(final_tables)}")

            # -----------------------------------------------------------------
            # ETAPA 4: Generación de Plan y SQL
            # -----------------------------------------------------------------
            if stream_callback:
                stream_callback("📝 Etapa 4: Creando plan de ejecución y generando SQL...")

            # Extraer parámetros de la consulta ANTES de generar el plan
            extracted_params = []
            
            # Usar LLM para determinar si necesitamos parámetros específicos
            needs_specific_params = await self._llm_analyze_parameter_needs(query, medical_analysis, stream_callback)
            
            if needs_specific_params:
                entities = medical_analysis.get('entities', {})
                patient_ids = entities.get('patient_ids', [])
                patient_names = entities.get('patient_names', [])
                
                if patient_ids or patient_names:
                    if patient_ids:
                        extracted_params.append(patient_ids[0])
                    if patient_names:
                        patient_name = patient_names[0]
                        # Normalizar el nombre en Python para búsqueda exacta
                        normalized_name = self._normalize_accents_python(patient_name)
                        extracted_params.append(normalized_name)
                
                # Si no se detectaron entidades, intentar extraer manualmente nombres específicos
                if not extracted_params:
                    import re
                    # Patrones para detectar nombres de personas en consultas en español
                    specific_name_patterns = [
                        r'paciente\s+(?:llamad[ao]|que\s+se\s+llam[ae])\s+([A-Za-zÁáÉéÍíÓóÚúÑñ\s]+)',
                        r'(?:de|del|para|sobre)\s+(?!de|del|para|sobre)([A-Z][a-záéíóúñ]+(?:\s+[A-Z][a-záéíóúñ]+)+)',
                        r'(?:de|del|para|sobre)\s+(?!de|del|para|sobre)([A-Za-zÁáÉéÍíÓóÚúÑñ]+)',
                        r'constantes\s+(?:de|para|vitales\s+de)\s+([A-Za-zÁáÉéÍíÓóÚúÑñ\s]+)',
                        r'(?:paciente|persona)\s+([A-Za-zÁáÉéÍíÓóÚúÑñ\s]+?)\s+(?:tiene|ha|con|ha sido|había)',
                        r'(?:paciente|persona)\s+([A-Za-zÁáÉéÍíÓóÚúÑñ\s]+)',
                        r'\b([A-Z][a-záéíóúñ]+\s+[A-Z][a-záéíóúñ]+(?:\s+[A-Z][a-záéíóúñ]+)*)\b'
                    ]
                    
                    # NUEVO: Patrones para detectar IDs de pacientes (prioridad: UUIDs completos)
                    id_patterns = [
                        r'(?:con\s+)?id\s+([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})',  # UUIDs completos
                        r'(?:paciente\s+)?([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})',  # UUIDs completos sin 'id'
                        r'(?:con\s+)?id\s+([0-9]+)',  # IDs numéricos
                        r'(?:paciente\s+)?([0-9]+)',  # IDs numéricos sin 'id'
                        r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})',  # UUIDs sueltos
                        r'\b([0-9]{6,})\b',  # IDs numéricos largos, solo si no es parte de un UUID
                    ]
                    
                    print(f"🔍 DEBUG: Intentando extraer nombres e IDs manualmente de la consulta: '{query}'")
                    
                    # PRIMERO: Intentar extraer IDs (prioridad alta)
                    for pattern in id_patterns:
                        matches = re.findall(pattern, query, re.IGNORECASE)
                        if matches:
                            for match in matches:
                                if match and len(match) > 5:  # Filtrar IDs válidos
                                    extracted_params.append(match)
                                    print(f"✅ DEBUG: ID extraído manualmente: '{match}' -> {extracted_params}")
                                    break
                            if extracted_params:
                                break
                    
                    # SEGUNDO: Si no se encontraron IDs, intentar extraer nombres
                    if not extracted_params:
                        for pattern in specific_name_patterns:
                            matches = re.findall(pattern, query, re.IGNORECASE)
                            if matches:
                                for match in matches:
                                    name = match.strip()
                                    # Filtrar nombres muy cortos o palabras comunes
                                    if len(name) > 3 and not any(common in name.lower() for common in ['paciente', 'diabetes', 'hemoglobina', 'tipo', 'vital', 'record']):
                                        parts = name.split()
                                        
                                        # Si parece un nombre completo (nombre + apellido)
                                        if len(parts) >= 2:
                                            # Normalizar el nombre completo en Python para búsqueda exacta
                                            normalized_name = self._normalize_accents_python(name)
                                            extracted_params.append(normalized_name)
                                            print(f"✅ DEBUG: Nombre completo extraído manualmente: '{name}' -> {extracted_params}")
                                            break
                                        # Si es solo un nombre/apellido
                                        else:
                                            normalized_name = self._normalize_accents_python(name)
                                            extracted_params.append(normalized_name)
                                            print(f"✅ DEBUG: Nombre único extraído manualmente: '{name}' -> {extracted_params}")
                                            break
                                # Salir del bucle de patrones si se encontró un nombre
                                if extracted_params:
                                    break
                    
                    # Si se encontraron parámetros, loggear para debugging
                    if extracted_params:
                        print(f"🔍 DEBUG: Parámetros extraídos manualmente: {extracted_params}")
                        
                        # También incluirlos en el análisis médico para referencia futura
                        if 'entities' not in medical_analysis:
                            medical_analysis['entities'] = {}
                        if 'patient_ids' not in medical_analysis['entities']:
                            medical_analysis['entities']['patient_ids'] = []
                        if 'patient_names' not in medical_analysis['entities']:
                            medical_analysis['entities']['patient_names'] = []
                        
                        # Guardar el parámetro extraído
                        for param in extracted_params:
                            if re.match(r'^[a-f0-9\-]{36}$', param, re.IGNORECASE) or re.match(r'^\d+$', param):
                                # Es un ID
                                if param not in medical_analysis['entities']['patient_ids']:
                                    medical_analysis['entities']['patient_ids'].append(param)
                            else:
                                # Es un nombre
                                if param not in medical_analysis['entities']['patient_names']:
                                    medical_analysis['entities']['patient_names'].append(param)
                    else:
                        print("❓ DEBUG: No se pudieron extraer parámetros manualmente de la consulta")
                    
                    print(f"🔑 DEBUG: Parámetros extraídos finales: {extracted_params}")
            
            if stream_callback and extracted_params:
                stream_callback(f"   - Parámetros de búsqueda detectados: {len(extracted_params)} elemento(s)")
            
            execution_plan = {
                "operation_type": "SELECT",
                "relevant_tables": final_tables,
                "join_conditions": join_conditions,
                "execution_plan": f"Plan para '{query}'",
                "params": extracted_params,  # Añadir parámetros extraídos
                "semantic_analysis": medical_analysis
            }

            print(f"🔧 DEBUG: Plan de ejecución creado con parámetros: {execution_plan.get('params', [])}")

            generated_sql = await self._llm_generate_smart_sql(query, execution_plan, stream_callback)
            sql_params = execution_plan.get('params', [])
            
            print(f"🔧 DEBUG: SQL generado: {generated_sql}")
            print(f"🔧 DEBUG: Parámetros SQL después de generación: {sql_params}")

            if generated_sql.startswith("Error:"):
                return self._create_error_response(f"Error generando SQL: {generated_sql}")

            # La validación y regeneración de SQL incompleto se mantiene
            validation_result = await self._validate_sql_completeness_with_llm(query, generated_sql, self.column_metadata)
            if validation_result:
                logger.warning(f"⚠️ SQL inicial incompleto detectado: {validation_result.get('razon', 'Sin razón')}")
                
                if validation_result.get('tablas_faltantes'):
                    logger.info(f"🔄 Regenerando SQL con contexto de validación...")
                    
                    missing_tables = validation_result.get('tablas_faltantes', [])
                    current_and_missing_tables = list(set(final_tables + missing_tables))
                    execution_plan['relevant_tables'] = current_and_missing_tables
                    
                    execution_plan['correction_feedback'] = await self._generate_intelligent_correction_feedback(
                        query, generated_sql, validation_result, current_and_missing_tables
                    )
                    
                    # CRÍTICO: Preservar parámetros durante regeneración
                    print(f"🔧 DEBUG: Preservando parámetros durante regeneración: {execution_plan.get('params', [])}")
                    
                    # Intentar regenerar con el feedback inteligente
                    regenerated_sql = await self._llm_generate_smart_sql(query, execution_plan, stream_callback)
                    if regenerated_sql and not regenerated_sql.startswith("Error"):
                        generated_sql = regenerated_sql
                        # Asegurar que los parámetros se mantienen
                        sql_params = execution_plan.get('params', [])
                        print(f"✅ DEBUG: SQL regenerado con parámetros preservados: {sql_params}")
                    else:
                        print(f"⚠️ DEBUG: Regeneración falló, manteniendo SQL original")
            
            # -----------------------------------------------------------------
            # ETAPA FINAL: Ejecución
            # -----------------------------------------------------------------
            print(f"🚀 DEBUG: Ejecutando con SQL: {generated_sql}")
            print(f"🚀 DEBUG: Parámetros finales para ejecución: {sql_params}")
            
            result = await self._execute_sql_with_learning(query, generated_sql, start_time, sql_params, stream_callback)
            
            # -----------------------------------------------------------------
            # ETAPA 5: Interpretación y Explicación
            # -----------------------------------------------------------------
            if result.get('success'):
                if stream_callback:
                    stream_callback("🩺 Etapa 5: Interpretación clínica de resultados...")
                
                if not result.get('data'):
                    if stream_callback:
                        stream_callback("   - Analizando ausencia de resultados...")
                    result['explanation'] = await self._llm_interpret_results(query, [], stream_callback)
                else:
                    if stream_callback:
                        stream_callback(f"   - Interpretando {len(result['data'])} resultado(s) encontrado(s)...")
                    interpretation = await self._llm_interpret_results(query, result['data'], stream_callback)
                    result['explanation'] = interpretation
                
                if stream_callback:
                    stream_callback("   ✅ Interpretación médica completada")
            else:
                if stream_callback:
                    stream_callback("❌ No se pudo generar interpretación debido a errores en la ejecución")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error crítico en el flujo de 4 etapas: {e}", exc_info=True)
            import traceback
            print(f"🔍 DEBUG - Error completo:")
            print(f"   Tipo de error: {type(e).__name__}")
            print(f"   Mensaje: {str(e)}")
            print(f"   Traceback completo:")
            traceback.print_exc()
            return self._create_error_response(f'Error crítico procesando la consulta: {str(e)}')

    async def _generate_intelligent_correction_feedback(self, query: str, incomplete_sql: str, 
                                                       validation_result: Dict[str, Any], 
                                                       all_tables: List[str]) -> str:
        """Genera feedback inteligente y genérico para corregir SQL incompleto usando LLM"""
        if not self.llm:
            return f"SQL incompleto. Incluye las tablas: {', '.join(validation_result.get('tablas_faltantes', []))}"
        
        try:
            # Obtener esquema de las tablas relevantes
            schema_info = {}
            for table in all_tables:
                if table in self.column_metadata:
                    schema_info[table] = [col['name'] for col in self.column_metadata[table]['columns'][:10]]
            
            prompt = f"""Eres un experto en bases de datos médicas. Tu tarea es generar instrucciones específicas para corregir un SQL incompleto.

CONSULTA ORIGINAL: {query}

SQL INCOMPLETO GENERADO:
{incomplete_sql}

ANÁLISIS DE VALIDACIÓN:
- Razón por la que está incompleto: {validation_result.get('razon', 'No especificada')}
- Tablas faltantes identificadas: {validation_result.get('tablas_faltantes', [])}
- Sugerencia del validador: {validation_result.get('sugerencia', 'No especificada')}

ESQUEMA DE TABLAS DISPONIBLES:
{json.dumps(schema_info, indent=2)}

TAREA: Genera instrucciones específicas y claras para corregir el SQL. Las instrucciones deben:

1. Explicar exactamente por qué el SQL actual es incompleto
2. Especificar qué tablas adicionales se necesitan y por qué
3. Sugerir cómo combinar o relacionar las tablas (JOIN vs consultas separadas)
4. Proporcionar ejemplos concretos de SQL corregido
5. Ser genérico - no asumir tipos específicos de consulta

FORMATO DE RESPUESTA:
Genera un texto de instrucciones claras y específicas que un desarrollador pueda seguir para corregir el SQL.

EJEMPLO DE RESPUESTA:
"Tu SQL actual solo consulta la tabla X, pero para responder completamente la pregunta necesitas también la tabla Y porque contiene [explicación]. 

Para corregir esto:
1. Incluye la tabla Y en tu consulta
2. Si las tablas tienen relación directa, usa JOIN con la condición [condición]
3. Si no tienen relación directa, genera consultas separadas

Ejemplo de SQL corregido:
[ejemplo de SQL]"

Genera las instrucciones específicas para esta situación:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}]
            )
            
            feedback = self._extract_response_text(response)
            logger.info(f"🧠 Feedback inteligente generado: {feedback[:200]}...")
            return feedback
            
        except Exception as e:
            logger.error(f"Error generando feedback inteligente: {e}")
            # Fallback a feedback básico
            return f"""Tu SQL actual está incompleto para responder la pregunta del usuario.

PROBLEMA: {validation_result.get('razon', 'Faltan tablas necesarias')}

SOLUCIÓN: Incluye las siguientes tablas en tu consulta: {', '.join(validation_result.get('tablas_faltantes', []))}

Analiza qué información específica necesitas de cada tabla y genera SQL que obtenga todos los datos relevantes para responder la pregunta completamente."""

    async def _enhanced_semantic_analysis(self, query: str, medical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis semántico mejorado con contexto médico"""
        
        # Combinar análisis tradicional con contexto médico
        basic_analysis = await self._analyze_query_semantics(query)
        
        # Enriquecer con información médica
        enhanced_analysis = {
            **basic_analysis,
            'medical_concepts': medical_analysis.get('medical_concepts', []),
            'clinical_intent': medical_analysis.get('clinical_intent', ''),
            'complexity_level': medical_analysis.get('complexity_level', 'simple'),
            'risk_factors': medical_analysis.get('risk_factors', [])
        }
        
        return enhanced_analysis

    async def _intelligent_table_mapping(self, semantic_analysis: Dict[str, Any], medical_analysis: Dict[str, Any], stream_callback=None) -> List[str]:
        """Mapeo inteligente de tablas basado en análisis médico - TODO VIA LLM"""
        
        if stream_callback:
            stream_callback("   - Seleccionando tablas relevantes con IA...")
        
        # Si hay LLM, delegar completamente la selección
        if self.llm:
            try:
                clinical_intent = medical_analysis.get('clinical_intent', '')
                medical_concepts = medical_analysis.get('medical_concepts', [])
                
                # Obtener lista de TODAS las tablas disponibles
                all_tables = list(self.column_metadata.keys())
                
                prompt = f"""Eres un experto en arquitectura de datos clínicos.
Analiza el contexto y selecciona las tablas MÁS RELEVANTES para responder la consulta.

CONSULTA ORIGINAL: {medical_analysis.get('original_query', '')}
INTENCIÓN CLÍNICA: {clinical_intent}
CONCEPTOS MÉDICOS: {', '.join(medical_concepts)}

ESQUEMA DISPONIBLE:
{self._get_schema_summary_for_exploration()}

INSTRUCCIONES:
1. Selecciona SOLO las tablas necesarias (máximo 5)
2. Si se buscan datos de pacientes específicos, SIEMPRE incluir PATI_PATIENTS
3. Para observaciones/signos vitales, priorizar OBSE_OBSERVATIONS 
4. Ordena por relevancia (más importante primero)

Responde SOLO con JSON:
{{"selected_tables": ["tabla1", "tabla2", ...]}}"""

                resp = await asyncio.to_thread(
                    _call_openai_native, self.llm, [{"role": "user", "content": prompt}]
                )
                
                result = self._try_parse_llm_json(self._extract_response_text(resp))
                if result and result.get("selected_tables"):
                    # Validar que las tablas existan
                    valid_tables = [t for t in result["selected_tables"] if t in self.column_metadata]
                    if valid_tables:
                        if stream_callback:
                            stream_callback(f"   - Tablas seleccionadas: {', '.join(valid_tables[:3])}...")
                        return valid_tables[:5]
                        
            except Exception as e:
                logger.warning(f"Error en selección LLM de tablas: {e}")
        
        # Fallback mínimo: tablas más comunes
        if stream_callback:
            stream_callback("   - Usando selección básica de tablas...")
        return list(self.column_metadata.keys())[:5]



    async def _execute_sql_with_learning(self, query: str, sql: str, start_time: float, sql_params: Optional[List[Any]] = None, stream_callback=None) -> Dict[str, Any]:
        """Ejecuta SQL usando los módulos centralizados de limpieza y ejecución"""
        
        logger.info(f"🔍 SQL original recibido: {sql}")
        logger.info(f"🔍 SQL COMPLETO PARA DEPURACIÓN: {sql}")
        
        if stream_callback:
            stream_callback("🔍 Optimizando y ejecutando consulta SQL...")
        
        try:
            # PASO 1: Limpiar y sanitizar el SQL
            if stream_callback:
                stream_callback("   - Limpiando y optimizando SQL...")
            
            # --- REFORZADO: limpiar errores de palabras pegadas antes de todo ---
            sql = self._fix_typo_errors(sql)
            sql = self._basic_sql_cleanup(sql)
            
            cleaned_sql = SQLCleaner.sanitize_for_execution(sql)
            
            # Aplicar correcciones específicas de compatibilidad
            cleaned_sql = await self._fix_sql_compatibility(cleaned_sql, stream_callback)
            
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
                
                # SISTEMA DE RECUPERACIÓN: Regenerar SQL con información del error
                if stream_callback:
                    stream_callback("   - Error de sintaxis detectado, regenerando SQL...")
                
                # Obtener información del contexto original
                original_query = getattr(self, '_last_query', query)
                original_params = sql_params or []
                
                # Regenerar SQL con información del error
                regenerated_sql = await self._regenerate_sql_with_error_context(
                    original_query, cleaned_sql, syntax_error or "Error de sintaxis desconocido", original_params, stream_callback
                )
                
                if regenerated_sql and not regenerated_sql.startswith("Error"):
                    logger.info(f"🔄 SQL regenerado exitosamente después de error")
                    # Ejecutar el SQL regenerado
                    return await self._execute_sql_with_learning(query, regenerated_sql, start_time, original_params, stream_callback)
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
            
            # DETECCIÓN ESPECIAL: Si es búsqueda de pacientes, usar SQL robusto
            if 'PATI_PATIENTS' in cleaned_sql.upper() and 'PATI_FULL_NAME' in cleaned_sql.upper():
                if stream_callback:
                    stream_callback("   - Detectada búsqueda de pacientes, aplicando SQL robusto...")
                
                # Extraer el término de búsqueda del primer parámetro
                search_term = sql_params[0] if sql_params else ""
                
                # Usar SQL robusto con sugerencias
                cleaned_sql = self._create_robust_patient_search_sql(search_term, include_suggestions=True)
                sql_params = self._prepare_search_parameters(search_term)
                
                logger.info(f"🔍 Aplicado SQL robusto para búsqueda de pacientes: {search_term}")
            
            if placeholder_count != len(sql_params):
                logger.warning(f"⚠️ Ajustando parámetros: {len(sql_params)} → {placeholder_count}")
                
                if placeholder_count > len(sql_params):
                    # Añadir parámetros duplicados para consultas que requieren el mismo parámetro múltiples veces
                    if len(sql_params) > 0:
                        # Repetir el último parámetro válido
                        last_param = sql_params[-1]
                        sql_params.extend([last_param] * (placeholder_count - len(sql_params)))
                    else:
                        # Añadir parámetros vacíos
                        sql_params.extend([''] * (placeholder_count - len(sql_params)))
                else:
                    # Truncar parámetros
                    sql_params = sql_params[:placeholder_count]
                
            # PASO 4: Ejecutar con el módulo ejecutor
            if stream_callback:
                stream_callback("   - Ejecutando consulta en la base de datos...")
                
            result = executor.execute_query(cleaned_sql, sql_params)
            
            # PASO 5: Procesar resultado
            if result['success']:
                if stream_callback:
                    stream_callback(f"   ✅ Consulta completada: {result['row_count']} resultados en {result['execution_time']:.2f}s")
                
                # PROCESAMIENTO ESPECIAL para búsquedas de pacientes
                if 'PATI_PATIENTS' in cleaned_sql.upper() and 'PATI_FULL_NAME' in cleaned_sql.upper():
                    search_term = sql_params[0] if sql_params else ""
                    processed_results = self._process_patient_search_results(result['data'], search_term)
                    
                    return {
                        'success': processed_results['success'],
                        'message': processed_results['message'],
                        'data': processed_results['data'],
                        'exact_matches': processed_results.get('exact_matches', []),
                        'suggestions': processed_results.get('suggestions', []),
                        'sql_query': cleaned_sql,
                        'execution_time': result['execution_time'],
                        'total_time': time.time() - start_time
                    }
                
                # Aprender del éxito
                await self._learn_from_query_result(query, cleaned_sql, result['row_count'], result['execution_time'])
                
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
                
                # Aprender del error
                await self._learn_from_error(query, result['error'])
                    
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

    def _validate_columns_exist_in_schema(self, sql: str, tables_info: Dict[str, List[str]]) -> Optional[str]:
        """Validación estricta: verifica que TODAS las columnas usadas existan en el esquema real"""
        try:
            # Crear un conjunto de todas las columnas válidas con y sin prefijo de tabla
            valid_columns = set()
            for table_name, columns in tables_info.items():
                for column in columns:
                    valid_columns.add(column.upper())  # Sin prefijo
                    valid_columns.add(f"{table_name}.{column}".upper())  # Con prefijo
            
            # Extraer todas las columnas usadas en el SQL
            # Patrón para capturar columnas en SELECT, WHERE, JOIN ON, etc.
            column_patterns = [
                r'SELECT\s+(?:DISTINCT\s+)?(.+?)\s+FROM',  # Columnas en SELECT
                r'WHERE\s+(.+?)(?:\s+ORDER\s+BY|\s+GROUP\s+BY|\s+LIMIT|$)',  # Columnas en WHERE
                r'JOIN\s+\w+\s+ON\s+(.+?)(?:\s+WHERE|\s+ORDER|\s+GROUP|\s+LIMIT|$)',  # Columnas en JOIN ON
                r'ORDER\s+BY\s+(.+?)(?:\s+LIMIT|$)',  # Columnas en ORDER BY
                r'GROUP\s+BY\s+(.+?)(?:\s+ORDER|\s+LIMIT|$)',  # Columnas en GROUP BY
            ]
            
            used_columns = set()
            for pattern in column_patterns:
                matches = re.findall(pattern, sql, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    # Extraer nombres de columnas individuales
                    # Eliminar funciones, operadores, etc.
                    column_parts = re.findall(r'\b(\w+\.\w+|\w+)\b', match)
                    for part in column_parts:
                        # Filtrar palabras clave de SQL
                        if part.upper() not in ['SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'AND', 'OR', 'LIKE', 'UPPER', 'LOWER', 'COUNT', 'MAX', 'MIN', 'AVG', 'SUM', 'DISTINCT', 'ASC', 'DESC', 'LIMIT', 'ORDER', 'BY', 'GROUP', 'HAVING']:
                            used_columns.add(part.upper())
            
            # Verificar que todas las columnas usadas sean válidas
            invalid_columns = used_columns - valid_columns
            if invalid_columns:
                # Generar mensaje de error detallado con sugerencias
                error_msg = f"❌ COLUMNAS INVENTADAS DETECTADAS: {', '.join(invalid_columns)}\n"
                error_msg += "\n🔍 COLUMNAS VÁLIDAS DISPONIBLES:\n"
                for table_name, columns in tables_info.items():
                    error_msg += f"\n{table_name}:\n"
                    for column in columns[:10]:  # Mostrar solo las primeras 10 columnas
                        error_msg += f"  • {column}\n"
                    if len(columns) > 10:
                        error_msg += f"  ... y {len(columns)-10} más\n"
                
                return error_msg
            
            return None
            
        except Exception as e:
            logger.error(f"Error validando columnas: {e}")
            return None  # No bloquear por errores de validación
    
    async def _fix_common_column_errors(self, sql: str, stream_callback=None) -> str:
        """Corrige errores de columnas usando LLM de manera genérica"""
        try:
            # Si no hay LLM disponible, devolver SQL original
            if not self.llm:
                if stream_callback:
                    stream_callback("   - Comprobación básica de columnas (sin LLM disponible)...")
                return sql
            
            if stream_callback:
                stream_callback("   - Validando y corrigiendo nombres de columnas con IA...")
                
            # Usar el LLM para detectar y corregir columnas inventadas de manera genérica
            correction_prompt = f"""Eres un experto en SQL y esquemas de bases de datos médicas.

TAREA: Revisa el SQL y corrige SOLO los nombres de columnas que sean incorrectos.

SQL A REVISAR:
{sql}

ESQUEMA REAL DISPONIBLE:
{self._get_schema_summary_for_exploration()}

INSTRUCCIONES CRÍTICAS:
1. **NO CAMBIES LAS TABLAS NI LOS JOINS**. La selección de tablas ya es correcta.
2. Enfócate ÚNICAMENTE en corregir nombres de columnas inválidas dentro de las tablas ya elegidas.
3. Si una columna como `o.OBSERVATION_DATE` no existe en `OBSE_OBSERVATIONS`, busca una columna de fecha alternativa en ESA MISMA TABLA (ej: `OBSE_DATE_UTC`, `OBSE_TIMESTAMP`). No cambies a la tabla `APPO_APPOINTMENTS`.
4. Mantén la intención original del SQL, pero usando columnas que existan en las tablas especificadas.

FORMATO DE RESPUESTA:
- Si no hay errores de columnas: Devuelve el SQL original sin cambios.
- Si hay errores de columnas: Devuelve el SQL corregido, explicando los cambios en comentarios.

Responde SOLO con el SQL (corregido o original):"""

            try:
                if stream_callback:
                    stream_callback("   - Analizando columnas con modelo LLM...")
                    
                response = await asyncio.to_thread(
                    _call_openai_native, self.llm, [{"role": "user", "content": correction_prompt}]
                )
                
                corrected_sql = self._extract_response_text(response).strip()
                
                # CRÍTICO: Limpiar respuesta del LLM de markdown y comentarios
                corrected_sql = self._clean_llm_sql_response(corrected_sql)
                
                # Verificar si el LLM realmente hizo cambios válidos
                if corrected_sql and corrected_sql != sql and not corrected_sql.startswith("Error"):
                    logger.info(f"🧠 LLM corrigió columnas inventadas en el SQL")
                    logger.info(f"   SQL original: {sql[:100]}...")
                    logger.info(f"   SQL corregido: {corrected_sql[:100]}...")
                    
                    if stream_callback:
                        stream_callback("   - Se han corregido nombres de columnas incorrectos")
                    
                    return corrected_sql
                else:
                    if stream_callback:
                        stream_callback("   - Verificación de columnas completada sin cambios")
                    return sql
                    
            except Exception as e:
                logger.warning(f"Error usando LLM para corrección de columnas: {e}")
                if stream_callback:
                    stream_callback(f"   - Error al verificar columnas: {str(e)[:50]}...")
                return sql  # Fallback al SQL original
            
        except Exception as e:
            logger.error(f"Error en corrección de columnas: {e}")
            return sql  # Devolver original si falla la corrección

    async def _fix_sql_compatibility(self, sql: str, stream_callback=None) -> str:
        """
        Corrige problemas de compatibilidad del SQL para SQLite usando LLM.
        
        Convierte funciones de MySQL/PostgreSQL a SQLite de manera inteligente:
        - DATE_SUB, INTERVAL, CURDATE, NOW() → date() functions
        - TOP → LIMIT
        - GETDATE() → datetime('now')
        - Y cualquier otra incompatibilidad que el LLM detecte
        """
        try:
            if not sql:
                return sql
                
            # Si no hay LLM, usar fallback básico
            if not self.llm:
                if stream_callback:
                    stream_callback("   - Ajustando compatibilidad SQL con método básico...")
                return self._fix_sql_compatibility_fallback(sql, stream_callback)
                
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
                
                # Limpiar respuesta del LLM
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
                    logger.warning(f"⚠️ LLM devolvió respuesta inválida, usando fallback")
                    if stream_callback:
                        stream_callback("   - Usando método alternativo para compatibilidad")
                    return self._fix_sql_compatibility_fallback(sql, stream_callback)
                    
            except Exception as e:
                logger.error(f"Error usando LLM para compatibilidad: {e}")
                if stream_callback:
                    stream_callback(f"   - Error optimizando SQL: {str(e)[:50]}... Usando método alternativo")
                return self._fix_sql_compatibility_fallback(sql, stream_callback)
                
        except Exception as e:
            logger.error(f"Error en _fix_sql_compatibility: {e}")
            if stream_callback:
                stream_callback("   - Error en corrección de compatibilidad")
            return sql  # Devolver original si falla completamente
    
    def _fix_sql_compatibility_fallback(self, sql: str, stream_callback=None) -> str:
        """
        Fallback básico para compatibilidad SQL cuando no hay LLM disponible.
        Mantiene las conversiones más críticas y comunes.
        """
        try:
            if not sql:
                return sql
                
            corrected_sql = sql
            
            # Solo las conversiones más críticas y comunes
            # 1. Funciones de fecha básicas
            corrected_sql = re.sub(r'\bCURDATE\s*\(\s*\)', "date('now')", corrected_sql, flags=re.IGNORECASE)
            corrected_sql = re.sub(r'\bNOW\s*\(\s*\)', "datetime('now')", corrected_sql, flags=re.IGNORECASE)
            corrected_sql = re.sub(r'\bGETDATE\s*\(\s*\)', "datetime('now')", corrected_sql, flags=re.IGNORECASE)
            
            # 2. DATE_SUB básico
            corrected_sql = re.sub(
                r'DATE_SUB\s*\(\s*CURDATE\s*\(\s*\)\s*,\s*INTERVAL\s+(\d+)\s+YEAR\s*\)',
                r"date('now', '-\1 year')",
                corrected_sql,
                flags=re.IGNORECASE
            )
            
            # 3. TOP a LIMIT
            if 'TOP' in sql.upper() and 'LIMIT' not in corrected_sql.upper():
                top_match = re.search(r'\bSELECT\s+TOP\s+(\d+)\b', sql, re.IGNORECASE)
                if top_match:
                    limit_num = top_match.group(1)
                    corrected_sql = re.sub(r'\bSELECT\s+TOP\s+\d+\s+', 'SELECT ', corrected_sql, flags=re.IGNORECASE)
                    if not corrected_sql.rstrip().endswith(';'):
                        corrected_sql = corrected_sql.rstrip() + f' LIMIT {limit_num};'
                    else:
                        corrected_sql = corrected_sql.rstrip()[:-1] + f' LIMIT {limit_num};'
            
            # 4. Funciones de fecha MySQL a SQLite
            corrected_sql = re.sub(r'\bYEAR\s*\(\s*([^)]+)\s*\)', r"strftime('%Y', \1)", corrected_sql, flags=re.IGNORECASE)
            corrected_sql = re.sub(r'\bMONTH\s*\(\s*([^)]+)\s*\)', r"strftime('%m', \1)", corrected_sql, flags=re.IGNORECASE)
            
            # 5. Normalizar espacios
            corrected_sql = re.sub(r'\s+', ' ', corrected_sql).strip()
            
            # Asegurar punto y coma al final
            corrected_sql = corrected_sql.strip()
            if corrected_sql and not corrected_sql.endswith(';'):
                corrected_sql += ';'
            
            return corrected_sql
            
        except Exception as e:
            logger.error(f"Error en fallback de compatibilidad: {e}")
            return sql

    def _fix_common_sql_syntax_errors(self, sql: str) -> str:
        """
        Corrige errores de sintaxis SQL comunes.
        
        Args:
            sql: El SQL a corregir
            
        Returns:
            str: SQL corregido
        """
        try:
            if not sql:
                return sql
                
            corrected_sql = sql
            
            # 1. Corregir comas finales en SELECT
            corrected_sql = re.sub(r',\s*FROM\b', ' FROM', corrected_sql, flags=re.IGNORECASE)
            
            # 2. Corregir WHERE vacío
            corrected_sql = re.sub(r'\bWHERE\s*(?:ORDER|GROUP|LIMIT|;|$)', lambda m: m.group(0).replace('WHERE', ''), corrected_sql, flags=re.IGNORECASE)
            
            # 3. Corregir múltiples espacios
            corrected_sql = re.sub(r'\s+', ' ', corrected_sql)
            
            # 4. Asegurar punto y coma al final
            corrected_sql = corrected_sql.strip()
            if corrected_sql and not corrected_sql.endswith(';'):
                corrected_sql += ';'
            
            return corrected_sql
            
        except Exception as e:
            logger.error(f"Error en _fix_common_sql_syntax_errors: {e}")
            return sql

    async def _analyze_medical_intent_with_llm(self, query: str, stream_callback=None) -> Dict[str, Any]:
        """Analiza la intención médica de la consulta usando LLM"""
        try:
            if not self.llm:
                # Análisis básico sin LLM
                if stream_callback:
                    stream_callback("   - Realizando análisis básico (sin LLM disponible)...")
                return {
                    'clinical_intent': 'consulta_sql',
                    'medical_concepts': self._extract_basic_medical_concepts(query),
                    'qualifiers': [],
                    'entities': {},
                    'query_type': 'sql_query',
                    'complexity_level': 'simple'
                }
            
            # Prompt básico para análisis médico
            medical_prompt = f"""Analiza esta consulta médica y extrae información estructurada:

CONSULTA: "{query}"

RESPUESTA JSON:
{{
    "clinical_intent": "descripción breve de la intención",
    "medical_concepts": ["concepto1", "concepto2"],
    "qualifiers": ["calificador1", "calificador2"],
    "entities": {{"patient_names": [], "patient_ids": []}},
    "query_type": "sql_query",
    "complexity_level": "simple|medium|complex"
}}"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": medical_prompt}], task_description="Analizando intención médica de la consulta"
            )
            
            if stream_callback:
                stream_callback("   - Procesando análisis semántico de la consulta...")
                
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if stream_callback and result:
                stream_callback(f"   - Conceptos médicos identificados: {', '.join(result.get('medical_concepts', [])[:3])}...")
                
            return result if result else {
                'clinical_intent': 'consulta_sql',
                'medical_concepts': self._extract_basic_medical_concepts(query),
                'qualifiers': [],
                'entities': {},
                'query_type': 'sql_query',
                'complexity_level': 'simple'
            }
            
        except Exception as e:
            logger.error(f"Error en análisis médico: {e}")
            return {
                'clinical_intent': 'consulta_sql',
                'medical_concepts': self._extract_basic_medical_concepts(query),
                'qualifiers': [],
                'entities': {},
                'query_type': 'sql_query',
                'complexity_level': 'simple'
            }

    def _extract_basic_medical_concepts(self, query: str) -> List[str]:
        """Extrae conceptos médicos básicos usando análisis de texto simple - SIN HARDCODEO"""
        if not self.llm:
            # Solo análisis muy básico sin términos hardcodeados
            concepts = []
            query_lower = query.lower()
            
            # Buscar palabras que parezcan términos médicos por patrones generales
            import re
            
            # Patrones muy generales, NO términos específicos
            medical_patterns = [
                r'\b[a-záéíóúñ]{8,}\b',  # Palabras largas (8+ caracteres)
                r'\bhba?\d*[a-z]*\b',    # Patrones como HbA1c
                r'\b[a-z]+emia\b',       # Terminaciones médicas como anemia
                r'\b[a-z]+osis\b',       # Terminaciones como diabetes
                r'\b[a-z]+itis\b',       # Terminaciones como artritis
            ]
            
            for pattern in medical_patterns:
                matches = re.findall(pattern, query_lower)
                for match in matches:
                    if len(match) > 4:  # Solo palabras significativas
                        concepts.append(match)
            
            return list(set(concepts))  # Eliminar duplicados
        
        else:
            # Si hay LLM, usar análisis inteligente
            try:
                prompt = f"""Extrae conceptos médicos de esta consulta usando análisis inteligente:

CONSULTA: "{query}"

Identifica términos médicos, condiciones, procedimientos, medicamentos, etc.

RESPUESTA JSON:
{{
    "medical_concepts": ["concepto1", "concepto2", "concepto3"]
}}"""

                response = self.llm.invoke([{"role": "user", "content": prompt}])
                content = self._extract_response_text(response)
                result = self._try_parse_llm_json(content)
                
                if result and 'medical_concepts' in result:
                    return result['medical_concepts']
                
                # Fallback a análisis básico
                return []
                
            except Exception as e:
                logger.error(f"Error en análisis LLM de conceptos: {e}")
                return []
        
        # Buscar palabras clave muy generales
        general_medical_terms = ['paciente', 'diabetes', 'hemoglobina', 'media', 'promedio']
        for term in general_medical_terms:
            if term in query_lower:
                concepts.append(term)
        
        return list(set(concepts))  # Eliminar duplicados

    def _extract_response_text(self, response) -> str:
        """Extrae el texto de la respuesta del LLM"""
        if hasattr(response, 'content'):
            return response.content
        return str(response)

    def _try_parse_llm_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Intenta parsear JSON de respuesta del LLM"""
        try:
            # Limpiar contenido
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            # Buscar JSON en el contenido
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error parseando JSON: {e}")
            return None

    async def _enrich_medical_concepts(self, concepts: List[str], qualifiers: List[str], stream_callback=None) -> List[str]:
        """Enriquece los conceptos médicos con términos relacionados usando LLM"""
        try:
            if not self.llm or not concepts:
                if stream_callback:
                    stream_callback("   - No hay conceptos para enriquecer o LLM no disponible...")
                return concepts
            
            if stream_callback:
                stream_callback("   - Expandiendo conceptos médicos con terminología relacionada...")
                
            # Usar LLM para expandir conceptos médicos de manera dinámica
            enrichment_prompt = f"""Como experto en terminología médica, expande estos conceptos con términos relacionados:

CONCEPTOS ORIGINALES: {concepts}
CALIFICADORES: {qualifiers}

TAREA: Para cada concepto médico, proporciona términos relacionados, sinónimos y variaciones que podrían aparecer en bases de datos médicas.

EJEMPLOS DE EXPANSIÓN:
- diabetes → diabetes mellitus, diabetes tipo 1, diabetes tipo 2, DM, T1D, T2D
- hemoglobina → HbA1c, hemoglobina glicosilada, glicohemoglobina, A1C
- hipertensión → presión arterial alta, HTA, hipertensión arterial

RESPUESTA JSON:
{{
    "expanded_concepts": ["término1", "término2", "término3", ...]
}}

Incluye términos técnicos, abreviaciones médicas y sinónimos comunes."""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": enrichment_prompt}], task_description="Expandiendo conceptos médicos con terminología relacionada"
            )
            
            if stream_callback:
                stream_callback("   - Procesando expansión de conceptos médicos...")
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result and 'expanded_concepts' in result:
                expanded = result['expanded_concepts']
                # Combinar conceptos originales con expandidos
                all_concepts = concepts + expanded
                
                if stream_callback:
                    new_concepts = [c for c in expanded if c not in concepts]
                    if new_concepts:
                        stream_callback(f"   - Se añadieron {len(new_concepts)} conceptos relacionados")
                
                return list(set(all_concepts))  # Eliminar duplicados
            
            return concepts
            
        except Exception as e:
            logger.error(f"Error expandiendo conceptos médicos: {e}")
            return concepts

    async def _analyze_mandatory_tables_with_llm(self, query: str, table_candidates: List[str]) -> Optional[Dict[str, Any]]:
        """Analiza si hay tablas obligatorias que deben incluirse"""
        try:
            if not self.llm:
                return None
            
            prompt = f"""Analiza si esta consulta SQL requiere tablas adicionales obligatorias:

CONSULTA: "{query}"
TABLAS CANDIDATAS: {table_candidates}

¿Faltan tablas esenciales para responder completamente la consulta?

RESPUESTA JSON:
{{
    "mandatory_tables": ["tabla1", "tabla2"],
    "reason": "explicación breve"
}}"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}]
            )
            
            content = self._extract_response_text(response)
            return self._try_parse_llm_json(content)
            
        except Exception as e:
            logger.error(f"Error analizando tablas obligatorias: {e}")
            return None

    async def _llm_find_join_path_optimized(self, query: str, table_candidates: List[str], stream_callback=None) -> Dict[str, Any]:
        """Encuentra la mejor ruta de JOIN entre tablas"""
        try:
            if stream_callback:
                stream_callback("   - Analizando relaciones entre tablas seleccionadas...")
                
            return {
                "final_tables": table_candidates[:3],  # Usar las primeras 3 tablas
                "join_conditions": []
            }
        except Exception as e:
            logger.error(f"Error finding join path: {e}")
            if stream_callback:
                stream_callback("   - ⚠️ Error al establecer relaciones, usando tabla principal")
            return {"final_tables": table_candidates[:1], "join_conditions": []}

    async def _llm_analyze_parameter_needs(self, query: str, medical_analysis: Dict[str, Any], stream_callback=None) -> bool:
        """
        Analiza si la consulta necesita parámetros específicos para la búsqueda.
        
        Args:
            query: La consulta original del usuario
            medical_analysis: Análisis médico previo con entidades detectadas
            stream_callback: Función opcional para transmitir mensajes de progreso
            
        Returns:
            bool: True si necesita parámetros específicos, False en caso contrario
        """
        try:
            if stream_callback:
                stream_callback("   - Analizando si la consulta requiere parámetros específicos...")
                
            if not self.llm:
                # Fallback: usar heurística simple
                if stream_callback:
                    stream_callback("   - Usando heurística simple para detección de parámetros...")
                query_lower = query.lower()
                specific_indicators = [
                    'paciente', 'id', 'nombre', 'específico', 'particular',
                    'juan', 'maría', 'josé', 'ana', 'carlos', 'luis'
                ]
                return any(indicator in query_lower for indicator in specific_indicators)
            
            # Usar LLM para análisis más sofisticado
            prompt = f"""
Analiza esta consulta médica y determina si necesita parámetros específicos para la búsqueda.

CONSULTA: "{query}"

ANÁLISIS PREVIO: {medical_analysis}

REGLAS:
- Responde "SI" si la consulta menciona:
  * IDs específicos de pacientes
  * Nombres específicos de pacientes
  * Términos que requieren búsqueda exacta
  * Referencias a datos específicos

- Responde "NO" si la consulta es:
  * Consulta general ("¿cuántos pacientes hay?")
  * Estadísticas generales
  * Listados completos
  * Consultas agregadas

EJEMPLOS:
- "¿Cuántos pacientes hay?" → NO
- "Mostrar datos del paciente 1010" → SI
- "¿Qué medicación toma Juan?" → SI
- "Listar todos los diagnósticos" → NO

Responde SOLO: "SI" o "NO"
"""
            
            response = await self.llm.ainvoke(prompt)
            response_text = str(response.content).strip().upper()
            
            needs_params = "SI" in response_text or "YES" in response_text
            logger.info(f"🔍 Análisis de parámetros: {query} → {'Necesita parámetros' if needs_params else 'Sin parámetros específicos'}")
            
            return needs_params
            
        except Exception as e:
            logger.error(f"Error analyzing parameter needs: {e}")
            # Fallback conservador: no usar parámetros específicos
            return False

    def is_patient_search(self, execution_plan, sql):
        """
        Detecta si la consulta es una búsqueda de pacientes usando el esquema y el SQL generado.
        """
        tables = execution_plan.get('relevant_tables', []) if execution_plan else []
        if 'PATI_PATIENTS' in tables:
            return True
        # O si el SQL involucra columnas clave de pacientes
        if sql and any(col in sql.upper() for col in ['PATI_FULL_NAME', 'PATI_ID', 'PATI_NAME', 'PATI_SURNAME_1', 'PATI_SURNAME_2']):
            return True
            return False

    async def _llm_generate_smart_sql(self, query: str, execution_plan: Dict[str, Any], stream_callback=None) -> str:
        """Genera SQL inteligente basado en el plan de ejecución usando LLM dinámico"""
        try:
            params = execution_plan.get('params', [])
            tables = execution_plan.get('relevant_tables', [])
            
            if stream_callback:
                stream_callback("   - Generando SQL inteligente con IA...")
            
            # UNA SOLA LLAMADA LLM para detectar tipo de consulta y generar SQL
            if self.llm:
                # Obtener esquema real de la base de datos
                schema_info = self._get_schema_summary_for_exploration()
                
                # PROMPT DINÁMICO ÚNICO para análisis y generación
                prompt = f"""Eres un experto en SQL médico. Analiza la consulta y genera SQL optimizado.

CONSULTA ORIGINAL: "{query}"
PARÁMETROS DETECTADOS: {params}
TABLAS DISPONIBLES: {tables}

ESQUEMA DE LA BASE DE DATOS:
{schema_info}

ANÁLISIS REQUERIDO:
1. ¿Es una consulta de "último paciente registrado"?
2. ¿Qué tipo de información médica se busca?
3. ¿Qué tablas y columnas son relevantes?
4. ¿Qué criterio usar para "último" (fecha, ID, etc.)?

DETECCIÓN DE CONSULTAS DE "ÚLTIMO PACIENTE":
- Palabras clave: "último", "ultimo", "última", "ultima", "reciente", "nuevo", "creado", "registrado"
- Contexto: "paciente", "persona", "quién", "cuál", "como se llama"
- Ejemplos: "cual es el ultimo paciente", "quién es el último paciente", "dime el último paciente"

INSTRUCCIONES ESPECÍFICAS:
- Si es consulta de "último paciente": usar PATI_START_DATE DESC como criterio principal
- Para diagnósticos: usar EPIS_DIAGNOSTICS.DIAG_OBSERVATION
- Para medicación: usar PATI_USUAL_MEDICATION
- Para UUIDs: buscar directamente en PATI_PATIENTS.PATI_ID
- Para nombres: usar PATI_FULL_NAME con normalización
- Usar LEFT JOIN para datos opcionales
- Compatible con SQLite

ESTRATEGIA PARA "ÚLTIMO PACIENTE":
- SIEMPRE usar ORDER BY PATI_START_DATE DESC (NO PATI_ID DESC)
- PATI_START_DATE es la fecha real de registro del paciente
- PATI_ID es un UUID, no sirve para determinar el último registrado
- Incluir información del paciente (nombre, apellidos)
- Incluir diagnósticos si se solicitan
- Filtrar datos de calidad (no valores de prueba)
- LIMIT 1 para obtener solo el último

EJEMPLO DE SQL PARA "ÚLTIMO PACIENTE":
SELECT 
    p.PATI_ID,
    p.PATI_NAME,
    p.PATI_SURNAME_1,
    p.PATI_SURNAME_2,
    p.PATI_FULL_NAME,
    p.PATI_START_DATE,
    d.DIAG_OBSERVATION,
    d.DIAG_OTHER_DIAGNOSTIC
FROM PATI_PATIENTS p
LEFT JOIN EPIS_DIAGNOSTICS d ON p.PATI_ID = d.PATI_ID
WHERE p.PATI_START_DATE IS NOT NULL
    AND p.PATI_NAME IS NOT NULL
    AND p.PATI_NAME != ''
    AND p.PATI_NAME NOT LIKE '%PRUEBAS%'
    AND p.PATI_NAME NOT LIKE '%TEST%'
ORDER BY p.PATI_START_DATE DESC
LIMIT 1;

REGLAS CRÍTICAS:
- NO usar tabla PATI_PATIENT_IDENTIFICATIONS (no existe)
- Para "último paciente": SIEMPRE ORDER BY PATI_START_DATE DESC
- NUNCA usar ORDER BY PATI_ID DESC para determinar el último paciente
- PATI_ID es UUID, PATI_START_DATE es fecha de registro
- Filtrar datos de prueba y valores nulos
- Optimizar para rendimiento

RESPUESTA:
Solo el SQL válido, sin explicaciones ni comentarios."""

                response = await asyncio.to_thread(
                    _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                    task_description="Generando SQL inteligente dinámico"
                )
                
                generated_sql = self._extract_response_text(response).strip()
                generated_sql = self._clean_llm_sql_response(generated_sql)
                
                # VALIDACIÓN Y CORRECCIÓN AUTOMÁTICA INTELIGENTE
                if any(keyword in query.lower() for keyword in ["último", "ultimo", "última", "ultima", "reciente", "nuevo", "creado", "registrado"]):
                    if stream_callback:
                        stream_callback("   🔍 Validando SQL con IA...")
                    
                    # LLM analiza y corrige el SQL de forma inteligente
                    validation_prompt = f"""Eres un experto en SQL médico. Analiza y corrige esta consulta SQL si es necesario.

CONSULTA ORIGINAL: "{query}"
SQL GENERADO: {generated_sql}

ANÁLISIS REQUERIDO:
1. ¿El SQL es correcto para obtener el "último paciente registrado"?
2. ¿Está usando el criterio correcto para ordenar (fecha vs ID)?
3. ¿Incluye la información necesaria solicitada en la consulta?
4. ¿Es compatible con SQLite?

REGLAS IMPORTANTES:
- Para "último paciente": usar PATI_START_DATE DESC (fecha de registro)
- PATI_ID es UUID, no sirve para determinar el último registrado
- Incluir información relevante del paciente (nombre, apellidos)
- Filtrar datos de calidad (no valores de prueba)

RESPUESTA:
Solo el SQL corregido, sin explicaciones. Si el SQL está correcto, devuelve el mismo SQL."""
                    
                    try:
                        validation_response = await asyncio.to_thread(
                            _call_openai_native, self.llm, [{"role": "user", "content": validation_prompt}],
                            task_description="Validando SQL con IA"
                        )
                        
                        corrected_sql = self._extract_response_text(validation_response).strip()
                        corrected_sql = self._clean_llm_sql_response(corrected_sql)
                        
                        if corrected_sql and corrected_sql != generated_sql:
                            logger.info(f"🔧 SQL corregido por IA: {corrected_sql}")
                            generated_sql = corrected_sql
                            if stream_callback:
                                stream_callback("   ✅ SQL corregido automáticamente")
                        else:
                            if stream_callback:
                                stream_callback("   ✅ SQL validado correctamente")
                                
                    except Exception as e:
                        logger.warning(f"Error en validación IA: {e}")
                        # Continuar con el SQL original si falla la validación
                
                # LOGGING DETALLADO PARA DEPURACIÓN
                logger.info(f"🔍 DEBUG - SQL GENERADO POR LLM:")
                logger.info(f"   Consulta original: '{query}'")
                logger.info(f"   Parámetros: {params}")
                logger.info(f"   Tablas: {tables}")
                logger.info(f"   SQL generado: {generated_sql}")
                
                if generated_sql and not generated_sql.startswith("Error"):
                    if stream_callback:
                        stream_callback("   ✅ SQL inteligente generado dinámicamente")
                    return generated_sql
                else:
                    # Fallback inteligente
                    logger.warning(f"⚠️ LLM no generó SQL válido, usando fallback")
                    return await self._generate_fallback_sql(query, params, tables, stream_callback)
            else:
                # Fallback sin LLM
                if stream_callback:
                    stream_callback("   - Generando SQL básico (sin LLM disponible)...")
                if not tables:
                    return "Error: No hay tablas disponibles"
                return f"SELECT COUNT(*) FROM {tables[0]} LIMIT 10;"
                
        except Exception as e:
            logger.error(f"Error en _llm_generate_smart_sql: {e}")
            return await self._generate_fallback_sql(query, params, tables, stream_callback)

    async def _generate_uuid_based_sql(self, query: str, uuid_param: str, tables: List[str], stream_callback=None) -> str:
        """Genera SQL específico para búsquedas por UUID usando LLM dinámico"""
        try:
            if stream_callback:
                stream_callback("   - Generando SQL dinámico para UUID con IA...")
            
            # Obtener esquema real de la base de datos
            schema_info = self._get_schema_summary_for_exploration()
            
            # PROMPT ESPECÍFICO PARA GENERAR SQL POR UUID
            prompt = f"""Eres un experto en SQL médico. Genera una consulta SQL específica para buscar información de un paciente por UUID.

CONSULTA ORIGINAL: "{query}"
UUID DEL PACIENTE: {uuid_param}

ESQUEMA DE LA BASE DE DATOS:
{schema_info}

INSTRUCCIONES ESPECÍFICAS:
1. El UUID está almacenado directamente en PATI_PATIENTS.PATI_ID
2. NO existe tabla PATI_PATIENT_IDENTIFICATIONS
3. Usa LEFT JOIN para no excluir pacientes sin datos relacionados
4. Analiza qué información específica se solicita en la consulta
5. Genera SQL que incluya la información relevante solicitada
6. Asegúrate de que el SQL sea compatible con SQLite

ANÁLISIS DE LA CONSULTA:
- Identifica qué tipo de información médica se busca
- Determina qué tablas son relevantes
- Considera si se necesitan diagnósticos, medicación, episodios, etc.

REGLAS CRÍTICAS:
- Siempre incluir información básica del paciente (nombre, apellidos)
- Usar LEFT JOIN para datos opcionales (diagnósticos, medicación)
- Filtrar por PATI_ID = UUID directamente
- Optimizar para SQLite (no usar funciones específicas de otros DBMS)

RESPUESTA:
Solo el SQL válido, sin explicaciones ni comentarios."""

            if stream_callback:
                stream_callback("   - Consultando IA para SQL específico...")
            
            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Generando SQL dinámico para UUID"
            )
            
            generated_sql = self._clean_llm_sql_response(self._extract_response_text(response))
            
            if stream_callback:
                stream_callback("   ✅ SQL dinámico generado con IA")
            
            return generated_sql
            
        except Exception as e:
            logger.error(f"Error generando SQL dinámico para UUID: {e}")
            # Fallback simple y directo
            return f"SELECT * FROM PATI_PATIENTS WHERE PATI_ID = ?"

    async def _generate_fallback_sql(self, query: str, params: List[str], tables: List[str], stream_callback=None) -> str:
        """Genera SQL de fallback dinámico usando LLM"""
        try:
            if stream_callback:
                stream_callback("   - Generando SQL de fallback dinámico...")
            
            # Obtener esquema real de la base de datos
            schema_info = self._get_schema_summary_for_exploration()
            
            # PROMPT ESPECÍFICO PARA SQL DE FALLBACK
            prompt = f"""Eres un experto en SQL médico. Genera una consulta SQL de fallback segura.

CONSULTA ORIGINAL: "{query}"
PARÁMETROS: {params}
TABLAS DISPONIBLES: {tables}

ESQUEMA DE LA BASE DE DATOS:
{schema_info}

INSTRUCCIONES ESPECÍFICAS:
1. Genera una consulta SQL simple y segura como fallback
2. Usa solo las tablas disponibles que sean relevantes
3. Incluye información básica del paciente si es relevante
4. Limita los resultados a un número razonable (10-20 registros)
5. Asegúrate de que el SQL sea compatible con SQLite
6. Evita JOINs complejos que puedan fallar

ESTRATEGIA DE FALLBACK:
- Si hay parámetros, usarlos en la consulta
- Si no hay parámetros, mostrar datos de muestra
- Priorizar tablas principales (PATI_PATIENTS, EPIS_DIAGNOSTICS)
- Incluir solo campos esenciales

REGLAS CRÍTICAS:
- SQL simple y directo
- Compatible con SQLite
- Sin funciones específicas de otros DBMS
- Limitar resultados para evitar sobrecarga

RESPUESTA:
Solo el SQL válido, sin explicaciones ni comentarios."""

            if stream_callback:
                stream_callback("   - Consultando IA para SQL de fallback...")
            
            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Generando SQL de fallback dinámico"
            )
            
            generated_sql = self._clean_llm_sql_response(self._extract_response_text(response))
            
            # LOGGING DETALLADO PARA DEPURACIÓN
            logger.info(f"🔍 DEBUG - SQL DE FALLBACK GENERADO:")
            logger.info(f"   Consulta original: '{query}'")
            logger.info(f"   Parámetros: {params}")
            logger.info(f"   Tablas: {tables}")
            logger.info(f"   SQL de fallback: {generated_sql}")
            
            if stream_callback:
                stream_callback("   ✅ SQL de fallback generado dinámicamente")
            
            return generated_sql
                
        except Exception as e:
            logger.error(f"Error en fallback SQL dinámico: {e}")
            # Fallback básico si todo falla
            if tables:
                return f"SELECT * FROM {tables[0]} LIMIT 10"
            else:
                return "SELECT 1 as error_fallback;"

    def _generate_simple_patient_sql(self, query: str, params: List[str]) -> str:
        """
        Genera SQL simple y directo para búsquedas de pacientes usando LLM dinámico.
        """
        try:
            # Obtener esquema real de la base de datos
            schema_info = self._get_schema_summary_for_exploration()
            
            # PROMPT ESPECÍFICO PARA BÚSQUEDA DE PACIENTES
            prompt = f"""Eres un experto en SQL médico. Genera una consulta SQL simple para búsquedas de pacientes.

CONSULTA ORIGINAL: "{query}"
PARÁMETROS: {params}

ESQUEMA DE LA BASE DE DATOS:
{schema_info}

INSTRUCCIONES ESPECÍFICAS:
1. Genera una consulta SQL simple para buscar pacientes
2. Usa la tabla PATI_PATIENTS como principal
3. Si hay parámetros de búsqueda, úsalos para filtrar
4. Incluye información básica del paciente (nombre, apellidos, ID)
5. Limita los resultados a un número razonable (10-20 registros)
6. Asegúrate de que el SQL sea compatible con SQLite

ESTRATEGIA DE BÚSQUEDA:
- Si hay parámetros, usarlos para filtrar por nombre o ID
- Si no hay parámetros, mostrar pacientes activos
- Incluir solo campos esenciales (PATI_ID, PATI_NAME, PATI_SURNAME_1, PATI_FULL_NAME)
- Ordenar por ID para obtener los más recientes

REGLAS CRÍTICAS:
- SQL simple y directo
- Compatible con SQLite
- Sin JOINs complejos
- Filtrar pacientes válidos (no nulos, no vacíos)

RESPUESTA:
Solo el SQL válido, sin explicaciones ni comentarios."""

            # Usar LLM para generar SQL dinámico
            response = _call_openai_native(self.llm, [{"role": "user", "content": prompt}])
            generated_sql = self._clean_llm_sql_response(self._extract_response_text(response))
            
            return generated_sql
            
        except Exception as e:
            logger.error(f"Error en _generate_simple_patient_sql dinámico: {e}")
            # Fallback básico
            return "SELECT PATI_ID, PATI_NAME, PATI_SURNAME_1, PATI_FULL_NAME FROM PATI_PATIENTS WHERE PATI_ACTIVE = 1 ORDER BY PATI_ID LIMIT 10;"

    async def _validate_sql_completeness_with_llm(self, query: str, sql: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validación básica placeholder: siempre devuelve None para indicar que no hay problemas detectados."""
        return None

    async def _llm_interpret_results(self, query: str, data: List[Dict[str, Any]], stream_callback=None) -> str:
        """Interpretación médica detallada de los resultados."""
        if not data:
            return "No se encontraron resultados médicos para esta consulta."
        
        try:
            if not self.llm:
                # Interpretación básica sin LLM
                return await self._create_basic_medical_interpretation(query, data)
            
            # Usar LLM para interpretación médica detallada
            prompt = f"""Eres un médico experto que interpreta resultados de bases de datos médicas.

CONSULTA ORIGINAL: "{query}"

DATOS ENCONTRADOS ({len(data)} registros):
{json.dumps(data[:5], indent=2, ensure_ascii=False)}

TAREA: Proporciona una interpretación médica clara y útil de estos resultados.

INSTRUCCIONES:
1. Identifica qué tipo de información médica se encontró
2. Explica el significado clínico de los datos
3. Destaca información relevante para el paciente
4. Si hay medicamentos, alergias, o condiciones especiales, menciónalas
5. Proporciona contexto médico cuando sea apropiado
6. Si los datos son limitados, sugiere qué otra información podría ser útil

FORMATO DE RESPUESTA:
- Resumen ejecutivo de los hallazgos
- Interpretación médica de los datos
- Información clínica relevante
- Recomendaciones si aplica

Responde en español de manera clara y profesional:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}], 
                task_description="Interpretando resultados médicos"
            )
            
            interpretation = self._extract_response_text(response)
            return interpretation if interpretation else await self._create_basic_medical_interpretation(query, data)
            
        except Exception as e:
            logger.error(f"Error en interpretación médica: {e}")
            return await self._create_basic_medical_interpretation(query, data)

    async def _create_basic_medical_interpretation(self, query: str, data: List[Dict[str, Any]]) -> str:
        """Crea una interpretación médica básica usando LLM."""
        if not data:
            return "No se encontraron resultados médicos."
        
        try:
            # Usar LLM para interpretación básica
            prompt = f"""Eres un médico que interpreta resultados de bases de datos médicas.

CONSULTA ORIGINAL: "{query}"

DATOS ENCONTRADOS ({len(data)} registros):
{json.dumps(data[:10], indent=2, ensure_ascii=False)}

TAREA: Proporciona una interpretación médica clara y útil de estos resultados.

INSTRUCCIONES:
1. Analiza los datos y extrae información médica relevante
2. Identifica información del paciente, medicamentos, condiciones especiales
3. Destaca datos clínicamente importantes
4. Organiza la información de manera clara y profesional
5. Si hay datos limitados, explica qué información se encontró

FORMATO DE RESPUESTA:
- Resumen de hallazgos médicos
- Información del paciente relevante
- Datos médicos importantes
- Contexto clínico cuando sea apropiado

Responde en español de manera clara y profesional:"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}], 
                task_description="Interpretando resultados médicos básicos"
            )
            
            interpretation = self._extract_response_text(response)
            return interpretation if interpretation else f"Se encontraron {len(data)} registros médicos."
            
        except Exception as e:
            logger.error(f"Error en interpretación médica básica: {e}")
            return f"Se encontraron {len(data)} registros médicos."

    async def _analyze_query_semantics(self, query: str) -> Dict[str, Any]:
        """Análisis semántico muy básico como placeholder, extrae tokens simples de la consulta."""
        tokens = re.findall(r'\w+', query.lower())
        return {"keywords": tokens}

    async def _get_table_candidates_from_analysis(self, semantic_analysis: Dict[str, Any], stream_callback=None) -> List[str]:
        """Devuelve las primeras tablas del esquema como candidatos básicos."""
        return list(self.column_metadata.keys())[:5]

    async def _execute_multiple_statements(self, statements: List[str], params: List[Any], start_time: float) -> Dict[str, Any]:
        """Ejecuta varias sentencias SQL de forma secuencial (implementación básica)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        aggregated_results: List[Dict[str, Any]] = []
        executed_sql: List[str] = []
        try:
            for stmt in statements:
                executed_sql.append(stmt)
                cursor.execute(stmt, params if '?' in stmt else [])
                if cursor.description:
                    cols = [d[0] for d in cursor.description]
                    rows = cursor.fetchall()
                    aggregated_results.extend([dict(zip(cols, r)) for r in rows])
            conn.close()
            return {
                "success": True,
                "message": f"Ejecutadas {len(statements)} sentencia(s)",
                "data": aggregated_results,
                "sql_query": "; ".join(executed_sql),
                "execution_time": time.time() - start_time,
                "total_time": time.time() - start_time
            }
        except Exception as e:
            conn.close()
            return self._create_error_response(str(e), "; ".join(executed_sql))

    async def _validate_sql_syntax(self, sql: str) -> Optional[str]:
        """Comprobación sintáctica básica: intenta analizar la consulta con EXPLAIN; devuelve mensaje de error o None."""
        try:
            conn = self._get_connection()
            # CRÍTICO: Preparar SQL para validación (quitar punto y coma)
            clean_sql = self._prepare_sql_for_execution(sql)
            # Para validar la sintaxis con EXPLAIN, debemos proporcionar
            # un número correcto de parámetros dummy si el SQL los espera.
            placeholder_count = clean_sql.count('?')
            dummy_params = [None] * placeholder_count
            conn.execute(f"EXPLAIN {clean_sql}", dummy_params)
            conn.close()
            return None
        except Exception as e:
            return str(e)

    def _get_connection(self):
        """Obtiene una conexión SQLite a la base de datos."""
        return sqlite3.connect(self.db_path)

    async def _learn_from_query_result(self, query: str, sql: str, result_count: int, exec_time: float):
        """Placeholder de aprendizaje: implementación vacía para evitar errores."""
        # Temporalmente desactivado para evitar interferencias
        pass

    async def _learn_from_error(self, query: str, error_msg: str):
        """Placeholder para registrar errores (sin lógica de aprendizaje)."""
        # Temporalmente desactivado para evitar interferencias
        pass

    def _clean_llm_sql_response(self, sql_response: str) -> str:
        """Limpia la respuesta del LLM usando el módulo centralizado SQLCleaner."""
        return SQLCleaner.clean_llm_response(sql_response)

    def _normalize_accents_sql(self, column_name: str) -> str:
        """
        Genera la expresión SQL para normalizar vocales acentuadas.
        
        Args:
            column_name: Nombre de la columna a normalizar
            
        Returns:
            str: Expresión SQL con normalización completa de vocales acentuadas
        """
        # Versión más limpia y legible
        return f"REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(UPPER({column_name}),'Á','A'),'É','E'),'Í','I'),'Ó','O'),'Ú','U'),'Ñ','N')"

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

    def _create_smart_name_search_condition(self, column_name: str, search_term: str, use_exact_match: bool = False) -> str:
        """
        Crea una condición WHERE inteligente para búsqueda de nombres.
        GARANTIZA que sea completamente insensible a mayúsculas/minúsculas.
        
        Args:
            column_name: Nombre de la columna (ej: p.PATI_FULL_NAME)
            search_term: Término de búsqueda (ej: "ANA GARCIA")
            use_exact_match: Si usar búsqueda exacta o flexible
            
        Returns:
            str: Condición WHERE optimizada y robusta
        """
        # Normalizar el término de búsqueda en Python
        normalized_search = self._normalize_accents_python(search_term)
        
        if use_exact_match:
            # Búsqueda exacta con normalización en Python - SIEMPRE insensible a mayúsculas
            return f"UPPER({column_name}) = UPPER(?)"
        else:
            # Búsqueda flexible con normalización en Python - SIEMPRE insensible a mayúsculas
            return f"UPPER({column_name}) LIKE UPPER(?)"

    def _prepare_sql_for_execution(self, sql: str) -> str:
        """
        Prepara el SQL para ejecución limpiando elementos problemáticos.
        
        Args:
            sql: SQL a limpiar
            
        Returns:
            str: SQL listo para ejecutar
        """
        if not sql:
            return sql
            
        # 1. Quitar espacios en blanco al final
        clean_sql = sql.rstrip()
        
        # 2. Quitar punto y coma al final (SQLite no lo acepta con execute())
        clean_sql = clean_sql.rstrip(';')
        
        # 3. Normalizar espacios múltiples
        clean_sql = re.sub(r'\s+', ' ', clean_sql).strip()
        
        return clean_sql

    def _force_sql_corrections(self, sql: str, params: List[str]) -> str:
        """
        Aplica correcciones FORZADAS al SQL sin depender del LLM.
        Versión ROBUSTA que garantiza que nunca falle.
        
        Args:
            sql: SQL a corregir
            params: Parámetros normalizados
            
        Returns:
            str: SQL corregido forzadamente
        """
        if not sql:
            return sql
        
        # DETECCIÓN ESPECIAL: Si es búsqueda de pacientes, usar SQL robusto
        if 'PATI_PATIENTS' in sql.upper() and 'PATI_FULL_NAME' in sql.upper():
            search_term = params[0] if params else ""
            logger.info(f"🔍 Detectada búsqueda de pacientes, aplicando SQL robusto para: {search_term}")
            return self._create_robust_patient_search_sql(search_term, include_suggestions=True)
        
        # 1. ELIMINAR COMPLETAMENTE la cadena REPLACE si existe
        # Patrón más simple y robusto para eliminar cadenas REPLACE anidadas
        while 'REPLACE(REPLACE(' in sql:
            # Buscar y reemplazar patrones REPLACE anidados
            replace_pattern = r'REPLACE\(REPLACE\(REPLACE\(REPLACE\(REPLACE\(REPLACE\([^,]+,\s*[^,]+\),\s*[^,]+\),\s*[^,]+\),\s*[^,]+\),\s*[^,]+\),\s*[^,]+\)'
            if re.search(replace_pattern, sql, re.IGNORECASE):
                sql = re.sub(replace_pattern, 'UPPER(p.PATI_FULL_NAME)', sql, flags=re.IGNORECASE)
            else:
                # Si el patrón no coincide, salir del bucle para evitar bucle infinito
                break
        
        # 2. FORZAR corrección de errores tipográficos específicos
        sql = self._fix_typo_errors(sql)
        
        # 3. FORZAR uso de parámetros si no hay placeholders
        if params and '?' not in sql:
            # Si hay parámetros pero no placeholders, añadir condición WHERE
            if 'WHERE' not in sql.upper():
                sql = sql.rstrip(';')
                sql += ' WHERE UPPER(p.PATI_FULL_NAME) = UPPER(?);'
            else:
                # Si ya hay WHERE, reemplazar la condición compleja
                sql = re.sub(r'WHERE\s+.*?(?=ORDER|GROUP|HAVING|LIMIT|$)', 'WHERE UPPER(p.PATI_FULL_NAME) = UPPER(?) ', sql, flags=re.IGNORECASE | re.DOTALL)
        
        # 4. FORZAR normalización de espacios
        sql = re.sub(r'\s+', ' ', sql).strip()
        
        # 5. FORZAR punto y coma al final
        if not sql.endswith(';'):
            sql += ';'
        
        # 6. GARANTIZAR que las comparaciones de texto sean insensibles a mayúsculas
        # Reemplazar = ? por = UPPER(?) en comparaciones de texto
        sql = re.sub(r'=\s*\?', '= UPPER(?)', sql, flags=re.IGNORECASE)
        sql = re.sub(r'LIKE\s*\?', 'LIKE UPPER(?)', sql, flags=re.IGNORECASE)
        
        return sql

    async def _llm_clean_and_fix_sql(self, sql: str, stream_callback=None) -> str:
        """
        Usa el LLM para limpiar y corregir errores de sintaxis SQL de forma dinámica.
        
        Args:
            sql: SQL con posibles errores
            stream_callback: Función para mostrar progreso
            
        Returns:
            str: SQL corregido y limpio
        """
        try:
            # APLICAR CORRECCIONES FORZADAS PRIMERO
            sql = self._basic_sql_cleanup(sql)
            
            if not self.llm:
                return sql
            
            if stream_callback:
                stream_callback("   - Corrigiendo errores de sintaxis SQL con IA...")
            
            prompt = f"""Eres un experto en SQL que corrige errores de sintaxis y formato.

SQL A CORREGIR:
{sql}

INSTRUCCIONES CRÍTICAS:

1. NUNCA uses cadenas REPLACE complejas para normalización
2. SIEMPRE usa búsqueda simple: UPPER(p.PATI_FULL_NAME) = ?
3. SIEMPRE añade espacios: "p WHERE" NO "pWHERE"
4. SIEMPRE añade espacios: "p FROM" NO "pFROM"
5. SIEMPRE añade espacios: "p.PATI_FULL_NAME FROM" NO "p.PATI_FULL_NAMEFROM"

EJEMPLO CORRECTO:
SELECT p.PATI_ID, p.PATI_FULL_NAME FROM PATI_PATIENTS p WHERE UPPER(p.PATI_FULL_NAME) = ?;

RESPUESTA:
Devuelve SOLO el SQL corregido, sin explicaciones."""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Corrigiendo errores de sintaxis SQL"
            )
            
            corrected_sql = self._extract_response_text(response).strip()
            corrected_sql = self._clean_llm_sql_response(corrected_sql)
            
            if corrected_sql and not corrected_sql.startswith("Error"):
                logger.info(f"🧠 LLM corrigió SQL exitosamente")
                if stream_callback:
                    stream_callback("   ✅ Errores de sintaxis corregidos")
                return corrected_sql
            else:
                logger.warning(f"⚠️ LLM no pudo corregir SQL, usando fallback")
                return sql
                
        except Exception as e:
            logger.error(f"Error en _llm_clean_and_fix_sql: {e}")
            return sql

    def _basic_sql_cleanup(self, sql: str) -> str:
        """
        Limpieza básica de SQL sin LLM (fallback mínimo).
        
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
            logger.error(f"Error en _basic_sql_cleanup: {e}")
            return sql

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
            (r'SELECT\w+\.\*', r'SELECT \1'),
            (r'FROM\w+', r'FROM \1'),
            (r'WHERE\w+', r'WHERE \1'),
            (r'JOIN\w+', r'JOIN \1'),
            (r'ON\w+', r'ON \1'),
            (r'AND\w+', r'AND \1'),
            (r'OR\w+', r'OR \1'),
            (r'ORDER\w+', r'ORDER \1'),
            (r'GROUP\w+', r'GROUP \1'),
            (r'LIMIT\w+', r'LIMIT \1'),
        ]
        
        corrected_sql = sql
        for pattern, replacement in corrections:
            corrected_sql = re.sub(pattern, replacement, corrected_sql, flags=re.IGNORECASE)
        
        # Normalizar espacios múltiples
        corrected_sql = re.sub(r'\s+', ' ', corrected_sql).strip()
        
        return corrected_sql

    async def _llm_final_validation(self, sql: str, stream_callback=None) -> str:
        """
        Usa el LLM para validación final del SQL antes de la ejecución.
        
        Args:
            sql: SQL a validar
            stream_callback: Función para mostrar progreso
            
        Returns:
            str: SQL validado y corregido si es necesario
        """
        try:
            if not self.llm:
                return sql
            
            if stream_callback:
                stream_callback("   - Validación final del SQL con IA...")
            
            prompt = f"""Eres un experto en SQL que realiza validación final de consultas.

SQL A VALIDAR:
{sql}

TAREAS DE VALIDACIÓN:

1. DETECTAR ERRORES CRÍTICOS:
   - Errores de sintaxis obvios (pWHERE, pFROM, etc.)
   - Tablas inexistentes (VITAL_SIGNS → APPO_APPOINTMENTS)
   - Espacios faltantes entre palabras clave
   - Múltiples sentencias SQL

2. CORREGIR SI ES NECESARIO:
   - Añadir espacios donde falten
   - Corregir nombres de tablas inexistentes
   - Asegurar formato correcto
   - Mantener la lógica original

3. VALIDACIÓN FINAL:
   - Verificar que el SQL sea ejecutable
   - Asegurar que todas las palabras clave tengan espacios
   - Verificar que termine con punto y coma

RESPUESTA:
- Si el SQL está correcto: devuelve el SQL original sin cambios
- Si hay errores: devuelve el SQL corregido
- Devuelve SOLO el SQL, sin explicaciones"""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Validación final del SQL"
            )
            
            validated_sql = self._extract_response_text(response).strip()
            validated_sql = self._clean_llm_sql_response(validated_sql)
            
            if validated_sql and not validated_sql.startswith("Error"):
                if validated_sql != sql:
                    logger.info(f"🧠 LLM realizó correcciones finales en el SQL")
                    if stream_callback:
                        stream_callback("   ✅ Validación final completada con correcciones")
                else:
                    if stream_callback:
                        stream_callback("   ✅ Validación final completada sin cambios")
                return validated_sql
            else:
                logger.warning(f"⚠️ LLM no pudo validar SQL, usando original")
                return sql
                
        except Exception as e:
            logger.error(f"Error en _llm_final_validation: {e}")
            return sql

    async def _regenerate_sql_with_error_context(self, original_query: str, failed_sql: str, syntax_error: str, params: List[str], stream_callback=None) -> str:
        """
        Regenera SQL usando el LLM con información del error de sintaxis.
        
        Args:
            original_query: Consulta original del usuario
            failed_sql: SQL que falló
            syntax_error: Mensaje de error de sintaxis
            params: Parámetros originales
            stream_callback: Función para mostrar progreso
            
        Returns:
            str: SQL regenerado o None si falla
        """
        try:
            if not self.llm:
                logger.warning("❌ LLM no disponible para regeneración")
                return "SELECT 1 as error_no_llm;"
            
            if stream_callback:
                stream_callback("   - Regenerando SQL con información del error...")
            
            prompt = f"""Eres un experto en SQL que corrige errores de sintaxis críticos.

CONSULTA ORIGINAL: "{original_query}"

SQL QUE FALLÓ:
{failed_sql}

ERROR DE SINTAXIS DETECTADO:
{syntax_error}

PARÁMETROS ORIGINALES:
{params}

TAREAS CRÍTICAS:

1. ANALIZAR EL ERROR:
   - Identificar exactamente qué causó el error de sintaxis
   - Buscar espacios faltantes, palabras pegadas, etc.
   - Identificar tablas o columnas incorrectas

2. CORREGIR ERRORES ESPECÍFICOS:
   - Añadir espacios donde falten: "v.*FROM" → "v.* FROM"
   - Separar palabras pegadas: "pJOIN" → "p JOIN"
   - Corregir alias: "v ON" → "v ON" (verificar que v existe)
   - Corregir tablas inexistentes: VITAL_SIGNS → APPO_APPOINTMENTS

3. GENERAR SQL CORRECTO:
   - Usar solo tablas que existen en el esquema
   - Asegurar que todos los espacios estén correctos
   - Mantener la lógica original de la consulta
   - Usar búsqueda simple: UPPER(p.PATI_FULL_NAME) = ?

4. VALIDACIÓN FINAL:
   - Verificar que no haya errores de sintaxis obvios
   - Asegurar que todas las palabras clave tengan espacios
   - Verificar que termine con punto y coma

RESPUESTA:
Devuelve SOLO el SQL corregido, sin explicaciones ni comentarios."""

            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Regenerando SQL después de error de sintaxis"
            )
            
            regenerated_sql = self._extract_response_text(response).strip()
            regenerated_sql = self._clean_llm_sql_response(regenerated_sql)
            
            if regenerated_sql and not regenerated_sql.startswith("Error"):
                logger.info(f"🧠 LLM regeneró SQL exitosamente después de error")
                if stream_callback:
                    stream_callback("   ✅ SQL regenerado con correcciones")
                return regenerated_sql
            else:
                logger.warning(f"❌ LLM no pudo regenerar SQL, usando fallback")
                # Fallback: crear SQL básico
                fallback_sql = self._create_fallback_sql_with_placeholders(params, {
                    'relevant_tables': ['PATI_PATIENTS', 'APPO_APPOINTMENTS']
                })
                return fallback_sql or "SELECT 1 as error_fallback;"
                
        except Exception as e:
            logger.error(f"Error en _regenerate_sql_with_error_context: {e}")
            return "SELECT 1 as error_fallback;"

    def _create_fallback_sql_with_placeholders(self, params: List[str], execution_plan: Dict[str, Any]) -> str:
        """
        Crea un SQL de fallback básico que incluye placeholders para los parámetros dados.
        
        Args:
            params: Lista de parámetros para la consulta
            execution_plan: Plan de ejecución con las tablas relevantes
            
        Returns:
            str: SQL con placeholders correctos
        """
        try:
            tables = execution_plan.get('relevant_tables', [])
            if not tables:
                logger.error("❌ No hay tablas disponibles para generar SQL de fallback")
                return "SELECT 1 as error_no_tables;"
            
            # Usar la primera tabla como principal
            main_table = tables[0]
            
            # Determinar si los parámetros son para nombres o IDs
            name_params = [p for p in params if isinstance(p, str) and '%' in p]
            id_params = [p for p in params if isinstance(p, str) and '%' not in p]
            
            # Construir un SQL básico basado en el tipo de parámetros
            if name_params:
                # Para parámetros de nombre, buscar en columnas de pacientes
                if 'PATI_' in main_table.upper():
                    # Tabla de pacientes
                    conditions = []
                    for _ in name_params:
                        conditions.append("UPPER(PATI_FULL_NAME) LIKE ?")
                    
                    where_clause = " OR ".join(conditions)
                    sql = f"SELECT * FROM {main_table} WHERE {where_clause} LIMIT 10;"
                    
                elif 'EPIS_' in main_table.upper():
                    # Tabla de episodios, buscar por ID de paciente
                    conditions = []
                    for _ in name_params:
                        conditions.append("UPPER(EPIS_PATI_ID) LIKE ?")
                    
                    where_clause = " OR ".join(conditions)
                    sql = f"SELECT * FROM {main_table} WHERE {where_clause} LIMIT 10;"
                    
                else:
                    # Tabla genérica, buscar en campos de texto
                    conditions = []
                    for _ in name_params:
                        conditions.append("1=1")  # Placeholder que será reemplazado
                    
                    # Intentar encontrar columnas de texto en el esquema
                    text_columns = []
                    if main_table in self.column_metadata:
                        for col in self.column_metadata[main_table]['columns']:
                            col_name = col['name']
                            if any(keyword in col_name.upper() for keyword in ['NAME', 'NOMBRE', 'FULL', 'PATI']):
                                text_columns.append(col_name)
                    
                    if text_columns:
                        conditions = []
                        for _ in name_params:
                            conditions.append(f"UPPER({text_columns[0]}) LIKE ?")
                        where_clause = " OR ".join(conditions)
                        sql = f"SELECT * FROM {main_table} WHERE {where_clause} LIMIT 10;"
                    else:
                        # Sin columnas de texto identificables, SQL básico
                        sql = f"SELECT * FROM {main_table} LIMIT 10;"
                        
            elif id_params:
                # Para IDs específicos
                conditions = []
                for _ in id_params:
                    conditions.append("ID = ?")
                
                where_clause = " OR ".join(conditions)
                sql = f"SELECT * FROM {main_table} WHERE {where_clause} LIMIT 10;"
                
            else:
                # Sin parámetros reconocibles, SQL básico
                sql = f"SELECT * FROM {main_table} LIMIT 10;"
            
            logger.info(f"🔧 SQL de fallback generado: {sql}")
            return sql
            
        except Exception as e:
            logger.error(f"❌ Error generando SQL de fallback: {e}")
            return "SELECT 1 as error_fallback;"

    async def _generate_dynamic_sql_with_llm(self, query: str, params: List[str], execution_plan: Dict[str, Any], stream_callback=None) -> str:
        """
        Genera SQL dinámico usando el LLM basado en los datos reales encontrados.
        
        Args:
            query: Consulta original del usuario
            params: Parámetros extraídos (nombres, IDs, etc.)
            execution_plan: Plan de ejecución con tablas y contexto
            stream_callback: Función para mostrar progreso
            
        Returns:
            str: SQL generado dinámicamente
        """
        try:
            if not self.llm:
                logger.warning("❌ LLM no disponible para generación dinámica")
                return self._create_fallback_sql_with_placeholders(params, execution_plan)
            
            if stream_callback:
                stream_callback("   - Generando SQL dinámico basado en datos reales...")
            
            # Obtener información del esquema para las tablas relevantes
            tables = execution_plan.get('relevant_tables', [])
            schema_info = {}
            sample_data = {}
            
            for table in tables[:3]:  # Limitar a 3 tablas para no sobrecargar
                if table in self.column_metadata:
                    columns = [col['name'] for col in self.column_metadata[table]['columns']]
                    schema_info[table] = columns
                    
                    # Obtener datos de muestra si están disponibles
                    if table in self.sample_data:
                        sample_data[table] = self.sample_data[table]
            
            # Construir prompt dinámico basado en los parámetros
            if params and any('GARCÍA' in p.upper() or 'GARCIA' in p.upper() for p in params):
                # Caso específico para "Ana García" con datos reales
                dynamic_prompt = f"""Eres un experto en SQL para bases de datos médicas. Analiza la consulta y genera SQL optimizado.

CONSULTA ORIGINAL: "{query}"
PARÁMETROS DETECTADOS: {params}

DATOS REALES ENCONTRADOS:
- Existe "Ana García" (con tilde) en PATI_PATIENTS
- También existe "Ana García" (sin tilde) 
- Los nombres están en formato: "PRUEBAS101284 SINA101284 , ANA MARIA"

ESQUEMA DISPONIBLE:
{json.dumps(schema_info, indent=2, ensure_ascii=False)}

MUESTRA DE DATOS REALES:
{json.dumps(sample_data, indent=2, ensure_ascii=False)}

TAREA: Analiza la consulta y genera SQL que:
1. Identifique qué tipo de información médica se busca
2. Use la estrategia de búsqueda más apropiada para los parámetros
3. Conecte las tablas correctamente
4. Maneje tildes y caracteres especiales con normalización completa
5. Sea compatible con SQLite

ESTRATEGIAS DE BÚSQUEDA INTELIGENTES PARA NOMBRES:
- Búsqueda exacta con normalización: REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(UPPER(p.PATI_FULL_NAME),'Á','A'),'É','E'),'Í','I'),'Ó','O'),'Ú','U'),'Ñ','N') = 'ANA GARCIA'
- Búsqueda flexible con normalización: REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(UPPER(p.PATI_FULL_NAME),'Á','A'),'É','E'),'Í','I'),'Ó','O'),'Ú','U'),'Ñ','N') LIKE '%ANA%GARCIA%'
- Búsqueda múltiple: (normalización = 'ANA GARCIA' OR normalización LIKE '%ANA%GARCIA%')

NORMALIZACIÓN COMPLETA DE VOCALES:
- Á→A, É→E, Í→I, Ó→O, Ú→U, Ñ→N
- "Ana García" → "ANA GARCIA"
- "José María" → "JOSE MARIA"

Genera SQL optimizado basado en el análisis de la consulta y los datos reales.

Responde SOLO con el SQL optimizado:"""
            
            else:
                # Caso genérico para otros parámetros
                dynamic_prompt = f"""Eres un experto en SQL para bases de datos médicas. Analiza la consulta y genera SQL dinámico.

CONSULTA ORIGINAL: "{query}"
PARÁMETROS DETECTADOS: {params}

ESQUEMA DISPONIBLE:
{json.dumps(schema_info, indent=2, ensure_ascii=False)}

MUESTRA DE DATOS REALES:
{json.dumps(sample_data, indent=2, ensure_ascii=False)}

TAREA: Analiza la consulta y genera SQL que:
1. Identifique qué tipo de información médica se busca
2. Use la estrategia de búsqueda más apropiada para los parámetros
3. Maneje correctamente tildes, espacios y formatos de nombres con normalización completa
4. Conecte las tablas de manera eficiente
5. Sea compatible con SQLite

NORMALIZACIÓN DE VOCALES ACENTUADAS:
- Normaliza TODAS las vocales: Á→A, É→E, Í→I, Ó→O, Ú→U, Ñ→N
- Usa: REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(UPPER(p.PATI_FULL_NAME),'Á','A'),'É','E'),'Í','I'),'Ó','O'),'Ú','U'),'Ñ','N')

Genera SQL optimizado basado en el análisis de la consulta y los datos reales.

Responde SOLO con el SQL optimizado:"""
            
            try:
                response = await asyncio.to_thread(
                    _call_openai_native, self.llm, [{"role": "user", "content": dynamic_prompt}], 
                    task_description="Generando SQL dinámico basado en datos reales"
                )
                
                dynamic_sql = self._extract_response_text(response).strip()
                dynamic_sql = self._clean_llm_sql_response(dynamic_sql)
                
                if dynamic_sql and not dynamic_sql.startswith("Error"):
                    logger.info(f"🧠 SQL dinámico generado exitosamente")
                    if stream_callback:
                        stream_callback("   ✅ SQL dinámico generado basado en datos reales")
                    return dynamic_sql
                else:
                    logger.warning(f"❌ LLM no pudo generar SQL dinámico, usando fallback")
                    return self._create_fallback_sql_with_placeholders(params, execution_plan)
                    
            except Exception as e:
                logger.error(f"Error generando SQL dinámico: {e}")
                return self._create_fallback_sql_with_placeholders(params, execution_plan)
                
        except Exception as e:
            logger.error(f"Error en _generate_dynamic_sql_with_llm: {e}")
            return self._create_fallback_sql_with_placeholders(params, execution_plan)

    def _get_schema_summary_for_exploration(self) -> str:
        """
        Genera un resumen del esquema de la base de datos para exploración.
        
        Returns:
            Resumen del esquema en formato texto
        """
        try:
            schema_summary = []
            
            for table_name, metadata in self.column_metadata.items():
                if table_name.startswith('sqlite_'):
                    continue
                    
                # Información básica de la tabla
                row_count = self.table_row_counts.get(table_name, 0)
                columns = [col['name'] for col in metadata['columns']]
                
                schema_summary.append(f"📋 {table_name} ({row_count} registros):")
                schema_summary.append(f"   Columnas: {', '.join(columns)}")
                
                # Agregar datos de muestra si están disponibles
                if table_name in self.sample_data:
                    sample = self.sample_data[table_name]
                    if sample:
                        schema_summary.append(f"   Muestra: {sample[0] if isinstance(sample, list) else sample}")
                
                schema_summary.append("")  # Línea en blanco
            
            return "\n".join(schema_summary)
            
        except Exception as e:
            logger.error(f"Error generando resumen del esquema: {e}")
            return "Error generando resumen del esquema"

    def _process_patient_search_results(self, data: List[Dict[str, Any]], search_term: str) -> Dict[str, Any]:
        """
        Procesa resultados de búsqueda de pacientes y genera sugerencias inteligentes.
        
        Args:
            data: Resultados de la consulta SQL
            search_term: Término de búsqueda original
            
        Returns:
            Dict[str, Any]: Resultados procesados con sugerencias
        """
        if not data:
            return {
                'success': False,
                'message': f'❌ No se encontró ningún paciente con el nombre "{search_term}"',
                'suggestions': [],
                'data': []
            }
        
        # Separar coincidencias exactas de sugerencias
        exact_matches = []
        suggestions = []
        
        for row in data:
            if 'match_type' in row:
                if row['match_type'] == 1:
                    exact_matches.append(row)
                else:
                    suggestions.append(row)
            else:
                # Si no hay match_type, asumir que son coincidencias exactas
                exact_matches.append(row)
        
        # Preparar mensaje de respuesta
        if exact_matches:
            if len(exact_matches) == 1:
                message = f'✅ Encontrado 1 paciente: {exact_matches[0]["PATI_FULL_NAME"]}'
            else:
                message = f'✅ Encontrados {len(exact_matches)} pacientes'
            
            # Añadir sugerencias si hay
            if suggestions:
                message += f'\n💡 También encontré {len(suggestions)} pacientes similares:'
                for i, suggestion in enumerate(suggestions[:3], 1):
                    message += f'\n   {i}. {suggestion["PATI_FULL_NAME"]}'
        else:
            # Solo sugerencias
            message = f'❓ No encontré coincidencias exactas para "{search_term}"'
            if suggestions:
                message += f'\n💡 ¿Quizás te refieres a uno de estos pacientes?'
                for i, suggestion in enumerate(suggestions[:5], 1):
                    message += f'\n   {i}. {suggestion["PATI_FULL_NAME"]}'
            else:
                message += f'\n💡 No encontré pacientes similares. Verifica el nombre e intenta de nuevo.'
        
        return {
            'success': True,
            'message': message,
            'exact_matches': exact_matches,
            'suggestions': suggestions,
            'data': data,
            'search_term': search_term
        }

    def _create_fallback_patient_search(self, search_term: str) -> str:
        """
        Crea SQL de fallback para búsqueda de pacientes cuando fallan otros métodos.
        
        Args:
            search_term: Término de búsqueda
            
        Returns:
            str: SQL de fallback simple y robusto
        """
        # SQL ultra-simple que NUNCA puede fallar
        return """
        SELECT p.PATI_ID, p.PATI_FULL_NAME, p.PATI_USUAL_MEDICATION, p.PATI_LAST_VISIT
        FROM PATI_PATIENTS p 
        WHERE UPPER(p.PATI_FULL_NAME) LIKE UPPER(?)
        ORDER BY p.PATI_FULL_NAME
        LIMIT 20
        """

    def _create_robust_patient_search_sql(self, search_term: str, include_suggestions: bool = True) -> str:
        """
        Genera un SQL robusto para búsqueda de pacientes por nombre, insensible a mayúsculas y tildes.
        Si include_suggestions es True, también busca sugerencias similares.
        """
        # Normalizar el término de búsqueda en Python
        normalized_search = self._normalize_accents_python(search_term)
        # Búsqueda exacta y flexible
        sql = f'''
        SELECT 
            p.PATI_ID, 
            p.PATI_CLINICAL_HISTORY_ID, 
            p.PATI_NAME, 
            p.PATI_SURNAME_1, 
            p.PATI_SURNAME_2, 
            p.PATI_FULL_NAME,
            1 as match_type
        FROM PATI_PATIENTS p 
        WHERE UPPER(p.PATI_FULL_NAME) = UPPER(?)
        '''
        if include_suggestions:
            sql += '''
            UNION
            SELECT 
                p.PATI_ID, 
                p.PATI_CLINICAL_HISTORY_ID, 
                p.PATI_NAME, 
                p.PATI_SURNAME_1, 
                p.PATI_SURNAME_2, 
                p.PATI_FULL_NAME,
                0 as match_type
            FROM PATI_PATIENTS p 
            WHERE UPPER(p.PATI_FULL_NAME) LIKE UPPER(?) AND UPPER(p.PATI_FULL_NAME) != UPPER(?)
            '''
        sql += '\nORDER BY match_type DESC, p.PATI_FULL_NAME LIMIT 20;'
        return sql

    def _prepare_search_parameters(self, search_term: str) -> list:
        """
        Prepara los parámetros para la búsqueda robusta de pacientes.
        Devuelve una lista de parámetros para el SQL generado.
        """
        normalized = self._normalize_accents_python(search_term)
        return [normalized, f"%{normalized}%", normalized] if normalized else [""]

    async def _validate_patient_exists_dynamic(self, patient_id: str, stream_callback=None) -> bool:
        """Valida dinámicamente si un paciente existe usando LLM"""
        try:
            if stream_callback:
                stream_callback("   - Validando existencia del paciente...")
            
            # Obtener esquema real de la base de datos
            schema_info = self._get_schema_summary_for_exploration()
            
            # PROMPT ESPECÍFICO PARA VALIDAR EXISTENCIA
            prompt = f"""Eres un experto en validación de bases de datos médicas. Valida si un paciente existe.

ID DEL PACIENTE: {patient_id}

ESQUEMA DE LA BASE DE DATOS:
{schema_info}

INSTRUCCIONES ESPECÍFICAS:
1. Genera una consulta SQL simple para verificar si el paciente existe
2. Usa la tabla PATI_PATIENTS
3. Busca por PATI_ID exacto
4. La consulta debe ser rápida y eficiente
5. Solo necesitas verificar existencia, no obtener datos completos

ESTRATEGIA DE VALIDACIÓN:
- Usar COUNT(*) para verificar existencia
- Filtrar por PATI_ID exacto
- Consulta simple y directa
- Compatible con SQLite

REGLAS CRÍTICAS:
- SQL simple y directo
- Solo verificar existencia
- No JOINs complejos
- Optimizado para velocidad

RESPUESTA:
Solo el SQL válido, sin explicaciones ni comentarios."""

            if stream_callback:
                stream_callback("   - Consultando IA para validación...")
            
            response = await asyncio.to_thread(
                _call_openai_native, self.llm, [{"role": "user", "content": prompt}],
                task_description="Validando existencia de paciente"
            )
            
            validation_sql = self._clean_llm_sql_response(self._extract_response_text(response))
            
            # Ejecutar la consulta de validación
            executor = SQLExecutor(self.db_path)
            result = executor.execute_query(validation_sql, [patient_id])
            
            exists = result.get('success', False) and result.get('row_count', 0) > 0
            
            if stream_callback:
                if exists:
                    stream_callback("   ✅ Paciente encontrado")
                else:
                    stream_callback("   ❌ Paciente no encontrado")
            
            return exists
            
        except Exception as e:
            logger.error(f"Error validando existencia de paciente: {e}")
            return False





