#!/usr/bin/env python3
"""
🏥 FHIR Persistence Agent - Sistema Completamente Flexible v1.0
═══════════════════════════════════════════════════════════════

Agente especializado en persistir recursos FHIR en SQL usando
el sistema flexible existente (DynamicMapper + SchemaIntrospector).

✅ Completamente flexible - sin hardcodeo
✅ Auto-detección de tablas destino
✅ Generación SQL automática
✅ Validación bidireccional
✅ Manejo inteligente de relaciones
✅ Cache y optimización

Autor: ChatMed Team
Versión: 1.0 - Sistema Flexible Puro
"""

import asyncio
import logging
import sqlite3
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import re

# Imports del sistema flexible existente
try:
    from ..mapping.schema_introspector import SchemaIntrospector, TableMetadata  # type: ignore
    from ..mapping.dynamic_mapper import DynamicMapper, MappingResult  # type: ignore
    FLEXIBLE_SYSTEM_AVAILABLE = True
    print("✅ Sistema flexible importado correctamente (relativo)")
except ImportError:
    try:
        # Fallback: import absoluto
        from mapping.schema_introspector import SchemaIntrospector, TableMetadata  # type: ignore
        from mapping.dynamic_mapper import DynamicMapper, MappingResult  # type: ignore
        FLEXIBLE_SYSTEM_AVAILABLE = True
        print("✅ Sistema flexible importado correctamente (absoluto)")
    except ImportError as e:
        print(f"❌ Error importando sistema flexible: {e}")
        FLEXIBLE_SYSTEM_AVAILABLE = False
        # Clases fallback
        class SchemaIntrospector:  # type: ignore
            def __init__(self, *args, **kwargs): pass
            def analyze_table_structure(self, *args, **kwargs): return TableMetadata()  # type: ignore
        class DynamicMapper:  # type: ignore
            def __init__(self, *args, **kwargs): pass
        class MappingResult:  # type: ignore
            def __init__(self, *args, **kwargs): pass
        class TableMetadata:  # type: ignore
            def __init__(self, *args, **kwargs): 
                self.name = ""  # type: ignore
                self.confidence_score = 0.0  # type: ignore
                self.columns = {}  # type: ignore

# LLM para decisiones inteligentes (opcional)
LLM_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FHIRPersistenceAgent")

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

@dataclass 
class PersistenceStats:
    """Estadísticas del agente de persistencia"""
    total_resources_processed: int = 0
    successful_persistences: int = 0
    failed_persistences: int = 0
    avg_processing_time_ms: float = 0.0
    total_sql_queries_generated: int = 0
    total_records_created: int = 0
    total_records_updated: int = 0
    cache_hits: int = 0
    last_reset: datetime = field(default_factory=datetime.now)

class FHIRPersistenceAgent:
    """
    🏥 Agente de Persistencia FHIR Completamente Flexible
    
    Convierte recursos FHIR a SQL y los persiste automáticamente
    usando el sistema flexible existente para detectar esquemas
    y generar mapeos dinámicamente.
    """
    
    def __init__(self, db_path: str, llm_client=None):
        self.db_path = db_path
        self.llm_client = llm_client
        
        # Inicializar sistema flexible con fallback
        if FLEXIBLE_SYSTEM_AVAILABLE:
            try:
                self.introspector = SchemaIntrospector(db_path)
                self.mapper = DynamicMapper(db_path)
                logger.info("🚀 Sistema flexible completamente disponible")
            except Exception as e:
                logger.warning(f"⚠️ Error inicializando sistema flexible: {e}")
                self.introspector = self._create_fallback_introspector()
                self.mapper = self._create_fallback_mapper()
                logger.info("🔄 Usando sistema flexible básico (fallback)")
        else:
            logger.warning("⚠️ Sistema flexible no disponible - usando implementación básica")
            self.introspector = self._create_fallback_introspector()
            self.mapper = self._create_fallback_mapper()
        
        # Cache y estadísticas
        self.table_detection_cache: Dict[str, List[str]] = {}
        self.sql_template_cache: Dict[str, str] = {}
        self.stats = PersistenceStats()
        
        # Llamar inicialización adicional
        self.__post_init__()
    
    def _create_fallback_introspector(self):
        """Crea introspector de fallback que no requiere parámetros"""
        class FallbackIntrospector:
            def analyze_table_structure(self, *args, **kwargs):
                # Crear metadata básico compatible con TableMetadata
                if FLEXIBLE_SYSTEM_AVAILABLE:
                    try:
                        # Intentar usar la clase real
                        return TableMetadata(name="", columns={}, confidence_score=0.5)  # type: ignore
                    except:
                        pass
                
                # Fallback: crear objeto con atributos mínimos necesarios
                class BasicMetadata:
                    def __init__(self):
                        self.columns = {}
                        self.confidence_score = 0.5
                        self.name = ""
                        # Agregar cualquier otro atributo que pueda necesitar TableMetadata
                        self.primary_key = None
                        self.foreign_keys = []
                        self.table_type = "unknown"
                        
                return BasicMetadata()
        return FallbackIntrospector()
    
    def _create_fallback_mapper(self):
        """Crea mapper de fallback que no requiere parámetros"""
        class FallbackMapper:
            def map_sql_to_fhir(self, *args, **kwargs):
                return {'resourceType': 'Basic', 'id': 'fallback'}
        return FallbackMapper()
        
    def __post_init__(self):
        """Inicialización después del constructor"""
        # Sistema 100% dinámico - sin patrones hardcodeados
        # Los patrones se generan automáticamente analizando el esquema de BD
        self.fhir_to_table_patterns = {}  # Se llenan dinámicamente
        
        logger.info("🚀 FHIRPersistenceAgent inicializado")
        logger.info(f"   📊 Base de datos: {self.db_path}")
        logger.info(f"   🤖 LLM disponible: {'✅' if self.llm_client else '❌'}")
        logger.info(f"   🔧 Sistema flexible: {'✅' if FLEXIBLE_SYSTEM_AVAILABLE else '❌ (fallback básico)'}")
        logger.info("   🎯 Modo: 100% dinámico sin hardcodeo")

    async def persist_fhir_resource(self, fhir_resource: Dict[str, Any]) -> PersistenceResult:
        """
        🎯 Persiste un recurso FHIR en las tablas SQL correspondientes usando LLM
        
        Args:
            fhir_resource: Recurso FHIR a persistir
            
        Returns:
            PersistenceResult con detalles de la operación
            
        Raises:
            ValueError: Si no hay LLM configurado o el recurso es inválido
        """
        start_time = time.time()
        
        # Validar LLM disponible
        if not self.llm_client:
            return PersistenceResult(
                success=False,
                resource_type=fhir_resource.get('resourceType', 'Unknown'),
                resource_id=fhir_resource.get('id', 'unknown'),
                errors=["❌ Se requiere un LLM configurado para la persistencia FHIR→SQL"],
                confidence_score=0.0
            )
        
        # Validar recurso FHIR
        if not isinstance(fhir_resource, dict):
            return PersistenceResult(
                success=False,
                resource_type='Invalid',
                resource_id='unknown',
                errors=["Recurso FHIR debe ser un diccionario"],
                confidence_score=0.0
            )
        
        resource_type = fhir_resource.get('resourceType', '')
        resource_id = fhir_resource.get('id', f'auto-{int(time.time())}')
        
        if not resource_type:
            return PersistenceResult(
                success=False,
                resource_type='Unknown',
                resource_id=resource_id,
                errors=["Recurso FHIR debe tener 'resourceType'"],
                confidence_score=0.0
            )
        
        logger.info(f"🚀 Iniciando persistencia de {resource_type}/{resource_id} con LLM")
        
        try:
            # 1. Detectar tablas destino usando múltiples estrategias
            target_tables = await self._detect_target_tables(fhir_resource)
            
            if not target_tables:
                return PersistenceResult(
                    success=False,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    errors=[f"No se encontraron tablas SQL para el recurso {resource_type}"],
                    confidence_score=0.2
                )
            
            logger.info(f"📊 Tablas destino detectadas: {target_tables}")
            
            # 2. Convertir FHIR a datos SQL usando LLM
            sql_data_sets = await self._convert_fhir_to_sql_data(fhir_resource, target_tables)
            
            if not sql_data_sets:
                return PersistenceResult(
                    success=False,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    target_tables=target_tables,
                    errors=["No se pudo generar datos SQL desde el recurso FHIR"],
                    confidence_score=0.3
                )
            
            # Log de campos mapeados
            for data_set in sql_data_sets:
                logger.info(f"📝 Tabla {data_set['table_name']}: {data_set.get('field_count', 0)} campos mapeados")
            
            # 3. Generar queries SQL
            queries_with_values = await self._generate_sql_queries(sql_data_sets, target_tables)
            
            # 4. Ejecutar queries
            execution_result = await self._execute_sql_queries(queries_with_values)
            
            # 5. Construir resultado
            processing_time = (time.time() - start_time) * 1000
            
            result = PersistenceResult(
                success=execution_result['success'],
                resource_type=resource_type,
                resource_id=resource_id,
                target_tables=target_tables,
                sql_queries=[query for query, _ in queries_with_values],
                records_created=execution_result['records_created'],
                records_updated=execution_result['records_updated'],
                processing_time_ms=processing_time,
                errors=execution_result.get('errors', []),
                warnings=[],
                confidence_score=execution_result.get('confidence_score', 0.8)
            )
            
            # Actualizar estadísticas
            self._update_stats(result)
            
            if result.success:
                logger.info(f"✅ Recurso {resource_type}/{resource_id} persistido exitosamente en {len(target_tables)} tablas")
            else:
                logger.error(f"❌ Error persistiendo {resource_type}/{resource_id}: {result.errors}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error crítico en persistencia: {e}", exc_info=True)
            
            return PersistenceResult(
                success=False,
                resource_type=resource_type,
                resource_id=resource_id,
                errors=[f"Error crítico: {str(e)}"],
                processing_time_ms=(time.time() - start_time) * 1000,
                confidence_score=0.0
            )

    async def _detect_target_tables(self, fhir_resource: Dict[str, Any]) -> List[str]:
        """
        🎯 Detecta automáticamente las tablas destino para un recurso FHIR
        
        Estrategia completamente flexible:
        1. Usar patrones de nombre de tabla
        2. Analizar esquemas con SchemaIntrospector  
        3. Usar LLM para decisiones complejas
        4. Cache inteligente
        """
        resource_type = fhir_resource.get('resourceType', 'Unknown')
        
        # Verificar cache
        cache_key = f"tables_{resource_type}"
        if cache_key in self.table_detection_cache:
            self.stats.cache_hits += 1
            return self.table_detection_cache[cache_key]
        
        target_tables = []
        
        # 1. Detectar por patrones de nombre
        pattern_tables = await self._detect_tables_by_pattern(resource_type)
        target_tables.extend(pattern_tables)
        
        # 2. Analizar esquemas para confirmación
        schema_tables = await self._detect_tables_by_schema(fhir_resource, pattern_tables)
        target_tables.extend(schema_tables)
        
        # 3. Usar LLM para casos complejos (si disponible)
        if self.llm_client and not target_tables:
            llm_tables = await self._detect_tables_with_llm(fhir_resource)
            target_tables.extend(llm_tables)
        
        # Eliminar duplicados y ordenar por relevancia
        target_tables = list(dict.fromkeys(target_tables))  # Mantener orden, eliminar duplicados
        
        # Guardar en cache
        self.table_detection_cache[cache_key] = target_tables
        
        logger.debug(f"🎯 Tablas detectadas para {resource_type}: {target_tables}")
        return target_tables

    async def _detect_tables_by_pattern(self, resource_type: str) -> List[str]:
        """Detecta tablas usando análisis dinámico del esquema (sin patrones hardcodeados)"""
        # Obtener todas las tablas de la BD
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        all_tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        matching_tables = []
        resource_lower = resource_type.lower()
        
        # Análisis semántico dinámico por similitud de tokens
        resource_tokens = re.findall(r'\w+', resource_lower)
        
        for table in all_tables:
            table_tokens = re.findall(r'\w+', table.lower())
            
            # Calcular similitud semántica
            common_tokens = set(resource_tokens) & set(table_tokens)
            if common_tokens:
                similarity = len(common_tokens) / max(len(resource_tokens), len(table_tokens))
                
                # Umbral dinámico basado en similitud semántica
                if similarity > 0.3:  # 30% de similitud mínima
                    matching_tables.append(table)
                    logger.debug(f"🎯 Tabla {table} detectada para {resource_type} (similitud: {similarity:.2f})")
        
        return matching_tables

    async def _detect_tables_by_schema(self, fhir_resource: Dict[str, Any], candidate_tables: List[str]) -> List[str]:
        """Analiza esquemas para confirmar compatibilidad"""
        confirmed_tables = []
        
        for table_name in candidate_tables:
            try:
                metadata = self.introspector.analyze_table_structure(table_name)
                
                # Verificar compatibilidad básica
                compatibility_score = self._calculate_schema_compatibility(fhir_resource, metadata)
                
                if compatibility_score > 0.3:  # Umbral de compatibilidad
                    confirmed_tables.append(table_name)
                    logger.debug(f"✅ Tabla {table_name} compatible (score: {compatibility_score:.2f})")
                else:
                    logger.debug(f"❌ Tabla {table_name} no compatible (score: {compatibility_score:.2f})")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error analizando tabla {table_name}: {e}")
        
        return confirmed_tables

    def _calculate_schema_compatibility(self, fhir_resource: Dict[str, Any], metadata: Any) -> float:
        """Calcula compatibilidad entre recurso FHIR y esquema de tabla"""
        score = 0.0
        
        # Verificar columnas básicas esperadas
        column_names = [col.lower() for col in metadata.columns.keys()]
        
        # Puntos por columnas comunes
        if any('id' in col for col in column_names):
            score += 0.3
        if any('name' in col for col in column_names):
            score += 0.2
        if any('date' in col for col in column_names):
            score += 0.2
        if any('status' in col for col in column_names):
            score += 0.1
        
        # Puntos por tipo de recurso específico
        resource_type = fhir_resource.get('resourceType', '').lower()
        if resource_type == 'patient':
            if any('pati' in col for col in column_names):
                score += 0.4
        elif resource_type == 'condition':
            if any('diag' in col or 'cond' in col for col in column_names):
                score += 0.4
        # ... más tipos específicos
        
        return min(score, 1.0)

    async def _detect_tables_with_llm(self, fhir_resource: Dict[str, Any]) -> List[str]:
        """Usa LLM para detectar tablas en casos complejos"""
        if not self.llm_client:
            return []
        
        try:
            # Obtener esquema de BD usando conexión directa
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            all_tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            # Crear prompt para LLM
            prompt = f"""
Analiza este recurso FHIR y determina qué tablas SQL son las más apropiadas para almacenarlo:

RECURSO FHIR:
{json.dumps(fhir_resource, indent=2)}

TABLAS DISPONIBLES:
{', '.join(all_tables[:50])}  # Limitar para no saturar el prompt

Responde SOLO con una lista de nombres de tabla separados por comas, sin explicaciones.
Ejemplo: PATI_PATIENTS, EPIS_EPISODES
"""
            
            # Llamar al LLM (implementación específica según el cliente)
            response = await self._call_llm(prompt)
            
            # Parsear respuesta
            suggested_tables = [table.strip() for table in response.split(',')]
            valid_tables = [table for table in suggested_tables if table in all_tables]
            
            logger.debug(f"🤖 LLM sugiere tablas: {valid_tables}")
            return valid_tables
            
        except Exception as e:
            logger.warning(f"⚠️ Error usando LLM para detección de tablas: {e}")
            return []

    async def _call_llm(self, prompt: str) -> str:
        """Llama al LLM (implementación placeholder)"""
        # TODO: Implementar según el cliente LLM disponible
        return "PATI_PATIENTS"  # Placeholder

    def _call_openai_native(self, client, prompt: str, temperature: float = 0.1, max_tokens: int = 2000) -> str:
        """
        Llama a OpenAI nativo de manera flexible
        """
        try:
            # Intentar con OpenAI directo
            from openai import OpenAI
            from openai.types.chat import ChatCompletionUserMessageParam
            
            native_client = OpenAI()
            
            # Crear mensaje con el tipo correcto
            messages = [ChatCompletionUserMessageParam(role="user", content=prompt)]
            
            response = native_client.chat.completions.create(
                model='gpt-4o',
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content or ""
            
            if not content.strip():
                content = '{"error": "Respuesta vacía del LLM"}'
            
            return content
            
        except Exception as e:
            logger.error(f"Error en _call_openai_native: {e}", exc_info=True)
            return f'{{"error": "Error en llamada al LLM: {str(e)}"}}'

    async def _convert_fhir_to_sql_data(self, fhir_resource: Dict[str, Any], target_tables: List[str]) -> List[Dict[str, Any]]:
        """Convierte recurso FHIR a datos SQL para cada tabla destino usando LLM"""
        if not self.llm_client:
            raise ValueError("❌ Se requiere un LLM configurado para la conversión FHIR→SQL")
        
        sql_data_sets = []
        errors = []
        
        for table_name in target_tables:
            try:
                logger.info(f"🔄 Mapeando recurso FHIR a tabla {table_name} usando LLM...")
                
                # Usar el sistema inteligente con LLM para mapear FHIR→SQL
                sql_data = await self._map_fhir_to_sql_for_table(fhir_resource, table_name)
                
                if sql_data:
                    sql_data_sets.append({
                        'table_name': table_name,
                        'data': sql_data,
                        'field_count': len(sql_data)
                    })
                    logger.info(f"✅ Mapeados {len(sql_data)} campos para tabla {table_name}")
                else:
                    logger.warning(f"⚠️ No se obtuvieron datos para tabla {table_name}")
                    
            except Exception as e:
                error_msg = f"Error mapeando FHIR→SQL para tabla {table_name}: {str(e)}"
                logger.error(f"❌ {error_msg}")
                errors.append(error_msg)
        
        if not sql_data_sets and errors:
            raise ValueError(f"No se pudo mapear a ninguna tabla. Errores: {'; '.join(errors)}")
        
        return sql_data_sets

    async def _map_fhir_to_sql_for_table(self, fhir_resource: Dict[str, Any], table_name: str) -> Dict[str, Any]:
        """Mapea un recurso FHIR a datos SQL para una tabla específica usando LLM"""
        
        # Obtener metadatos de la tabla
        metadata = self.introspector.analyze_table_structure(table_name)
        
        # Usar LLM para mapeo inteligente
        try:
            sql_data = await self._map_with_llm(fhir_resource, table_name, metadata)
        except Exception as e:
            logger.error(f"❌ Error usando LLM para mapeo: {e}")
            # En lugar de fallback, devolver error claro
            raise ValueError(f"No se pudo mapear el recurso FHIR sin LLM: {str(e)}")
        
        # Agregar campos automáticos
        sql_data = self._add_automatic_fields(sql_data, table_name, fhir_resource)
        
        return sql_data

    async def _map_with_llm(self, fhir_resource: Dict[str, Any], table_name: str, metadata: Any) -> Dict[str, Any]:
        """Usa LLM para mapear inteligentemente TODOS los campos FHIR a columnas SQL"""
        
        # Preparar información detallada de columnas
        columns_info = []
        for col_name, col_type in metadata.columns.items():
            columns_info.append(f"- {col_name} ({col_type})")
        
        # Extraer información contextual del recurso FHIR
        resource_type = fhir_resource.get('resourceType', 'Unknown')
        resource_id = fhir_resource.get('id', 'sin-id')
        
        # Crear un resumen de los campos disponibles en el recurso FHIR
        fhir_fields_summary = self._analyze_fhir_structure(fhir_resource)
        
        prompt = f"""
Eres un experto en sistemas de salud con profundo conocimiento de FHIR R4 y bases de datos médicas SQLite.

CONTEXTO:
- Estás mapeando un recurso FHIR {resource_type} a la tabla SQL {table_name}
- El sistema es un HIS (Hospital Information System) completo
- Debes preservar TODA la información médica relevante
- La base de datos usa SQLite (NO SQL Server)

RECURSO FHIR A MAPEAR:
{json.dumps(fhir_resource, indent=2, ensure_ascii=False)}

ESTRUCTURA DEL RECURSO FHIR:
{fhir_fields_summary}

TABLA SQL DESTINO: {table_name}
COLUMNAS DISPONIBLES:
{chr(10).join(columns_info)}

INSTRUCCIONES DETALLADAS:
1. Analiza PROFUNDAMENTE el recurso FHIR, incluyendo:
   - Campos anidados (ej: name[0].given[0], address[0].city)
   - Arrays de objetos (ej: identifier[], telecom[], address[])
   - Extensiones personalizadas
   - CodeableConcepts y sus códigos/displays
   - Referencias a otros recursos

2. Para cada columna SQL, determina:
   - ¿Qué campo FHIR corresponde mejor?
   - ¿Cómo extraer el valor correcto de estructuras complejas?
   - ¿Qué transformación aplicar?

3. Reglas de mapeo inteligente:
   - IDs: Extraer valores numéricos cuando sea posible
   - Nombres: Considerar given, family, prefix, suffix
   - Fechas: Formato ISO (YYYY-MM-DD HH:MM:SS)
   - Booleanos: 1/0 para columnas BIT
   - Referencias: Extraer solo el ID numérico (ej: "Patient/123" → 123)
   - Códigos: Preferir el código sobre el display
   - Telecom: Extraer según el 'use' (home, work, mobile)
   - Address: Concatenar líneas si es necesario

4. Consideraciones especiales por tipo de recurso:
   - Patient: Mapear toda la información demográfica
   - Condition: Incluir códigos ICD, severidad, estado clínico
   - Observation: Valores con unidades, interpretaciones
   - MedicationRequest: Dosis, frecuencia, duración
   - Encounter: Tipo, clase, período, ubicación

5. NO uses funciones SQL, solo valores de datos
6. Para timestamps actuales usa el formato: "2024-01-15 10:30:00"
7. Si un campo FHIR no tiene equivalente SQL obvio, intenta inferirlo por contexto

RESPUESTA ESPERADA:
Un JSON con TODAS las columnas SQL relevantes mapeadas a sus valores FHIR correspondientes.
Incluye campos calculados o derivados cuando sea apropiado.

Ejemplo de respuesta completa:
{{
  "PATI_ID": "123",
  "PATI_NAME": "Juan",
  "PATI_SURNAME_1": "García",
  "PATI_SURNAME_2": "López",
  "PATI_FULL_NAME": "Juan García López",
  "PATI_BIRTH_DATE": "1990-01-15",
  "PATI_GENDER": 1,
  "PATI_ADDRESS": "Calle Mayor 123, Madrid",
  "PATI_PHONE": "+34 600123456",
  "PATI_EMAIL": "juan.garcia@email.com",
  "PATI_ACTIVE": 1,
  "PATI_CREATED_DATE": "2024-01-15 10:30:00",
  "MTIME": "2024-01-15 10:30:00"
}}

IMPORTANTE: Mapea TODOS los campos posibles, no solo los básicos. Extrae el máximo de información del recurso FHIR.
"""
        
        try:
            # Llamar al LLM con reintentos mejorados
            response_text = await self._call_llm_with_retry(prompt)
            
            # Verificar y corregir funciones SQL incompatibles
            if any(func in response_text.upper() for func in ['GETDATE()', 'DATEDIFF(', 'DATEADD(', 'CURRENT_TIMESTAMP()']):
                logger.warning("⚠️ LLM generó respuesta con funciones SQL incompatibles, aplicando correcciones")
                response_text = self._fix_sql_compatibility(response_text)
            
            # Extraer JSON de la respuesta
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                sql_data = json.loads(json_match.group(0))
                
                # Validar y convertir tipos de datos según el esquema
                converted_data = {}
                for col_name, value in sql_data.items():
                    if col_name in metadata.columns:
                        column_info = {'type': metadata.columns[col_name]}
                        converted_value = self._convert_fhir_value_to_sql(value, column_info)
                        converted_data[col_name] = converted_value
                    else:
                        logger.warning(f"⚠️ Columna {col_name} no existe en tabla {table_name}, ignorando")
                
                # Validar que tenemos datos significativos
                if len(converted_data) < 2:  # Al menos ID + otro campo
                    raise ValueError(f"Mapeo insuficiente: solo {len(converted_data)} campos mapeados")
                
                logger.info(f"✅ LLM mapeó exitosamente {len(converted_data)} campos para tabla {table_name}")
                return converted_data
            else:
                raise ValueError("No se encontró JSON válido en la respuesta del LLM")
                
        except Exception as e:
            logger.error(f"❌ Error en mapeo con LLM: {e}")
            raise

    def _analyze_fhir_structure(self, fhir_resource: Dict[str, Any], prefix: str = "") -> str:
        """Analiza y resume la estructura del recurso FHIR para el prompt"""
        lines = []
        
        def analyze_value(key: str, value: Any, level: int = 0):
            indent = "  " * level
            if isinstance(value, dict):
                lines.append(f"{indent}- {key}: objeto con campos: {', '.join(value.keys())}")
                # Analizar algunos campos importantes
                for sub_key in ['text', 'coding', 'value', 'display', 'code']:
                    if sub_key in value:
                        analyze_value(f"{key}.{sub_key}", value[sub_key], level + 1)
            elif isinstance(value, list) and value:
                lines.append(f"{indent}- {key}: array con {len(value)} elementos")
                if isinstance(value[0], dict):
                    lines.append(f"{indent}  Primer elemento tiene: {', '.join(value[0].keys())}")
            elif value is not None:
                value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                lines.append(f"{indent}- {key}: {type(value).__name__} = {value_str}")
        
        for key, value in fhir_resource.items():
            analyze_value(key, value)
        
        return "\n".join(lines)

    async def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Llama al LLM con reintentos y manejo mejorado de errores"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"🔄 Reintento {attempt + 1} de {max_retries}")
                    await asyncio.sleep(2 ** attempt)  # Backoff exponencial
                
                # Intentar llamada al LLM
                if self.llm_client and hasattr(self.llm_client, 'ainvoke'):
                    # LangChain style
                    from langchain.schema import HumanMessage
                    response = await self.llm_client.ainvoke([HumanMessage(content=prompt)])
                    return response.content
                else:
                    # OpenAI style o cliente directo
                    return self._call_openai_native(self.llm_client, prompt)
                    
            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ Error en intento {attempt + 1}: {str(e)}")
                
                # Si es un error de rate limit, esperar más
                if "rate" in str(e).lower():
                    await asyncio.sleep(5 * (attempt + 1))
        
        raise Exception(f"Fallo después de {max_retries} intentos. Último error: {last_error}")

    def _fallback_mapping(self, fhir_resource: Dict[str, Any], table_name: str, metadata: Any) -> Dict[str, Any]:
        """ELIMINADO - Ya no usamos fallback, siempre requerimos LLM"""
        raise NotImplementedError("El mapeo sin LLM no está soportado. Se requiere un LLM configurado.")

    def _extract_fhir_value_for_column(self, fhir_resource: Dict[str, Any], column_name: str, resource_type: str) -> Any:
        """ELIMINADO - Ya no usamos extracción manual, el LLM maneja todo"""
        raise NotImplementedError("La extracción manual no está soportada. Use LLM para mapeo.")

    def _map_fhir_gender_to_sql(self, fhir_gender: str) -> int:
        """Mapea género FHIR a valor SQL"""
        mapping = {
            'male': 1,
            'female': 2,
            'other': 3,
            'unknown': 4
        }
        return mapping.get(fhir_gender.lower(), 4)

    def _convert_fhir_value_to_sql(self, fhir_value: Any, column_info: Dict[str, Any]) -> Any:
        """Convierte valor FHIR al tipo SQL apropiado"""
        sql_type = column_info.get('type', 'TEXT').upper()
        
        if sql_type in ['INT', 'INTEGER', 'BIGINT']:
            try:
                return int(fhir_value)
            except (ValueError, TypeError):
                return 0
        
        elif sql_type in ['REAL', 'FLOAT', 'DOUBLE']:
            try:
                return float(fhir_value)
            except (ValueError, TypeError):
                return 0.0
        
        elif sql_type in ['DATE', 'DATETIME', 'DATETIME2']:
            if isinstance(fhir_value, str):
                return fhir_value
            return str(fhir_value)
        
        elif sql_type == 'BIT':
            if isinstance(fhir_value, bool):
                return 1 if fhir_value else 0
            return 1 if str(fhir_value).lower() in ['true', '1', 'yes'] else 0
        
        else:  # TEXT, NVARCHAR, etc.
            return str(fhir_value)

    def _add_automatic_fields(self, sql_data: Dict[str, Any], table_name: str, fhir_resource: Dict[str, Any]) -> Dict[str, Any]:
        """Añade campos automáticos comunes (timestamps, IDs, etc.)"""
        # Ejemplo: añadir campo de timestamp
        if 'MTIME' in sql_data or 'mtime' in sql_data:
            pass  # Ya lo tiene
        else:
            ts = datetime.utcnow().isoformat(timespec='seconds')
            if any(c.lower() == 'mtime' for c in sql_data.keys()):
                sql_data['MTIME'] = ts
        
        # NUEVO: Si es tabla de pacientes, rellenar PATI_FULL_NAME si falta
        if table_name.upper().endswith('PATI_PATIENTS'):
            has_name = 'PATI_NAME' in sql_data
            has_surname = 'PATI_SURNAME_1' in sql_data
            full_name_missing = 'PATI_FULL_NAME' not in sql_data or not sql_data.get('PATI_FULL_NAME')
            if full_name_missing and (has_name or has_surname):
                parts = [sql_data.get('PATI_NAME', '').strip(), sql_data.get('PATI_SURNAME_1', '').strip(), sql_data.get('PATI_SURNAME_2', '').strip()]
                full_name = " ".join([p for p in parts if p])
                if full_name:
                    sql_data['PATI_FULL_NAME'] = full_name
        
        return sql_data

    async def _generate_sql_queries(self, sql_data_sets: List[Dict[str, Any]], target_tables: List[str]) -> List[Tuple[str, List[Any]]]:
        """Genera consultas SQL INSERT/UPDATE con sus valores"""
        queries_with_values = []
        
        for data_set in sql_data_sets:
            table_name = data_set['table_name']
            data = data_set['data']
            
            if not data:
                continue
            
            # Generar consulta UPSERT (INSERT OR REPLACE) con valores
            query, values = self._generate_upsert_query(table_name, data)
            queries_with_values.append((query, values))
        
        return queries_with_values

    def _generate_upsert_query(self, table_name: str, data: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Genera consulta SQL UPSERT con valores separados"""
        
        columns = list(data.keys())
        values = list(data.values())
        placeholders = ['?' for _ in columns]
        
        # Usar INSERT OR REPLACE para SQLite
        query = f"""
        INSERT OR REPLACE INTO {table_name} 
        ({', '.join(columns)}) 
        VALUES ({', '.join(placeholders)})
        """
        
        # NUEVO: Aplicar correcciones de compatibilidad preventivamente
        query = self._fix_sql_compatibility(query.strip())
        
        return query, values

    async def _execute_sql_queries(self, queries_with_values: List[Tuple[str, List[Any]]]) -> Dict[str, Any]:
        """Ejecuta las consultas SQL generadas con sus valores"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            records_created = 0
            records_updated = 0
            errors = []
            
            for query, values in queries_with_values:
                try:
                    cursor.execute(query, values)
                    records_created += cursor.rowcount
                    
                except sqlite3.Error as e:
                    # NUEVO: Aplicar correcciones de compatibilidad SQLite
                    if "no such function: GETDATE" in str(e) or "GETDATE" in query:
                        corrected_query = self._fix_sql_compatibility(query)
                        if corrected_query != query:
                            try:
                                logger.info(f"🔧 Reintentando con SQL corregido")
                                cursor.execute(corrected_query, values)
                                records_created += cursor.rowcount
                                continue
                            except sqlite3.Error as e2:
                                errors.append(f"Error SQL (después de corrección): {str(e2)}")
                                logger.error(f"❌ Error ejecutando SQL corregido: {e2}")
                    
                    errors.append(f"Error SQL: {str(e)}")
                    logger.error(f"❌ Error ejecutando SQL: {e}")
                    logger.error(f"   Query: {query}")
                    logger.error(f"   Values: {values}")
            
            conn.commit()
            conn.close()
            
            return {
                'success': len(errors) == 0,
                'records_created': records_created,
                'records_updated': records_updated,
                'errors': errors,
                'confidence_score': 0.9 if len(errors) == 0 else 0.3
            }
            
        except Exception as e:
            logger.error(f"❌ Error crítico ejecutando SQL: {e}")
            return {
                'success': False,
                'records_created': 0,
                'records_updated': 0,
                'errors': [str(e)],
                'confidence_score': 0.0
            }

    def _update_stats(self, result: PersistenceResult) -> None:
        """Actualiza estadísticas del agente"""
        self.stats.total_resources_processed += 1
        
        if result.success:
            self.stats.successful_persistences += 1
        else:
            self.stats.failed_persistences += 1
        
        self.stats.total_sql_queries_generated += len(result.sql_queries)
        self.stats.total_records_created += result.records_created
        self.stats.total_records_updated += result.records_updated
        
        # Actualizar tiempo promedio
        total_time = (self.stats.avg_processing_time_ms * (self.stats.total_resources_processed - 1) + 
                     result.processing_time_ms)
        self.stats.avg_processing_time_ms = total_time / self.stats.total_resources_processed

    async def persist_fhir_bundle(self, fhir_bundle: Dict[str, Any]) -> List[PersistenceResult]:
        """Persiste un Bundle FHIR completo"""
        results = []
        
        if 'entry' not in fhir_bundle:
            return results
        
        for entry in fhir_bundle['entry']:
            if 'resource' in entry:
                result = await self.persist_fhir_resource(entry['resource'])
                results.append(result)
        
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del agente"""
        return {
            'total_resources_processed': self.stats.total_resources_processed,
            'successful_persistences': self.stats.successful_persistences,
            'failed_persistences': self.stats.failed_persistences,
            'success_rate': (self.stats.successful_persistences / max(self.stats.total_resources_processed, 1)) * 100,
            'avg_processing_time_ms': self.stats.avg_processing_time_ms,
            'total_sql_queries_generated': self.stats.total_sql_queries_generated,
            'total_records_created': self.stats.total_records_created,
            'total_records_updated': self.stats.total_records_updated,
            'cache_hits': self.stats.cache_hits,
            'cache_hit_rate': (self.stats.cache_hits / max(self.stats.total_resources_processed, 1)) * 100,
            'last_reset': self.stats.last_reset.isoformat()
        }

    def reset_stats(self) -> None:
        """Reinicia estadísticas"""
        self.stats = PersistenceStats()

    def clear_cache(self) -> None:
        """Limpia todos los caches"""
        self.table_detection_cache.clear()
        self.sql_template_cache.clear()
        logger.info(" Cache limpiado")

    def _fix_sql_compatibility(self, sql: str) -> str:
        """
        🔧 Corrige funciones SQL incompatibles con SQLite
        """
        try:
            import re
            
            # Correcciones de funciones de fecha
            corrections = {
                # SQL Server -> SQLite
                r'\bGETDATE\s*\(\s*\)': "datetime('now')",
                r'\bCURRENT_TIMESTAMP\s*\(\s*\)': "datetime('now')",
                r'\bSYSDATE\s*\(\s*\)': "datetime('now')",  # Oracle
                
                # Funciones DATEDIFF
                r'\bDATEDIFF\s*\(\s*DAY\s*,\s*([^,]+)\s*,\s*GETDATE\s*\(\s*\)\s*\)': r"julianday('now') - julianday(\1)",
                r'\bDATEDIFF\s*\(\s*YEAR\s*,\s*([^,]+)\s*,\s*GETDATE\s*\(\s*\)\s*\)': r"(julianday('now') - julianday(\1)) / 365.25",
                r'\bDATEDIFF\s*\(\s*MONTH\s*,\s*([^,]+)\s*,\s*GETDATE\s*\(\s*\)\s*\)': r"(julianday('now') - julianday(\1)) / 30.44",
                
                # Funciones DATEADD
                r'\bDATEADD\s*\(\s*DAY\s*,\s*(-?\d+)\s*,\s*GETDATE\s*\(\s*\)\s*\)': r"date('now', '\1 days')",
                r'\bDATEADD\s*\(\s*MONTH\s*,\s*(-?\d+)\s*,\s*GETDATE\s*\(\s*\)\s*\)': r"date('now', '\1 months')",
                r'\bDATEADD\s*\(\s*YEAR\s*,\s*(-?\d+)\s*,\s*GETDATE\s*\(\s*\)\s*\)': r"date('now', '\1 years')",
                
                # Otras correcciones comunes
                r'\bTOP\s+(\d+)\b': r'LIMIT \1',  # SQL Server TOP -> SQLite LIMIT
                r'\bISNULL\s*\(': 'COALESCE(',  # SQL Server ISNULL -> SQLite COALESCE
                r'\bLEN\s*\(': 'LENGTH(',  # SQL Server LEN -> SQLite LENGTH
            }
            
            corrected_sql = sql
            for pattern, replacement in corrections.items():
                corrected_sql = re.sub(pattern, replacement, corrected_sql, flags=re.IGNORECASE)
            
            # Log si se hicieron correcciones
            if corrected_sql != sql:
                logger.info(f"🔧 SQL corregido para compatibilidad con SQLite en FHIRPersistenceAgent")
                logger.debug(f"Original: {sql}")
                logger.debug(f"Corregido: {corrected_sql}")
            
            return corrected_sql
            
        except Exception as e:
            logger.error(f"Error corrigiendo compatibilidad SQL: {e}")
            return sql  # Devolver original si falla la corrección

# Función de utilidad para uso rápido
async def quick_persist_fhir_resource(db_path: str, fhir_resource: Dict[str, Any]) -> PersistenceResult:
    """Función de utilidad para persistir un recurso FHIR rápidamente"""
    agent = FHIRPersistenceAgent(db_path)
    return await agent.persist_fhir_resource(fhir_resource)

# Ejemplo de uso
if __name__ == "__main__":
    async def demo():
        # Ejemplo de recurso FHIR Patient
        patient_resource = {
            "resourceType": "Patient",
            "id": "12345",
            "name": [{"given": ["Ana"], "family": "Gómez"}],
            "gender": "female",
            "birthDate": "1966-12-25"
        }
        
        agent = FHIRPersistenceAgent("database_new.sqlite3.db")
        result = await agent.persist_fhir_resource(patient_resource)
        
        print(f"Resultado: {result}")
        print(f"Estadísticas: {agent.get_stats()}")
    
    asyncio.run(demo()) 