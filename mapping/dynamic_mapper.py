"""
🎯 DYNAMIC MAPPER: Mapeador Dinámico SQL→FHIR Ultra-Rápido
═══════════════════════════════════════════════════════════════

Mapeador inteligente que usa el SchemaIntrospector para generar
mapeos FHIR automáticamente en tiempo real con cache inteligente.

✅ Auto-mapeo basado en patrones de la bibliografía
✅ Cache multinivel ultra-rápido
✅ Validación FHIR opcional
✅ Métricas de rendimiento
✅ Fallback inteligente

Tiempo objetivo: < 10ms por mapeo
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import json
import re
from pathlib import Path

try:
    from .schema_introspector import SchemaIntrospector, TableMetadata  # type: ignore
except ImportError:
    # Fallback en caso de problemas de import
    class SchemaIntrospector:  # type: ignore
        def __init__(self, *args, **kwargs): pass
        def analyze_table_structure(self, *args, **kwargs): return TableMetadata()  # type: ignore
        def analyze_field_mappings(self, *args, **kwargs): return {}  # type: ignore
    class TableMetadata:  # type: ignore
        def __init__(self, *args, **kwargs): 
            self.confidence_score = 0.0  # type: ignore
            self.fhir_resource = ""  # type: ignore
            self.primary_key = None  # type: ignore
            self.name = ""  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MappingResult:
    """Resultado de mapeo SQL→FHIR"""
    fhir_resource: Dict[str, Any]
    resource_type: str
    confidence_score: float
    mapping_time_ms: float
    source_table: str
    field_count: int
    validation_errors: List[str] = field(default_factory=list)
    cache_hit: bool = False

@dataclass
class MappingStats:
    """Estadísticas de rendimiento del mapeador"""
    total_mappings: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_mapping_time_ms: float = 0.0
    total_time_ms: float = 0.0
    errors: int = 0
    last_reset: datetime = field(default_factory=datetime.now)

class DynamicMapper:
    """
    🎯 Mapeador Dinámico SQL→FHIR Ultra-Rápido
    
    Convierte registros SQL en recursos FHIR usando patrones automáticos
    del SchemaIntrospector con cache inteligente multinivel.
    """
    
    def __init__(self, db_path: str, cache_ttl: int = 3600):
        self.db_path = db_path
        self.cache_ttl = cache_ttl
        
        # Inicializar introspector
        self.introspector = SchemaIntrospector(db_path, cache_ttl)
        
        # Cache multinivel
        self.mapping_cache: Dict[str, MappingResult] = {}
        self.schema_cache: Dict[str, TableMetadata] = {}
        self.fhir_template_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Estadísticas
        self.stats = MappingStats()
        
        # Templates FHIR base para rendimiento
        self.fhir_templates = {
            'Patient': {
                'resourceType': 'Patient',
                'id': '',
                'meta': {'lastUpdated': ''},
                'identifier': [],
                'name': [{'family': '', 'given': ['']}],
                'telecom': [],
                'gender': '',
                'birthDate': '',
                'address': [],
                'active': True
            },
            'Encounter': {
                'resourceType': 'Encounter',
                'id': '',
                'meta': {'lastUpdated': ''},
                'identifier': [],
                'status': 'finished',
                'class': {'system': 'http://terminology.hl7.org/CodeSystem/v3-ActCode', 'code': 'AMB'},
                'subject': {'reference': ''},
                'period': {'start': '', 'end': ''},
                'reasonCode': [],
                'location': []
            },
            'Procedure': {
                'resourceType': 'Procedure',
                'id': '',
                'meta': {'lastUpdated': ''},
                'identifier': [],
                'status': 'completed',
                'code': {'text': ''},
                'subject': {'reference': ''},
                'performedDateTime': '',
                'reasonCode': []
            },
            'Observation': {
                'resourceType': 'Observation',
                'id': '',
                'meta': {'lastUpdated': ''},
                'identifier': [],
                'status': 'final',
                'code': {'text': ''},
                'subject': {'reference': ''},
                'effectiveDateTime': '',
                'valueString': ''
            },
            'Medication': {
                'resourceType': 'Medication',
                'id': '',
                'meta': {'lastUpdated': ''},
                'identifier': [],
                'code': {'text': ''},
                'status': 'active',
                'manufacturer': {'display': ''}
            },
            'AllergyIntolerance': {
                'resourceType': 'AllergyIntolerance',
                'id': '',
                'meta': {'lastUpdated': ''},
                'identifier': [],
                'clinicalStatus': {'coding': [{'system': 'http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical', 'code': 'active'}]},
                'verificationStatus': {'coding': [{'system': 'http://terminology.hl7.org/CodeSystem/allergyintolerance-verification', 'code': 'confirmed'}]},
                'patient': {'reference': ''},
                'code': {'text': ''},
                'reaction': []
            },
            'Condition': {
                'resourceType': 'Condition',
                'id': '',
                'meta': {'lastUpdated': ''},
                'identifier': [],
                'clinicalStatus': {'coding': [{'system': 'http://terminology.hl7.org/CodeSystem/condition-clinical', 'code': 'active'}]},
                'verificationStatus': {'coding': [{'system': 'http://terminology.hl7.org/CodeSystem/condition-ver-status', 'code': 'confirmed'}]},
                'subject': {'reference': ''},
                'code': {'text': ''},
                'onsetDateTime': ''
            }
        }
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Verifica validez del cache"""
        if cache_key not in self.cache_timestamps:
            return False
        age = datetime.now() - self.cache_timestamps[cache_key]
        return age.total_seconds() < self.cache_ttl
    
    def _get_from_cache(self, cache_key: str, cache_dict: Dict) -> Optional[Any]:
        """Obtiene del cache si es válido"""
        if self._is_cache_valid(cache_key) and cache_key in cache_dict:
            self.stats.cache_hits += 1
            return cache_dict[cache_key]
        
        self.stats.cache_misses += 1
        return None
    
    def _set_cache(self, cache_key: str, data: Any, cache_dict: Dict) -> None:
        """Guarda en cache"""
        cache_dict[cache_key] = data
        self.cache_timestamps[cache_key] = datetime.now()
    
    async def map_sql_to_fhir(self, 
                             table_name: str, 
                             sql_data: Dict[str, Any],
                             validate: bool = False) -> MappingResult:
        """
        🚀 Mapeo principal SQL→FHIR ultra-rápido
        
        Args:
            table_name: Nombre de la tabla SQL
            sql_data: Datos del registro SQL
            validate: Si validar el recurso FHIR generado
            
        Returns:
            MappingResult: Resultado completo del mapeo
        """
        start_time = time.time()
        
        # Generar clave de cache única
        cache_key = f"{table_name}_{hash(str(sorted(sql_data.items())))}"
        
        # Verificar cache de mapeo
        cached_result = self._get_from_cache(cache_key, self.mapping_cache)
        if cached_result:
            cached_result.cache_hit = True
            return cached_result
        
        try:
            # 1. Obtener metadatos de la tabla (con cache)
            table_metadata = await self._get_table_metadata(table_name)
            
            # 2. Crear recurso FHIR base
            fhir_resource = await self._create_base_fhir_resource(table_metadata, sql_data)
            
            # 3. Mapear campos específicos
            await self._map_fields_to_fhir(table_metadata, sql_data, fhir_resource)
            
            # 4. Post-procesamiento
            await self._post_process_fhir_resource(fhir_resource, table_metadata)
            
            # 5. Validación opcional
            validation_errors = []
            if validate:
                validation_errors = await self._validate_fhir_resource(fhir_resource)
            
            # 6. Calcular métricas
            mapping_time_ms = (time.time() - start_time) * 1000
            confidence = table_metadata.confidence_score
            
            # 7. Crear resultado
            result = MappingResult(
                fhir_resource=fhir_resource,
                resource_type=table_metadata.fhir_resource,
                confidence_score=confidence,
                mapping_time_ms=mapping_time_ms,
                source_table=table_name,
                field_count=len(sql_data),
                validation_errors=validation_errors,
                cache_hit=False
            )
            
            # 8. Guardar en cache si es de alta confianza
            if confidence >= 0.7:
                self._set_cache(cache_key, result, self.mapping_cache)
            
            # 9. Actualizar estadísticas
            self._update_stats(mapping_time_ms, len(validation_errors) > 0)
            
            logger.debug(f"⚡ Mapeo {table_name}: {mapping_time_ms:.2f}ms (confianza: {confidence:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error mapeando {table_name}: {e}")
            self.stats.errors += 1
            
            # Fallback básico
            return MappingResult(
                fhir_resource={'resourceType': 'Basic', 'id': str(sql_data.get('ID', 'unknown'))},
                resource_type='Basic',
                confidence_score=0.1,
                mapping_time_ms=(time.time() - start_time) * 1000,
                source_table=table_name,
                field_count=len(sql_data),
                validation_errors=[f"Mapping error: {str(e)}"]
            )
    
    async def _get_table_metadata(self, table_name: str) -> TableMetadata:
        """Obtiene metadatos de tabla con cache"""
        cache_key = f"metadata_{table_name}"
        
        cached_metadata = self._get_from_cache(cache_key, self.schema_cache)
        if cached_metadata:
            return cached_metadata
        
        # Analizar tabla
        metadata = self.introspector.analyze_table_structure(table_name)
        self._set_cache(cache_key, metadata, self.schema_cache)
        
        return metadata
    
    async def _create_base_fhir_resource(self, metadata: TableMetadata, sql_data: Dict[str, Any]) -> Dict[str, Any]:
        """Crea recurso FHIR base desde template"""
        resource_type = metadata.fhir_resource
        
        # Obtener template desde cache
        cache_key = f"template_{resource_type}"
        template = self._get_from_cache(cache_key, self.fhir_template_cache)
        
        if not template:
            template = self.fhir_templates.get(resource_type, {
                'resourceType': resource_type,
                'id': '',
                'meta': {'lastUpdated': ''}
            }).copy()
            self._set_cache(cache_key, template, self.fhir_template_cache)
        
        # Clonar template
        fhir_resource = json.loads(json.dumps(template))  # Deep copy rápido
        
        # Campos básicos obligatorios
        primary_key = metadata.primary_key or 'ID'
        resource_id = str(sql_data.get(primary_key, sql_data.get('ID', 'unknown')))
        
        fhir_resource['id'] = resource_id
        fhir_resource['meta']['lastUpdated'] = sql_data.get('MTIME', datetime.now().isoformat())
        
        return fhir_resource
    
    async def _map_fields_to_fhir(self, metadata: TableMetadata, sql_data: Dict[str, Any], fhir_resource: Dict[str, Any]) -> None:
        """Mapea campos SQL específicos a FHIR"""
        
        # Obtener mapeos de campos
        field_mappings = self.introspector.analyze_field_mappings(metadata)
        
        for sql_field, sql_value in sql_data.items():
            if sql_value is None or sql_field in ['MTIME', metadata.primary_key]:
                continue  # Saltar campos ya procesados o nulos
            
            field_info = field_mappings.get(sql_field, {})
            fhir_path = field_info.get('fhir_path', sql_field.lower())
            fhir_type = field_info.get('fhir_type', 'string')
            
            # Mapeo específico por tipo de recurso
            await self._map_field_by_resource_type(
                fhir_resource, sql_field, sql_value, fhir_path, fhir_type, metadata.fhir_resource
            )
    
    async def _map_field_by_resource_type(self, fhir_resource: Dict[str, Any], 
                                        sql_field: str, sql_value: Any,
                                        fhir_path: str, fhir_type: str, 
                                        resource_type: str) -> None:
        """Mapeo específico por tipo de recurso FHIR"""
        
        # Transformar valor según tipo FHIR
        transformed_value = self._transform_sql_value(sql_value, fhir_type)
        
        if resource_type == 'Patient':
            await self._map_patient_field(fhir_resource, sql_field, transformed_value)
        elif resource_type == 'Encounter':
            await self._map_encounter_field(fhir_resource, sql_field, transformed_value)
        elif resource_type == 'Procedure':
            await self._map_procedure_field(fhir_resource, sql_field, transformed_value)
        elif resource_type == 'Observation':
            await self._map_observation_field(fhir_resource, sql_field, transformed_value)
        else:
            # Mapeo genérico
            self._set_fhir_value(fhir_resource, fhir_path, transformed_value)
    
    async def _map_patient_field(self, fhir_resource: Dict[str, Any], sql_field: str, value: Any) -> None:
        """Mapeo específico para Patient"""
        field_upper = sql_field.upper()
        
        if 'NAME' in field_upper:
            if 'family' not in fhir_resource['name'][0] or not fhir_resource['name'][0]['family']:
                fhir_resource['name'][0]['family'] = str(value)
        elif 'SURNAME' in field_upper or 'LASTNAME' in field_upper:
            fhir_resource['name'][0]['family'] = str(value)
        elif 'BIRTH' in field_upper and 'DATE' in field_upper:
            fhir_resource['birthDate'] = self._format_date(value)
        elif 'GENDER' in field_upper or 'GEND' in field_upper:
            fhir_resource['gender'] = self._map_gender(value)
        elif 'PHONE' in field_upper:
            fhir_resource['telecom'].append({
                'system': 'phone',
                'value': str(value),
                'use': 'home'
            })
        elif 'EMAIL' in field_upper:
            fhir_resource['telecom'].append({
                'system': 'email',
                'value': str(value)
            })
        elif 'ADDRESS' in field_upper:
            if not fhir_resource['address']:
                fhir_resource['address'] = [{}]
            fhir_resource['address'][0].setdefault('line', []).append(str(value))
        elif 'ACTIVE' in field_upper:
            fhir_resource['active'] = bool(value)
        elif 'DESCRIPTION' in field_upper:
            fhir_resource.setdefault('text', {})['div'] = f"<div>{value}</div>"
    
    async def _map_encounter_field(self, fhir_resource: Dict[str, Any], sql_field: str, value: Any) -> None:
        """Mapeo específico para Encounter"""
        field_upper = sql_field.upper()
        
        if 'START' in field_upper and 'DATE' in field_upper:
            fhir_resource['period']['start'] = self._format_datetime(value)
        elif 'END' in field_upper and 'DATE' in field_upper:
            fhir_resource['period']['end'] = self._format_datetime(value)
        elif 'PATI_ID' in field_upper:
            fhir_resource['subject']['reference'] = f"Patient/{value}"
        elif 'STATUS' in field_upper or 'STATE' in field_upper:
            fhir_resource['status'] = self._map_encounter_status(value)
        elif 'CLASS' in field_upper:
            fhir_resource['class']['code'] = str(value)
        elif 'REASON' in field_upper:
            fhir_resource['reasonCode'].append({
                'text': str(value)
            })
    
    async def _map_procedure_field(self, fhir_resource: Dict[str, Any], sql_field: str, value: Any) -> None:
        """Mapeo específico para Procedure"""
        field_upper = sql_field.upper()
        
        if 'PATI_ID' in field_upper:
            fhir_resource['subject']['reference'] = f"Patient/{value}"
        elif 'DATE' in field_upper:
            fhir_resource['performedDateTime'] = self._format_datetime(value)
        elif 'CODE' in field_upper:
            fhir_resource['code']['coding'] = [{'code': str(value)}]
        elif 'DESCRIPTION' in field_upper:
            fhir_resource['code']['text'] = str(value)
        elif 'STATUS' in field_upper:
            fhir_resource['status'] = self._map_procedure_status(value)
    
    async def _map_observation_field(self, fhir_resource: Dict[str, Any], sql_field: str, value: Any) -> None:
        """Mapeo específico para Observation"""
        field_upper = sql_field.upper()
        
        if 'PATI_ID' in field_upper:
            fhir_resource['subject']['reference'] = f"Patient/{value}"
        elif 'DATE' in field_upper:
            fhir_resource['effectiveDateTime'] = self._format_datetime(value)
        elif 'VALUE' in field_upper or 'RESULT' in field_upper:
            fhir_resource['valueString'] = str(value)
        elif 'CODE' in field_upper:
            fhir_resource['code']['coding'] = [{'code': str(value)}]
        elif 'DESCRIPTION' in field_upper:
            fhir_resource['code']['text'] = str(value)
    
    def _transform_sql_value(self, value: Any, fhir_type: str) -> Any:
        """Transforma valor SQL al tipo FHIR apropiado"""
        if value is None:
            return None
        
        if fhir_type == 'boolean':
            return bool(value) if isinstance(value, (int, str)) else value
        elif fhir_type == 'integer':
            return int(value) if isinstance(value, (str, float)) else value
        elif fhir_type == 'decimal':
            return float(value) if isinstance(value, (str, int)) else value
        elif fhir_type in ['dateTime', 'date']:
            return self._format_datetime(value) if fhir_type == 'dateTime' else self._format_date(value)
        else:
            return str(value)
    
    def _format_datetime(self, value: Any) -> str:
        """Formatea datetime para FHIR"""
        if isinstance(value, str):
            try:
                # Intentar parsear diferentes formatos
                from dateutil import parser
                dt = parser.parse(value)
                return dt.isoformat()
            except:
                return str(value)
        elif hasattr(value, 'isoformat'):
            return value.isoformat()
        else:
            return str(value)
    
    def _format_date(self, value: Any) -> str:
        """Formatea date para FHIR"""
        if isinstance(value, str):
            try:
                from dateutil import parser
                dt = parser.parse(value)
                return dt.date().isoformat()
            except:
                return str(value)
        elif hasattr(value, 'date'):
            return value.date().isoformat()
        else:
            return str(value)
    
    def _map_gender(self, value: Any) -> str:
        """Mapea género a valores FHIR"""
        if not value:
            return 'unknown'
        
        value_str = str(value).lower()
        if value_str in ['m', 'male', 'masculino', '1']:
            return 'male'
        elif value_str in ['f', 'female', 'femenino', '2']:
            return 'female'
        else:
            return 'unknown'
    
    def _map_encounter_status(self, value: Any) -> str:
        """Mapea estado de encuentro"""
        if not value:
            return 'finished'
        
        value_str = str(value).lower()
        if 'active' in value_str or 'curso' in value_str:
            return 'in-progress'
        elif 'finished' in value_str or 'alta' in value_str:
            return 'finished'
        elif 'cancelled' in value_str or 'cancelado' in value_str:
            return 'cancelled'
        else:
            return 'finished'
    
    def _map_procedure_status(self, value: Any) -> str:
        """Mapea estado de procedimiento"""
        if not value:
            return 'completed'
        
        value_str = str(value).lower()
        if 'progress' in value_str or 'curso' in value_str:
            return 'in-progress'
        elif 'completed' in value_str or 'realizado' in value_str:
            return 'completed'
        elif 'cancelled' in value_str or 'cancelado' in value_str:
            return 'stopped'
        else:
            return 'completed'
    
    def _set_fhir_value(self, fhir_resource: Dict[str, Any], path: str, value: Any) -> None:
        """Establece valor en path FHIR anidado"""
        parts = path.split('.')
        current = fhir_resource
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    async def _post_process_fhir_resource(self, fhir_resource: Dict[str, Any], metadata: TableMetadata) -> None:
        """Post-procesamiento del recurso FHIR"""
        
        # Limpiar campos vacíos
        self._clean_empty_fields(fhir_resource)
        
        # Agregar identifier automático
        if 'identifier' in fhir_resource and not fhir_resource['identifier']:
            fhir_resource['identifier'] = [{
                'system': f'http://hospital.local/{metadata.name}',
                'value': fhir_resource.get('id', '')
            }]
        
        # Agregar text narrative si no existe
        if 'text' not in fhir_resource:
            fhir_resource['text'] = {
                'status': 'generated',
                'div': f'<div>Generated from {metadata.name}</div>'
            }
    
    def _clean_empty_fields(self, obj: Union[Dict, List]) -> None:
        """Limpia campos vacíos recursivamente"""
        if isinstance(obj, dict):
            empty_keys = []
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    self._clean_empty_fields(value)
                    if not value:  # Si quedó vacío después de limpiar
                        empty_keys.append(key)
                elif value == '' or value is None:
                    empty_keys.append(key)
            
            for key in empty_keys:
                del obj[key]
        
        elif isinstance(obj, list):
            obj[:] = [item for item in obj if item != '' and item is not None]
            for item in obj:
                if isinstance(item, (dict, list)):
                    self._clean_empty_fields(item)
    
    async def _validate_fhir_resource(self, fhir_resource: Dict[str, Any]) -> List[str]:
        """Validación básica del recurso FHIR"""
        errors = []
        
        # Validaciones básicas obligatorias
        if 'resourceType' not in fhir_resource:
            errors.append('Missing required field: resourceType')
        
        if 'id' not in fhir_resource or not fhir_resource['id']:
            errors.append('Missing required field: id')
        
        # Validaciones específicas por tipo
        resource_type = fhir_resource.get('resourceType', '')
        
        if resource_type == 'Patient':
            if 'name' not in fhir_resource or not fhir_resource['name']:
                errors.append('Patient must have at least one name')
        
        elif resource_type == 'Encounter':
            if 'subject' not in fhir_resource or not fhir_resource.get('subject', {}).get('reference'):
                errors.append('Encounter must have a subject reference')
        
        return errors
    
    def _update_stats(self, mapping_time_ms: float, has_errors: bool) -> None:
        """Actualiza estadísticas de rendimiento"""
        self.stats.total_mappings += 1
        self.stats.total_time_ms += mapping_time_ms
        
        # Calcular promedio móvil
        self.stats.avg_mapping_time_ms = self.stats.total_time_ms / self.stats.total_mappings
        
        if has_errors:
            self.stats.errors += 1
    
    async def batch_map_sql_to_fhir(self, 
                                  table_name: str, 
                                  sql_records: List[Dict[str, Any]],
                                  batch_size: int = 100) -> List[MappingResult]:
        """
        🚀 Mapeo por lotes ultra-optimizado
        
        Args:
            table_name: Nombre de la tabla
            sql_records: Lista de registros SQL
            batch_size: Tamaño del lote para procesamiento
            
        Returns:
            Lista de resultados de mapeo
        """
        logger.info(f"🚀 Iniciando mapeo por lotes: {len(sql_records)} registros de {table_name}")
        
        results = []
        
        # Procesar por lotes para optimizar memoria
        for i in range(0, len(sql_records), batch_size):
            batch = sql_records[i:i + batch_size]
            
            # Mapeo secuencial del lote (para SQLite)
            batch_results = []
            for record in batch:
                result = await self.map_sql_to_fhir(table_name, record)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Log de progreso
            progress = min(i + batch_size, len(sql_records))
            logger.info(f"📈 Progreso: {progress}/{len(sql_records)} registros mapeados")
        
        logger.info(f"✅ Mapeo por lotes completado: {len(results)} recursos FHIR generados")
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de rendimiento"""
        cache_hit_rate = (
            self.stats.cache_hits / (self.stats.cache_hits + self.stats.cache_misses) * 100
            if (self.stats.cache_hits + self.stats.cache_misses) > 0 else 0
        )
        
        return {
            'total_mappings': self.stats.total_mappings,
            'avg_mapping_time_ms': round(self.stats.avg_mapping_time_ms, 2),
            'cache_hit_rate_percent': round(cache_hit_rate, 2),
            'cache_hits': self.stats.cache_hits,
            'cache_misses': self.stats.cache_misses,
            'errors': self.stats.errors,
            'error_rate_percent': round((self.stats.errors / max(self.stats.total_mappings, 1)) * 100, 2),
            'uptime_seconds': (datetime.now() - self.stats.last_reset).total_seconds()
        }
    
    def reset_stats(self) -> None:
        """Resetea estadísticas"""
        self.stats = MappingStats()
        logger.info("📊 Estadísticas reseteadas")
    
    def clear_cache(self) -> None:
        """Limpia todos los caches"""
        self.mapping_cache.clear()
        self.schema_cache.clear()
        self.fhir_template_cache.clear()
        self.cache_timestamps.clear()
        logger.info("🧹 Cache limpiado completamente")


# 🚀 FUNCIONES DE UTILIDAD RÁPIDA

async def quick_map_table(db_path: str, table_name: str, limit: int = 100) -> List[MappingResult]:
    """
    ⚡ Mapeo rápido de tabla completa
    
    Args:
        db_path: Ruta a la base de datos
        table_name: Nombre de la tabla
        limit: Límite de registros
        
    Returns:
        Lista de resultados de mapeo
    """
    import sqlite3
    
    mapper = DynamicMapper(db_path)
    
    # Obtener datos SQL
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        rows = cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}").fetchall()
        sql_records = [dict(row) for row in rows]
    
    # Mapear a FHIR
    return await mapper.batch_map_sql_to_fhir(table_name, sql_records)

def create_fhir_bundle(mapping_results: List[MappingResult]) -> Dict[str, Any]:
    """
    📦 Crea un Bundle FHIR desde resultados de mapeo
    
    Args:
        mapping_results: Lista de resultados de mapeo
        
    Returns:
        Bundle FHIR completo
    """
    entries = []
    
    for result in mapping_results:
        if result.fhir_resource and result.confidence_score >= 0.5:
            entries.append({
                'resource': result.fhir_resource,
                'request': {
                    'method': 'PUT',
                    'url': f"{result.resource_type}/{result.fhir_resource.get('id', '')}"
                }
            })
    
    return {
        'resourceType': 'Bundle',
        'id': f'batch-{datetime.now().strftime("%Y%m%d%H%M%S")}',
        'type': 'batch',
        'entry': entries,
        'total': len(entries)
    }


# 📊 EJEMPLO DE USO
if __name__ == "__main__":
    async def demo():
        # Ejemplo de uso rápido
        db_path = "path/to/medical_db.db"
        mapper = DynamicMapper(db_path)
        
        # Datos de ejemplo
        patient_data = {
            'PATI_ID': 12345,
            'PATI_NAME': 'Juan',
            'PATI_SURNAME_1': 'Pérez',
            'PATI_BIRTH_DATE': '1985-03-15',
            'PATI_GENDER': 'M',
            'MTIME': '2024-01-15T10:30:00'
        }
        
        # Mapear a FHIR
        result = await mapper.map_sql_to_fhir('PATI_PATIENTS', patient_data)
        
        print(f"✅ Mapeo completado en {result.mapping_time_ms:.2f}ms")
        print(f"🎯 Confianza: {result.confidence_score:.2f}")
        print(f"📄 Recurso FHIR: {result.resource_type}")
        
        # Estadísticas
        stats = mapper.get_performance_stats()
        print(f"📊 Estadísticas: {stats}")
    
    # Ejecutar demo
    # asyncio.run(demo())
    pass 