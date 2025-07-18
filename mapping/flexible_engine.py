"""
🚀 ChatMed Flexible Engine v2.0 - Motor Principal Flexible
===========================================================

Motor principal flexible que reemplaza el sistema hardcodeado con configuración 100% dinámica

Características principales:
- ✅ Configuración 100% dinámica vía YAML/JSON
- ✅ API compatible con sistema actual  
- ✅ Rendimiento superior con cache inteligente
- ✅ Mapeo SQL→FHIR completamente flexible
- ✅ Auto-detección de esquemas dinámicos

Basado en: FLEXIBLE_MIGRATION_PLAN.md - FASE 1: FUNDACIÓN
Autor: Carmen Pascual
Versión: 2.0 - Flexible
"""

import json
import sqlite3
import logging
import asyncio
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib
import yaml
import os

# Configuración de logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("FlexibleEngine")

class ConversionDirection(Enum):
    """Direcciones de conversión soportadas"""
    SQL_TO_FHIR = "sql_to_fhir"
    FHIR_TO_SQL = "fhir_to_sql"
    BIDIRECTIONAL = "bidirectional"

class MappingType(Enum):
    """Tipos de mapeo disponibles"""
    DETAILED = "detailed"  # Mapeo detallado específico
    PATTERN = "pattern"    # Mapeo basado en patrones
    DYNAMIC = "dynamic"    # Mapeo dinámico auto-detectado
    HYBRID = "hybrid"      # Combinación de tipos

@dataclass
class FlexibleConversionResult:
    """Resultado de conversión con metadatos completos"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    fhir_resource: Optional[Dict[str, Any]] = None
    sql_query: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    mapping_type: MappingType = MappingType.DYNAMIC
    conversion_time: float = 0.0
    cache_hit: bool = False
    validation_passed: bool = False

@dataclass
class CacheEntry:
    """Entrada de cache inteligente"""
    data: Any
    timestamp: datetime
    hit_count: int = 0
    conversion_time: float = 0.0
    
    def is_expired(self, ttl_minutes: int = 60) -> bool:
        return datetime.now() - self.timestamp > timedelta(minutes=ttl_minutes)

class FlexibleEngine:
    """
    🎯 Motor Flexible v2.0 - Core del Sistema
    
    Características principales:
    - Configuración 100% dinámica desde archivos YAML/JSON
    - Cache inteligente multinivel 
    - Auto-detección de esquemas de base de datos
    - Mapeo SQL→FHIR completamente flexible
    - Rendimiento optimizado con métricas en tiempo real
    - API compatible con sistema legacy
    """
    
    def __init__(self,
                 db_path: str,
                 config_dir: str = "config",
                 enable_cache: bool = True,
                 cache_ttl_minutes: int = 60,
                 enable_validation: bool = True):
        """
        Inicializa el Motor Flexible v2.0
        
        Args:
            db_path: Ruta a la base de datos SQLite
            config_dir: Directorio con configuraciones YAML/JSON
            enable_cache: Habilitar cache inteligente
            cache_ttl_minutes: TTL del cache en minutos
            enable_validation: Habilitar validación FHIR
        """
        logger.info("🚀 Inicializando ChatMed Flexible Engine v2.0...")
        
        # Configuración básica  
        self.db_path = db_path
        self.config_dir = config_dir
        self.enable_cache = enable_cache
        self.cache_ttl_minutes = cache_ttl_minutes
        self.enable_validation = enable_validation
        
        # Cache inteligente multinivel
        self.conversion_cache = {} if enable_cache else None
        self.schema_cache = {} if enable_cache else None
        self.config_cache = {} if enable_cache else None
        
        # Configuraciones flexibles cargadas dinámicamente
        self.mapping_rules = {}
        self.type_mappings = {}
        self.transformation_rules = {}
        self.validation_rules = {}
        
        # Métricas de rendimiento
        self.metrics = {
            'conversions_total': 0,
            'conversions_successful': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_conversion_time': 0.0,
            'config_reloads': 0,
            'auto_detections': 0
        }
        
        # Esquemas detectados dinámicamente
        self.detected_schemas = {}
        self.table_mappings = {}
        
        # Cargar configuraciones
        self._load_flexible_configurations()
        
        # Auto-detectar esquemas de BD
        self._auto_detect_database_schemas()
        
        logger.info("✅ Motor Flexible v2.0 inicializado correctamente")
        self._log_initialization_summary()
    
    def _load_flexible_configurations(self):
        """Carga configuraciones flexibles desde archivos YAML/JSON"""
        logger.info("📂 Cargando configuraciones flexibles...")
        
        config_files = {
            'mapping_rules': 'mapping_rules.yaml',
            'type_mappings': 'type_mappings.yaml', 
            'transformation_rules': 'transformation_rules.yaml',
            'validation_rules': 'validation_rules.yaml'
        }
        
        # Intentar cargar desde múltiples ubicaciones
        search_paths = [
            self.config_dir,
            f"chatmed_v2_flexible/{self.config_dir}",
            f"../chatmed_v2_flexible/{self.config_dir}",
            "chatmed_fhir_system/config",  # Fallback al sistema actual
            "../chatmed_fhir_system/config"
        ]
        
        for config_name, filename in config_files.items():
            config_loaded = False
            
            for search_path in search_paths:
                config_path = os.path.join(search_path, filename)
                
                if os.path.exists(config_path):
                    try:
                        if filename.endswith('.yaml') or filename.endswith('.yml'):
                            with open(config_path, 'r', encoding='utf-8') as f:
                                config_data = yaml.safe_load(f)
                        else:
                            with open(config_path, 'r', encoding='utf-8') as f:
                                config_data = json.load(f)
                        
                        setattr(self, config_name, config_data)
                        logger.info(f"✅ {config_name} cargado desde: {config_path}")
                        config_loaded = True
                        break
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Error cargando {config_path}: {e}")
            
            if not config_loaded:
                logger.warning(f"⚠️ {config_name} no encontrado, usando configuración por defecto")
                setattr(self, config_name, self._get_default_config(config_name))
        
        self.metrics['config_reloads'] += 1
    
    def _get_default_config(self, config_name: str) -> Dict[str, Any]:
        """Configuraciones por defecto para fallback"""
        defaults = {
            'mapping_rules': {
                'tables': {
                    'patients': {
                        'fhir_resource': 'Patient',
                        'fields': {
                            'id': {'fhir_path': 'id', 'type': 'string'},
                            'name': {'fhir_path': 'name[0].text', 'type': 'string'},
                            'birthdate': {'fhir_path': 'birthDate', 'type': 'date'}
                        }
                    }
                },
                'patterns': {
                    'id_patterns': ['*_id', 'id', '*_pk'],
                    'name_patterns': ['*_name', 'name*', '*_desc*'],
                    'date_patterns': ['*_date', '*_time', 'created*']
                }
            },
            'type_mappings': {
                'sql_to_fhir': {
                    'TEXT': 'string',
                    'VARCHAR': 'string', 
                    'INTEGER': 'integer',
                    'REAL': 'decimal',
                    'DATE': 'date',
                    'DATETIME': 'dateTime'
                }
            },
            'transformation_rules': {
                'date_format': 'YYYY-MM-DD',
                'boolean_mapping': {'1': True, '0': False, 'true': True, 'false': False}
            },
            'validation_rules': {
                'required_fields': ['id'],
                'fhir_resource_validation': True
            }
        }
        return defaults.get(config_name, {})
    
    def _auto_detect_database_schemas(self):
        """Auto-detección inteligente de esquemas de base de datos"""
        logger.info("🔍 Auto-detectando esquemas de base de datos...")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Obtener todas las tablas
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                logger.info(f"📊 Detectadas {len(tables)} tablas en la base de datos")
                
                # Analizar esquema de cada tabla
                for table_name in tables:
                    try:
                        cursor.execute(f"PRAGMA table_info({table_name})")
                        columns = cursor.fetchall()
                        
                        schema_info = {
                            'table_name': table_name,
                            'columns': {},
                            'primary_keys': [],
                            'inferred_fhir_type': self._infer_fhir_resource_type(table_name)
                        }
                        
                        for col_info in columns:
                            col_name = col_info[1]
                            col_type = col_info[2]
                            is_pk = col_info[5] == 1
                            
                            schema_info['columns'][col_name] = {
                                'sql_type': col_type,
                                'fhir_type': self._map_sql_type_to_fhir(col_type),
                                'is_primary_key': is_pk,
                                'inferred_fhir_path': self._infer_fhir_path(col_name, table_name)
                            }
                            
                            if is_pk:
                                schema_info['primary_keys'].append(col_name)
                        
                        self.detected_schemas[table_name] = schema_info
                        self.metrics['auto_detections'] += 1
                        
                    except Exception as e:
                        logger.error(f"❌ Error analizando tabla {table_name}: {e}")
                
                logger.info(f"✅ Esquemas auto-detectados para {len(self.detected_schemas)} tablas")
                
        except Exception as e:
            logger.error(f"❌ Error en auto-detección de esquemas: {e}")
    
    def _infer_fhir_resource_type(self, table_name: str) -> str:
        """Infiere el tipo de recurso FHIR basado en el nombre de tabla"""
        table_lower = table_name.lower()
        
        # Mapeos comunes
        fhir_mappings = {
            'patient': 'Patient',
            'condition': 'Condition', 
            'medication': 'MedicationRequest',
            'observation': 'Observation',
            'procedure': 'Procedure',
            'encounter': 'Encounter',
            'practitioner': 'Practitioner',
            'organization': 'Organization'
        }
        
        for key, fhir_type in fhir_mappings.items():
            if key in table_lower:
                return fhir_type
        
        return 'Basic'  # Tipo genérico por defecto
    
    def _map_sql_type_to_fhir(self, sql_type: str) -> str:
        """Mapea tipos SQL a tipos FHIR"""
        sql_type_upper = sql_type.upper()
        return self.type_mappings.get('sql_to_fhir', {}).get(sql_type_upper, 'string')
    
    def _infer_fhir_path(self, column_name: str, table_name: str) -> str:
        """Infiere la ruta FHIR basada en el nombre de columna"""
        col_lower = column_name.lower()
        
        # Mapeos comunes
        common_paths = {
            'id': 'id',
            'name': 'name[0].text',
            'first_name': 'name[0].given[0]',
            'last_name': 'name[0].family', 
            'birthdate': 'birthDate',
            'birth_date': 'birthDate',
            'gender': 'gender',
            'phone': 'telecom[0].value',
            'email': 'telecom[1].value',
            'address': 'address[0].text',
            'status': 'status'
        }
        
        return common_paths.get(col_lower, f'extension[0].valueString')
    
    async def convert_sql_to_fhir_flexible(self, 
                                          table_name: str, 
                                          sql_row: Dict[str, Any],
                                          target_resource_type: Optional[str] = None) -> FlexibleConversionResult:
        """
        🎯 Conversión flexible SQL→FHIR con cache y auto-detección
        """
        start_time = time.time()
        self.metrics['conversions_total'] += 1
        
        # Cache hit check
        if self.enable_cache and self.conversion_cache:
            cache_key = self._generate_cache_key(table_name, sql_row)
            if cache_key in self.conversion_cache:
                cached_entry = self.conversion_cache[cache_key]
                if not cached_entry.is_expired(self.cache_ttl_minutes):
                    cached_entry.hit_count += 1
                    self.metrics['cache_hits'] += 1
                    
                    result = FlexibleConversionResult(
                        success=True,
                        fhir_resource=cached_entry.data,
                        mapping_type=MappingType.HYBRID,
                        conversion_time=cached_entry.conversion_time,
                        cache_hit=True
                    )
                    return result
        
        self.metrics['cache_misses'] += 1
        
        try:
            # 1. Determinar tipo de mapeo a usar
            mapping_strategy = self._determine_mapping_strategy(table_name)
            
            # 2. Ejecutar conversión según estrategia
            if mapping_strategy == MappingType.DETAILED:
                fhir_resource = await self._convert_with_detailed_mapping(table_name, sql_row)
            elif mapping_strategy == MappingType.PATTERN:
                fhir_resource = await self._convert_with_pattern_mapping(table_name, sql_row)
            else:  # DYNAMIC
                fhir_resource = await self._convert_with_dynamic_mapping(table_name, sql_row)
            
            # 3. Validación FHIR si está habilitada
            validation_passed = True
            if self.enable_validation:
                validation_passed = self._validate_fhir_resource(fhir_resource)
            
            # 4. Guardar en cache
            conversion_time = time.time() - start_time
            if self.enable_cache and self.conversion_cache is not None:
                cache_key = self._generate_cache_key(table_name, sql_row)
                self.conversion_cache[cache_key] = CacheEntry(
                    data=fhir_resource,
                    timestamp=datetime.now(),
                    conversion_time=conversion_time
                )
            
            # 5. Actualizar métricas
            self.metrics['conversions_successful'] += 1
            self.metrics['avg_conversion_time'] = (
                (self.metrics['avg_conversion_time'] * (self.metrics['conversions_total'] - 1) + conversion_time) 
                / self.metrics['conversions_total']
            )
            
            return FlexibleConversionResult(
                success=True,
                fhir_resource=fhir_resource,
                mapping_type=mapping_strategy,
                conversion_time=conversion_time,
                validation_passed=validation_passed
            )
            
        except Exception as e:
            logger.error(f"❌ Error en conversión flexible: {e}")
            return FlexibleConversionResult(
                success=False,
                errors=[str(e)],
                conversion_time=time.time() - start_time
            )
    
    def _determine_mapping_strategy(self, table_name: str) -> MappingType:
        """Determina la estrategia de mapeo más apropiada"""
        
        # 1. Mapeo detallado si existe configuración específica
        if table_name in self.mapping_rules.get('tables', {}):
            return MappingType.DETAILED
        
        # 2. Mapeo por patrones si coincide con algún patrón
        patterns = self.mapping_rules.get('patterns', {})
        if self._matches_pattern(table_name, patterns):
            return MappingType.PATTERN
        
        # 3. Mapeo dinámico por defecto
        return MappingType.DYNAMIC
    
    def _matches_pattern(self, table_name: str, patterns: Dict[str, Any]) -> bool:
        """Verifica si una tabla coincide con algún patrón configurado"""
        # Implementación simplificada - expandir según necesidades
        return len(patterns) > 0
    
    async def _convert_with_detailed_mapping(self, table_name: str, sql_row: Dict[str, Any]) -> Dict[str, Any]:
        """Conversión usando mapeo detallado específico"""
        mapping_config = self.mapping_rules['tables'][table_name]
        fhir_resource_type = mapping_config['fhir_resource']
        
        fhir_resource = {
            'resourceType': fhir_resource_type,
            'id': sql_row.get('id', f"{table_name}_{hash(str(sql_row))}")
        }
        
        # Mapear campos según configuración
        for sql_field, field_config in mapping_config['fields'].items():
            if sql_field in sql_row:
                fhir_path = field_config['fhir_path']
                value = sql_row[sql_field]
                
                # Aplicar transformaciones si existen
                if 'transformation' in field_config:
                    value = self._apply_transformation(value, field_config['transformation'])
                
                self._set_fhir_value_by_path(fhir_resource, fhir_path, value)
        
        return fhir_resource
    
    async def _convert_with_pattern_mapping(self, table_name: str, sql_row: Dict[str, Any]) -> Dict[str, Any]:
        """Conversión usando mapeo basado en patrones"""
        fhir_resource_type = self._infer_fhir_resource_type(table_name)
        
        fhir_resource = {
            'resourceType': fhir_resource_type,
            'id': self._find_id_value(sql_row) or f"{table_name}_{hash(str(sql_row))}"
        }
        
        # Aplicar patrones configurados
        patterns = self.mapping_rules.get('patterns', {})
        
        for sql_field, value in sql_row.items():
            if value is None:
                continue
                
            fhir_path = self._match_field_to_pattern(sql_field, patterns)
            if fhir_path:
                self._set_fhir_value_by_path(fhir_resource, fhir_path, value)
        
        return fhir_resource
    
    async def _convert_with_dynamic_mapping(self, table_name: str, sql_row: Dict[str, Any]) -> Dict[str, Any]:
        """Conversión usando mapeo dinámico auto-detectado"""
        schema_info = self.detected_schemas.get(table_name)
        if not schema_info:
            raise ValueError(f"Esquema no detectado para tabla: {table_name}")
        
        fhir_resource_type = schema_info['inferred_fhir_type']
        
        fhir_resource = {
            'resourceType': fhir_resource_type,
            'id': self._find_id_value(sql_row) or f"{table_name}_{hash(str(sql_row))}"
        }
        
        # Mapear usando esquema auto-detectado
        for sql_field, value in sql_row.items():
            if value is None or sql_field not in schema_info['columns']:
                continue
            
            column_info = schema_info['columns'][sql_field]
            fhir_path = column_info['inferred_fhir_path']
            
            # Convertir valor según tipo
            converted_value = self._convert_value_by_type(value, column_info['fhir_type'])
            self._set_fhir_value_by_path(fhir_resource, fhir_path, converted_value)
        
        return fhir_resource
    
    def _apply_transformation(self, value: Any, transformation_name: str) -> Any:
        """Aplica transformación configurada a un valor"""
        transformations = self.transformation_rules
        
        if transformation_name == 'date_format':
            if isinstance(value, str):
                try:
                    dt = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                    return dt.strftime('%Y-%m-%d')
                except:
                    return value
        elif transformation_name == 'boolean':
            boolean_mapping = transformations.get('boolean_mapping', {})
            return boolean_mapping.get(str(value).lower(), value)
        
        return value
    
    def _set_fhir_value_by_path(self, fhir_resource: Dict[str, Any], fhir_path: str, value: Any):
        """Establece valor en recurso FHIR usando ruta de puntos"""
        try:
            # Implementación simplificada - expandir para rutas complejas
            if '.' not in fhir_path and '[' not in fhir_path:
                fhir_resource[fhir_path] = value
            else:
                # Para rutas complejas como 'name[0].text'
                parts = fhir_path.replace('[', '.').replace(']', '').split('.')
                current = fhir_resource
                
                for i, part in enumerate(parts[:-1]):
                    if part.isdigit():
                        continue
                    if part not in current:
                        if i < len(parts) - 2 and parts[i+1].isdigit():
                            current[part] = []
                        else:
                            current[part] = {}
                    current = current[part]
                
                current[parts[-1]] = value
        except Exception as e:
            logger.warning(f"⚠️ Error estableciendo valor FHIR: {e}")
    
    def _find_id_value(self, sql_row: Dict[str, Any]) -> Optional[str]:
        """Busca el valor ID en la fila SQL"""
        id_candidates = ['id', 'ID', 'pk', 'primary_key']
        
        for candidate in id_candidates:
            if candidate in sql_row:
                return str(sql_row[candidate])
        
        # Buscar campos que terminen en _id
        for key in sql_row.keys():
            if key.lower().endswith('_id'):
                return str(sql_row[key])
        
        return None
    
    def _match_field_to_pattern(self, field_name: str, patterns: Dict[str, Any]) -> Optional[str]:
        """Coincide campo con patrón configurado"""
        # Implementación simplificada
        field_lower = field_name.lower()
        
        if 'name' in field_lower:
            return 'name[0].text'
        elif 'date' in field_lower:
            return 'birthDate' if 'birth' in field_lower else 'date'
        elif 'phone' in field_lower:
            return 'telecom[0].value'
        elif 'email' in field_lower:
            return 'telecom[1].value'
        
        return None
    
    def _convert_value_by_type(self, value: Any, fhir_type: str) -> Any:
        """Convierte valor según el tipo FHIR"""
        if fhir_type == 'date' and isinstance(value, str):
            return value.split(' ')[0]  # Extraer solo fecha
        elif fhir_type == 'integer':
            try:
                return int(value)
            except:
                return value
        elif fhir_type == 'decimal':
            try:
                return float(value)
            except:
                return value
        
        return str(value) if value is not None else None
    
    def _validate_fhir_resource(self, fhir_resource: Dict[str, Any]) -> bool:
        """Validación básica de recurso FHIR"""
        required_fields = self.validation_rules.get('required_fields', ['id'])
        
        for field in required_fields:
            if field not in fhir_resource:
                return False
        
        return True
    
    def _generate_cache_key(self, table_name: str, sql_row: Dict[str, Any]) -> str:
        """Genera clave de cache única"""
        content = f"{table_name}_{hash(str(sorted(sql_row.items())))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de rendimiento del motor"""
        cache_hit_rate = 0.0
        if self.metrics['cache_hits'] + self.metrics['cache_misses'] > 0:
            cache_hit_rate = self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses'])
        
        return {
            **self.metrics,
            'cache_hit_rate': f"{cache_hit_rate:.1%}",
            'cache_entries': len(self.conversion_cache) if self.conversion_cache else 0,
            'detected_tables': len(self.detected_schemas),
            'config_loaded': bool(self.mapping_rules)
        }
    
    def _log_initialization_summary(self):
        """Log resumen de inicialización"""
        logger.info("📊 MOTOR FLEXIBLE v2.0 - RESUMEN:")
        logger.info(f"   🗃️ Base de datos: {self.db_path}")
        logger.info(f"   ⚡ Cache: {'✅ Habilitado' if self.enable_cache else '❌ Deshabilitado'}")
        logger.info(f"   🔍 Tablas detectadas: {len(self.detected_schemas)}")
        logger.info(f"   📋 Reglas de mapeo: {len(self.mapping_rules.get('tables', {}))}")
        logger.info(f"   🎯 Patrones: {len(self.mapping_rules.get('patterns', {}))}")
        logger.info(f"   ✅ Validación FHIR: {'✅ Habilitada' if self.enable_validation else '❌ Deshabilitada'}")


# Función de utilidad para inicialización rápida
def create_flexible_engine(db_path: str, config_dir: str = "config") -> FlexibleEngine:
    """
    Crea instancia del Motor Flexible con configuración optimizada
    """
    return FlexibleEngine(
        db_path=db_path,
        config_dir=config_dir,
        enable_cache=True,
        cache_ttl_minutes=60,
        enable_validation=True
    )


# Ejemplo de uso
async def main():
    """Función principal para pruebas"""
    engine = create_flexible_engine("database_new.sqlite3.db")
    
    # Ejemplo de conversión
    sample_data = {
        'id': 1,
        'name': 'Juan Pérez',
        'birthdate': '1980-05-15',
        'gender': 'M'
    }
    
    result = await engine.convert_sql_to_fhir_flexible('patients', sample_data)
    
    if result.success:
        print("✅ Conversión exitosa:")
        print(json.dumps(result.fhir_resource, indent=2))
        print(f"⚡ Tiempo: {result.conversion_time:.3f}s")
        print(f"📊 Métricas: {engine.get_performance_metrics()}")
    else:
        print(f"❌ Error: {result.errors}")


if __name__ == "__main__":
    asyncio.run(main()) 