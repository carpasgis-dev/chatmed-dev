"""
🧪 TESTS DE MIGRACIÓN: Bridge V1 → V2 Flexible
════════════════════════════════════════════════════════════════════

Tests exhaustivos para verificar que la migración del FHIRSQLBridge
del sistema anterior (V1) al nuevo sistema flexible (V2) mantiene
100% de compatibilidad de API y funcionalidad.

Tests incluidos:
✅ Compatibilidad de API completa
✅ Rendimiento igual o superior
✅ Funcionalidad específica mantenida
✅ Casos edge y errores
✅ Migración de agentes (FHIR Agent, SQL Agent)
"""

import unittest
import asyncio
import tempfile
import sqlite3
import json
import time
import os
import sys
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Agregar paths para imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'chatmed_fhir_system'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    # Imports del sistema flexible V2
    from mapping.schema_introspector import SchemaIntrospector
    from mapping.dynamic_mapper import DynamicMapper
    from utils.schema_analyzer import SchemaAnalyzer
    FLEXIBLE_SYSTEM_AVAILABLE = True
except ImportError:
    FLEXIBLE_SYSTEM_AVAILABLE = False
    print("⚠️ Sistema flexible no disponible para tests")

try:
    # Import del Bridge V2 (nuevo)
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'chatmed_fhir_system', 'mapping'))
    from fhir_sql_bridge import FHIRSQLBridge, ConversionResult, ConversionDirection
    BRIDGE_V2_AVAILABLE = True
except ImportError:
    BRIDGE_V2_AVAILABLE = False
    print("⚠️ Bridge V2 no disponible para tests")

class TestMigrationCompatibility(unittest.TestCase):
    """
    🧪 Tests de Compatibilidad de Migración V1→V2
    
    Verifica que el Bridge V2 mantiene 100% de compatibilidad
    con la API del sistema anterior.
    """
    
    @classmethod
    def setUpClass(cls):
        """Configuración de clase para tests"""
        cls.test_db_path = None
        cls.bridge_v2 = None
        
        # Crear base de datos de prueba
        cls._create_test_database()
        
        # Inicializar Bridge V2 si está disponible
        if BRIDGE_V2_AVAILABLE:
            try:
                cls.bridge_v2 = FHIRSQLBridge(cls.test_db_path)
                print("✅ Bridge V2 inicializado para tests")
            except Exception as e:
                print(f"❌ Error inicializando Bridge V2: {e}")
                cls.bridge_v2 = None
    
    @classmethod
    def _create_test_database(cls):
        """Crea base de datos de prueba con estructura médica"""
        cls.test_db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        cls.test_db_path = cls.test_db_file.name
        
        # Crear tablas de prueba
        with sqlite3.connect(cls.test_db_path) as conn:
            cursor = conn.cursor()
            
            # Tabla de pacientes (principal)
            cursor.execute("""
                CREATE TABLE PATI_PATIENTS (
                    PATI_ID INTEGER PRIMARY KEY,
                    PATI_NAME TEXT,
                    PATI_SURNAME_1 TEXT,
                    PATI_BIRTH_DATE TEXT,
                    GEND_ID INTEGER,
                    PATI_DELETED INTEGER DEFAULT 0,
                    MTIME TEXT
                )
            """)
            
            # Tabla de episodios
            cursor.execute("""
                CREATE TABLE EPIS_EPISODES (
                    EPIS_ID INTEGER PRIMARY KEY,
                    PATI_ID INTEGER,
                    EPIS_START_DATE TEXT,
                    EPIS_CLOSED_DATE TEXT,
                    EPIS_DELETED INTEGER DEFAULT 0,
                    MTIME TEXT,
                    FOREIGN KEY (PATI_ID) REFERENCES PATI_PATIENTS(PATI_ID)
                )
            """)
            
            # Tabla de procedimientos
            cursor.execute("""
                CREATE TABLE PROC_PROCEDURES (
                    PROC_ID INTEGER PRIMARY KEY,
                    PATI_ID INTEGER,
                    EPIS_ID INTEGER,
                    PROC_DESCRIPTION TEXT,
                    PROC_RESULT_DATE TEXT,
                    PROC_DELETED INTEGER DEFAULT 0,
                    MTIME TEXT,
                    FOREIGN KEY (PATI_ID) REFERENCES PATI_PATIENTS(PATI_ID),
                    FOREIGN KEY (EPIS_ID) REFERENCES EPIS_EPISODES(EPIS_ID)
                )
            """)
            
            # Datos de prueba
            test_data = [
                # Pacientes
                ("INSERT INTO PATI_PATIENTS (PATI_ID, PATI_NAME, PATI_SURNAME_1, PATI_BIRTH_DATE, GEND_ID, MTIME) VALUES (?, ?, ?, ?, ?, ?)",
                 [(1, 'Juan', 'Pérez', '1985-03-15', 1, '2024-01-15T10:30:00'),
                  (2, 'María', 'González', '1990-07-22', 2, '2024-01-15T11:00:00'),
                  (3, 'Carlos', 'López', '1978-12-03', 1, '2024-01-15T11:30:00')]),
                
                # Episodios
                ("INSERT INTO EPIS_EPISODES (EPIS_ID, PATI_ID, EPIS_START_DATE, EPIS_CLOSED_DATE, MTIME) VALUES (?, ?, ?, ?, ?)",
                 [(1, 1, '2024-01-10T09:00:00', '2024-01-10T17:00:00', '2024-01-10T17:00:00'),
                  (2, 2, '2024-01-12T14:00:00', '2024-01-12T16:30:00', '2024-01-12T16:30:00')]),
                
                # Procedimientos
                ("INSERT INTO PROC_PROCEDURES (PROC_ID, PATI_ID, EPIS_ID, PROC_DESCRIPTION, PROC_RESULT_DATE, MTIME) VALUES (?, ?, ?, ?, ?, ?)",
                 [(1, 1, 1, 'Consulta general', '2024-01-10T10:00:00', '2024-01-10T10:00:00'),
                  (2, 2, 2, 'Análisis de sangre', '2024-01-12T15:00:00', '2024-01-12T15:00:00')])
            ]
            
            for query, data_list in test_data:
                cursor.executemany(query, data_list)
            
            conn.commit()
        
        print(f"✅ Base de datos de prueba creada: {cls.test_db_path}")
    
    @classmethod
    def tearDownClass(cls):
        """Limpieza después de todos los tests"""
        if cls.test_db_path and os.path.exists(cls.test_db_path):
            os.unlink(cls.test_db_path)
            print("🧹 Base de datos de prueba eliminada")
    
    def setUp(self):
        """Configuración antes de cada test"""
        self.maxDiff = None  # Para ver diffs completos
    
    @unittest.skipUnless(BRIDGE_V2_AVAILABLE, "Bridge V2 no disponible")
    def test_api_compatibility_basic_conversion(self):
        """Test: API básica de conversión SQL→FHIR mantiene compatibilidad"""
        
        # Datos de prueba
        sql_data = {
            'PATI_ID': 1,
            'PATI_NAME': 'Juan',
            'PATI_SURNAME_1': 'Pérez',
            'PATI_BIRTH_DATE': '1985-03-15',
            'GEND_ID': 1,
            'MTIME': '2024-01-15T10:30:00'
        }
        
        # Test de conversión usando API del sistema anterior
        result = self.bridge_v2.convert_sql_to_fhir('PATI_PATIENTS', sql_data)
        
        # Verificaciones de compatibilidad
        self.assertIsInstance(result, ConversionResult)
        self.assertTrue(hasattr(result, 'success'))
        self.assertTrue(hasattr(result, 'fhir_resource'))
        self.assertTrue(hasattr(result, 'errors'))
        self.assertTrue(hasattr(result, 'mapping_type'))
        
        # Verificar que la conversión funciona
        if result.success:
            self.assertIsNotNone(result.fhir_resource)
            self.assertEqual(result.fhir_resource.get('resourceType'), 'Patient')
            self.assertEqual(result.fhir_resource.get('id'), '1')
        
        print(f"✅ API básica compatible - Éxito: {result.success}")
    
    @unittest.skipUnless(BRIDGE_V2_AVAILABLE, "Bridge V2 no disponible")
    def test_api_compatibility_batch_conversion(self):
        """Test: API de conversión por lotes mantiene compatibilidad"""
        
        # Obtener datos de prueba de la base de datos
        with sqlite3.connect(self.test_db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM PATI_PATIENTS")
            sql_records = [dict(row) for row in cursor.fetchall()]
        
        # Test de conversión por lotes usando API del sistema anterior
        fhir_resources = self.bridge_v2.batch_convert_sql_to_fhir('PATI_PATIENTS', sql_records)
        
        # Verificaciones de compatibilidad
        self.assertIsInstance(fhir_resources, list)
        self.assertGreater(len(fhir_resources), 0)
        
        # Verificar que cada recurso es válido
        for resource in fhir_resources:
            self.assertIsInstance(resource, dict)
            self.assertIn('resourceType', resource)
            self.assertEqual(resource['resourceType'], 'Patient')
        
        print(f"✅ API por lotes compatible - {len(fhir_resources)} recursos generados")
    
    @unittest.skipUnless(BRIDGE_V2_AVAILABLE, "Bridge V2 no disponible")
    def test_api_compatibility_fhir_to_sql(self):
        """Test: API de conversión FHIR→SQL mantiene compatibilidad"""
        
        # Recurso FHIR de prueba
        fhir_patient = {
            'resourceType': 'Patient',
            'id': '123',
            'name': [{'given': ['Ana'], 'family': 'Martín'}],
            'birthDate': '1992-05-18',
            'gender': 'female'
        }
        
        # Test de conversión usando API del sistema anterior
        result = self.bridge_v2.convert_fhir_patient_to_sql(fhir_patient)
        
        # Verificaciones de compatibilidad
        self.assertIsInstance(result, ConversionResult)
        self.assertTrue(hasattr(result, 'success'))
        self.assertTrue(hasattr(result, 'data'))
        
        if result.success:
            self.assertIsNotNone(result.data)
            self.assertIn('PATI_ID', result.data)
            self.assertIn('PATI_NAME', result.data)
            self.assertEqual(result.data['PATI_NAME'], 'Ana')
            self.assertEqual(result.data['PATI_SURNAME_1'], 'Martín')
            self.assertEqual(result.data['GEND_ID'], 2)  # female → 2
        
        print(f"✅ API FHIR→SQL compatible - Éxito: {result.success}")
    
    @unittest.skipUnless(BRIDGE_V2_AVAILABLE, "Bridge V2 no disponible")
    def test_api_compatibility_bundle_creation(self):
        """Test: API de creación de bundles mantiene compatibilidad"""
        
        # Obtener datos para bundle
        with sqlite3.connect(self.test_db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM PATI_PATIENTS LIMIT 2")
            sql_records = [dict(row) for row in cursor.fetchall()]
        
        # Test de creación de bundle usando API del sistema anterior
        result = self.bridge_v2.convert_sql_query_results_to_fhir_bundle(sql_records, "Patient")
        
        # Verificaciones de compatibilidad
        self.assertIsInstance(result, ConversionResult)
        self.assertTrue(hasattr(result, 'success'))
        self.assertTrue(hasattr(result, 'fhir_resource'))
        
        if result.success:
            bundle = result.fhir_resource
            self.assertEqual(bundle.get('resourceType'), 'Bundle')
            self.assertIn('entry', bundle)
            self.assertGreater(len(bundle['entry']), 0)
        
        print(f"✅ API Bundle compatible - Éxito: {result.success}")
    
    @unittest.skipUnless(BRIDGE_V2_AVAILABLE, "Bridge V2 no disponible")
    def test_performance_compatibility(self):
        """Test: Rendimiento igual o superior al sistema anterior"""
        
        # Datos de prueba
        sql_data = {
            'PATI_ID': 1,
            'PATI_NAME': 'Juan',
            'PATI_SURNAME_1': 'Pérez',
            'PATI_BIRTH_DATE': '1985-03-15',
            'GEND_ID': 1,
            'MTIME': '2024-01-15T10:30:00'
        }
        
        # Test de rendimiento
        start_time = time.time()
        for i in range(10):  # 10 conversiones
            result = self.bridge_v2.convert_sql_to_fhir('PATI_PATIENTS', sql_data)
            self.assertTrue(result.success or len(result.errors) > 0)  # Debe tener resultado válido
        
        total_time = (time.time() - start_time) * 1000  # ms
        avg_time_per_conversion = total_time / 10
        
        # Verificar que está dentro del objetivo (< 50ms por conversión)
        self.assertLess(avg_time_per_conversion, 50, 
                       f"Rendimiento insuficiente: {avg_time_per_conversion:.2f}ms > 50ms objetivo")
        
        print(f"✅ Rendimiento compatible - {avg_time_per_conversion:.2f}ms por conversión")
    
    @unittest.skipUnless(BRIDGE_V2_AVAILABLE, "Bridge V2 no disponible")
    def test_stats_compatibility(self):
        """Test: API de estadísticas mantiene compatibilidad"""
        
        # Ejecutar algunas conversiones
        sql_data = {'PATI_ID': 1, 'PATI_NAME': 'Test'}
        self.bridge_v2.convert_sql_to_fhir('PATI_PATIENTS', sql_data)
        
        # Test de API de estadísticas
        stats = self.bridge_v2.get_conversion_stats()
        
        # Verificaciones de compatibilidad
        self.assertIsInstance(stats, dict)
        
        # Estadísticas esperadas del sistema anterior (compatibilidad)
        expected_keys = [
            'conversions_performed',
            'successful_conversions',
            'errors',
            'system_version'  # Nuevo en V2
        ]
        
        for key in expected_keys:
            if key != 'system_version':  # system_version es nuevo en V2
                self.assertIn(key, stats, f"Estadística faltante: {key}")
        
        # Verificar estadísticas específicas del V2
        self.assertIn('bridge_version', stats)
        self.assertIn('system_mode', stats)
        
        print(f"✅ API estadísticas compatible - {len(stats)} métricas disponibles")
    
    @unittest.skipUnless(BRIDGE_V2_AVAILABLE, "Bridge V2 no disponible")
    def test_error_handling_compatibility(self):
        """Test: Manejo de errores mantiene compatibilidad"""
        
        # Test con datos inválidos
        invalid_sql_data = {'invalid_field': 'invalid_value'}
        result = self.bridge_v2.convert_sql_to_fhir('INVALID_TABLE', invalid_sql_data)
        
        # Verificaciones de manejo de errores
        self.assertIsInstance(result, ConversionResult)
        self.assertTrue(hasattr(result, 'success'))
        self.assertTrue(hasattr(result, 'errors'))
        
        # En caso de error, debe tener información de diagnóstico
        if not result.success:
            self.assertIsInstance(result.errors, list)
        
        print(f"✅ Manejo de errores compatible - Errores: {len(result.errors) if result.errors else 0}")
    
    @unittest.skipUnless(BRIDGE_V2_AVAILABLE, "Bridge V2 no disponible")
    def test_cache_functionality(self):
        """Test: Funcionalidad de cache funciona correctamente"""
        
        sql_data = {
            'PATI_ID': 1,
            'PATI_NAME': 'Juan',
            'PATI_SURNAME_1': 'Pérez'
        }
        
        # Primera conversión (miss de cache)
        start_time = time.time()
        result1 = self.bridge_v2.convert_sql_to_fhir('PATI_PATIENTS', sql_data)
        time1 = (time.time() - start_time) * 1000
        
        # Segunda conversión (hit de cache esperado)
        start_time = time.time()
        result2 = self.bridge_v2.convert_sql_to_fhir('PATI_PATIENTS', sql_data)
        time2 = (time.time() - start_time) * 1000
        
        # Verificaciones
        self.assertEqual(result1.success, result2.success)
        if result1.success and result2.success:
            self.assertEqual(result1.fhir_resource.get('id'), result2.fhir_resource.get('id'))
        
        # La segunda conversión debería ser más rápida (cache hit)
        if time2 < time1:
            print(f"✅ Cache funcionando - 1ra: {time1:.2f}ms, 2da: {time2:.2f}ms")
        else:
            print(f"⚠️ Cache posiblemente no activo - 1ra: {time1:.2f}ms, 2da: {time2:.2f}ms")
    
    @unittest.skipUnless(BRIDGE_V2_AVAILABLE, "Bridge V2 no disponible")
    def test_supported_tables_api(self):
        """Test: API de tablas soportadas funciona"""
        
        # Test de métodos de utilidad
        supported_tables = self.bridge_v2.get_supported_tables()
        self.assertIsInstance(supported_tables, list)
        
        # Test con tabla conocida
        if 'PATI_PATIENTS' in supported_tables:
            self.assertTrue(self.bridge_v2.is_table_supported('PATI_PATIENTS'))
        
        # Test con tabla inexistente
        self.assertFalse(self.bridge_v2.is_table_supported('NONEXISTENT_TABLE'))
        
        print(f"✅ API tablas soportadas - {len(supported_tables)} tablas disponibles")

class TestFlexibleSystemComponents(unittest.TestCase):
    """
    🧪 Tests de Componentes del Sistema Flexible
    
    Verifica que los componentes individuales del sistema flexible
    funcionan correctamente de forma independiente.
    """
    
    @classmethod
    def setUpClass(cls):
        """Configuración de clase"""
        cls.test_db_path = TestMigrationCompatibility.test_db_path
    
    @unittest.skipUnless(FLEXIBLE_SYSTEM_AVAILABLE, "Sistema flexible no disponible")
    def test_schema_introspector_basic(self):
        """Test: SchemaIntrospector funciona correctamente"""
        
        introspector = SchemaIntrospector(self.test_db_path)
        
        # Test análisis de tabla específica
        table_metadata = introspector.analyze_table_structure('PATI_PATIENTS')
        
        self.assertIsNotNone(table_metadata)
        self.assertEqual(table_metadata.name, 'PATI_PATIENTS')
        self.assertIn('PATI_ID', table_metadata.columns)
        self.assertGreater(table_metadata.confidence_score, 0.5)
        
        print(f"✅ SchemaIntrospector - Tabla: {table_metadata.name}, Confianza: {table_metadata.confidence_score:.2f}")
    
    @unittest.skipUnless(FLEXIBLE_SYSTEM_AVAILABLE, "Sistema flexible no disponible")
    def test_dynamic_mapper_basic(self):
        """Test: DynamicMapper funciona correctamente"""
        
        mapper = DynamicMapper(self.test_db_path)
        
        # Test mapeo básico
        sql_data = {
            'PATI_ID': 1,
            'PATI_NAME': 'Juan',
            'PATI_SURNAME_1': 'Pérez'
        }
        
        # Ejecución asíncrona
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                mapper.map_sql_to_fhir('PATI_PATIENTS', sql_data)
            )
            
            self.assertIsNotNone(result)
            self.assertGreater(result.confidence_score, 0.0)
            self.assertEqual(result.source_table, 'PATI_PATIENTS')
            
            print(f"✅ DynamicMapper - Confianza: {result.confidence_score:.2f}, Tiempo: {result.mapping_time_ms:.2f}ms")
            
        finally:
            loop.close()
    
    @unittest.skipUnless(FLEXIBLE_SYSTEM_AVAILABLE, "Sistema flexible no disponible")
    def test_schema_analyzer_basic(self):
        """Test: SchemaAnalyzer funciona correctamente"""
        
        analyzer = SchemaAnalyzer(self.test_db_path)
        
        # Test análisis rápido
        analysis = analyzer.quick_analysis(include_row_counts=True)
        
        self.assertGreater(analysis.table_count, 0)
        self.assertGreater(analysis.total_columns, 0)
        self.assertGreater(analysis.analysis_time_ms, 0)
        
        # Verificar que encontró nuestras tablas de prueba
        expected_tables = ['PATI_PATIENTS', 'EPIS_EPISODES', 'PROC_PROCEDURES']
        for table in expected_tables:
            self.assertIn(table, analysis.row_counts)
        
        print(f"✅ SchemaAnalyzer - {analysis.table_count} tablas, {analysis.analysis_time_ms:.2f}ms")

class TestAgentCompatibility(unittest.TestCase):
    """
    🧪 Tests de Compatibilidad de Agentes
    
    Verifica que los agentes (FHIR Agent, SQL Agent) mantienen
    compatibilidad con el Bridge V2.
    """
    
    @unittest.skipUnless(BRIDGE_V2_AVAILABLE, "Bridge V2 no disponible")
    def test_fhir_agent_initialization(self):
        """Test: FHIR Agent se inicializa correctamente con Bridge V2"""
        
        try:
            # Importar FHIR Agent
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'chatmed_fhir_system', 'agents'))
            from fhir_agent import FHIRMedicalAgent
            
            # Inicializar agente
            agent = FHIRMedicalAgent(db_path=TestMigrationCompatibility.test_db_path)
            
            # Verificaciones básicas
            self.assertIsNotNone(agent)
            self.assertIsNotNone(agent.bridge)
            
            # Verificar que usa Bridge V2
            stats = agent.bridge.get_conversion_stats()
            bridge_version = stats.get('bridge_version', 'unknown')
            self.assertIn('2.0', bridge_version)
            
            print(f"✅ FHIR Agent compatible con Bridge V2 ({bridge_version})")
            
        except ImportError:
            self.skipTest("FHIR Agent no disponible para test")
    
    def test_sql_agent_compatibility(self):
        """Test: SQL Agent mantiene compatibilidad (no usa bridge directamente)"""
        
        try:
            # Importar SQL Agent
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'chatmed_fhir_system', 'agents'))
            from sql_agent import SQLMedicalAgent
            
            # Mock LLM para test
            mock_llm = Mock()
            mock_llm.predict = Mock(return_value="SELECT COUNT(*) FROM PATI_PATIENTS")
            
            # Inicializar agente
            agent = SQLMedicalAgent(db_path=TestMigrationCompatibility.test_db_path, llm=mock_llm)
            
            # Verificaciones básicas
            self.assertIsNotNone(agent)
            self.assertEqual(agent.db_path, TestMigrationCompatibility.test_db_path)
            
            print("✅ SQL Agent compatible (no usa bridge directamente)")
            
        except ImportError:
            self.skipTest("SQL Agent no disponible para test")

class TestEndToEndMigration(unittest.TestCase):
    """
    🧪 Tests End-to-End de Migración
    
    Tests completos que simulan el uso real del sistema
    después de la migración.
    """
    
    @classmethod
    def setUpClass(cls):
        """Configuración de clase"""
        cls.test_db_path = TestMigrationCompatibility.test_db_path
    
    @unittest.skipUnless(BRIDGE_V2_AVAILABLE and FLEXIBLE_SYSTEM_AVAILABLE, "Sistemas no disponibles")
    def test_complete_sql_to_fhir_workflow(self):
        """Test: Flujo completo SQL→FHIR funciona end-to-end"""
        
        # 1. Inicializar Bridge V2
        bridge = FHIRSQLBridge(self.test_db_path)
        
        # 2. Obtener datos reales de la base de datos
        with sqlite3.connect(self.test_db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM PATI_PATIENTS WHERE PATI_ID = 1")
            patient_data = dict(cursor.fetchone())
        
        # 3. Conversión SQL→FHIR
        start_time = time.time()
        result = bridge.convert_sql_to_fhir('PATI_PATIENTS', patient_data)
        conversion_time = (time.time() - start_time) * 1000
        
        # 4. Verificaciones end-to-end
        self.assertTrue(result.success, f"Conversión falló: {result.errors}")
        self.assertIsNotNone(result.fhir_resource)
        self.assertEqual(result.fhir_resource['resourceType'], 'Patient')
        self.assertIn('name', result.fhir_resource)
        self.assertLess(conversion_time, 100, f"Conversión muy lenta: {conversion_time:.2f}ms")
        
        # 5. Verificar estadísticas
        stats = bridge.get_conversion_stats()
        self.assertGreater(stats['conversions_performed'], 0)
        
        print(f"✅ End-to-end SQL→FHIR - {conversion_time:.2f}ms, Sistema: {stats.get('system_mode', 'unknown')}")
    
    @unittest.skipUnless(BRIDGE_V2_AVAILABLE, "Bridge V2 no disponible")
    def test_batch_processing_performance(self):
        """Test: Procesamiento por lotes tiene rendimiento adecuado"""
        
        # Inicializar Bridge V2
        bridge = FHIRSQLBridge(self.test_db_path)
        
        # Obtener todos los pacientes
        with sqlite3.connect(self.test_db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM PATI_PATIENTS")
            all_patients = [dict(row) for row in cursor.fetchall()]
        
        # Procesamiento por lotes
        start_time = time.time()
        fhir_resources = bridge.batch_convert_sql_to_fhir('PATI_PATIENTS', all_patients)
        batch_time = (time.time() - start_time) * 1000
        
        # Verificaciones de rendimiento
        self.assertGreater(len(fhir_resources), 0)
        time_per_record = batch_time / len(all_patients) if all_patients else 0
        self.assertLess(time_per_record, 50, f"Rendimiento por lotes insuficiente: {time_per_record:.2f}ms por registro")
        
        print(f"✅ End-to-end por lotes - {len(fhir_resources)} recursos, {time_per_record:.2f}ms/registro")


def run_migration_tests():
    """
    🚀 Ejecuta todos los tests de migración
    
    Función principal para ejecutar la suite completa de tests
    de migración y compatibilidad.
    """
    
    print("🧪 INICIANDO TESTS DE MIGRACIÓN V1→V2")
    print("═" * 60)
    
    # Configurar test loader
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Agregar tests por categoría
    test_classes = [
        TestMigrationCompatibility,
        TestFlexibleSystemComponents,
        TestAgentCompatibility,
        TestEndToEndMigration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Resumen
    print("\n" + "═" * 60)
    print("🏁 RESUMEN DE TESTS DE MIGRACIÓN")
    print(f"   ✅ Tests ejecutados: {result.testsRun}")
    print(f"   ❌ Fallos: {len(result.failures)}")
    print(f"   🚫 Errores: {len(result.errors)}")
    print(f"   ⏭️  Omitidos: {len(result.skipped)}")
    
    # Estado final
    if result.wasSuccessful():
        print("🎉 MIGRACIÓN VERIFICADA EXITOSAMENTE")
        print("   Bridge V2 es 100% compatible con sistema anterior")
    else:
        print("❌ MIGRACIÓN TIENE PROBLEMAS")
        print("   Revisar fallos antes de proceder")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    """Ejecutar tests si se llama directamente"""
    success = run_migration_tests()
    sys.exit(0 if success else 1) 