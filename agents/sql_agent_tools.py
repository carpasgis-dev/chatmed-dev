import sqlite3
import json
import threading
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
import asyncio
import hashlib
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)

class SchemaIntrospectionError(Exception):
    """Custom exception for schema introspection failures."""
    pass

class SQLLearningSystem:
    """
    Sistema de aprendizaje de errores SQL que recuerda y aprende de errores previos.
    """
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.error_patterns_file = cache_dir / "sql_error_patterns.json"
        self.success_patterns_file = cache_dir / "sql_success_patterns.json"
        self.error_history_file = cache_dir / "sql_error_history.json"
        
        # Cargar patrones existentes
        self.error_patterns = self._load_json_file(self.error_patterns_file, {})
        self.success_patterns = self._load_json_file(self.success_patterns_file, {})
        self.error_history = self._load_json_file(self.error_history_file, [])
        
        # Configuración de aprendizaje
        self.max_retry_attempts = 3
        self.learning_threshold = 0.7  # Confianza mínima para aplicar aprendizaje
        self.pattern_expiry_days = 30  # Días antes de que un patrón expire
        
        logger.info("🧠 Sistema de aprendizaje SQL inicializado")

    def _load_json_file(self, file_path: Path, default: Any) -> Any:
        """Carga archivo JSON con manejo de errores"""
        try:
            if file_path.exists():
                with file_path.open('r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"No se pudo cargar {file_path}: {e}")
        return default

    def _save_json_file(self, file_path: Path, data: Any):
        """Guarda archivo JSON con manejo de errores"""
        try:
            with file_path.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"No se pudo guardar {file_path}: {e}")

    def record_error(self, query: str, failed_sql: str, error_message: str, corrected_sql: Optional[str] = None):
        """
        Registra un error SQL para aprendizaje futuro.
        
        Args:
            query: Consulta original del usuario
            failed_sql: SQL que falló
            error_message: Mensaje de error
            corrected_sql: SQL corregido (si se generó)
        """
        try:
            # Crear hash único para el error
            error_hash = self._create_error_hash(query, failed_sql, error_message)
            
            # Extraer patrón del error
            error_pattern = self._extract_error_pattern(error_message)
            
            # Registrar en historial
            error_record = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'failed_sql': failed_sql,
                'error_message': error_message,
                'error_pattern': error_pattern,
                'error_hash': error_hash,
                'corrected_sql': corrected_sql,
                'success': corrected_sql is not None
            }
            
            self.error_history.append(error_record)
            
            # Limpiar historial antiguo (mantener solo últimos 1000 errores)
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-1000:]
            
            # Actualizar patrones de error
            self._update_error_patterns(error_pattern, error_record)
            
            # Guardar datos
            self._save_json_file(self.error_history_file, self.error_history)
            self._save_json_file(self.error_patterns_file, self.error_patterns)
            
            logger.info(f"🧠 Error registrado para aprendizaje: {error_pattern}")
            
        except Exception as e:
            logger.error(f"Error registrando error para aprendizaje: {e}")

    def record_success(self, query: str, sql: str, execution_time: float):
        """
        Registra un SQL exitoso para aprendizaje futuro.
        
        Args:
            query: Consulta original
            sql: SQL que funcionó
            execution_time: Tiempo de ejecución
        """
        try:
            # Crear hash único para el éxito
            success_hash = self._create_success_hash(query, sql)
            
            # Extraer patrón del éxito
            success_pattern = self._extract_success_pattern(query, sql)
            
            # Registrar patrón exitoso
            if success_pattern not in self.success_patterns:
                self.success_patterns[success_pattern] = {
                    'count': 0,
                    'avg_execution_time': 0,
                    'last_used': None,
                    'examples': []
                }
            
            pattern_data = self.success_patterns[success_pattern]
            pattern_data['count'] += 1
            pattern_data['last_used'] = datetime.now().isoformat()
            pattern_data['avg_execution_time'] = (
                (pattern_data['avg_execution_time'] * (pattern_data['count'] - 1) + execution_time) 
                / pattern_data['count']
            )
            
            # Mantener solo los últimos 5 ejemplos
            pattern_data['examples'].append({
                'query': query,
                'sql': sql,
                'execution_time': execution_time
            })
            if len(pattern_data['examples']) > 5:
                pattern_data['examples'] = pattern_data['examples'][-5:]
            
            # Guardar patrones exitosos
            self._save_json_file(self.success_patterns_file, self.success_patterns)
            
            logger.info(f"✅ Patrón exitoso registrado: {success_pattern}")
            
        except Exception as e:
            logger.error(f"Error registrando éxito para aprendizaje: {e}")

    def get_error_correction_suggestion(self, query: str, failed_sql: str, error_message: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene sugerencia de corrección basada en errores previos similares.
        
        Args:
            query: Consulta original
            failed_sql: SQL que falló
            error_message: Mensaje de error
            
        Returns:
            Dict con sugerencia de corrección o None si no hay patrón similar
        """
        try:
            error_pattern = self._extract_error_pattern(error_message)
            
            # Buscar patrones similares en el historial
            similar_errors = []
            for error_record in self.error_history:
                if error_record['error_pattern'] == error_pattern:
                    similar_errors.append(error_record)
            
            if not similar_errors:
                return None
            
            # Filtrar por consultas similares
            query_similarity_threshold = 0.6
            most_similar_error = None
            highest_similarity = 0
            
            for error_record in similar_errors:
                similarity = self._calculate_query_similarity(query, error_record['query'])
                if similarity > query_similarity_threshold and similarity > highest_similarity:
                    highest_similarity = similarity
                    most_similar_error = error_record
            
            if most_similar_error and most_similar_error.get('corrected_sql'):
                return {
                    'suggested_sql': most_similar_error['corrected_sql'],
                    'confidence': highest_similarity,
                    'pattern': error_pattern,
                    'based_on_query': most_similar_error['query']
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error obteniendo sugerencia de corrección: {e}")
            return None

    def get_success_pattern_suggestion(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene sugerencia basada en patrones exitosos previos.
        
        Args:
            query: Consulta original
            
        Returns:
            Dict con sugerencia de patrón exitoso o None
        """
        try:
            query_pattern = self._extract_success_pattern(query, "")
            
            # Buscar patrones exitosos similares
            best_match = None
            highest_similarity = 0
            
            for pattern, data in self.success_patterns.items():
                if data['count'] >= 2:  # Solo patrones probados
                    similarity = self._calculate_query_similarity(query_pattern, pattern)
                    if similarity > 0.7 and similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = {
                            'pattern': pattern,
                            'data': data,
                            'confidence': similarity
                        }
            
            if best_match:
                # Usar el ejemplo más reciente
                latest_example = best_match['data']['examples'][-1]
                return {
                    'suggested_sql': latest_example['sql'],
                    'confidence': best_match['confidence'],
                    'pattern': best_match['pattern'],
                    'avg_execution_time': best_match['data']['avg_execution_time']
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error obteniendo sugerencia de patrón exitoso: {e}")
            return None

    def _create_error_hash(self, query: str, failed_sql: str, error_message: str) -> str:
        """Crea hash único para un error"""
        content = f"{query}:{failed_sql}:{error_message}"
        return hashlib.md5(content.encode()).hexdigest()

    def _create_success_hash(self, query: str, sql: str) -> str:
        """Crea hash único para un éxito"""
        content = f"{query}:{sql}"
        return hashlib.md5(content.encode()).hexdigest()

    def _extract_error_pattern(self, error_message: str) -> str:
        """Extrae patrón del mensaje de error"""
        # Patrones comunes de errores SQL
        patterns = [
            r'no such column: ([^"]+)',
            r'no such table: ([^"]+)',
            r'near "([^"]+)": syntax error',
            r'UNIQUE constraint failed',
            r'NOT NULL constraint failed',
            r'FOREIGN KEY constraint failed'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                return f"{pattern}:{match.group(1)}"
        
        # Si no coincide con patrones conocidos, usar el error completo
        return f"unknown:{error_message[:50]}"

    def _extract_success_pattern(self, query: str, sql: str) -> str:
        """Extrae patrón de consulta exitosa"""
        # Simplificar la consulta para crear patrón
        query_lower = query.lower()
        
        # Identificar tipo de consulta
        if 'paciente' in query_lower or 'patient' in query_lower:
            return "patient_query"
        elif 'diabetes' in query_lower or 'diabético' in query_lower:
            return "diabetes_query"
        elif 'medicación' in query_lower or 'medication' in query_lower:
            return "medication_query"
        elif 'contar' in query_lower or 'count' in query_lower or 'cuántos' in query_lower:
            return "count_query"
        else:
            return "general_query"

    def _calculate_query_similarity(self, query1: str, query2: str) -> float:
        """Calcula similitud entre dos consultas"""
        try:
            # Normalizar consultas
            q1_words = set(re.findall(r'\w+', query1.lower()))
            q2_words = set(re.findall(r'\w+', query2.lower()))
            
            if not q1_words or not q2_words:
                return 0.0
            
            # Calcular similitud Jaccard
            intersection = len(q1_words.intersection(q2_words))
            union = len(q1_words.union(q2_words))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0

    def _update_error_patterns(self, error_pattern: str, error_record: Dict[str, Any]):
        """Actualiza patrones de error con nueva información"""
        if error_pattern not in self.error_patterns:
            self.error_patterns[error_pattern] = {
                'count': 0,
                'successful_corrections': [],
                'failed_corrections': [],
                'last_seen': None
            }
        
        pattern_data = self.error_patterns[error_pattern]
        pattern_data['count'] += 1
        pattern_data['last_seen'] = datetime.now().isoformat()
        
        # Registrar corrección si existe
        if error_record.get('corrected_sql'):
            pattern_data['successful_corrections'].append({
                'original_sql': error_record['failed_sql'],
                'corrected_sql': error_record['corrected_sql'],
                'timestamp': error_record['timestamp']
            })
        else:
            pattern_data['failed_corrections'].append({
                'original_sql': error_record['failed_sql'],
                'timestamp': error_record['timestamp']
            })
        
        # Mantener solo las últimas 10 correcciones
        if len(pattern_data['successful_corrections']) > 10:
            pattern_data['successful_corrections'] = pattern_data['successful_corrections'][-10:]
        if len(pattern_data['failed_corrections']) > 10:
            pattern_data['failed_corrections'] = pattern_data['failed_corrections'][-10:]

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de aprendizaje"""
        try:
            total_errors = len(self.error_history)
            successful_corrections = sum(1 for e in self.error_history if e.get('success', False))
            success_rate = (successful_corrections / total_errors * 100) if total_errors > 0 else 0
            
            return {
                'total_errors': total_errors,
                'successful_corrections': successful_corrections,
                'success_rate': success_rate,
                'error_patterns': len(self.error_patterns),
                'success_patterns': len(self.success_patterns),
                'recent_errors': len([e for e in self.error_history if datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(days=7)])
            }
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}

class SQLAgentTools:
    """
    SQLAgentToolsV2 - Enterprise-grade schema introspection and FK graph builder.
    Version: 2.0 - Enhanced with LLM-based validation and learning system
    """

    def __init__(self, db_path: str, cache_dir: Optional[str] = None, llm=None):
        self.db_path = Path(db_path)
        self.schema: Dict[str, List[Dict[str, Any]]] = {}
        self.fk_graph: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.RLock()
        self.cache_dir = Path(cache_dir or self.db_path.parent)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / f"schema_cache_{self.db_path.stem}.json"
        self.llm = llm  # LLM para validación inteligente
        
        # NUEVO: Sistema de aprendizaje
        self.learning_system = SQLLearningSystem(self.cache_dir)
        self.retry_count = 0
        self.max_retries = 3

    def introspect_schema(self, use_cache: bool = True) -> None:
        """
        Introspects the SQLite schema and foreign key relationships.
        If use_cache is True, attempts to load from cache first.
        """
        with self._lock:
            if use_cache and self.load_schema_cache():
                logger.info("Schema loaded from cache.")
                return

            logger.info("Introspecting database schema.")
            if not self.db_path.exists():
                raise SchemaIntrospectionError(f"Database file not found: {self.db_path}")

            try:
                conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Fetch all user tables
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
                )
                tables = [row['name'] for row in cursor.fetchall()]
                if not tables:
                    raise SchemaIntrospectionError("No user tables found in the database.")

                for table in tables:
                    self.schema[table] = self._get_table_info(cursor, table)
                    self.fk_graph[table] = self._get_foreign_keys(cursor, table)

                conn.close()
                self.save_schema_cache()
                logger.info("Schema introspection completed and cached.")

            except sqlite3.Error as e:
                raise SchemaIntrospectionError(f"SQLite error during introspection: {e}")

    def _get_table_info(self, cursor: sqlite3.Cursor, table: str) -> List[Dict[str, Any]]:
        """Fetch column metadata for a table with validation."""
        cursor.execute(f"PRAGMA table_info('{table}');")
        cols = cursor.fetchall()
        if not cols:
            logger.warning(f"Table '{table}' has no columns or PRAGMA failed.")
        return [
            {
                'cid': row['cid'],
                'name': row['name'],
                'type': row['type'],
                'notnull': bool(row['notnull']),
                'default': row['dflt_value'],
                'pk': bool(row['pk'])
            }
            for row in cols
        ]

    def _get_foreign_keys(self, cursor: sqlite3.Cursor, table: str) -> List[Dict[str, Any]]:
        """Fetch foreign key relationships for a table."""
        try:
            cursor.execute(f"PRAGMA foreign_key_list('{table}');")
            fks = cursor.fetchall()
            return [
                {
                    'id': fk['id'],
                    'seq': fk['seq'],
                    'table': fk['table'],
                    'from': fk['from'],
                    'to': fk['to'],
                    'on_update': fk['on_update'],
                    'on_delete': fk['on_delete'],
                    'match': fk['match']
                }
                for fk in fks
            ]
        except sqlite3.Error:
            logger.warning(f"PRAGMA foreign_key_list failed for table '{table}'. Trying fallback detection.")
            return self._fallback_detect_fks(table)

    def _fallback_detect_fks(self, table: str) -> List[Dict[str, Any]]:
        """
        Fallback detection: identifies FK relationships by naming conventions (columns ending with '_id').
        """
        detected = []
        col_names = [col['name'] for col in self.schema.get(table, [])]
        for col in col_names:
            if col.lower().endswith('_id') and col.lower() != 'id':
                ref_table = col[:-3].upper()
                if ref_table in self.schema:
                    detected.append({
                        'id': None,
                        'seq': None,
                        'table': ref_table,
                        'from': col,
                        'to': 'id',
                        'on_update': None,
                        'on_delete': None,
                        'match': None
                    })
        return detected

    def save_schema_cache(self) -> None:
        """Saves schema and fk_graph to a JSON cache."""
        data = {'schema': self.schema, 'fk_graph': self.fk_graph}
        try:
            with self._cache_file.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Schema cache saved to {self._cache_file}")
        except IOError as e:
            logger.error(f"Failed to save schema cache: {e}")

    def load_schema_cache(self) -> bool:
        """Loads schema and fk_graph from cache if it exists. Returns True if successful."""
        if not self._cache_file.exists():
            return False
        try:
            with self._cache_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
                self.schema = data.get('schema', {})
                self.fk_graph = data.get('fk_graph', {})
            return True
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load schema cache: {e}")
            return False

    def find_join_path(self, start: str, end: str) -> List[Tuple[str, str, str]]:
        """
        Finds the shortest join path between two tables using BFS on fk_graph.
        Returns a list of tuples: (from_table, column_from, to_table).
        """
        if start not in self.fk_graph or end not in self.fk_graph:
            raise SchemaIntrospectionError(f"Tables '{start}' or '{end}' not in fk_graph.")

        # BFS
        queue = [(start, [])]
        visited = set()
        while queue:
            current, path = queue.pop(0)
            if current == end:
                return path
            visited.add(current)
            for fk in self.fk_graph.get(current, []):
                neigh = fk['table']
                if neigh in visited:
                    continue
                new_path = path + [(current, fk['from'], neigh)]
                queue.append((neigh, new_path))
        raise SchemaIntrospectionError(f"No join path found between '{start}' and '{end}'")

    def get_schema(self) -> Dict[str, Any]:
        """Returns the introspected schema."""
        return self.schema

    def get_fk_graph(self) -> Dict[str, Any]:
        """Returns the foreign key relationship graph."""
        return self.fk_graph
    
    def validate_sql_uses_only_valid_tables(
        self, sql: str, valid_tables: List[str]
    ) -> Optional[str]:
        """
        Verifica que el SQL solo use tablas que estén en valid_tables.
        Devuelve mensaje de error si encuentra otra tabla, o None si todo OK.
        """
        # Normalizar SQL para mejor parsing
        sql_normalized = ' ' + sql + ' '  # Añadir espacios para boundary detection
        
        # Extraer nombres de tablas usados en FROM y JOIN
        found_tables = set()
        
        # Patrones mejorados para capturar tablas
        patterns = [
            r'\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)\b',
            r'\bJOIN\s+([A-Za-z_][A-Za-z0-9_]*)\b',
            r'\bINTO\s+([A-Za-z_][A-Za-z0-9_]*)\b',  # Para INSERT INTO
            r'\bUPDATE\s+([A-Za-z_][A-Za-z0-9_]*)\b',  # Para UPDATE
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, sql_normalized, flags=re.IGNORECASE):
                table_name = match.group(1)
                # Ignorar palabras clave SQL que podrían capturarse erróneamente
                if table_name.upper() not in ['AS', 'ON', 'AND', 'OR', 'WHERE', 'SET', 'VALUES']:
                    found_tables.add(table_name)
        
        # Comparar con válidas (case-insensitive)
        valid_tables_upper = {t.upper() for t in valid_tables}
        invalid_tables = []
        
        for table in found_tables:
            if table.upper() not in valid_tables_upper:
                invalid_tables.append(table)
        
        if invalid_tables:
            # Crear mensaje de error mejorado
            error_parts = [f"La tabla «{t}» no está permitida" for t in invalid_tables]
            error_msg = ". ".join(error_parts) + ". "
            error_msg += f"Tablas válidas: {', '.join(sorted(valid_tables))}"
            
            # Sugerir tablas similares si es posible
            suggestions = []
            for invalid in invalid_tables:
                for valid in valid_tables:
                    if (invalid.lower() in valid.lower() or 
                        valid.lower() in invalid.lower() or
                        self._calculate_similarity(invalid, valid) > 0.6):
                        suggestions.append(f"¿Quisiste decir '{valid}' en lugar de '{invalid}'?")
                        break
            
            if suggestions:
                error_msg += ". " + " ".join(suggestions)
            
            return error_msg
        
        return None
    
    def validate_sql_uses_only_valid_columns(
        self, sql: str, tables_info: Dict[str, List[str]]
    ) -> Optional[str]:
        """
        Verifica que cada referencia tabla.columna exista en tables_info.
        Devuelve mensaje de error si detecta alguna columna inválida, o None si todo OK.
        """
        errors = []
        
        # Patrón mejorado para capturar tabla.columna
        pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b'
        
        for match in re.finditer(pattern, sql):
            table = match.group(1)
            column = match.group(2)
            
            # Verificar si es una función SQL (ignorar)
            if table.upper() in ['DATE', 'DATETIME', 'TIME', 'UPPER', 'LOWER', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN']:
                continue
            
            # Buscar la tabla (case-insensitive)
            table_found = None
            for t in tables_info.keys():
                if t.upper() == table.upper():
                    table_found = t
                    break
            
            if table_found is None:
                errors.append(f"Tabla desconocida en SQL: {table}")
                continue
            
            # Obtener columnas de la tabla
            valid_columns = tables_info[table_found]
            
            # Verificar si la columna existe (case-insensitive)
            column_found = False
            for col in valid_columns:
                if col.upper() == column.upper():
                    column_found = True
                    break
            
            if not column_found:
                # Buscar columnas similares para sugerir
                similar_columns = []
                for col in valid_columns:
                    if (column.lower() in col.lower() or 
                        col.lower() in column.lower() or
                        self._calculate_similarity(column, col) > 0.6):
                        similar_columns.append(col)
                
                error_msg = f"La columna «{column}» no existe en la tabla «{table}». "
                error_msg += f"Columnas válidas: {', '.join(valid_columns[:10])}"
                
                if len(valid_columns) > 10:
                    error_msg += f" (y {len(valid_columns) - 10} más)"
                
                if similar_columns:
                    error_msg += f". ¿Quisiste decir: {', '.join(similar_columns[:3])}?"
                
                errors.append(error_msg)
        
        if errors:
            return ". ".join(errors)
        
        return None
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calcula la similitud entre dos strings usando distancia de Levenshtein normalizada."""
        try:
            # Implementación simple de similitud
            str1 = str1.lower()
            str2 = str2.lower()
            
            if str1 == str2:
                return 1.0
            
            # Verificar si uno contiene al otro
            if str1 in str2 or str2 in str1:
                return 0.8
            
            # Verificar prefijos comunes
            common_prefix_len = 0
            for i in range(min(len(str1), len(str2))):
                if str1[i] == str2[i]:
                    common_prefix_len += 1
                else:
                    break
            
            if common_prefix_len > 3:
                return 0.7
            
            return 0.0
            
        except Exception:
            return 0.0

    async def llm_validate_and_correct_tables(self, sql: str, stream_callback=None) -> str:
        """
        Usa LLM para validar y corregir nombres de tablas de forma inteligente.
        
        Args:
            sql: SQL a validar
            stream_callback: Función para mostrar progreso
            
        Returns:
            str: SQL con tablas corregidas
        """
        try:
            if not self.llm:
                logger.warning("LLM no disponible para validación de tablas")
                return sql
            
            if stream_callback:
                stream_callback("   - Validando tablas con IA...")
            
            # Obtener lista de tablas disponibles
            available_tables = list(self.schema.keys())
            
            prompt = f"""Eres un experto en SQL que valida y corrige nombres de tablas.

SQL A VALIDAR:
{sql}

TABLAS DISPONIBLES EN EL ESQUEMA:
{', '.join(available_tables)}

TAREA: Analiza el SQL y corrige cualquier nombre de tabla que no exista en el esquema.

INSTRUCCIONES:
1. Identifica todas las tablas mencionadas en el SQL
2. Verifica si cada tabla existe en la lista de tablas disponibles
3. Si encuentras tablas inexistentes, reemplázalas con la tabla correcta más apropiada
4. Para tablas de pacientes, usa PATI_PATIENTS
5. Para diagnósticos/condiciones, usa EPIS_EPISODES
6. Para observaciones/signos vitales, usa OBSE_OBSERVATIONS
7. Para citas, usa APPO_APPOINTMENTS
8. Mantén la lógica original del SQL
9. Solo corrige nombres de tablas, no cambies la estructura

EJEMPLOS DE CORRECCIÓN:
- "Pacientes" → "PATI_PATIENTS"
- "Patients" → "PATI_PATIENTS"
- "Diagnósticos" → "EPIS_EPISODES"
- "Observaciones" → "OBSE_OBSERVATIONS"

RESPUESTA:
Devuelve SOLO el SQL corregido, sin explicaciones."""

            response = await asyncio.to_thread(
                self._call_openai_native, [{"role": "user", "content": prompt}],
                task_description="Validando tablas con IA"
            )
            
            corrected_sql = self._extract_response_text(response).strip()
            corrected_sql = self._clean_llm_sql_response(corrected_sql)
            
            if corrected_sql and not corrected_sql.startswith("Error"):
                if stream_callback:
                    stream_callback("   ✅ Tablas validadas y corregidas")
                return corrected_sql
            else:
                logger.warning("LLM no pudo corregir tablas, usando SQL original")
                return sql
                
        except Exception as e:
            logger.error(f"Error en validación de tablas con LLM: {e}")
            return sql

    async def llm_validate_and_correct_columns(self, sql: str, stream_callback=None) -> str:
        """
        Valida y corrige columnas usando LLM con información específica del esquema real.
        Le da al LLM las herramientas para verificar qué columnas existen realmente.
        """
        try:
            if not self.llm:
                if stream_callback:
                    stream_callback("   - Validación básica de columnas (sin LLM)...")
                return sql
            
            if stream_callback:
                stream_callback("   - Validando columnas con IA y esquema real...")
            
            # Obtener esquema detallado para que el LLM pueda validar columnas
            schema_details = {}
            for table_name, metadata in self.schema.items():
                columns = [col['name'] for col in metadata]
                schema_details[table_name] = columns
            
            # Crear prompt que le dé al LLM las herramientas para validar columnas
            validation_prompt = f"""Eres un experto en SQL que valida columnas usando el esquema real de la base de datos.

SQL A VALIDAR:
{sql}

ESQUEMA REAL DE COLUMNAS POR TABLA:
{json.dumps(schema_details, indent=2, ensure_ascii=False)}

INSTRUCCIONES CRÍTICAS:
1. **VERIFICA CADA COLUMNA** mencionada en el SQL contra el esquema real
2. **NO INVENTES COLUMNAS** que no estén en el esquema
3. **CORRIGE NOMBRES** de columnas que no existan
4. **SUGIERE ALTERNATIVAS** si una columna no existe pero hay una similar
5. **MANTÉN LA LÓGICA** original del SQL

REGLAS DE CORRECCIÓN ESPECÍFICAS:
- Para diagnósticos: usar DIAG_OBSERVATION (existe) vs DIAGNOSIS_CODE (no existe)
- Para pacientes: usar PATI_FULL_NAME (existe) vs PATIENT_NAME (no existe)
- Para medicamentos: usar PAUM_OBSERVATIONS (existe) vs MEDICATION_NAME (no existe)
- Para observaciones: usar DIAG_OBSERVATION (existe) vs OBSERVATION_VALUE (no existe)

EJEMPLOS DE CORRECCIÓN:
- EPIS_DIAGNOSTICS.DIAGNOSIS_CODE → EPIS_DIAGNOSTICS.DIAG_OBSERVATION
- PATI_PATIENTS.PATIENT_NAME → PATI_PATIENTS.PATI_FULL_NAME
- PATI_USUAL_MEDICATION.MEDICATION_NAME → PATI_USUAL_MEDICATION.PAUM_OBSERVATIONS

TAREA:
Analiza el SQL y corrige SOLO los nombres de columnas que no existan en el esquema real.
Si todas las columnas son correctas, devuelve el SQL original sin cambios.

RESPUESTA:
Devuelve SOLO el SQL corregido, sin explicaciones adicionales."""

            try:
                if stream_callback:
                    stream_callback("   - Analizando columnas con modelo LLM...")
                    
                response = await asyncio.to_thread(
                    self._call_openai_native, [{"role": "user", "content": validation_prompt}]
                )
                
                corrected_sql = self._extract_response_text(response).strip()
                corrected_sql = self._clean_llm_sql_response(corrected_sql)
                
                if corrected_sql and corrected_sql != sql and not corrected_sql.startswith("Error"):
                    logger.info(f"🧠 LLM corrigió columnas inexistentes en el SQL")
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
                logger.warning(f"Error usando LLM para validación de columnas: {e}")
                if stream_callback:
                    stream_callback(f"   - Error al verificar columnas: {str(e)[:50]}...")
                return sql  # Fallback al SQL original
            
        except Exception as e:
            logger.error(f"Error en llm_validate_and_correct_columns: {e}")
            return sql

    async def llm_validate_sql_completeness(self, query: str, sql: str, stream_callback=None) -> Dict[str, Any]:
        """
        Usa LLM para validar si el SQL está completo para responder la consulta.
        
        Args:
            query: Consulta original del usuario
            sql: SQL generado
            stream_callback: Función para mostrar progreso
            
        Returns:
            Dict[str, Any]: Resultado de la validación
        """
        try:
            if not self.llm:
                return {"is_complete": True, "missing_tables": [], "suggestions": []}
            
            if stream_callback:
                stream_callback("   - Validando completitud del SQL...")
            
            # Obtener esquema disponible
            available_tables = list(self.schema.keys())
            
            prompt = f"""Eres un experto en SQL que valida si una consulta está completa.

CONSULTA ORIGINAL: "{query}"

SQL GENERADO:
{sql}

TABLAS DISPONIBLES:
{', '.join(available_tables)}

TAREA: Analiza si el SQL está completo para responder la consulta.

INSTRUCCIONES:
1. Analiza la consulta original para entender qué información se necesita
2. Revisa el SQL generado para ver si obtiene toda la información necesaria
3. Identifica si faltan tablas o JOINs para obtener la información completa
4. Considera el contexto médico de la consulta

EJEMPLOS:
- Consulta: "pacientes con diabetes" → Necesita PATI_PATIENTS + EPIS_EPISODES + EPIS_DIAGNOSTICS
- Consulta: "medicación de pacientes" → Necesita PATI_PATIENTS (tiene PATI_USUAL_MEDICATION)
- Consulta: "constantes vitales" → Necesita OBSE_OBSERVATIONS

RESPUESTA JSON:
{{
    "is_complete": true/false,
    "missing_tables": ["tabla1", "tabla2"],
    "suggestions": ["sugerencia1", "sugerencia2"],
    "reason": "explicación de por qué está incompleto"
}}"""

            response = await asyncio.to_thread(
                self._call_openai_native, [{"role": "user", "content": prompt}],
                task_description="Validando completitud del SQL"
            )
            
            content = self._extract_response_text(response)
            result = self._try_parse_llm_json(content)
            
            if result:
                if stream_callback:
                    if result.get("is_complete", True):
                        stream_callback("   ✅ SQL validado como completo")
                    else:
                        stream_callback(f"   ⚠️ SQL incompleto: {result.get('reason', 'Sin razón')}")
                return result
            else:
                return {"is_complete": True, "missing_tables": [], "suggestions": []}
                
        except Exception as e:
            logger.error(f"Error en validación de completitud: {e}")
            return {"is_complete": True, "missing_tables": [], "suggestions": []}

    async def llm_generate_corrected_sql(self, query: str, failed_sql: str, error_message: str, stream_callback=None) -> str:
        """
        Usa LLM para regenerar SQL corregido basado en el error y aprendizaje previo.
        
        Args:
            query: Consulta original
            failed_sql: SQL que falló
            error_message: Mensaje de error
            stream_callback: Función para mostrar progreso
            
        Returns:
            str: SQL corregido
        """
        try:
            if not self.llm:
                logger.warning("LLM no disponible para regeneración")
                return failed_sql
            
            if stream_callback:
                stream_callback("   - Regenerando SQL con información del error...")
            
            # PASO 1: Verificar si tenemos sugerencias de aprendizaje
            learning_suggestion = self.learning_system.get_error_correction_suggestion(query, failed_sql, error_message)
            
            if learning_suggestion and learning_suggestion.get('confidence', 0) > 0.8:
                if stream_callback:
                    stream_callback(f"   🧠 Aplicando corrección aprendida (confianza: {learning_suggestion['confidence']:.2f})")
                
                # Usar corrección aprendida
                corrected_sql = learning_suggestion['suggested_sql']
                
                # Registrar el uso del aprendizaje
                self.learning_system.record_error(query, failed_sql, error_message, corrected_sql)
                
                return corrected_sql
            
            # PASO 2: Usar LLM para generar corrección
            schema_info = {}
            for table_name, columns in self.schema.items():
                schema_info[table_name] = [col['name'] for col in columns]
            
            # Obtener estadísticas de aprendizaje para contexto
            learning_stats = self.learning_system.get_learning_statistics()
            
            prompt = f"""Eres un experto en SQL que corrige errores específicos usando aprendizaje previo.

CONSULTA ORIGINAL: "{query}"

SQL QUE FALLÓ:
{failed_sql}

ERROR DETECTADO:
{error_message}

ESTADÍSTICAS DE APRENDIZAJE:
- Total de errores registrados: {learning_stats.get('total_errors', 0)}
- Correcciones exitosas: {learning_stats.get('successful_corrections', 0)}
- Tasa de éxito: {learning_stats.get('success_rate', 0):.1f}%

ESQUEMA DISPONIBLE:
{json.dumps(schema_info, indent=2, ensure_ascii=False)}

TAREA: Corrige el SQL basándote en el error específico y patrones de éxito previos.

INSTRUCCIONES:
1. Analiza el error para entender qué está mal
2. Corrige el problema específico (tabla inexistente, columna incorrecta, etc.)
3. Usa solo tablas y columnas que existen en el esquema
4. Mantén la lógica original de la consulta
5. Asegúrate de que el SQL sea sintácticamente correcto
6. Considera patrones de éxito similares

EJEMPLOS DE CORRECCIÓN:
- "no such column: PATI_PATIENTS.ID" → Usar "PATI_PATIENTS.PATI_ID"
- "no such table: VITAL_SIGNS" → Usar "OBSE_OBSERVATIONS"
- "syntax error" → Corregir espacios faltantes, comas, etc.

RESPUESTA:
Devuelve SOLO el SQL corregido, sin explicaciones."""

            response = await asyncio.to_thread(
                self._call_openai_native, [{"role": "user", "content": prompt}],
                task_description="Regenerando SQL después de error"
            )
            
            corrected_sql = self._extract_response_text(response).strip()
            corrected_sql = self._clean_llm_sql_response(corrected_sql)
            
            if corrected_sql and not corrected_sql.startswith("Error"):
                if stream_callback:
                    stream_callback("   ✅ SQL regenerado exitosamente")
                
                # Registrar el error y la corrección para aprendizaje futuro
                self.learning_system.record_error(query, failed_sql, error_message, corrected_sql)
                
                return corrected_sql
            else:
                logger.warning("LLM no pudo regenerar SQL, usando fallback")
                fallback_sql = self._create_fallback_sql(query)
                
                # Registrar el error sin corrección
                self.learning_system.record_error(query, failed_sql, error_message, None)
                
                return fallback_sql
                
        except Exception as e:
            logger.error(f"Error en regeneración de SQL: {e}")
            return self._create_fallback_sql(query)

    def _call_openai_native(self, messages, task_description="Consultando modelo de IA"):
        """Función de compatibilidad para llamar a OpenAI nativo"""
        try:
            from openai import OpenAI
            native_client = OpenAI()

            if isinstance(messages, list):
                openai_messages = []
                for msg in messages:
                    role = "user"
                    content = ""
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        role = str(msg["role"])
                        content = str(msg["content"])
                    else:
                        content = str(msg)
                    openai_messages.append({"role": role, "content": content})
            else:
                content = str(messages)
                openai_messages = [{"role": "user", "content": content}]

            print(f"   💡 {task_description}...", end="", flush=True)
            
            resp_stream = native_client.chat.completions.create(
                model="gpt-4o",
                messages=openai_messages,  # type: ignore
                temperature=0.1,
                max_tokens=4000,
                stream=True,
            )
            
            stream_buffer = []
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

            class MockResponse:
                def __init__(self, content):
                    self.content = content

            return MockResponse(content)

        except Exception as e:
            error_msg = f"Error en llamada OpenAI: {str(e)}"
            print(f"   ❌ ERROR EN LLM: {error_msg}")
            logger.error(f"Error en _call_openai_native: {e}", exc_info=True)
            return MockResponse('{"success": false, "message": "Error crítico en la llamada al LLM."}')

    def _extract_response_text(self, response) -> str:
        """Extrae el texto de la respuesta del LLM"""
        if hasattr(response, 'content'):
            return response.content
        return str(response)

    def _clean_llm_sql_response(self, sql_response: str) -> str:
        """Limpia la respuesta del LLM de markdown y comentarios"""
        if not sql_response:
            return sql_response
        
        # Eliminar markdown
        if sql_response.startswith('```sql'):
            sql_response = sql_response[6:]
        if sql_response.startswith('```'):
            sql_response = sql_response[3:]
        if sql_response.endswith('```'):
            sql_response = sql_response[:-3]
        
        # Eliminar comentarios SQL
        sql_response = re.sub(r'--.*$', '', sql_response, flags=re.MULTILINE)
        sql_response = re.sub(r'/\*.*?\*/', '', sql_response, flags=re.DOTALL)
        
        # Normalizar espacios
        sql_response = re.sub(r'\s+', ' ', sql_response).strip()
        
        return sql_response

    def _try_parse_llm_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Intenta parsear JSON de respuesta del LLM"""
        try:
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error parseando JSON: {e}")
            return None

    def _create_fallback_sql(self, query: str) -> str:
        """Crea SQL de fallback básico"""
        try:
            # Buscar tabla principal
            main_table = None
            for table in self.schema.keys():
                if 'PATI' in table.upper():
                    main_table = table
                    break
            
            if not main_table:
                main_table = list(self.schema.keys())[0] if self.schema else "PATI_PATIENTS"
            
            return f"SELECT * FROM {main_table} LIMIT 10;"
            
        except Exception as e:
            logger.error(f"Error creando SQL de fallback: {e}")
            return "SELECT 1 as error_fallback;"

# Example usage:
# tools = SQLAgentToolsV2('medical.db', cache_dir='./cache')\#
# tools.introspect_schema()
# path = tools.find_join_path('PATI_PATIENTS', 'PATI_PATIENT_ALLERGIES')
# print(path)
