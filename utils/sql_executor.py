#!/usr/bin/env python3
"""
SQL Executor Module
===================
Maneja la ejecución de consultas SQL con reintentos y manejo robusto de errores.
"""
import sqlite3
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import time

logger = logging.getLogger(__name__)


class SQLExecutor:
    """Ejecutor de SQL con manejo robusto de errores y reintentos."""
    
    def __init__(self, db_path: str):
        """
        Inicializa el ejecutor con la ruta a la base de datos.
        
        Args:
            db_path: Ruta al archivo de base de datos SQLite
        """
        self.db_path = db_path
        
    def execute_query(self, sql: str, params: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Ejecuta una consulta SQL con manejo robusto de errores.
        
        Args:
            sql: SQL a ejecutar (debe estar ya sanitizado)
            params: Parámetros para la consulta
            
        Returns:
            Dict con los resultados o información del error
        """
        start_time = time.time()
        params = params or []
        
        try:
            # Intentar ejecutar normalmente
            results = self._execute_single_query(sql, params)
            execution_time = time.time() - start_time
            
            result = {
                'success': True,
                'data': results['data'],
                'columns': results['columns'],
                'row_count': len(results['data']),
                'execution_time': execution_time,
                'sql_query': sql
            }
            
            # Agregar lastrowid si está disponible
            if 'lastrowid' in results:
                result['lastrowid'] = results['lastrowid']
            
            return result
            
        except sqlite3.OperationalError as e:
            error_msg = str(e)
            logger.error(f"OperationalError ejecutando SQL: {error_msg}")
            
            # Manejo específico del error "one statement at a time"
            if "You can only execute one statement at a time" in error_msg:
                logger.info("Detectado error 'one statement at a time', reintentando...")
                
                # Intentar quitar punto y coma y reintentar
                if sql.rstrip().endswith(';'):
                    clean_sql = sql.rstrip().rstrip(';')
                    try:
                        results = self._execute_single_query(clean_sql, params)
                        execution_time = time.time() - start_time
                        
                        result = {
                            'success': True,
                            'data': results['data'],
                            'columns': results['columns'],
                            'row_count': len(results['data']),
                            'execution_time': execution_time,
                            'sql_query': clean_sql,
                            'warning': 'SQL corregido automáticamente (punto y coma eliminado)'
                        }
                        
                        # Agregar lastrowid si está disponible
                        if 'lastrowid' in results:
                            result['lastrowid'] = results['lastrowid']
                        
                        return result
                    except Exception as retry_error:
                        logger.error(f"Reintento falló: {retry_error}")
                        
            # Si no se pudo manejar, devolver error
            return {
                'success': False,
                'error': error_msg,
                'error_type': 'OperationalError',
                'sql_query': sql,
                'execution_time': time.time() - start_time
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error ejecutando SQL: {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'error_type': type(e).__name__,
                'sql_query': sql,
                'execution_time': time.time() - start_time
            }
    
    def _execute_single_query(self, sql: str, params: List[Any]) -> Dict[str, Any]:
        """
        Ejecuta una sola consulta SQL.
        
        Args:
            sql: SQL a ejecutar
            params: Parámetros
            
        Returns:
            Dict con datos y metadatos
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(sql, params)
            
            # Si es una consulta SELECT, obtener resultados
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                # Convertir a lista de diccionarios
                data = []
                for row in rows:
                    data.append(dict(zip(columns, row)))
                
                return {
                    'data': data,
                    'columns': columns
                }
            else:
                # Para INSERT, UPDATE, DELETE
                conn.commit()
                result = {
                    'data': [],
                    'columns': [],
                    'rows_affected': cursor.rowcount
                }
                
                # Agregar lastrowid para INSERT
                if sql.strip().upper().startswith('INSERT'):
                    result['lastrowid'] = cursor.lastrowid
                
                return result
                
        finally:
            conn.close()
    
    def execute_multiple_statements(self, statements: List[str], 
                                  params: Optional[List[List[Any]]] = None) -> Dict[str, Any]:
        """
        Ejecuta múltiples sentencias SQL.
        
        Args:
            statements: Lista de sentencias SQL
            params: Lista de listas de parámetros (una por sentencia)
            
        Returns:
            Dict con resultados agregados
        """
        start_time = time.time()
        all_results = []
        errors = []
        
        params = params or [[] for _ in statements]
        
        for i, (stmt, stmt_params) in enumerate(zip(statements, params)):
            logger.info(f"Ejecutando sentencia {i+1}/{len(statements)}")
            result = self.execute_query(stmt, stmt_params)
            
            if result['success']:
                all_results.extend(result.get('data', []))
            else:
                errors.append({
                    'statement_index': i,
                    'sql': stmt,
                    'error': result['error']
                })
        
        execution_time = time.time() - start_time
        
        if errors:
            return {
                'success': False,
                'partial_results': all_results,
                'errors': errors,
                'execution_time': execution_time
            }
        else:
            return {
                'success': True,
                'data': all_results,
                'statement_count': len(statements),
                'execution_time': execution_time
            }
    
    def validate_connection(self) -> Tuple[bool, Optional[str]]:
        """
        Valida que se pueda conectar a la base de datos.
        
        Returns:
            Tuple[bool, Optional[str]]: (es_válida, mensaje_error)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            return True, None
        except Exception as e:
            return False, str(e)
    
    def test_query_syntax(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Prueba la sintaxis de una consulta usando EXPLAIN.
        
        Args:
            sql: SQL a validar
            
        Returns:
            Tuple[bool, Optional[str]]: (es_válida, mensaje_error)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Contar placeholders para crear parámetros dummy
            placeholder_count = sql.count('?')
            dummy_params = [None] * placeholder_count
            
            cursor.execute(f"EXPLAIN {sql}", dummy_params)
            conn.close()
            
            return True, None
            
        except Exception as e:
            return False, str(e) 