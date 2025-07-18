#!/usr/bin/env python3
"""
SQL Cleaner Module
==================
Centraliza todas las operaciones de limpieza y sanitización de SQL.
"""
import re
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class SQLCleaner:
    """Clase para centralizar la limpieza y sanitización de SQL."""
    
    @staticmethod
    def sanitize_for_execution(sql: str) -> str:
        """
        Sanitiza completamente el SQL para ejecución segura en SQLite.
        
        Esta es la función principal que debe usarse antes de ejecutar cualquier SQL.
        
        Args:
            sql: SQL a limpiar
            
        Returns:
            str: SQL limpio y listo para ejecutar
        """
        if not sql:
            return sql
            
        # 1. Eliminar comentarios SQL
        clean_sql = SQLCleaner.remove_comments(sql)
        
        # 2. Eliminar caracteres de control problemáticos
        clean_sql = SQLCleaner.remove_control_characters(clean_sql)
        
        # 3. Normalizar espacios
        clean_sql = SQLCleaner.normalize_whitespace(clean_sql)
        
        # 4. Eliminar punto y coma al final (crítico para SQLite)
        clean_sql = SQLCleaner.remove_trailing_semicolon(clean_sql)
        
        # 5. Validación final
        clean_sql = clean_sql.strip()
        
        logger.debug(f"SQL sanitizado: {clean_sql[:100]}...")
        
        return clean_sql
    
    @staticmethod
    def remove_comments(sql: str) -> str:
        """Elimina todos los comentarios SQL."""
        # Eliminar comentarios de línea (--)
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        
        # Eliminar comentarios multilínea (/* */)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        
        return sql
    
    @staticmethod
    def remove_control_characters(sql: str) -> str:
        """Elimina caracteres de control que pueden causar problemas."""
        # Eliminar caracteres de control ASCII (0x00-0x1f, 0x7f-0x9f)
        sql = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sql)
        return sql
    
    @staticmethod
    def normalize_whitespace(sql: str) -> str:
        """Normaliza espacios en blanco múltiples."""
        # Convertir múltiples espacios/tabs/newlines en un solo espacio
        sql = re.sub(r'\s+', ' ', sql)
        return sql.strip()
    
    @staticmethod
    def remove_trailing_semicolon(sql: str) -> str:
        """Elimina punto y coma al final (SQLite no lo acepta con execute())."""
        return sql.rstrip().rstrip(';')
    
    @staticmethod
    def clean_llm_response(sql_response: str) -> str:
        """
        Limpia respuestas de LLM que pueden contener markdown o formato extra.
        
        Args:
            sql_response: Respuesta cruda del LLM
            
        Returns:
            str: SQL limpio
        """
        if not sql_response:
            return ""
            
        cleaned = sql_response.strip()
        
        # Eliminar bloques de código markdown
        cleaned = re.sub(r'^```[a-zA-Z]*\n', '', cleaned)
        cleaned = re.sub(r'\n```$', '', cleaned)
        cleaned = re.sub(r'^```', '', cleaned)
        cleaned = re.sub(r'```$', '', cleaned)
        
        # Aplicar limpieza estándar
        return SQLCleaner.sanitize_for_execution(cleaned)
    
    @staticmethod
    def split_statements(sql: str) -> List[str]:
        """
        Divide SQL en múltiples sentencias si las hay.
        
        Args:
            sql: SQL que puede contener múltiples sentencias
            
        Returns:
            Lista de sentencias SQL individuales
        """
        # Primero limpiar el SQL
        clean_sql = SQLCleaner.remove_comments(sql)
        
        # Dividir por punto y coma
        statements = [stmt.strip() for stmt in clean_sql.split(';') if stmt.strip()]
        
        return statements
    
    @staticmethod
    def validate_single_statement(sql: str) -> Tuple[bool, Optional[str]]:
        """
        Valida que el SQL contenga solo una sentencia.
        
        Returns:
            Tuple[bool, Optional[str]]: (es_válido, mensaje_error)
        """
        statements = SQLCleaner.split_statements(sql)
        
        if len(statements) == 0:
            return False, "SQL vacío"
        elif len(statements) > 1:
            return False, f"Múltiples sentencias detectadas ({len(statements)})"
        else:
            return True, None
    
    @staticmethod
    def fix_common_syntax_errors(sql: str) -> str:
        """Corrige errores de sintaxis SQL comunes."""
        if not sql:
            return sql
            
        # 1. Corregir comas finales antes de FROM
        sql = re.sub(r',\s*FROM\b', ' FROM', sql, flags=re.IGNORECASE)
        
        # 2. Corregir WHERE vacío
        sql = re.sub(r'\bWHERE\s*(?:ORDER|GROUP|LIMIT|;|$)', 
                     lambda m: m.group(0).replace('WHERE', ''), 
                     sql, flags=re.IGNORECASE)
        
        # 3. Corregir paréntesis desbalanceados en IN ()
        sql = re.sub(r'\bIN\s*\(\s*\)', 'IN (NULL)', sql, flags=re.IGNORECASE)
        
        return sql 