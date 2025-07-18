#!/usr/bin/env python3
"""
Schema Discovery Tool - Herramienta de Descubrimiento de Esquemas
===============================================================

Descubre automáticamente la estructura y valores de la base de datos
para generar prompts dinámicos y precisos de forma completamente genérica.
"""

import sqlite3
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TableInfo:
    """Información de una tabla"""
    name: str
    columns: List[Dict[str, Any]]
    row_count: int
    sample_data: List[Dict[str, Any]]

@dataclass
class ForeignKeyInfo:
    """Información de claves foráneas"""
    table: str
    column: str
    references_table: str
    references_column: str

@dataclass
class ColumnValues:
    """Valores únicos de una columna"""
    table: str
    column: str
    values: List[Any]
    value_count: int

class SchemaDiscovery:
    """
    Herramienta genérica para descubrir automáticamente la estructura de la base de datos
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        
    def connect(self):
        """Conecta a la base de datos"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            return True
        except Exception as e:
            logger.error(f"Error conectando a la base de datos: {e}")
            return False
    
    def get_table_list(self) -> List[str]:
        """Obtiene lista de todas las tablas"""
        if not self.connection:
            return []
        
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in cursor.fetchall()]
    
    def get_table_info(self, table_name: str) -> Optional[TableInfo]:
        """Obtiene información completa de una tabla"""
        if not self.connection:
            return None
        
        cursor = self.connection.cursor()
        
        # Obtener estructura de columnas
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = []
        for row in cursor.fetchall():
            columns.append({
                'name': row[1],
                'type': row[2],
                'not_null': bool(row[3]),
                'default': row[4],
                'primary_key': bool(row[5])
            })
        
        # Obtener número de filas
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        
        # Obtener datos de muestra (máximo 5 filas)
        sample_data = []
        if row_count > 0:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
            rows = cursor.fetchall()
            column_names = [col['name'] for col in columns]
            
            for row in rows:
                sample_data.append(dict(zip(column_names, row)))
        
        return TableInfo(
            name=table_name,
            columns=columns,
            row_count=row_count,
            sample_data=sample_data
        )
    
    def get_foreign_keys(self, table_name: str) -> List[ForeignKeyInfo]:
        """Obtiene información de claves foráneas"""
        if not self.connection:
            return []
        
        cursor = self.connection.cursor()
        cursor.execute(f"PRAGMA foreign_key_list({table_name})")
        
        foreign_keys = []
        for row in cursor.fetchall():
            foreign_keys.append(ForeignKeyInfo(
                table=table_name,
                column=row[3],
                references_table=row[2],
                references_column=row[4]
            ))
        
        return foreign_keys
    
    def get_column_values(self, table_name: str, column_name: str, limit: int = 20) -> ColumnValues:
        """Obtiene valores únicos de una columna de forma genérica"""
        if not self.connection:
            return ColumnValues(table_name, column_name, [], 0)
        
        cursor = self.connection.cursor()
        
        # Obtener valores únicos
        cursor.execute(f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL LIMIT {limit}")
        values = [row[0] for row in cursor.fetchall()]
        
        # Obtener conteo total de valores únicos
        cursor.execute(f"SELECT COUNT(DISTINCT {column_name}) FROM {table_name} WHERE {column_name} IS NOT NULL")
        value_count = cursor.fetchone()[0]
        
        return ColumnValues(table_name, column_name, values, value_count)
    
    def discover_all_column_values(self, table_name: str) -> Dict[str, ColumnValues]:
        """Descubre valores de todas las columnas de una tabla"""
        table_info = self.get_table_info(table_name)
        if not table_info:
            return {}
        
        column_values = {}
        for col in table_info.columns:
            # Solo para columnas con pocos valores únicos (menos de 50)
            if col['type'] in ['TEXT', 'VARCHAR', 'CHAR']:
                values = self.get_column_values(table_name, col['name'], 50)
                if values.value_count <= 50:  # Solo columnas con valores limitados
                    column_values[col['name']] = values
        
        return column_values
    
    def generate_table_summary(self, table_name: str) -> str:
        """Genera un resumen genérico de una tabla para prompts"""
        table_info = self.get_table_info(table_name)
        if not table_info:
            return f"Tabla {table_name}: No disponible"
        
        # Información básica
        summary = f"Tabla {table_name}:\n"
        summary += f"- Filas: {table_info.row_count}\n"
        summary += f"- Columnas: {', '.join([col['name'] for col in table_info.columns])}\n"
        
        # Descubrir valores de columnas con valores limitados
        column_values = self.discover_all_column_values(table_name)
        if column_values:
            summary += "- Valores de columnas:\n"
            for col_name, values in column_values.items():
                if values.values:
                    summary += f"  * {col_name}: {', '.join(map(str, values.values[:10]))}"
                    if values.value_count > 10:
                        summary += f" (y {values.value_count - 10} más)"
                    summary += "\n"
        
        return summary
    
    def generate_sql_prompt_context(self, relevant_tables: List[str]) -> str:
        """Genera contexto genérico para prompts SQL basado en tablas relevantes"""
        context = "ESQUEMA DE BASE DE DATOS:\n\n"
        
        for table_name in relevant_tables:
            context += self.generate_table_summary(table_name) + "\n"
        
        # Agregar información de relaciones
        context += "\nRELACIONES IMPORTANTES:\n"
        for table_name in relevant_tables:
            foreign_keys = self.get_foreign_keys(table_name)
            for fk in foreign_keys:
                context += f"- {table_name}.{fk.column} → {fk.references_table}.{fk.references_column}\n"
        
        return context
    
    def discover_table_relationships(self) -> Dict[str, List[ForeignKeyInfo]]:
        """Descubre todas las relaciones entre tablas"""
        tables = self.get_table_list()
        relationships = {}
        
        for table in tables:
            foreign_keys = self.get_foreign_keys(table)
            if foreign_keys:
                relationships[table] = foreign_keys
        
        return relationships
    
    def find_related_tables(self, table_name: str) -> List[str]:
        """Encuentra tablas relacionadas a través de claves foráneas"""
        relationships = self.discover_table_relationships()
        related = set()
        
        # Tablas que referencian a esta tabla
        for table, fks in relationships.items():
            for fk in fks:
                if fk.references_table == table_name:
                    related.add(table)
        
        # Tablas que esta tabla referencia
        if table_name in relationships:
            for fk in relationships[table_name]:
                related.add(fk.references_table)
        
        return list(related)
    
    def get_table_sample_data(self, table_name: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Obtiene datos de muestra de una tabla"""
        if not self.connection:
            return []
        
        cursor = self.connection.cursor()
        cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
        rows = cursor.fetchall()
        
        # Obtener nombres de columnas
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        
        return [dict(zip(columns, row)) for row in rows]
    
    def close(self):
        """Cierra la conexión"""
        if self.connection:
            self.connection.close()

# Función de utilidad para usar en agentes
def get_dynamic_sql_context(db_path: str, relevant_tables: List[str]) -> str:
    """
    Obtiene contexto dinámico genérico para prompts SQL
    
    Args:
        db_path: Ruta a la base de datos
        relevant_tables: Lista de tablas relevantes
        
    Returns:
        str: Contexto formateado para prompts SQL
    """
    discovery = SchemaDiscovery(db_path)
    if discovery.connect():
        try:
            return discovery.generate_sql_prompt_context(relevant_tables)
        finally:
            discovery.close()
    return "Error: No se pudo conectar a la base de datos"

def discover_table_structure(db_path: str, table_name: str) -> Dict[str, Any]:
    """
    Descubre la estructura completa de una tabla
    
    Args:
        db_path: Ruta a la base de datos
        table_name: Nombre de la tabla
        
    Returns:
        Dict con información de la tabla
    """
    discovery = SchemaDiscovery(db_path)
    if discovery.connect():
        try:
            table_info = discovery.get_table_info(table_name)
            column_values = discovery.discover_all_column_values(table_name)
            foreign_keys = discovery.get_foreign_keys(table_name)
            
            return {
                'table_info': table_info,
                'column_values': column_values,
                'foreign_keys': foreign_keys
            }
        finally:
            discovery.close()
    return {} 