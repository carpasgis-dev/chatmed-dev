#!/usr/bin/env python3
"""
SQL Generator Tool - Herramienta de Generación Automática de SQL
===============================================================

Genera SQL automáticamente de forma completamente genérica usando LLM para mapeo.
"""

import re
import sqlite3
import json
from typing import Dict, List, Any, Optional, Tuple
from .schema_discovery import SchemaDiscovery, ColumnValues

class SQLGenerator:
    """
    Generador automático de SQL genérico basado en esquemas descubiertos y LLM
    """
    
    def __init__(self, db_path: str, llm=None):
        self.db_path = db_path
        self.llm = llm
        self.discovery = SchemaDiscovery(db_path)
        
    def connect(self):
        """Conecta a la base de datos"""
        return self.discovery.connect()
    
    def find_main_table(self, query: str) -> str:
        """
        Encuentra la tabla principal basada en la consulta
        
        Args:
            query: Consulta en lenguaje natural
            
        Returns:
            str: Nombre de la tabla principal
        """
        tables = self.discovery.get_table_list()
        query_lower = query.lower()
        
        # Buscar palabras clave que indiquen la tabla principal
        for table in tables:
            table_lower = table.lower()
            
            # Si la consulta menciona la tabla directamente
            if table_lower in query_lower:
                return table
            
            # Buscar patrones comunes
            if 'paciente' in query_lower and 'pati' in table_lower:
                return table
            elif 'usuario' in query_lower and 'user' in table_lower:
                return table
            elif 'medico' in query_lower and 'med' in table_lower:
                return table
        
        # Por defecto, usar la primera tabla que parezca principal
        for table in tables:
            if any(keyword in table.lower() for keyword in ['pati', 'user', 'med', 'main']):
                return table
        
        return tables[0] if tables else ""
    
    async def extract_conditions_with_llm(self, query: str, main_table: str) -> Dict[str, Any]:
        """
        Extrae condiciones usando LLM para mapeo genérico
        
        Args:
            query: Consulta en lenguaje natural
            main_table: Tabla principal
            
        Returns:
            Dict[str, Any]: Condiciones extraídas
        """
        if not self.llm:
            return self._extract_conditions_fallback(query, main_table)
        
        # Obtener estructura de la tabla principal
        table_info = self.discovery.get_table_info(main_table)
        column_values = self.discovery.discover_all_column_values(main_table) if table_info else {}
        
        # Crear prompt para LLM
        schema_info = self._get_schema_info_for_llm(main_table, column_values)
        
        prompt = f"""
Eres un experto en bases de datos médicas. Analiza esta consulta y extrae las condiciones para generar SQL.

CONSULTA: "{query}"
TABLA PRINCIPAL: {main_table}

ESQUEMA DE LA TABLA:
{schema_info}

INSTRUCCIONES:
1. Identifica qué condiciones se necesitan para esta consulta
2. Mapea los términos de la consulta a columnas y valores de la base de datos
3. Identifica si es una consulta de conteo (COUNT) o selección (SELECT)
4. Extrae filtros de edad, género, fechas, etc.

EJEMPLOS DE MAPEO:
- "mujer" → GEND_ID = 'F'
- "hombre" → GEND_ID = 'M' 
- "mayor de 40 años" → edad > 40
- "cuántos" → COUNT(*)

Responde SOLO con este JSON:
{{
  "conditions": {{
    "column_name": "value",
    "gend_id": "F"
  }},
  "operation": "count|select",
  "filters": [
    {{
      "column": "column_name",
      "operator": "=",
      "value": "value"
    }}
  ],
  "is_count": true/false,
  "age_filter": {{
    "operator": ">|<",
    "value": 40
  }}
}}
"""
        
        try:
            response = await self.llm.ainvoke(prompt)
            content = self._extract_response_text(response)
            
            # Extraer JSON de la respuesta
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group(0))
                return result
            else:
                return self._extract_conditions_fallback(query, main_table)
                
        except Exception as e:
            print(f"Error en LLM mapping: {e}")
            return self._extract_conditions_fallback(query, main_table)
    
    def _extract_conditions_fallback(self, query: str, main_table: str) -> Dict[str, Any]:
        """Fallback sin LLM - detección básica"""
        conditions = {}
        query_lower = query.lower()
        
        # Detección básica
        if any(word in query_lower for word in ['cuántos', 'cuantos', 'número', 'numero', 'total']):
            conditions['is_count'] = True
        
        if any(word in query_lower for word in ['mujer', 'mujeres', 'femenino']):
            conditions['gend_id'] = 'F'
        elif any(word in query_lower for word in ['hombre', 'hombres', 'masculino']):
            conditions['gend_id'] = 'M'
        
        return conditions
    
    def _get_schema_info_for_llm(self, table_name: str, column_values: Dict[str, Any]) -> str:
        """Genera información del esquema para el LLM"""
        table_info = self.discovery.get_table_info(table_name)
        if not table_info:
            return "Tabla no encontrada"
        
        schema_info = f"Tabla: {table_name}\nColumnas:\n"
        
        for col in table_info.columns:
            col_name = col['name']
            col_type = col['type']
            
            # Obtener valores de muestra para esta columna
            sample_values = []
            if col_name in column_values:
                col_data = column_values[col_name]
                if isinstance(col_data, ColumnValues) and col_data.values:
                    sample_values = [str(v) for v in col_data.values[:5]]  # Máximo 5 valores
            
            schema_info += f"- {col_name} ({col_type})"
            if sample_values:
                schema_info += f" - Valores: {', '.join(sample_values)}"
            schema_info += "\n"
        
        return schema_info
    
    def _extract_response_text(self, response) -> str:
        """Extrae texto de respuesta del LLM"""
        if hasattr(response, 'content'):
            return str(response.content)
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    
    def build_where_clause(self, main_table: str, conditions: Dict[str, Any]) -> List[str]:
        """
        Construye cláusula WHERE basada en condiciones del LLM
        
        Args:
            main_table: Tabla principal
            conditions: Condiciones extraídas por LLM
            
        Returns:
            List[str]: Condiciones WHERE
        """
        where_conditions = []
        
        # Obtener estructura de la tabla
        table_info = self.discovery.get_table_info(main_table)
        
        if not table_info:
            return where_conditions
        
        # Procesar condiciones directas
        if 'conditions' in conditions:
            for col_name, value in conditions['conditions'].items():
                # Verificar si la columna existe
                column_exists = any(col['name'] == col_name for col in table_info.columns)
                if column_exists:
                    if isinstance(value, (int, float)):
                        where_conditions.append(f"{main_table}.{col_name} = {value}")
                    else:
                        where_conditions.append(f"{main_table}.{col_name} = '{value}'")
        
        # Procesar filtros específicos
        if 'filters' in conditions:
            for filter_item in conditions['filters']:
                col_name = filter_item.get('column')
                operator = filter_item.get('operator', '=')
                value = filter_item.get('value')
                
                if col_name and value:
                    column_exists = any(col['name'] == col_name for col in table_info.columns)
                    if column_exists:
                        if isinstance(value, (int, float)):
                            where_conditions.append(f"{main_table}.{col_name} {operator} {value}")
                        else:
                            where_conditions.append(f"{main_table}.{col_name} {operator} '{value}'")
        
        # Procesar filtros de edad
        if 'age_filter' in conditions:
            age_filter = conditions['age_filter']
            operator = age_filter.get('operator', '>')
            value = age_filter.get('value')
            
            if value:
                # Buscar columna de fecha de nacimiento
                for col in table_info.columns:
                    if 'birth' in col['name'].lower() or 'nacimiento' in col['name'].lower():
                        where_conditions.append(f"(strftime('%Y', 'now') - strftime('%Y', {main_table}.{col['name']})) {operator} {value}")
                        break
        
        return where_conditions
    
    def generate_count_query(self, main_table: str, conditions: Dict[str, Any]) -> str:
        """
        Genera consulta COUNT basada en condiciones del LLM
        
        Args:
            main_table: Tabla principal
            conditions: Condiciones extraídas
            
        Returns:
            str: SQL generado
        """
        sql_parts = [f"SELECT COUNT(*)"]
        sql_parts.append(f"FROM {main_table}")
        
        # Construir WHERE
        where_conditions = self.build_where_clause(main_table, conditions)
        
        if where_conditions:
            sql_parts.append("WHERE " + " AND ".join(where_conditions))
        
        return " ".join(sql_parts)
    
    def generate_select_query(self, main_table: str, conditions: Dict[str, Any], search_terms: Optional[List[str]] = None) -> str:
        """
        Genera consulta SELECT basada en condiciones del LLM
        
        Args:
            main_table: Tabla principal
            conditions: Condiciones extraídas
            search_terms: Términos de búsqueda opcionales
            
        Returns:
            str: SQL generado
        """
        sql_parts = [f"SELECT {main_table}.*"]
        sql_parts.append(f"FROM {main_table}")
        
        # Construir WHERE
        where_conditions = self.build_where_clause(main_table, conditions)
        
        # Agregar búsqueda por términos si se proporcionan
        if search_terms:
            table_info = self.discovery.get_table_info(main_table)
            
            if table_info:
                # Buscar columnas de texto para búsqueda
                text_columns = [col['name'] for col in table_info.columns if col['type'] in ['TEXT', 'VARCHAR', 'CHAR']]
                
                if text_columns:
                    search_conditions = []
                    for term in search_terms:
                        term_conditions = [f"{main_table}.{col} LIKE '%{term}%'" for col in text_columns]
                        search_conditions.append(f"({' OR '.join(term_conditions)})")
                    
                    if search_conditions:
                        where_conditions.append(f"({' OR '.join(search_conditions)})")
        
        if where_conditions:
            sql_parts.append("WHERE " + " AND ".join(where_conditions))
        
        # Ordenar y limitar
        sql_parts.append(f"ORDER BY {main_table}.ID DESC")
        
        if conditions.get('limit'):
            sql_parts.append("LIMIT 10")
        else:
            sql_parts.append("LIMIT 50")
        
        return " ".join(sql_parts)
    
    async def generate_sql_from_query(self, query: str) -> str:
        """
        Genera SQL automáticamente desde una consulta en lenguaje natural usando LLM
        
        Args:
            query: Consulta en lenguaje natural
            
        Returns:
            str: SQL generado
        """
        if not self.connect():
            return "Error: No se pudo conectar a la base de datos"
        
        try:
            # Encontrar tabla principal
            main_table = self.find_main_table(query)
            if not main_table:
                return "Error: No se pudo identificar la tabla principal"
            
            # Extraer condiciones usando LLM
            conditions = await self.extract_conditions_with_llm(query, main_table)
            
            # Extraer términos de búsqueda
            words = query.split()
            search_terms = [word for word in words if word.isalpha() and len(word) > 2]
            
            # Generar SQL apropiado
            if conditions.get('is_count', False):
                return self.generate_count_query(main_table, conditions)
            else:
                return self.generate_select_query(main_table, conditions, search_terms)
                
        finally:
            self.discovery.close()

# Función de utilidad para usar en agentes
async def generate_sql_automatically(db_path: str, query: str, llm=None) -> str:
    """
    Genera SQL automáticamente para una consulta usando LLM para mapeo genérico
    
    Args:
        db_path: Ruta a la base de datos
        query: Consulta en lenguaje natural
        llm: Cliente LLM (opcional)
        
    Returns:
        str: SQL generado
    """
    generator = SQLGenerator(db_path, llm=llm)
    return await generator.generate_sql_from_query(query) 