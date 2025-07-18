#!/usr/bin/env python3
"""
Sistema Inteligente de Gestión de Esquemas de Base de Datos
Detecta automáticamente problemas y los corrige usando IA.
"""

import logging
import sqlite3
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI
from datetime import datetime

logger = logging.getLogger(__name__)

class IntelligentSchemaManager:
    """
    Gestor inteligente de esquemas que detecta y corrige problemas automáticamente
    usando análisis de errores y generación dinámica de SQL.
    """
    
    def __init__(self, db_path: str, openai_client: Optional[OpenAI] = None):
        self.db_path = db_path
        self.openai_client = openai_client or OpenAI()
        self.logger = logger
        self.error_patterns = self._load_error_patterns()
        self.schema_cache = {}
        
    def _load_error_patterns(self) -> Dict[str, Any]:
        """Patrones de errores conocidos y sus soluciones."""
        return {
            "no such table": {
                "pattern": r"no such table: (\w+)",
                "type": "missing_table",
                "priority": "high"
            },
            "no such column": {
                "pattern": r"no such column: (\w+)",
                "type": "missing_column", 
                "priority": "high"
            }
        }
    
    def analyze_error(self, error_message: str) -> Dict[str, Any]:
        """
        Analiza un error de base de datos usando IA para determinar la causa y solución.
        """
        try:
            prompt = f"""
            Eres un experto en bases de datos SQLite. Analiza este error y proporciona:
            
            ERROR: {error_message}
            
            ANÁLISIS REQUERIDO:
            1. TIPO DE ERROR (uno de):
               - missing_table: Falta tabla
               - missing_column: Falta columna
               - duplicate_table: Tabla duplicada
               - duplicate_column: Columna duplicada
               - fk_constraint: Error de clave foránea
               - syntax_error: Error de sintaxis
               - other: Otro tipo
            
            2. ENTIDAD AFECTADA:
               - Nombre de tabla o columna
            
            3. SOLUCIÓN RECOMENDADA:
               - SQL para corregir el problema
            
            Responde en formato JSON:
            {{
                "error_type": "tipo",
                "affected_entity": "entidad",
                "solution_sql": "SQL de solución",
                "confidence": 0.95
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            if not content:
                return self._fallback_error_analysis(error_message)
            
            import json
            analysis = json.loads(content)
            
            self.logger.info(f"🧠 [SCHEMA] Análisis inteligente: {analysis['error_type']} - {analysis['affected_entity']}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error en análisis inteligente: {e}")
            return self._fallback_error_analysis(error_message)
    
    def _fallback_error_analysis(self, error_message: str) -> Dict[str, Any]:
        """Análisis de fallback usando patrones predefinidos."""
        for error_name, pattern_info in self.error_patterns.items():
            if error_name in error_message.lower():
                match = re.search(pattern_info["pattern"], error_message, re.IGNORECASE)
                if match:
                    entity = match.group(1)
                    return {
                        "error_type": pattern_info["type"],
                        "affected_entity": entity,
                        "solution_sql": self._generate_fallback_sql(pattern_info["type"], entity),
                        "confidence": 0.7
                    }
        
        return {
            "error_type": "unknown",
            "affected_entity": "unknown",
            "solution_sql": "",
            "confidence": 0.3
        }
    
    def _generate_fallback_sql(self, error_type: str, entity: str) -> str:
        """Genera SQL de fallback completamente genérico."""
        if error_type == "missing_table":
            # SQL genérico para cualquier tabla
            return f"CREATE TABLE IF NOT EXISTS {entity} (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP);"
        elif error_type == "missing_column":
            # Extraer tabla y columna de forma genérica
            parts = entity.split('.')
            if len(parts) == 2:
                table, column = parts
                return f"ALTER TABLE {table} ADD COLUMN {column} TEXT DEFAULT '';"
            else:
                # Fallback genérico
                return f"ALTER TABLE {entity} ADD COLUMN new_column TEXT DEFAULT '';"
        return ""
    
    def auto_fix_schema_error(self, error_message: str) -> bool:
        """
        Corrige automáticamente errores de esquema usando IA.
        """
        try:
            self.logger.info(f"🔧 [SCHEMA] Intentando auto-corrección: {error_message}")
            
            # 1. Analizar el error
            analysis = self.analyze_error(error_message)
            
            if analysis["confidence"] < 0.5:
                self.logger.warning(f"🔧 [SCHEMA] Baja confianza en análisis ({analysis['confidence']}), saltando auto-corrección")
                return False
            
            # 2. Aplicar la corrección
            if analysis["solution_sql"]:
                success = self._execute_schema_fix(analysis["solution_sql"])
                
                if success:
                    self.logger.info(f"✅ [SCHEMA] Auto-corrección exitosa: {analysis['error_type']}")
                
                return success
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error en auto-corrección: {e}")
            return False
    
    def _execute_schema_fix(self, sql: str) -> bool:
        """Ejecuta la corrección del esquema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Ejecutar múltiples comandos SQL si es necesario
                sql_commands = [cmd.strip() for cmd in sql.split(';') if cmd.strip()]
                
                for command in sql_commands:
                    if command:
                        cursor.execute(command)
                        self.logger.info(f"🔧 [SCHEMA] Ejecutado: {command[:50]}...")
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Error ejecutando corrección: {e}")
            return False 