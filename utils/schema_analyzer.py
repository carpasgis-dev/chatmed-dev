"""
🔍 SCHEMA ANALYZER: Analizador de Esquemas Ultra-Rápido
═══════════════════════════════════════════════════════════

Analizador complementario que proporciona utilidades rápidas
para análisis de esquemas y optimización de consultas.

Diseñado para máximo rendimiento y mínima latencia.
"""

import sqlite3
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import re
import time

logger = logging.getLogger(__name__)

@dataclass
class QuickAnalysis:
    """Análisis rápido de esquema"""
    table_count: int
    total_columns: int
    primary_keys: Dict[str, str]
    foreign_keys: Dict[str, List[Tuple[str, str]]]
    indexes: Dict[str, List[str]]
    row_counts: Dict[str, int]
    analysis_time_ms: float

class SchemaAnalyzer:
    """
    🔍 Analizador de Esquemas Ultra-Rápido
    
    Proporciona análisis rápidos complementarios al SchemaIntrospector
    para optimización de rendimiento y consultas específicas.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._cache = {}
        
    def quick_analysis(self, include_row_counts: bool = False) -> QuickAnalysis:
        """
        ⚡ Análisis ultra-rápido del esquema completo
        
        Args:
            include_row_counts: Si incluir conteos de filas (más lento)
            
        Returns:
            QuickAnalysis: Análisis básico del esquema
        """
        start_time = time.time()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 1. Obtener todas las tablas
            tables = self._get_all_tables(cursor)
            
            # 2. Análisis rápido de estructura
            primary_keys = {}
            foreign_keys = defaultdict(list)
            indexes = defaultdict(list)
            total_columns = 0
            
            for table in tables:
                # Primary keys
                pk = self._get_primary_key(cursor, table)
                if pk:
                    primary_keys[table] = pk
                
                # Foreign keys
                fks = self._get_foreign_keys(cursor, table)
                foreign_keys[table] = fks
                
                # Indexes
                idx = self._get_indexes(cursor, table)
                indexes[table] = idx
                
                # Column count
                cols = self._get_column_count(cursor, table)
                total_columns += cols
            
            # 3. Row counts (opcional)
            row_counts = {}
            if include_row_counts:
                row_counts = self._get_row_counts(cursor, tables)
            
            analysis_time_ms = (time.time() - start_time) * 1000
            
            return QuickAnalysis(
                table_count=len(tables),
                total_columns=total_columns,
                primary_keys=primary_keys,
                foreign_keys=dict(foreign_keys),
                indexes=dict(indexes),
                row_counts=row_counts,
                analysis_time_ms=analysis_time_ms
            )
    
    def _get_all_tables(self, cursor: sqlite3.Cursor) -> List[str]:
        """Obtiene todas las tablas médicas"""
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        return [row[0] for row in cursor.fetchall()]
    
    def _get_primary_key(self, cursor: sqlite3.Cursor, table: str) -> Optional[str]:
        """Obtiene la clave primaria de una tabla"""
        try:
            cursor.execute(f"PRAGMA table_info({table})")
            for row in cursor.fetchall():
                if row[5]:  # pk column
                    return row[1]  # name column
            return None
        except:
            return None
    
    def _get_foreign_keys(self, cursor: sqlite3.Cursor, table: str) -> List[Tuple[str, str]]:
        """Obtiene las claves foráneas de una tabla"""
        try:
            cursor.execute(f"PRAGMA foreign_key_list({table})")
            return [(row[3], row[2]) for row in cursor.fetchall()]  # (from, table)
        except:
            return []
    
    def _get_indexes(self, cursor: sqlite3.Cursor, table: str) -> List[str]:
        """Obtiene los índices de una tabla"""
        try:
            cursor.execute(f"PRAGMA index_list({table})")
            return [row[1] for row in cursor.fetchall()]  # name column
        except:
            return []
    
    def _get_column_count(self, cursor: sqlite3.Cursor, table: str) -> int:
        """Obtiene el número de columnas de una tabla"""
        try:
            cursor.execute(f"PRAGMA table_info({table})")
            return len(cursor.fetchall())
        except:
            return 0
    
    def _get_row_counts(self, cursor: sqlite3.Cursor, tables: List[str]) -> Dict[str, int]:
        """Obtiene conteos de filas para todas las tablas"""
        row_counts = {}
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_counts[table] = cursor.fetchone()[0]
            except:
                row_counts[table] = 0
        return row_counts
    
    def get_largest_tables(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        📊 Obtiene las N tablas más grandes por número de filas
        
        Args:
            top_n: Número de tablas a retornar
            
        Returns:
            Lista de tuplas (table_name, row_count)
        """
        analysis = self.quick_analysis(include_row_counts=True)
        
        # Ordenar por número de filas
        sorted_tables = sorted(
            analysis.row_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_tables[:top_n]
    
    def find_related_tables(self, table_name: str) -> Dict[str, List[str]]:
        """
        🔗 Encuentra tablas relacionadas por claves foráneas
        
        Args:
            table_name: Nombre de la tabla base
            
        Returns:
            Dict con tablas que referencian y tablas referenciadas
        """
        analysis = self.quick_analysis()
        
        # Tablas que esta tabla referencia
        references_to = []
        for fk_col, ref_table in analysis.foreign_keys.get(table_name, []):
            references_to.append(ref_table)
        
        # Tablas que referencian a esta tabla
        referenced_by = []
        for table, fks in analysis.foreign_keys.items():
            for fk_col, ref_table in fks:
                if ref_table == table_name:
                    referenced_by.append(table)
        
        return {
            'references_to': references_to,
            'referenced_by': referenced_by
        }
    
    def suggest_optimization_queries(self, table_name: str) -> List[str]:
        """
        ⚡ Sugiere consultas optimizadas para una tabla
        
        Args:
            table_name: Nombre de la tabla
            
        Returns:
            Lista de consultas SQL optimizadas
        """
        analysis = self.quick_analysis()
        queries = []
        
        # Consulta básica con límite
        queries.append(f"SELECT * FROM {table_name} LIMIT 1000")
        
        # Si tiene clave primaria, consulta ordenada
        if table_name in analysis.primary_keys:
            pk = analysis.primary_keys[table_name]
            queries.append(f"SELECT * FROM {table_name} ORDER BY {pk} DESC LIMIT 1000")
        
        # Si tiene MTIME, consulta por fecha reciente
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [row[1] for row in cursor.fetchall()]
                
                if 'MTIME' in columns:
                    queries.append(f"""
                        SELECT * FROM {table_name} 
                        WHERE MTIME >= datetime('now', '-30 days')
                        ORDER BY MTIME DESC
                        LIMIT 1000
                    """)
                
                # Si tiene campo DELETED, excluir borrados
                if any('DELETED' in col for col in columns):
                    deleted_col = next(col for col in columns if 'DELETED' in col)
                    queries.append(f"""
                        SELECT * FROM {table_name} 
                        WHERE {deleted_col} = 0 OR {deleted_col} IS NULL
                        LIMIT 1000
                    """)
                    
            except:
                pass
        
        return queries
    
    def get_medical_priority_tables(self) -> List[str]:
        """
        🏥 Obtiene tablas médicas por orden de prioridad
        
        Returns:
            Lista de tablas ordenadas por importancia médica
        """
        analysis = self.quick_analysis(include_row_counts=True)
        
        # Sistema dinámico: prioridades calculadas automáticamente
        # Basado en número de relaciones y cantidad de datos
        
        # Calcular puntuación para cada tabla (sistema dinámico)
        table_scores = []
        for table in analysis.row_counts.keys():
            score = 0
            
            # Puntuación por número de filas (cantidad de datos)
            row_count = analysis.row_counts.get(table, 0)
            if row_count > 0:
                score += min(row_count / 1000, 20)  # Máximo 20 puntos por datos
            
            # Puntuación por relaciones (tablas centrales)
            related = self.find_related_tables(table)
            score += len(related['referenced_by']) * 5  # Más puntos si otras tablas la referencian
            score += len(related['references']) * 2   # Puntos por referencias que hace
            
            # Puntuación por estructura (claves primarias/foráneas)
            if analysis.primary_keys.get(table):
                score += 5  # Tabla bien estructurada
            if analysis.foreign_keys.get(table):
                score += len(analysis.foreign_keys[table]) * 3  # Por cada FK
            
            table_scores.append((table, score))
        
        # Ordenar por puntuación
        table_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [table for table, score in table_scores]
    
    def export_schema_summary(self, output_path: str) -> None:
        """
        📄 Exporta resumen del esquema
        
        Args:
            output_path: Ruta del archivo de salida
        """
        analysis = self.quick_analysis(include_row_counts=True)
        
        summary = {
            'schema_summary': {
                'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'database_path': self.db_path,
                'analysis_time_ms': analysis.analysis_time_ms,
                'statistics': {
                    'total_tables': analysis.table_count,
                    'total_columns': analysis.total_columns,
                    'tables_with_pk': len(analysis.primary_keys),
                    'tables_with_fk': len([t for t in analysis.foreign_keys.values() if t]),
                    'total_rows': sum(analysis.row_counts.values())
                },
                'largest_tables': self.get_largest_tables(10),
                'priority_tables': self.get_medical_priority_tables()[:20],
                'table_details': {
                    table: {
                        'primary_key': analysis.primary_keys.get(table),
                        'foreign_keys': analysis.foreign_keys.get(table, []),
                        'indexes': analysis.indexes.get(table, []),
                        'row_count': analysis.row_counts.get(table, 0)
                    }
                    for table in analysis.row_counts.keys()
                }
            }
        }
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Resumen de esquema exportado a: {output_path}")


# 🚀 FUNCIONES DE UTILIDAD

def quick_schema_stats(db_path: str) -> Dict[str, Any]:
    """
    ⚡ Estadísticas rápidas del esquema
    
    Args:
        db_path: Ruta a la base de datos
        
    Returns:
        Dict con estadísticas básicas
    """
    analyzer = SchemaAnalyzer(db_path)
    analysis = analyzer.quick_analysis(include_row_counts=True)
    
    return {
        'total_tables': analysis.table_count,
        'total_columns': analysis.total_columns,
        'total_rows': sum(analysis.row_counts.values()),
        'analysis_time_ms': analysis.analysis_time_ms,
        'largest_table': max(analysis.row_counts.items(), key=lambda x: x[1]) if analysis.row_counts else None,
        'tables_with_data': len([t for t, c in analysis.row_counts.items() if c > 0])
    }

def find_central_tables(db_path: str) -> List[str]:
    """
    🏥 Encuentra tablas centrales (más referenciadas) de forma dinámica
    
    Args:
        db_path: Ruta a la base de datos
        
    Returns:
        Lista de tablas centrales ordenadas por importancia
    """
    analyzer = SchemaAnalyzer(db_path)
    analysis = analyzer.quick_analysis()
    
    # Contar referencias por tabla (dinámico)
    reference_count = {}
    
    for table, fks in analysis.foreign_keys.items():
        for fk_col, ref_table in fks:
            if ref_table not in reference_count:
                reference_count[ref_table] = 0
            reference_count[ref_table] += 1
    
    # Ordenar por número de referencias (tablas más centrales)
    sorted_tables = sorted(reference_count.items(), key=lambda x: x[1], reverse=True)
    
    return [table for table, count in sorted_tables]


# 📊 EJEMPLO DE USO
if __name__ == "__main__":
    # Ejemplo de análisis rápido
    def demo_analysis():
        db_path = "path/to/medical_db.db"
        analyzer = SchemaAnalyzer(db_path)
        
        # Análisis rápido
        analysis = analyzer.quick_analysis(include_row_counts=True)
        print(f"📊 Análisis completado en {analysis.analysis_time_ms:.2f}ms")
        print(f"📋 {analysis.table_count} tablas, {analysis.total_columns} columnas")
        
        # Tablas más grandes
        largest = analyzer.get_largest_tables(5)
        print(f"📈 Tablas más grandes: {largest}")
        
        # Tablas prioritarias
        priority = analyzer.get_medical_priority_tables()[:10]
        print(f"🏥 Tablas prioritarias: {priority}")
        
        # Estadísticas rápidas
        stats = quick_schema_stats(db_path)
        print(f"⚡ Estadísticas: {stats}")
    
    # Ejecutar demo
    # demo_analysis()
    pass 