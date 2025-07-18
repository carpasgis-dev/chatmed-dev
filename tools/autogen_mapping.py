#!/usr/bin/env python3
"""
Autogen Mapping Tool
====================

Uso:
    python -m chatmed_v2_flexible.tools.autogen_mapping --db database.sqlite --out chatmed_v2_flexible/config

Genera/actualiza:
  • mapping_rules.yaml
  • type_mappings.yaml (solo si falta el tipo)

No reemplaza configuraciones existentes a menos que se use --force.
"""

from __future__ import annotations

import argparse
import sqlite3
import yaml
import json
from pathlib import Path
from typing import Dict, Any
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autogen_mapping")

FHIR_MAP = {
    'patient': 'Patient',
    'condition': 'Condition',
    'medication': 'MedicationRequest',
    'observation': 'Observation',
    'procedure': 'Procedure',
}

def infer_fhir_resource(table: str) -> str:
    t = table.lower()
    for key, res in FHIR_MAP.items():
        if key in t:
            return res
    return 'Basic'

def sql_to_fhir_type(sql_type: str) -> str:
    s = sql_type.upper()
    if any(tok in s for tok in ['INT', 'NUMBER', 'BIGINT', 'SMALLINT']):
        return 'integer'
    if any(tok in s for tok in ['CHAR', 'TEXT', 'CLOB', 'VARCHAR']):
        return 'string'
    if 'DATE' in s:
        return 'date'
    if 'TIME' in s:
        return 'dateTime'
    if any(tok in s for tok in ['REAL', 'DOUBLE', 'FLOAT', 'DECIMAL', 'NUMERIC']):
        return 'decimal'
    return 'string'

def load_yaml(path: Path) -> Dict[str, Any]:
    if path.exists():
        with path.open('r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}

def main():
    parser = argparse.ArgumentParser(description="Auto-genera reglas de mapeo a partir de una BD SQLite")
    parser.add_argument('--db', required=True, help='Ruta a la base de datos SQLite')
    parser.add_argument('--out', default='chatmed_v2_flexible/config', help='Directorio de salida')
    parser.add_argument('--force', action='store_true', help='Sobrescribir reglas existentes')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping_file = out_dir / 'mapping_rules.yaml'
    type_file = out_dir / 'type_mappings.yaml'

    existing_map = load_yaml(mapping_file)
    existing_types = load_yaml(type_file)

    if 'tables' not in existing_map:
        existing_map['tables'] = {}
    if 'patterns' not in existing_map:
        existing_map['patterns'] = {}

    if 'sql_to_fhir' not in existing_types:
        existing_types['sql_to_fhir'] = {}

    with sqlite3.connect(args.db) as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [r[0] for r in cur.fetchall()]
        logger.info(f"Detectadas {len(tables)} tablas")

        for table in tables:
            if table in existing_map['tables'] and not args.force:
                logger.info(f"⏭️  Tabla {table} ya en mapping_rules, omitiendo (usa --force para sobrescribir)")
                continue
            cur.execute(f"PRAGMA table_info({table})")
            cols = cur.fetchall()
            fields: Dict[str, Any] = {}
            for cid, name, coltype, notnull, default, pk in cols:
                fhir_type = sql_to_fhir_type(coltype or '')
                existing_types['sql_to_fhir'].setdefault(coltype.upper(), fhir_type)
                if pk and name.upper().endswith('ID'):
                    path = 'id'
                else:
                    # simple snake to camel path heuristic
                    clean = re.sub(r'[^a-zA-Z0-9]', '', name)
                    path = f"extension.{clean}"
                fields[name] = {'fhir_path': path, 'type': fhir_type}

            existing_map['tables'][table] = {
                'fhir_resource': infer_fhir_resource(table),
                'fields': fields
            }

    # Guardar YAML
    with mapping_file.open('w', encoding='utf-8') as f:
        yaml.safe_dump(existing_map, f, sort_keys=False, allow_unicode=True)
    with type_file.open('w', encoding='utf-8') as f:
        yaml.safe_dump(existing_types, f, sort_keys=False, allow_unicode=True)

    logger.info(f"✅ Reglas generadas/actualizadas en {out_dir}")

if __name__ == '__main__':
    main() 