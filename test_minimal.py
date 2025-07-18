#!/usr/bin/env python3
"""
🧪 Script de prueba minimalista
===============================
"""

import asyncio
import sys
import os

# Añadir el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_minimal():
    """Prueba minimalista del sistema"""
    
    print("🧪 PRUEBA MINIMALISTA")
    print("=" * 20)
    
    # Verificar que la base de datos existe
    db_path = "database_new.sqlite3.db"
    if not os.path.exists(db_path):
        print(f"❌ Base de datos no encontrada: {db_path}")
        return
    
    print("✅ Base de datos encontrada")
    
    # Verificar que las tablas de alergias existen
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Verificar tablas de alergias
        allergy_tables = ['PATI_PATIENT_ALLERGIES', 'ALLE_ALLERGY_TYPES', 'ALLE_ALLERGY_CATEGORIES']
        for table in allergy_tables:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            if cursor.fetchone():
                print(f"✅ Tabla {table} existe")
            else:
                print(f"❌ Tabla {table} NO existe")
        
        # Verificar tabla de pacientes
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='PATI_PATIENTS'")
        if cursor.fetchone():
            print("✅ Tabla PATI_PATIENTS existe")
        else:
            print("❌ Tabla PATI_PATIENTS NO existe")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error verificando tablas: {e}")
        return
    
    print("\n📊 Verificación completada")

if __name__ == "__main__":
    asyncio.run(test_minimal()) 