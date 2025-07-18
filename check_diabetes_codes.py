#!/usr/bin/env python3
"""
🔍 Script para verificar códigos de diabetes en la base de datos
"""

import sqlite3
import os

def check_diabetes_codes():
    """Verifica si hay códigos de diabetes en CODR_TABULAR_DIAGNOSTICS"""
    
    # Buscar la base de datos
    db_paths = [
        "database_new.sqlite3.db",
        "../database_new.sqlite3.db",
        "chatmed_v2_flexible/database_new.sqlite3.db"
    ]
    
    db_path = None
    for path in db_paths:
        if os.path.exists(path):
            db_path = path
            break
    
    if not db_path:
        print("❌ No se encontró la base de datos")
        return
    
    print(f"🔍 Usando base de datos: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Verificar si la tabla CODR_TABULAR_DIAGNOSTICS tiene datos
        cursor.execute("SELECT COUNT(*) FROM CODR_TABULAR_DIAGNOSTICS WHERE COTA_DESCRIPTION_ES IS NOT NULL")
        count = cursor.fetchone()[0]
        print(f"📊 Registros con descripciones en CODR_TABULAR_DIAGNOSTICS: {count}")
        
        if count == 0:
            print("⚠️ La tabla CODR_TABULAR_DIAGNOSTICS está vacía o no tiene descripciones")
            print("🔍 Verificando EPIS_DIAGNOSTICS...")
            
            cursor.execute("SELECT COUNT(*) FROM EPIS_DIAGNOSTICS WHERE DIAG_OBSERVATION IS NOT NULL")
            epis_count = cursor.fetchone()[0]
            print(f"📊 Registros con DIAG_OBSERVATION: {epis_count}")
            
            if epis_count > 0:
                cursor.execute("SELECT DISTINCT DIAG_OBSERVATION FROM EPIS_DIAGNOSTICS WHERE DIAG_OBSERVATION IS NOT NULL LIMIT 10")
                examples = cursor.fetchall()
                print("📋 Ejemplos de DIAG_OBSERVATION:")
                for i, (obs,) in enumerate(examples, 1):
                    print(f"   {i}. {obs}")
        else:
            # Buscar códigos de diabetes
            diabetes_terms = ['diabetes', 'diabético', 'diabética', 'DM', 'diabetes mellitus']
            
            print("🔍 Buscando códigos de diabetes...")
            for term in diabetes_terms:
                cursor.execute("""
                    SELECT COTA_ID, COTA_DESCRIPTION_ES 
                    FROM CODR_TABULAR_DIAGNOSTICS 
                    WHERE LOWER(COTA_DESCRIPTION_ES) LIKE LOWER(?)
                """, (f'%{term}%',))
                
                results = cursor.fetchall()
                if results:
                    print(f"✅ Encontrados {len(results)} códigos para '{term}':")
                    for code, desc in results[:5]:  # Mostrar solo los primeros 5
                        print(f"   - {code}: {desc}")
                else:
                    print(f"❌ No se encontraron códigos para '{term}'")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_diabetes_codes() 