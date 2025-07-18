#!/usr/bin/env python3
"""
Test del nuevo formato de terminal mejorado para ChatMed v2.0
============================================================

Este script demuestra cómo se verá el nuevo formato de terminal
para las respuestas de operaciones clínicas.
"""

import sys
import os

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(__file__))

from chat_real import RealChatInterface

def test_new_format():
    """Prueba el nuevo formato de terminal con una respuesta simulada."""
    
    # Crear instancia de la interfaz
    chat = RealChatInterface()
    
    # Respuesta simulada de una operación clínica exitosa
    sample_response = """
✅ Operación completada exitosamente
   - Paciente creado con ID: 1751178298
   - Nombre: lamine yamal
   - Edad: 72 años
   - Diagnósticos guardados: 1
   - Medicamentos guardados: 6
   - Observaciones guardadas: 3
   - Total de registros clínicos añadidos: 10
   - ⚠️ Errores encontrados: 4
     - Condición 'Diabetes mellitus tipo 2 (DM2)': No se pudo parsear la respuesta del LLM.
     - Condición 'Cardiopatía isquémica': Error de BD en transacción: UNIQUE constraint failed: ACCI_PATIENT_CONDITIONS.APCO_ID
     - Condición 'Infarto agudo de miocardio sin elevación del ST': Error de BD en transacción: UNIQUE constraint failed: ACCI_PATIENT_CONDITIONS.APCO_ID
     - Condición 'Infección por S. aureus': Error de BD en transacción: UNIQUE constraint failed: ACCI_PATIENT_CONDITIONS.APCO_ID
"""
    
    print("🔬 Probando nuevo formato de terminal...")
    print("=" * 60)
    
    # Formatear la respuesta
    formatted_response = chat.format_clinical_operation_response(sample_response)
    
    # Mostrar el resultado
    print(formatted_response)
    
    print("\n" + "=" * 60)
    print("✅ Prueba completada. El nuevo formato debería verse mucho más limpio y profesional.")

def test_error_format():
    """Prueba el formato con errores."""
    
    chat = RealChatInterface()
    
    # Respuesta con errores
    error_response = """
✅ Operación completada exitosamente
   - Paciente creado con ID: 12345
   - Nombre: Juan Pérez
   - Edad: 45 años
   - Diagnósticos guardados: 0
   - Medicamentos guardados: 0
   - Observaciones guardadas: 0
   - Total de registros clínicos añadidos: 0
   - ⚠️ Errores encontrados: 2
     - ❌ Error de conexión: Base de datos no disponible
     - ❌ Error de validación: Datos del paciente incompletos
"""
    
    print("\n🔬 Probando formato con errores...")
    print("=" * 60)
    
    formatted_response = chat.format_clinical_operation_response(error_response)
    print(formatted_response)

def test_normal_response():
    """Prueba el formato normal para respuestas que no son operaciones clínicas."""
    
    chat = RealChatInterface()
    
    # Respuesta normal (no operación clínica)
    normal_response = """
🏥 Consulta SQL procesada exitosamente

📊 Resultados encontrados: 15 pacientes diabéticos
🗄️ Tabla consultada: PATI_PATIENTS
⏱️ Tiempo de respuesta: 0.45 segundos

Los pacientes encontrados tienen diabetes tipo 2 y están siendo tratados
con medicamentos hipoglucemiantes según el protocolo establecido.
"""
    
    print("\n🔬 Probando formato normal...")
    print("=" * 60)
    
    formatted_response = chat.format_response(normal_response)
    print(formatted_response)

if __name__ == "__main__":
    print("🚀 Iniciando pruebas del nuevo formato de terminal...")
    
    # Ejecutar pruebas
    test_new_format()
    test_error_format()
    test_normal_response()
    
    print("\n🎉 Todas las pruebas completadas.")
    print("💡 El nuevo formato proporciona una experiencia de usuario mucho más profesional y clara.") 