# 💾 Agente de Persistencia FHIR - Documentación Técnica

## 📋 Información General

**Nombre del Agente:** `FHIRPersistenceAgent`  
**Clase Principal:** `agents.fhir_persistence_agent_old.FHIRPersistenceAgent`  
**Versión:** v3.0  
**Tipo:** Agente de persistencia y gestión de recursos FHIR en base de datos  

## 🎯 Propósito y Funcionalidad

El Agente de Persistencia FHIR es el componente especializado en el almacenamiento, gestión y recuperación de recursos FHIR en la base de datos. Se encarga de convertir recursos FHIR a SQL y mantener la integridad referencial entre recursos.

### Funciones Principales:
- **Persistencia de recursos** FHIR en SQL
- **Gestión de referencias** entre recursos
- **Validación de integridad** de datos
- **Conversión automática** FHIR↔SQL
- **Manejo de transacciones** seguras
- **Recuperación de recursos** FHIR

## 🏗️ Arquitectura Técnica

### Componentes Principales:

#### 1. **Sistema de Conversión FHIR↔SQL**
```python
# Conversión de recursos FHIR a SQL
fhir_to_sql_prompt = f"""Convierte el siguiente recurso FHIR a SQL:

RECURSO FHIR:
{fhir_resource}

ESQUEMA DE BASE DE DATOS:
{schema_info}

INSTRUCCIONES:
1. Identifica el tipo de recurso FHIR
2. Mapea campos FHIR a columnas SQL
3. Genera INSERT/UPDATE apropiado
4. Maneja referencias entre recursos
5. Valida integridad de datos

SQL GENERADO:"""
```

#### 2. **Sistema de Gestión de Referencias**
```python
# Gestión de referencias entre recursos FHIR
reference_management = {
    "patient_references": "Patient/{id}",
    "condition_references": "Condition/{id}",
    "medication_references": "MedicationRequest/{id}",
    "observation_references": "Observation/{id}",
    "procedure_references": "Procedure/{id}"
}
```

#### 3. **Sistema de Validación de Integridad**
- **Validación de recursos** FHIR
- **Verificación de referencias** existentes
- **Comprobación de constraints** de base de datos
- **Validación de tipos** de datos

## 📊 Preguntas Clínicamente Relevantes

### 1. **Persistencia de Recursos FHIR**
```
❓ "Guarda este recurso Patient en la base de datos"
❓ "Persiste este diagnóstico FHIR"
❓ "Almacena esta medicación en formato FHIR"
❓ "Convierte y guarda estos recursos FHIR"
```

**Recurso FHIR de Entrada:**
```json
{
  "resourceType": "Patient",
  "id": "1751178304",
  "identifier": [{"value": "123456"}],
  "name": [{"text": "María del Carmen López de la Cruz"}],
  "birthDate": "1980-01-01",
  "gender": "female",
  "active": true
}
```

**SQL Generado:**
```sql
INSERT INTO PATI_PATIENTS (
    PATI_ID, 
    PATI_NAME, 
    PATI_SURNAME_1, 
    PATI_FULL_NAME, 
    PATI_BIRTH_DATE, 
    PATI_GENDER, 
    PATI_ACTIVE
) VALUES (
    1751178304, 
    'María', 
    'López', 
    'María del Carmen López de la Cruz', 
    '1980-01-01', 
    'female', 
    1
);
```

### 2. **Gestión de Referencias FHIR**
```
❓ "Crea un diagnóstico que referencie al paciente 1751178304"
❓ "Guarda una medicación que apunte al paciente correcto"
❓ "Persiste una observación con referencia al paciente"
❓ "Maneja las referencias entre recursos FHIR"
```

**Recurso con Referencias:**
```json
{
  "resourceType": "Condition",
  "id": "diag_1751178304_001",
  "subject": {"reference": "Patient/1751178304"},
  "code": {"text": "Diabetes mellitus tipo 2"},
  "clinicalStatus": "active",
  "onsetDateTime": "2023-10-05"
}
```

**SQL con Referencias:**
```sql
INSERT INTO EPIS_DIAGNOSTICS (
    DIAG_ID,
    PATI_ID,
    DIAG_OBSERVATION,
    DIAG_DESCRIPTION,
    DIAG_CLINICAL_STATUS,
    DIAG_ONSET_DATE
) VALUES (
    'diag_1751178304_001',
    1751178304,
    'Diabetes mellitus tipo 2',
    'Diabetes mellitus tipo 2',
    'active',
    '2023-10-05'
);
```

### 3. **Recuperación de Recursos FHIR**
```
❓ "Recupera el paciente con ID 1751178304 en formato FHIR"
❓ "Obtén todos los diagnósticos de un paciente"
❓ "Busca medicaciones prescritas en formato FHIR"
❓ "Convierte datos SQL a recursos FHIR"
```

**Recurso FHIR Recuperado:**
```json
{
  "resourceType": "Patient",
  "id": "1751178304",
  "identifier": [{"value": "123456"}],
  "name": [{"text": "María del Carmen López de la Cruz"}],
  "birthDate": "1980-01-01",
  "gender": "female",
  "active": true,
  "extension": [
    {
      "url": "http://example.com/patient/start_date",
      "valueDateTime": "2023-01-15"
    }
  ]
}
```

### 4. **Validación de Integridad**
```
❓ "Valida que este recurso FHIR sea correcto"
❓ "Verifica que las referencias existan"
❓ "Comprueba la integridad de los datos FHIR"
❓ "Valida la estructura del recurso"
```

**Validación Automática:**
```python
# Validación de recurso FHIR
validation_result = {
    "is_valid": True,
    "errors": [],
    "warnings": [],
    "references_valid": True,
    "data_types_valid": True,
    "constraints_satisfied": True
}
```

## 🔧 Funciones Técnicas Principales

### 1. **`persist_fhir_resource(fhir_resource: Dict)`**
**Propósito:** Persistencia de recursos FHIR en base de datos  
**Entrada:** Recurso FHIR válido  
**Salida:** Confirmación de persistencia exitosa  

```python
async def persist_fhir_resource(self, fhir_resource: Dict) -> Dict[str, Any]:
    """
    💾 Persiste un recurso FHIR en la base de datos
    """
    # 1. Validación del recurso FHIR
    # 2. Conversión a SQL
    # 3. Validación de referencias
    # 4. Ejecución de transacción
    # 5. Confirmación de persistencia
```

### 2. **`convert_fhir_to_sql(fhir_resource: Dict)`**
**Propósito:** Conversión de recursos FHIR a SQL  
**Características:** Mapeo dinámico sin hardcodeo  

```python
async def convert_fhir_to_sql(self, fhir_resource: Dict) -> str:
    """
    Convierte recurso FHIR a SQL usando mapeo dinámico
    """
    # 1. Identificación del tipo de recurso
    # 2. Mapeo de campos usando LLM
    # 3. Generación de SQL optimizado
    # 4. Validación de esquema
    # 5. Manejo de referencias
```

### 3. **`retrieve_fhir_resource(resource_type: str, resource_id: str)`**
**Propósito:** Recuperación de recursos FHIR desde SQL  
**Características:** Conversión inversa SQL↔FHIR  

```python
async def retrieve_fhir_resource(self, resource_type: str, resource_id: str) -> Dict[str, Any]:
    """
    Recupera recurso FHIR desde la base de datos
    """
    # 1. Consulta SQL del recurso
    # 2. Conversión a formato FHIR
    # 3. Inclusión de referencias
    # 4. Validación de estructura
    # 5. Retorno del recurso FHIR
```

## 🗃️ Estructura de Mapeo FHIR↔SQL

### Mapeo de Recursos:

#### **Patient ↔ PATI_PATIENTS**
```python
patient_mapping = {
    "id": "PATI_ID",
    "name[0].text": "PATI_FULL_NAME",
    "birthDate": "PATI_BIRTH_DATE",
    "gender": "PATI_GENDER",
    "active": "PATI_ACTIVE",
    "identifier[0].value": "PATI_IDENTIFIER"
}
```

#### **Condition ↔ EPIS_DIAGNOSTICS**
```python
condition_mapping = {
    "id": "DIAG_ID",
    "subject.reference": "PATI_ID",
    "code.text": "DIAG_OBSERVATION",
    "clinicalStatus": "DIAG_CLINICAL_STATUS",
    "onsetDateTime": "DIAG_ONSET_DATE"
}
```

#### **MedicationRequest ↔ PATI_USUAL_MEDICATION**
```python
medication_mapping = {
    "id": "PAUM_ID",
    "subject.reference": "PATI_ID",
    "medicationCodeableConcept.text": "PAUM_OBSERVATIONS",
    "status": "PAUM_STATUS",
    "dosageInstruction[0].text": "PAUM_DOSAGE"
}
```

#### **Observation ↔ PROC_PROCEDURES**
```python
observation_mapping = {
    "id": "PROC_ID",
    "subject.reference": "PATI_ID",
    "code.text": "PROC_DESCRIPTION",
    "valueQuantity.value": "PROC_VALUE",
    "valueQuantity.unit": "PROC_UNIT"
}
```

## 🔍 Algoritmos de Persistencia

### 1. **Validación de Recursos FHIR**
```python
# Algoritmo de validación
def validate_fhir_resource(resource: Dict) -> ValidationResult:
    """
    Valida un recurso FHIR según especificaciones
    """
    # 1. Validación de sintaxis
    # 2. Verificación de campos requeridos
    # 3. Validación de referencias
    # 4. Comprobación de integridad
    # 5. Validación de tipos de datos
```

### 2. **Conversión Dinámica FHIR↔SQL**
```python
# Algoritmo de conversión
def convert_fhir_to_sql_dynamic(fhir_resource: Dict, schema: Dict) -> str:
    """
    Convierte FHIR a SQL usando mapeo dinámico
    """
    # 1. Identificación del tipo de recurso
    # 2. Análisis del esquema de base de datos
    # 3. Mapeo de campos usando LLM
    # 4. Generación de SQL optimizado
    # 5. Validación de compatibilidad
```

### 3. **Gestión de Transacciones**
```python
# Algoritmo de transacciones
def manage_fhir_transaction(resources: List[Dict]) -> TransactionResult:
    """
    Gestiona transacción de múltiples recursos FHIR
    """
    # 1. Validación de todos los recursos
    # 2. Ordenamiento por dependencias
    # 3. Ejecución en transacción
    # 4. Rollback en caso de error
    # 5. Confirmación de persistencia
```

## 📈 Métricas de Rendimiento

### Indicadores Clave:
- **Tiempo de persistencia:** < 5 segundos para recursos complejos
- **Tasa de conversión:** > 95% de recursos FHIR válidos
- **Precisión de mapeo:** > 90% de campos mapeados correctamente
- **Tasa de transacciones:** > 98% de transacciones exitosas

### Logs de Rendimiento:
```python
logger.info(f"💾 Recurso persistido: {resource_type} con ID {resource_id}")
logger.info(f"✅ Conversión FHIR↔SQL completada en {conversion_time:.2f}s")
logger.info(f"🔄 Transacción exitosa: {resources_count} recursos")
```

## 🛠️ Configuración y Uso

### Inicialización:
```python
fhir_persistence_agent = FHIRPersistenceAgent(
    db_path="database_new.sqlite3.db",
    llm=llm_instance,
    sql_agent=sql_agent
)
```

### Ejemplo de Uso:
```python
# Persistencia de recurso FHIR
result = await fhir_persistence_agent.persist_fhir_resource(fhir_patient)

# Conversión FHIR a SQL
sql_query = await fhir_persistence_agent.convert_fhir_to_sql(fhir_resource)

# Recuperación de recurso FHIR
patient_fhir = await fhir_persistence_agent.retrieve_fhir_resource("Patient", "1751178304")
```

## 🔧 Troubleshooting

### Problemas Comunes:

#### 1. **Recurso FHIR Inválido**
**Síntoma:** `Invalid FHIR resource: missing required field`  
**Solución:** Verificar campos requeridos y validación de recursos

#### 2. **Error de Referencia**
**Síntoma:** `Reference not found: Patient/999999`  
**Solución:** Verificar que los recursos referenciados existan

#### 3. **Error de Conversión**
**Síntoma:** `Cannot map FHIR field to SQL column`  
**Solución:** Verificar esquema de base de datos y mapeo dinámico

## 📚 Referencias Técnicas

### Archivos Principales:
- `agents/fhir_persistence_agent_old.py` - Implementación principal
- `mapping/fhir_sql_bridge.py` - Puente FHIR↔SQL
- `utils/fhir_mapping_corrector.py` - Corrección de mapeos
- `config/` - Configuración de mapeos

### Dependencias:
- `fhirclient` - Cliente FHIR
- `langchain_openai` - LLM para mapeo
- `sqlite3` - Base de datos
- `json` - Manejo de recursos FHIR

### Estándares FHIR:
- **FHIR R4** - Versión de recursos utilizada
- **HL7 FHIR** - Estándar de interoperabilidad
- **FHIR Validator** - Validación de recursos

---

**Versión:** 1.0  
**Última actualización:** 2025-07-18  
**Mantenido por:** Equipo de Desarrollo ChatMed 