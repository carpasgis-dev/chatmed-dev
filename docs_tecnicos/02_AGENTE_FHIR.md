# 🏥 Agente FHIR - Documentación Técnica

## 📋 Información General

**Nombre del Agente:** `FHIRMedicalAgent`  
**Clase Principal:** `agents.fhir_agent_complete.FHIRMedicalAgent`  
**Versión:** v4.0  
**Tipo:** Agente de procesamiento y gestión de recursos FHIR médicos  

## 🎯 Propósito y Funcionalidad

El Agente FHIR es el componente especializado en el procesamiento de información clínica estructurada según el estándar FHIR (Fast Healthcare Interoperability Resources). Se encarga de convertir notas clínicas en recursos FHIR y mapearlos a la base de datos SQL.

### Funciones Principales:
- **Procesamiento de notas clínicas** con IA
- **Conversión automática** SQL↔FHIR
- **Validación FHIR** automática
- **Gestión inteligente** de recursos relacionados
- **Mapeo dinámico** de campos sin hardcodeo
- **Persistencia** de recursos FHIR

## 🏗️ Arquitectura Técnica

### Componentes Principales:

#### 1. **Sistema de Procesamiento de Notas Clínicas**
```python
# Procesamiento de notas clínicas con LLM
clinical_note_prompt = f"""Procesa la siguiente nota clínica y extrae información estructurada:

NOTA CLÍNICA:
{clinical_note}

INSTRUCCIONES:
1. Identifica información del paciente (nombre, edad, sexo)
2. Extrae diagnósticos y condiciones médicas
3. Identifica medicaciones prescritas
4. Detecta resultados de laboratorio
5. Identifica procedimientos realizados
6. Extrae observaciones clínicas relevantes

FORMATO DE SALIDA: JSON estructurado con recursos FHIR"""
```

#### 2. **Sistema de Conversión FHIR↔SQL**
```python
# Mapeo dinámico FHIR a SQL
mapping_prompt = f"""Convierte el siguiente recurso FHIR a SQL:

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

#### 3. **Sistema de Validación FHIR**
- **Validación de sintaxis** FHIR
- **Verificación de recursos** requeridos
- **Validación de referencias** entre recursos
- **Comprobación de integridad** de datos

## 📊 Preguntas Clínicamente Relevantes

### 1. **Procesamiento de Notas Clínicas**
```
❓ "Procesa esta nota clínica: Paciente María López, 45 años, presenta diabetes tipo 2, 
    presión arterial 140/90, se prescribe metformina 500mg 2x día"
❓ "Convierte a FHIR: Paciente varón, 62 años, hipertensión arterial, 
    tratamiento con enalapril 10mg diario"
❓ "Extrae información FHIR de: Paciente femenina, 38 años, embarazada 12 semanas, 
    sin complicaciones"
```

**Recursos FHIR Generados:**
```json
{
  "resourceType": "Patient",
  "id": "1751178304",
  "name": [{"text": "María López"}],
  "birthDate": "1980-01-01",
  "gender": "female"
},
{
  "resourceType": "Condition",
  "id": "diag_1751178304_001",
  "subject": {"reference": "Patient/1751178304"},
  "code": {"text": "Diabetes tipo 2"},
  "clinicalStatus": "active"
},
{
  "resourceType": "MedicationRequest",
  "id": "med_1751178304_001",
  "subject": {"reference": "Patient/1751178304"},
  "medicationCodeableConcept": {"text": "Metformina 500mg"},
  "dosageInstruction": [{"text": "2x día"}]
}
```

### 2. **Búsquedas de Recursos FHIR**
```
❓ "Busca todos los pacientes con diabetes en formato FHIR"
❓ "Encuentra medicaciones prescritas para el paciente 1751178304"
❓ "Obtén diagnósticos activos en formato FHIR"
❓ "Lista de procedimientos realizados esta semana"
```

**Query FHIR Generado:**
```json
{
  "resourceType": "Bundle",
  "type": "searchset",
  "entry": [
    {
      "resource": {
        "resourceType": "Patient",
        "id": "1751178304",
        "name": [{"text": "María del Carmen López de la Cruz"}]
      }
    }
  ]
}
```

### 3. **Conversión de Formatos**
```
❓ "Convierte estos datos SQL a FHIR: PATI_ID=1751178304, PATI_NAME='María López'"
❓ "Transforma a SQL: Patient resource con id 1751178304"
❓ "Mapea diagnóstico FHIR a tabla SQL"
```

**Conversión Automática:**
```sql
-- SQL generado desde FHIR
INSERT INTO PATI_PATIENTS (PATI_ID, PATI_NAME, PATI_FULL_NAME) 
VALUES (1751178304, 'María', 'María del Carmen López de la Cruz');
```

### 4. **Validación de Recursos**
```
❓ "Valida este recurso FHIR: Patient con id inválido"
❓ "Verifica integridad de referencias en Bundle FHIR"
❓ "Comprueba que el diagnóstico tenga paciente asociado"
```

**Validación Automática:**
```python
# Validación de recursos FHIR
def validate_fhir_resource(resource: Dict) -> ValidationResult:
    """
    Valida un recurso FHIR según especificaciones
    """
    # 1. Validación de sintaxis
    # 2. Verificación de campos requeridos
    # 3. Validación de referencias
    # 4. Comprobación de integridad
```

## 🔧 Funciones Técnicas Principales

### 1. **`process_query(query: str)`**
**Propósito:** Procesamiento principal de consultas FHIR  
**Entrada:** Consulta en lenguaje natural o nota clínica  
**Salida:** Recursos FHIR estructurados  

```python
async def process_query(self, query: str, stream_callback=None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    🏥 Procesamiento de consultas FHIR y notas clínicas
    """
    # 1. Detección de tipo de consulta
    # 2. Procesamiento de nota clínica (si aplica)
    # 3. Generación de recursos FHIR
    # 4. Mapeo a SQL (si necesario)
    # 5. Validación y persistencia
```

### 2. **`process_clinical_note(note: str)`**
**Propósito:** Procesamiento de notas clínicas a recursos FHIR  
**Características:** Extracción inteligente de información médica  

```python
async def process_clinical_note(self, note: str) -> Dict[str, Any]:
    """
    Procesa nota clínica y genera recursos FHIR
    """
    # 1. Análisis semántico de la nota
    # 2. Extracción de entidades médicas
    # 3. Generación de recursos FHIR
    # 4. Validación de recursos
    # 5. Persistencia en base de datos
```

### 3. **`convert_fhir_to_sql(fhir_resource: Dict)`**
**Propósito:** Conversión de recursos FHIR a SQL  
**Características:** Mapeo dinámico sin hardcodeo  

```python
async def convert_fhir_to_sql(self, fhir_resource: Dict) -> str:
    """
    Convierte recurso FHIR a SQL usando mapeo dinámico
    """
    # 1. Identificación del tipo de recurso
    # 2. Mapeo dinámico de campos
    # 3. Generación de SQL optimizado
    # 4. Validación de esquema
    # 5. Manejo de referencias
```

## 🗃️ Estructura de Recursos FHIR

### Recursos Principales:

#### **Patient** (Paciente)
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

#### **Condition** (Diagnóstico)
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

#### **MedicationRequest** (Medicación)
```json
{
  "resourceType": "MedicationRequest",
  "id": "med_1751178304_001",
  "subject": {"reference": "Patient/1751178304"},
  "medicationCodeableConcept": {"text": "Metformina 500mg"},
  "dosageInstruction": [{"text": "2 comprimidos al día"}],
  "status": "active"
}
```

#### **Observation** (Observación)
```json
{
  "resourceType": "Observation",
  "id": "obs_1751178304_001",
  "subject": {"reference": "Patient/1751178304"},
  "code": {"text": "Presión arterial"},
  "valueQuantity": {
    "value": 140,
    "unit": "mmHg"
  }
}
```

## 🔍 Algoritmos de Procesamiento

### 1. **Extracción de Entidades Médicas**
```python
# Algoritmo de extracción usando LLM
def extract_medical_entities(note: str) -> Dict[str, Any]:
    """
    Extrae entidades médicas de nota clínica
    """
    # 1. Identificación de paciente
    # 2. Extracción de diagnósticos
    # 3. Detección de medicaciones
    # 4. Identificación de observaciones
    # 5. Extracción de procedimientos
```

### 2. **Generación de Recursos FHIR**
```python
# Algoritmo de generación FHIR
def generate_fhir_resources(entities: Dict) -> List[Dict]:
    """
    Genera recursos FHIR desde entidades extraídas
    """
    # 1. Creación de Patient resource
    # 2. Generación de Condition resources
    # 3. Creación de MedicationRequest resources
    # 4. Generación de Observation resources
    # 5. Manejo de referencias entre recursos
```

### 3. **Mapeo Dinámico FHIR↔SQL**
```python
# Algoritmo de mapeo dinámico
def map_fhir_to_sql(fhir_resource: Dict, schema: Dict) -> str:
    """
    Mapea recurso FHIR a SQL usando esquema dinámico
    """
    # 1. Identificación del tipo de recurso
    # 2. Mapeo de campos usando LLM
    # 3. Generación de SQL optimizado
    # 4. Validación de compatibilidad
    # 5. Manejo de referencias
```

## 📈 Métricas de Rendimiento

### Indicadores Clave:
- **Tiempo de procesamiento:** < 10 segundos para notas complejas
- **Precisión de extracción:** > 90% de entidades correctas
- **Tasa de conversión:** > 95% de recursos FHIR válidos
- **Tasa de mapeo:** > 85% de conversiones FHIR↔SQL exitosas

### Logs de Rendimiento:
```python
logger.info(f"🏥 Nota clínica procesada: {len(fhir_resources)} recursos FHIR generados")
logger.info(f"✅ Recursos FHIR validados y persistidos")
logger.info(f"🔄 Conversión FHIR↔SQL completada: {sql_queries} queries generadas")
```

## 🛠️ Configuración y Uso

### Inicialización:
```python
fhir_agent = FHIRMedicalAgent(
    db_path="database_new.sqlite3.db",
    llm=llm_instance,
    sql_agent=sql_agent,
    medgemma_agent=medgemma_agent
)
```

### Ejemplo de Uso:
```python
# Procesamiento de nota clínica
result = await fhir_agent.process_query(
    "Paciente María López, 45 años, diabetes tipo 2, metformina 500mg 2x día"
)

# Conversión FHIR a SQL
sql_result = await fhir_agent.convert_fhir_to_sql(fhir_resource)

# Búsqueda de recursos FHIR
search_result = await fhir_agent.process_query(
    "Busca todos los pacientes con diabetes en formato FHIR"
)
```

## 🔧 Troubleshooting

### Problemas Comunes:

#### 1. **Recursos FHIR Inválidos**
**Síntoma:** `Invalid FHIR resource: missing required field`  
**Solución:** Verificar campos requeridos y validación de recursos

#### 2. **Error de Referencias**
**Síntoma:** `Reference not found: Patient/999999`  
**Solución:** Verificar que los recursos referenciados existan

#### 3. **Mapeo Fallido**
**Síntoma:** `Cannot map FHIR field to SQL column`  
**Solución:** Verificar esquema de base de datos y mapeo dinámico

## 📚 Referencias Técnicas

### Archivos Principales:
- `agents/fhir_agent_complete.py` - Implementación principal
- `agents/fhir_persistence_agent_old.py` - Persistencia FHIR
- `mapping/fhir_sql_bridge.py` - Puente FHIR↔SQL
- `utils/fhir_mapping_corrector.py` - Corrección de mapeos

### Dependencias:
- `fhirclient` - Cliente FHIR
- `langchain_openai` - LLM para procesamiento
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