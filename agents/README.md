# 🧠 ChatMed v2.0 Flexible - Masterclass de Agentes

## 📋 Índice
1. [Introducción](#introducción)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Agentes Principales](#agentes-principales)
4. [Agentes Especializados](#agentes-especializados)
5. [Agentes de Soporte](#agentes-de-soporte)
6. [Flujo de Trabajo](#flujo-de-trabajo)
7. [Casos de Uso](#casos-de-uso)

---

## 🎯 Introducción

ChatMed v2.0 Flexible es un sistema de **Inteligencia Artificial médica multi-agente** diseñado para asistir a profesionales de la salud. La filosofía del sistema es **100% dinámico sin hardcodeo** - todo se decide usando IA en tiempo real.

### 🧠 Filosofía del Sistema
- **Dinámico**: Sin patrones predefinidos, todo se resuelve con LLM
- **Adaptativo**: Los agentes aprenden y se adaptan a cada consulta
- **Transparente**: El usuario siempre sabe qué está pasando
- **Modular**: Cada agente tiene una responsabilidad específica

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GreetingAgent │    │  BioChatAgent   │    │  FHIRMedicalAgent│
│   (Conversación)│    │  (Investigación)│    │  (Datos Clínicos)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ SQLAgentEnhanced│
                    │ (Base de Datos) │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │MedGemma Clinical│
                    │   (Análisis)    │
                    └─────────────────┘
```

---

## 🔄 **Conversión SQL ↔ FHIR - Explicación Detallada**

### 🎯 **¿Por qué es importante esta conversión?**

En el mundo médico, existen **dos estándares principales** para manejar datos:

1. **🗄️ SQL (Base de Datos)**: Formato tradicional para almacenar datos
2. **🏥 FHIR (Fast Healthcare Interoperability Resources)**: Estándar moderno para intercambio de datos médicos

### 🔄 **El Puente Inteligente SQL ↔ FHIR**

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONVERSIÓN DINÁMICA                        │
│                        SQL ↔ FHIR                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌─────────────────┐
                    │   LLM Analiza   │
                    │   y Decide      │
                    └─────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────────────┐    ┌─────────────────┐    ┌───────────────┐
│   SQL → FHIR  │    │  Validación     │    │  FHIR → SQL   │
│   (Consulta)  │    │  Inteligente    │    │  (Inserción)  │
└───────────────┘    └─────────────────┘    └───────────────┘
```

### 📊 **Ejemplo Práctico: Paciente con Medicación**

#### **🗄️ Datos en SQL (Base de Datos):**
```sql
-- Tabla PATI_PATIENTS
PATI_ID: 12345
PATI_FIRST_NAME: "María"
PATI_LAST_NAME: "García"
PATI_BIRTH_DATE: "1978-05-15"
PATI_GENDER: "F"

-- Tabla PATI_USUAL_MEDICATION  
PAUM_ID: 67890
PATI_ID: 12345  -- Referencia al paciente
PAUM_OBSERVATIONS: "Paracetamol 500mg cada 8 horas"
PAUM_START_DATE: "2024-01-15"
```

#### **🏥 Conversión a FHIR (Estándar Médico):**
```json
{
  "resourceType": "Patient",
  "id": "12345",
  "name": [
    {
      "use": "official",
      "text": "María García",
      "family": "García",
      "given": ["María"]
    }
  ],
  "gender": "female",
  "birthDate": "1978-05-15"
}

{
  "resourceType": "MedicationRequest",
  "id": "67890",
  "subject": {
    "reference": "Patient/12345"
  },
  "medicationCodeableConcept": {
    "text": "Paracetamol 500mg"
  },
  "dosageInstruction": [
    {
      "text": "500mg cada 8 horas"
    }
  ],
  "authoredOn": "2024-01-15"
}
```

### 🧠 **Proceso de Conversión Inteligente:**

#### **1️⃣ SQL → FHIR (Consulta de Datos):**
```python
# Usuario pregunta: "Muéstrame los datos FHIR del paciente María García"

# 1. SQLAgentEnhanced genera SQL:
SELECT p.*, m.* 
FROM PATI_PATIENTS p 
LEFT JOIN PATI_USUAL_MEDICATION m ON p.PATI_ID = m.PATI_ID 
WHERE p.PATI_FIRST_NAME = 'María' AND p.PATI_LAST_NAME = 'García'

# 2. FHIRMedicalAgent convierte resultados a FHIR:
- Mapea PATI_FIRST_NAME + PATI_LAST_NAME → Patient.name
- Mapea PATI_BIRTH_DATE → Patient.birthDate
- Mapea PAUM_OBSERVATIONS → MedicationRequest.medicationCodeableConcept
- Crea referencias entre recursos (Patient ↔ MedicationRequest)
```

#### **2️⃣ FHIR → SQL (Inserción de Datos):**
```python
# Usuario envía: "Procesa esta nota: Paciente Carlos López, 35 años, toma enalapril 10mg"

# 1. FHIRMedicalAgent extrae datos:
{
  "resourceType": "Patient",
  "name": "Carlos López",
  "age": 35
}

{
  "resourceType": "MedicationRequest", 
  "medicationCodeableConcept": {
    "text": "Enalapril 10mg"
  }
}

# 2. LLM mapea a SQL:
INSERT INTO PATI_PATIENTS (PATI_FIRST_NAME, PATI_LAST_NAME, PATI_BIRTH_DATE)
VALUES ('Carlos', 'López', '1989-01-01')

INSERT INTO PATI_USUAL_MEDICATION (PATI_ID, PAUM_OBSERVATIONS)
VALUES (LAST_INSERT_ID(), 'Enalapril 10mg')
```

### 🎯 **Ventajas de la Conversión Dinámica:**

#### **✅ Para Profesionales de la Salud:**
- **Interoperabilidad**: Los datos pueden compartirse con otros sistemas médicos
- **Estándares**: Cumple con normativas internacionales (HL7 FHIR)
- **Flexibilidad**: Puede trabajar con datos SQL existentes y nuevos datos FHIR

#### **✅ Para Desarrolladores:**
- **Sin Hardcodeo**: El LLM decide cómo mapear campos automáticamente
- **Adaptativo**: Se adapta a diferentes esquemas de base de datos
- **Validación**: Verifica que los datos cumplan estándares FHIR

#### **✅ Para el Sistema:**
- **Bidireccional**: SQL ↔ FHIR y FHIR ↔ SQL
- **Inteligente**: Usa LLM para entender contexto médico
- **Robusto**: Maneja errores y casos edge automáticamente

### 🔧 **Componentes Técnicos de la Conversión:**

#### **1. 🧠 LLM como Traductor:**
```python
# El LLM entiende:
- Qué campos SQL corresponden a qué campos FHIR
- Cómo validar datos médicos
- Cómo manejar relaciones entre recursos
- Cómo corregir errores automáticamente
```

#### **2. 🔍 Validación Inteligente:**
```python
# Verifica:
- Que los datos cumplan estándares FHIR
- Que las fechas sean válidas
- Que los códigos médicos sean correctos
- Que las referencias entre recursos sean coherentes
```

#### **3. 🔄 Mapeo Dinámico:**
```python
# Sin reglas hardcodeadas:
- El LLM analiza el contexto de cada campo
- Decide el mapeo más apropiado
- Aprende de mapeos exitosos previos
- Se adapta a nuevos esquemas automáticamente
```

### 💡 **Casos de Uso Reales:**

#### **🏥 Hospital que migra de SQL a FHIR:**
```
1. Tiene base de datos SQL con 10,000 pacientes
2. Quiere interoperar con otros sistemas médicos
3. FHIRMedicalAgent convierte automáticamente
4. Mantiene compatibilidad con sistema existente
```

#### **🔬 Investigador que analiza datos:**
```
1. Recibe datos en formato FHIR de múltiples fuentes
2. Quiere analizarlos en su base de datos SQL
3. FHIRMedicalAgent convierte y normaliza
4. SQLAgentEnhanced permite consultas complejas
```

#### **📊 Sistema de Salud Pública:**
```
1. Necesita estandarizar datos de múltiples hospitales
2. Cada hospital tiene esquema SQL diferente
3. FHIRMedicalAgent unifica en formato FHIR
4. Permite análisis epidemiológico centralizado
```

---

## 🗄️ **Esquema de Base de Datos - Explicación Detallada**

### 🎯 **¿Cómo está organizada la información médica?**

La base de datos de ChatMed sigue un **esquema médico especializado** que simula un sistema de información clínica real:

### 📊 **Estructura de Tablas Principales:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    ESQUEMA MÉDICO                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ PATI_PATIENTS  │    │PATI_USUAL_MEDIC.│    │  MEDI_MEDICAT.  │
│ (Pacientes)    │◄──►│ (Medicación)    │◄──►│ (Medicamentos)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  DIAG_DIAGNOSIS│    │  ACIN_ACTIVE_   │    │  CODR_CODES    │
│  (Diagnósticos) │    │  INGREDIENTS    │    │  (Códigos)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 🏥 **Tablas Principales Explicadas:**

#### **1. 🧑‍⚕️ PATI_PATIENTS (Pacientes):**
```sql
PATI_ID (AUTOINCREMENT)     -- ID único del paciente
PATI_FIRST_NAME             -- Nombre
PATI_LAST_NAME              -- Apellido  
PATI_BIRTH_DATE             -- Fecha de nacimiento
PATI_GENDER                 -- Género (M/F)
PATI_ADDRESS                -- Dirección
PATI_PHONE                  -- Teléfono
PATI_EMAIL                  -- Email
```

#### **2. 💊 PATI_USUAL_MEDICATION (Medicación del Paciente):**
```sql
PAUM_ID (AUTOINCREMENT)     -- ID único de la medicación
PATI_ID                     -- Referencia al paciente
PAUM_OBSERVATIONS           -- Texto libre: "Paracetamol 500mg cada 8h"
PAUM_START_DATE             -- Fecha de inicio
PAUM_END_DATE               -- Fecha de fin (opcional)
PAUM_DOSAGE                 -- Dosis
PAUM_FREQUENCY              -- Frecuencia
```

#### **3. 🏥 DIAG_DIAGNOSIS (Diagnósticos):**
```sql
DIAG_ID (AUTOINCREMENT)     -- ID único del diagnóstico
PATI_ID                     -- Referencia al paciente
DIAG_DESCRIPTION            -- Descripción del diagnóstico
DIAG_DATE                   -- Fecha del diagnóstico
DIAG_SEVERITY               -- Severidad
DIAG_STATUS                 -- Estado (activo, resuelto, etc.)
```

#### **4. 💊 MEDI_MEDICATIONS (Catálogo de Medicamentos):**
```sql
MEDI_ID (AUTOINCREMENT)     -- ID único del medicamento
MEDI_NAME                   -- Nombre del medicamento
MEDI_ACTIVE_INGREDIENT      -- Principio activo
MEDI_DOSAGE_FORM            -- Forma farmacéutica
MEDI_STRENGTH               -- Concentración
```

#### **5. 🧪 ACIN_ACTIVE_INGREDIENTS (Principios Activos):**
```sql
ACIN_ID (AUTOINCREMENT)     -- ID único del principio activo
ACIN_NAME                   -- Nombre del principio activo
ACIN_DESCRIPTION            -- Descripción
ACIN_THERAPEUTIC_CLASS      -- Clase terapéutica
```

#### **6. 📋 CODR_CODES (Códigos Médicos):**
```sql
CODR_ID (AUTOINCREMENT)     -- ID único del código
CODR_CODE                   -- Código (ICD-10, ATC, etc.)
CODR_DESCRIPTION            -- Descripción
CODR_TYPE                   -- Tipo de código
CODR_SYSTEM                 -- Sistema de codificación
```

### 🔗 **Relaciones entre Tablas:**

#### **👥 Paciente ↔ Medicación:**
```sql
-- Un paciente puede tener múltiples medicaciones
PATI_PATIENTS.PATI_ID ←→ PATI_USUAL_MEDICATION.PATI_ID
```

#### **👥 Paciente ↔ Diagnósticos:**
```sql
-- Un paciente puede tener múltiples diagnósticos
PATI_PATIENTS.PATI_ID ←→ DIAG_DIAGNOSIS.PATI_ID
```

#### **💊 Medicación ↔ Medicamentos:**
```sql
-- La medicación del paciente se relaciona con el catálogo
PATI_USUAL_MEDICATION.PAUM_OBSERVATIONS ←→ MEDI_MEDICATIONS.MEDI_NAME
```

### 🧠 **Cómo el LLM Entiende el Esquema:**

#### **1️⃣ Análisis Automático:**
```python
# El LLM explora automáticamente:
- Qué tablas existen
- Qué columnas tiene cada tabla
- Cómo se relacionan las tablas
- Qué tipo de datos contiene cada campo
```

#### **2️⃣ Mapeo Inteligente:**
```python
# Cuando el usuario dice: "Paciente con diabetes que toma metformina"

# El LLM entiende:
- "Paciente" → tabla PATI_PATIENTS
- "diabetes" → tabla DIAG_DIAGNOSIS (diagnóstico)
- "metformina" → tabla PATI_USUAL_MEDICATION (medicación)
- Necesita JOIN entre las 3 tablas
```

#### **3️⃣ Generación de SQL:**
```sql
SELECT 
    p.PATI_FIRST_NAME, p.PATI_LAST_NAME,
    d.DIAG_DESCRIPTION,
    m.PAUM_OBSERVATIONS
FROM PATI_PATIENTS p
JOIN DIAG_DIAGNOSIS d ON p.PATI_ID = d.PATI_ID
JOIN PATI_USUAL_MEDICATION m ON p.PATI_ID = m.PATI_ID
WHERE d.DIAG_DESCRIPTION LIKE '%diabetes%'
AND m.PAUM_OBSERVATIONS LIKE '%metformina%'
```

### 💡 **Ejemplos Prácticos de Consultas:**

#### **🔍 "Muéstrame todos los pacientes con diabetes":**
```sql
SELECT p.*, d.DIAG_DESCRIPTION
FROM PATI_PATIENTS p
JOIN DIAG_DIAGNOSIS d ON p.PATI_ID = d.PATI_ID
WHERE d.DIAG_DESCRIPTION LIKE '%diabetes%'
```

#### **💊 "¿Qué medicamentos toma María García?":**
```sql
SELECT p.PATI_FIRST_NAME, p.PATI_LAST_NAME, m.PAUM_OBSERVATIONS
FROM PATI_PATIENTS p
JOIN PATI_USUAL_MEDICATION m ON p.PATI_ID = m.PATI_ID
WHERE p.PATI_FIRST_NAME = 'María' AND p.PATI_LAST_NAME = 'García'
```

#### **📊 "Pacientes con HTA que toman enalapril":**
```sql
SELECT p.*, d.DIAG_DESCRIPTION, m.PAUM_OBSERVATIONS
FROM PATI_PATIENTS p
JOIN DIAG_DIAGNOSIS d ON p.PATI_ID = d.PATI_ID
JOIN PATI_USUAL_MEDICATION m ON p.PATI_ID = m.PATI_ID
WHERE d.DIAG_DESCRIPTION LIKE '%HTA%'
AND m.PAUM_OBSERVATIONS LIKE '%enalapril%'
```

### 🎯 **Ventajas de este Esquema:**

#### **✅ Realista:**
- Simula un sistema de información clínica real
- Incluye relaciones médicas complejas
- Maneja datos médicos especializados

#### **✅ Flexible:**
- El LLM puede explorar y entender el esquema
- Se adapta a diferentes tipos de consultas
- Permite consultas complejas y anidadas

#### **✅ Extensible:**
- Fácil agregar nuevas tablas
- Mantiene relaciones coherentes
- Compatible con estándares médicos

### 🔧 **Proceso de Exploración del LLM:**

```python
# 1. El LLM explora el esquema:
"¿Qué tablas hay en la base de datos?"

# 2. Analiza las columnas:
"¿Qué campos tiene la tabla PATI_PATIENTS?"

# 3. Entiende las relaciones:
"¿Cómo se relacionan pacientes con medicaciones?"

# 4. Genera consultas inteligentes:
"Muéstrame pacientes con diabetes que toman metformina"
→ JOIN automático entre PATI_PATIENTS, DIAG_DIAGNOSIS, PATI_USUAL_MEDICATION
```

---

## 🎭 Agentes Principales

### 1. 🤖 **IntelligentGreetingAgent** (`greeting_agent.py`)
**El "Cerebro Conversacional" del Sistema**

#### 🎯 **¿Qué hace realmente?**
- **Analiza el contexto** de cada consulta usando LLM
- **Determina la intención** del usuario (saludo, ayuda, consulta técnica)
- **Adapta el tono** según el nivel de experiencia del usuario
- **Proporciona información** sobre las capacidades del sistema
- **Mantiene memoria** de la conversación para contexto

#### 🧠 **Cómo funciona internamente:**
```python
# Analiza cada consulta con LLM para entender:
- Tipo de interacción (saludo, ayuda, consulta técnica)
- Nivel de experiencia del usuario (principiante, intermedio, experto)
- Intención específica (qué quiere realmente el usuario)
- Tono apropiado para la respuesta
```

#### 💡 **Casos de uso típicos:**
- "Hola, ¿qué puedes hacer?"
- "¿Cómo funciona el sistema?"
- "Necesito ayuda con una consulta médica"
- "¿Qué bases de datos tienes disponibles?"

---

### 2. 🔬 **BioChatAgent** (`biochat_agent.py`)
**El "Investigador Biomédico" Multi-Fuente**

#### 🎯 **¿Qué hace realmente?**
- **Busca en múltiples fuentes** biomédicas simultáneamente
- **Planifica automáticamente** qué herramientas usar según la consulta
- **Sintetiza resultados** de diferentes fuentes en una respuesta coherente
- **Mantiene contexto** para preguntas de seguimiento
- **Optimiza queries** para cada base de datos específica

#### 🧠 **Fuentes de datos que consulta:**
- **PubMed**: Literatura médica y estudios científicos
- **ClinicalTrials.gov**: Ensayos clínicos activos
- **GenBank**: Datos genómicos y secuencias
- **Europe PMC**: Literatura biomédica europea
- **AEMPS**: Medicamentos autorizados en España
- **Semantic Scholar**: Literatura académica con análisis de impacto

#### 💡 **Ejemplo de flujo interno:**
```python
# 1. Usuario pregunta: "¿Hay estudios sobre diabetes tipo 2?"
# 2. LLM decide usar: PubMed + ClinicalTrials + Semantic Scholar
# 3. Ejecuta búsquedas en paralelo
# 4. Sintetiza resultados en una respuesta coherente
# 5. Guarda contexto para preguntas de seguimiento
```

#### 💡 **Casos de uso típicos:**
- "Busca estudios sobre diabetes tipo 2"
- "¿Qué ensayos clínicos hay sobre cáncer de pulmón?"
- "Encuentra información sobre el gen BRCA1"
- "¿Qué medicamentos están aprobados para la hipertensión?"

---

### 3. 🏥 **FHIRMedicalAgent** (`fhir_agent_complete.py`)
**El "Procesador de Datos Clínicos" Inteligente**

#### 🎯 **¿Qué hace realmente?**
- **Procesa notas clínicas** usando IA para extraer información estructurada
- **Convierte automáticamente** entre formatos FHIR y SQL
- **Valida datos médicos** según estándares FHIR
- **Mapea dinámicamente** campos sin hardcodeo
- **Gestiona relaciones** entre recursos médicos (pacientes, medicamentos, etc.)

#### 🧠 **Proceso interno detallado:**
```python
# 1. Recibe nota clínica: "Paciente María García, 45 años, toma paracetamol 500mg"
# 2. LLM extrae datos estructurados:
#    - Paciente: María García, 45 años
#    - Medicamento: Paracetamol 500mg
# 3. Genera recursos FHIR automáticamente
# 4. Mapea a tablas SQL usando LLM
# 5. Valida y corrige datos antes de insertar
```

#### 💡 **Casos de uso típicos:**
- "Procesa esta nota clínica: [texto]"
- "Convierte estos datos FHIR a SQL"
- "Inserta este paciente en la base de datos"
- "Busca medicaciones del paciente Carlos López"

---

### 4. 🗄️ **SQLAgentIntelligentEnhanced** (`sql_agent_flexible_enhanced.py`)
**El "Experto en Base de Datos" con IA**

#### 🎯 **¿Qué hace realmente?**
- **Analiza consultas en lenguaje natural** y las convierte a SQL
- **Explora automáticamente** el esquema de la base de datos
- **Mapea conceptos médicos** a tablas y columnas específicas
- **Corrige errores SQL** automáticamente usando LLM
- **Optimiza consultas** para mejor rendimiento
- **Interpreta resultados** médicos de forma inteligente

#### 🧠 **Sistema de caché inteligente:**
```python
# Mantiene caché de:
- Esquemas de tablas
- Mapeos exitosos de conceptos médicos
- Validaciones previas
- Consultas optimizadas
```

#### 💡 **Ejemplo de procesamiento:**
```python
# Usuario: "Muéstrame el último paciente con su medicación"
# 1. LLM analiza: "último paciente" = ORDER BY fecha DESC LIMIT 1
# 2. LLM mapea: "paciente" = tabla PATI_PATIENTS
# 3. LLM mapea: "medicación" = tabla PATI_USUAL_MEDICATION
# 4. LLM genera SQL con JOIN correcto
# 5. Ejecuta y interpreta resultados
```

#### 💡 **Casos de uso típicos:**
- "¿Cuántos pacientes tenemos en la base de datos?"
- "Muéstrame todos los pacientes con diabetes"
- "¿Qué medicamentos toma María García?"
- "Busca pacientes que tomen paracetamol"

---

### 5. 🧬 **MedGemmaClinicalAgent** (`medgemma_clinical_agent.py`)
**El "Analista Clínico" Especializado**

#### 🎯 **¿Qué hace realmente?**
- **Analiza datos clínicos** usando el modelo MedGemma
- **Valida diagnósticos** contra síntomas y evidencia
- **Explica conceptos médicos** de forma comprensible
- **Genera reportes clínicos** estructurados
- **Recomienda tratamientos** basados en evidencia
- **Proporciona fallback** con LLM cuando MedGemma no está disponible

#### 🧠 **Capacidades específicas:**
- **Análisis clínico**: Interpreta datos médicos complejos
- **Validación de diagnósticos**: Verifica coherencia entre síntomas y diagnóstico
- **Explicación médica**: Traduce conceptos técnicos a lenguaje comprensible
- **Generación de reportes**: Crea documentos clínicos estructurados
- **Recomendaciones**: Sugiere tratamientos basados en evidencia

#### 💡 **Casos de uso típicos:**
- "Analiza estos síntomas: [lista de síntomas]"
- "¿Es coherente este diagnóstico con estos síntomas?"
- "Explícame qué es la diabetes tipo 2"
- "Genera un reporte clínico para este paciente"

---

## 🔧 Agentes Especializados

### 6. 🔍 **IntelligentQueryClassifier** (`intelligent_query_classifier.py`)
**El "Clasificador de Consultas" Inteligente**

#### 🎯 **¿Qué hace realmente?**
- **Clasifica automáticamente** cada consulta según su tipo
- **Dirige la consulta** al agente más apropiado
- **Optimiza el flujo** de procesamiento
- **Aprende patrones** de clasificación exitosos

#### 💡 **Tipos de consultas que clasifica:**
- **Investigación**: Búsquedas en PubMed, ensayos clínicos
- **Datos clínicos**: Procesamiento de notas, FHIR
- **Base de datos**: Consultas SQL, análisis de datos
- **Análisis clínico**: Diagnósticos, validaciones médicas
- **Conversación**: Saludos, ayuda, información general

---

### 7. 🧪 **DrugDetectionAEMPS** (`drug_detection_aemps.py`)
**El "Detector de Medicamentos" Español**

#### 🎯 **¿Qué hace realmente?**
- **Detecta medicamentos** mencionados en texto
- **Verifica autorización** en la base de datos AEMPS
- **Proporciona información** sobre medicamentos españoles
- **Valida dosis** y presentaciones autorizadas

---

### 8. 📚 **PubMedQueryGenerator** (`pubmed_query_generator.py`)
**El "Generador de Queries" para PubMed**

#### 🎯 **¿Qué hace realmente?**
- **Optimiza queries** para búsquedas en PubMed
- **Usa términos MeSH** para búsquedas más precisas
- **Adapta queries** según el tipo de consulta
- **Mejora la relevancia** de los resultados

---

### 9. 🧠 **AdaptiveLearningSystem** (`adaptive_learning_system.py`)
**El "Sistema de Aprendizaje" Adaptativo**

#### 🎯 **¿Qué hace realmente?**
- **Aprende de patrones** exitosos de consultas
- **Adapta respuestas** según el historial del usuario
- **Optimiza automáticamente** el rendimiento del sistema
- **Mejora continuamente** la precisión de las respuestas

---

## 🛠️ Agentes de Soporte

### 10. 📦 **BatchProcessor** (`batch_processor.py`)
**El "Procesador por Lotes"**

#### 🎯 **¿Qué hace realmente?**
- **Procesa múltiples consultas** de forma eficiente
- **Optimiza recursos** del sistema
- **Maneja colas** de procesamiento
- **Proporciona feedback** de progreso

---

### 11. 🔧 **SQLAgentTools** (`sql_agent_tools.py`)
**Las "Herramientas SQL" Modulares**

#### 🎯 **¿Qué hace realmente?**
- **Proporciona herramientas** auxiliares para SQL
- **Limpia y valida** consultas SQL
- **Optimiza rendimiento** de consultas
- **Maneja errores** de base de datos

---

## 🔄 Flujo de Trabajo

### 📋 **Proceso Típico de una Consulta:**

1. **🎭 GreetingAgent** recibe la consulta
   - Analiza el contexto y la intención
   - Determina si es una consulta técnica o conversacional

2. **🔍 IntelligentQueryClassifier** clasifica la consulta
   - Decide qué agente debe procesarla
   - Optimiza el flujo de trabajo

3. **🎯 Agente Especializado** procesa la consulta:
   - **BioChatAgent**: Para investigación biomédica
   - **FHIRMedicalAgent**: Para datos clínicos
   - **SQLAgentEnhanced**: Para consultas de base de datos
   - **MedGemmaClinicalAgent**: Para análisis clínico

4. **🔄 Sistema de Aprendizaje** adapta y mejora
   - Aprende de patrones exitosos
   - Optimiza futuras consultas

---

## 💡 Casos de Uso

### 🏥 **Para Profesionales de la Salud:**

#### **Investigación Clínica:**
```
Usuario: "Busca estudios recientes sobre tratamientos para diabetes tipo 2"
→ BioChatAgent busca en PubMed, ClinicalTrials, Semantic Scholar
→ Sintetiza resultados en una respuesta coherente
```

#### **Análisis de Datos Clínicos:**
```
Usuario: "Procesa esta nota: Paciente de 45 años con HTA, toma enalapril"
→ FHIRMedicalAgent extrae datos estructurados
→ Convierte a recursos FHIR
→ Inserta en base de datos SQL
```

#### **Consultas de Base de Datos:**
```
Usuario: "¿Cuántos pacientes tenemos con diabetes?"
→ SQLAgentEnhanced genera SQL automáticamente
→ Ejecuta consulta optimizada
→ Interpreta resultados médicamente
```

### 🔬 **Para Investigadores:**

#### **Búsqueda de Literatura:**
```
Usuario: "Encuentra estudios sobre el gen BRCA1 en cáncer de mama"
→ BioChatAgent busca en múltiples fuentes
→ PubMedQueryGenerator optimiza la query
→ Devuelve resultados relevantes y actualizados
```

#### **Análisis de Ensayos Clínicos:**
```
Usuario: "¿Qué ensayos clínicos hay sobre inmunoterapia?"
→ BioChatAgent consulta ClinicalTrials.gov
→ Filtra por criterios específicos
→ Proporciona información detallada
```

---

## 🎯 **Conclusión**

ChatMed v2.0 Flexible representa un **paradigma avanzado** en sistemas de IA médica:

- **🧠 100% Dinámico**: Sin hardcodeo, todo se resuelve con LLM
- **🔄 Adaptativo**: Los agentes aprenden y mejoran continuamente
- **🎯 Especializado**: Cada agente tiene una función específica y optimizada
- **🔗 Integrado**: Los agentes trabajan en conjunto de forma fluida
- **📊 Transparente**: El usuario siempre sabe qué está pasando

Este sistema está diseñado para **revolucionar** la forma en que los profesionales de la salud interactúan con la información médica, proporcionando **herramientas inteligentes** que se adaptan a las necesidades específicas de cada usuario.

---

## 📞 **Soporte y Desarrollo**

**Desarrollado por:** Carmen Pascual (@carpasgis-dev) en Laberit

**Versión:** 2.0 Flexible

**Filosofía:** 100% dinámico sin hardcodeo - Todo se decide usando IA en tiempo real

---

*"La medicina del futuro no será solo más precisa, sino más inteligente y adaptativa."* 🧠🏥 