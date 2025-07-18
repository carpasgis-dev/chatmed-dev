# 📚 BIBLIOGRAFÍA DE REFERENCIA: Esquema de Base de Datos Médica

## 🎯 RESUMEN EJECUTIVO
Documentación completa extraída del esquema `schema_complete.dot` con 243 tablas identificadas para el sistema de introspección automática SQL→FHIR.

---

## 📊 ESTADÍSTICAS GENERALES
- **Total tablas**: 243
- **Prefijos identificados**: 52
- **Dominios funcionales**: 15
- **Tipos de datos**: 8 principales

---

## 🏗️ PATRONES DE PREFIJOS Y FUNCIONALIDAD

### **CORE MÉDICO**
```yaml
# Pacientes y Demographics
PATI_: 15 tablas
  - PATI_PATIENTS (tabla principal)
  - PATI_PATIENT_ALLERGIES
  - PATI_PATIENT_ADDRESSES
  - PATI_PATIENT_CONTACTS
  - PATI_PATIENT_HEALTH_IDS
  - PATI_PATIENT_IDENTIFICATIONS
  - PATI_PATIENT_PHONES
  - PATI_PATIENT_TYPES
  - PATI_USUAL_MEDICATION
  - PATI_EXITUS
  - PATI_MERGE_PATIENTS
  - PATI_ORIGIN_PROFESSIONALS
  - PATI_PATIENT_CONTACT_PERSONS
  - Mapeo FHIR: Patient + RelatedPerson + ContactPoint

# Episodios Clínicos
EPIS_: 12 tablas
  - EPIS_EPISODES (tabla principal)
  - EPIS_DIAGNOSTICS
  - EPIS_PROCEDURES
  - EPIS_ADMIN_DISCHARGE_REASONS
  - EPIS_DESTINATIONS
  - EPIS_STATES
  - EPIS_ISOLATION_TYPES
  - EPIS_DIET_TYPES
  - EPIS_DISCHARGE_SHIFTS
  - Mapeo FHIR: Encounter + EpisodeOfCare
```

### **PROCEDIMIENTOS Y TRATAMIENTOS**
```yaml
# Procedimientos
PROC_: 8 tablas
  - PROC_PROCEDURES (tabla principal)
  - PROC_PROCEDURE_TYPES
  - PROC_PROCEDURE_VIEW_TYPES
  - PROC_SURGICAL_DESTINATIONS
  - PROC_SURGICAL_TIMES
  - Mapeo FHIR: Procedure + ServiceRequest

# Tratamientos
APPR_: 7 tablas
  - APPR_TREATMENTS (tabla principal)
  - APPR_TREATMENT_STATES
  - APPR_TREATMENT_ACTIONS
  - APPR_TREATMENT_DISCHARGE_TYPE
  - APPR_TREATMENT_DEST_DISCHARGE
  - Mapeo FHIR: CarePlan + Task
```

### **MEDICAMENTOS Y ALERGIAS**
```yaml
# Medicamentos
MEDI_: 10 tablas
  - MEDI_MEDICATIONS (tabla principal)
  - MEDI_ACTIVE_INGREDIENTS
  - MEDI_MEDICATION_GROUPS
  - MEDI_GROUPS
  - MEDI_PHARMA_THERAPEUTIC_GROUPS
  - Mapeo FHIR: Medication + MedicationRequest

# Alergias
ALLE_: 8 tablas
  - ALLE_ALLERGIES (tabla principal)
  - ALLE_ALLERGY_TYPES
  - ALLE_ALLERGY_SEVERITY_LEVELS
  - ALLE_ALLERGY_RESULTS
  - ALLE_ALLERGY_CATEGORIES
  - ALLE_NOT_MEDICINAL_ALLERGENS
  - Mapeo FHIR: AllergyIntolerance
```

### **DIAGNÓSTICOS Y CODIFICACIÓN**
```yaml
# Diagnósticos
DIAG_: 3 tablas
  - DIAG_DIAGNOSTIC_STATES
  - DIAG_LATERALITIES
  - Mapeo FHIR: Condition + DiagnosticReport

# Codificación (ICD-10, SNOMED)
CODR_: 8 tablas
  - CODR_TABULAR_DIAGNOSTICS (tabla principal)
  - CODR_TABULAR_PROCEDURES
  - CODR_CATALOGS
  - CODR_CODING_TYPES
  - CODR_DIAGNOSTIC_GROUPS
  - CODR_DIAGNOSTIC_TYPES
  - Mapeo FHIR: CodeSystem + ValueSet
```

### **OBSERVACIONES Y PRUEBAS**
```yaml
# Observaciones Clínicas
OBSE_: 6 tablas
  - OBSE_OBSERVATIONS (tabla principal)
  - OBSE_OBSERVATION_TYPES
  - OBSE_OBSERVATION_STATES
  - Mapeo FHIR: Observation + DiagnosticReport

# Pruebas y Tests
TEST_: 5 tablas
  - TEST_TESTS (tabla principal)
  - TEST_TEST_TYPES
  - TEST_TEST_SUBTYPES
  - TEST_PROFILES
  - TEST_STATES
  - Mapeo FHIR: DiagnosticReport + Observation
```

### **RECURSOS HOSPITALARIOS**
```yaml
# Camas y Recursos
BEDS_: 3 tablas
  - BEDS_BEDS (tabla principal)
  - BEDS_BED_TYPES
  - BEDS_CONSTRAINTS
  - Mapeo FHIR: Location + Device

# Organización
ORMA_: 4 tablas
  - ORMA_SERVICES
  - ORMA_UNITS
  - ORMA_DEPARTMENTS
  - Mapeo FHIR: Organization + Location
```

### **CUIDADOS Y CURACIONES**
```yaml
# Cuidados
CARE_: 2 tablas
  - CARE_CARES
  - Mapeo FHIR: CarePlan + Task

# Curaciones
CURE_: 15 tablas
  - CURE_CURES (tabla principal)
  - CURE_PHASES
  - CURE_TYPES
  - CURE_CLEAN_TYPES
  - CURE_DRAINAGE_TYPES
  - CURE_EXUDE_TYPES
  - CURE_PRIMARY_DRESSINGS
  - CURE_SKIN_VALUATIONS
  - Mapeo FHIR: Procedure + Observation
```

### **PROPUESTAS Y APPOINTMENTS**
```yaml
# Propuestas
APPO_: 8 tablas
  - APPO_PROPOSALS (tabla principal)
  - APPO_PROPOSAL_TYPES
  - APPO_PROP_ATTENTION_TYPES
  - Mapeo FHIR: Appointment + ServiceRequest

# Documentos
DOCS_: 3 tablas
  - DOCS_DOCUMENTS (tabla principal)
  - DOCS_DOCUMENT_TYPES
  - DOCS_TYPE_CATEGORIES
  - Mapeo FHIR: DocumentReference + Media
```

---

## 🔍 PATRONES DE CAMPOS CRÍTICOS

### **CAMPOS UNIVERSALES**
```yaml
# Campos en TODAS las tablas
MTIME: datetime2          # Modificación timestamp
_ID: PK                   # Clave primaria
_DELETED: bit             # Soft delete flag
_DESCRIPTION_ES: nvarchar # Descripción en español

# Campos de Auditoría
_CREATION_DATE: datetime2
_UPDATE_DATE: datetime2
_START_DATE: datetime2
_END_DATE: datetime2
```

### **CAMPOS DE RELACIÓN**
```yaml
# Referencias a Pacientes
PATI_ID: INT             # Referencia a PATI_PATIENTS

# Referencias a Episodios
EPIS_ID: bigint          # Referencia a EPIS_EPISODES

# Referencias a Procedimientos
PROC_ID: INT             # Referencia a PROC_PROCEDURES

# Referencias a Codificación
CDTE_ID: bigint          # Referencia a CODR_TABULAR_DIAGNOSTICS
CPTE_ID: bigint          # Referencia a CODR_TABULAR_PROCEDURES
```

### **CAMPOS DE ESTADO**
```yaml
# Estados Generales
*_STATE: tinyint/smallint
*_STATUS: tinyint/smallint
*_ACTIVE: bit
*_ENABLED: bit

# Tipos y Categorías
*_TYPE: tinyint/smallint
*_CATEGORY: tinyint/smallint
*_CLASS: tinyint/smallint
```

---

## 🎨 TIPOS DE DATOS Y TRANSFORMACIONES

### **MAPEO SQL → FHIR**
```yaml
# Tipos Básicos
INT/bigint → integer
tinyint/smallint → integer
nvarchar → string
datetime2 → dateTime
date → date
bit → boolean
decimal/float → decimal

# Tipos Especiales
BLOB → base64Binary
JSON → json
XML → xml
```

### **TRANSFORMACIONES DE VALORES**
```yaml
# Booleanos
bit 1 → true
bit 0 → false
NULL → null

# Fechas
datetime2 → ISO 8601 format
date → YYYY-MM-DD format

# Códigos
*_CODE → CodeableConcept
*_CODING → Coding
```

---

## 🔗 RELACIONES CRÍTICAS IDENTIFICADAS

### **RELACIONES PRIMARIAS**
```yaml
# Paciente Central
PATI_PATIENTS ← EPIS_EPISODES
PATI_PATIENTS ← PATI_PATIENT_ALLERGIES
PATI_PATIENTS ← PROC_PROCEDURES
PATI_PATIENTS ← OBSE_OBSERVATIONS

# Episodio Clínico
EPIS_EPISODES ← EPIS_DIAGNOSTICS
EPIS_EPISODES ← EPIS_PROCEDURES
EPIS_EPISODES ← APPR_TREATMENTS

# Codificación
CODR_TABULAR_DIAGNOSTICS ← EPIS_DIAGNOSTICS
CODR_TABULAR_PROCEDURES ← EPIS_PROCEDURES
```

### **RELACIONES JERÁRQUICAS**
```yaml
# Organizacional
ORMA_SERVICES → ORMA_UNITS → ORMA_DEPARTMENTS

# Medicamentos
MEDI_MEDICATIONS → MEDI_GROUPS → MEDI_PHARMA_THERAPEUTIC_GROUPS

# Síntomas (Auto-referencia)
TRIA_SYMPTOMS.TRSY_ID_PARENT → TRIA_SYMPTOMS.TRSY_ID
```

---

## 🎯 PATRONES DE INFERENCIA FHIR

### **REGLAS DE INFERENCIA AUTOMÁTICA**
```yaml
# Por Prefijo de Tabla
PATI_* → Patient resource
EPIS_* → Encounter resource
PROC_* → Procedure resource
OBSE_* → Observation resource
MEDI_* → Medication resource
ALLE_* → AllergyIntolerance resource

# Por Sufijo de Campo
*_DATE → dateTime element
*_CODE → CodeableConcept
*_DESCRIPTION → text element
*_OBSERVATION → text element
*_RESULT → value element

# Por Tipo de Dato
bit → boolean
datetime2 → dateTime
nvarchar → string
INT → integer
```

### **PATRONES DE CARDINALIDAD**
```yaml
# Uno a Muchos
PATI_PATIENTS → EPIS_EPISODES (1:N)
EPIS_EPISODES → EPIS_DIAGNOSTICS (1:N)
EPIS_EPISODES → PROC_PROCEDURES (1:N)

# Muchos a Uno
EPIS_DIAGNOSTICS → CODR_TABULAR_DIAGNOSTICS (N:1)
PROC_PROCEDURES → PROC_PROCEDURE_TYPES (N:1)

# Muchos a Muchos (través de tablas intermedias)
PATI_PATIENTS ←→ MEDI_MEDICATIONS (via PATI_USUAL_MEDICATION)
```

---

## 🚀 CONFIGURACIÓN PARA INTROSPECTOR

### **TABLAS PRIORITARIAS** (Top 20)
```yaml
# Núcleo Esencial
1. PATI_PATIENTS
2. EPIS_EPISODES
3. EPIS_DIAGNOSTICS
4. PROC_PROCEDURES
5. OBSE_OBSERVATIONS
6. MEDI_MEDICATIONS
7. ALLE_ALLERGIES
8. CODR_TABULAR_DIAGNOSTICS
9. CODR_TABULAR_PROCEDURES
10. APPR_TREATMENTS

# Recursos Complementarios
11. PATI_PATIENT_ALLERGIES
12. DOCS_DOCUMENTS
13. ORMA_SERVICES
14. BEDS_BEDS
15. TEST_TESTS
16. VACC_VACCINES
17. CURE_CURES
18. APPO_PROPOSALS
19. EPIS_PROCEDURES
20. TRIA_SYMPTOMS
```

### **CAMPOS OBLIGATORIOS**
```yaml
# Campos que SIEMPRE deben estar en FHIR
*_ID → id
*_DESCRIPTION_ES → text
MTIME → meta.lastUpdated
PATI_ID → subject.reference
EPIS_ID → encounter.reference
```

---

## 📝 NOTAS DE IMPLEMENTACIÓN

### **PRIORIDADES DE DESARROLLO**
1. **Fase 1**: Implementar 20 tablas prioritarias
2. **Fase 2**: Agregar patrones de inferencia automática
3. **Fase 3**: Completar todas las 243 tablas
4. **Fase 4**: Optimizar rendimiento y cache

### **CONSIDERACIONES ESPECIALES**
- Tablas con prefijo `sqlite_autoindex_*` → Ignorar (índices automáticos)
- Tablas con `_DELETED = 1` → Manejar soft deletes
- Campos con `MTIME` → Usar para versionado
- Relaciones FK → Generar references automáticamente

---

**📊 ESTADÍSTICAS FINALES**
- Prefijos únicos: 52
- Patrones de inferencia: 15
- Reglas de transformación: 25
- Relaciones identificadas: 150+

**🎯 OBJETIVO**: Sistema 100% automático de introspección SQL→FHIR basado en esta bibliografía. 