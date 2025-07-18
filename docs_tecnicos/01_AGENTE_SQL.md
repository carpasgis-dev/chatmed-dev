# 🗄️ Agente SQL - Documentación Técnica

## 📋 Información General

**Nombre del Agente:** `SQLAgentIntelligentEnhanced`  
**Clase Principal:** `agents.sql_agent_flexible_enhanced.SQLAgentIntelligentEnhanced`  
**Versión:** v5.0  
**Tipo:** Agente de consulta y análisis de base de datos médica  

## 🎯 Propósito y Funcionalidad

El Agente SQL es el componente especializado en consultas y análisis de datos médicos almacenados en la base de datos. Proporciona acceso inteligente a información de pacientes, diagnósticos, medicaciones y resultados de laboratorio para apoyar la toma de decisiones clínicas.

### Funciones Principales:
- **Consultas epidemiológicas** de datos médicos
- **Análisis estadístico** de información clínica
- **Búsquedas de pacientes** por criterios médicos
- **Generación automática** de SQL optimizado
- **Interpretación clínica** de resultados
- **Análisis de patrones** de prescripción y diagnóstico

## 🏗️ Arquitectura Técnica

### Componentes Principales:

#### 1. **Sistema de Análisis Clínico**
```python
# Análisis de consultas médicas usando LLM
clinical_analysis_prompt = f"""Analiza esta consulta médica y determina:
1. Tipo de análisis requerido (epidemiológico, clínico, farmacológico)
2. Criterios de filtrado (edad, patología, medicación)
3. Relaciones entre tablas necesarias
4. Métricas clínicas relevantes

CONSULTA: "{query}"

RESPONDE EN JSON:
{{
    "tipo_analisis": "epidemiologico|clinico|farmacologico",
    "criterios_filtrado": ["edad", "patologia", "medicacion"],
    "tablas_requeridas": ["PATI_PATIENTS", "EPIS_DIAGNOSTICS"],
    "metricas_clinicas": ["prevalencia", "incidencia", "distribucion"]
}}"""
```

#### 2. **Generación de SQL Clínico**
```python
# Generación de SQL para análisis médico
clinical_sql_prompt = f"""Genera una consulta SQL para análisis médico:

ANÁLISIS REQUERIDO:
{tipo_analisis}

CRITERIOS CLÍNICOS:
{criterios_filtrado}

ESQUEMA DE BASE DE DATOS:
{schema_info}

REGLAS CLÍNICAS:
1. Calcular edad desde PATI_BIRTH_DATE
2. Filtrar por patologías en DIAG_OBSERVATION
3. Buscar medicaciones en PAUM_OBSERVATIONS
4. Incluir estadísticas relevantes
5. Agrupar por criterios médicos

SQL GENERADO:"""
```

#### 3. **Sistema de Interpretación Clínica**
- **Análisis de prevalencia** de patologías
- **Estadísticas de prescripción** médica
- **Correlaciones** entre diagnósticos y tratamientos
- **Identificación de patrones** clínicos

## 📊 Preguntas Clínicamente Relevantes

### 1. **Análisis Epidemiológico**
```
❓ "¿Cuántos pacientes mayores de 65 años toman metformina?"
❓ "¿Qué patologías tienen los pacientes que toman insulina?"
❓ "¿Cuál es la prevalencia de diabetes por grupos de edad?"
❓ "¿Cuántos pacientes con hipertensión toman múltiples medicamentos?"
```

**SQL Generado:**
```sql
SELECT 
    COUNT(*) as total_pacientes,
    AVG(CAST((julianday('now') - julianday(p.PATI_BIRTH_DATE))/365.25 AS INTEGER)) as edad_promedio,
    GROUP_CONCAT(DISTINCT d.DIAG_OBSERVATION) as patologias
FROM PATI_PATIENTS p
JOIN PATI_USUAL_MEDICATION m ON p.PATI_ID = m.PATI_ID
LEFT JOIN EPIS_DIAGNOSTICS d ON p.PATI_ID = d.PATI_ID
WHERE CAST((julianday('now') - julianday(p.PATI_BIRTH_DATE))/365.25 AS INTEGER) > 65
AND m.PAUM_OBSERVATIONS LIKE '%metformina%'
GROUP BY p.PATI_ID
```

**Resultado Clínico:**
```
📊 ANÁLISIS EPIDEMIOLÓGICO:
├── Pacientes >65 años con metformina: 23
├── Edad promedio: 72.3 años
├── Patologías asociadas:
│   ├── Diabetes mellitus tipo 2: 18 pacientes
│   ├── Hipertensión arterial: 15 pacientes
│   └── Dislipidemia: 12 pacientes
└── Prevalencia: 15.2% de pacientes >65 años
```

### 2. **Análisis de Prescripción Médica**
```
❓ "¿Qué medicamentos se prescriben más en pacientes con diabetes?"
❓ "¿Cuántos pacientes toman múltiples antihipertensivos?"
❓ "¿Cuál es la combinación más frecuente de medicamentos?"
❓ "¿Qué pacientes tienen polimedicación (>5 fármacos)?"
```

**SQL Generado:**
```sql
SELECT 
    m.PAUM_OBSERVATIONS as medicamento,
    COUNT(*) as frecuencia_prescripcion,
    COUNT(DISTINCT p.PATI_ID) as pacientes_unicos,
    GROUP_CONCAT(DISTINCT d.DIAG_OBSERVATION) as indicaciones
FROM PATI_USUAL_MEDICATION m
JOIN PATI_PATIENTS p ON m.PATI_ID = p.PATI_ID
LEFT JOIN EPIS_DIAGNOSTICS d ON p.PATI_ID = d.PATI_ID
WHERE d.DIAG_OBSERVATION LIKE '%diabetes%'
GROUP BY m.PAUM_OBSERVATIONS
ORDER BY frecuencia_prescripcion DESC
```

**Resultado Clínico:**
```
💊 ANÁLISIS DE PRESCRIPCIÓN:
├── Metformina: 45 prescripciones (32 pacientes)
├── Insulina glargina: 28 prescripciones (22 pacientes)
├── Glimepirida: 15 prescripciones (12 pacientes)
└── Indicaciones principales: Diabetes mellitus tipo 2
```

### 3. **Análisis de Comorbilidades**
```
❓ "¿Qué pacientes tienen diabetes + hipertensión + dislipidemia?"
❓ "¿Cuántos pacientes con insuficiencia cardíaca toman betabloqueantes?"
❓ "¿Cuál es la prevalencia de síndrome metabólico?"
❓ "¿Qué pacientes tienen múltiples factores de riesgo cardiovascular?"
```

**SQL Generado:**
```sql
SELECT 
    p.PATI_ID,
    p.PATI_FULL_NAME,
    COUNT(DISTINCT d.DIAG_OBSERVATION) as numero_comorbilidades,
    GROUP_CONCAT(DISTINCT d.DIAG_OBSERVATION) as lista_comorbilidades,
    GROUP_CONCAT(DISTINCT m.PAUM_OBSERVATIONS) as medicacion_actual
FROM PATI_PATIENTS p
JOIN EPIS_DIAGNOSTICS d ON p.PATI_ID = d.PATI_ID
LEFT JOIN PATI_USUAL_MEDICATION m ON p.PATI_ID = m.PATI_ID
WHERE d.DIAG_OBSERVATION IN ('diabetes', 'hipertensión', 'dislipidemia')
GROUP BY p.PATI_ID
HAVING COUNT(DISTINCT d.DIAG_OBSERVATION) >= 3
```

**Resultado Clínico:**
```
🏥 ANÁLISIS DE COMORBILIDADES:
├── Pacientes con síndrome metabólico: 18
├── Comorbilidades promedio: 3.2 por paciente
├── Medicación más frecuente:
│   ├── Metformina + Enalapril + Atorvastatina: 8 pacientes
│   └── Insulina + Amlodipino + Simvastatina: 6 pacientes
└── Riesgo cardiovascular: ALTO en 15 pacientes
```

### 4. **Análisis por Grupos de Edad**
```
❓ "¿Cuántos pacientes jóvenes (<40) tienen diabetes tipo 1?"
❓ "¿Qué medicamentos toman los pacientes de 40-60 años?"
❓ "¿Cuál es la prevalencia de hipertensión por décadas de edad?"
❓ "¿Qué pacientes geriátricos (>80) tienen polimedicación?"
```

**SQL Generado:**
```sql
SELECT 
    CASE 
        WHEN CAST((julianday('now') - julianday(p.PATI_BIRTH_DATE))/365.25 AS INTEGER) < 40 THEN 'Jóvenes (<40)'
        WHEN CAST((julianday('now') - julianday(p.PATI_BIRTH_DATE))/365.25 AS INTEGER) < 60 THEN 'Adultos (40-60)'
        WHEN CAST((julianday('now') - julianday(p.PATI_BIRTH_DATE))/365.25 AS INTEGER) < 80 THEN 'Mayores (60-80)'
        ELSE 'Geriatría (>80)'
    END as grupo_edad,
    COUNT(DISTINCT p.PATI_ID) as total_pacientes,
    COUNT(DISTINCT CASE WHEN d.DIAG_OBSERVATION LIKE '%diabetes%' THEN p.PATI_ID END) as pacientes_diabetes,
    COUNT(DISTINCT CASE WHEN d.DIAG_OBSERVATION LIKE '%hipertensión%' THEN p.PATI_ID END) as pacientes_hipertension
FROM PATI_PATIENTS p
LEFT JOIN EPIS_DIAGNOSTICS d ON p.PATI_ID = d.PATI_ID
GROUP BY grupo_edad
ORDER BY grupo_edad
```

**Resultado Clínico:**
```
📊 ANÁLISIS POR GRUPOS DE EDAD:
├── Jóvenes (<40): 45 pacientes
│   ├── Diabetes tipo 1: 8 pacientes (17.8%)
│   └── Hipertensión: 3 pacientes (6.7%)
├── Adultos (40-60): 78 pacientes
│   ├── Diabetes tipo 2: 23 pacientes (29.5%)
│   └── Hipertensión: 31 pacientes (39.7%)
├── Mayores (60-80): 92 pacientes
│   ├── Diabetes: 34 pacientes (37.0%)
│   └── Hipertensión: 67 pacientes (72.8%)
└── Geriatría (>80): 23 pacientes
    ├── Diabetes: 8 pacientes (34.8%)
    └── Hipertensión: 19 pacientes (82.6%)
```

### 5. **Análisis de Seguimiento Clínico**
```
❓ "¿Qué pacientes no han tenido seguimiento en los últimos 6 meses?"
❓ "¿Cuántos pacientes tienen valores de HbA1c > 7%?"
❓ "¿Qué pacientes necesitan ajuste de medicación?"
❓ "¿Cuál es el control glucémico promedio por paciente?"
```

**SQL Generado:**
```sql
SELECT 
    p.PATI_ID,
    p.PATI_FULL_NAME,
    d.DIAG_OBSERVATION as diagnostico_principal,
    m.PAUM_OBSERVATIONS as medicacion_actual,
    l.PROC_DESCRIPTION as ultimo_laboratorio,
    CASE 
        WHEN l.PROC_DESCRIPTION LIKE '%HbA1c%' AND l.PROC_DESCRIPTION LIKE '%>7%' THEN 'Control deficiente'
        WHEN l.PROC_DESCRIPTION LIKE '%HbA1c%' AND l.PROC_DESCRIPTION LIKE '%<7%' THEN 'Control adecuado'
        ELSE 'Sin datos recientes'
    END as estado_control
FROM PATI_PATIENTS p
LEFT JOIN EPIS_DIAGNOSTICS d ON p.PATI_ID = d.PATI_ID
LEFT JOIN PATI_USUAL_MEDICATION m ON p.PATI_ID = m.PATI_ID
LEFT JOIN PROC_PROCEDURES l ON p.PATI_ID = l.PATI_ID
WHERE d.DIAG_OBSERVATION LIKE '%diabetes%'
ORDER BY p.PATI_ID
```

## 🔄 Flujo de Trabajo Clínico

### **Proceso de Análisis Epidemiológico:**

```
1. 📋 CONSULTA CLÍNICA
   "¿Cuántos pacientes mayores de 65 años toman metformina?"

2. 🧠 ANÁLISIS SEMÁNTICO
   - Tipo: Análisis epidemiológico
   - Criterios: Edad >65, medicación = metformina
   - Tablas: PATI_PATIENTS, PATI_USUAL_MEDICATION, EPIS_DIAGNOSTICS

3. 🔍 GENERACIÓN SQL
   - Cálculo de edad desde fecha de nacimiento
   - Filtrado por medicación específica
   - JOIN con diagnósticos para contexto clínico

4. 📊 EJECUCIÓN Y ANÁLISIS
   - Conteo de pacientes que cumplen criterios
   - Análisis de comorbilidades asociadas
   - Cálculo de prevalencia en la población

5. 🏥 INTERPRETACIÓN CLÍNICA
   - Identificación de patrones de prescripción
   - Análisis de efectividad del tratamiento
   - Recomendaciones para seguimiento
```

### **Proceso de Análisis de Prescripción:**

```
1. 📋 CONSULTA FARMACOLÓGICA
   "¿Qué medicamentos se prescriben más en diabetes?"

2. 🧠 ANÁLISIS DE PATRONES
   - Identificación de medicamentos más frecuentes
   - Análisis de combinaciones terapéuticas
   - Evaluación de adherencia al tratamiento

3. 🔍 SQL EPIDEMIOLÓGICO
   - Agrupación por medicamento
   - Conteo de prescripciones y pacientes únicos
   - Correlación con diagnósticos

4. 📊 ESTADÍSTICAS CLÍNICAS
   - Frecuencia de prescripción por medicamento
   - Distribución por grupos de edad
   - Análisis de polimedicación

5. 💊 RECOMENDACIONES CLÍNICAS
   - Identificación de tratamientos estándar
   - Detección de prescripciones atípicas
   - Sugerencias de optimización terapéutica
```

## 🔧 Funciones Técnicas Principales

### 1. **`process_clinical_query(query: str)`**
**Propósito:** Procesamiento de consultas clínicas complejas  
**Entrada:** Consulta médica en lenguaje natural  
**Salida:** Análisis epidemiológico con interpretación clínica  

```python
async def process_clinical_query(self, query: str) -> Dict[str, Any]:
    """
    🏥 Procesamiento de consultas clínicas con análisis epidemiológico
    """
    # 1. Análisis semántico clínico
    # 2. Identificación de criterios médicos
    # 3. Generación de SQL epidemiológico
    # 4. Interpretación clínica de resultados
    # 5. Recomendaciones médicas
```

### 2. **`analyze_epidemiological_patterns(criteria: Dict)`**
**Propósito:** Análisis de patrones epidemiológicos  
**Características:** Análisis de prevalencia, incidencia y correlaciones  

```python
async def analyze_epidemiological_patterns(self, criteria: Dict) -> Dict[str, Any]:
    """
    📊 Análisis epidemiológico de patrones clínicos
    """
    # 1. Cálculo de prevalencia por grupos
    # 2. Análisis de factores de riesgo
    # 3. Correlación entre diagnósticos y tratamientos
    # 4. Identificación de patrones de prescripción
```

### 3. **`generate_clinical_sql(analysis: Dict)`**
**Propósito:** Generación de SQL para análisis clínico  
**Características:** SQL optimizado para consultas médicas complejas  

```python
async def generate_clinical_sql(self, analysis: Dict) -> str:
    """
    🔍 Genera SQL optimizado para análisis clínico
    """
    # 1. Mapeo de criterios clínicos a SQL
    # 2. Optimización para consultas epidemiológicas
    # 3. Inclusión de estadísticas relevantes
    # 4. Validación de esquema médico
```

## 🗃️ Estructura de Base de Datos Médica

### Tablas Principales:

#### **PATI_PATIENTS** (Pacientes)
```sql
- PATI_ID (PRIMARY KEY) - Identificador único del paciente
- PATI_NAME (Nombre) - Nombre del paciente
- PATI_SURNAME_1 (Primer apellido) - Primer apellido
- PATI_FULL_NAME (Nombre completo) - Nombre completo
- PATI_BIRTH_DATE (Fecha de nacimiento) - Para cálculo de edad
- PATI_START_DATE (Fecha de inicio de atención) - Seguimiento temporal
- PATI_ACTIVE (Estado activo) - Paciente activo en el sistema
```

#### **EPIS_DIAGNOSTICS** (Diagnósticos)
```sql
- DIAG_ID (PRIMARY KEY) - Identificador del diagnóstico
- PATI_ID (FOREIGN KEY) - Referencia al paciente
- DIAG_OBSERVATION (Observación diagnóstica) - Diagnóstico principal
- DIAG_DESCRIPTION (Descripción del diagnóstico) - Detalles clínicos
```

#### **PATI_USUAL_MEDICATION** (Medicación)
```sql
- PAUM_ID (PRIMARY KEY) - Identificador de la prescripción
- PATI_ID (FOREIGN KEY) - Referencia al paciente
- PAUM_OBSERVATIONS (Observaciones de medicación) - Medicamento y dosis
```

#### **PROC_PROCEDURES** (Procedimientos/Laboratorio)
```sql
- PROC_ID (PRIMARY KEY) - Identificador del procedimiento
- PATI_ID (FOREIGN KEY) - Referencia al paciente
- PROC_DESCRIPTION (Descripción del procedimiento) - Resultados de laboratorio
```

## 📈 Métricas Clínicas

### Indicadores Epidemiológicos:
- **Prevalencia:** Porcentaje de pacientes con una condición específica
- **Incidencia:** Nuevos casos por período de tiempo
- **Distribución por edad:** Análisis por grupos demográficos
- **Comorbilidades:** Múltiples condiciones en un mismo paciente
- **Polimedicación:** Pacientes con múltiples medicamentos

### Logs de Análisis Clínico:
```python
logger.info(f"📊 Análisis epidemiológico completado: {prevalencia}% prevalencia")
logger.info(f"💊 Patrones de prescripción identificados: {num_patrones}")
logger.info(f"🏥 Comorbilidades analizadas: {num_comorbilidades}")
```

## 🛠️ Configuración y Uso Clínico

### Inicialización:
```python
sql_agent = SQLAgentIntelligentEnhanced(
    db_path="database_new.sqlite3.db",
    llm=llm_instance,
    medgemma_agent=medgemma_agent
)
```

### Ejemplos de Uso Clínico:
```python
# Análisis epidemiológico
result = await sql_agent.process_query("¿Cuántos pacientes mayores de 65 años toman metformina?")

# Análisis de prescripción
result = await sql_agent.process_query("¿Qué medicamentos se prescriben más en diabetes?")

# Análisis de comorbilidades
result = await sql_agent.process_query("¿Qué pacientes tienen diabetes + hipertensión + dislipidemia?")

# Análisis por grupos de edad
result = await sql_agent.process_query("¿Cuál es la prevalencia de hipertensión por décadas de edad?")
```

## 🔧 Troubleshooting Clínico

### Problemas Comunes:

#### 1. **Cálculo Incorrecto de Edad**
**Síntoma:** Edades negativas o incorrectas  
**Solución:** Verificar formato de fecha y fórmula de cálculo

#### 2. **Filtrado Incompleto de Medicamentos**
**Síntoma:** No encuentra medicamentos específicos  
**Solución:** Usar LIKE con variaciones del nombre del medicamento

#### 3. **Análisis de Comorbilidades Incompleto**
**Síntoma:** No detecta múltiples diagnósticos  
**Solución:** Usar GROUP_CONCAT y HAVING para múltiples condiciones

## 📚 Referencias Clínicas

### Archivos Principales:
- `agents/sql_agent_flexible_enhanced.py` - Implementación principal
- `utils/sql_cleaner.py` - Limpieza de SQL clínico
- `utils/sql_executor.py` - Ejecución de consultas médicas
- `utils/sql_generator.py` - Generación de SQL epidemiológico

### Dependencias:
- `sqlite3` - Base de datos médica
- `langchain_openai` - LLM para análisis clínico
- `asyncio` - Procesamiento asíncrono
- `logging` - Sistema de logs clínicos

---

**Versión:** 2.0 - Enfoque Clínico  
**Última actualización:** 2025-01-18  
**Mantenido por:** Equipo de Desarrollo ChatMed
