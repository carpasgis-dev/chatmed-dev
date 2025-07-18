# 🗄️ Agente SQL - Documentación Técnica

## 📋 Información General

**Nombre del Agente:** `SQLAgentIntelligentEnhanced`  
**Clase Principal:** `agents.sql_agent_flexible_enhanced.SQLAgentIntelligentEnhanced`  
**Versión:** v5.0  
**Tipo:** Agente de consulta y análisis de base de datos médica  

## 🎯 Propósito y Funcionalidad

El Agente SQL es el componente especializado en consultas y análisis de datos médicos almacenados en la base de datos. Proporciona acceso inteligente a información de pacientes, diagnósticos, medicaciones y resultados de laboratorio.

### Funciones Principales:
- **Consultas dinámicas** de datos médicos
- **Análisis estadístico** de información clínica
- **Búsquedas inteligentes** de pacientes
- **Generación automática** de SQL optimizado
- **Validación y corrección** de consultas
- **Interpretación clínica** de resultados

## 🏗️ Arquitectura Técnica

### Componentes Principales:

#### 1. **Sistema de Detección Inteligente**
```python
# Detección de consultas de último paciente usando LLM
detection_prompt = f"""Analiza esta consulta y determina si se refiere al ÚLTIMO PACIENTE registrado en la base de datos.

CONSULTA: "{query}"

CRITERIOS PARA DETECTAR CONSULTAS DE ÚLTIMO PACIENTE:
- Palabras clave: "último", "ultimo", "última", "ultima", "reciente", "creado", "registrado"
- Frases: "último paciente", "ultimo paciente", "último paciente creado", "ultimo paciente creado"
- Preguntas: "¿Cuál es el último paciente?", "¿Quién es el último paciente?", "¿Dime el último paciente?"
- Variaciones: "cual es el ultimo", "cuál es el último", "dime el ultimo", "dime el último", "quien es el ultimo", "quién es el último"

Responde SOLO con "SÍ" si es una consulta de último paciente, o "NO" si no lo es."""
```

#### 2. **Generación de SQL con Doble LLM**
```python
# PRIMERA LLAMADA: Detectar tipo de consulta
# SEGUNDA LLAMADA: Generar SQL optimizado
sql_prompt = f"""Genera una consulta SQL optimizada para obtener información del ÚLTIMO PACIENTE registrado en la base de datos.

REGLAS OBLIGATORIAS:
- Usar SOLO PATI_ID DESC para determinar el último paciente (NO usar PATI_START_DATE ni PATI_UPDATE_DATE)
- Incluir campos: PATI_ID, PATI_NAME, PATI_SURNAME_1, PATI_FULL_NAME
- Usar ORDER BY PATI_ID DESC LIMIT 1
- Tabla: PATI_PATIENTS

EJEMPLO CORRECTO:
SELECT PATI_ID, PATI_NAME, PATI_SURNAME_1, PATI_FULL_NAME 
FROM PATI_PATIENTS 
ORDER BY PATI_ID DESC 
LIMIT 1"""
```

#### 3. **Sistema de Validación Robusta**
- **Validación de sintaxis** SQL
- **Compatibilidad** con SQLite
- **Corrección automática** de errores
- **Validación de esquema** en tiempo real

## 📊 Preguntas Clínicamente Relevantes

### 1. **Consultas de Último Paciente**
```
❓ "¿Cuál es el último paciente creado?"
❓ "¿Cómo se llama el último paciente registrado?"
❓ "Dime el último paciente"
❓ "Quién es el último paciente"
```

**SQL Generado:**
```sql
SELECT PATI_ID, PATI_NAME, PATI_SURNAME_1, PATI_FULL_NAME 
FROM PATI_PATIENTS 
ORDER BY PATI_ID DESC 
LIMIT 1
```

### 2. **Búsquedas de Pacientes Específicos**
```
❓ "Muéstrame todos los datos de María del Carmen incluyendo diagnósticos, medicación y laboratorio"
❓ "Busca pacientes con diabetes"
❓ "Encuentra pacientes con hipertensión"
❓ "Pacientes con diagnóstico de cáncer"
```

**SQL Generado:**
```sql
SELECT 
    p.*,
    d.DIAG_OBSERVATION,
    m.PAUM_OBSERVATIONS,
    l.PROC_DESCRIPTION
FROM PATI_PATIENTS p
LEFT JOIN EPIS_DIAGNOSTICS d ON p.PATI_ID = d.PATI_ID
LEFT JOIN PATI_USUAL_MEDICATION m ON p.PATI_ID = m.PATI_ID
LEFT JOIN PROC_PROCEDURES l ON p.PATI_ID = l.PATI_ID
WHERE p.PATI_NAME LIKE '%María del Carmen%'
```

### 3. **Análisis Estadístico**
```
❓ "¿Cuántos pacientes hay en total?"
❓ "¿Cuántos pacientes tienen diabetes?"
❓ "Estadísticas de pacientes por edad"
❓ "Distribución de diagnósticos"
```

**SQL Generado:**
```sql
SELECT COUNT(DISTINCT p.PATI_ID) as total_pacientes
FROM PATI_PATIENTS p
```

### 4. **Consultas de Diagnósticos**
```
❓ "Pacientes con diagnóstico de diabetes"
❓ "Busca diagnósticos relacionados con cardiología"
❓ "Pacientes con múltiples diagnósticos"
```

**SQL Generado:**
```sql
SELECT p.PATI_ID, p.PATI_NAME, d.DIAG_OBSERVATION 
FROM PATI_PATIENTS p 
JOIN EPIS_DIAGNOSTICS d ON p.PATI_ID = d.PATI_ID 
WHERE d.DIAG_OBSERVATION LIKE '%diabetes%'
```

### 5. **Consultas de Medicación**
```
❓ "¿Qué medicamentos se prescriben más?"
❓ "Pacientes que toman metformina"
❓ "Medicación habitual de pacientes"
```

**SQL Generado:**
```sql
SELECT PAUM_OBSERVATIONS, COUNT(*) as frecuencia
FROM PATI_USUAL_MEDICATION 
WHERE PAUM_OBSERVATIONS IS NOT NULL
GROUP BY PAUM_OBSERVATIONS
ORDER BY frecuencia DESC
```

## 🔧 Funciones Técnicas Principales

### 1. **`process_query(query: str)`**
**Propósito:** Procesamiento principal de consultas SQL  
**Entrada:** Consulta en lenguaje natural  
**Salida:** Resultados estructurados con interpretación clínica  

```python
async def process_query(self, query: str, stream_callback=None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    🧠 Procesamiento genérico de consultas SQL usando LLM para mapeo automático
    """
    # 1. Análisis semántico con LLM
    # 2. Generación de SQL optimizado
    # 3. Validación y corrección
    # 4. Ejecución y interpretación
```

### 2. **`_generate_last_patient_sql_simple(query: str)`**
**Propósito:** Generación específica para consultas de último paciente  
**Características:** Doble llamada al LLM para detección y generación  

```python
async def _generate_last_patient_sql_simple(self, query: str) -> str:
    """
    Genera SQL específico para último paciente con doble llamada al LLM
    """
    # PRIMERA LLAMADA: Detectar si es consulta de último paciente
    # SEGUNDA LLAMADA: Generar SQL optimizado
    # Validación: ORDER BY PATI_ID DESC LIMIT 1
```

### 3. **`_execute_sql_with_llm_validation(query: str, sql: str)`**
**Propósito:** Ejecución robusta de SQL con validación LLM  
**Características:** Manejo de errores, corrección automática, interpretación clínica  

```python
async def _execute_sql_with_llm_validation(self, query: str, sql: str, start_time: float, sql_params: Optional[List[Any]] = None, stream_callback=None) -> Dict[str, Any]:
    """
    Ejecuta SQL con validación LLM y manejo robusto de errores
    """
    # 1. Limpieza y optimización de SQL
    # 2. Validación de compatibilidad SQLite
    # 3. Ejecución con manejo de errores
    # 4. Interpretación clínica de resultados
```

## 🗃️ Estructura de Base de Datos

### Tablas Principales:

#### **PATI_PATIENTS** (Pacientes)
```sql
- PATI_ID (PRIMARY KEY)
- PATI_NAME (Nombre)
- PATI_SURNAME_1 (Primer apellido)
- PATI_FULL_NAME (Nombre completo)
- PATI_BIRTH_DATE (Fecha de nacimiento)
- PATI_START_DATE (Fecha de inicio de atención)
- PATI_ACTIVE (Estado activo)
```

#### **EPIS_DIAGNOSTICS** (Diagnósticos)
```sql
- DIAG_ID (PRIMARY KEY)
- PATI_ID (FOREIGN KEY)
- DIAG_OBSERVATION (Observación diagnóstica)
- DIAG_DESCRIPTION (Descripción del diagnóstico)
```

#### **PATI_USUAL_MEDICATION** (Medicación)
```sql
- PAUM_ID (PRIMARY KEY)
- PATI_ID (FOREIGN KEY)
- PAUM_OBSERVATIONS (Observaciones de medicación)
```

#### **PROC_PROCEDURES** (Procedimientos/Laboratorio)
```sql
- PROC_ID (PRIMARY KEY)
- PATI_ID (FOREIGN KEY)
- PROC_DESCRIPTION (Descripción del procedimiento)
```

## 🔍 Algoritmos de Detección

### 1. **Detección de Consultas de Último Paciente**
```python
# Algoritmo de detección usando LLM
def detect_last_patient_query(query: str) -> bool:
    """
    Detecta si una consulta se refiere al último paciente
    """
    keywords = ['último', 'ultimo', 'última', 'ultima', 'reciente', 'creado', 'registrado']
    phrases = ['último paciente', 'ultimo paciente', 'último paciente creado']
    questions = ['¿Cuál es el último paciente?', '¿Quién es el último paciente?']
    
    # Análisis semántico con LLM
    # Validación de contexto médico
    # Confirmación de intención
```

### 2. **Generación de SQL Optimizado**
```python
# Algoritmo de generación con validación
def generate_optimized_sql(query: str, analysis: Dict) -> str:
    """
    Genera SQL optimizado basado en análisis semántico
    """
    # 1. Análisis de entidades médicas
    # 2. Mapeo a tablas relevantes
    # 3. Generación de JOINs apropiados
    # 4. Optimización para SQLite
    # 5. Validación de esquema
```

## 📈 Métricas de Rendimiento

### Indicadores Clave:
- **Tiempo de respuesta:** < 5 segundos para consultas simples
- **Precisión de SQL:** > 95% de consultas válidas
- **Tasa de detección:** > 90% para consultas de último paciente
- **Tasa de corrección:** > 85% de errores corregidos automáticamente

### Logs de Rendimiento:
```python
logger.info(f"✅ Consulta completada: {len(results)} resultados en {execution_time:.2f}s")
logger.info(f"🧠 SQL validado con esquema real")
logger.info(f"✅ SQL limpio y listo: {sql}")
```

## 🛠️ Configuración y Uso

### Inicialización:
```python
sql_agent = SQLAgentIntelligentEnhanced(
    db_path="database_new.sqlite3.db",
    llm=llm_instance,
    medgemma_agent=medgemma_agent
)
```

### Ejemplo de Uso:
```python
# Consulta simple
result = await sql_agent.process_query("¿Cuál es el último paciente creado?")

# Consulta compleja
result = await sql_agent.process_query(
    "Muéstrame todos los datos de María del Carmen incluyendo diagnósticos, medicación y laboratorio"
)
```

## 🔧 Troubleshooting

### Problemas Comunes:

#### 1. **SQL Generado Incorrecto**
**Síntoma:** `SELECT * FROM PATIENTS ORDER BY created_at DESC LIMIT 1`  
**Solución:** Verificar que se use `PATI_PATIENTS` y `PATI_ID DESC`

#### 2. **Error de Esquema**
**Síntoma:** `no such table: PATIENTS`  
**Solución:** Usar nombres de tablas correctos: `PATI_PATIENTS`

#### 3. **Detección Fallida**
**Síntoma:** No detecta consultas de último paciente  
**Solución:** Verificar keywords y prompts de detección

## 📚 Referencias Técnicas

### Archivos Principales:
- `agents/sql_agent_flexible_enhanced.py` - Implementación principal
- `utils/sql_cleaner.py` - Limpieza de SQL
- `utils/sql_executor.py` - Ejecución de consultas
- `utils/sql_generator.py` - Generación de SQL

### Dependencias:
- `sqlite3` - Base de datos
- `langchain_openai` - LLM para generación
- `asyncio` - Procesamiento asíncrono
- `logging` - Sistema de logs

---

**Versión:** 1.0  
**Última actualización:** 2025-07-18  
**Mantenido por:** Equipo de Desarrollo ChatMed 