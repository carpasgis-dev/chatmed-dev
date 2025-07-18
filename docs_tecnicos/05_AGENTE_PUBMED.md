# 📚 Agente PubMed - Documentación Técnica

## 📋 Información General

**Nombre del Agente:** `PubMedQueryGenerator`  
**Clase Principal:** `agents.pubmed_query_generator.PubMedQueryGenerator`  
**Versión:** v2.0  
**Tipo:** Agente de búsqueda y análisis de literatura científica biomédica  

## 🎯 Propósito y Funcionalidad

El Agente PubMed es el componente especializado en la búsqueda, recuperación y análisis de literatura científica biomédica desde la base de datos PubMed. Proporciona acceso inteligente a la investigación médica más reciente y relevante.

### Funciones Principales:
- **Búsqueda inteligente** en PubMed
- **Análisis de literatura** científica
- **Síntesis de evidencia** médica
- **Filtrado por relevancia** y calidad
- **Generación de resúmenes** de investigación
- **Análisis de tendencias** en investigación

## 🏗️ Arquitectura Técnica

### Componentes Principales:

#### 1. **Sistema de Búsqueda Inteligente**
```python
# Generación de queries PubMed optimizadas
pubmed_query_prompt = f"""Genera una consulta PubMed optimizada para la siguiente pregunta:

PREGUNTA: "{user_query}"

INSTRUCCIONES:
1. Identifica términos médicos clave
2. Usa operadores booleanos apropiados (AND, OR, NOT)
3. Incluye filtros de fecha si es relevante
4. Especifica tipos de estudio si es necesario
5. Optimiza para relevancia clínica

QUERY PUBMED:"""
```

#### 2. **Sistema de Análisis de Literatura**
```python
# Análisis de resultados de PubMed
literature_analysis_prompt = f"""Analiza los siguientes artículos de PubMed:

ARTÍCULOS:
{pubmed_results}

INSTRUCCIONES:
1. Identifica hallazgos principales
2. Evalúa calidad de evidencia
3. Sintetiza conclusiones clave
4. Identifica limitaciones
5. Proporciona recomendaciones clínicas

ANÁLISIS:"""
```

#### 3. **Sistema de Filtrado y Ranking**
- **Filtrado por relevancia** clínica
- **Ranking por impacto** científico
- **Filtrado por fecha** de publicación
- **Evaluación de calidad** metodológica

## 📊 Preguntas Clínicamente Relevantes

### 1. **Búsquedas de Evidencia Clínica**
```
❓ "Busca evidencia sobre el tratamiento de diabetes tipo 2"
❓ "¿Cuáles son los últimos estudios sobre COVID-19?"
❓ "Encuentra artículos sobre cáncer de mama en 2024"
❓ "Busca metaanálisis sobre hipertensión arterial"
```

**Query PubMed Generada:**
```
("diabetes mellitus type 2"[MeSH Terms] OR "diabetes type 2"[Title/Abstract]) 
AND ("therapy"[MeSH Terms] OR "treatment"[Title/Abstract]) 
AND ("2020"[Date - Publication] : "3000"[Date - Publication])
```

**Resultados Analizados:**
```
📚 EVIDENCIA CLÍNICA ENCONTRADA:

🔬 **Estudios Principales:**
1. "Efficacy of SGLT2 inhibitors in T2DM" (2024)
   - N=2,500 pacientes
   - Reducción HbA1c: 0.8%
   - Nivel de evidencia: A

2. "Metformin vs. newer agents" (2023)
   - Metaanálisis de 15 estudios
   - Metformina sigue siendo primera línea
   - Nivel de evidencia: A

📊 **Conclusiones Clínicas:**
- Metformina: Tratamiento de primera línea
- SGLT2 inhibitors: Beneficios cardiovasculares
- GLP-1 agonists: Control de peso adicional

⚠️ **Limitaciones:**
- Estudios principalmente en población caucásica
- Seguimiento limitado a 2 años
```

### 2. **Búsquedas de Fármacos y Tratamientos**
```
❓ "Busca información sobre metformina y efectos secundarios"
❓ "¿Cuáles son las contraindicaciones de la warfarina?"
❓ "Encuentra estudios sobre inmunoterapia en cáncer"
❓ "Busca evidencia sobre antibióticos en neumonía"
```

**Análisis Generado:**
```
💊 ANÁLISIS DE FÁRMACO: METFORMINA

📋 **Información Básica:**
- Clase: Biguanida
- Indicaciones: Diabetes tipo 2
- Mecanismo: Inhibición gluconeogénesis

🔍 **Efectos Secundarios (Evidencia 2024):**
- Gastrointestinales: 30-50% de pacientes
- Acidosis láctica: <0.01% (raro)
- Deficiencia B12: 10-30% a largo plazo

📊 **Estudios Recientes:**
1. "Metformin safety profile" (2024)
   - N=15,000 pacientes
   - Seguridad cardiovascular confirmada
   - Beneficios adicionales: anti-aging

2. "Long-term metformin use" (2023)
   - Seguimiento 10 años
   - Reducción mortalidad: 15%
   - Prevención cáncer: 20%
```

### 3. **Búsquedas de Diagnósticos y Síntomas**
```
❓ "¿Cuáles son los síntomas del COVID-19?"
❓ "Busca información sobre diagnóstico de cáncer de pulmón"
❓ "Encuentra artículos sobre depresión y tratamiento"
❓ "¿Cuáles son los marcadores de infarto agudo?"
```

**Resultados Generados:**
```
🔍 DIAGNÓSTICO: COVID-19

📋 **Síntomas Principales (Evidencia 2024):**
- Fiebre: 80-90% de casos
- Tos seca: 60-70%
- Fatiga: 50-60%
- Pérdida olfato/gusto: 40-50%

⚠️ **Síntomas de Alarma:**
- Dificultad respiratoria
- Dolor torácico
- Confusión
- Cianosis

📊 **Estudios Epidemiológicos:**
1. "COVID-19 symptoms evolution" (2024)
   - Variantes actuales: Síntomas más leves
   - Duración promedio: 5-7 días
   - Secuelas: 10-20% de casos

2. "Long COVID prevalence" (2023)
   - Síntomas persistentes: 15-30%
   - Factores de riesgo identificados
```

### 4. **Búsquedas de Investigación Avanzada**
```
❓ "Busca artículos sobre CRISPR en terapia génica"
❓ "Encuentra estudios sobre inteligencia artificial en medicina"
❓ "¿Cuáles son los avances en inmunoterapia?"
❓ "Busca investigación sobre medicina personalizada"
```

**Análisis Generado:**
```
🧬 INVESTIGACIÓN AVANZADA: CRISPR

📚 **Estado Actual (2024):**
- Aplicaciones clínicas: 15+ ensayos activos
- Enfermedades objetivo: Anemia falciforme, cáncer
- Eficacia: 70-90% en modelos preclínicos

🔬 **Estudios Destacados:**
1. "CRISPR-Cas9 in sickle cell disease" (2024)
   - N=45 pacientes
   - Eficacia: 85% de curación
   - Seguridad: Perfil favorable

2. "Gene editing safety" (2023)
   - Revisión de 50+ estudios
   - Riesgo off-target: <1%
   - Regulación: FDA aprobación pendiente

⚠️ **Consideraciones Éticas:**
- Edición germinal: Controversial
- Acceso equitativo: Desafío global
- Regulación: Necesaria
```

## 🔧 Funciones Técnicas Principales

### 1. **`process_query(query: str)`**
**Propósito:** Procesamiento principal de consultas PubMed  
**Entrada:** Consulta sobre literatura médica  
**Salida:** Análisis estructurado de evidencia científica  

```python
async def process_query(self, query: str, stream_callback=None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    📚 Procesamiento de consultas de literatura médica
    """
    # 1. Generación de query PubMed optimizada
    # 2. Búsqueda en base de datos PubMed
    # 3. Filtrado y ranking de resultados
    # 4. Análisis de literatura encontrada
    # 5. Síntesis de evidencia clínica
```

### 2. **`generate_pubmed_query(user_query: str)`**
**Propósito:** Generación de queries PubMed optimizadas  
**Características:** Uso de LLM para optimización  

```python
async def generate_pubmed_query(self, user_query: str) -> str:
    """
    Genera query PubMed optimizada usando LLM
    """
    # 1. Análisis semántico de la consulta
    # 2. Identificación de términos médicos
    # 3. Construcción de query con operadores booleanos
    # 4. Optimización para relevancia clínica
    # 5. Validación de sintaxis PubMed
```

### 3. **`analyze_literature_results(results: List[Dict])`**
**Propósito:** Análisis de resultados de PubMed  
**Características:** Síntesis de evidencia científica  

```python
async def analyze_literature_results(self, results: List[Dict]) -> Dict[str, Any]:
    """
    Analiza resultados de PubMed y sintetiza evidencia
    """
    # 1. Evaluación de calidad metodológica
    # 2. Extracción de hallazgos principales
    # 3. Síntesis de conclusiones
    # 4. Identificación de limitaciones
    # 5. Generación de recomendaciones clínicas
```

## 🗃️ Estructura de Datos PubMed

### Campos de Artículo:
```python
pubmed_article = {
    "pmid": "12345678",
    "title": "Efficacy of new treatment for diabetes",
    "abstract": "This study evaluated...",
    "authors": ["Smith J", "Johnson A"],
    "journal": "New England Journal of Medicine",
    "publication_date": "2024-01-15",
    "mesh_terms": ["Diabetes Mellitus", "Therapy"],
    "study_type": "Randomized Controlled Trial",
    "sample_size": 2500,
    "impact_factor": 74.699
}
```

### Tipos de Estudios:
```python
study_types = {
    "RCT": "Randomized Controlled Trial",
    "META": "Meta-analysis",
    "REVIEW": "Systematic Review",
    "OBSERVATIONAL": "Observational Study",
    "CASE_CONTROL": "Case-Control Study",
    "COHORT": "Cohort Study"
}
```

### Niveles de Evidencia:
```python
evidence_levels = {
    "A": "Meta-análisis de RCTs",
    "B": "RCT individual",
    "C": "Estudios observacionales",
    "D": "Opinión de expertos",
    "E": "Evidencia insuficiente"
}
```

## 🔍 Algoritmos de Búsqueda

### 1. **Generación de Queries**
```python
# Algoritmo de generación de queries
def generate_optimized_query(user_query: str) -> str:
    """
    Genera query PubMed optimizada
    """
    # 1. Análisis de términos médicos
    # 2. Mapeo a MeSH terms
    # 3. Construcción de query booleana
    # 4. Optimización para relevancia
    # 5. Validación de sintaxis
```

### 2. **Filtrado de Resultados**
```python
# Algoritmo de filtrado
def filter_results(articles: List[Dict]) -> List[Dict]:
    """
    Filtra resultados por relevancia y calidad
    """
    # 1. Filtrado por fecha
    # 2. Filtrado por tipo de estudio
    # 3. Ranking por impacto
    # 4. Evaluación de calidad
    # 5. Eliminación de duplicados
```

### 3. **Análisis de Literatura**
```python
# Algoritmo de análisis
def analyze_literature(articles: List[Dict]) -> Dict[str, Any]:
    """
    Analiza literatura y sintetiza evidencia
    """
    # 1. Extracción de hallazgos
    # 2. Evaluación de calidad
    # 3. Síntesis de conclusiones
    # 4. Identificación de limitaciones
    # 5. Generación de recomendaciones
```

## 📈 Métricas de Rendimiento

### Indicadores Clave:
- **Tiempo de búsqueda:** < 10 segundos para consultas complejas
- **Precisión de query:** > 90% de queries válidas
- **Relevancia de resultados:** > 85% de artículos relevantes
- **Calidad de síntesis:** > 80% de análisis precisos

### Logs de Rendimiento:
```python
logger.info(f"📚 Búsqueda completada: {hits} resultados en {search_time:.2f}s")
logger.info(f"🔍 Query generada: {pubmed_query}")
logger.info(f"📊 Análisis completado: {summary_length} palabras")
```

## 🛠️ Configuración y Uso

### Inicialización:
```python
pubmed_agent = PubMedQueryGenerator(
    llm=llm_instance,
    api_key="your_pubmed_api_key",
    max_results=50,
    date_filter="2020:3000"
)
```

### Ejemplo de Uso:
```python
# Búsqueda de evidencia clínica
result = await pubmed_agent.process_query(
    "Busca evidencia sobre el tratamiento de diabetes tipo 2"
)

# Análisis de fármaco
drug_analysis = await pubmed_agent.process_query(
    "Busca información sobre metformina y efectos secundarios"
)

# Búsqueda de investigación avanzada
research_result = await pubmed_agent.process_query(
    "Busca artículos sobre CRISPR en terapia génica"
)
```

## 🔧 Troubleshooting

### Problemas Comunes:

#### 1. **Error de API PubMed**
**Síntoma:** `PubMed API error: Rate limit exceeded`  
**Solución:** Implementar rate limiting y reintentos

#### 2. **Query Inválida**
**Síntoma:** `Invalid PubMed query syntax`  
**Solución:** Validar sintaxis antes de ejecutar

#### 3. **Sin Resultados**
**Síntoma:** `No results found for query`  
**Solución:** Ampliar términos de búsqueda o fechas

## 📚 Referencias Técnicas

### Archivos Principales:
- `agents/pubmed_query_generator.py` - Implementación principal
- `config/ncbi_config.py` - Configuración de API
- `utils/` - Utilidades de búsqueda y análisis

### Dependencias:
- `biopython` - Acceso a PubMed
- `requests` - Llamadas HTTP
- `langchain_openai` - LLM para generación
- `xml` - Parsing de respuestas PubMed

### APIs Integradas:
- **PubMed API** - Búsqueda de literatura
- **NCBI E-utilities** - Acceso a bases de datos
- **MeSH Database** - Términos médicos controlados

---

**Versión:** 1.0  
**Última actualización:** 2025-07-18  
**Mantenido por:** Equipo de Desarrollo ChatMed 