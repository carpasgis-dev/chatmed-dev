# 👋 Agente de Saludos - Documentación Técnica

## 📋 Información General

**Nombre del Agente:** `IntelligentGreetingAgent`  
**Clase Principal:** `agents.greeting_agent.IntelligentGreetingAgent`  
**Versión:** v3.0  
**Tipo:** Agente de interacción inicial y gestión de contexto de usuario  

## 🎯 Propósito y Funcionalidad

El Agente de Saludos es el componente especializado en la gestión de la experiencia inicial del usuario con el sistema ChatMed. Se encarga de proporcionar una interfaz amigable, detectar la intención del usuario y establecer el contexto apropiado para la interacción médica.

### Funciones Principales:
- **Gestión de saludos** personalizados
- **Detección de intención** del usuario
- **Establecimiento de contexto** médico
- **Orientación** sobre capacidades del sistema
- **Gestión de sesiones** de usuario
- **Transición inteligente** a otros agentes

## 🏗️ Arquitectura Técnica

### Componentes Principales:

#### 1. **Sistema de Detección de Intención**
```python
# Detección de intención usando LLM
intent_detection_prompt = f"""Analiza la siguiente consulta del usuario y determina su intención principal:

CONSULTA: "{user_query}"

CATEGORÍAS DE INTENCIÓN:
1. SALUDO - Usuario saluda o inicia conversación
2. CONSULTA_MEDICA - Pregunta sobre salud, síntomas, diagnósticos
3. BUSQUEDA_PACIENTE - Busca información de pacientes específicos
4. PROCESAMIENTO_NOTA - Quiere procesar una nota clínica
5. CONSULTA_TECNICA - Pregunta sobre el funcionamiento del sistema
6. DESPEDIDA - Usuario se despide o termina conversación

Responde SOLO con la categoría más apropiada."""
```

#### 2. **Sistema de Respuestas Contextuales**
```python
# Generación de respuestas contextuales
greeting_prompt = f"""Genera una respuesta apropiada para el usuario basada en:

CONTEXTO: {context}
INTENCIÓN: {intent}
HISTORIAL: {conversation_history}

REGLAS:
- Mantén un tono profesional pero amigable
- Proporciona información útil sobre capacidades
- Orienta al usuario sobre cómo proceder
- Sé conciso pero informativo

RESPUESTA:"""
```

#### 3. **Sistema de Gestión de Contexto**
- **Detección de estado** de la conversación
- **Mantenimiento de contexto** médico
- **Transición inteligente** entre agentes
- **Gestión de sesiones** de usuario

## 📊 Preguntas Clínicamente Relevantes

### 1. **Saludos y Presentaciones**
```
❓ "Hola, ¿cómo estás?"
❓ "Buenos días, ¿qué puedes hacer?"
❓ "Hola, soy médico y necesito ayuda"
❓ "¿Puedes ayudarme con información médica?"
```

**Respuestas Generadas:**
```
👋 ¡Hola! Soy ChatMed, tu asistente médico inteligente. 

Puedo ayudarte con:
• 📋 Procesamiento de notas clínicas
• 🔍 Búsqueda de información de pacientes
• 💊 Análisis de medicaciones y diagnósticos
• 📊 Consultas a la base de datos médica
• 🏥 Conversión de datos a formato FHIR

¿En qué puedo ayudarte hoy?
```

### 2. **Orientación sobre Capacidades**
```
❓ "¿Qué puedes hacer?"
❓ "¿Cuáles son tus funciones?"
❓ "¿Puedes procesar notas clínicas?"
❓ "¿Cómo funciona el sistema?"
```

**Respuestas Generadas:**
```
🤖 Mis capacidades principales:

📝 **Procesamiento de Notas Clínicas**
- Extrae información estructurada de notas médicas
- Convierte a recursos FHIR estándar
- Almacena en base de datos SQL

🔍 **Búsqueda Inteligente**
- Consultas de pacientes por nombre
- Búsqueda de diagnósticos y medicaciones
- Análisis estadístico de datos médicos

💊 **Análisis Clínico**
- Interpretación de resultados de laboratorio
- Análisis de medicaciones prescritas
- Validación de información médica

¿Te gustaría probar alguna de estas funciones?
```

### 3. **Transiciones a Otros Agentes**
```
❓ "Quiero procesar una nota clínica"
❓ "Busca información de un paciente"
❓ "Necesito consultar la base de datos"
❓ "Analiza esta información médica"
```

**Respuestas Generadas:**
```
✅ Perfecto, te ayudo con eso.

Para procesar una nota clínica, simplemente compártela conmigo y la procesaré automáticamente, extrayendo la información del paciente, diagnósticos, medicaciones y otros datos relevantes.

Para buscar información de pacientes, puedes preguntarme por nombre, diagnóstico o cualquier criterio específico.

¿Qué información necesitas procesar o consultar?
```

### 4. **Gestión de Despedidas**
```
❓ "Gracias, hasta luego"
❓ "Adiós"
❓ "Terminamos por hoy"
❓ "Hasta la próxima"
```

**Respuestas Generadas:**
```
👋 ¡Ha sido un placer ayudarte! 

Recuerda que estoy aquí para asistirte con cualquier consulta médica o procesamiento de información clínica.

¡Que tengas un excelente día! 🏥
```

## 🔧 Funciones Técnicas Principales

### 1. **`process_query(query: str)`**
**Propósito:** Procesamiento principal de consultas de saludo  
**Entrada:** Consulta del usuario  
**Salida:** Respuesta contextual apropiada  

```python
async def process_query(self, query: str, stream_callback=None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    👋 Procesamiento de saludos y gestión de contexto inicial
    """
    # 1. Detección de intención del usuario
    # 2. Análisis de contexto de conversación
    # 3. Generación de respuesta apropiada
    # 4. Establecimiento de contexto para otros agentes
    # 5. Transición inteligente si es necesario
```

### 2. **`detect_intent(query: str)`**
**Propósito:** Detección de la intención del usuario  
**Características:** Análisis semántico con LLM  

```python
async def detect_intent(self, query: str) -> str:
    """
    Detecta la intención principal del usuario
    """
    # 1. Análisis semántico de la consulta
    # 2. Clasificación de intención
    # 3. Validación de contexto
    # 4. Retorno de categoría apropiada
```

### 3. **`generate_contextual_response(intent: str, context: Dict)`**
**Propósito:** Generación de respuestas contextuales  
**Características:** Respuestas personalizadas según contexto  

```python
async def generate_contextual_response(self, intent: str, context: Dict) -> str:
    """
    Genera respuesta contextual basada en intención y contexto
    """
    # 1. Análisis de intención detectada
    # 2. Evaluación de contexto actual
    # 3. Generación de respuesta apropiada
    # 4. Inclusión de información útil
    # 5. Orientación sobre próximos pasos
```

## 🗃️ Estructura de Contexto

### Variables de Contexto:

#### **Estado de Conversación**
```python
conversation_state = {
    "is_first_interaction": True,
    "user_intent": "SALUDO",
    "previous_queries": [],
    "current_context": "INITIAL",
    "session_start_time": "2025-07-18T08:21:00Z"
}
```

#### **Información de Usuario**
```python
user_info = {
    "user_type": "MEDICAL_PROFESSIONAL",
    "expertise_level": "EXPERT",
    "preferred_language": "es",
    "session_id": "session_1751178304"
}
```

#### **Capacidades del Sistema**
```python
system_capabilities = {
    "fhir_processing": True,
    "sql_queries": True,
    "clinical_analysis": True,
    "patient_search": True,
    "note_processing": True
}
```

## 🔍 Algoritmos de Detección

### 1. **Detección de Intención**
```python
# Algoritmo de detección de intención
def detect_user_intent(query: str) -> str:
    """
    Detecta la intención principal del usuario
    """
    # 1. Análisis de palabras clave
    # 2. Clasificación semántica con LLM
    # 3. Validación de contexto
    # 4. Confirmación de intención
```

### 2. **Generación de Respuestas Contextuales**
```python
# Algoritmo de generación contextual
def generate_contextual_response(intent: str, context: Dict) -> str:
    """
    Genera respuesta contextual apropiada
    """
    # 1. Selección de template base
    # 2. Personalización según contexto
    # 3. Inclusión de información relevante
    # 4. Orientación sobre próximos pasos
    # 5. Validación de coherencia
```

### 3. **Gestión de Transiciones**
```python
# Algoritmo de transición inteligente
def manage_agent_transition(intent: str, context: Dict) -> str:
    """
    Gestiona transición a otros agentes
    """
    # 1. Evaluación de intención
    # 2. Selección de agente apropiado
    # 3. Preparación de contexto
    # 4. Transición suave
    # 5. Confirmación de cambio
```

## 📈 Métricas de Rendimiento

### Indicadores Clave:
- **Tiempo de respuesta:** < 2 segundos para saludos
- **Precisión de detección:** > 95% de intenciones correctas
- **Tasa de satisfacción:** > 90% de usuarios satisfechos
- **Tasa de transición:** > 85% de transiciones exitosas

### Logs de Rendimiento:
```python
logger.info(f"👋 Saludo procesado: {intent} detectado en {response_time:.2f}s")
logger.info(f"✅ Contexto establecido para agente: {next_agent}")
logger.info(f"🔄 Transición exitosa a: {target_agent}")
```

## 🛠️ Configuración y Uso

### Inicialización:
```python
greeting_agent = IntelligentGreetingAgent(
    llm=llm_instance,
    system_context=system_capabilities,
    user_preferences=user_info
)
```

### Ejemplo de Uso:
```python
# Procesamiento de saludo inicial
result = await greeting_agent.process_query("Hola, ¿cómo estás?")

# Detección de intención
intent = await greeting_agent.detect_intent("Quiero procesar una nota clínica")

# Generación de respuesta contextual
response = await greeting_agent.generate_contextual_response(intent, context)
```

## 🔧 Troubleshooting

### Problemas Comunes:

#### 1. **Detección Incorrecta de Intención**
**Síntoma:** Clasifica consulta médica como saludo  
**Solución:** Mejorar prompts de detección y validación

#### 2. **Respuesta Inapropiada**
**Síntoma:** Respuesta no contextualizada  
**Solución:** Verificar contexto y personalización

#### 3. **Transición Fallida**
**Síntoma:** No redirige a agente apropiado  
**Solución:** Verificar lógica de transición y contexto

## 📚 Referencias Técnicas

### Archivos Principales:
- `agents/greeting_agent.py` - Implementación principal
- `core/orchestrator_v2.py` - Gestión de transiciones
- `utils/llm_utils.py` - Utilidades de LLM
- `config/` - Configuración de contexto

### Dependencias:
- `langchain_openai` - LLM para detección y generación
- `asyncio` - Procesamiento asíncrono
- `logging` - Sistema de logs
- `datetime` - Gestión de sesiones

### Integración:
- **Orquestador Principal** - Coordinación con otros agentes
- **Sistema de Memoria** - Persistencia de contexto
- **Gestión de Sesiones** - Control de estado de usuario

---

**Versión:** 1.0  
**Última actualización:** 2025-07-18  
**Mantenido por:** Equipo de Desarrollo ChatMed 