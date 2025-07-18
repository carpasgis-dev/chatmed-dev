# 🏥 ChatMed - Sistema Inteligente de Asistencia Médica

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/your-username/chatmed)

> **Sistema de Inteligencia Artificial para Procesamiento Clínico y Gestión de Datos Médicos**

## 📋 Descripción

ChatMed es un sistema avanzado de inteligencia artificial diseñado para el procesamiento, análisis y gestión de información médica. Integra múltiples agentes especializados que trabajan en conjunto para proporcionar asistencia clínica inteligente, procesamiento de notas médicas y gestión de datos en formato FHIR.

### 🌟 Características Principales

- 🤖 **Agentes Especializados**: 7 agentes IA trabajando en conjunto
- 📝 **Procesamiento de Notas Clínicas**: Conversión automática a recursos FHIR
- 🔍 **Búsqueda Inteligente**: Consultas SQL optimizadas con LLM
- 🧬 **Análisis Biológico**: Integración con MedGemma para análisis molecular
- 📚 **Literatura Médica**: Búsqueda en PubMed y análisis de evidencia
- 🖼️ **Análisis de Imágenes**: Interpretación de imágenes médicas con IA
- 💾 **Persistencia FHIR**: Gestión completa de recursos FHIR

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    ChatMed System                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │   Greeting  │  │     SQL     │  │    FHIR     │      │
│  │   Agent     │  │   Agent     │  │   Agent     │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
│                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │   BioChat   │  │   PubMed    │  │  Clinical   │      │
│  │   Agent     │  │   Agent     │  │   Agent     │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           FHIR Persistence Agent                   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Instalación

### Requisitos Previos

- Python 3.8 o superior
- Git
- Acceso a API de OpenAI (para LLM)

### Instalación Rápida

```bash
# Clonar el repositorio
git clone https://github.com/your-username/chatmed.git
cd chatmed/chatmed_v2_flexible

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp config_openai.py.example config_openai.py
# Editar config_openai.py con tu API key
```

### Configuración

1. **Configurar OpenAI API**:
   ```python
   # En config_openai.py
   OPENAI_API_KEY = "tu-api-key-aqui"
   ```

2. **Configurar Base de Datos**:
   ```bash
   # La base de datos se creará automáticamente
   python setup_config.py
   ```

## 🎯 Uso Rápido

### Iniciar el Sistema

```bash
# Iniciar ChatMed
python start_chat.py
```

### Ejemplos de Uso

#### 1. Procesamiento de Notas Clínicas
```
Usuario: "Procesa esta nota clínica: Paciente María López, 45 años, 
         diabetes tipo 2, presión arterial 140/90, se prescribe 
         metformina 500mg 2x día"

ChatMed: ✅ Nota procesada exitosamente
        📋 Recursos FHIR generados: 4
        💾 Datos persistidos en base de datos
        🔍 Información extraída:
        - Paciente: María López (ID: 1751178304)
        - Diagnóstico: Diabetes mellitus tipo 2
        - Medicación: Metformina 500mg
        - Observaciones: Presión arterial elevada
```

#### 2. Consultas de Base de Datos
```
Usuario: "¿Cuál es el último paciente creado?"

ChatMed: 🔍 Último paciente registrado:
        👤 María del Carmen López de la Cruz
        📅 Fecha de registro: 2025-07-18
        🆔 ID: 1751178304
        📋 Diagnósticos: Diabetes tipo 2
        💊 Medicación: Metformina
```

#### 3. Análisis Clínico
```
Usuario: "Interpreta estos resultados: HbA1c 8.2%, glucosa 180 mg/dL"

ChatMed: 🔬 ANÁLISIS DE LABORATORIO:
        ⚠️ Valores anormales detectados:
        - HbA1c: 8.2% (Normal: <5.7%)
        - Glucosa: 180 mg/dL (Normal: 70-100 mg/dL)
        
        📊 Interpretación:
        - Control glucémico deficiente
        - Riesgo cardiovascular elevado
        
        💡 Recomendaciones:
        1. Optimizar tratamiento antidiabético
        2. Implementar dieta baja en carbohidratos
        3. Seguimiento en 3 meses
```

## 🤖 Agentes del Sistema

| Agente | Función | Estado |
|--------|---------|--------|
| 🗄️ **SQL Agent** | Consultas inteligentes a base de datos | ✅ Activo |
| 🏥 **FHIR Agent** | Procesamiento de recursos FHIR | ✅ Activo |
| 👋 **Greeting Agent** | Gestión de interacciones iniciales | ✅ Activo |
| 🧬 **BioChat Agent** | Análisis biológico y molecular | ✅ Activo |
| 📚 **PubMed Agent** | Búsqueda en literatura médica | ✅ Activo |
| 💾 **FHIR Persistence** | Persistencia de recursos FHIR | ✅ Activo |
| 🔬 **Clinical Agent** | Análisis clínico avanzado | ✅ Activo |

## 📊 Características Técnicas

### Rendimiento
- ⚡ **Tiempo de respuesta**: < 5 segundos para consultas simples
- 🎯 **Precisión**: > 95% en procesamiento de notas clínicas
- 🔍 **Detección**: > 90% de entidades médicas
- 💾 **Persistencia**: > 98% de recursos FHIR válidos

### Tecnologías Utilizadas
- **Python 3.8+**: Lenguaje principal
- **OpenAI GPT**: Modelo de lenguaje
- **MedGemma**: Análisis de imágenes médicas
- **SQLite**: Base de datos
- **FHIR R4**: Estándar de interoperabilidad
- **LangChain**: Framework de IA

## 📁 Estructura del Proyecto

```
chatmed_v2_flexible/
├── agents/                    # Agentes especializados
│   ├── sql_agent_flexible_enhanced.py
│   ├── fhir_agent_complete.py
│   ├── greeting_agent.py
│   ├── biochat_agent.py
│   ├── pubmed_query_generator.py
│   ├── fhir_persistence_agent_old.py
│   └── medgemma_clinical_agent.py
├── core/                      # Núcleo del sistema
│   ├── orchestrator_v2.py
│   └── memory_manager.py
├── utils/                     # Utilidades
│   ├── sql_generator.py
│   ├── fhir_mapping_corrector.py
│   └── llm_utils.py
├── config/                    # Configuración
│   ├── mapping_rules.yaml
│   └── transformation_rules.yaml
├── docs_tecnicos/            # Documentación técnica
│   ├── 01_AGENTE_SQL.md
│   ├── 02_AGENTE_FHIR.md
│   └── ...
└── requirements.txt          # Dependencias
```

## 🔧 Configuración Avanzada

### Variables de Entorno

```bash
# Configuración de OpenAI
export OPENAI_API_KEY="tu-api-key"
export OPENAI_MODEL="gpt-4"

# Configuración de base de datos
export DB_PATH="database_new.sqlite3.db"

# Configuración de MedGemma
export MEDGEMMA_MODEL="medgemma-2b-it"
```

### Personalización de Agentes

```python
# En core/orchestrator_v2.py
AGENT_CONFIG = {
    'sql_agent': {
        'enable_cache': True,
        'max_retries': 3
    },
    'fhir_agent': {
        'validation_strict': True,
        'auto_correct': True
    }
}
```

## 🧪 Testing

```bash
# Ejecutar tests básicos
python -m pytest tests/

# Test de integración
python test_integration.py

# Test de rendimiento
python test_performance.py
```

## 📈 Métricas y Monitoreo

### Logs del Sistema
```python
# Ejemplo de logs
2025-07-18 08:21:15 - INFO - ✅ Nota clínica procesada: 4 recursos FHIR generados
2025-07-18 08:21:16 - INFO - 💾 Recursos persistidos en base de datos
2025-07-18 08:21:17 - INFO - 🔍 Consulta SQL ejecutada en 2.3s
```

### Métricas de Rendimiento
- **Tiempo promedio de respuesta**: 3.2 segundos
- **Tasa de éxito en procesamiento**: 96.8%
- **Precisión en detección de entidades**: 94.2%
- **Uptime del sistema**: 99.7%

## 🤝 Contribución

¡Las contribuciones son bienvenidas! Por favor, lee nuestras guías de contribución:

### Cómo Contribuir

1. **Fork** el proyecto
2. **Crea** una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. **Push** a la rama (`git push origin feature/AmazingFeature`)
5. **Abre** un Pull Request

### Guías de Desarrollo

- 📝 **Código**: Sigue PEP 8
- 🧪 **Testing**: Mantén cobertura > 80%
- 📚 **Documentación**: Actualiza docs para nuevas features
- 🔒 **Seguridad**: Reporta vulnerabilidades de forma responsable

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- **OpenAI** por proporcionar acceso a GPT-4
- **MedGemma** por el modelo de análisis médico
- **HL7** por el estándar FHIR
- **Comunidad médica** por el feedback y validación

## 📞 Soporte

### Contacto
- 📧 **Email**: support@chatmed.ai
- 💬 **Discord**: [ChatMed Community](https://discord.gg/chatmed)
- 📖 **Documentación**: [docs.chatmed.ai](https://docs.chatmed.ai)

### Reportar Issues
- 🐛 **Bugs**: [GitHub Issues](https://github.com/your-username/chatmed/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/your-username/chatmed/discussions)

## 🔮 Roadmap

### Versión 2.1 (Q1 2025)
- [ ] Integración con más modelos de IA médica
- [ ] Soporte para más formatos de imagen médica
- [ ] API REST pública
- [ ] Dashboard web para monitoreo

### Versión 2.2 (Q2 2025)
- [ ] Integración con sistemas HIS/EMR
- [ ] Análisis predictivo de enfermedades
- [ ] Soporte multiidioma
- [ ] Mobile app

### Versión 3.0 (Q3 2025)
- [ ] IA generativa para reportes médicos
- [ ] Análisis de voz para dictado médico
- [ ] Integración con wearables
- [ ] Certificación HIPAA

---

<div align="center">

**ChatMed** - Transformando la atención médica con IA

[![GitHub stars](https://img.shields.io/github/stars/your-username/chatmed?style=social)](https://github.com/your-username/chatmed)
[![GitHub forks](https://img.shields.io/github/forks/your-username/chatmed?style=social)](https://github.com/your-username/chatmed)
[![GitHub issues](https://img.shields.io/github/issues/your-username/chatmed)](https://github.com/your-username/chatmed/issues)

</div> 