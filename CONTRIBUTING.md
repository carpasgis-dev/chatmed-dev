# 🤝 Guía de Contribución - ChatMed

¡Gracias por tu interés en contribuir a ChatMed! Este documento proporciona las pautas para contribuir al proyecto.

## 📋 Tabla de Contenidos

- [Cómo Contribuir](#cómo-contribuir)
- [Configuración del Entorno](#configuración-del-entorno)
- [Estándares de Código](#estándares-de-código)
- [Proceso de Pull Request](#proceso-de-pull-request)
- [Reportar Bugs](#reportar-bugs)
- [Solicitar Features](#solicitar-features)
- [Preguntas Frecuentes](#preguntas-frecuentes)

## 🚀 Cómo Contribuir

### Tipos de Contribuciones

- 🐛 **Reportar Bugs**: Ayuda a mejorar la estabilidad
- 💡 **Solicitar Features**: Sugiere nuevas funcionalidades
- 📝 **Mejorar Documentación**: Ayuda a otros desarrolladores
- 🔧 **Arreglar Bugs**: Resuelve problemas existentes
- ✨ **Implementar Features**: Agrega nuevas funcionalidades
- 🧪 **Escribir Tests**: Mejora la calidad del código

### Antes de Contribuir

1. **Revisa los Issues existentes** para evitar duplicados
2. **Lee la documentación** del proyecto
3. **Prueba el código** en tu entorno local
4. **Sigue las convenciones** del proyecto

## ⚙️ Configuración del Entorno

### Requisitos

- Python 3.8+
- Git
- Acceso a API de OpenAI

### Configuración Local

```bash
# 1. Fork el repositorio
# 2. Clona tu fork
git clone https://github.com/tu-usuario/chatmed.git
cd chatmed/chatmed_v2_flexible

# 3. Crea un entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 4. Instala dependencias
pip install -r requirements.txt

# 5. Configura las variables de entorno
cp config_openai.py.example config_openai.py
# Edita config_openai.py con tu API key

# 6. Ejecuta tests
python -m pytest tests/
```

## 📝 Estándares de Código

### Convenciones de Python

- **PEP 8**: Sigue las convenciones de estilo
- **Docstrings**: Documenta todas las funciones
- **Type Hints**: Usa anotaciones de tipo
- **Nombres descriptivos**: Variables y funciones claras

### Ejemplo de Código

```python
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

def process_medical_data(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Procesa datos médicos y retorna resultados estructurados.
    
    Args:
        data: Diccionario con datos médicos
        
    Returns:
        Diccionario con resultados procesados o None si hay error
        
    Raises:
        ValueError: Si los datos son inválidos
    """
    try:
        # Tu código aquí
        return processed_data
    except Exception as e:
        logger.error(f"Error procesando datos: {e}")
        return None
```

### Estructura de Archivos

```
agents/
├── __init__.py
├── sql_agent_flexible_enhanced.py
├── fhir_agent_complete.py
└── ...

core/
├── __init__.py
├── orchestrator_v2.py
└── ...

utils/
├── __init__.py
├── sql_generator.py
└── ...

tests/
├── test_sql_agent.py
├── test_fhir_agent.py
└── ...
```

## 🔄 Proceso de Pull Request

### 1. Preparación

```bash
# Crea una nueva rama
git checkout -b feature/tu-nueva-feature

# Haz tus cambios
# ...

# Agrega los archivos
git add .

# Commit con mensaje descriptivo
git commit -m "feat: agrega nueva funcionalidad de análisis clínico

- Implementa análisis de laboratorio avanzado
- Agrega validación de resultados
- Mejora precisión de diagnósticos

Closes #123"
```

### 2. Mensajes de Commit

Usa el formato **Conventional Commits**:

- `feat:` Nueva funcionalidad
- `fix:` Corrección de bug
- `docs:` Documentación
- `style:` Formato de código
- `refactor:` Refactorización
- `test:` Tests
- `chore:` Tareas de mantenimiento

### 3. Pull Request

1. **Título descriptivo**: "feat: Agrega análisis de imágenes médicas"
2. **Descripción detallada**: Explica qué hace y por qué
3. **Referencia issues**: "Closes #123" o "Fixes #456"
4. **Screenshots**: Si aplica para cambios de UI
5. **Tests**: Incluye tests para nuevas funcionalidades

### 4. Template de Pull Request

```markdown
## 📋 Descripción
Breve descripción de los cambios

## 🔧 Cambios Realizados
- [ ] Nueva funcionalidad agregada
- [ ] Bug corregido
- [ ] Documentación actualizada
- [ ] Tests agregados

## 🧪 Tests
- [ ] Tests unitarios pasan
- [ ] Tests de integración pasan
- [ ] Cobertura de código > 80%

## 📸 Screenshots
Si aplica, incluye screenshots

## ✅ Checklist
- [ ] Código sigue PEP 8
- [ ] Documentación actualizada
- [ ] Tests agregados/actualizados
- [ ] No hay conflictos de merge
```

## 🐛 Reportar Bugs

### Template de Bug Report

```markdown
## 🐛 Descripción del Bug
Descripción clara y concisa del problema

## 🔄 Pasos para Reproducir
1. Ve a '...'
2. Haz clic en '...'
3. Desplázate hacia abajo hasta '...'
4. Ve el error

## ✅ Comportamiento Esperado
Descripción de lo que debería pasar

## 📱 Información del Sistema
- OS: [ej. Windows 10]
- Python: [ej. 3.8.5]
- ChatMed: [ej. v2.0.0]

## 📋 Información Adicional
Cualquier contexto adicional, logs, screenshots
```

## 💡 Solicitar Features

### Template de Feature Request

```markdown
## 💡 Descripción de la Feature
Descripción clara de la funcionalidad deseada

## 🎯 Caso de Uso
Explica cómo esta feature sería útil

## 🔧 Implementación Sugerida
Si tienes ideas sobre la implementación

## 📋 Alternativas Consideradas
Otras soluciones que consideraste

## 📱 Información Adicional
Cualquier contexto adicional
```

## 🧪 Tests

### Escribir Tests

```python
import pytest
from agents.sql_agent_flexible_enhanced import SQLAgentIntelligentEnhanced

class TestSQLAgent:
    """Tests para el agente SQL."""
    
    def test_process_query(self):
        """Test del procesamiento de consultas."""
        agent = SQLAgentIntelligentEnhanced()
        result = agent.process_query("¿Cuál es el último paciente?")
        assert result is not None
        assert "sql" in result
        
    def test_generate_last_patient_sql(self):
        """Test de generación de SQL para último paciente."""
        agent = SQLAgentIntelligentEnhanced()
        sql = agent._generate_last_patient_sql_simple("último paciente")
        assert "PATI_ID DESC" in sql
        assert "LIMIT 1" in sql
```

### Ejecutar Tests

```bash
# Todos los tests
python -m pytest

# Tests específicos
python -m pytest tests/test_sql_agent.py

# Con cobertura
python -m pytest --cov=agents tests/

# Tests de integración
python -m pytest tests/integration/
```

## 📚 Documentación

### Actualizar Documentación

1. **README.md**: Para cambios importantes
2. **docs_tecnicos/**: Para documentación técnica
3. **Docstrings**: Para funciones y clases
4. **Ejemplos**: Para casos de uso

### Estilo de Documentación

```python
def analyze_medical_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analiza datos médicos y proporciona insights clínicos.
    
    Esta función procesa información médica estructurada y genera
    análisis clínicos basados en patrones y evidencia científica.
    
    Args:
        data: Diccionario con datos médicos del paciente
            - lab_results: Resultados de laboratorio
            - symptoms: Síntomas reportados
            - history: Historial médico
            
    Returns:
        Diccionario con análisis clínico:
            - diagnosis: Diagnósticos sugeridos
            - risk_factors: Factores de riesgo
            - recommendations: Recomendaciones clínicas
            
    Raises:
        ValueError: Si los datos son inválidos
        ProcessingError: Si hay error en el análisis
        
    Example:
        >>> data = {"lab_results": {"glucose": 180}}
        >>> result = analyze_medical_data(data)
        >>> print(result["diagnosis"])
        ['Diabetes mellitus']
    """
```

## 🤝 Preguntas Frecuentes

### ¿Cómo empiezo a contribuir?

1. **Fork el repositorio**
2. **Configura tu entorno local**
3. **Encuentra un issue etiquetado como "good first issue"**
4. **Crea una rama y haz tus cambios**
5. **Envía un Pull Request**

### ¿Qué hago si encuentro un bug crítico?

1. **Crea un issue con etiqueta "bug" y "high priority"**
2. **Proporciona toda la información posible**
3. **Si puedes, propón una solución**

### ¿Cómo puedo solicitar una nueva feature?

1. **Busca si ya existe un issue similar**
2. **Crea un issue con etiqueta "enhancement"**
3. **Describe el caso de uso y beneficio**

### ¿Cuál es el proceso de review?

1. **Automated checks** deben pasar
2. **Code review** por maintainers
3. **Tests** deben pasar
4. **Documentación** debe estar actualizada

## 📞 Contacto

- 📧 **Email**: dev@chatmed.ai
- 💬 **Discord**: [ChatMed Community](https://discord.gg/chatmed)
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-username/chatmed/issues)
- 💡 **Discussions**: [GitHub Discussions](https://github.com/your-username/chatmed/discussions)

---

**¡Gracias por contribuir a ChatMed! 🏥**

Tu contribución ayuda a mejorar la atención médica con IA. 