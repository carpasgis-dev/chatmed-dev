"""
Agente Inteligente de Detección de Nombres y Apellidos
=======================================================

Este agente especializado usa LLM para detectar y normalizar nombres de pacientes
de manera inteligente, manejando diferentes formatos y variaciones.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging

# Importar utilidades de parseo JSON
try:
    from ..utils.json_parser import robust_json_parse
except ImportError:
    # Fallback si no se puede importar
    def robust_json_parse(content: str, fallback=None):
        try:
            return json.loads(content.strip())
        except:
            return fallback

logger = logging.getLogger(__name__)

@dataclass
class PatientName:
    """Estructura normalizada para nombres de pacientes"""
    first_name: str
    last_name: str
    full_name: str
    original_query: str
    confidence: float
    variations: List[str]
    search_patterns: List[str]

class IntelligentNameDetector:
    """
    Agente especializado en detección inteligente de nombres de pacientes
    """
    
    def __init__(self, llm=None):
        self.llm = llm
        self.name_cache = {}
    
    async def detect_patient_name(self, query: str) -> PatientName:
        """
        Detecta el nombre del paciente en una consulta usando LLM inteligente
        """
        try:
            if not self.llm:
                return self._basic_name_detection(query)
            
            # Cache para evitar llamadas repetidas
            cache_key = query.lower().strip()
            if cache_key in self.name_cache:
                return self.name_cache[cache_key]
            
            prompt = f"""
Eres un experto en detección de nombres de pacientes en consultas médicas. 
Analiza esta consulta y extrae el nombre del paciente de manera inteligente.

CONSULTA: "{query}"

INSTRUCCIONES:
1. Identifica el nombre completo del paciente mencionado
2. Separa nombre y apellido de manera inteligente
3. Maneja variaciones: "Juan Carlos", "Juan Carlos Pascual", "Pascual, Juan Carlos"
4. Considera apellidos compuestos: "De la Rosa", "Van der Berg"
5. Normaliza el formato: primer nombre + apellidos
6. Genera patrones de búsqueda para SQL

EJEMPLOS:
- "¿qué datos tiene Juan Carlos Pascual?" → Juan Carlos (nombre), Pascual (apellido)
- "mostrar constantes vitales de María José García López" → María José (nombre), García López (apellido)
- "buscar paciente Ana" → Ana (nombre), "" (apellido)

Responde SOLO con este JSON:
{{
  "first_name": "nombre del paciente",
  "last_name": "apellido del paciente", 
  "full_name": "nombre completo",
  "confidence": 0.0-1.0,
  "variations": ["variación1", "variación2"],
  "search_patterns": [
    "patrón SQL para búsqueda exacta",
    "patrón SQL para búsqueda parcial"
  ],
  "reasoning": "explicación de la detección"
}}
"""
            
            response = await self.llm.ainvoke(prompt)
            content = str(response.content).strip()
            
            # Parsear respuesta usando utilidad robusta
            result = robust_json_parse(content, fallback=self._extract_name_with_regex(content, query))
            
            # Asegurar que result sea un diccionario
            if result is None:
                result = self._extract_name_with_regex(content, query)
            
            # Crear objeto PatientName
            patient_name = PatientName(
                first_name=result.get('first_name', ''),
                last_name=result.get('last_name', ''),
                full_name=result.get('full_name', ''),
                original_query=query,
                confidence=result.get('confidence', 0.5),
                variations=result.get('variations', []),
                search_patterns=result.get('search_patterns', [])
            )
            
            # Guardar en cache
            self.name_cache[cache_key] = patient_name
            
            logger.info(f"🧠 Nombre detectado: {patient_name.full_name} (confianza: {patient_name.confidence:.2f})")
            return patient_name
            
        except Exception as e:
            logger.error(f"❌ Error en detección de nombre: {e}")
            return self._basic_name_detection(query)
    
    def _extract_name_with_regex(self, content: str, original_query: str) -> Dict[str, Any]:
        """Extrae nombre usando regex como fallback"""
        try:
            # Limpiar contenido
            content = content.strip()
            
            # Remover markdown si está presente
            if content.startswith('```json'):
                content = content[7:]
            elif content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            
            # Buscar JSON en la respuesta con múltiples patrones
            json_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # JSON simple
                r'\{.*\}',  # JSON con cualquier contenido
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, content, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        continue
            
            # Si no se encuentra JSON válido, intentar parsear todo el contenido
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass
                
        except Exception as e:
            logger.warning(f"⚠️ Error en extracción regex: {e}")
        
        # Fallback básico
        return self._basic_name_detection(original_query).__dict__
    
    def _basic_name_detection(self, query: str) -> PatientName:
        """Detección básica sin LLM"""
        query_lower = query.lower()
        
        # Patrones comunes de nombres
        name_patterns = [
            r'(\w+)\s+(\w+)\s+(\w+)',  # Juan Carlos Pascual
            r'(\w+)\s+(\w+)',           # Juan Pascual
            r'(\w+)',                   # Juan
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, query_lower)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    first_name = groups[0].title()
                    last_name = groups[1].title()
                    if len(groups) >= 2:
                        full_name = f"{first_name} {last_name}"
                    elif len(groups) == 1:
                        first_name = groups[0].title()
                        last_name = ""
                        full_name = first_name
                    else:
                        first_name = ""
                        last_name = ""
                        full_name = query

                    return PatientName(
                    first_name=first_name,
                    last_name=last_name,
                    full_name=full_name,
                    original_query=query,
                    confidence=0.6,
                    variations=[full_name.lower(), full_name],
                    search_patterns=[
                        f"PATI_NAME LIKE '%{first_name}%'",
                        f"PATI_FULL_NAME LIKE '%{full_name}%'"
                    ]
                )
        
        # Si no se encuentra, devolver consulta completa
        return PatientName(
            first_name="",
            last_name="",
            full_name=query,
            original_query=query,
            confidence=0.3,
            variations=[query],
            search_patterns=[f"PATI_FULL_NAME LIKE '%{query}%'"]
        )
    
    def generate_sql_search_conditions(self, patient_name: PatientName) -> List[str]:
        """
        Genera condiciones SQL para buscar el paciente en la base de datos
        """
        conditions = []
        
        if patient_name.first_name and patient_name.last_name:
            # Búsqueda por nombre y apellido
            conditions.extend([
                f"PATI_NAME LIKE '%{patient_name.first_name}%' AND PATI_SURNAME_1 LIKE '%{patient_name.last_name}%'",
                f"PATI_FULL_NAME LIKE '%{patient_name.first_name} {patient_name.last_name}%'",
                f"PATI_FULL_NAME LIKE '%{patient_name.full_name}%'"
            ])
        elif patient_name.full_name:
            # Búsqueda por nombre completo
            conditions.extend([
                f"PATI_FULL_NAME LIKE '%{patient_name.full_name}%'",
                f"PATI_NAME LIKE '%{patient_name.full_name}%'"
            ])
        
        # Añadir variaciones
        for variation in patient_name.variations:
            conditions.append(f"PATI_FULL_NAME LIKE '%{variation}%'")
        
        return conditions
    
    def normalize_name_for_search(self, name: str) -> str:
        """
        Normaliza un nombre para búsqueda en la base de datos
        """
        # Convertir a minúsculas y limpiar
        normalized = name.lower().strip()
        
        # Remover caracteres especiales
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Normalizar espacios
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()

class IntelligentConceptDetector:
    """
    Agente especializado en detección de conceptos médicos
    """
    
    def __init__(self, llm=None):
        self.llm = llm
        self.concept_cache = {}
    
    async def detect_medical_concepts(self, query: str) -> Dict[str, Any]:
        """
        Detecta conceptos médicos en una consulta usando LLM inteligente
        """
        try:
            if not self.llm:
                return self._basic_concept_detection(query)
            
            # Cache para evitar llamadas repetidas
            cache_key = query.lower().strip()
            if cache_key in self.concept_cache:
                return self.concept_cache[cache_key]
            
            prompt = f"""
Eres un experto en terminología médica. Analiza esta consulta y detecta TODOS los conceptos médicos.

CONSULTA: "{query}"

INSTRUCCIONES:
1. Identifica TODOS los conceptos médicos mencionados
2. Clasifica por tipo: signos_vitales, diagnostico, medicamento, procedimiento, datos_paciente
3. Extrae términos relacionados y sinónimos
4. Genera patrones de búsqueda para SQL
5. Considera variaciones y abreviaciones

CONCEPTOS A DETECTAR:
- Signos vitales: tensión, presión, frecuencia, temperatura, peso, talla, saturación
- Diagnósticos: diabetes, hipertensión, enfermedades, condiciones
- Medicamentos: nombres de fármacos, tratamientos, dosis
- Procedimientos: análisis, pruebas, exploraciones
- Datos de paciente: información personal, historial

Responde SOLO con este JSON:
{{
  "concepts": [
    {{
      "concept": "nombre del concepto",
      "type": "signos_vitales|diagnostico|medicamento|procedimiento|datos_paciente",
      "variations": ["variación1", "variación2"],
      "sql_patterns": ["patrón SQL 1", "patrón SQL 2"],
      "confidence": 0.0-1.0
    }}
  ],
  "primary_intent": "tipo de consulta principal",
  "overall_confidence": 0.0-1.0,
  "reasoning": "explicación de la detección"
}}
"""
            
            response = await self.llm.ainvoke(prompt)
            content = str(response.content).strip()
            
            # Parsear respuesta usando utilidad robusta
            result = robust_json_parse(content, fallback=self._basic_concept_detection(query))
            
            # Asegurar que result sea un diccionario
            if result is None:
                result = self._basic_concept_detection(query)
            
            # Guardar en cache
            self.concept_cache[cache_key] = result
            
            logger.info(f"🧠 Conceptos detectados: {len(result.get('concepts', []))} (confianza: {result.get('overall_confidence', 0):.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en detección de conceptos: {e}")
            return self._basic_concept_detection(query)
    
    def _basic_concept_detection(self, query: str) -> Dict[str, Any]:
        """Detección básica sin LLM"""
        query_lower = query.lower()
        concepts = []
        
        # Detectar signos vitales
        vital_signs = ['tensión', 'presión', 'frecuencia', 'temperatura', 'peso', 'talla', 'saturación', 'constantes vitales']
        for sign in vital_signs:
            if sign in query_lower:
                concepts.append({
                    "concept": sign,
                    "type": "signos_vitales",
                    "variations": [sign, f"{sign} vitales"],
                    "sql_patterns": [f"DIAG_OBSERVATION LIKE '%{sign}%'", f"APPO_OBSERVATIONS LIKE '%{sign}%'"],
                    "confidence": 0.8
                })
        
        # Detectar datos de paciente
        if any(word in query_lower for word in ['datos', 'información', 'historial']):
            concepts.append({
                "concept": "datos de paciente",
                "type": "datos_paciente",
                "variations": ["datos", "información", "historial"],
                "sql_patterns": ["PATI_PATIENTS.*"],
                "confidence": 0.7
            })
        
        return {
            "concepts": concepts,
            "primary_intent": "consulta_general",
            "overall_confidence": 0.6,
            "reasoning": "Detección básica sin LLM"
        }

# Función de utilidad para importar fácilmente
def create_intelligent_detectors(llm=None) -> Tuple[IntelligentNameDetector, IntelligentConceptDetector]:
    """
    Crea instancias de los detectores inteligentes
    """
    name_detector = IntelligentNameDetector(llm)
    concept_detector = IntelligentConceptDetector(llm)
    return name_detector, concept_detector 