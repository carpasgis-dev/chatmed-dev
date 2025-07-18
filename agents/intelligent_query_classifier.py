#!/usr/bin/env python3
"""
Clasificador inteligente de consultas médicas que optimiza automáticamente
la estrategia de búsqueda y respuesta según el tipo de consulta.
"""

import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI
from enum import Enum

logger = logging.getLogger(__name__)

class QueryType(Enum):
    SYSTEMATIC_REVIEW = "systematic_review"
    CLINICAL_GUIDELINE = "clinical_guideline"
    DRUG_SAFETY = "drug_safety"
    TREATMENT_EFFICACY = "treatment_efficacy"
    DIAGNOSIS = "diagnosis"
    PROGNOSIS = "prognosis"
    GENERAL_INFO = "general_info"
    SYSTEM_QUESTION = "system_question"  # Nuevo tipo para preguntas sobre el sistema

class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

class IntelligentQueryClassifier:
    """
    Clasificador inteligente que analiza consultas médicas y optimiza
    automáticamente la estrategia de búsqueda.
    """
    
    def __init__(self, openai_client: Optional[OpenAI] = None):
        self.openai_client = openai_client or OpenAI()
        self.logger = logger
    
    def analyze_query(self, user_query: str) -> Dict[str, Any]:
        """
        Analiza una consulta médica de forma completamente genérica usando IA.
        """
        try:
            # Primero detectar si es una pregunta sobre el sistema
            system_keywords = [
                "langchain", "openai", "gpt", "llm", "modelo", 
                "usas", "utilizas", "versión", "framework",
                "librería", "biblioteca", "api", "tecnología"
            ]
            
            query_lower = user_query.lower()
            if any(keyword in query_lower for keyword in system_keywords):
                return {
                    "query_type": "system_question",
                    "complexity": "simple",
                    "strategy": {
                        "primary_sources": ["system_docs"],
                        "filters": {},
                        "max_results": 1
                    },
                    "confidence": 0.9
                }

            # Si no es pregunta del sistema, continuar con el análisis normal
            prompt = f"""
            Eres un experto en análisis de consultas médicas. Analiza esta consulta de forma genérica:
            
            CONSULTA: {user_query}
            
            ANÁLISIS REQUERIDO:
            1. TIPO DE CONSULTA (detecta automáticamente):
               - systematic_review: Revisiones sistemáticas
               - clinical_guideline: Guías clínicas
               - drug_safety: Seguridad de medicamentos
               - treatment_efficacy: Eficacia de tratamientos
               - diagnosis: Diagnóstico
               - prognosis: Pronóstico
               - general_info: Información general
               - system_question: Pregunta sobre el sistema
            
            2. COMPLEJIDAD (evalúa automáticamente):
               - simple: Consulta básica
               - moderate: Consulta intermedia
               - complex: Consulta compleja
            
            3. ESTRATEGIA DE BÚSQUEDA (genera automáticamente):
               - Fuentes prioritarias
               - Filtros recomendados
               - Número de resultados óptimo
            
            Responde en formato JSON:
            {{
                "query_type": "tipo_detectado",
                "complexity": "complejidad_detectada",
                "strategy": {{
                    "primary_sources": ["fuente1", "fuente2"],
                    "filters": {{}},
                    "max_results": número_óptimo
                }},
                "confidence": nivel_confianza
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            if not content:
                return self._get_default_analysis(user_query)
            
            import json
            analysis = json.loads(content)
            
            self.logger.info(f"🧠 [CLASSIFIER] Análisis genérico: {analysis['query_type']} ({analysis['complexity']})")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error en análisis genérico: {e}")
            return self._get_default_analysis(user_query)
    
    def _get_default_analysis(self, query: str) -> Dict[str, Any]:
        """Análisis por defecto completamente genérico."""
        return {
            "query_type": "general_info",
            "complexity": "moderate",
            "strategy": {
                "primary_sources": ["PubMed", "Google Scholar"],
                "filters": {
                    "year_from": 2018,
                    "study_types": ["systematic_review", "randomized_trial"],
                    "languages": ["english"]
                },
                "max_results": 15
            },
            "confidence": 0.7
        }
    
    def get_optimized_search_params(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera parámetros de búsqueda optimizados basados en el análisis.
        """
        strategy = analysis.get("strategy", {})
        
        # Ajustar parámetros según complejidad
        complexity = analysis.get("complexity", "moderate")
        if complexity == "simple":
            strategy["max_results"] = min(strategy.get("max_results", 10), 8)
        elif complexity == "complex":
            strategy["max_results"] = max(strategy.get("max_results", 10), 20)
        
        return {
            "max_results": strategy.get("max_results", 15),
            "filters": strategy.get("filters", {}),
            "sources": strategy.get("primary_sources", ["PubMed"]),
            "query_type": analysis.get("query_type", "general_info")
        } 