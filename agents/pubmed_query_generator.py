#!/usr/bin/env python3
"""
Módulo especializado en generar consultas PubMed optimizadas usando LLM.
Este módulo conoce las reglas específicas de búsqueda de PubMed y genera
consultas efectivas basadas en la consulta del usuario.
"""

import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

class PubMedQueryGenerator:
    """
    Generador especializado de consultas PubMed usando LLM.
    Conoce las reglas de búsqueda de PubMed y genera consultas optimizadas.
    """
    
    def __init__(self, openai_client: Optional[OpenAI] = None):
        self.openai_client = openai_client or OpenAI()
        self.logger = logger
    
    def generate_pubmed_query(self, user_query: str, max_results: int = 15) -> Dict[str, Any]:
        """
        Genera una consulta PubMed optimizada usando LLM especializado.
        
        Args:
            user_query: Consulta del usuario en español
            max_results: Número máximo de resultados
            
        Returns:
            Dict con la query optimizada y metadatos
        """
        try:
            # Prompt especializado en PubMed
            prompt = f"""
            Eres un experto en búsquedas de PubMed (NCBI) con amplio conocimiento de:
            - Reglas de sintaxis de PubMed
            - Operadores booleanos (AND, OR, NOT)
            - Campos de búsqueda ([tiab], [title], [abstract], etc.)
            - MeSH terms y vocabulario médico
            - Estrategias de búsqueda efectivas
            
            TAREA: Convierte esta consulta del usuario en una query PubMed optimizada.
            
            CONSULTA DEL USUARIO: {user_query}
            
            REGLAS IMPORTANTES:
            1. Usa AND para conceptos que deben aparecer JUNTOS
            2. Usa OR para sinónimos o términos relacionados
            3. Usa comillas para frases exactas: "systematic review"
            4. Usa [tiab] para buscar en título y abstract
            5. Usa [title] para buscar solo en título
            6. Usa [pt] para tipos de publicación
            7. Limita a revisiones sistemáticas si es apropiado: "systematic review"[pt]
            8. Usa MeSH terms cuando sea posible
            9. Mantén la query simple pero efectiva
            
            EJEMPLOS:
            - "vitamin D" AND "respiratory infections" AND "prevention"
            - ("systematic review"[pt] OR "meta-analysis"[pt]) AND "diabetes"
            - "metformin" AND "side effects" AND "clinical trials"
            
            GENERA SOLO LA QUERY PubMed (sin explicaciones):
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            if content is None:
                return {
                    "query": f'"{user_query}"',
                    "original_query": user_query,
                    "max_results": max_results,
                    "source": "fallback"
                }
            
            query = content.strip()
            
            # Limpiar la query
            query = self._clean_pubmed_query(query)
            
            self.logger.info(f"PubMed Query generada por LLM: {query}")
            
            return {
                "query": query,
                "original_query": user_query,
                "max_results": max_results,
                "source": "PubMedQueryGenerator"
            }
            
        except Exception as e:
            self.logger.error(f"Error generando query PubMed: {e}")
            # Query de fallback
            return {
                "query": f'"{user_query}"',
                "original_query": user_query,
                "max_results": max_results,
                "source": "fallback"
            }
    
    def _clean_pubmed_query(self, query: str) -> str:
        """
        Limpia y valida la query PubMed generada.
        """
        # Remover comillas extra al inicio/final
        query = query.strip().strip('"').strip("'")
        
        # Asegurar que no empiece con operadores booleanos
        if query.upper().startswith(('AND ', 'OR ', 'NOT ')):
            query = query[4:].strip()
        
        # Si la query está vacía, usar fallback
        if not query:
            return '"medical research"'
        
        return query
    
    def generate_advanced_query(self, user_query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Genera una query PubMed avanzada con filtros específicos.
        
        Args:
            user_query: Consulta del usuario
            filters: Filtros adicionales (años, tipos de estudio, etc.)
        """
        try:
            base_query = self.generate_pubmed_query(user_query)
            
            # Si no hay base_query válido, devolver fallback
            if not base_query:
                return {
                    "query": f'"{user_query}"',
                    "original_query": user_query,
                    "max_results": 15,
                    "source": "fallback"
                }
            
            # Si no hay filtros, devolver base_query
            if not filters:
                return base_query
            
            # Añadir filtros
            query_parts = [base_query["query"]]
            
            # Filtro por años
            if "year_from" in filters:
                query_parts.append(f'"{filters["year_from"]}"[dp] : "3000"[dp]')
            
            # Filtro por tipo de publicación
            if "publication_type" in filters:
                pt = filters["publication_type"]
                if pt == "systematic_review":
                    query_parts.append('"systematic review"[pt]')
                elif pt == "randomized_trial":
                    query_parts.append('"randomized controlled trial"[pt]')
                elif pt == "meta_analysis":
                    query_parts.append('"meta-analysis"[pt]')
            
            # Filtro por idioma
            if filters.get("language") == "english":
                query_parts.append('"english"[la]')
            
            # Combinar con AND
            final_query = " AND ".join(query_parts)
            
            return {
                "query": final_query,
                "original_query": user_query,
                "filters": filters,
                "source": "PubMedQueryGenerator_Advanced"
            }
            
        except Exception as e:
            self.logger.error(f"Error generando query avanzada: {e}")
            # Devolver fallback en lugar de llamar al método que puede devolver None
            return {
                "query": f'"{user_query}"',
                "original_query": user_query,
                "max_results": 15,
                "source": "fallback"
            }
    
    def generate_optimized_query(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Genera query PubMed optimizada para velocidad.
        """
        try:
            if not self.openai_client:
                return self._get_fallback_query(query)
            
            # Prompt más corto y directo
            prompt = f"""
            Genera una query PubMed optimizada para: "{query}"
            
            REGLAS:
            - Usa términos MeSH cuando sea posible
            - Incluye sinónimos con OR
            - Evita términos muy específicos
            - Máximo 3-4 conceptos principales
            
            Responde SOLO la query sin explicaciones.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            generated_query = response.choices[0].message.content
            
            if not generated_query:
                return self._get_fallback_query(query)
            
            # Limpiar y optimizar la query
            clean_query = self._clean_query(generated_query.strip())
            
            # Aplicar filtros básicos si se especifican
            if filters:
                clean_query = self._apply_basic_filters(clean_query, filters)
            
            return {
                "query": clean_query,
                "original_query": query,
                "optimized": True
            }
            
        except Exception as e:
            self.logger.error(f"Error generando query optimizada: {e}")
            return self._get_fallback_query(query)
    
    def _clean_query(self, query: str) -> str:
        """Limpia y optimiza la query."""
        # Remover markdown si existe
        if "```" in query:
            query = query.split("```")[1] if len(query.split("```")) > 1 else query
        
        # Remover líneas vacías y espacios extra
        query = " ".join(query.split())
        
        # Asegurar que no tenga comillas extra
        query = query.strip('"\'')
        
        return query
    
    def _apply_basic_filters(self, query: str, filters: Dict[str, Any]) -> str:
        """Aplica filtros básicos a la query."""
        # Solo aplicar filtros esenciales para velocidad
        if filters.get("year_from"):
            year = filters["year_from"]
            query += f" AND {year}[dp]"
        
        if filters.get("study_types") and "Systematic Reviews" in filters["study_types"]:
            query += " AND systematic review[pt]"
        
        return query
    
    def _get_fallback_query(self, query: str) -> Dict[str, Any]:
        """Devuelve una query de fallback."""
        return {
            "query": f'"{query}"',
            "original_query": query,
            "max_results": 15,
            "source": "fallback",
            "optimized": False
        } 