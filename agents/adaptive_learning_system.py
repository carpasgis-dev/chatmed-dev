#!/usr/bin/env python3
"""
Sistema de aprendizaje adaptativo que mejora automáticamente las búsquedas
basándose en el feedback del usuario y el rendimiento histórico.
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

class AdaptiveLearningSystem:
    """
    Sistema que aprende de las interacciones del usuario para optimizar
    futuras búsquedas y respuestas.
    """
    
    def __init__(self, storage_file: str = "adaptive_learning.json"):
        self.storage_file = storage_file
        self.logger = logger
        self.learning_data = self._load_learning_data()
        
    def _load_learning_data(self) -> Dict[str, Any]:
        """Carga datos de aprendizaje desde archivo."""
        try:
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "query_patterns": {},
                "successful_strategies": {},
                "user_preferences": {},
                "performance_metrics": {},
                "last_updated": datetime.now().isoformat()
            }
    
    def _save_learning_data(self):
        """Guarda datos de aprendizaje en archivo."""
        try:
            self.learning_data["last_updated"] = datetime.now().isoformat()
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.learning_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error guardando datos de aprendizaje: {e}")
    
    def record_query_execution(self, query: str, strategy: Dict[str, Any], 
                             results: Dict[str, Any], execution_time: float):
        """
        Registra la ejecución de una consulta para aprendizaje.
        """
        try:
            query_key = self._normalize_query(query)
            
            # Registrar métricas de rendimiento
            if query_key not in self.learning_data["performance_metrics"]:
                self.learning_data["performance_metrics"][query_key] = []
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "strategy": strategy,
                "results_count": results.get("total_sources_found", 0),
                "execution_time": execution_time,
                "has_pubmed_results": len(results.get("pubmed_search", {}).get("articles", [])) > 0,
                "has_synthesis": bool(results.get("synthesis", ""))
            }
            
            self.learning_data["performance_metrics"][query_key].append(metrics)
            
            # Mantener solo los últimos 10 registros por consulta
            if len(self.learning_data["performance_metrics"][query_key]) > 10:
                self.learning_data["performance_metrics"][query_key] = \
                    self.learning_data["performance_metrics"][query_key][-10:]
            
            self._save_learning_data()
            
        except Exception as e:
            self.logger.error(f"Error registrando ejecución: {e}")
    
    def record_user_feedback(self, query: str, feedback_type: str, 
                           feedback_value: Any, user_rating: Optional[int] = None):
        """
        Registra feedback del usuario para mejorar futuras búsquedas.
        """
        try:
            query_key = self._normalize_query(query)
            
            if query_key not in self.learning_data["user_preferences"]:
                self.learning_data["user_preferences"][query_key] = {
                    "feedback_history": [],
                    "preferred_strategies": [],
                    "average_rating": 0.0
                }
            
            feedback = {
                "timestamp": datetime.now().isoformat(),
                "type": feedback_type,
                "value": feedback_value,
                "rating": user_rating
            }
            
            self.learning_data["user_preferences"][query_key]["feedback_history"].append(feedback)
            
            # Actualizar rating promedio
            if user_rating is not None:
                ratings = [f["rating"] for f in self.learning_data["user_preferences"][query_key]["feedback_history"] 
                          if f["rating"] is not None]
                if ratings:
                    self.learning_data["user_preferences"][query_key]["average_rating"] = sum(ratings) / len(ratings)
            
            self._save_learning_data()
            
        except Exception as e:
            self.logger.error(f"Error registrando feedback: {e}")
    
    def get_optimized_strategy(self, query: str, base_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Obtiene una estrategia optimizada basada en el aprendizaje previo.
        """
        try:
            query_key = self._normalize_query(query)
            optimized_strategy = base_strategy.copy()
            
            # Aplicar aprendizajes de consultas similares
            similar_queries = self._find_similar_queries(query_key)
            
            for similar_query in similar_queries:
                # Ajustar parámetros basándose en éxito previo
                if similar_query in self.learning_data["performance_metrics"]:
                    metrics = self.learning_data["performance_metrics"][similar_query]
                    
                    # Encontrar la estrategia más exitosa
                    successful_strategies = [m for m in metrics if m.get("results_count", 0) > 0]
                    if successful_strategies:
                        best_strategy = max(successful_strategies, 
                                          key=lambda x: x.get("results_count", 0))
                        
                        # Aplicar ajustes basándose en la estrategia exitosa
                        if best_strategy["strategy"].get("max_results", 0) > optimized_strategy.get("max_results", 0):
                            optimized_strategy["max_results"] = best_strategy["strategy"]["max_results"]
                        
                        # Ajustar filtros si fueron exitosos
                        if best_strategy["strategy"].get("filters"):
                            optimized_strategy["filters"] = best_strategy["strategy"]["filters"]
                
                # Aplicar preferencias del usuario
                if similar_query in self.learning_data["user_preferences"]:
                    user_prefs = self.learning_data["user_preferences"][similar_query]
                    if user_prefs.get("average_rating", 0) > 4.0:  # Si el usuario dio buena calificación
                        # Usar estrategias similares
                        pass
            
            self.logger.info(f"Estrategia optimizada aplicada para: {query_key}")
            return optimized_strategy
            
        except Exception as e:
            self.logger.error(f"Error optimizando estrategia: {e}")
            return base_strategy
    
    def _normalize_query(self, query: str) -> str:
        """Normaliza una consulta para usar como clave."""
        return query.lower().strip()
    
    def _find_similar_queries(self, query_key: str) -> List[str]:
        """Encuentra consultas similares basándose en palabras clave."""
        similar_queries = []
        
        # Buscar consultas que contengan palabras clave similares
        query_words = set(query_key.split())
        
        for stored_query in self.learning_data["performance_metrics"].keys():
            stored_words = set(stored_query.split())
            
            # Calcular similitud (intersección de palabras)
            intersection = len(query_words.intersection(stored_words))
            if intersection >= 2:  # Al menos 2 palabras en común
                similar_queries.append(stored_query)
        
        return similar_queries[:5]  # Máximo 5 consultas similares
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """
        Obtiene insights del sistema de aprendizaje.
        """
        try:
            insights = {
                "total_queries_learned": len(self.learning_data["performance_metrics"]),
                "average_execution_time": 0.0,
                "most_successful_strategies": [],
                "user_satisfaction": 0.0
            }
            
            # Calcular tiempo promedio de ejecución
            all_times = []
            for query_metrics in self.learning_data["performance_metrics"].values():
                for metric in query_metrics:
                    if metric.get("execution_time"):
                        all_times.append(metric["execution_time"])
            
            if all_times:
                insights["average_execution_time"] = sum(all_times) / len(all_times)
            
            # Calcular satisfacción promedio del usuario
            all_ratings = []
            for user_prefs in self.learning_data["user_preferences"].values():
                if user_prefs.get("average_rating"):
                    all_ratings.append(user_prefs["average_rating"])
            
            if all_ratings:
                insights["user_satisfaction"] = sum(all_ratings) / len(all_ratings)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error obteniendo insights: {e}")
            return {}
    
    def optimize_strategy(self, query_type: str, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimiza la estrategia de búsqueda de forma completamente genérica basada en métricas.
        """
        try:
            # Análisis genérico de métricas
            avg_time = performance_metrics.get('avg_search_time', 0)
            relevance_score = performance_metrics.get('avg_relevance', 0)
            user_satisfaction = performance_metrics.get('user_satisfaction', 0)
            
            # Optimización genérica basada en patrones
            optimized_strategy = {
                "query_type": query_type,
                "optimizations": []
            }
            
            # Optimización por tiempo de búsqueda
            if avg_time > 30:
                optimized_strategy["optimizations"].append({
                    "type": "speed_optimization",
                    "action": "reduce_sources",
                    "description": "Reducir fuentes para mejorar velocidad"
                })
            
            # Optimización por relevancia
            if relevance_score < 0.7:
                optimized_strategy["optimizations"].append({
                    "type": "relevance_optimization", 
                    "action": "improve_filters",
                    "description": "Mejorar filtros para mayor relevancia"
                })
            
            # Optimización por satisfacción del usuario
            if user_satisfaction < 0.8:
                optimized_strategy["optimizations"].append({
                    "type": "quality_optimization",
                    "action": "enhance_synthesis",
                    "description": "Mejorar síntesis para mayor calidad"
                })
            
            # Estrategia adaptativa genérica
            if query_type in ["systematic_review", "clinical_guideline"]:
                optimized_strategy["recommended_sources"] = ["PubMed", "Cochrane", "Guidelines"]
                optimized_strategy["max_results"] = 20
            elif query_type in ["drug_safety", "treatment_efficacy"]:
                optimized_strategy["recommended_sources"] = ["PubMed", "FDA", "EMA"]
                optimized_strategy["max_results"] = 15
            elif query_type in ["diagnosis", "prognosis"]:
                optimized_strategy["recommended_sources"] = ["PubMed", "UpToDate", "ClinicalKey"]
                optimized_strategy["max_results"] = 12
            else:
                # Estrategia genérica por defecto
                optimized_strategy["recommended_sources"] = ["PubMed", "Google Scholar"]
                optimized_strategy["max_results"] = 10
            
            self.logger.info(f"🎯 [ADAPTIVE] Optimización genérica para {query_type}: {len(optimized_strategy['optimizations'])} mejoras")
            return optimized_strategy
            
        except Exception as e:
            self.logger.error(f"Error en optimización genérica: {e}")
            return self._get_default_strategy(query_type)
    
    def _get_default_strategy(self, query_type: str) -> Dict[str, Any]:
        """Estrategia por defecto completamente genérica."""
        return {
            "query_type": query_type,
            "optimizations": [],
            "recommended_sources": ["PubMed", "Google Scholar"],
            "max_results": 10
        } 