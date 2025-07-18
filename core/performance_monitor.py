"""
Performance Monitor - Monitor de Rendimiento
==========================================

Módulo simple para monitoreo de rendimiento del sistema.
"""

import time
from typing import Dict, Any

class PerformanceMonitor:
    """Monitor de rendimiento básico"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {}
    
    def record_metric(self, name: str, value: Any):
        """Registrar una métrica"""
        self.metrics[name] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener todas las métricas"""
        return self.metrics.copy()
    
    def get_uptime(self) -> float:
        """Obtener tiempo de actividad"""
        return time.time() - self.start_time 