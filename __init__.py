"""
ChatMed V2 Flexible - Sistema de IA Médica Avanzado
==================================================

Sistema principal de ChatMed con arquitectura flexible y modular.
"""

__version__ = "2.0.0"
__author__ = "ChatMed Team"
__description__ = "Sistema de IA Médica con arquitectura flexible"

# Importar componentes principales
try:
    from .core.simple_orchestrator import SimpleOrchestrator
    from .agents.sql_agent_flexible_enhanced import SQLAgentIntelligentEnhanced
    from .agents.fhir_agent_complete import FHIRMedicalAgent
    from .agents.medgemma_clinical_agent import MedGemmaClinicalAgent
    from .agents.greeting_agent import IntelligentGreetingAgent
    
    __all__ = [
        'SimpleOrchestrator',
        'SQLAgentIntelligentEnhanced', 
        'FHIRMedicalAgent',
        'MedGemmaClinicalAgent',
        'IntelligentGreetingAgent'
    ]
except ImportError as e:
    # Si hay errores de importación, solo definir la versión
    __all__ = [] 