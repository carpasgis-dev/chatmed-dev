"""
Módulo de Agentes ChatMed v2 Flexible
====================================

Agentes especializados para el sistema ChatMed v2.0 Flexible.
"""

# Importar agentes disponibles con manejo de errores
__all__ = []

# Agentes críticos
try:
    from .sql_agent_flexible_enhanced import SQLAgentIntelligentEnhanced
    __all__.append('SQLAgentIntelligentEnhanced')
except ImportError:
    pass

try:
    from .fhir_agent_complete import FHIRMedicalAgent
    __all__.append('FHIRMedicalAgent')
except ImportError:
    pass

try:
    from .greeting_agent import IntelligentGreetingAgent
    __all__.append('IntelligentGreetingAgent')
except ImportError:
    pass

# Agentes opcionales
try:
    from .biochat_agent import BioChatAgent
    __all__.append('BioChatAgent')
except ImportError:
    pass

try:
    from .pubmed_query_generator import PubMedQueryGenerator
    __all__.append('PubMedQueryGenerator')
except ImportError:
    pass

try:
    from .medgemma_clinical_agent import MedGemmaClinicalAgent
    __all__.append('MedGemmaClinicalAgent')
except ImportError:
    pass

try:
    from .fhir_persistence_agent_old import FHIRPersistenceAgent
    __all__.append('FHIRPersistenceAgent')
except ImportError:
    pass

# Agentes legacy (mantener compatibilidad)
try:
    from .sql_agent_flexible import SQLAgentIntelligent
    __all__.append('SQLAgentIntelligent')
except ImportError:
    pass

try:
    from .sql_agent_clean import SQLAgentRobust
    __all__.append('SQLAgentRobust')
except ImportError:
    pass 