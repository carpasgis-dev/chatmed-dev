#!/usr/bin/env python3
"""
Módulo para procesamiento eficiente en lotes de recursos FHIR
Optimiza la persistencia agrupando recursos por tipo y procesándolos en lotes
"""

import asyncio
from typing import Dict, List, Any, Optional
import json
import logging
import sqlite3

logger = logging.getLogger(__name__)

class FHIRBatchProcessor:
    """Procesador eficiente de recursos FHIR en lotes"""
    
    def __init__(self, sql_agent, fhir_agent=None):
        self.sql_agent = sql_agent
        self.fhir_agent = fhir_agent  # Referencia al agente FHIR para corrección de UUIDs
        
    async def process_fhir_batch(self, resources: List[Dict], patient_id: str) -> Dict[str, Any]:
        """Procesa recursos FHIR en lotes usando el SQLAgent flexible/inteligente"""
        try:
            print(f"🚀 Procesando {len(resources)} recursos en lotes con ID de paciente: {patient_id}")
            results = []
            total_processed = 0
            
            for i, resource in enumerate(resources, 1):
                print(f"      - Procesando {i}/{len(resources)}: {resource.get('resourceType')}")
                
                # CORRECCIÓN DE UUIDs ANTES DEL PROCESAMIENTO
                corrected_resource = resource
                if self.fhir_agent and hasattr(self.fhir_agent, '_fix_fhir_uuid_mapping'):
                    try:
                        print(f"      🔧 Corrigiendo UUIDs para {resource.get('resourceType')}...")
                        corrected_resource = await self.fhir_agent._fix_fhir_uuid_mapping(resource, patient_id)
                        print(f"      ✅ UUIDs corregidos para {resource.get('resourceType')}")
                        
                        # Verificar que los UUIDs se corrigieron
                        if 'id' in corrected_resource and 'urn:uuid:' in str(corrected_resource['id']):
                            print(f"      ⚠️ UUID no corregido: {corrected_resource['id']}")
                            # Aplicar corrección básica
                            corrected_resource = await self.fhir_agent._fix_fhir_uuid_mapping(corrected_resource, patient_id)
                            print(f"      🔧 Corrección básica aplicada")
                        
                    except Exception as e:
                        print(f"      ❌ Error corrigiendo UUIDs: {e}")
                        print(f"      🔧 Usando recurso original")
                        corrected_resource = resource
                else:
                    print(f"      ⚠️ Agente FHIR no disponible para corrección de UUIDs")
                
                # El SQLAgentFlexibleEnhanced se encarga del mapeo y persistencia inteligente
                result = await self.sql_agent.process_data_manipulation(
                    operation='INSERT',
                    data=corrected_resource,
                    context={'resource_type': corrected_resource.get('resourceType', 'Unknown'), 'patient_id': patient_id}
                )
                
                # CORRECCIÓN: Asegurar que result sea un diccionario
                if not isinstance(result, dict):
                    print(f"      ⚠️ Result no es un diccionario, convirtiendo...")
                    result = {
                        'success': False,
                        'error': f'Result no es un diccionario: {type(result)}',
                        'data': result if result else []
                    }
                
                results.append(result)
                if result.get('success'):
                    total_processed += 1
                    print(f"      ✅ Recurso {i} procesado exitosamente")
                else:
                    print(f"      ❌ Error procesando recurso {i}: {result.get('error', 'Error desconocido')}")
            
            print(f"✅ Procesamiento en lotes completado: {total_processed} recursos insertados")
            return {
                'success': True,
                'results': results,
                'total_resources': len(resources),
                'total_processed': total_processed,
                'patient_id': patient_id
            }
        except Exception as e:
            logger.error(f"Error en procesamiento por lotes: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_resources': len(resources)
            } 