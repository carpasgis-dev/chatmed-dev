#!/usr/bin/env python3
"""
Módulo dinámico para corregir problemas de mapeo FHIR→SQL.
Usa LLM para corrección adaptativa sin hardcodeos.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class FHIRMappingCorrector:
    """
    Corrector dinámico de mapeos FHIR→SQL usando LLM.
    Detecta y corrige problemas automáticamente sin hardcodeos.
    """
    
    def __init__(self, llm=None):
        self.llm = llm
        self.fictitious_id_patterns = [
            'unico', 'urn:uuid:', 'patient-id', 'observation-id', 
            'encounter-id', 'medication-id', 'id-', 'ficticio', 
            'mock', 'fake', 'id-unico', 'patient-id-unico'
        ]
    
    async def correct_mapping_result(self, result: Dict[str, Any], fhir_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Corrige dinámicamente un resultado de mapeo FHIR→SQL usando LLM.
        PRESERVA valores válidos y solo corrige cuando es necesario.
        
        Args:
            result: Resultado del mapeo del LLM
            fhir_data: Datos FHIR originales
            
        Returns:
            Dict[str, Any]: Resultado corregido
        """
        if not result or not isinstance(result, dict):
            return result
        
        try:
            # Importar logger
            import logging
            logger = logging.getLogger(__name__)
            
            # VERIFICACIÓN CRÍTICA: Si ya tenemos valores válidos, ser conservador
            current_values = result.get('values', [])
            if current_values and any(val != "Unknown" and val is not None and val != "" for val in current_values):
                # Si ya tenemos valores válidos, solo aplicar corrección básica
                logger.info(f"PRESERVANDO valores válidos en correct_mapping_result: {current_values}")
                
                # Solo aplicar corrección básica para IDs ficticios
                corrected_result = result.copy()
                if corrected_result.get('values'):
                    corrected_values = []
                    for val in corrected_result['values']:
                        if isinstance(val, str) and any(pattern in val.lower() for pattern in [
                            'unico', 'urn:uuid:', 'patient-id', 'id-', 'ficticio', 'mock', 'fake', 'unknown'
                        ]):
                            corrected_values.append(None)
                        else:
                            corrected_values.append(val)
                    corrected_result['values'] = corrected_values
                
                return corrected_result
            
            # Si no tenemos valores válidos, aplicar corrección completa
            logger.info(f"Aplicando corrección completa para valores: {current_values}")
            
            # 1. Corregir valores usando LLM dinámico
            corrected_result = await self._correct_values_with_llm(result, fhir_data)
            
            # 2. Validar y corregir IDs ficticios
            corrected_result = await self._validate_ids_with_llm(corrected_result)
            
            # 3. Asegurar columnas requeridas dinámicamente
            corrected_result = await self._ensure_columns_with_llm(corrected_result, fhir_data)
            
            return corrected_result
            
        except Exception as e:
            logger.error(f"Error corrigiendo mapeo: {e}")
            return result
    
    async def _correct_values_with_llm(self, result: Dict[str, Any], fhir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Corrige valores usando LLM dinámico - PRESERVA valores válidos."""
        if not self.llm or not result.get('columns') or not result.get('values'):
            return result
        
        try:
            # Importar logger
            import logging
            logger = logging.getLogger(__name__)
            
            # VERIFICAR SI YA TENEMOS VALORES VÁLIDOS
            current_values = result.get('values', [])
            if current_values and any(val != "Unknown" and val is not None and val != "" for val in current_values):
                # Si ya tenemos valores válidos, solo aplicar corrección básica
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"PRESERVANDO valores válidos en LLM correction: {current_values}")
                return result
            
            # PROMPT DINÁMICO PARA CORRECCIÓN DE VALORES
            correction_prompt = f"""Eres un experto en corrección de mapeos FHIR→SQL médico.

DATOS FHIR ORIGINALES:
{json.dumps(fhir_data, indent=2, ensure_ascii=False)}

RESULTADO DEL MAPEO A CORREGIR:
{json.dumps(result, indent=2, ensure_ascii=False)}

TAREA: Corrige SOLO los valores que necesitan corrección. PRESERVA valores válidos.

REGLAS DE CORRECCIÓN CRÍTICAS:
1. NUNCA cambies valores válidos por "Unknown"
2. Solo corrige valores que son claramente incorrectos
3. Extrae valores simples desde objetos anidados
4. Convierte listas a strings apropiados
5. Maneja valores nulos correctamente
6. Preserva fechas en formato correcto

EJEMPLOS DE CORRECCIÓN:
- {{"given": ["Lidia"]}} → "Lidia"
- {{"family": "Gómez"}} → "Gómez"
- [{{"code": "GLUCOSE"}}] → "GLUCOSE"
- {{"value": 120, "unit": "mg/dL"}} → "120 mg/dL"

IMPORTANTE: Si un valor ya es válido (no "Unknown"), NO lo cambies.

RESPUESTA JSON:
{{
    "columns": ["columna1", "columna2"],
    "values": ["valor1_corregido", "valor2_corregido"],
    "corrections_applied": ["corrección1", "corrección2"]
}}

IMPORTANTE: Solo responde con el JSON corregido. NUNCA uses "Unknown"."""

            # Llamada al LLM para corrección
            response = await self._call_llm(correction_prompt)
            corrected = self._try_parse_llm_json(response)
            
            if corrected and isinstance(corrected, dict):
                # VERIFICAR que no se hayan introducido valores "Unknown"
                corrected_values = corrected.get('values', [])
                if any(val == "Unknown" for val in corrected_values):
                    # Si el LLM introdujo "Unknown", usar el resultado original
                    logger.warning(f"LLM introdujo 'Unknown', preservando valores originales")
                    return result
                
                return corrected
            else:
                return result
                
        except Exception as e:
            logger.error(f"Error en corrección con LLM: {e}")
            return result
    
    async def _validate_ids_with_llm(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Valida IDs usando LLM dinámico."""
        if not self.llm or not result.get('values'):
            return result
        
        try:
            # PROMPT DINÁMICO PARA VALIDACIÓN DE IDs
            validation_prompt = f"""Eres un experto en validación de IDs en bases de datos médicas.

VALORES A VALIDAR:
{json.dumps(result.get('values', []), indent=2, ensure_ascii=False)}

COLUMNAS CORRESPONDIENTES:
{json.dumps(result.get('columns', []), indent=2, ensure_ascii=False)}

TAREA: Detecta y corrige IDs ficticios o problemáticos.

REGLAS DE VALIDACIÓN:
1. IDs válidos: números, strings numéricos, UUIDs reales
2. IDs NO válidos: "patient-id-unico", "urn:uuid:", "id-unico", "ficticio"
3. Convierte IDs problemáticos a null
4. Preserva IDs válidos

RESPUESTA JSON:
{{
    "values": ["valor1_corregido", "valor2_corregido"],
    "ids_corrected": ["id1_corregido", "id2_corregido"]
}}

IMPORTANTE: Solo responde con el JSON corregido."""

            # Llamada al LLM para validación
            response = await self._call_llm(validation_prompt)
            validated = self._try_parse_llm_json(response)
            
            if validated and isinstance(validated, dict):
                result['values'] = validated.get('values', result.get('values', []))
                return result
            else:
                return result
                
        except Exception as e:
            logger.error(f"Error en validación con LLM: {e}")
            return result
    
    async def _ensure_columns_with_llm(self, result: Dict[str, Any], fhir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Asegura columnas requeridas usando LLM dinámico."""
        if not self.llm:
            return result
        
        try:
            # PROMPT DINÁMICO PARA COLUMNAS REQUERIDAS
            columns_prompt = f"""Eres un experto en esquemas de bases de datos médicas.

DATOS FHIR:
{json.dumps(fhir_data, indent=2, ensure_ascii=False)}

COLUMNAS ACTUALES:
{json.dumps(result.get('columns', []), indent=2, ensure_ascii=False)}

VALORES ACTUALES:
{json.dumps(result.get('values', []), indent=2, ensure_ascii=False)}

TAREA: Identifica columnas requeridas que faltan y añádelas.

REGLAS:
1. Analiza el tipo de recurso FHIR
2. Identifica columnas esenciales que faltan
3. Añade valores apropiados para columnas faltantes
4. Mantén coherencia con el esquema

RESPUESTA JSON:
{{
    "columns": ["columna1", "columna2", "columna_faltante"],
    "values": ["valor1", "valor2", "valor_faltante"],
    "columns_added": ["columna_faltante"]
}}

IMPORTANTE: Solo responde con el JSON corregido."""

            # Llamada al LLM para columnas
            response = await self._call_llm(columns_prompt)
            enhanced = self._try_parse_llm_json(response)
            
            if enhanced and isinstance(enhanced, dict):
                return enhanced
            else:
                return result
                
        except Exception as e:
            logger.error(f"Error en columnas con LLM: {e}")
            return result
    
    async def _call_llm(self, prompt: str) -> str:
        """Llama al LLM de forma segura."""
        try:
            if not self.llm:
                return '{"error": "LLM no disponible"}'
            
            # Importar función de llamada LLM
            from agents.sql_agent_flexible_enhanced import _call_openai_native
            
            response = await asyncio.to_thread(
                _call_openai_native, 
                self.llm, 
                [{"role": "user", "content": prompt}],
                task_description="Corrección FHIR dinámica"
            )
            
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"Error llamando LLM: {e}")
            return '{"error": "Error en llamada LLM"}'
    
    def _try_parse_llm_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Parsea JSON del LLM con múltiples estrategias."""
        if not content or not isinstance(content, str):
            return None
        
        # Estrategia 1: Parseo directo
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Estrategia 2: Buscar JSON en el contenido
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                json_str = content[start:end]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Estrategia 3: Limpiar y reintentar
        try:
            cleaned = content.replace('\n', ' ').replace('\r', ' ')
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        return None

# Instancia global para uso fácil
fhir_mapping_corrector = None

async def correct_fhir_mapping_result(result: Dict[str, Any], fhir_data: Dict[str, Any], llm=None) -> Dict[str, Any]:
    """
    Función de conveniencia para corregir resultados de mapeo FHIR→SQL.
    
    Args:
        result: Resultado del mapeo del LLM
        fhir_data: Datos FHIR originales
        llm: Cliente LLM (opcional)
        
    Returns:
        Dict[str, Any]: Resultado corregido
    """
    global fhir_mapping_corrector
    
    # Importar logger
    import logging
    logger = logging.getLogger(__name__)
    
    if fhir_mapping_corrector is None:
        fhir_mapping_corrector = FHIRMappingCorrector(llm)
    
    # VERIFICACIÓN CRÍTICA: Si el resultado ya tiene valores válidos, NO lo corrijas
    if result and isinstance(result, dict):
        values = result.get('values', [])
        columns = result.get('columns', [])
        
        # Si ya tenemos valores válidos (no "Unknown"), preservarlos
        if values and any(val != "Unknown" and val is not None and val != "" for val in values):
            # Solo aplicar corrección básica para IDs ficticios
            corrected_result = _basic_correction(result, fhir_data)
            
            # DEBUG: Log de preservación de valores
            logger.info(f"PRESERVANDO valores válidos: {values}")
            
            return corrected_result
    
    # Si no hay LLM, usar corrección básica
    if not llm:
        return _basic_correction(result, fhir_data)
    
    # Usar corrección con LLM solo si es necesario
    return await fhir_mapping_corrector.correct_mapping_result(result, fhir_data)

def _basic_correction(result: Dict[str, Any], fhir_data: Dict[str, Any]) -> Dict[str, Any]:
    """Corrección básica sin LLM - PRESERVA valores válidos."""
    if not result or not isinstance(result, dict):
        return result
    
    # Importar logger
    import logging
    logger = logging.getLogger(__name__)
    
    # DEBUG: Log de entrada
    logger.info(f"BASIC_CORRECTION - Entrada: {result}")
    
    # Corrección básica de IDs ficticios - SOLO corregir IDs problemáticos
    if result.get('values'):
        corrected_values = []
        for i, val in enumerate(result['values']):
            # PRESERVAR valores válidos
            if val is not None and val != "Unknown" and val != "":
                # Solo corregir si es claramente un ID ficticio
                if isinstance(val, str) and any(pattern in val.lower() for pattern in [
                    'unico', 'urn:uuid:', 'patient-id', 'id-', 'ficticio', 'mock', 'fake', 'unknown'
                ]):
                    logger.info(f"Corrigiendo ID ficticio: {val} → null")
                    corrected_values.append(None)
                else:
                    # PRESERVAR valor válido
                    logger.info(f"Preservando valor válido: {val}")
                    corrected_values.append(val)
            else:
                # Corregir valores nulos/vacíos/"Unknown"
                if val == "Unknown":
                    logger.info(f"Corrigiendo valor 'Unknown': {val} → null")
                    corrected_values.append(None)
                else:
                    corrected_values.append(val)
        
        result['values'] = corrected_values
        logger.info(f"BASIC_CORRECTION - Salida: {result}")
    
    return result 