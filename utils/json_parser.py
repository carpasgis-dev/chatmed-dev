"""
Utilidades mejoradas para parseo JSON robusto
=============================================

Este módulo proporciona funciones para parsear JSON de respuestas del LLM
de manera robusta, manejando errores comunes como texto adicional,
formato markdown, y JSON malformado.
"""

import json
import re
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

def robust_json_parse(content: str, fallback: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Parsea JSON de manera robusta desde respuestas del LLM
    
    Args:
        content: Contenido de la respuesta del LLM
        fallback: Valor a devolver si no se puede parsear
        
    Returns:
        Dict parseado o fallback
    """
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
        
        # Limpiar líneas vacías y espacios extra
        content = content.strip()
        
        # Si el contenido empieza con { y termina con }, intentar parsear directamente
        if content.startswith('{') and content.endswith('}'):
            try:
                result = json.loads(content)
                logger.debug("✅ JSON parseado directamente (formato limpio)")
                return result
            except json.JSONDecodeError:
                pass
        
        # Estrategia 1: Intentar parsear directamente
        try:
            result = json.loads(content)
            logger.debug("✅ JSON parseado directamente")
            return result
        except json.JSONDecodeError:
            pass
        
        # Estrategia 2: Buscar JSON con regex mejorado
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # JSON simple anidado
            r'\{.*\}',  # JSON con cualquier contenido
            r'\[.*\]',  # Array JSON
        ]
        
        # Estrategia 2.1: Buscar JSON dentro de bloques de código markdown
        markdown_json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        markdown_match = re.search(markdown_json_pattern, content, re.DOTALL)
        if markdown_match:
            try:
                json_str = markdown_match.group(1)
                result = json.loads(json_str)
                logger.debug("✅ JSON parseado desde bloque markdown")
                return result
            except json.JSONDecodeError as e:
                logger.debug(f"⚠️ Error parseando JSON desde markdown: {e}")
        
        for pattern in json_patterns:
            json_match = re.search(pattern, content, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    result = json.loads(json_str)
                    logger.debug(f"✅ JSON parseado con patrón: {pattern}")
                    return result
                except json.JSONDecodeError as e:
                    logger.debug(f"⚠️ Error parseando JSON con patrón {pattern}: {e}")
                    continue
        
        # Estrategia 3: Buscar líneas individuales que parezcan JSON
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    result = json.loads(line)
                    logger.debug("✅ JSON encontrado en línea individual")
                    return result
                except:
                    continue
        
        # Estrategia 4: Intentar reparar JSON común
        repaired_content = _repair_common_json_issues(content)
        if repaired_content:
            try:
                result = json.loads(repaired_content)
                logger.debug("✅ JSON reparado exitosamente")
                return result
            except json.JSONDecodeError:
                pass
        
        logger.warning(f"❌ No se pudo parsear JSON del contenido")
        logger.debug(f"🔍 DEBUG: Contenido recibido: {content[:200]}...")
        return fallback
        
    except Exception as e:
        logger.error(f"❌ Error general parseando JSON: {e}")
        return fallback

def _repair_common_json_issues(content: str) -> Optional[str]:
    """
    Intenta reparar problemas comunes en JSON malformado
    """
    try:
        # Remover comentarios de línea
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        
        # Remover comentarios de bloque
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Reparar comillas no escapadas en strings
        content = re.sub(r'(?<!\\)"(?=.*?":)', r'\\"', content)
        
        # Reparar comas finales
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        
        # Reparar valores booleanos/nulos mal escritos
        content = re.sub(r'\btrue\b', 'true', content)
        content = re.sub(r'\bfalse\b', 'false', content)
        content = re.sub(r'\bnull\b', 'null', content)
        
        return content.strip()
        
    except Exception:
        return None

def extract_json_objects(content: str) -> List[Dict[str, Any]]:
    """
    Extrae múltiples objetos JSON de una respuesta del LLM
    
    Args:
        content: Contenido de la respuesta
        
    Returns:
        Lista de objetos JSON encontrados
    """
    objects = []
    
    try:
        # Limpiar contenido
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        elif content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        
        # Buscar múltiples objetos JSON
        object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(object_pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                obj = json.loads(match)
                if isinstance(obj, dict):
                    objects.append(obj)
            except json.JSONDecodeError:
                continue
        
        return objects
        
    except Exception as e:
        logger.error(f"❌ Error extrayendo objetos JSON: {e}")
        return []

def validate_json_structure(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Valida que un JSON tenga los campos requeridos
    
    Args:
        data: Datos a validar
        required_fields: Lista de campos requeridos
        
    Returns:
        True si todos los campos están presentes
    """
    try:
        for field in required_fields:
            if field not in data:
                logger.warning(f"⚠️ Campo requerido faltante: {field}")
                return False
        return True
    except Exception as e:
        logger.error(f"❌ Error validando estructura JSON: {e}")
        return False 