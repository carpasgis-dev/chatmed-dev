#!/usr/bin/env python3
"""
💊 Drug Detection AEMPS - Detector de Fármacos ChatMed
=====================================================

Módulo para detección y análisis de fármacos usando datos de AEMPS
(Agencia Española de Medicamentos y Productos Sanitarios)

Autor: ChatMed System
Versión: 1.0
"""

import logging
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger("DrugDetectionAEMPS")

@dataclass
class DrugInfo:
    """Información de un fármaco detectado"""
    name: str
    active_ingredient: Optional[str] = None
    dosage: Optional[str] = None
    route: Optional[str] = None
    confidence: float = 0.0
    aemps_code: Optional[str] = None
    category: Optional[str] = None

class DrugDetectionAEMPS:
    """
    Detector de fármacos usando patrones y base de datos AEMPS
    """
    
    def __init__(self):
        """Inicializa el detector de fármacos"""
        self.drug_patterns = self._load_drug_patterns()
        self.common_drugs = self._load_common_drugs()
        logger.info("✅ DrugDetectionAEMPS inicializado")
    
    def _load_drug_patterns(self) -> Dict[str, str]:
        """Carga patrones de reconocimiento de fármacos"""
        return {
            # Antibióticos
            r'\b(amoxicilina|penicilina|azitromicina|ciprofloxacino)\b': 'antibiotico',
            
            # Analgésicos
            r'\b(paracetamol|ibuprofeno|aspirina|diclofenaco|naproxeno)\b': 'analgesico',
            
            # Antihipertensivos
            r'\b(enalapril|losartan|amlodipino|atenolol|metoprolol)\b': 'antihipertensivo',
            
            # Antidiabéticos
            r'\b(metformina|glibenclamida|insulina|gliclazida)\b': 'antidiabetico',
            
            # Antihistamínicos
            r'\b(loratadina|cetirizina|desloratadina|fexofenadina)\b': 'antihistaminico',
            
            # Protectores gástricos
            r'\b(omeprazol|lansoprazol|pantoprazol|esomeprazol)\b': 'protector_gastrico',
            
            # Ansiolíticos
            r'\b(lorazepam|alprazolam|diazepam|clonazepam)\b': 'ansiolitico',
            
            # Antidepresivos
            r'\b(sertralina|fluoxetina|paroxetina|citalopram)\b': 'antidepresivo',
            
            # Patrones de dosificación
            r'\b(\d+)\s*(mg|g|ml|mcg|ui)\b': 'dosage',
            
            # Patrones de frecuencia
            r'\b(cada|c/)\s*(\d+)\s*(horas?|h|días?|d)\b': 'frequency'
        }
    
    def _load_common_drugs(self) -> Dict[str, DrugInfo]:
        """Carga base de datos de fármacos comunes"""
        return {
            'paracetamol': DrugInfo(
                name='Paracetamol',
                active_ingredient='Paracetamol',
                category='Analgésico/Antipirético',
                aemps_code='CN123456'
            ),
            'ibuprofeno': DrugInfo(
                name='Ibuprofeno',
                active_ingredient='Ibuprofeno',
                category='Antiinflamatorio no esteroideo',
                aemps_code='CN234567'
            ),
            'amoxicilina': DrugInfo(
                name='Amoxicilina',
                active_ingredient='Amoxicilina',
                category='Antibiótico betalactámico',
                aemps_code='CN345678'
            ),
            'omeprazol': DrugInfo(
                name='Omeprazol',
                active_ingredient='Omeprazol',
                category='Inhibidor bomba protones',
                aemps_code='CN456789'
            ),
            'enalapril': DrugInfo(
                name='Enalapril',
                active_ingredient='Enalapril maleato',
                category='IECA - Antihipertensivo',
                aemps_code='CN567890'
            )
        }
    
    def detect_drugs_in_text(self, text: str) -> List[DrugInfo]:
        """
        Detecta fármacos en un texto dado
        
        Args:
            text: Texto a analizar
            
        Returns:
            Lista de fármacos detectados
        """
        detected_drugs = []
        text_lower = text.lower()
        
        # Buscar fármacos conocidos
        for drug_name, drug_info in self.common_drugs.items():
            if drug_name in text_lower:
                # Buscar dosificación cercana
                dosage = self._extract_dosage_near_drug(text, drug_name)
                
                # Crear copia del fármaco con dosificación
                detected_drug = DrugInfo(
                    name=drug_info.name,
                    active_ingredient=drug_info.active_ingredient,
                    dosage=dosage,
                    category=drug_info.category,
                    aemps_code=drug_info.aemps_code,
                    confidence=0.9
                )
                
                detected_drugs.append(detected_drug)
        
        # Buscar patrones adicionales
        for pattern, category in self.drug_patterns.items():
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                drug_name = match.group(1) if match.groups() else match.group(0)
                
                # Evitar duplicados
                if not any(d.name.lower() == drug_name.lower() for d in detected_drugs):
                    detected_drug = DrugInfo(
                        name=drug_name.title(),
                        category=category,
                        confidence=0.7
                    )
                    detected_drugs.append(detected_drug)
        
        logger.info(f"🔍 Detectados {len(detected_drugs)} fármacos en el texto")
        return detected_drugs
    
    def _extract_dosage_near_drug(self, text: str, drug_name: str) -> Optional[str]:
        """Extrae dosificación cercana a un fármaco"""
        # Buscar patrones de dosificación cerca del nombre del fármaco
        dosage_pattern = r'(\d+)\s*(mg|g|ml|mcg|ui)'
        
        # Buscar en ventana de ±50 caracteres alrededor del fármaco
        drug_pos = text.lower().find(drug_name.lower())
        if drug_pos != -1:
            start = max(0, drug_pos - 50)
            end = min(len(text), drug_pos + len(drug_name) + 50)
            context = text[start:end]
            
            dosage_match = re.search(dosage_pattern, context, re.IGNORECASE)
            if dosage_match:
                return f"{dosage_match.group(1)} {dosage_match.group(2)}"
        
        return None
    
    def get_drug_info(self, drug_name: str) -> Optional[DrugInfo]:
        """
        Obtiene información detallada de un fármaco
        
        Args:
            drug_name: Nombre del fármaco
            
        Returns:
            Información del fármaco o None si no se encuentra
        """
        drug_key = drug_name.lower()
        return self.common_drugs.get(drug_key)
    
    def validate_drug_interaction(self, drugs: List[str]) -> Dict[str, Any]:
        """
        Valida posibles interacciones entre fármacos (versión básica)
        
        Args:
            drugs: Lista de nombres de fármacos
            
        Returns:
            Información sobre interacciones
        """
        # Interacciones conocidas básicas
        interactions = {
            ('ibuprofeno', 'enalapril'): 'Puede reducir efecto antihipertensivo',
            ('omeprazol', 'clopidogrel'): 'Puede reducir efecto antiagregante',
            ('paracetamol', 'warfarina'): 'Puede potenciar efecto anticoagulante'
        }
        
        found_interactions = []
        drugs_lower = [d.lower() for d in drugs]
        
        for (drug1, drug2), interaction in interactions.items():
            if drug1 in drugs_lower and drug2 in drugs_lower:
                found_interactions.append({
                    'drugs': [drug1, drug2],
                    'interaction': interaction,
                    'severity': 'moderate'
                })
        
        return {
            'interactions_found': len(found_interactions),
            'interactions': found_interactions,
            'recommendation': 'Consultar con farmacéutico' if found_interactions else 'No se detectaron interacciones conocidas'
        }
    
    def get_drug_info_with_aemps_from_detected(self, detected_drugs: List[DrugInfo]) -> Dict[str, Any]:
        """
        Obtiene información completa de AEMPS para los fármacos detectados
        (Método de compatibilidad con BioChatAgent)
        
        Args:
            detected_drugs: Lista de fármacos detectados
            
        Returns:
            Información estructurada con datos AEMPS
        """
        drug_info = {
            'detected_drugs': [],
            'aemps_results': [],
            'total_detected': len(detected_drugs)
        }
        
        for drug in detected_drugs:
            # Convertir DrugInfo a formato compatible
            drug_dict = {
                'spanish_name': drug.name.lower(),
                'confidence': 'alta' if drug.confidence > 0.8 else 'media' if drug.confidence > 0.5 else 'baja',
                'type': 'principio_activo' if drug.active_ingredient else 'nombre_comercial',
                'detected_in': 'pattern_matching',
                'reason': f'Detectado como {drug.category}' if drug.category else 'Detectado por patrones'
            }
            drug_info['detected_drugs'].append(drug_dict)
            
            # Simular búsqueda AEMPS (versión simplificada)
            try:
                aemps_result = self._simulate_aemps_search(drug.name)
                if aemps_result:
                    drug_info['aemps_results'].append({
                        'drug': drug_dict,
                        'aemps_info': aemps_result
                    })
            except Exception as e:
                logger.warning(f"Error simulando búsqueda AEMPS para {drug.name}: {e}")
        
        return drug_info
    
    def _simulate_aemps_search(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """Simula búsqueda en AEMPS para compatibilidad"""
        # Crear enlace de búsqueda AEMPS
        from urllib.parse import quote
        search_url = f"https://cima.aemps.es/cima/publico/lista.html?texto={quote(drug_name)}"
        
        return {
            'drug_name': drug_name,
            'title': f'Información de {drug_name.title()} en AEMPS',
            'url': search_url,
            'source': 'AEMPS-CIMA',
            'type': 'Búsqueda'
        }

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del detector"""
        return {
            'total_patterns': len(self.drug_patterns),
            'known_drugs': len(self.common_drugs),
            'categories': list(set(drug.category for drug in self.common_drugs.values() if drug.category))
        }

# Función de conveniencia para importación
def create_drug_detector() -> DrugDetectionAEMPS:
    """Crea una instancia del detector de fármacos"""
    return DrugDetectionAEMPS()

# Función para compatibilidad con imports existentes
def initialize_drug_detector(openai_client=None) -> DrugDetectionAEMPS:
    """
    Inicializa el detector de fármacos (compatibilidad con versiones anteriores)
    
    Args:
        openai_client: Cliente OpenAI (ignorado por ahora, para compatibilidad)
    
    Returns:
        Instancia del detector de fármacos
    """
    return DrugDetectionAEMPS()

# Para compatibilidad con imports existentes
DrugDetector = DrugDetectionAEMPS 