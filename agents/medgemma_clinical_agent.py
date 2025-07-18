#!/usr/bin/env python3
"""
MedGemma Clinical Agent - Agente de Análisis Clínico Avanzado
=============================================================

Este agente utiliza MedGemma para análisis clínico especializado.
Basado en la documentación oficial de MedGemma.

Autor: Carmen Pascual
Versión: 2.0
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
import torch
from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# Función auxiliar para llamar a LLM
def _call_openai_native(client, messages, temperature=0.1, max_tokens=4000, task_description="Consultando modelo de IA"):
    """Función auxiliar para llamar a LLM de forma nativa"""
    try:
        if hasattr(client, 'invoke'):
            response = client.invoke(messages[0]["content"])
            return response
        elif hasattr(client, '__call__'):
            response = client(messages[0]["content"])
            return response
        else:
            return "Error: Cliente LLM no compatible"
    except Exception as e:
        return f"Error llamando LLM: {str(e)}"

class MockResponse:
    def __init__(self, content: str):
        self.content = content

class MedGemmaClinicalAgent:
    """
    Agente clínico especializado usando MedGemma para análisis médico avanzado.
    
    Características:
    - Análisis clínico de datos médicos
    - Validación de diagnósticos
    - Explicación de conceptos médicos
    - Generación de reportes clínicos
    - Recomendaciones de tratamiento
    """
    
    def __init__(self, model_id: str = "google/medgemma-27b-text-it", device: str = "auto", llm=None):
        """
        Inicializa el agente MedGemma con el modelo correcto.
        
        Args:
            model_id: ID del modelo MedGemma en Hugging Face
            device: Dispositivo a usar ("auto", "cuda", "cpu")
            llm: Cliente LLM para fallback dinámico
        """
        self.model_id = model_id
        self.device = device
        self.llm = llm
        self.pipe = None
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        
        print("🏥 Inicializando MedGemma Clinical Agent...")
        
        # Detectar dispositivo
        if torch.cuda.is_available():
            print("   ✅ CUDA disponible - Usando GPU")
            self.device = "cuda"
        else:
            print("   ⚠️ CUDA no disponible - Usando CPU")
            self.device = "cpu"
        
        # Inicializar modelo
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa el modelo MedGemma usando el pipeline de transformers"""
        try:
            print("   🔧 Inicializando agente clínico...")
            logger.info("[DEBUG] Iniciando inicialización del agente clínico...")
            
            # Intentar obtener token de Hugging Face si está disponible
            import os
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            
            # Intentar cargar MedGemma si está disponible
            try:
                if hf_token:
                    logger.info("[DEBUG] Intentando cargar MedGemma con token...")
                    self.pipe = pipeline(
                        "text-generation",
                        model="google/medgemma-27b-text-it",
                        token=hf_token,
                        torch_dtype=torch.bfloat16,
                        device_map="auto" if self.device == "cuda" else "cpu"
                    )
                    self.is_initialized = True
                    self.model_id = "google/medgemma-27b-text-it"
                    print("   ✅ MedGemma inicializado correctamente")
                    logger.info("[DEBUG] MedGemma inicializado correctamente")
                    return
                else:
                    logger.warning("[DEBUG] No se encontró HUGGINGFACE_TOKEN, MedGemma no disponible")
            except Exception as e:
                logger.warning(f"[DEBUG] Error cargando MedGemma: {e}")
            
            # Si MedGemma no está disponible, usar fallback con LLM
            logger.info("[DEBUG] Usando fallback con LLM")
            print("   ⚠️ MedGemma no disponible, usando LLM como fallback")
            self.is_initialized = False
            self.pipe = self._create_fallback_pipeline()
            
        except Exception as e:
            print(f"   ❌ Error crítico inicializando agente: {e}")
            logger.error(f"[DEBUG] Error crítico inicializando agente: {e}")
            self.is_initialized = False
            self.pipe = self._create_fallback_pipeline()
            logger.info("[DEBUG] Usando pipeline de fallback por error crítico")
    
    def _create_fallback_pipeline(self):
        """Crea un pipeline de fallback dinámico usando LLM"""
        logger.info("[DEBUG] Creando pipeline de fallback dinámico...")
        
        class DynamicFallbackPipeline:
            def __init__(self, llm=None):
                self.llm = llm
                logger.info("[DEBUG] DynamicFallbackPipeline inicializado")
                
            async def __call__(self, text, **kwargs):
                logger.info(f"[DEBUG] DynamicFallbackPipeline llamado con texto: {text[:100]}...")
                
                if not self.llm:
                    return [{"generated_text": "Lo siento, no puedo proporcionar información médica específica en este momento. Por favor, consulte con un profesional de la salud."}]
                
                try:
                    # Prompt dinámico para análisis médico
                    prompt = f"""Eres un asistente médico experto. Analiza esta consulta y proporciona información médica útil y precisa.

CONSULTA: "{text}"

TAREA: Proporciona información médica relevante, segura y útil.

REGLAS:
- Si es sobre medicamentos, menciona efectos adversos comunes y graves
- Si es sobre enfermedades, explica síntomas y tratamientos generales
- Si es sobre diagnósticos, proporciona información educativa
- SIEMPRE recomienda consultar con un profesional de la salud
- Sé preciso pero no hagas diagnósticos específicos
- Usa lenguaje médico apropiado pero comprensible

RESPUESTA: Proporciona información médica útil y estructurada."""

                    # Usar LLM para respuesta dinámica
                    if self.llm:
                        try:
                            llm = self.llm  # Capturar referencia local
                            response = await asyncio.to_thread(
                                lambda: llm.invoke(prompt) if hasattr(llm, 'invoke') else str(llm(prompt))
                            )
                            content = str(response)
                        except Exception as e:
                            logger.error(f"[DEBUG] Error llamando LLM: {e}")
                            content = "Lo siento, no puedo proporcionar información médica específica en este momento. Por favor, consulte con un profesional de la salud."
                    else:
                        content = "Lo siento, no puedo proporcionar información médica específica en este momento. Por favor, consulte con un profesional de la salud."
                    
                    logger.info("[DEBUG] Respuesta dinámica generada")
                    return [{"generated_text": content}]
                    
                except Exception as e:
                    logger.error(f"[DEBUG] Error en fallback dinámico: {e}")
                    return [{"generated_text": "Lo siento, no puedo proporcionar información médica específica en este momento. Por favor, consulte con un profesional de la salud."}]
        
        return DynamicFallbackPipeline(self.llm)
    
    def _extract_response_text(self, response) -> str:
        """Extrae texto de respuesta del LLM"""
        if hasattr(response, 'content'):
            return str(response.content)
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    
    def _create_medical_messages(self, prompt_type: str, **kwargs) -> List[Dict[str, str]]:
        """
        Crea mensajes médicos estructurados para MedGemma.
        
        Args:
            prompt_type: Tipo de prompt ("analysis", "validation", "explanation", "report", "treatment")
            **kwargs: Datos específicos para el prompt
            
        Returns:
            List[Dict[str, str]]: Mensajes estructurados
        """
        system_prompts = {
            "analysis": "You are an expert clinical analyst. Analyze the provided medical data and provide detailed clinical insights.",
            "validation": "You are a medical validation specialist. Validate the provided diagnosis against the symptoms and medical history.",
            "explanation": "You are a medical educator. Explain medical concepts in clear, understandable terms.",
            "report": "You are a clinical report generator. Create comprehensive medical reports based on the provided data.",
            "treatment": "You are a treatment recommendation specialist. Provide evidence-based treatment recommendations."
        }
        
        system_prompt = system_prompts.get(prompt_type, "You are a helpful medical assistant.")
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # Agregar contenido específico según el tipo
        if prompt_type == "analysis" and "medical_data" in kwargs:
            messages.append({
                "role": "user",
                "content": f"Please analyze this medical data and provide clinical insights:\n\n{kwargs['medical_data']}"
            })
        elif prompt_type == "validation" and "diagnosis" in kwargs and "symptoms" in kwargs:
            messages.append({
                "role": "user",
                "content": f"Validate this diagnosis: {kwargs['diagnosis']}\n\nSymptoms: {kwargs['symptoms']}\n\nMedical history: {kwargs.get('medical_history', 'Not provided')}"
            })
        elif prompt_type == "explanation" and "concept" in kwargs:
            messages.append({
                "role": "user",
                "content": f"Explain this medical concept: {kwargs['concept']}"
            })
        elif prompt_type == "report" and "patient_data" in kwargs and "medical_results" in kwargs:
            messages.append({
                "role": "user",
                "content": f"Generate a clinical report based on:\n\nPatient Data: {kwargs['patient_data']}\n\nMedical Results: {kwargs['medical_results']}"
            })
        elif prompt_type == "treatment" and "diagnosis" in kwargs and "medical_history" in kwargs:
            messages.append({
                "role": "user",
                "content": f"Provide treatment recommendations for:\n\nDiagnosis: {kwargs['diagnosis']}\n\nMedical History: {kwargs['medical_history']}"
            })
        
        return messages
    
    async def analyze_clinical_data(self, medical_data: str, stream_callback=None) -> Dict[str, Any]:
        """
        Analiza datos clínicos usando MedGemma.
        
        Args:
            medical_data: Datos médicos a analizar
            stream_callback: Función para mostrar progreso
            
        Returns:
            Dict[str, Any]: Análisis clínico
        """
        if not self.is_initialized:
            return {
                "success": False,
                "error": "MedGemma no está inicializado",
                "analysis": "Análisis no disponible - MedGemma no inicializado"
            }
        
        try:
            if stream_callback:
                stream_callback("   🧠 Analizando datos clínicos con MedGemma...")
            
            # Crear mensajes para análisis clínico
            messages = self._create_medical_messages("analysis", medical_data=medical_data)
            
            # Usar pipeline para generación
            if not self.pipe:
                return {
                    "success": False,
                    "error": "Pipeline no inicializado",
                    "analysis": "Análisis no disponible - Pipeline no inicializado"
                }
            
            prompt_text = messages[-1]["content"] if messages else ""
            output = self.pipe(
                prompt_text, 
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7
            )
            
            # Extraer el contenido generado
            if output and isinstance(output, list) and len(output) > 0:
                generated_text = output[0]["generated_text"]
                response = str(generated_text)
                
                if stream_callback:
                    stream_callback("   ✅ Análisis clínico completado")
                
                return {
                    "success": True,
                    "analysis": response,
                    "model": "MedGemma",
                    "tokens_generated": len(response.split())
                }
            else:
                return {
                    "success": False,
                    "error": "No se generó respuesta",
                    "analysis": "No se pudo generar análisis clínico"
                }
                
        except Exception as e:
            logger.error(f"Error en análisis clínico: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis": f"Error en análisis: {str(e)}"
            }
    
    async def validate_diagnosis(self, diagnosis: str, symptoms: str, stream_callback=None) -> Dict[str, Any]:
        """
        Valida un diagnóstico usando MedGemma.
        
        Args:
            diagnosis: Diagnóstico a validar
            symptoms: Síntomas del paciente
            stream_callback: Función para mostrar progreso
            
        Returns:
            Dict[str, Any]: Validación del diagnóstico
        """
        if not self.is_initialized:
            return {
                "success": False,
                "error": "MedGemma no está inicializado",
                "validation": "Validación no disponible"
            }
        
        try:
            if stream_callback:
                stream_callback("   🔍 Validando diagnóstico con MedGemma...")
            
            messages = self._create_medical_messages(
                "validation", 
                diagnosis=diagnosis, 
                symptoms=symptoms
            )
            
            if not self.pipe:
                return {
                    "success": False,
                    "error": "Pipeline no inicializado",
                    "validation": "Validación no disponible"
                }
            
            prompt_text = messages[-1]["content"] if messages else ""
            output = self.pipe(
                prompt_text, 
                max_new_tokens=250,
                do_sample=True,
                temperature=0.6
            )
            
            if output and isinstance(output, list) and len(output) > 0:
                generated_text = output[0]["generated_text"]
                response = str(generated_text)
                
                if stream_callback:
                    stream_callback("   ✅ Validación completada")
                
                return {
                    "success": True,
                    "validation": response,
                    "diagnosis": diagnosis,
                    "symptoms": symptoms,
                    "model": "MedGemma"
                }
            else:
                return {
                    "success": False,
                    "error": "No se generó validación",
                    "validation": "No se pudo validar el diagnóstico"
                }
                
        except Exception as e:
            logger.error(f"Error en validación de diagnóstico: {e}")
            return {
                "success": False,
                "error": str(e),
                "validation": f"Error en validación: {str(e)}"
            }
    
    async def explain_medical_concept(self, concept: str, stream_callback=None) -> Dict[str, Any]:
        """
        Explica un concepto médico usando MedGemma.
        
        Args:
            concept: Concepto médico a explicar
            stream_callback: Función para mostrar progreso
            
        Returns:
            Dict[str, Any]: Explicación del concepto
        """
        if not self.is_initialized:
            return {
                "success": False,
                "error": "MedGemma no está inicializado",
                "explanation": "Explicación no disponible"
            }
        
        try:
            if stream_callback:
                stream_callback("   📚 Explicando concepto médico con MedGemma...")
            
            messages = self._create_medical_messages("explanation", concept=concept)
            
            if not self.pipe:
                return {
                    "success": False,
                    "error": "Pipeline no inicializado",
                    "explanation": "Explicación no disponible"
                }
            
            prompt_text = messages[-1]["content"] if messages else ""
            output = self.pipe(
                prompt_text, 
                max_new_tokens=200,
                do_sample=True,
                temperature=0.8
            )
            
            if output and isinstance(output, list) and len(output) > 0:
                generated_text = output[0]["generated_text"]
                response = str(generated_text)
                
                if stream_callback:
                    stream_callback("   ✅ Explicación completada")
                
                return {
                    "success": True,
                    "explanation": response,
                    "concept": concept,
                    "model": "MedGemma"
                }
            else:
                return {
                    "success": False,
                    "error": "No se generó explicación",
                    "explanation": "No se pudo explicar el concepto"
                }
                
        except Exception as e:
            logger.error(f"Error explicando concepto médico: {e}")
            return {
                "success": False,
                "error": str(e),
                "explanation": f"Error en explicación: {str(e)}"
            }
    
    async def generate_clinical_report(self, patient_data: str, medical_results: str, stream_callback=None) -> Dict[str, Any]:
        """
        Genera un reporte clínico usando MedGemma.
        
        Args:
            patient_data: Datos del paciente
            medical_results: Resultados médicos
            stream_callback: Función para mostrar progreso
            
        Returns:
            Dict[str, Any]: Reporte clínico generado
        """
        if not self.is_initialized:
            return {
                "success": False,
                "error": "MedGemma no está inicializado",
                "report": "Reporte no disponible"
            }
        
        try:
            if stream_callback:
                stream_callback("   📋 Generando reporte clínico con MedGemma...")
            
            messages = self._create_medical_messages(
                "report", 
                patient_data=patient_data, 
                medical_results=medical_results
            )
            
            if not self.pipe:
                return {
                    "success": False,
                    "error": "Pipeline no inicializado",
                    "report": "Reporte no disponible"
                }
            
            prompt_text = messages[-1]["content"] if messages else ""
            output = self.pipe(
                prompt_text, 
                max_new_tokens=400,
                do_sample=True,
                temperature=0.7
            )
            
            if output and isinstance(output, list) and len(output) > 0:
                generated_text = output[0]["generated_text"]
                response = str(generated_text)
                
                if stream_callback:
                    stream_callback("   ✅ Reporte clínico generado")
                
                return {
                    "success": True,
                    "report": response,
                    "patient_data": patient_data,
                    "medical_results": medical_results,
                    "model": "MedGemma"
                }
            else:
                return {
                    "success": False,
                    "error": "No se generó reporte",
                    "report": "No se pudo generar el reporte clínico"
                }
                
        except Exception as e:
            logger.error(f"Error generando reporte clínico: {e}")
            return {
                "success": False,
                "error": str(e),
                "report": f"Error generando reporte: {str(e)}"
            }
    
    async def recommend_treatment(self, diagnosis: str, medical_history: str, stream_callback=None) -> Dict[str, Any]:
        """
        Recomienda tratamiento usando MedGemma.
        
        Args:
            diagnosis: Diagnóstico del paciente
            medical_history: Historia médica
            stream_callback: Función para mostrar progreso
            
        Returns:
            Dict[str, Any]: Recomendaciones de tratamiento
        """
        if not self.is_initialized:
            return {
                "success": False,
                "error": "MedGemma no está inicializado",
                "recommendations": "Recomendaciones no disponibles"
            }
        
        try:
            if stream_callback:
                stream_callback("   💊 Generando recomendaciones de tratamiento con MedGemma...")
            
            messages = self._create_medical_messages(
                "treatment", 
                diagnosis=diagnosis, 
                medical_history=medical_history
            )
            
            if not self.pipe:
                return {
                    "success": False,
                    "error": "Pipeline no inicializado",
                    "recommendations": "Recomendaciones no disponibles"
                }
            
            prompt_text = messages[-1]["content"] if messages else ""
            output = self.pipe(
                prompt_text, 
                max_new_tokens=350,
                do_sample=True,
                temperature=0.6
            )
            
            if output and isinstance(output, list) and len(output) > 0:
                generated_text = output[0]["generated_text"]
                response = str(generated_text)
                
                if stream_callback:
                    stream_callback("   ✅ Recomendaciones generadas")
                
                return {
                    "success": True,
                    "recommendations": response,
                    "diagnosis": diagnosis,
                    "medical_history": medical_history,
                    "model": "MedGemma"
                }
            else:
                return {
                    "success": False,
                    "error": "No se generaron recomendaciones",
                    "recommendations": "No se pudieron generar recomendaciones"
                }
                
        except Exception as e:
            logger.error(f"Error generando recomendaciones: {e}")
            return {
                "success": False,
                "error": str(e),
                "recommendations": f"Error generando recomendaciones: {str(e)}"
            }
    
    async def process_query(self, query: str, stream_callback=None) -> Dict[str, Any]:
        """
        Procesa consultas médicas con análisis clínico avanzado y streaming en tiempo real.
        """
        try:
            if not self.llm:
                return {
                    "success": False,
                    "message": "❌ LLM no disponible para análisis clínico",
                    "model": "Sin LLM"
                }
            
            # Análisis de la consulta médica
            if stream_callback:
                stream_callback("🧠 Analizando consulta médica...")
            
            # Generar respuesta con streaming
            response_text = await self._generate_streaming_response(query, stream_callback)
            
            return {
                "success": True,
                "response": response_text,
                "query": query,
                "model": "LLM Médico Especializado"
            }
            
        except Exception as e:
            logger.error(f"Error en análisis clínico: {e}")
            return {
                "success": False,
                "message": f"❌ Error en análisis clínico: {str(e)}",
                "model": "Error"
            }

    async def _generate_streaming_response(self, query: str, stream_callback=None) -> str:
        """
        Genera respuesta con streaming en tiempo real.
        """
        try:
            if not self.llm:
                return "❌ LLM no disponible para análisis clínico"
            
            # Prompt para análisis clínico
            prompt = f"""Eres un médico experto que proporciona información clínica precisa y útil.

CONSULTA DEL PACIENTE: "{query}"

INSTRUCCIONES:
1. Proporciona información médica clara y precisa
2. Incluye dosis recomendadas cuando sea apropiado
3. Menciona efectos secundarios importantes
4. Advierte sobre cuándo consultar al médico
5. Usa lenguaje comprensible para el paciente
6. Sé específico y práctico en las recomendaciones

FORMATO DE RESPUESTA:
- Información clara y estructurada
- Dosis y frecuencia cuando aplique
- Efectos secundarios importantes
- Cuándo buscar atención médica
- Precauciones alimentarias si aplica

Responde de manera profesional y útil:"""

            if stream_callback:
                stream_callback("🤖 Generando respuesta médica...")
            
            # Usar streaming real del LLM
            response = await self.llm.ainvoke(prompt)
            
            # Extraer el texto de la respuesta
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            if stream_callback:
                stream_callback("✅ Análisis clínico completado")
            
            return response_text.strip()
            
        except Exception as e:
            logger.error(f"Error generando respuesta con streaming: {e}")
            return f"❌ Error generando respuesta: {str(e)}"

async def test_medgemma_agent():
    """Función de prueba para el agente MedGemma"""
    print("🧪 Probando agente MedGemma...")
    
    agent = MedGemmaClinicalAgent()
    
    if agent.is_initialized:
        print("✅ MedGemma inicializado correctamente")
        
        # Prueba de análisis clínico
        test_data = """
        Paciente masculino, 52 años
        Diagnóstico: Hipertensión arterial
        Presión arterial: 150/95 mmHg
        Medicación: Enalapril 10mg/día
        """
        
        result = await agent.analyze_clinical_data(test_data)
        print(f"📊 Resultado del análisis: {result}")
        
    else:
        print("❌ MedGemma no se pudo inicializar")

if __name__ == "__main__":
    asyncio.run(test_medgemma_agent()) 