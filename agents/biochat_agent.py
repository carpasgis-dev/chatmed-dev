#!/usr/bin/env python3
"""
BioChatAgent - Agente de Búsqueda Biomédica Multi-Herramienta de Nueva Generación
================================================================================

Arquitectura basada en "Agentes como Herramientas" (Tools as Agents). Un LLM planificador
decide qué herramientas especializadas (PubMed, Google Scholar, GenBank, etc.) usar
y las ejecuta en paralelo para una búsqueda rápida, inteligente y transparente.
"""

import asyncio
import logging
import json
import re
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, TypedDict, cast
from langchain.llms.base import BaseLLM
from duckduckgo_search import DDGS
from Bio import Entrez
from langchain.schema import AIMessage, BaseMessage
import aiohttp
from datetime import datetime

# Tipos personalizados para resultados de Entrez
class EntrezSearchResult(TypedDict):
    """Tipo personalizado para resultados de búsqueda de Entrez."""
    IdList: List[str]
    RetMax: str
    RetStart: str
    QueryKey: str
    WebEnv: str
    Count: str
    TranslationSet: List[Dict[str, Any]]
    QueryTranslation: str

class EntrezRecord(TypedDict, total=False):
    """Tipo personalizado para registros de Entrez."""
    TI: str  # Título
    PMID: str  # ID de PubMed
    AU: List[str]  # Autores
    TA: str  # Título abreviado del journal
    DP: str  # Fecha de publicación
    AB: str  # Abstract
    MH: List[str]  # Términos MeSH

class GenBankSearchResult(TypedDict):
    """Tipo personalizado para resultados de búsqueda de GenBank."""
    IdList: List[str]
    RetMax: str
    RetStart: str
    QueryKey: str
    WebEnv: str
    Count: str
    TranslationSet: List[Dict[str, Any]]

# Configuración de logging
logging.basicConfig(level="INFO", format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuración de NCBI
Entrez.email = "carpasgis99@gmail.com"
Entrez.api_key = "c404a87e96b8328d6bb5d34da565c1c7a308"
logger.info(f"✅ Configuración NCBI cargada: {Entrez.email}")

class BioChatAgent:
    """
    Agente de investigación biomédica que utiliza múltiples fuentes especializadas.
    """
    
    def __init__(self, llm: Optional[BaseLLM] = None, medgemma_agent=None):
        if not llm:
            raise ValueError("Se requiere una instancia de un modelo de lenguaje (LLM).")
        self.llm = llm
        self.medgemma_agent = medgemma_agent
        self.search_tool = DDGS(proxy=None)
        
        # --- GESTOR DE MEMORIA CONVERSACIONAL ---
        self.last_search_context = None
        self.last_query = None
        
        # Configuración de APIs
        self.semantic_scholar_api = "https://api.semanticscholar.org/v1/"
        self.europe_pmc_api = "https://www.ebi.ac.uk/europepmc/webservices/rest/"
        self.clinical_trials_api = "https://clinicaltrials.gov/api/"
        
        logger.info("✅ BioChatAgent (Multi-Source) inicializado.")

    def _extract_response_text(self, response: Union[str, BaseMessage, Any]) -> str:
        """Extrae el texto de una respuesta del LLM de forma segura."""
        try:
            if isinstance(response, str):
                return response
            if isinstance(response, (AIMessage, BaseMessage)):
                return str(response.content)
            if hasattr(response, 'content'):
                return str(response.content)
            # Si es una lista o diccionario, convertirlo a JSON
            if isinstance(response, (list, dict)):
                return json.dumps(response, ensure_ascii=False)
            return str(response)
        except Exception as e:
            logger.error(f"Error extrayendo texto de respuesta: {e}")
            return str(response)

    async def process_query(self, query: str, stream_callback=None) -> Dict[str, Any]:
        """Procesa una consulta, decidiendo si es una nueva búsqueda o una pregunta de seguimiento."""
        self.stream_callback = stream_callback or (lambda msg: None)
        
        # --- CLASIFICADOR DE INTENCIÓN ---
        if self.last_search_context:
            is_follow_up = await self._is_follow_up_question(query)
            if is_follow_up:
                logger.info("❓ Detectada pregunta de seguimiento. Usando memoria...")
                return await self._answer_follow_up_question(query)

        logger.info("🚀 Detectada nueva búsqueda. Iniciando proceso completo...")
        return await self._execute_new_search(query)

    async def _is_follow_up_question(self, query: str) -> bool:
        """Determina si una consulta es una pregunta de seguimiento sobre la búsqueda anterior."""
        if not self.last_query:
            return False

        prompt = f"""
        Analiza si la "Nueva Pregunta" es una pregunta de seguimiento sobre los "Resultados Anteriores".
        Una pregunta de seguimiento busca aclarar, resumir o pedir más detalles sobre la información ya presentada. No inicia un tema completamente nuevo.

        Búsqueda Anterior: "{self.last_query}"
        Resultados Anteriores: {json.dumps(self.last_search_context, indent=2, ensure_ascii=False, default=str, skipkeys=True)}
        
        Nueva Pregunta: "{query}"

        Ejemplos de seguimiento: "dime más sobre el primer artículo", "¿cuál es el NCT ID del tercer ensayo?", "resume el abstract del estudio de Patel".
        Ejemplos de NO seguimiento: "¿y sobre la diabetes?", "buscar guías de HTA".

        Responde solo con "true" o "false".
        """
        try:
            response = await self.llm.ainvoke(prompt, temperature=0.0)
            answer = self._extract_response_text(response).lower().strip()
            return "true" in answer
        except Exception as e:
            logger.error(f"Error clasificando intención de seguimiento: {e}")
            return False

    async def _answer_follow_up_question(self, query: str) -> Dict[str, Any]:
        """Responde una pregunta de seguimiento usando el contexto en memoria."""
        if not self.last_search_context:
            return {
                "success": False,
                "message": "No tengo un contexto de búsqueda anterior para responder a tu pregunta."
            }

        # Reconstruir el contexto para el LLM
        context = "--- CONTEXTO DE LA BÚSQUEDA ANTERIOR ---\\n"
        for source, data in self.last_search_context.items():
            context += f"\\n## Resultados de {source}:\\n"
            if isinstance(data, dict):
                results = data.get('results', [])
                if results:
                    for i, res in enumerate(results, 1):
                        context += f"\\n--- Evidencia {source} #{i} ---\\n"
                        context += f"ID: {res.get('id', res.get('pmid', res.get('nctId', 'N/A')))}\\n"
                        context += f"Título: {res.get('title', 'N/A')}\\n"
                        context += f"Resumen: {res.get('abstract', 'No disponible.')}\\n"
        
        prompt = f"""
        Eres un asistente de investigación. Responde la "Pregunta de Seguimiento" basándote únicamente en el "Contexto de la Búsqueda Anterior".

        {context}

        Pregunta de Seguimiento: "{query}"

        Tu Respuesta:
        """
        try:
            response = await self.llm.ainvoke(prompt, temperature=0.1)
            answer = self._extract_response_text(response)
            return {"success": True, "message": answer}
        except Exception as e:
            logger.error(f"Error respondiendo pregunta de seguimiento: {e}")
            return {"success": False, "message": "Lo siento, tuve un problema al responder tu pregunta."}
            
    async def _execute_new_search(self, query: str, is_follow_up: bool = False) -> Dict[str, Any]:
        """Ejecuta el flujo completo para una nueva consulta de búsqueda, o refina una existente."""
        try:
            # Si es una pregunta de seguimiento, la planificación es más simple
            if is_follow_up:
                # Reutilizar y refinar la búsqueda en PubMed. Para otras fuentes, se puede decidir no volver a buscar.
                tool_calls = [{"tool_name": "search_pubmed", "query": await self._build_pubmed_query(query, is_follow_up=True)}]
            else:
                # Proceso completo para una nueva búsqueda
                tool_calls = await self._decide_tools_to_use(query)
            
            if not tool_calls:
                self.stream_callback("⚠️ No se pudo determinar un plan de búsqueda.")
                return {"success": False, "message": "No se pudo procesar la consulta."}

            # 2. Ejecutar búsquedas en paralelo
            self.stream_callback(f"⚡ Ejecutando {len(tool_calls)} búsquedas especializadas en paralelo...")
            
            tasks = []
            for call in tool_calls:
                tool_name = call.get("tool_name")
                tool_query = call.get("query")
                
                if not tool_name or not tool_query:
                    logger.warning(f"Herramienta inválida: {call}")
                    continue
                        
                tool_methods = {
                    "search_pubmed": self._tool_search_pubmed,
                    "search_semantic_scholar": self._tool_search_semantic_scholar,
                    "search_clinical_trials": self._tool_search_clinical_trials,
                    "search_europe_pmc": self._tool_search_europe_pmc,
                    "search_genbank": self._tool_search_genbank,
                    "search_aemps": self._tool_search_aemps
                }
                
                if tool_name in tool_methods:
                    logger.info(f"Añadiendo tarea: {tool_name} con query: {tool_query[:50]}...")
                    tasks.append(tool_methods[tool_name](tool_query))
                else:
                    logger.warning(f"Herramienta no reconocida: {tool_name}")
            
            if not tasks:
                logger.error(f"No se crearon tareas. tool_calls: {tool_calls}")
                # Fallback: usar búsqueda web básica
                self.stream_callback("⚠️ Usando búsqueda web como fallback...")
                fallback_result = await self._tool_search_web(query)
                if fallback_result.get("results"):
                    # Convertir al formato esperado por _generate_expert_synthesis
                    fallback_context = {"Web": fallback_result.get("results", [])}
                    synthesis = await self._generate_expert_synthesis(query, fallback_context)
                    return {"success": True, "message": synthesis}
                return {"success": False, "message": "No se pudo ejecutar ninguna búsqueda."}

            search_results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 3. Sintetizar resultados
            self.stream_callback("🔬 Consolidando resultados de múltiples fuentes...")
            
            final_context = {}
            for result in search_results_list:
                if isinstance(result, dict) and result.get("source"):
                    final_context[result["source"]] = result
                elif isinstance(result, Exception):
                    logger.error(f"Error en búsqueda: {result}")

            if not final_context:
                # Limpiar memoria si no hay resultados
                self.last_search_context = None
                logger.warning("No se obtuvieron resultados de ninguna fuente.")
                return {
                    "success": False,
                    "message": "No se encontraron resultados para tu consulta en ninguna de las fuentes de datos biomédicas consultadas."
                }

            synthesis = await self._generate_expert_synthesis(query, final_context)
            
            # --- GUARDAR EN MEMORIA ---
            self.last_search_context = final_context
            self.last_query = query
            logger.info("✅ Contexto de búsqueda guardado en memoria.")
            
            self.stream_callback("✅ Análisis completado.")
            return {"success": True, "message": synthesis}
            
        except Exception as e:
            logger.error(f"Error en proceso principal: {e}", exc_info=True)
            return {"success": False, "message": f"Error procesando la consulta: {e}"}

    async def _decide_tools_to_use(self, main_query: str) -> List[Dict[str, str]]:
        """Usa el LLM para decidir qué fuentes de búsqueda son las más apropiadas, enriquecido con detección de entidades."""
        
        # PASO 1: Detección de entidades biomédicas
        entities = await self._detect_biomedical_entities(main_query)
        
        # PASO 2: Decisión de herramientas principales con el LLM
        prompt = f"""Eres un experto en investigación biomédica. Analiza esta consulta y decide qué fuentes usar para obtener la información más completa y relevante.

**Pregunta del Usuario:** "{main_query}"

**Fuentes Disponibles (Usa ESTOS nombres de herramienta):**

1. `search_pubmed`: Para literatura médica primaria, ensayos clínicos, evidencia científica, estudios clínicos.
2. `search_semantic_scholar`: Para literatura académica con análisis de impacto, papers muy citados, tendencias recientes.
3. `search_clinical_trials`: Para buscar tratamientos en investigación, estudios en curso y completados.
4. `search_europe_pmc`: Para literatura biomédica europea y preprints.
5. `search_genbank`: Para datos genéticos, secuencias, variantes genéticas, si hay términos genéticos específicos.
6. `search_aemps`: Para buscar información de medicamentos en la AEMPS (Agencia Española de Medicamentos).

**Instrucciones:**
1. Analiza la pregunta cuidadosamente
2. Selecciona TODAS las fuentes relevantes (mínimo 2 si es posible)
3. Para cada fuente, genera una consulta optimizada
4. Prioriza la combinación de fuentes para obtener una visión completa

**Formato de Respuesta:**
[
  {{
    "tool_name": "nombre_fuente",
    "query": "consulta optimizada en inglés"
  }},
  ...
]

**Ejemplos de Buenas Combinaciones:**
- Para preguntas sobre tratamientos: PubMed + ClinicalTrials.gov
- Para investigación genética: PubMed + GenBank
- Para temas emergentes: Semantic Scholar + Europe PMC
- Para revisiones completas: PubMed + Semantic Scholar + ClinicalTrials.gov

**Tu Respuesta (solo el JSON):**"""

        try:
            response = await self.llm.ainvoke(prompt)
            response_text = self._extract_response_text(response)
            cleaned_response = response_text.strip().replace("```json", "").replace("```", "").strip()
            logger.info(f"Respuesta LLM para decisión de herramientas: {cleaned_response[:200]}...")
            
            try:
                tool_calls = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                logger.warning(f"Error al parsear JSON de decisión de herramientas: {e}. Contenido: {cleaned_response}")
                tool_calls = []

            # PASO 3: Enriquecer con búsquedas especializadas basadas en entidades
            if entities.get("genes"):
                for gene in entities["genes"]:
                    logger.info(f"🧬 Añadiendo búsqueda en GenBank para el gen: {gene}")
                    tool_calls.append({"tool_name": "search_genbank", "query": f"{gene}[Gene Name]"})

            if entities.get("drugs"):
                for drug in entities["drugs"]:
                    logger.info(f"💊 Añadiendo búsqueda en AEMPS para el fármaco: {drug}")
                    tool_calls.append({"tool_name": "search_aemps", "query": drug})

            # Asegurar que haya al menos una búsqueda si la detección falla pero las entidades existen
            if not tool_calls and (entities.get("genes") or entities.get("drugs") or entities.get("diseases")):
                tool_calls.append({"tool_name": "search_pubmed", "query": main_query})

            logger.info(f"Herramientas finales seleccionadas: {[call.get('tool_name') for call in tool_calls]}")
            return tool_calls
                
        except Exception as e:
            logger.error(f"Error en _decide_tools_to_use: {e}")
            return [{"tool_name": "search_pubmed", "query": main_query}]

    # --- Implementaciones de Herramientas de Búsqueda ---

    async def _tool_search_web(self, query: str) -> Dict[str, Any]:
        """Búsqueda web general como fallback."""
        self.stream_callback(f"   • 🌐 Buscando en la web: '{query[:30]}...'")
        results = await self._execute_ddg_search(query)
        return {"source": "Web", "results": results}

    async def _tool_search_google_scholar(self, query: str) -> Dict[str, Any]:
        """Busca en Google Scholar."""
        self.stream_callback(f"   • 🎓 Buscando en Google Scholar: '{query[:30]}...'")
        scholar_query = f"{query} site:scholar.google.com"
        results = await self._execute_ddg_search(scholar_query, 7)
        return {"source": "Google Scholar", "results": results}

    async def _tool_search_clinical_guidelines(self, query: str) -> Dict[str, Any]:
        """Busca guías de práctica clínica."""
        self.stream_callback(f"   • 📖 Buscando Guías Clínicas: '{query[:30]}...'")
        guidelines_query = f'{query} clinical practice guidelines OR treatment recommendations'
        results = await self._execute_ddg_search(guidelines_query, 7)
        return {"source": "Guías Clínicas", "results": results}

    async def _build_pubmed_query(self, query: str, is_follow_up: bool = False) -> str:
        """
        Construye una consulta PubMed experta y contextual.
        Si es una pregunta de seguimiento, refina la consulta anterior.
        """
        if is_follow_up and self.last_search_context and self.last_search_context.get("PubMed", {}).get("entrez_query"):
            base_query = self.last_search_context["PubMed"]["entrez_query"]
            instruction = f"Refina o expande la siguiente consulta Entrez existente para responder a la nueva pregunta."
            context_query = f"CONSULTA ENTREZ ANTERIOR:\n{base_query}\n\nNUEVA PREGUNTA DEL USUARIO:\n'{query}'"
        else:
            instruction = "Convierte la pregunta del usuario en una consulta Entrez óptima y experta."
            context_query = f"PREGUNTA DEL USUARIO:\n'{query}'"

        prompt = f"""Eres un bibliotecario médico experto en la sintaxis de PubMed Entrez. {instruction}

{context_query}

**PROCESO DE RAZONAMIENTO EXPERTO (que debes seguir):**
1.  **Deconstruye la Pregunta**: Identifica los conceptos médicos clave (genes, fármacos, enfermedades, tipo de estudio).
2.  **Mapeo a MeSH**: Para cada concepto, encuentra el término MeSH más preciso.
3.  **Construcción de Cláusulas Robustas**: Combina el término MeSH con una búsqueda de texto libre en Título/Abstract para máxima cobertura. Sintaxis: `("Termino"[MeSH Terms] OR "Termino"[Title/Abstract])`
4.  **Ensamblaje Booleano**: Une las cláusulas con `AND`.
5.  **Filtros**: Si aplica, añade filtros de tipo de publicación (ej: `Clinical Trial[ptyp]`, `Review[ptyp]`, `Guideline[ptyp]`).

**GENERA ÚNICAMENTE LA CONSULTA ENTREZ FINAL. NO INCLUYAS EXPLICACIONES.**
"""
        try:
            response = await self.llm.ainvoke(prompt, temperature=0.0)
            query_text = self._extract_response_text(response).strip().replace('\n', ' ')
            logger.info(f"PubMed Query Construida: {query_text}")
            return query_text
        except Exception as e:
            logger.error(f"Error construyendo la query de PubMed: {e}")
            return query # Fallback a la query original

    async def _build_genbank_query(self, query: str) -> str:
        """Construye una consulta GenBank usando sintaxis Entrez correcta."""
        prompt = f"""Construye una consulta GenBank usando la sintaxis correcta de Entrez.

CONSULTA ORIGINAL: "{query}"

INSTRUCCIONES:
1. Usa etiquetas de campo específicas de GenBank:
   - [ACCN] para números de acceso
   - [ORGN] para organismos
   - [TI] para título/definición
   - [SLEN] para longitud de secuencia
   - [PROP] para propiedades
2. Usa operadores booleanos cuando sea necesario
3. Incluye rangos numéricos si aplica (ej: 1000:2000[SLEN])

EJEMPLO DE SALIDA:
"Homo sapiens"[ORGN] AND BRCA1[TI] AND "complete cds"[PROP]

GENERA SOLO LA CONSULTA:"""
        
        response = await self.llm.ainvoke(prompt)
        query_text = self._extract_response_text(response)
        return query_text.strip().strip('"\'')

    async def _tool_search_pubmed(self, query: str) -> Dict[str, Any]:
        """Busca en PubMed usando la API de Entrez con sintaxis correcta."""
        try:
            self.stream_callback(f"   • 📚 Buscando en PubMed: '{query}'")
            
            # Construir consulta Entrez
            entrez_query = await self._build_pubmed_query(query)
            self.stream_callback(f"      Query Entrez: {entrez_query}")
            
            loop = asyncio.get_running_loop()
            search_handle = await loop.run_in_executor(
                None, 
                lambda: Entrez.esearch(
                    db="pubmed",
                    term=entrez_query,
                    retmax="10",  # Aumentado para más resultados
                    usehistory="y"
                )
            )
            
            # Convertir el resultado a nuestro tipo personalizado
            search_result = cast(EntrezSearchResult, Entrez.read(search_handle))
            search_handle.close()
            
            pmids = search_result['IdList']
            if not pmids:
                return {"source": "PubMed", "results": []}
            
            # Obtener detalles de los artículos
            fetch_handle = await loop.run_in_executor(
                None,
                lambda: Entrez.efetch(
                    db="pubmed",
                    id=pmids,
                    rettype="medline",
                    retmode="text",
                    webenv=search_result.get('WebEnv'),
                    query_key=search_result.get('QueryKey')
                )
            )
            
            records_str = fetch_handle.read()
            fetch_handle.close()
            
            from Bio import Medline
            records = list(Medline.parse(records_str.splitlines()))
            
            articles = []
            for record in records:
                record = cast(EntrezRecord, record)
                if 'TI' in record and 'PMID' in record:
                    article = {
                        "title": record['TI'],
                        "authors": record.get('AU', ["N/A"]),
                        "journal": record.get('TA', "N/A"),
                        "year": record.get('DP', "N/A").split(" ")[0] if 'DP' in record else "N/A",
                        "abstract": record.get('AB', "Sin resumen disponible."),
                        "mesh_terms": record.get('MH', []),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{record['PMID']}/",
                        "pmid": record['PMID']
                    }
                    # No truncar el abstract
                    if 'AB' in record:
                        article["snippet"] = record['AB']
                    articles.append(article)
            
            # Incluir la query Entrez en los resultados para transparencia
            return {
                "source": "PubMed", 
                "results": articles,
                "entrez_query": entrez_query,
                "total_found": len(pmids)
            }
            
        except Exception as e:
            logger.error(f"Error durante la búsqueda en PubMed: {e}")
            return {"source": "PubMed", "results": [], "error": str(e)}

    async def _tool_search_genbank(self, query: str) -> Dict[str, Any]:
        """Busca en GenBank y formatea los resultados de manera similar a PubMed."""
        try:
            self.stream_callback(f"   • 🧬 Buscando en GenBank: '{query}'")
            entrez_query = await self._build_genbank_query(query)
            
            loop = asyncio.get_running_loop()
            search_handle = await loop.run_in_executor(None, lambda: Entrez.esearch(db="nuccore", term=entrez_query, retmax="3", usehistory="y"))
            search_result = cast(GenBankSearchResult, Entrez.read(search_handle))
            search_handle.close()
            
            ids = search_result['IdList']
            if not ids: return {"source": "GenBank", "results": []}

            fetch_handle = await loop.run_in_executor(None, lambda: Entrez.efetch(db="nuccore", id=ids, rettype="gb", retmode="text"))
            records_text = fetch_handle.read()
            fetch_handle.close()

            results = []
            for record_text in records_text.split("//\n"):
                if "ACCESSION" in record_text:
                    accession = re.search(r"ACCESSION\s+([\w\d_.-]+)", record_text)
                    definition = re.search(r"DEFINITION\s+((?:.|\n(?!SOURCE))*)", record_text)
                    organism = re.search(r"ORGANISM\s+(.+)", record_text)
                    journal = re.search(r"JOURNAL\s+((?:.|\n(?!REFERENCE))*)", record_text)
                    
                    if accession:
                        # Extraer primer autor y año si está disponible en la referencia
                        authors = ["N/A"]
                        year = "N/A"
                        if journal:
                            author_match = re.search(r"AUTHORS\s+(.+)", journal.group(1))
                            if author_match:
                                authors = author_match.group(1).split(', ')
                            
                            year_match = re.search(r"JOURNAL\s+.*?\((\d{4})\)", journal.group(1))
                            if year_match:
                                year = year_match.group(1)

                        results.append({
                            "id": accession.group(1),
                            "title": definition.group(1).replace('\\n', ' ').strip() if definition else "N/A",
                            "authors": authors,
                            "year": year,
                            "journal": "GenBank",
                            "url": f"https://www.ncbi.nlm.nih.gov/nuccore/{accession.group(1)}",
                            "abstract": f"Secuencia de {organism.group(1).strip() if organism else 'N/A'}."
                        })
            
            return {"source": "GenBank", "results": results}
            
        except Exception as e:
            logger.error(f"Error durante la búsqueda en GenBank: {e}")
            return {"source": "GenBank", "results": [], "error": str(e)}

    async def _execute_ddg_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Función auxiliar para ejecutar búsquedas web con DuckDuckGo."""
        try:
            results = await asyncio.to_thread(self.search_tool.text, query, max_results=max_results)
            return [{"url": r["href"], "title": r["title"], "snippet": r["body"]} for r in results]
        except Exception as e:
            logger.error(f"Error durante la búsqueda web de '{query}': {e}")
            return []

    async def _generate_expert_synthesis(self, original_query: str, results_dict: Dict[str, List[Dict]]) -> str:
        """Genera un resumen experto a partir de los resultados de múltiples fuentes usando MedGemma cuando está disponible."""
        
        # USAR MEDGEMMA SI ESTÁ DISPONIBLE
        if self.medgemma_agent:
            try:
                # Preparar datos para MedGemma
                medical_data = {
                    'query': original_query,
                    'results': results_dict,
                    'context': 'Resultados de búsqueda biomédica multi-fuente'
                }
                
                # Analizar con MedGemma
                medgemma_result = await self.medgemma_agent.analyze_clinical_data(
                    json.dumps(medical_data, indent=2, ensure_ascii=False),
                    self.stream_callback
                )
                
                if medgemma_result and medgemma_result.get('success'):
                    synthesis = medgemma_result.get('analysis', '')
                    if synthesis:
                        return synthesis
                        
            except Exception as e:
                logger.error(f"Error con MedGemma en BioChat: {e}")
                # Continuar con LLM si MedGemma falla
        
        # FALLBACK A LLM
        context = f"Pregunta Original: \"{original_query}\"\n\n--- EVIDENCIA ENCONTRADA ---\n"
        bibliography = "\n### Bibliografía\n"
        
        for source, data in results_dict.items():
            context += f"\n## Resultados de {source}:\n"
            if isinstance(data, dict):
                results = data.get('results', [])
                if results:
                    for i, res in enumerate(results, 1):
                        # Añadir al contexto principal
                        context += f"\n--- Evidencia {source} #{i} ---\n"
                        context += f"ID: {res.get('id', res.get('pmid', res.get('nctId', 'N/A')))}\n"
                        context += f"Título: {res.get('title', 'N/A')}\n"
                        authors = res.get('authors', [])
                        if isinstance(authors, list) and authors:
                            context += f"Autores: {', '.join(authors[:3])}{' et al.' if len(authors) > 3 else ''}\n"
                        elif isinstance(authors, str):
                            context += f"Autores: {authors}\n"
                        context += f"Fuente/Revista: {res.get('journal', source)}\n"
                        context += f"Año: {res.get('year', 'N/A')}\n"
                        context += f"URL: {res.get('url', '#')}\n"
                        context += f"Resumen: {res.get('abstract', 'No disponible.')}\n"
                        
                        # Generar referencia bibliográfica mejorada
                        if source == "PubMed":
                            pmid = res.get('pmid', 'N/A')
                            authors_str = ""
                            if isinstance(authors, list) and authors:
                                if len(authors) > 3:
                                    authors_str = f"{authors[0]}, {authors[1]}, {authors[2]} et al."
                                else:
                                    authors_str = ", ".join(authors)
                            elif isinstance(authors, str):
                                authors_str = authors
                            
                            year = res.get('year', 'N/A')
                            title = res.get('title', 'N/A')
                            journal = res.get('journal', 'N/A')
                            
                            # Formato más completo con URL
                            bibliography += f"- **[PubMed ID: {pmid}]** {authors_str} ({year}). *{title}*. {journal}. Disponible en: {res.get('url', '#')}\n"
                            
                        elif source == "ClinicalTrials.gov":
                            nct_id = res.get('nctId', 'N/A')
                            title = res.get('title', 'N/A')
                            status = res.get('status', 'N/A')
                            bibliography += f"- **[NCT: {nct_id}]** {title}. Estado: {status}. Disponible en: {res.get('url', '#')}\n"
                            
                        elif source == "GenBank":
                            genbank_id = res.get('id', 'N/A')
                            title = res.get('title', 'N/A')
                            bibliography += f"- **[GenBank: {genbank_id}]** {title}. Disponible en: {res.get('url', '#')}\n"
                            
                        elif source == "AEMPS":
                            drug_name = res.get('title', 'N/A')
                            bibliography += f"- **[AEMPS]** {drug_name}. Ficha técnica disponible en: {res.get('url', '#')}\n"
                else:
                    context += "- No se encontraron resultados en esta fuente.\n"

        prompt = f"""Eres un experto en investigación médica. Tu tarea es analizar la evidencia encontrada y sintetizar una respuesta clara, concisa y profesional.

PREGUNTA ORIGINAL: "{original_query}"

EVIDENCIA DISPONIBLE:
{context}

INSTRUCCIONES CRÍTICAS:
1. Analiza y sintetiza la evidencia encontrada
2. **INTERCALA LAS REFERENCIAS DIRECTAMENTE EN EL TEXTO** donde menciones cada hallazgo
   - Ejemplo: "Se ha demostrado que la metformina reduce el riesgo cardiovascular (PubMed ID: 12345678)"
   - Ejemplo: "Un ensayo clínico en fase III está evaluando esta terapia (NCT12345678)"
3. Usa el formato de referencia apropiado según la fuente:
   - PubMed: (PubMed ID: XXXXX)
   - ClinicalTrials: (NCT: XXXXX)
   - GenBank: (GenBank: XXXXX)
   - AEMPS: (AEMPS: nombre del medicamento)
4. Organiza la información de manera lógica y coherente
5. Incluye una conclusión basada en la evidencia

FORMATO DE LA RESPUESTA:
1. **Resumen principal** con referencias intercaladas
2. **Hallazgos clave** con sus respectivas referencias
3. **Conclusión** basada en la evidencia citada
4. **Referencias completas** al final (lista bibliográfica)

EJEMPLO DE TEXTO CON REFERENCIAS INTERCALADAS:
"Los estudios recientes han demostrado que la terapia CAR-T es efectiva en leucemia linfoblástica aguda refractaria (PubMed ID: 40458395), con tasas de remisión del 80% en pacientes pediátricos. Además, se están explorando nuevas aplicaciones en tumores sólidos (NCT05205902), aunque los resultados preliminares son mixtos..."

RESPUESTA:"""

        try:
            response = await self.llm.ainvoke(prompt)
            synthesis = self._extract_response_text(response)
            
            # Asegurar que la bibliografía se incluya al final
            if "### Bibliografía" not in synthesis:
                synthesis += bibliography
            
            return synthesis
            
        except Exception as e:
            logger.error(f"Error generando síntesis: {e}")
            return f"Error al generar la síntesis. Por favor, revise los resultados directamente:\n{context}"

    async def _tool_search_semantic_scholar(self, query: str) -> Dict[str, Any]:
        """Busca en Semantic Scholar usando su API."""
        try:
            self.stream_callback(f"   • 📚 Buscando en Semantic Scholar: '{query}'")
            
            async with aiohttp.ClientSession() as session:
                params = {
                    "query": query,
                    "limit": 10,
                    "fields": "title,abstract,authors,year,venue,paperId,citationCount"
                }
                async with session.get(f"{self.semantic_scholar_api}paper/search", params=params) as response:
                    data = await response.json()
                    
                    results = []
                    for paper in data.get("data", []):
                        results.append({
                            "title": paper.get("title"),
                            "authors": [a.get("name") for a in paper.get("authors", [])],
                            "year": paper.get("year"),
                            "journal": paper.get("venue"),
                            "abstract": paper.get("abstract"),
                            "citations": paper.get("citationCount"),
                            "url": f"https://www.semanticscholar.org/paper/{paper.get('paperId')}"
                        })
                    
                    return {
                        "source": "Semantic Scholar",
                        "results": results,
                        "total_found": len(results)
                    }
                    
        except Exception as e:
            logger.error(f"Error en Semantic Scholar: {e}")
            return {"source": "Semantic Scholar", "results": [], "error": str(e)}

    async def _tool_search_clinical_trials(self, query: str) -> Dict[str, Any]:
        """Busca ensayos clínicos en ClinicalTrials.gov usando la nueva API v2."""
        try:
            self.stream_callback(f"   • 🏥 Buscando ensayos clínicos (API v2): '{query}'")
            
            async with aiohttp.ClientSession() as session:
                # Parámetros para la nueva API v2
                params = {
                    "query.term": query,
                    "format": "json",
                    "pageSize": 10
                }
                # La URL correcta para la API v2
                api_url = "https://clinicaltrials.gov/api/v2/studies"
                
                async with session.get(api_url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error en ClinicalTrials.gov API v2: {response.status}, {error_text}")
                        return {"source": "ClinicalTrials.gov", "results": [], "error": f"HTTP {response.status}"}
                    
                    data = await response.json()
                    
                    results = []
                    for study_wrapper in data.get("studies", []):
                        study = study_wrapper.get("protocolSection", {})
                        
                        # Extraer información de la nueva estructura
                        status_module = study.get("statusModule", {})
                        id_module = study.get("identificationModule", {})
                        conditions_module = study.get("conditionsModule", {})
                        interventions_module = study.get("armsInterventionsModule", {})
                        
                        results.append({
                            "title": id_module.get("briefTitle", "N/A"),
                            "status": status_module.get("overallStatus", "N/A"),
                            "phase": study.get("designModule", {}).get("phases", ["N/A"])[0],
                            "conditions": conditions_module.get("conditions", []),
                            "interventions": [i.get("name") for i in interventions_module.get("interventions", [])],
                            "nctId": id_module.get("nctId"),
                            "url": f"https://clinicaltrials.gov/study/{id_module.get('nctId')}"
                        })
                    
                    return {
                        "source": "ClinicalTrials.gov",
                        "results": results,
                        "total_found": data.get("totalCount", 0)
                    }
                    
        except Exception as e:
            logger.error(f"Error en ClinicalTrials.gov: {e}", exc_info=True)
            return {"source": "ClinicalTrials.gov", "results": [], "error": str(e)}

    async def _tool_search_europe_pmc(self, query: str) -> Dict[str, Any]:
        """Busca en Europe PMC usando su API."""
        try:
            self.stream_callback(f"   • 🔬 Buscando en Europe PMC: '{query}'")
            
            async with aiohttp.ClientSession() as session:
                params = {
                    "query": query,
                    "format": "json",
                    "pageSize": 10,
                    "resultType": "core"
                }
                async with session.get(f"{self.europe_pmc_api}search", params=params) as response:
                    data = await response.json()
                    
                    results = []
                    for result in data.get("resultList", {}).get("result", []):
                        results.append({
                            "title": result.get("title"),
                            "authors": result.get("authorString", "").split(", "),
                            "journal": result.get("journalTitle"),
                            "year": result.get("pubYear"),
                            "abstract": result.get("abstractText"),
                            "pmid": result.get("pmid"),
                            "doi": result.get("doi"),
                            "url": f"https://europepmc.org/article/{result.get('source')}/{result.get('id')}"
                        })
                    
                    return {
                        "source": "Europe PMC",
                        "results": results,
                        "total_found": data.get("hitCount", 0)
                    }
                    
        except Exception as e:
            logger.error(f"Error en Europe PMC: {e}")
            return {"source": "Europe PMC", "results": [], "error": str(e)}

    async def _tool_search_aemps(self, drug_name: str) -> Dict[str, Any]:
        """Busca información de un medicamento en la AEMPS y extrae metadatos clave."""
        try:
            self.stream_callback(f"   • 🇪🇸 Buscando en AEMPS: '{drug_name}'")
            api_url = f"https://cima.aemps.es/cima/rest/medicamentos"
            params = {"nombre": drug_name}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, params=params) as response:
                    if response.status != 200:
                        return {"source": "AEMPS", "results": [], "error": f"HTTP {response.status}"}
                    
                    data = await response.json()
                    results = []
                    for med in data.get("resultados", [])[:2]: # Limitar a 2 para no saturar
                        doc_url = med.get("docs", [{}])[0].get("urlHtml", "#") if med.get("docs") else "#"
                        results.append({
                            "id": med.get("nregistro"), # Usar nregistro como ID único
                            "title": f"Ficha Técnica de '{med.get('nombre')}'",
                            "authors": [med.get("labtitular", "AEMPS")], # Usar laboratorio como autor
                            "year": str(datetime.now().year), # Usar año actual
                            "journal": "AEMPS CIMA",
                            "url": doc_url,
                            "abstract": f"Medicamento autorizado y registrado en España. Estado: {med.get('estado', {}).get('nombre', 'N/A')}. Laboratorio titular: {med.get('labtitular', 'N/A')}."
                        })
                    return {"source": "AEMPS", "results": results}
        except Exception as e:
            logger.error(f"Error en la búsqueda de AEMPS: {e}")
            return {"source": "AEMPS", "results": [], "error": str(e)}

    async def _detect_biomedical_entities(self, query: str) -> Dict[str, List[str]]:
        """Usa un LLM para detectar entidades biomédicas (genes, fármacos, enfermedades) en la consulta."""
        prompt = f"""
        Analiza la siguiente consulta biomédica e identifica las entidades clave.

        Consulta: "{query}"

        Extrae las siguientes entidades si existen:
        - "genes": Nombres de genes (ej: CYP2C19, BRCA1).
        - "drugs": Nombres de fármacos (ej: Clopidogrel, Atorvastatina).
        - "diseases": Nombres de enfermedades o condiciones (ej: Infarto de miocardio, Alzheimer).

        Responde SOLO con un objeto JSON con las listas de las entidades encontradas. Si no encuentras ninguna, devuelve una lista vacía.
        
        Formato de Respuesta (JSON):
        {{
          "genes": ["gen1", "gen2"],
          "drugs": ["farmaco1"],
          "diseases": ["enfermedad1"]
        }}
        """
        try:
            response = await self.llm.ainvoke(prompt, temperature=0.0)
            content = self._extract_response_text(response)
            entities = json.loads(content.strip())
            
            # Log de entidades detectadas
            detected_summary = ", ".join([f"{k}: {v}" for k, v in entities.items() if v])
            if detected_summary:
                logger.info(f"🧬 Entidades detectadas: {detected_summary}")
            
            return entities
        except Exception as e:
            logger.error(f"Error detectando entidades biomédicas: {e}")
            return {"genes": [], "drugs": [], "diseases": []}