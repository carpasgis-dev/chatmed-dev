import json
from typing import Dict, Any, Optional, List

class FHIRSQLLLMMapper:
    """
    Clase para mapear recursos FHIR a SQL de forma flexible usando LLM y prompts en español.
    """
    def __init__(self, llm, schema_getter):
        """
        llm: objeto LLM compatible (debe tener un método ainvoke o similar)
        schema_getter: función que recibe el nombre de la tabla y devuelve la lista de columnas válidas
        """
        self.llm = llm
        self.schema_getter = schema_getter

    async def map_fhir_to_sql(self, fhir_resource: Dict[str, Any], db_schema: Dict[str, List[str]], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Flujo flexible:
        1. Determina la tabla SQL adecuada usando el LLM
        2. Mapea los campos FHIR a columnas SQL usando el LLM
        3. Valida los campos con el esquema real
        4. Devuelve el mapeo listo para construir el SQL
        """
        # 1. Prompt para determinar la tabla
        prompt_tabla = f"""
Eres un experto en integración FHIR-SQL médico. Analiza el recurso FHIR y selecciona la tabla SQL más adecuada.

RECURSO FHIR:
{json.dumps(fhir_resource, indent=2, ensure_ascii=False)}

TABLAS DISPONIBLES EN LA BASE DE DATOS:
{list(db_schema.keys())}

REGLAS DE SELECCIÓN:
- Patient → PATI_PATIENTS
- MedicationRequest → PATI_USUAL_MEDICATION o MEDI_MEDICATIONS
- Condition → EPIS_DIAGNOSTICS
- Observation → EPIS_DIAGNOSTICS
- AllergyIntolerance → PATI_PATIENT_ALLERGIES
- Procedure → EPIS_PROCEDURES

Responde SOLO con el nombre exacto de la tabla (ejemplo: PATI_PATIENTS, EPIS_DIAGNOSTICS, etc.)
"""
        response_tabla = await self.llm.ainvoke(prompt_tabla)
        table_name = str(response_tabla.content).strip().split()[0].replace('"','').replace("'","")

        # 2. Prompt para mapear campos FHIR a columnas SQL
        prompt_campos = f"""
Eres un experto en mapeo de datos médicos FHIR a SQL. Analiza el recurso FHIR y mapea sus campos a las columnas disponibles en la tabla '{table_name}'.

RECURSO FHIR:
{json.dumps(fhir_resource, indent=2, ensure_ascii=False)}

COLUMNAS DISPONIBLES EN LA TABLA '{table_name}':
{db_schema.get(table_name, [])}

INSTRUCCIONES ESPECÍFICAS:
1. Analiza el tipo de recurso FHIR y su contenido
2. Identifica qué campos del FHIR se pueden mapear a las columnas de la tabla
3. Transforma los datos si es necesario (ej: texto a ID, arrays a texto, etc.)
4. Devuelve SOLO un JSON válido con el formato: {{"columna1": "valor1", "columna2": "valor2"}}

EJEMPLOS DE MAPEO:
- Para Patient: name.given[0] → PATI_NAME, name.family → PATI_SURNAME_1
- Para MedicationRequest: medicationCodeableConcept.text → MEDI_NAME, dosageInstruction[0].text → MEDI_DOSAGE
- Para Condition: code.text → DIAG_OBSERVATION

Responde SOLO con el JSON de mapeo:
"""
        response_campos = await self.llm.ainvoke(prompt_campos)
        
        # Mejorar parsing del JSON
        try:
            content = response_campos.content.strip()
            # Limpiar el contenido si tiene texto extra
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1]
            
            mapped_data = json.loads(content)
            print(f"   🔍 JSON parseado correctamente: {mapped_data}")
        except Exception as e:
            print(f"   ⚠️ Error parseando JSON: {e}")
            print(f"   📝 Contenido recibido: {response_campos.content}")
            mapped_data = {}

        # 3. Validar que los campos existen en la tabla
        valid_columns = set(db_schema.get(table_name, []))
        filtered_data = {k: v for k, v in mapped_data.items() if k in valid_columns}

        # 4. Devolver resultado
        return {
            'table': table_name,
            'columns': list(filtered_data.keys()),
            'values': list(filtered_data.values()),
            'mapped_data': filtered_data,
            'mapping_summary': {
                'resourceType': fhir_resource.get('resourceType'),
                'original_fields': list(fhir_resource.keys()),
                'used_columns': list(filtered_data.keys()),
                'ignored_fields': [k for k in mapped_data.keys() if k not in valid_columns]
            }
        } 