import logging
import os

def get_llm():
    """Inicializa y retorna una instancia del LLM (ChatOpenAI)"""
    try:
        from langchain_openai import ChatOpenAI
        
        # Verificar que tenemos API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY no está configurada")
        
        # Crear instancia del LLM
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1
        )
        
        return llm
        
    except Exception as e:
        logging.error(f"Error inicializando LLM: {e}")
        raise 