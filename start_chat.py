#!/usr/bin/env python3
"""
🚀 ChatMed v2.0 Flexible - Launcher Script
==========================================

Script de inicio optimizado para ChatMed v2.0 con:
- Configuración automática de paths
- Verificación de dependencias
- Agente FHIR completo restaurado
- Sistema completamente funcional

Uso: python start_chat.py
"""

import os
import sys
import asyncio
import warnings
import logging
from pathlib import Path

# Suprimir el warning específico de pkg_resources en eutils
warnings.filterwarnings("ignore", category=UserWarning, module="eutils")

# Configurar logging para ser menos verboso
logging.basicConfig(level=logging.ERROR, format='%(message)s')
# Silenciar loggers específicos que son muy verbosos
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("ChatMedFlexibleV2").setLevel(logging.ERROR)
logging.getLogger("FHIRAgent3.0").setLevel(logging.ERROR)
logging.getLogger("SQLAgentRobust").setLevel(logging.ERROR)
logging.getLogger("FlexibleEngine").setLevel(logging.ERROR)
logging.getLogger("FHIRPersistenceAgent").setLevel(logging.ERROR)
logging.getLogger("mapping.schema_introspector").setLevel(logging.ERROR)
logging.getLogger("chatmed_v2_flexible.mapping.schema_introspector").setLevel(logging.ERROR)
# Silenciar TODOS los loggers por defecto
logging.getLogger().setLevel(logging.ERROR)

# --- Mejoras de UI/UX para la Terminal ---

def print_header():
    """Imprime un encabezado visualmente atractivo."""
    header = """
    ┌──────────────────────────────────────────────────┐
    │  🩺   ChatMed v2.0 - Asistente Médico IA   🩺   │
    └──────────────────────────────────────────────────┘
    """
    print(header)

def print_section(title):
    """Imprime un título de sección."""
    print(f"\n--- {title.upper()} ---")

def print_check(message, success=True):
    """Imprime un item de verificación con un icono."""
    icon = "✅" if success else "⚠️"
    print(f"  {icon} {message}")

def setup_environment():
    """Configura el entorno de ejecución."""
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir.parent
    
    paths_to_add = [str(current_dir), str(project_root)]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
            
    print_check(f"Entorno configurado desde: {current_dir.name}")
    return current_dir, project_root

def check_dependencies():
    """Verifica dependencias críticas."""
    print_section("Verificación del sistema")
    
    # Verificar OpenAI API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print_check("OpenAI API Key configurada")
    else:
        print_check("⚠️ OPENAI_API_KEY no encontrada - funciones LLM limitadas", success=False)
        print("   💡 Para habilitar todas las funciones, configura la variable de entorno OPENAI_API_KEY")
    
    try:
        import openai
        print_check("OpenAI disponible")
    except ImportError:
        print_check("OpenAI no disponible - algunas funciones pueden estar limitadas", success=False)
    
    try:
        from langchain.llms.base import BaseLLM
        print_check("LangChain disponible")
    except ImportError:
        print_check("LangChain no disponible - funciones LLM limitadas", success=False)
    
    db_paths = ["database_new.sqlite3.db", "../database_new.sqlite3.db", "../../database_new.sqlite3.db"]
    db_found = False
    for db_path in db_paths:
        if os.path.exists(db_path):
            print_check(f"Base de datos encontrada: {db_path}")
            db_found = True
            break
    
    if not db_found:
        print_check("Base de datos no encontrada - se creará una nueva si es necesario", success=False)

async def create_orchestrator_with_fallback():
    """Crea el orquestador con manejo de errores y fallbacks"""
    try:
        from core.orchestrator_v2 import IntelligentOrchestratorV3
        
        print_section("Inicializando orquestador")
        
        # Intentar crear el orquestador mejorado
        orchestrator = IntelligentOrchestratorV3(
            enable_cache=True,
            enable_performance_monitoring=True
        )
        
        # Verificar que los agentes se inicializaron correctamente
        available_agents = list(orchestrator.agents.keys())
        if available_agents:
            print_check(f"Agentes inicializados: {', '.join(available_agents)}")
            return orchestrator
        else:
            print_check("⚠️ No se pudieron inicializar agentes", success=False)
            return None
            
    except ValueError as e:
        if "API Key" in str(e):
            print_check("❌ Error: API Key de OpenAI no configurada", success=False)
            print("   💡 Configura la variable de entorno OPENAI_API_KEY para usar todas las funciones")
            return None
        else:
            print_check(f"❌ Error de configuración: {e}", success=False)
            return None
    except FileNotFoundError as e:
        if "Base de datos" in str(e):
            print_check("❌ Error: Base de datos no encontrada", success=False)
            print("   💡 Asegúrate de que existe database_new.sqlite3.db en el directorio raíz")
            return None
        else:
            print_check(f"❌ Error de archivo: {e}", success=False)
            return None
    except Exception as e:
        print_check(f"❌ Error inesperado: {e}", success=False)
        return None

async def main():
    """Función principal"""
    print_header()
    
    print_section("Inicialización")
    current_dir, project_root = setup_environment()
    check_dependencies()
    
    try:
        if os.path.exists(os.path.join(current_dir, "chat_real.py")):
            print_check("Detectado 'chat_real.py', iniciando modo avanzado...")
            from chat_real import main as chat_main
            await chat_main()
            return 0
        else:
            print_check("No se encontró 'chat_real.py', usando chat interactivo integrado.", success=False)
            raise ImportError("chat_real not found")
        
    except ImportError:
        # Crear orquestador con manejo de errores
        orchestrator = await create_orchestrator_with_fallback()
        
        if not orchestrator:
            print_section("Sistema no disponible")
            print("❌ No se pudo inicializar el sistema ChatMed.")
            print("\n💡 Soluciones posibles:")
            print("   1. Configura OPENAI_API_KEY en tu entorno")
            print("   2. Verifica que existe database_new.sqlite3.db")
            print("   3. Instala las dependencias: pip install -r requirements.txt")
            print("   4. Revisa los logs para más detalles")
            return 1
        
        print_section("Sistema listo")
        print("💡 Comandos disponibles: 'salir', 'ayuda', 'stats'")
        print("="*50)
        
        print("👋 ¡Hola! Soy ChatMed. ¿En qué puedo ayudarte hoy?")
        
        while True:
            try:
                user_input = input("👤 Tu consulta: ").strip()
                
                if user_input.lower() in ['salir', 'exit', 'quit']:
                    print("\n👋 ¡Hasta luego! Ha sido un placer ayudarte.")
                    break
                
                if user_input.lower() == 'ayuda':
                    print("""
    ┌─────────────────────────── AYUDA ───────────────────────────┐
    │                                                             │
    │  Comandos:                                                  │
    │    • salir:      Termina la sesión.                         │
    │    • stats:      Muestra estadísticas de rendimiento.       │
    │    • ayuda:      Muestra este menú de ayuda.                │
    │                                                             │
    │  Tipos de consulta:                                         │
    │    • SQL:        "¿Cuántos pacientes hombres mayores de 40?"│
    │    • FHIR:       "Crea un recurso para un paciente..."      │
    │    • BioChat:    "Investigación sobre diabetes tipo 2"      │
    │    • Saludo:     "Hola, ¿qué tal?"                          │
    │                                                             │
    │  El sistema detectará automáticamente cómo ayudarte.        │
    └─────────────────────────────────────────────────────────────┘
                    """)
                    continue
                
                if user_input.lower() == 'stats':
                    try:
                        stats = orchestrator.get_performance_metrics()
                        print(f"""
    ┌────────────────── ESTADÍSTICAS DEL SISTEMA ──────────────────┐
    │                                                              │
    │   Consultas totales: {stats.get('total_queries', 0):<3}                                  │
    │   Tasa de acierto de caché: {stats.get('cache_hit_rate', '0%'):<8}                      │
    │   Tiempo de respuesta promedio: {stats.get('avg_response_time', '0.00s'):<8}                │
    │   Uso de agentes: {str(stats.get('agent_usage', {})):<35} │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
                        """)
                    except Exception as e:
                        print(f"❌ Error obteniendo estadísticas: {e}")
                    continue
                
                if not user_input:
                    continue
                
                print("🔄 Procesando tu solicitud...")
                
                # Callback para mostrar progreso
                def stream_callback(message: str):
                    print(f"   {message}")
                
                try:
                    result = await orchestrator.process_query_optimized(
                        user_input, 
                        stream_callback=stream_callback
                    )
                    
                    # DEBUG: Mostrar resultado completo para diagnóstico
                    print(f"\n🔍 DEBUG - Resultado completo:")
                    print(f"   Tipo: {type(result)}")
                    if isinstance(result, dict):
                        print(f"   Claves: {list(result.keys())}")
                        if 'sql' in result:
                            print(f"   SQL generado: {result['sql']}")
                        if 'data' in result:
                            print(f"   Datos encontrados: {len(result['data'])} registros")
                        if 'clinical_analysis' in result:
                            print(f"   Análisis clínico: {result['clinical_analysis']}")
                    
                    # Formatear la respuesta para el usuario
                    if isinstance(result, dict):
                        if result.get('success'):
                            # Extraer el mensaje principal
                            message = result.get('message', '')
                            if not message:
                                # Si no hay mensaje, intentar extraer de otros campos
                                if 'data' in result:
                                    message = f"Encontré {len(result['data'])} resultados."
                                elif 'sql' in result:
                                    message = "Consulta ejecutada correctamente."
                                else:
                                    message = "Operación completada exitosamente."
                            print(f"🩺 ChatMed: {message}")
                        else:
                            error_msg = result.get('error', 'Error desconocido')
                            print(f"❌ Error: {error_msg}")
                    else:
                        print(f"🩺 ChatMed: {result}")
                except Exception as e:
                    print(f"❌ Error procesando consulta: {e}")
                    print("💡 Intenta reformular tu pregunta o usa 'ayuda' para ver ejemplos")
                
            except KeyboardInterrupt:
                print("\n👋 ¡Sesión terminada bruscamente! Hasta luego.")
                break
            except Exception as e:
                print(f"❌ Error inesperado: {e}")
                continue
                    
    except Exception as e:
        print(f"❌ Error crítico al iniciar: {e}")
        print("  Verifica que la estructura del proyecto y las dependencias son correctas.")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta luego!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error fatal en el launcher: {e}")
        sys.exit(1) 