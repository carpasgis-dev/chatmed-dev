"""
ChatMed v2.0 - Interfaz de Chat Profesional
=========================================
Sistema Médico Inteligente - Versión de Producción
"""

# Suprimir warnings y logs técnicos
import warnings
warnings.filterwarnings("ignore")

import logging
import os
import sys
import asyncio
import re
from datetime import datetime
from typing import Union, Dict, Any, Optional

# Configurar codificación UTF-8 para Windows
if os.name == 'nt':
    os.system('chcp 65001 > nul')
    # Configurar variables de entorno para UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Configuración de logging personalizada para ChatMed
class ChatMedFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.colors = {
            'DEBUG': '\033[94m',    # Azul
            'INFO': '\033[92m',     # Verde
            'WARNING': '\033[93m',  # Amarillo
            'ERROR': '\033[91m',    # Rojo
            'CRITICAL': '\033[95m', # Magenta
            'RESET': '\033[0m'      # Reset
        }

    def format(self, record):
        # Añadir color según el nivel
        color = self.colors.get(record.levelname, self.colors['RESET'])
        reset = self.colors['RESET']
        
        # Formatear el mensaje con color y estructura
        message = record.getMessage()
        
        # Detectar tipo de mensaje por su contenido
        if '📚 SISTEMA DE APRENDIZAJE' in message:
            color = '\033[95m'  # Magenta para aprendizaje
        elif '⚡ EJECUCIÓN SQL' in message:
            color = '\033[93m'  # Amarillo para ejecución
        elif message.startswith('├') or message.startswith('│') or message.startswith('└'):
            # Mantener el color actual para la estructura de árbol
            pass
        elif 'ERROR' in message:
            color = self.colors['ERROR']
        
        return f"{color}{message}{reset}"

# Configurar handler principal
console_handler = logging.StreamHandler()
console_handler.setFormatter(ChatMedFormatter())

# Configurar logger para SQLAgent
sql_logger = logging.getLogger('SQLAgent')
sql_logger.setLevel(logging.INFO)
sql_logger.addHandler(console_handler)
sql_logger.propagate = False

# Suprimir otros logs no deseados
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('openai').setLevel(logging.ERROR)

# Suprimir logs de componentes del sistema excepto SQLAgent
for logger_name in ['ChatMedFlexibleV2', 'FHIRAgent3.0', 
                    'GreetingAgent', 'biochat_agent', 'FHIRPersistenceAgent', 'fhir_sql_bridge',
                    'sql_agent_tools', 'agents.biochat_agent']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

class Colors:
    HEADER = '\033[96m\033[1m'
    USER = '\033[92m\033[1m'
    ASSISTANT = '\033[94m\033[1m'
    SYSTEM = '\033[93m\033[1m'
    ERROR = '\033[91m\033[1m'
    SUCCESS = '\033[92m\033[1m'
    INFO = '\033[95m\033[1m'
    WARNING = '\033[93m\033[1m'
    ACCENT = '\033[35m\033[1m'
    RESET = '\033[0m'

class ChatMedInterface:
    def __init__(self):
        self.running = True
        self.session_start = datetime.now()
        self.query_count = 0
        self.colors = Colors()
        self.orchestrator = None
        self.loading_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.loading_index = 0
    
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_welcome(self):
        """Pantalla de bienvenida profesional"""
        welcome = f"""
{self.colors.HEADER}================================================================
                    ChatMed v2.0 Professional                    
                   Sistema Medico Inteligente                       
                    Powered by Advanced AI                          
================================================================{self.colors.RESET}

{self.colors.ACCENT}-- Capacidades del Sistema --{self.colors.RESET}
{self.colors.SUCCESS}* Consultas SQL Inteligentes  * Gestion de Registros FHIR{self.colors.RESET}
{self.colors.SUCCESS}* Investigacion Biomedica     * Asistencia Medica IA{self.colors.RESET}

{self.colors.INFO}Comandos: {self.colors.RESET}salir, limpiar, ayuda, stats
{self.colors.SYSTEM}Sesion iniciada: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}{self.colors.RESET}
"""
        print(welcome)
    
    def show_loading(self, message: str):
        """Muestra una animación de carga elegante"""
        char = self.loading_chars[self.loading_index % len(self.loading_chars)]
        self.loading_index += 1
        print(f"\r{self.colors.SYSTEM}{char} {message}...{self.colors.RESET}", end="", flush=True)
    
    def stream_callback(self, message: str):
        """Callback para mostrar progreso en tiempo real"""
        # Los mensajes ahora vienen directamente del sistema de logging mejorado
        # Solo necesitamos manejar mensajes especiales que no vienen del logger
        
        if message.startswith("🔄 INICIANDO"):
            print(f"{self.colors.HEADER}\n{'='*64}")
            print(f"Procesando consulta...")
            print(f"{'='*64}{self.colors.RESET}")
            return
            
        if "Insight para" in message:
            table = message.split("'")[1] if "'" in message else "tabla"
            print(f"{self.colors.ACCENT}🧠 Generando insights para {table}...{self.colors.RESET}")
            return
            
        if "SQL generado" in message:
            print(f"{self.colors.SUCCESS}✓ SQL generado exitosamente{self.colors.RESET}")
            return
            
        # Los demás mensajes son manejados por el logger
        pass
    
    def format_response(self, response: Union[str, Dict[str, Any]]) -> str:
        """Formatea respuestas de manera elegante y profesional"""
        if isinstance(response, dict):
            if not response.get('success', False):
                error_msg = response.get('message', 'Ocurrió un error inesperado.')
                return f"""
{self.colors.ERROR}================================================================
ERROR: {error_msg}
================================================================{self.colors.RESET}
"""

            # Respuesta exitosa
            main_message = response.get('message', 'Operación completada.')
            data = response.get('data')
            sql_query = response.get('sql_query')
            explanation = response.get('explanation', '')
            
            formatted_response = f"{self.colors.SUCCESS}================================================================\n"
            formatted_response += f"RESULTADO: {main_message}\n"
            formatted_response += f"================================================================{self.colors.RESET}\n"
            
            # Mostrar explicación detallada si está disponible
            if explanation:
                formatted_response += f"\n{self.colors.INFO}📋 EXPLICACIÓN DETALLADA:{self.colors.RESET}\n"
                formatted_response += f"{explanation}\n"
            
            # Mostrar SQL ejecutado si está disponible
            if sql_query:
                formatted_response += f"\n{self.colors.SYSTEM}🔧 SQL ejecutado:{self.colors.RESET}\n"
                formatted_response += f"{self.colors.ACCENT}   {sql_query}{self.colors.RESET}\n"
            
            # Mostrar datos si están disponibles (solo si no hay explicación detallada)
            if data and not explanation:
                if isinstance(data, list) and data:
                    formatted_response += f"\n{self.colors.INFO}📊 RESULTADOS ENCONTRADOS ({len(data)} registros):{self.colors.RESET}\n"
                    for i, item in enumerate(data, 1):
                        if isinstance(item, dict):
                            formatted_response += f"{self.colors.ACCENT}-- Registro {i} --{self.colors.RESET}\n"
                            for key, value in item.items():
                                formatted_response += f"{self.colors.ACCENT}  {key}:{self.colors.RESET} {value}\n"
                            formatted_response += "\n"
                        else:
                            formatted_response += f"   - {item}\n"
                else:
                    formatted_response += f"\n{self.colors.INFO}📋 DETALLE:{self.colors.RESET} {str(data)}\n"
            
            return formatted_response

        # Respuesta de string
        formatted_response = f"{self.colors.SUCCESS}================================================================\n"
        formatted_response += f"{response}\n"
        formatted_response += f"================================================================{self.colors.RESET}\n"
        return formatted_response
    
    def get_user_input(self) -> str:
        """Obtiene entrada del usuario con estilo"""
        prompt = f"\n{self.colors.USER}Tu consulta: {self.colors.RESET}"
        
        try:
            user_input = input(prompt).strip()
            return user_input
        except KeyboardInterrupt:
            print(f"\n\n{self.colors.SYSTEM}Hasta luego!{self.colors.RESET}")
            return "salir"
    
    async def initialize_system(self):
        """Inicializa el sistema de manera robusta"""
        try:
            # Configurar entorno para evitar errores de codificación
            import locale
            import codecs
            
            # Configurar el logger del SQLAgent para usar nuestro formateador personalizado
            sql_logger = logging.getLogger('SQLAgent')
            if not sql_logger.handlers:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(ChatMedFormatter())
                sql_logger.addHandler(console_handler)
                sql_logger.setLevel(logging.INFO)
                sql_logger.propagate = False
            
            # Forzar UTF-8 en todas las salidas
            if os.name == 'nt':
                try:
                    locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
                except:
                    try:
                        locale.setlocale(locale.LC_ALL, 'Spanish_Spain.UTF-8')
                    except:
                        pass
            
            # Redirigir completamente stdout/stderr durante inicialización
            class NullWriter:
                def write(self, text): pass
                def flush(self): pass
                def close(self): pass
                
            null_writer = NullWriter()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            try:
                sys.stdout = null_writer
                sys.stderr = null_writer
                
                # Suprimir completamente cualquier output durante importación
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    sys.path.append(os.path.dirname(__file__))
                    from core.orchestrator_v2 import create_flexible_orchestrator
                    
                    self.orchestrator = await create_flexible_orchestrator()
                
                # Restaurar stdout/stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                
                return True
                
            except Exception as init_error:
                # Restaurar stdout/stderr antes de mostrar error
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                
                # Convertir cualquier carácter problemático a ASCII
                error_text = str(init_error).encode('ascii', 'ignore').decode('ascii')
                
                print(f"""
{self.colors.ERROR}================================================================
ERROR DE INICIALIZACION: {error_text}
================================================================{self.colors.RESET}
""")
                return False
                
        except Exception as e:
            # Error crítico - mostrar error mínimo
            error_text = str(e).encode('ascii', 'ignore').decode('ascii')
            print(f"ERROR CRITICO: {error_text}")
            return False
    
    async def run_chat(self):
        """Ejecuta el chat principal"""
        self.clear_screen()
        self.print_welcome()
        
        # Mostrar inicialización
        print(f"\n{self.colors.SYSTEM}Inicializando sistema...{self.colors.RESET}")
        
        # Animación de carga
        for i in range(10):
            self.show_loading("Cargando componentes del sistema")
            await asyncio.sleep(0.1)
        
        print(f"\r{self.colors.SYSTEM}Inicializando sistema... OK{self.colors.RESET}")
        
        if not await self.initialize_system():
            return
        
        print(f"""
{self.colors.SUCCESS}================================================================
ChatMed v2.0 esta listo para ayudarte
Escribe tu consulta medica, pregunta o comando
================================================================{self.colors.RESET}
""")
        
        while self.running:
            user_input = self.get_user_input()
            
            if not user_input:
                continue
            
            # Comandos del sistema
            if user_input.lower() in ['salir', 'exit', 'quit']:
                print(f"""
{self.colors.SYSTEM}================================================================
Gracias por usar ChatMed v2.0
Consultas procesadas: {self.query_count}
Tiempo de sesión: {str(datetime.now() - self.session_start)}
================================================================{self.colors.RESET}
""")
                break
            elif user_input.lower() in ['limpiar', 'clear']:
                self.clear_screen()
                self.print_welcome()
                continue
            elif user_input.lower() == 'ayuda':
                self.show_help()
                continue
            elif user_input.lower() == 'stats':
                self.show_stats()
                continue
            
            # Procesar consulta
            try:
                # Mostrar procesamiento
                print(f"\n{self.colors.SYSTEM}Procesando consulta...{self.colors.RESET}")
                
                if self.orchestrator:
                    # Procesar con callback para mostrar progreso
                    response = await self.orchestrator.process_query_optimized(
                        user_input, 
                        stream_callback=self.stream_callback
                    )
                    formatted_response = self.format_response(response)
                    print(formatted_response)
                    print(f"\n{self.colors.SYSTEM}{'='*64}{self.colors.RESET}")
                else:
                    print(f"""
{self.colors.ERROR}================================================================
El sistema no esta disponible en este momento
================================================================{self.colors.RESET}
""")
                    
            except Exception as e:
                print(f"""
{self.colors.ERROR}================================================================
ERROR AL PROCESAR LA CONSULTA: {str(e)}
================================================================{self.colors.RESET}
""")

            self.query_count += 1

    def show_help(self):
        """Muestra la ayuda del sistema"""
        help_text = f"""
{self.colors.INFO}================================================================
AYUDA - ChatMed v2.0
================================================================

COMANDOS DISPONIBLES:
  salir/exit/quit  - Terminar la sesión
  limpiar/clear    - Limpiar la pantalla
  ayuda            - Mostrar esta ayuda
  stats            - Mostrar estadísticas del sistema

TIPOS DE CONSULTA:
  - Consultas SQL: "¿Cuántos pacientes diabéticos hay?"
  - Registros FHIR: "Registrar paciente Juan Pérez, 45 años"
  - Investigación: "Estudios sobre diabetes tipo 2"
  - Conversación: "Hola", "¿qué puedes hacer?"

El sistema detecta automáticamente el tipo de consulta
================================================================{self.colors.RESET}
"""
        print(help_text)

    def show_stats(self):
        """Muestra estadísticas del sistema"""
        if self.orchestrator:
            stats = self.orchestrator.get_performance_metrics()
            
            stats_text = f"""
{self.colors.INFO}================================================================
ESTADISTICAS DEL SISTEMA
================================================================

RENDIMIENTO:
  Consultas totales: {stats.get('total_queries', 0)}
  Cache: {stats.get('cache_hit_rate', '0%')}
  Tiempo promedio: {stats.get('avg_response_time', '0.00s')}

SESION ACTUAL:
  Consultas: {self.query_count}
  Tiempo activo: {str(datetime.now() - self.session_start)}

================================================================{self.colors.RESET}
"""
            print(stats_text)
        else:
            print(f"""
{self.colors.ERROR}================================================================
ESTADISTICAS NO DISPONIBLES
No se pueden mostrar las estadisticas en este momento
================================================================{self.colors.RESET}
""")

async def main():
    """Función principal"""
    try:
        chat = ChatMedInterface()
        await chat.run_chat()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.SYSTEM}👋 Sistema terminado por el usuario{Colors.RESET}")
    except Exception as e:
        print(f"""
{Colors.ERROR}================================================================
ERROR FATAL DEL SISTEMA: {str(e)}
================================================================{Colors.RESET}
""")

if __name__ == "__main__":
    asyncio.run(main())
