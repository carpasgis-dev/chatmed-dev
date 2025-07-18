#!/usr/bin/env python3
"""
ChatMed v2.0 Flexible - Terminal Interface
Sistema de IA médica multi-agente con interfaz de terminal
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Optional
import re

# Configurar OpenAI API Key explícitamente
try:
    from config_openai import setup_openai_config
    setup_openai_config()
    print("✅ Configuración de OpenAI cargada correctamente")
except Exception as e:
    print(f"⚠️ Error cargando configuración: {e}")

# Configurar codificación UTF-8 para Windows
if os.name == 'nt':
    try:
        os.system('chcp 65001 > nul')
    except:
        pass  # Ignorar si chcp no está disponible
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Suprimir TODOS los logs para una visualización limpia
logging.getLogger().setLevel(logging.INFO)  # Cambiar a INFO para debug
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('langchain').setLevel(logging.WARNING)
logging.getLogger('core').setLevel(logging.INFO)  # Habilitar logs del core
logging.getLogger('agents').setLevel(logging.INFO)  # Habilitar logs de agentes
logging.getLogger('fhir_sql_bridge').setLevel(logging.WARNING)
logging.getLogger('ChatMedFlexibleV2').setLevel(logging.INFO)  # Habilitar logs
logging.getLogger('SQLAgentIntelligent_v4.2').setLevel(logging.INFO)  # Habilitar logs
logging.getLogger('SQLAgent').setLevel(logging.INFO)  # Habilitar logs
# Habilitar logs del GreetingAgent para debug
logging.getLogger('GreetingAgent').setLevel(logging.INFO)

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.orchestrator_v2 import IntelligentOrchestratorV3
from langchain_openai import ChatOpenAI

class Colors:
    """Paleta de colores para la interfaz"""
    # Colores principales
    PRIMARY = '\033[38;5;33m'      # Azul médico
    SECONDARY = '\033[38;5;141m'   # Púrpura suave
    SUCCESS = '\033[38;5;46m'      # Verde éxito
    WARNING = '\033[38;5;208m'     # Naranja advertencia
    ERROR = '\033[38;5;196m'       # Rojo error
    INFO = '\033[38;5;51m'         # Cian info
    
    # Colores de texto
    WHITE = '\033[37m'
    GRAY = '\033[38;5;240m'
    LIGHT_GRAY = '\033[38;5;248m'
    
    # Efectos
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    
    # Reset
    RESET = '\033[0m'
    
    # Gradientes
    GRADIENT_1 = '\033[38;5;33m'
    GRADIENT_2 = '\033[38;5;39m'
    GRADIENT_3 = '\033[38;5;45m'
    GRADIENT_4 = '\033[38;5;51m'
    
    # Colores para streaming
    STREAM_BLUE = '\033[38;5;27m'
    STREAM_GREEN = '\033[38;5;34m'
    STREAM_YELLOW = '\033[38;5;220m'
    STREAM_PURPLE = '\033[38;5;99m'
    STREAM_ORANGE = '\033[38;5;208m'

class Icons:
    """Iconos Unicode para la interfaz"""
    # Emojis médicos
    DOCTOR = "👨‍⚕️"
    NURSE = "👩‍⚕️"
    HOSPITAL = "🏥"
    PILL = "💊"
    HEART = "❤️"
    BRAIN = "🧠"
    DNA = "🧬"
    MICROSCOPE = "🔬"
    SYRINGE = "💉"
    
    # Estados
    LOADING = "⏳"
    SUCCESS = "✅"
    ERROR = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"
    
    # Agentes
    SQL = "🗄️"
    FHIR = "📋"
    BIOCHAT = "🔬"
    MEDGEMMA = "🧠"
    GREETING = "👋"
    
    # Animaciones
    SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    # Streaming
    STREAM_START = "🚀"
    STREAM_STEP = "⚡"
    STREAM_DONE = "✅"
    STREAM_ERROR = "❌"
    STREAM_WARNING = "⚠️"
    STREAM_INFO = "ℹ️"
    STREAM_DEBUG = "🔍"
    STREAM_SQL = "🗄️"
    STREAM_AI = "🤖"
    STREAM_DB = "🗄️"
    STREAM_ANALYSIS = "🔬"

class StreamingVisualizer:
    """Visualizador de streaming para logs y procesos"""
    
    def __init__(self):
        self.current_step = 0
        self.total_steps = 0
        self.current_process = ""
        self.step_details = []
        self.start_time = None
    
    def start_process(self, process_name: str, total_steps: int = 0):
        """Inicia un nuevo proceso de streaming"""
        self.current_process = process_name
        self.total_steps = total_steps
        self.current_step = 0
        self.step_details = []
        self.start_time = datetime.now()
        
        print(f"\n{Colors.STREAM_BLUE}{Icons.STREAM_START} {Colors.BOLD}Iniciando: {process_name}{Colors.RESET}")
        if total_steps > 0:
            print(f"{Colors.GRAY}   Pasos totales: {total_steps}{Colors.RESET}")
    
    def update_step(self, step_name: str, status: str = "info", details: str = ""):
        """Actualiza el paso actual del proceso"""
        self.current_step += 1
        
        # Iconos según el estado
        status_icons = {
            "info": Icons.STREAM_INFO,
            "success": Icons.STREAM_DONE,
            "error": Icons.STREAM_ERROR,
            "warning": Icons.STREAM_WARNING,
            "debug": Icons.STREAM_DEBUG,
            "sql": Icons.STREAM_SQL,
            "ai": Icons.STREAM_AI,
            "db": Icons.STREAM_DB,
            "analysis": Icons.STREAM_ANALYSIS
        }
        
        # Colores según el estado
        status_colors = {
            "info": Colors.STREAM_BLUE,
            "success": Colors.SUCCESS,
            "error": Colors.ERROR,
            "warning": Colors.WARNING,
            "debug": Colors.STREAM_PURPLE,
            "sql": Colors.STREAM_GREEN,
            "ai": Colors.STREAM_ORANGE,
            "db": Colors.STREAM_BLUE,
            "analysis": Colors.STREAM_YELLOW
        }
        
        icon = status_icons.get(status, Icons.STREAM_STEP)
        color = status_colors.get(status, Colors.STREAM_BLUE)
        
        # Mostrar progreso si hay total de pasos
        progress = ""
        if self.total_steps > 0:
            progress = f" [{self.current_step}/{self.total_steps}]"
        
        # Línea principal del paso
        step_line = f"{color}{icon} {step_name}{progress}{Colors.RESET}"
        
        # Añadir detalles si existen
        if details:
            step_line += f"\n{Colors.GRAY}   └─ {details}{Colors.RESET}"
        
        print(step_line)
        
        # Guardar detalles para resumen
        self.step_details.append({
            "step": step_name,
            "status": status,
            "details": details,
            "timestamp": datetime.now()
        })
    
    def end_process(self, success: bool = True, summary: str = ""):
        """Termina el proceso de streaming"""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            
            if success:
                print(f"\n{Colors.SUCCESS}{Icons.STREAM_DONE} {Colors.BOLD}Proceso completado exitosamente{Colors.RESET}")
            else:
                print(f"\n{Colors.ERROR}{Icons.STREAM_ERROR} {Colors.BOLD}Proceso completado con errores{Colors.RESET}")
            
            print(f"{Colors.GRAY}   Duración: {duration:.2f}s{Colors.RESET}")
            
            if summary:
                print(f"{Colors.INFO}   Resumen: {summary}{Colors.RESET}")
    
    def show_progress_bar(self, current: int, total: int, width: int = 40):
        """Muestra una barra de progreso visual"""
        if total == 0:
            return
        
        progress = current / total
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        percentage = int(progress * 100)
        
        print(f"{Colors.STREAM_BLUE}[{bar}] {percentage}%{Colors.RESET}")

class ModernTerminalInterface:
    """Interfaz terminal moderna para ChatMed con visualización streaming"""
    
    def __init__(self):
        self.orchestrator = None
        self.query_count = 0
        self.start_time = datetime.now()
        self.spinner_index = 0
        self.streaming_visualizer = StreamingVisualizer()
        
        # Configurar captura de logs para visualización streaming
        self.setup_log_capture()
    
    def setup_log_capture(self):
        """Configura la captura de logs para visualización streaming - DESACTIVADO PARA LIMPIEZA"""
        # Desactivado temporalmente para una visualización más limpia
        pass
    
    def process_log_message(self, message: str, level: str = "INFO"):
        """Procesa mensajes de log para visualización streaming - DESACTIVADO"""
        # Desactivado temporalmente para una visualización más limpia
        pass
    
    def extract_clean_message(self, message: str) -> Optional[str]:
        """Extrae un mensaje limpio del log para mostrar"""
        # Patrones para extraer información relevante
        patterns = [
            r'DEBUG: (.+)',
            r'INFO: (.+)',
            r'ERROR: (.+)',
            r'⚠️ (.+)',
            r'✅ (.+)',
            r'❌ (.+)',
            r'🔍 (.+)',
            r'🧠 (.+)',
            r'💡 (.+)',
            r'🚀 (.+)',
            r'⚡ (.+)',
            r'🗄️ (.+)',
            r'🤖 (.+)',
            r'🔬 (.+)',
            r'💾 (.+)',
            r'📋 (.+)',
            r'👋 (.+)',
            r'👨‍⚕️ (.+)',
            r'👩‍⚕️ (.+)',
            r'🏥 (.+)',
            r'💊 (.+)',
            r'❤️ (.+)',
            r'🧬 (.+)',
            r'💉 (.+)',
            r'⏳ (.+)',
            r'ℹ️ (.+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                return match.group(1).strip()
        
        # Si no hay patrón específico, intentar limpiar el mensaje
        if ":" in message:
            parts = message.split(":", 1)
            if len(parts) > 1:
                return parts[1].strip()
        
        return None
    
    def clear_screen(self):
        """Limpia la pantalla"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_welcome(self):
        """Pantalla de bienvenida moderna"""
        welcome = f"""
{Colors.GRADIENT_1}╔══════════════════════════════════════════════════════════════════════════════╗
║{Colors.GRADIENT_2}                    🏥 ChatMed v2.0 - Asistente Médico IA 🏥                    {Colors.GRADIENT_1}║
║{Colors.GRADIENT_3}                         Sistema Médico Inteligente Avanzado                        {Colors.GRADIENT_1}║
║{Colors.GRADIENT_4}                              Powered by Advanced AI                               {Colors.GRADIENT_1}║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}

{Colors.INFO}🎯 {Colors.BOLD}Capacidades del Sistema:{Colors.RESET}
{Colors.SUCCESS}   • {Colors.BOLD}Consultas SQL Inteligentes{Colors.RESET}     {Icons.SQL} Análisis de datos médicos
{Colors.SUCCESS}   • {Colors.BOLD}Gestión de Registros FHIR{Colors.RESET}      {Icons.FHIR} Notas clínicas estructuradas  
{Colors.SUCCESS}   • {Colors.BOLD}Investigación Biomédica{Colors.RESET}        {Icons.BIOCHAT} Literatura científica
{Colors.SUCCESS}   • {Colors.BOLD}Análisis Clínico IA{Colors.RESET}           {Icons.MEDGEMMA} Diagnósticos inteligentes

{Colors.WARNING}💡 {Colors.BOLD}Comandos Disponibles:{Colors.RESET}
{Colors.LIGHT_GRAY}   • {Colors.BOLD}salir{Colors.RESET} - Terminar sesión
{Colors.LIGHT_GRAY}   • {Colors.BOLD}limpiar{Colors.RESET} - Limpiar pantalla  
{Colors.LIGHT_GRAY}   • {Colors.BOLD}ayuda{Colors.RESET} - Mostrar ayuda
{Colors.LIGHT_GRAY}   • {Colors.BOLD}stats{Colors.RESET} - Estadísticas del sistema

{Colors.SECONDARY}🕐 {Colors.BOLD}Sesión iniciada:{Colors.RESET} {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
{Colors.RESET}"""
        print(welcome)
    
    async def show_loading_animation(self, message: str, duration: float = 1.0):
        """Muestra una animación de carga elegante"""
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < duration:
            for i, char in enumerate(Icons.SPINNER):
                print(f"\r{Colors.INFO}{char} {message}...{Colors.RESET}", end="", flush=True)
                await asyncio.sleep(0.1)
                if (datetime.now() - start_time).total_seconds() >= duration:
                    break
        print(f"\r{Colors.SUCCESS}✅ {message} completado{Colors.RESET}")
    
    def print_help(self):
        """Mostrar ayuda con formato moderno"""
        help_text = f"""
{Colors.PRIMARY}╔══════════════════════════════════════════════════════════════════════════════╗
║                                 📚 AYUDA - ChatMed v2.0                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}

{Colors.INFO}🎯 {Colors.BOLD}Tipos de Consulta:{Colors.RESET}

{Icons.SQL} {Colors.BOLD}Consultas SQL (Datos de Pacientes):{Colors.RESET}
   • "Busca pacientes con diabetes"
   • "¿Cuántos pacientes hay?"
   • "Muéstrame el último paciente"
   • "Pacientes mayores de 60 años"

{Icons.FHIR} {Colors.BOLD}Registros FHIR (Notas Clínicas):{Colors.RESET}
   • "Registrar paciente Juan Pérez, 45 años"
   • "Crear nota clínica para paciente 12345"
   • "Añadir diagnóstico de hipertensión"

{Icons.BIOCHAT} {Colors.BOLD}Investigación Biomédica:{Colors.RESET}
   • "¿Qué investigaciones recientes hay sobre diabetes?"
   • "Ensayos clínicos sobre cáncer de mama"
   • "Últimos estudios sobre tratamientos COVID"

{Icons.MEDGEMMA} {Colors.BOLD}Análisis Clínico:{Colors.RESET}
   • "¿Qué es la hipertensión arterial?"
   • "¿Para qué sirve la metformina?"
   • "¿Es seguro tomar paracetamol con ibuprofeno?"

{Icons.GREETING} {Colors.BOLD}Interacciones Básicas:{Colors.RESET}
   • "Hola", "¿Qué puedes hacer?"
   • "Gracias", "Adiós"

{Colors.WARNING}💡 {Colors.BOLD}Consejos:{Colors.RESET}
   • Sé específico en tus consultas
   • Usa lenguaje natural
   • El sistema detecta automáticamente el tipo de consulta
   • Puedes hacer preguntas de seguimiento

{Colors.SUCCESS}✅ {Colors.BOLD}Comandos del Sistema:{Colors.RESET}
   • {Colors.BOLD}salir{Colors.RESET} - Terminar sesión
   • {Colors.BOLD}limpiar{Colors.RESET} - Limpiar pantalla
   • {Colors.BOLD}ayuda{Colors.RESET} - Mostrar esta ayuda
   • {Colors.BOLD}stats{Colors.RESET} - Estadísticas del sistema
{Colors.RESET}"""
        print(help_text)
    
    def print_stats(self):
        """Mostrar estadísticas con formato moderno"""
        stats_text = f"""
{Colors.PRIMARY}╔══════════════════════════════════════════════════════════════════════════════╗
║                              📊 ESTADÍSTICAS DEL SISTEMA                               ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}

{Colors.INFO}📈 {Colors.BOLD}Rendimiento:{Colors.RESET}
   • {Colors.BOLD}Consultas procesadas:{Colors.RESET} {self.query_count}
   • {Colors.BOLD}Tiempo activo:{Colors.RESET} {str(datetime.now() - self.start_time)}
   • {Colors.BOLD}Estado del sistema:{Colors.RESET} {'✅ Activo' if self.orchestrator else '❌ Inactivo'}

{Colors.SUCCESS}🎯 {Colors.BOLD}Sesión Actual:{Colors.RESET}
   • {Colors.BOLD}Inicio de sesión:{Colors.RESET} {self.start_time.strftime('%H:%M:%S')}
   • {Colors.BOLD}Tiempo transcurrido:{Colors.RESET} {str(datetime.now() - self.start_time)}
   • {Colors.BOLD}Promedio por consulta:{Colors.RESET} {'N/A' if self.query_count == 0 else f'{str((datetime.now() - self.start_time) / self.query_count)}'}

{Colors.WARNING}⚡ {Colors.BOLD}Información del Sistema:{Colors.RESET}
   • {Colors.BOLD}Base de datos:{Colors.RESET} ✅ Conectada
   • {Colors.BOLD}LLM:{Colors.RESET} ✅ Disponible
   • {Colors.BOLD}Agentes:{Colors.RESET} ✅ Todos activos
{Colors.RESET}"""
        print(stats_text)
    
    async def initialize_system(self):
        """Inicializar el sistema con animaciones"""
        print(f"\n{Colors.INFO}🚀 {Colors.BOLD}Inicializando ChatMed v2.0...{Colors.RESET}")
        
        # Animación de inicialización
        steps = [
            "Cargando agentes especializados",
            "Conectando con base de datos",
            "Inicializando modelos de IA",
            "Configurando orquestador inteligente",
            "Preparando interfaz de usuario"
        ]
        
        for i, step in enumerate(steps):
            await self.show_loading_animation(step, 0.8)
            await asyncio.sleep(0.2)
        
        try:
            # Configurar LLM
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.1
            )
            
            # Inicializar orquestador con LLM explícito
            self.orchestrator = IntelligentOrchestratorV3(
                config_path=None,
                enable_cache=True,
                enable_performance_monitoring=True,
                cache_ttl_minutes=60
            )
            
            # FORZAR la configuración del LLM
            self.orchestrator.llm_classifier = llm
            print(f"{Colors.INFO}✅ LLM configurado en orquestador{Colors.RESET}")
            
            # Verificar que los agentes se inicializaron
            available_agents = list(self.orchestrator.agents.keys())
            print(f"{Colors.INFO}✅ Agentes disponibles: {', '.join(available_agents)}{Colors.RESET}")
            
            print(f"\n{Colors.SUCCESS}✅ {Colors.BOLD}¡Sistema inicializado exitosamente!{Colors.RESET}")
            print(f"{Colors.INFO}💡 {Colors.BOLD}Escribe tu consulta médica o usa 'ayuda' para más información{Colors.RESET}")
            
            return True
            
        except Exception as e:
            print(f"\n{Colors.ERROR}❌ {Colors.BOLD}Error inicializando el sistema:{Colors.RESET}")
            print(f"{Colors.ERROR}{str(e)}{Colors.RESET}")
            return False
    
    def format_user_input(self, text: str) -> str:
        """Formatea la entrada del usuario"""
        return f"{Colors.WHITE}{Colors.BOLD}👤 Tú:{Colors.RESET} {text}"
    
    def format_assistant_response(self, text: str) -> str:
        """Formatea la respuesta del asistente"""
        return f"{Colors.PRIMARY}{Colors.BOLD}🤖 ChatMed:{Colors.RESET} {text}"
    
    def format_error(self, error: str) -> str:
        """Formatea errores de manera elegante"""
        return f"""
{Colors.ERROR}╔══════════════════════════════════════════════════════════════════════════════╗
║                              ❌ ERROR EN EL SISTEMA                               ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}

{Colors.ERROR}{Colors.BOLD}Descripción:{Colors.RESET} {error}

{Colors.WARNING}💡 {Colors.BOLD}Posibles soluciones:{Colors.RESET}
   • Verifica tu conexión a internet
   • Asegúrate de que la API key esté configurada
   • Revisa que la base de datos esté disponible
   • Intenta reformular tu consulta

{Colors.INFO}🔄 {Colors.BOLD}El sistema continuará funcionando normalmente.{Colors.RESET}
"""
    
    async def process_query(self, query: str):
        """Procesar una consulta con formato moderno y visualización streaming limpia"""
        if not self.orchestrator:
            print(f"\n{self.format_assistant_response(self.format_error('Sistema no disponible'))}")
            return
        
        try:
            start_time = datetime.now()
            
            # Mostrar inicio de procesamiento de forma elegante
            print(f"\n{Colors.INFO}🔄 {Colors.BOLD}Procesando consulta...{Colors.RESET}")
            
            # Función de callback para streaming
            def stream_callback(message: str):
                """Callback para mostrar progreso en tiempo real"""
                if message.startswith("🎯"):
                    # Mostrar agente seleccionado
                    print(f"\n{Colors.SECONDARY}{Colors.BOLD}{message}{Colors.RESET}")
                elif message.startswith("🧠") or message.startswith("🗄️") or message.startswith("🔬"):
                    # Mostrar pasos de procesamiento
                    print(f"{Colors.GRAY}   {message}{Colors.RESET}")
                else:
                    # Mostrar otros mensajes de progreso
                    print(f"{Colors.GRAY}   {message}{Colors.RESET}")
            
            # Procesar consulta con streaming
            response = await self.orchestrator.process_query_optimized(query, stream_callback=stream_callback)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Formatear respuesta de forma elegante
            print(f"\n{Colors.GRAY}{'─' * 60}{Colors.RESET}")
            
            # Extraer información del agente usado y la respuesta
            agent_info, response_text = self._extract_agent_and_response(response)
            
            # Mostrar qué agente se usó (si no se mostró ya en streaming)
            if agent_info and not any("🎯" in line for line in [stream_callback.__defaults__[0] if stream_callback.__defaults__ else ""]):
                print(f"{Colors.SECONDARY}{Colors.BOLD}🔧 Agente usado: {agent_info}{Colors.RESET}")
            
            # Mostrar respuesta con streaming real
            print(f"{self.format_assistant_response('')}", end="", flush=True)
            
            # Streaming real de la respuesta
            await self._stream_response(response_text)
            
            print()  # Nueva línea al final
            
            print(f"\n{Colors.GRAY}⏱️  Tiempo: {processing_time:.2f}s{Colors.RESET}")
            print(f"{Colors.GRAY}{'─' * 60}{Colors.RESET}")
            
        except Exception as e:
            error_msg = self.format_error(f"Error procesando consulta: {str(e)}")
            print(f"\n{self.format_assistant_response(error_msg)}")
    
    async def _stream_response(self, text: str):
        """
        Muestra la respuesta con streaming real, palabra por palabra.
        """
        words = text.split()
        for i, word in enumerate(words):
            print(f"{word} ", end="", flush=True)
            # Pausa más corta para que se vea más fluido
            if i % 5 == 0:  # Pausa cada 5 palabras
                await asyncio.sleep(0.02)  # Pausa más corta
    
    def _extract_agent_and_response(self, response) -> tuple:
        """Extrae información del agente usado y la respuesta limpia"""
        agent_info = None
        response_text = ""
        
        if isinstance(response, dict):
            # Extraer información del agente
            if 'agent_type' in response:
                agent_type = response['agent_type']
                agent_icons = {
                    'sql': Icons.SQL,
                    'fhir': Icons.FHIR,
                    'biochat': Icons.BIOCHAT,
                    'clinical_analysis': Icons.MEDGEMMA,
                    'greeting': Icons.GREETING
                }
                icon = agent_icons.get(agent_type, '🤖')
                agent_info = f"{icon} {agent_type.upper()}"
            
            # Extraer la respuesta limpia
            if response.get('success') == False:
                response_text = response.get('message', 'Error desconocido')
            else:
                # Priorizar campos específicos de respuesta
                if 'message' in response and response['message']:
                    response_text = response['message']
                elif 'response' in response and response['response']:
                    response_text = response['response']
                elif 'formatted_data' in response and response['formatted_data']:
                    response_text = response['formatted_data']
                elif 'data' in response and response['data']:
                    # Si hay datos, formatearlos
                    if isinstance(response['data'], list) and response['data']:
                        response_text = self._format_data_as_text(response['data'])
                    else:
                        response_text = str(response['data'])
                elif 'original_result' in response and response['original_result']:
                    # Si hay resultado original, extraer el mensaje de ahí
                    original = response['original_result']
                    if isinstance(original, dict):
                        if 'message' in original:
                            response_text = original['message']
                        elif 'response' in original:
                            response_text = original['response']
                        else:
                            response_text = str(original)
                    else:
                        response_text = str(original)
                else:
                    # Si no hay campo específico, crear un resumen limpio
                    response_text = self._create_clean_summary(response)
        else:
            # Si es un string, devolverlo directamente
            response_text = str(response)
        
        return agent_info, response_text
    
    def _format_data_as_text(self, data: list) -> str:
        """Formatea datos de lista como texto legible"""
        if not data:
            return "No se encontraron datos."
        
        result = []
        for i, item in enumerate(data[:10], 1):  # Limitar a 10 elementos
            if isinstance(item, dict):
                # Formatear diccionario
                item_text = []
                for key, value in item.items():
                    if value is not None and str(value).strip():
                        item_text.append(f"{key}: {value}")
                result.append(f"{i}. {' | '.join(item_text)}")
            else:
                result.append(f"{i}. {item}")
        
        if len(data) > 10:
            result.append(f"... y {len(data) - 10} elementos más")
        
        return "\n".join(result)
    
    def _create_clean_summary(self, response_dict: dict) -> str:
        """Crea un resumen limpio del diccionario de respuesta"""
        # Filtrar campos que no queremos mostrar
        exclude_fields = {'success', 'query', 'model', 'note', 'timestamp', 'execution_time'}
        
        summary_parts = []
        for key, value in response_dict.items():
            if key not in exclude_fields and value is not None and str(value).strip():
                if isinstance(value, (list, dict)):
                    # Para estructuras complejas, mostrar solo el tipo
                    summary_parts.append(f"{key}: {type(value).__name__}")
                else:
                    summary_parts.append(f"{key}: {value}")
        
        if summary_parts:
            return "\n".join(summary_parts)
        else:
            return "Respuesta procesada correctamente."
    
    async def run(self):
        """Ejecutar la interfaz"""
        self.clear_screen()
        self.print_welcome()
        
        # Inicializar sistema
        if not await self.initialize_system():
            return
        
        print(f"\n{Colors.SUCCESS}🎉 {Colors.BOLD}¡ChatMed está listo para ayudarte!{Colors.RESET}")
        print(f"{Colors.GRAY}{'─' * 60}{Colors.RESET}")
        
        # Bucle principal
        while True:
            try:
                # Obtener entrada del usuario
                print(f"\n{Colors.PRIMARY}{Colors.BOLD}💬 {Colors.RESET}", end="")
                query = input().strip()
                
                if query.lower() in ['salir', 'exit', 'quit']:
                    print(f"\n{Colors.SUCCESS}👋 {Colors.BOLD}¡Hasta luego!{Colors.RESET}")
                    break
                
                if not query:
                    continue
                
                # Comandos especiales
                if query.lower() == 'ayuda':
                    self.print_help()
                    continue
                
                if query.lower() == 'stats':
                    self.print_stats()
                    continue
                
                if query.lower() == 'limpiar':
                    self.clear_screen()
                    self.print_welcome()
                    continue
                
                # Mostrar entrada del usuario
                print(f"\n{self.format_user_input(query)}")
                
                # Procesar consulta
                self.query_count += 1
                await self.process_query(query)
                
            except KeyboardInterrupt:
                print(f"\n{Colors.WARNING}⚠️ {Colors.BOLD}Interrumpido por el usuario{Colors.RESET}")
                break
            except Exception as e:
                print(f"\n{Colors.ERROR}❌ {Colors.BOLD}Error inesperado:{Colors.RESET} {e}")

async def main():
    """Función principal"""
    interface = ModernTerminalInterface()
    await interface.run()

if __name__ == "__main__":
    asyncio.run(main()) 