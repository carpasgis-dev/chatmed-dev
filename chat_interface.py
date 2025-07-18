#!/usr/bin/env python3
"""
🎨 ChatMed - Interfaz Terminal Moderna
=====================================

Interfaz visual moderna para ChatMed con:
- Colores ANSI y iconos
- Animaciones de carga
- Barra de progreso
- Formato de respuestas elegante
- Comandos visuales
- Estadísticas en tiempo real
"""

import os
import sys
import asyncio
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, Union
import logging

# Configurar codificación UTF-8 para Windows
if os.name == 'nt':
    os.system('chcp 65001 > nul')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Suprimir logs técnicos
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('openai').setLevel(logging.ERROR)

class Colors:
    """Paleta de colores moderna para la interfaz"""
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
    DOTS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    PULSE = ["●", "○", "●", "○", "●"]

class ModernChatInterface:
    """Interfaz moderna para ChatMed"""
    
    def __init__(self):
        self.running = True
        self.session_start = datetime.now()
        self.query_count = 0
        self.orchestrator = None
        self.spinner_index = 0
        self.current_agent = None
        self.processing = False
        
        # Configurar pantalla
        self.clear_screen()
        self.print_welcome()
    
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
{Colors.LIGHT_GRAY}   • {Colors.BOLD}agentes{Colors.RESET} - Información de agentes

{Colors.SECONDARY}🕐 {Colors.BOLD}Sesión iniciada:{Colors.RESET} {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}
{Colors.RESET}"""
        print(welcome)
    
    def show_loading_animation(self, message: str, duration: float = 2.0):
        """Muestra una animación de carga elegante"""
        start_time = time.time()
        while time.time() - start_time < duration:
            for i, char in enumerate(Icons.SPINNER):
                print(f"\r{Colors.INFO}{char} {message}...{Colors.RESET}", end="", flush=True)
                time.sleep(0.1)
                if time.time() - start_time >= duration:
                    break
        print(f"\r{Colors.SUCCESS}✅ {message} completado{Colors.RESET}")
    
    def show_agent_indicator(self, agent_type: str):
        """Muestra indicador visual del agente activo"""
        agent_icons = {
            'sql': Icons.SQL,
            'fhir': Icons.FHIR, 
            'biochat': Icons.BIOCHAT,
            'clinical_analysis': Icons.MEDGEMMA,
            'medgemma': Icons.MEDGEMMA,
            'greeting': Icons.GREETING
        }
        
        agent_names = {
            'sql': 'SQL Agent',
            'fhir': 'FHIR Agent',
            'biochat': 'BioChat Agent', 
            'clinical_analysis': 'MedGemma Agent',
            'medgemma': 'MedGemma Agent',
            'greeting': 'Greeting Agent'
        }
        
        icon = agent_icons.get(agent_type.lower(), Icons.INFO)
        name = agent_names.get(agent_type.lower(), 'Unknown Agent')
        
        print(f"\n{Colors.PRIMARY}🎯 {Colors.BOLD}Agente Activo:{Colors.RESET} {icon} {name}")
        print(f"{Colors.GRAY}{'─' * 60}{Colors.RESET}")
    
    def format_user_input(self, text: str) -> str:
        """Formatea la entrada del usuario"""
        return f"{Colors.WHITE}{Colors.BOLD}👤 Tú:{Colors.RESET} {text}"
    
    def format_assistant_response(self, text: str, agent_type: str = None) -> str:
        """Formatea la respuesta del asistente"""
        agent_icons = {
            'sql': Icons.SQL,
            'fhir': Icons.FHIR,
            'biochat': Icons.BIOCHAT,
            'clinical_analysis': Icons.MEDGEMMA,
            'medgemma': Icons.MEDGEMMA,
            'greeting': Icons.GREETING
        }
        
        icon = agent_icons.get(agent_type.lower() if agent_type else 'unknown', Icons.DOCTOR)
        return f"{Colors.PRIMARY}{Colors.BOLD}🤖 ChatMed:{Colors.RESET} {text}"
    
    def show_processing_status(self, message: str):
        """Muestra estado de procesamiento"""
        char = Icons.SPINNER[self.spinner_index % len(Icons.SPINNER)]
        self.spinner_index += 1
        print(f"\r{Colors.INFO}{char} {message}...{Colors.RESET}", end="", flush=True)
    
    def show_progress_bar(self, current: int, total: int, width: int = 50):
        """Muestra una barra de progreso"""
        progress = int(width * current / total)
        bar = "█" * progress + "░" * (width - progress)
        percentage = int(100 * current / total)
        print(f"\r{Colors.PRIMARY}[{bar}] {percentage}%{Colors.RESET}", end="", flush=True)
    
    def format_sql_result(self, data: list, query: str) -> str:
        """Formatea resultados SQL de manera elegante"""
        if not data:
            return f"{Colors.WARNING}⚠️ No se encontraron resultados para esta consulta.{Colors.RESET}"
        
        result = f"{Colors.SUCCESS}📊 {Colors.BOLD}Resultados encontrados: {len(data)} registros{Colors.RESET}\n"
        result += f"{Colors.GRAY}{'─' * 60}{Colors.RESET}\n"
        
        for i, item in enumerate(data[:5], 1):  # Mostrar solo los primeros 5
            result += f"{Colors.INFO}📋 {Colors.BOLD}Registro {i}:{Colors.RESET}\n"
            for key, value in item.items():
                result += f"   {Colors.LIGHT_GRAY}{key}:{Colors.RESET} {value}\n"
            result += "\n"
        
        if len(data) > 5:
            result += f"{Colors.GRAY}... y {len(data) - 5} registros más{Colors.RESET}\n"
        
        return result
    
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
    
    def show_help(self):
        """Muestra ayuda del sistema"""
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
   • {Colors.BOLD}agentes{Colors.RESET} - Información de agentes
{Colors.RESET}"""
        print(help_text)
    
    def show_stats(self):
        """Muestra estadísticas del sistema"""
        if not self.orchestrator:
            print(f"{Colors.ERROR}❌ Sistema no disponible{Colors.RESET}")
            return
        
        try:
            stats = self.orchestrator.get_performance_metrics()
            
            stats_text = f"""
{Colors.PRIMARY}╔══════════════════════════════════════════════════════════════════════════════╗
║                              📊 ESTADÍSTICAS DEL SISTEMA                               ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}

{Colors.INFO}📈 {Colors.BOLD}Rendimiento:{Colors.RESET}
   • {Colors.BOLD}Consultas totales:{Colors.RESET} {stats.get('total_queries', 0)}
   • {Colors.BOLD}Tasa de cache:{Colors.RESET} {stats.get('cache_hit_rate', '0%')}
   • {Colors.BOLD}Tiempo promedio:{Colors.RESET} {stats.get('avg_response_time', '0.00s')}

{Colors.SUCCESS}🎯 {Colors.BOLD}Sesión Actual:{Colors.RESET}
   • {Colors.BOLD}Consultas procesadas:{Colors.RESET} {self.query_count}
   • {Colors.BOLD}Tiempo activo:{Colors.RESET} {str(datetime.now() - self.session_start)}
   • {Colors.BOLD}Agente actual:{Colors.RESET} {self.current_agent or 'Ninguno'}

{Colors.WARNING}⚡ {Colors.BOLD}Uso de Agentes:{Colors.RESET}
   • {Icons.SQL} SQL Agent: {stats.get('agent_usage', {}).get('sql', 0)} consultas
   • {Icons.FHIR} FHIR Agent: {stats.get('agent_usage', {}).get('fhir', 0)} consultas
   • {Icons.BIOCHAT} BioChat Agent: {stats.get('agent_usage', {}).get('biochat', 0)} consultas
   • {Icons.MEDGEMMA} MedGemma Agent: {stats.get('agent_usage', {}).get('clinical_analysis', 0)} consultas
{Colors.RESET}"""
            print(stats_text)
            
        except Exception as e:
            print(f"{Colors.ERROR}❌ Error obteniendo estadísticas: {e}{Colors.RESET}")
    
    def show_agents_info(self):
        """Muestra información de los agentes"""
        agents_info = f"""
{Colors.PRIMARY}╔══════════════════════════════════════════════════════════════════════════════╗
║                              🤖 INFORMACIÓN DE AGENTES                               ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}

{Icons.SQL} {Colors.BOLD}SQL Agent (Análisis de Datos):{Colors.RESET}
   • Consultas de base de datos médica
   • Búsquedas de pacientes
   • Estadísticas y reportes
   • Análisis de diagnósticos

{Icons.FHIR} {Colors.BOLD}FHIR Agent (Registros Médicos):{Colors.RESET}
   • Gestión de notas clínicas
   • Registro de pacientes
   • Estructuración de datos médicos
   • Interoperabilidad FHIR

{Icons.BIOCHAT} {Colors.BOLD}BioChat Agent (Investigación):{Colors.RESET}
   • Búsquedas en literatura médica
   • Ensayos clínicos
   • Artículos científicos
   • Investigación biomédica

{Icons.MEDGEMMA} {Colors.BOLD}MedGemma Agent (Análisis Clínico):{Colors.RESET}
   • Explicaciones médicas
   • Conceptos de salud
   • Información de medicamentos
   • Diagnósticos básicos

{Icons.GREETING} {Colors.BOLD}Greeting Agent (Interacción):{Colors.RESET}
   • Saludos y conversación
   • Ayuda del sistema
   • Información general
   • Navegación básica

{Colors.INFO}💡 {Colors.BOLD}El sistema selecciona automáticamente el agente más apropiado{Colors.RESET}
{Colors.RESET}"""
        print(agents_info)
    
    async def initialize_system(self):
        """Inicializa el sistema con animación"""
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
            self.show_loading_animation(step, 1.0)
            await asyncio.sleep(0.5)
        
        try:
            # Importar y crear orquestador
            sys.path.append(os.path.dirname(__file__))
            from core.orchestrator_v2 import create_flexible_orchestrator
            
            self.orchestrator = await create_flexible_orchestrator()
            
            print(f"\n{Colors.SUCCESS}✅ {Colors.BOLD}¡Sistema inicializado exitosamente!{Colors.RESET}")
            print(f"{Colors.INFO}💡 {Colors.BOLD}Escribe tu consulta médica o usa 'ayuda' para más información{Colors.RESET}")
            
            return True
            
        except Exception as e:
            print(f"\n{Colors.ERROR}❌ {Colors.BOLD}Error inicializando el sistema:{Colors.RESET}")
            print(f"{Colors.ERROR}{str(e)}{Colors.RESET}")
            return False
    
    def get_user_input(self) -> str:
        """Obtiene entrada del usuario con prompt elegante"""
        try:
            print(f"\n{Colors.PRIMARY}{Colors.BOLD}💬 {Colors.RESET}", end="")
            user_input = input().strip()
            return user_input
        except KeyboardInterrupt:
            return "salir"
        except EOFError:
            return "salir"
    
    async def run_chat(self):
        """Ejecuta el chat principal"""
        # Inicializar sistema
        if not await self.initialize_system():
            return
        
        print(f"\n{Colors.SUCCESS}🎉 {Colors.BOLD}¡ChatMed está listo para ayudarte!{Colors.RESET}")
        print(f"{Colors.GRAY}{'─' * 60}{Colors.RESET}")
        
        while self.running:
            try:
                # Obtener entrada del usuario
                user_input = self.get_user_input()
                
                if not user_input:
                    continue
                
                # Comandos del sistema
                if user_input.lower() in ['salir', 'exit', 'quit']:
                    self.show_farewell()
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
                elif user_input.lower() == 'agentes':
                    self.show_agents_info()
                    continue
                
                # Procesar consulta
                await self.process_query(user_input)
                
            except KeyboardInterrupt:
                print(f"\n{Colors.WARNING}⚠️ {Colors.BOLD}Interrumpido por el usuario{Colors.RESET}")
                break
            except Exception as e:
                print(f"\n{Colors.ERROR}❌ {Colors.BOLD}Error inesperado:{Colors.RESET} {e}")
    
    async def process_query(self, query: str):
        """Procesa una consulta del usuario"""
        self.query_count += 1
        self.processing = True
        
        # Mostrar entrada del usuario
        print(f"\n{self.format_user_input(query)}")
        
        try:
            # Mostrar procesamiento
            print(f"\n{Colors.INFO}🔄 {Colors.BOLD}Procesando consulta...{Colors.RESET}")
            
            if self.orchestrator:
                # Procesar con callback para mostrar progreso
                response = await self.orchestrator.process_query_optimized(
                    query, 
                    stream_callback=self.show_processing_status
                )
                
                # Formatear respuesta
                formatted_response = self.format_response(response)
                print(f"\n{self.format_assistant_response(formatted_response)}")
                
            else:
                print(f"\n{self.format_assistant_response(self.format_error('Sistema no disponible'))}")
                
        except Exception as e:
            error_msg = self.format_error(f"Error procesando consulta: {str(e)}")
            print(f"\n{self.format_assistant_response(error_msg)}")
        
        finally:
            self.processing = False
            print(f"\n{Colors.GRAY}{'─' * 60}{Colors.RESET}")
    
    def format_response(self, response: Union[str, Dict[str, Any]]) -> str:
        """Formatea la respuesta del sistema"""
        if isinstance(response, dict):
            if not response.get('success', False):
                return self.format_error(response.get('message', 'Error desconocido'))
            
            # Respuesta exitosa
            message = response.get('message', 'Operación completada')
            data = response.get('data', [])
            sql_query = response.get('sql', '')
            
            # Formatear según el tipo de respuesta
            if data and isinstance(data, list):
                return self.format_sql_result(data, message)
            elif sql_query:
                return f"{message}\n\n{Colors.GRAY}SQL ejecutado: {sql_query}{Colors.RESET}"
            else:
                return message
        else:
            return str(response)
    
    def show_farewell(self):
        """Muestra mensaje de despedida"""
        farewell = f"""
{Colors.PRIMARY}╔══════════════════════════════════════════════════════════════════════════════╗
║                              👋 ¡GRACIAS POR USAR CHATMED!                               ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}

{Colors.SUCCESS}📊 {Colors.BOLD}Resumen de la sesión:{Colors.RESET}
   • {Colors.BOLD}Consultas procesadas:{Colors.RESET} {self.query_count}
   • {Colors.BOLD}Tiempo de sesión:{Colors.RESET} {str(datetime.now() - self.session_start)}
   • {Colors.BOLD}Agentes utilizados:{Colors.RESET} Todos los agentes especializados

{Colors.INFO}💡 {Colors.BOLD}ChatMed v2.0 - Sistema Médico Inteligente{Colors.RESET}
{Colors.GRAY}   Desarrollado con tecnología de IA avanzada para asistencia médica{Colors.RESET}

{Colors.SUCCESS}✅ {Colors.BOLD}¡Hasta la próxima!{Colors.RESET}
{Colors.RESET}"""
        print(farewell)

async def main():
    """Función principal"""
    interface = ModernChatInterface()
    await interface.run_chat()

if __name__ == "__main__":
    asyncio.run(main()) 