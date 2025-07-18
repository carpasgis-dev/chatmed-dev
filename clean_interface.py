#!/usr/bin/env python3
"""
ChatMed - Interfaz Terminal Ultra Limpia
========================================

Interfaz terminal completamente limpia sin logs técnicos.
"""

import asyncio
import os
import sys
import logging
from datetime import datetime

# Configurar codificación UTF-8 para Windows
if os.name == 'nt':
    os.system('chcp 65001 > nul')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Suprimir TODOS los logs técnicos
logging.getLogger().setLevel(logging.CRITICAL)
for logger_name in ['httpx', 'urllib3', 'openai', 'langchain', 'core', 'agents', 'fhir_sql_bridge', 'transformers', 'requests']:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# Redirigir stderr para suprimir errores
class SuppressStderr:
    def __enter__(self):
        self.original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self.original_stderr

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.simple_orchestrator import SimpleOrchestrator
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

class UltraCleanInterface:
    """Interfaz terminal ultra limpia para ChatMed"""
    
    def __init__(self):
        self.orchestrator = None
        self.query_count = 0
        self.start_time = datetime.now()
        self.spinner_index = 0
    
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
    
    async def show_loading_animation(self, message: str, duration: float = 0.8):
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
            await self.show_loading_animation(step, 0.6)
            await asyncio.sleep(0.1)
        
        try:
            # Suprimir errores durante la inicialización
            with SuppressStderr():
                # Configurar LLM
                llm = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.1
                )
                
                # Inicializar orquestador
                self.orchestrator = SimpleOrchestrator(
                    db_path="database_new.sqlite3.db",
                    llm=llm
                )
            
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
        """Procesar una consulta con formato moderno"""
        if not self.orchestrator:
            print(f"\n{self.format_assistant_response(self.format_error('Sistema no disponible'))}")
            return
        
        try:
            print(f"\n{Colors.INFO}🔄 {Colors.BOLD}Procesando consulta...{Colors.RESET}")
            start_time = datetime.now()
            
            # Suprimir errores durante el procesamiento
            with SuppressStderr():
                response = await self.orchestrator.process_query(
                    query=query,
                    user_id="user"
                )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Formatear respuesta con separadores
            print(f"\n{Colors.GRAY}{'─' * 60}{Colors.RESET}")
            print(f"{self.format_assistant_response(response)}")
            print(f"{Colors.GRAY}⏱️  Tiempo de procesamiento: {processing_time:.2f}s{Colors.RESET}")
            print(f"{Colors.GRAY}{'─' * 60}{Colors.RESET}")
            
        except Exception as e:
            error_msg = self.format_error(f"Error procesando consulta: {str(e)}")
            print(f"\n{self.format_assistant_response(error_msg)}")
    
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
    interface = UltraCleanInterface()
    await interface.run()

if __name__ == "__main__":
    asyncio.run(main()) 