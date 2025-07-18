from setuptools import setup, find_packages

setup(
    name="chatmed-v2-flexible",
    version="2.0.0",
    description="Sistema de IA médica multi-agente con capacidades avanzadas de búsqueda biomédica",
    author="Carmen Pascual",
    author_email="chatmed.ai.assistant@gmail.com",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-openai>=0.1.0",
        "openai>=1.0.0",
        "biopython>=1.81",
        "duckduckgo-search>=4.1.1",
        "aiohttp>=3.9.1",
        "asyncio>=3.4.3",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1"
    ],
    python_requires=">=3.9",
) 