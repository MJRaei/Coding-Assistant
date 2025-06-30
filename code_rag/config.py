"""
Configuration settings for the Code RAG system.
"""

import os
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# OpenAI Configuration
DEFAULT_EMBEDDING_MODEL = os.getenv('DEFAULT_EMBEDDING_MODEL', 'text-embedding-3-small')
DEFAULT_CHAT_MODEL = os.getenv('DEFAULT_CHAT_MODEL', 'gpt-4o-mini')

EMBEDDING_MODELS = {
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
    'text-embedding-ada-002': 1536
}

DEFAULT_EXCLUDED_DIRS: List[str] = os.getenv(
    'EXCLUDED_DIRS', 
    '__pycache__,.git,.pytest_cache,node_modules,.venv,venv,.env,dist,build,.idea,.vscode'
).split(',')

DEFAULT_INCLUDED_EXTENSIONS: List[str] = os.getenv(
    'INCLUDED_EXTENSIONS',
    '.py,.md,.txt,.yml,.yaml,.json,.sql,.js,.ts,.jsx,.tsx,.css,.scss,.html,.xml,.toml,.cfg,.ini'
).split(',')

DEFAULT_MAX_TOKENS = int(os.getenv('DEFAULT_MAX_TOKENS', '2000'))
DEFAULT_OVERLAP_TOKENS = int(os.getenv('DEFAULT_OVERLAP_TOKENS', '200'))

DEFAULT_SEARCH_K = int(os.getenv('DEFAULT_SEARCH_K', '5'))
DEFAULT_MAX_CHUNKS_FOR_CONTEXT = int(os.getenv('DEFAULT_MAX_CHUNKS_FOR_CONTEXT', '5'))

OPENAI_BATCH_SIZE = int(os.getenv('OPENAI_BATCH_SIZE', '100'))
OPENAI_RATE_LIMIT_DELAY = float(os.getenv('OPENAI_RATE_LIMIT_DELAY', '0.1'))

DEFAULT_LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
VERBOSE_LOGGING = os.getenv('VERBOSE_LOGGING', 'false').lower() == 'true'

def get_openai_api_key() -> str:
    """Get OpenAI API key from environment variable."""
    api_key = os.getenv('OPENAI_API_KEY', '')
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "Please set it in your .env file or environment."
        )
    return api_key

def get_data_dir() -> str:
    """Get data directory path from environment or use default."""
    return os.getenv('CODE_RAG_DATA_DIR', './data')

def get_indices_dir() -> str:
    """Get indices directory path."""
    return os.getenv('CODE_RAG_INDICES_DIR', os.path.join(get_data_dir(), 'indices'))

def get_logs_dir() -> str:
    """Get logs directory path."""
    return os.getenv('CODE_RAG_LOGS_DIR', os.path.join(get_data_dir(), 'logs'))

def get_log_file() -> str:
    """Get log file path."""
    return os.getenv('LOG_FILE', os.path.join(get_logs_dir(), 'code_rag.log'))

def validate_embedding_model(model_name: str) -> bool:
    """Validate if the embedding model is supported."""
    return model_name in EMBEDDING_MODELS

def get_embedding_dimension(model_name: str) -> int:
    """Get the dimension for the specified embedding model."""
    if not validate_embedding_model(model_name):
        raise ValueError(f"Unsupported embedding model: {model_name}")
    return EMBEDDING_MODELS[model_name]

def ensure_directories_exist():
    """Create necessary directories if they don't exist."""
    import os
    from pathlib import Path
    
    dirs_to_create = [
        get_data_dir(),
        get_indices_dir(),
        get_logs_dir()
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True) 