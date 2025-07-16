"""
Factory for creating embedding providers.
"""

import os
from typing import Dict, Type, Optional
import logging

from .base_provider import BaseEmbeddingProvider, EmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider
from .huggingface_provider import HuggingFaceEmbeddingProvider
from .ollama_provider import OllamaEmbeddingProvider


class EmbeddingProviderFactory:
    """Factory for creating embedding providers"""
    
    _providers: Dict[EmbeddingProvider, Type[BaseEmbeddingProvider]] = {
        EmbeddingProvider.OPENAI: OpenAIEmbeddingProvider,
        EmbeddingProvider.HUGGINGFACE: HuggingFaceEmbeddingProvider,
        EmbeddingProvider.OLLAMA: OllamaEmbeddingProvider,
        EmbeddingProvider.SENTENCE_TRANSFORMERS: HuggingFaceEmbeddingProvider,
    }
    
    @classmethod
    def create_provider(cls, provider_type: str, model_name: str, **kwargs) -> BaseEmbeddingProvider:
        """
        Create an embedding provider
        
        Args:
            provider_type: Type of provider ('openai', 'huggingface', 'ollama', etc.)
            model_name: Name of the embedding model
            **kwargs: Provider-specific configuration
            
        Returns:
            Initialized embedding provider
        """
        logger = logging.getLogger(__name__)
        
        try:
            provider_enum = EmbeddingProvider(provider_type.lower())
        except ValueError:
            raise ValueError(f"Unsupported provider type: {provider_type}")
        
        provider_class = cls._providers.get(provider_enum)
        if not provider_class:
            raise ValueError(f"No implementation found for provider: {provider_type}")
        
        if provider_enum == EmbeddingProvider.OPENAI:
            api_key = kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
            kwargs['api_key'] = api_key
            
        elif provider_enum == EmbeddingProvider.HUGGINGFACE:
            kwargs.setdefault('device', 'cpu')
            
        elif provider_enum == EmbeddingProvider.OLLAMA:
            kwargs.setdefault('host', 'http://localhost:11434')
        
        try:
            provider = provider_class(model_name, **kwargs)
            provider.initialize()
            
            logger.info(f"Successfully created {provider_type} provider with model {model_name}")
            return provider
            
        except Exception as e:
            logger.error(f"Error creating {provider_type} provider: {e}")
            raise
    
    @classmethod
    def get_supported_providers(cls) -> list:
        """Get list of supported providers"""
        return [provider.value for provider in cls._providers.keys()]
    
    @classmethod
    def register_provider(cls, provider_type: EmbeddingProvider, provider_class: Type[BaseEmbeddingProvider]):
        """Register a new provider"""
        cls._providers[provider_type] = provider_class 