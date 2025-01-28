"""
Interface de base pour les modèles LLM.
Définit le contrat que tous les clients LLM doivent implémenter.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from enum import Enum

class ModelType(Enum):
    """Types de modèles supportés."""
    API = "api"
    LOCAL = "local"

class ProviderType(Enum):
    """Fournisseurs de modèles supportés."""
    MISTRAL = "mistral"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    LM_STUDIO = "lm_studio"
    OLLAMA = "ollama"

@dataclass
class ModelConfig:
    """Configuration d'un modèle LLM."""
    name: str
    provider: ProviderType
    type: ModelType
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    parameters: Dict[str, Any] = None

@dataclass
class Message:
    """Représente un message dans la conversation."""
    role: str  # 'user' ou 'assistant'
    content: str
    timestamp: float

class LLMError(Exception):
    """Exception de base pour les erreurs LLM."""
    pass

class APIError(LLMError):
    """Erreurs liées aux appels API."""
    pass

class ValidationError(LLMError):
    """Erreurs de validation des entrées/sorties."""
    pass

class LLMProvider(ABC):
    """
    Interface abstraite pour les fournisseurs de modèles LLM.
    Chaque fournisseur (Mistral, Gemini, etc.) doit implémenter cette interface.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialise le provider avec sa configuration.
        
        Args:
            config: Configuration du modèle
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialise la connexion avec le modèle.
        À implémenter par chaque provider.
        """
        pass

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Génère une complétion basée sur l'historique des messages.

        Args:
            messages: Liste des messages de la conversation
            temperature: Contrôle la créativité (0.0 à 1.0)
            max_tokens: Nombre maximum de tokens en sortie
            stream: Si True, retourne un générateur de réponses partielles

        Returns:
            La réponse du modèle

        Raises:
            APIError: En cas d'erreur d'API
            ValidationError: En cas de paramètres invalides
        """
        pass

    @abstractmethod
    async def validate_credentials(self) -> bool:
        """
        Vérifie que les credentials sont valides.
        
        Returns:
            True si les credentials sont valides
        
        Raises:
            APIError: En cas d'erreur de validation
        """
        pass

    def _validate_config(self) -> None:
        """
        Valide la configuration du provider.
        
        Raises:
            ValidationError: Si la configuration est invalide
        """
        if self.config.type == ModelType.API and not self.config.api_key:
            raise ValidationError(f"API key requise pour {self.config.provider}")
        
        if self.config.type == ModelType.LOCAL and not self.config.base_url:
            raise ValidationError(f"URL de base requise pour {self.config.provider}")