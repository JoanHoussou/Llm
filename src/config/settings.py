"""
Gestion de la configuration de l'application.
Gère le stockage sécurisé des clés API et la configuration des modèles.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import keyring
from loguru import logger
from pydantic import BaseModel, SecretStr

from api.base import ModelType, ProviderType, ModelConfig

class APIKeyConfig(BaseModel):
    """Configuration des clés API."""
    key: SecretStr
    provider: ProviderType
    is_valid: bool = False

class ModelParameters(BaseModel):
    """Paramètres configurables d'un modèle."""
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

class AppConfig(BaseModel):
    """Configuration globale de l'application."""
    theme: str = "light"
    audio_enabled: bool = True
    save_history: bool = True
    history_path: Path = Path.home() / ".llm-chat" / "history"
    max_history: int = 100

class ConfigManager:
    """
    Gestionnaire de configuration de l'application.
    Gère le stockage sécurisé des clés API et les paramètres des modèles.
    """

    def __init__(self):
        """Initialise le gestionnaire de configuration."""
        self.app_dir = Path.home() / ".llm-chat"
        self.config_file = self.app_dir / "config.json"
        self.app_config = AppConfig()
        self._initialize_directories()
        self._load_config()

    def _initialize_directories(self) -> None:
        """Crée les répertoires nécessaires s'ils n'existent pas."""
        self.app_dir.mkdir(parents=True, exist_ok=True)
        self.app_config.history_path.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> None:
        """Charge la configuration depuis le fichier."""
        try:
            if self.config_file.exists():
                config_data = json.loads(self.config_file.read_text())
                self.app_config = AppConfig(**config_data)
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            # Utilise la configuration par défaut

    def save_config(self) -> None:
        """Sauvegarde la configuration dans le fichier."""
        try:
            self.config_file.write_text(
                json.dumps(asdict(self.app_config), indent=2)
            )
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la configuration: {e}")

    def get_api_key(self, provider: ProviderType) -> Optional[str]:
        """
        Récupère la clé API pour un provider.
        
        Args:
            provider: Type de provider

        Returns:
            La clé API si elle existe
        """
        try:
            return keyring.get_password("llm-chat", provider.value)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la clé API: {e}")
            return None

    def save_api_key(self, provider: ProviderType, key: str) -> None:
        """
        Enregistre une clé API de manière sécurisée.
        
        Args:
            provider: Type de provider
            key: Clé API à sauvegarder
        """
        try:
            keyring.set_password("llm-chat", provider.value, key)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la clé API: {e}")
            raise

    def get_model_config(self, provider: ProviderType) -> ModelConfig:
        """
        Récupère la configuration d'un modèle.
        
        Args:
            provider: Type de provider

        Returns:
            Configuration du modèle
        """
        # Configuration par défaut selon le provider
        config = {
            ProviderType.MISTRAL: ModelConfig(
                name="codestral-2501",  # Utilisation du modèle Codestral
                provider=ProviderType.MISTRAL,
                type=ModelType.API,
                parameters=ModelParameters().dict()
            ),
            ProviderType.GEMINI: ModelConfig(
                name="gemini-pro",
                provider=ProviderType.GEMINI,
                type=ModelType.API,
                parameters=ModelParameters().dict()
            ),
            # Ajouter d'autres configurations par défaut
        }
        
        base_config = config.get(provider)
        if not base_config:
            raise ValueError(f"Configuration non trouvée pour {provider}")

        # Ajoute la clé API si disponible
        api_key = self.get_api_key(provider)
        if api_key:
            base_config.api_key = api_key

        return base_config

    def update_model_parameters(
        self, 
        provider: ProviderType, 
        parameters: Dict[str, Any]
    ) -> None:
        """
        Met à jour les paramètres d'un modèle.
        
        Args:
            provider: Type de provider
            parameters: Nouveaux paramètres
        """
        try:
            config = self.get_model_config(provider)
            config.parameters.update(parameters)
            # Sauvegarde dans le fichier de configuration
            self.save_config()
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des paramètres: {e}")
            raise