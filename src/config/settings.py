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

    DEFAULT_CONFIGS = {
        "mistral": {
            "name": "codestral-2501",
            "provider": "mistral",
            "type": "api",
            "parameters": ModelParameters().dict()
        },
        "gemini": {
            "name": "gemini-pro",
            "provider": "gemini",
            "type": "api",
            "parameters": ModelParameters().dict()
        },
        "lm_studio": {
            "name": "llama2",
            "provider": "lm_studio",
            "type": "local",
            "base_url": "http://localhost:1234",
            "parameters": ModelParameters().dict()
        },
        "ollama": {
            "name": "mistral",
            "provider": "ollama",
            "type": "local",
            "base_url": "http://localhost:11434",
            "parameters": ModelParameters().dict()
        }
    }

    def __init__(self):
        """Initialise le gestionnaire de configuration."""
        self.app_dir = Path.home() / ".llm-chat"
        self.config_file = self.app_dir / "config.json"
        self.models_file = self.app_dir / "models.json"
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

    def _config_to_dict(self, config: ModelConfig) -> dict:
        """
        Convertit une configuration ModelConfig en dictionnaire sérialisable.
        
        Args:
            config: Configuration à convertir

        Returns:
            Dictionnaire de la configuration
        """
        return {
            "name": config.name,
            "provider": config.provider.value,
            "type": config.type.value,
            "base_url": config.base_url,
            "parameters": config.parameters,
            "api_key": config.api_key
        }

    def _dict_to_config(self, data: dict) -> ModelConfig:
        """
        Convertit un dictionnaire en ModelConfig.
        
        Args:
            data: Données de configuration

        Returns:
            Instance de ModelConfig
        """
        return ModelConfig(
            name=data["name"],
            provider=ProviderType(data["provider"]),
            type=ModelType(data["type"]),
            base_url=data.get("base_url"),
            parameters=data.get("parameters", ModelParameters().dict()),
            api_key=data.get("api_key")
        )

    def get_model_config(self, provider: ProviderType) -> ModelConfig:
        """
        Récupère la configuration d'un modèle.
        
        Args:
            provider: Type de provider

        Returns:
            Configuration du modèle
        """
        provider_value = provider.value
        
        # Vérifie d'abord s'il existe une configuration sauvegardée
        try:
            if self.models_file.exists():
                saved_configs = json.loads(self.models_file.read_text())
                if provider_value in saved_configs:
                    return self._dict_to_config(saved_configs[provider_value])
        except Exception as e:
            logger.warning(f"Erreur lors du chargement de la configuration du modèle: {e}")

        # Utilise la configuration par défaut
        if provider_value not in self.DEFAULT_CONFIGS:
            raise ValueError(f"Configuration non trouvée pour {provider}")
            
        config = self._dict_to_config(self.DEFAULT_CONFIGS[provider_value])
        
        # Ajoute la clé API si disponible pour les modèles API
        if config.type == ModelType.API:
            api_key = self.get_api_key(provider)
            if api_key:
                config.api_key = api_key

        return config

    def save_model_config(self, config: ModelConfig) -> None:
        """
        Sauvegarde la configuration complète d'un modèle.
        
        Args:
            config: Configuration du modèle à sauvegarder
        """
        try:
            # Charge les configurations existantes
            saved_configs = {}
            if self.models_file.exists():
                saved_configs = json.loads(self.models_file.read_text())
            
            # Met à jour la configuration pour ce provider
            saved_configs[config.provider.value] = self._config_to_dict(config)
            
            # Sauvegarde dans le fichier
            self.models_file.write_text(json.dumps(saved_configs, indent=2))
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la configuration du modèle: {e}")
            raise

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
            # Sauvegarde la configuration mise à jour
            self.save_model_config(config)
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des paramètres: {e}")
            raise
