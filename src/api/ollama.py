"""
Provider pour Ollama.
Permet d'utiliser des modèles via l'API locale d'Ollama.
"""
import json
import aiohttp
from typing import List, Optional
from loguru import logger

from .base import (
    LLMProvider, 
    ModelConfig, 
    Message, 
    APIError,
    ValidationError
)

class OllamaProvider(LLMProvider):
    """Implémentation du provider pour Ollama."""
    
    async def __aenter__(self):
        """Support du context manager asynchrone."""
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Support du context manager asynchrone."""
        await self.close()

    async def close(self):
        """Ferme proprement la connexion."""
        pass

    async def initialize(self) -> None:
        """Initialise la connexion avec Ollama."""
        try:
            # Vérifie que le modèle est disponible
            logger.debug(f"Vérification du modèle Ollama: {self.config.name}")
            async with aiohttp.ClientSession() as session:
                # Vérifier que l'API est accessible
                async with session.get(f"{self.config.base_url}/api/version") as resp:
                    if resp.status != 200:
                        raise APIError("Impossible de se connecter à Ollama")
                    version_data = await resp.json()
                    logger.info(f"Version Ollama: {version_data.get('version')}")
                
                # Vérifier que le modèle est disponible
                async with session.post(
                    f"{self.config.base_url}/api/show",
                    json={"name": self.config.name},
                    headers={"Content-Type": "application/json"}
                    ) as resp:
                        if resp.status != 200:
                            error_detail = await resp.text()
                            logger.error(f"Erreur Ollama show ({resp.status}): {error_detail}")
                            raise APIError(f"Modèle {self.config.name} non disponible: {resp.status}")
                        data = await resp.json()
                        logger.info(f"Modèle Ollama chargé: {self.config.name}")
                        logger.debug(f"Détails du modèle: {json.dumps(data, indent=2, ensure_ascii=False)}")
        except Exception as e:
            logger.error(f"Erreur d'initialisation Ollama: {str(e)}")
            raise APIError(f"Erreur d'initialisation Ollama: {str(e)}")

    def _format_messages(self, messages: List[Message]) -> List[dict]:
        """
        Formate les messages pour Ollama.
        
        Args:
            messages: Liste des messages

        Returns:
            Liste des messages formatés
        """
        formatted = []
        for msg in messages:
            formatted.append({
                "role": "assistant" if msg.role == "assistant" else "user",
                "content": msg.content
            })
        logger.debug(f"Messages formatés: {json.dumps(formatted, indent=2)}")
        return formatted

    async def chat_completion(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Génère une complétion via l'API Ollama.

        Args:
            messages: Liste des messages de la conversation
            temperature: Contrôle de la créativité (0.0 à 1.0)
            max_tokens: Nombre maximum de tokens en sortie
            stream: Si True, retourne un générateur de réponses

        Returns:
            La réponse du modèle
        """
        if not messages:
            raise ValidationError("La liste des messages ne peut pas être vide")

        try:
            formatted_messages = self._format_messages(messages)

            # Configuration de la requête
            request_data = {
                "model": self.config.name,
                "messages": formatted_messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                }
            }
            logger.debug(f"Requête Ollama: {json.dumps(request_data, indent=2, ensure_ascii=False)}")

            # Appel de l'API chat
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.base_url}/api/chat",
                    json=request_data,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    }
                ) as resp:
                    if resp.status != 200:
                        error_detail = await resp.text()
                        logger.error(f"Erreur Ollama ({resp.status}): {error_detail}")
                        raise APIError(f"Erreur Ollama: {resp.status} - {error_detail}")
                    
                    data = await resp.json()
                    logger.debug(f"Réponse Ollama: {json.dumps(data, indent=2, ensure_ascii=False)}")
                    
                    # La réponse est dans message.content
                    if "message" in data and "content" in data["message"]:
                        return data["message"]["content"]
                    else:
                        raise APIError("Format de réponse Ollama invalide")

        except Exception as e:
            logger.error(f"Erreur lors de la génération: {str(e)}")
            raise APIError(f"Erreur lors de la génération: {str(e)}")

    async def validate_credentials(self) -> bool:
        """
        Vérifie que la connexion à Ollama est fonctionnelle.
        
        Returns:
            True si la connexion est valide
        """
        try:
            await self.initialize()
            return True
        except Exception:
            return False
