"""
Provider pour LM Studio.
Permet d'utiliser des modèles via l'API locale de LM Studio.
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

class LMStudioProvider(LLMProvider):
    """Implémentation du provider pour LM Studio."""
    
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
        """Initialise la connexion avec LM Studio."""
        try:
            # Valide la connexion en testant l'API
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config.base_url}/v1/models") as resp:
                    if resp.status != 200:
                        error_detail = await resp.text()
                        logger.error(f"Réponse LM Studio models ({resp.status}): {error_detail}")
                        raise APIError(f"Erreur de connexion à LM Studio: {resp.status}")
                    data = await resp.json()
                    logger.info(f"Modèles LM Studio disponibles: {json.dumps(data, indent=2)}")
        except Exception as e:
            raise APIError(f"Erreur d'initialisation LM Studio: {str(e)}")

    async def chat_completion(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Génère une complétion via l'API LM Studio.

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

        # Conversion des messages au format attendu par LM Studio
        formatted_messages = [
            {
                "role": msg.role,
                "content": msg.content
            } for msg in messages
        ]

        try:
            async with aiohttp.ClientSession() as session:
                request_data = {
                    "messages": formatted_messages,
                    "model": self.config.name,
                    "temperature": temperature,
                    "max_tokens": max_tokens if max_tokens else 2048,
                    "stream": False
                }
                logger.debug(f"Requête LM Studio: {json.dumps(request_data, indent=2)}")

                async with session.post(
                    f"{self.config.base_url}/v1/chat/completions",
                    json=request_data
                ) as resp:
                    if resp.status != 200:
                        error_detail = await resp.text()
                        logger.error(f"Erreur LM Studio ({resp.status}): {error_detail}")
                        raise APIError(f"Erreur LM Studio: {resp.status} - {error_detail}")
                    
                    data = await resp.json()
                    logger.debug(f"Réponse LM Studio: {json.dumps(data, indent=2)}")
                    return data["choices"][0]["message"]["content"]

        except Exception as e:
            raise APIError(f"Erreur lors de la génération: {str(e)}")

    async def validate_credentials(self) -> bool:
        """
        Vérifie que la connexion à LM Studio est fonctionnelle.
        
        Returns:
            True si la connexion est valide
        """
        try:
            await self.initialize()
            return True
        except Exception:
            return False
