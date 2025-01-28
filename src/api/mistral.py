"""
Client pour l'API Mistral.
Implémente l'interface LLMProvider pour interagir avec les modèles Mistral.
"""
import json
from typing import List, Optional, AsyncGenerator
import aiohttp
from loguru import logger

from api.base import (
    LLMProvider,
    ModelConfig,
    Message,
    APIError,
    ValidationError,
    ProviderType
)

class MistralProvider(LLMProvider):
    """
    Provider pour l'API Mistral.
    Implémente les méthodes de l'interface LLMProvider pour Mistral.
    """

    API_URL = "https://codestral.mistral.ai/v1/"

    def __init__(self, config: ModelConfig):
        """
        Initialise le provider Mistral.

        Args:
            config: Configuration du modèle Mistral
        """
        if config.provider != ProviderType.MISTRAL:
            raise ValidationError("Configuration invalide pour Mistral")
        super().__init__(config)
        self.session = None

    async def __aenter__(self):
        """Context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()

    async def initialize(self) -> None:
        """
        Initialise la session HTTP pour les appels API.
        
        Raises:
            APIError: Si l'initialisation échoue
        """
        try:
            if self.session is None:
                self.session = aiohttp.ClientSession(
                    base_url=self.API_URL,
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json"
                    }
                )
            await self.validate_credentials()
        except Exception as e:
            if self.session:
                await self.session.close()
                self.session = None
            raise APIError(f"Erreur d'initialisation Mistral: {e}")

    async def chat_completion(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Génère une réponse via l'API Mistral.

        Args:
            messages: Historique de la conversation
            temperature: Contrôle de la créativité (0.0 à 1.0)
            max_tokens: Nombre maximum de tokens en sortie
            stream: Si True, retourne un générateur de réponses

        Returns:
            Réponse du modèle ou générateur si stream=True

        Raises:
            APIError: En cas d'erreur d'API
            ValidationError: En cas de paramètres invalides
        """
        if not self.session:
            await self.initialize()

        try:
            # Conversion des messages au format Mistral
            formatted_messages = [
                {
                    "role": msg.role,
                    "content": msg.content
                }
                for msg in messages
            ]

            # Préparation des paramètres
            params = {
                "model": self.config.name,
                "messages": formatted_messages,
                "temperature": temperature,
            }
            
            if max_tokens:
                params["max_tokens"] = max_tokens

            endpoint = "chat/completions"
            if stream:
                return self._stream_response(endpoint, params)

            async with self.session.post(endpoint, json=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise APIError(
                        f"Erreur Mistral ({response.status}): {error_text}"
                    )
                
                data = await response.json()
                return data["choices"][0]["message"]["content"]

        except aiohttp.ClientError as e:
            raise APIError(f"Erreur de communication avec Mistral: {e}")
        except Exception as e:
            raise APIError(f"Erreur inattendue: {e}")

    async def _stream_response(
        self,
        endpoint: str,
        params: dict
    ) -> AsyncGenerator[str, None]:
        """
        Génère un stream de réponses depuis l'API.

        Args:
            endpoint: Point d'entrée API
            params: Paramètres de la requête

        Yields:
            Fragments de la réponse

        Raises:
            APIError: En cas d'erreur pendant le streaming
        """
        params["stream"] = True
        
        try:
            async with self.session.post(endpoint, json=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise APIError(
                        f"Erreur de streaming Mistral ({response.status}): {error_text}"
                    )

                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if choice := data.get("choices", [{}])[0]:
                                if delta := choice.get("delta", {}).get("content"):
                                    yield delta
                        except json.JSONDecodeError:
                            logger.warning(f"Ligne invalide ignorée: {line}")

        except aiohttp.ClientError as e:
            raise APIError(f"Erreur de streaming: {e}")
        except Exception as e:
            raise APIError(f"Erreur inattendue pendant le streaming: {e}")

    async def validate_credentials(self) -> bool:
        """
        Vérifie que les credentials sont valides en testant l'API.
        
        Returns:
            True si les credentials sont valides

        Raises:
            APIError: Si les credentials sont invalides
        """
        try:
            # Test simple avec un message court
            test_message = [
                Message(
                    role="user",
                    content="Test de connexion",
                    timestamp=0
                )
            ]
            await self.chat_completion(
                messages=test_message,
                temperature=0.1,
                max_tokens=10
            )
            return True

        except Exception as e:
            raise APIError(f"Validation des credentials échouée: {e}")

    async def close(self) -> None:
        """Ferme proprement la session."""
        if self.session:
            await self.session.close()
            self.session = None

    def __del__(self):
        """Destructeur pour s'assurer que la session est fermée."""
        if self.session and not self.session.closed:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
            except Exception as e:
                logger.error(f"Erreur lors de la fermeture de la session: {e}")