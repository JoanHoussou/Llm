"""
Client pour l'API Gemini.
Implémente l'interface LLMProvider pour interagir avec les modèles Google Gemini.
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

class GeminiProvider(LLMProvider):
    """
    Provider pour l'API Gemini.
    Implémente les méthodes de l'interface LLMProvider pour Google Gemini.
    """

    API_URL = "https://generativelanguage.googleapis.com/v1/"

    def __init__(self, config: ModelConfig):
        """
        Initialise le provider Gemini.

        Args:
            config: Configuration du modèle Gemini
        """
        if config.provider != ProviderType.GEMINI:
            raise ValidationError("Configuration invalide pour Gemini")
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
                    headers={"Content-Type": "application/json"}
                )
            await self.validate_credentials()
        except Exception as e:
            if self.session:
                await self.session.close()
                self.session = None
            raise APIError(f"Erreur d'initialisation Gemini: {e}")

    async def chat_completion(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Génère une réponse via l'API Gemini.

        Args:
            messages: Historique de la conversation
            temperature: Contrôle de la créativité (0.0 à 1.0)
            max_tokens: Nombre maximum de tokens en sortie
            stream: Si True, retourne un générateur de réponses

        Returns:
            Réponse du modèle

        Raises:
            APIError: En cas d'erreur d'API
            ValidationError: En cas de paramètres invalides
        """
        if not self.session:
            await self.initialize()

        try:
            # Conversion des messages au format Gemini
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": "user" if msg.role == "user" else "model",
                    "parts": [{"text": msg.content}]
                })

            # Préparation des paramètres
            params = {
                "contents": formatted_messages,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens if max_tokens else 2048,
                }
            }

            endpoint = f"models/{self.config.name}:generateContent"
            url = f"{endpoint}?key={self.config.api_key}"

            if stream:
                return self._stream_response(url, params)

            async with self.session.post(url, json=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise APIError(
                        f"Erreur Gemini ({response.status}): {error_text}"
                    )
                
                data = await response.json()
                if not data.get("candidates"):
                    raise APIError("Pas de réponse générée")
                
                return data["candidates"][0]["content"]["parts"][0]["text"]

        except aiohttp.ClientError as e:
            raise APIError(f"Erreur de communication avec Gemini: {e}")
        except Exception as e:
            raise APIError(f"Erreur inattendue: {e}")

    async def _stream_response(
        self,
        url: str,
        params: dict
    ) -> AsyncGenerator[str, None]:
        """
        Génère un stream de réponses depuis l'API.

        Args:
            url: URL de l'API avec la clé
            params: Paramètres de la requête

        Yields:
            Fragments de la réponse

        Raises:
            APIError: En cas d'erreur pendant le streaming
        """
        params["streamGenerationConfig"] = {"streamType": "tokens"}
        
        try:
            async with self.session.post(url, json=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise APIError(
                        f"Erreur de streaming Gemini ({response.status}): {error_text}"
                    )

                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if candidates := data.get("candidates", []):
                                if parts := candidates[0].get("content", {}).get("parts", []):
                                    if text := parts[0].get("text"):
                                        yield text
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