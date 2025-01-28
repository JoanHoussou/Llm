"""
Point d'entrée principal de l'application Chat LLM.
Configure et lance l'interface utilisateur Streamlit.
"""
import os
import time
import asyncio
from typing import Optional, Dict, List
import streamlit as st
from loguru import logger

from api.base import ProviderType, ModelType, Message, LLMProvider
from api.mistral import MistralProvider
from api.gemini import GeminiProvider
from config.settings import ConfigManager
from audio.manager import AudioManager, AudioError

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Chat LLM",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation des états de session
if "messages" not in st.session_state:
    st.session_state.messages = []

if "audio_manager" not in st.session_state:
    st.session_state.audio_manager = AudioManager()

if "config_manager" not in st.session_state:
    st.session_state.config_manager = ConfigManager()

if "loop" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)

if "selected_provider" not in st.session_state:
    st.session_state.selected_provider = "mistral"

if "current_provider" not in st.session_state:
    st.session_state.current_provider = None

def run_async(coro):
    """
    Exécute une coroutine de manière synchrone.
    
    Args:
        coro: Coroutine à exécuter
        
    Returns:
        Résultat de la coroutine
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # Si un event loop existe déjà
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

async def initialize_provider(provider_type: ProviderType) -> Optional[LLMProvider]:
    """
    Initialise un provider LLM.

    Args:
        provider_type: Type de provider à initialiser

    Returns:
        Instance du provider initialisé ou None en cas d'erreur
    """
    try:
        config = st.session_state.config_manager.get_model_config(provider_type)
        
        if provider_type == ProviderType.MISTRAL:
            provider = MistralProvider(config)
        elif provider_type == ProviderType.GEMINI:
            provider = GeminiProvider(config)
        else:
            st.warning(f"Provider {provider_type} non implémenté")
            return None

        await provider.initialize()
        return provider
    
    except Exception as e:
        st.error(f"Erreur d'initialisation du provider {provider_type}: {e}")
        return None

async def change_provider(new_provider: str):
    """
    Change le provider actuel de manière sécurisée.

    Args:
        new_provider: Nom du nouveau provider
    """
    # On met d'abord à jour le provider sélectionné
    st.session_state.selected_provider = new_provider
    
    # Si un provider est déjà initialisé, on le ferme proprement
    if st.session_state.current_provider:
        try:
            await st.session_state.current_provider.close()
        except Exception as e:
            logger.warning(f"Erreur lors de la fermeture du provider: {e}")
        finally:
            st.session_state.current_provider = None

def render_sidebar():
    """Affiche la barre latérale avec les configurations."""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Sélection du modèle
        provider_type = st.selectbox(
            "Modèle",
            options=[p.value for p in ProviderType],
            format_func=lambda x: x.capitalize(),
            key="provider_selector"
        )

        # Mise à jour du provider sélectionné
        if provider_type != st.session_state.selected_provider:
            run_async(change_provider(provider_type))

        # Configuration du modèle sélectionné
        if provider_type:
            provider = ProviderType(provider_type)
            
            # Affichage des champs selon le type de modèle
            if provider in [ProviderType.MISTRAL, ProviderType.GEMINI, ProviderType.DEEPSEEK]:
                api_key = st.text_input(
                    f"Clé API {provider.value}",
                    type="password",
                    value=st.session_state.config_manager.get_api_key(provider) or ""
                )
                
                if api_key:
                    st.session_state.config_manager.save_api_key(provider, api_key)
                    run_async(change_provider(provider_type))
                    
            elif provider in [ProviderType.LM_STUDIO, ProviderType.OLLAMA]:
                st.info("Configuration du modèle local à venir...")
            
        # Paramètres du modèle
        st.subheader("Paramètres")
        temperature = st.slider(
            "Température",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Contrôle la créativité des réponses"
        )
        st.session_state["temperature"] = temperature
        
        # Paramètres audio
        st.subheader("Audio")
        audio_enabled = st.checkbox(
            "Activer l'audio",
            value=st.session_state.config_manager.app_config.audio_enabled
        )
        
        if audio_enabled != st.session_state.config_manager.app_config.audio_enabled:
            st.session_state.config_manager.app_config.audio_enabled = audio_enabled
            st.session_state.config_manager.save_config()

async def process_chat_response(prompt: str):
    """
    Traite la réponse du chat de manière asynchrone.
    
    Args:
        prompt: Message de l'utilisateur
    """
    try:
        # Récupération du provider actuel
        if not st.session_state.get("current_provider"):
            provider_type = ProviderType(st.session_state.selected_provider)
            st.session_state.current_provider = await initialize_provider(
                provider_type
            )
        
        if not st.session_state.current_provider:
            st.error("Aucun provider LLM initialisé")
            return

        # Conversion des messages de la session en objets Message
        chat_messages: List[Message] = []
        for msg in st.session_state.messages:
            chat_messages.append(
                Message(
                    role=msg["role"],
                    content=msg["content"],
                    timestamp=msg.get("timestamp", time.time())
                )
            )

        # Ajout du nouveau message
        chat_messages.append(
            Message(
                role="user",
                content=prompt,
                timestamp=time.time()
            )
        )

        async with st.session_state.current_provider:
            # Génération de la réponse
            response = await st.session_state.current_provider.chat_completion(
                messages=chat_messages,
                temperature=st.session_state.get("temperature", 0.7)
            )
            
            return response

    except Exception as e:
        st.error(f"Erreur lors de la génération de la réponse: {e}")
        if st.session_state.get("current_provider"):
            await st.session_state.current_provider.close()
            st.session_state.current_provider = None
        return None

def render_chat_interface():
    """Affiche l'interface de chat principale."""
    st.title("💬 Chat LLM")
    
    # Affichage des messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Zone de saisie
    if prompt := st.chat_input("Votre message..."):
        # Ajout du message utilisateur
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": time.time()
        })
        
        with st.chat_message("user"):
            st.write(prompt)

        # Réponse du modèle
        with st.chat_message("assistant"):
            with st.spinner("Réflexion en cours..."):
                response = run_async(process_chat_response(prompt))
                
                if response:
                    st.write(response)
                    # Ajout de la réponse à l'historique
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": time.time()
                    })

def render_audio_controls():
    """Affiche les contrôles audio si activés."""
    if not st.session_state.config_manager.app_config.audio_enabled:
        return

    cols = st.columns(2)
    
    with cols[0]:
        if st.button("🎤 Enregistrer", key="record"):
            try:
                run_async(st.session_state.audio_manager.start_recording())
                st.success("Enregistrement démarré")
            except AudioError as e:
                st.error(f"Erreur d'enregistrement: {e}")
    
    with cols[1]:
        if st.button("⏹️ Stop", key="stop"):
            try:
                audio_data, sample_rate = run_async(
                    st.session_state.audio_manager.stop_recording()
                )
                # TODO: Conversion audio -> texte
                st.success("Enregistrement terminé")
            except AudioError as e:
                st.error(f"Erreur d'arrêt d'enregistrement: {e}")

def main():
    """Point d'entrée principal de l'application."""
    render_sidebar()
    render_chat_interface()
    render_audio_controls()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Erreur critique de l'application: {e}")
        logger.exception("Erreur critique")
