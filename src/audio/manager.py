"""
Gestionnaire des fonctionnalités audio.
Gère l'enregistrement vocal et la synthèse vocale.
"""
import os
import time
import wave
import tempfile
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.io import wavfile
import librosa
from loguru import logger

class AudioError(Exception):
    """Exception de base pour les erreurs audio."""
    pass

class AudioManager:
    """
    Gère les fonctionnalités d'enregistrement et de lecture audio.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 1,
        dtype: np.dtype = np.int16
    ):
        """
        Initialise le gestionnaire audio.

        Args:
            sample_rate: Taux d'échantillonnage en Hz
            channels: Nombre de canaux audio
            dtype: Type de données pour l'enregistrement
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self._recording = False
        self._frames = []
        self._temp_dir = Path(tempfile.gettempdir()) / "llm-chat-audio"
        self._temp_dir.mkdir(parents=True, exist_ok=True)

    async def start_recording(self) -> None:
        """
        Démarre l'enregistrement audio.

        Raises:
            AudioError: Si l'enregistrement ne peut pas démarrer
        """
        try:
            self._recording = True
            self._frames = []

            def callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Status audio: {status}")
                if self._recording:
                    self._frames.append(indata.copy())

            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                callback=callback
            )
            self.stream.start()
            logger.info("Enregistrement audio démarré")

        except Exception as e:
            self._recording = False
            raise AudioError(f"Erreur lors du démarrage de l'enregistrement: {e}")

    async def stop_recording(self) -> Tuple[np.ndarray, int]:
        """
        Arrête l'enregistrement et retourne les données audio.

        Returns:
            Tuple contenant:
                - Les données audio (numpy array)
                - Le taux d'échantillonnage

        Raises:
            AudioError: Si l'arrêt de l'enregistrement échoue
        """
        if not self._recording:
            raise AudioError("Aucun enregistrement en cours")

        try:
            self._recording = False
            self.stream.stop()
            self.stream.close()

            if not self._frames:
                raise AudioError("Aucune donnée audio enregistrée")

            # Combine tous les frames en un seul array
            audio_data = np.concatenate(self._frames, axis=0)
            logger.info(f"Enregistrement audio terminé: {len(audio_data)} échantillons")
            
            return audio_data, self.sample_rate

        except Exception as e:
            raise AudioError(f"Erreur lors de l'arrêt de l'enregistrement: {e}")

    def save_audio(self, audio_data: np.ndarray, filename: Optional[str] = None) -> Path:
        """
        Sauvegarde les données audio dans un fichier.

        Args:
            audio_data: Données audio à sauvegarder
            filename: Nom du fichier (optionnel)

        Returns:
            Chemin vers le fichier audio sauvegardé

        Raises:
            AudioError: Si la sauvegarde échoue
        """
        try:
            if filename is None:
                filename = f"audio_{int(time.time())}.wav"
            
            file_path = self._temp_dir / filename
            
            with wave.open(str(file_path), 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(np.dtype(self.dtype).itemsize)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())

            logger.info(f"Audio sauvegardé: {file_path}")
            return file_path

        except Exception as e:
            raise AudioError(f"Erreur lors de la sauvegarde audio: {e}")

    async def text_to_speech(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Convertit du texte en audio.
        Note: Cette implémentation est un placeholder.
        Dans une version réelle, nous utiliserions un service TTS comme gTTS.

        Args:
            text: Texte à convertir en audio

        Returns:
            Tuple contenant:
                - Les données audio (numpy array)
                - Le taux d'échantillonnage

        Raises:
            AudioError: Si la conversion échoue
        """
        # TODO: Implémenter avec un vrai service TTS
        raise NotImplementedError("La synthèse vocale n'est pas encore implémentée")

    def play_audio(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """
        Joue les données audio.

        Args:
            audio_data: Données audio à jouer
            sample_rate: Taux d'échantillonnage

        Raises:
            AudioError: Si la lecture échoue
        """
        try:
            sd.play(audio_data, sample_rate)
            sd.wait()  # Attend la fin de la lecture
            logger.info("Lecture audio terminée")
        
        except Exception as e:
            raise AudioError(f"Erreur lors de la lecture audio: {e}")

    def cleanup(self) -> None:
        """Nettoie les fichiers temporaires."""
        try:
            for file in self._temp_dir.glob("*.wav"):
                file.unlink()
            logger.info("Nettoyage des fichiers audio temporaires effectué")
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage des fichiers temporaires: {e}")