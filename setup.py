"""
Script de configuration pour l'application Chat LLM.
Génère le fichier .env et prépare l'environnement de développement.
"""
import os
import sys
from pathlib import Path
from typing import Dict

def create_env_file(env_path: Path) -> None:
    """
    Crée le fichier .env avec les variables d'environnement par défaut.
    
    Args:
        env_path: Chemin vers le fichier .env
    """
    env_content = """# Configuration Chat LLM
# Clés API (à remplir)
MISTRAL_API_KEY=
GEMINI_API_KEY=
DEEPSEEK_API_KEY=

# Configuration des modèles locaux
LM_STUDIO_URL=http://localhost:1234
OLLAMA_URL=http://localhost:11434

# Configuration de l'application
LOG_LEVEL=INFO
SAVE_HISTORY=true
MAX_HISTORY=100
AUDIO_ENABLED=true

# Configuration du modèle par défaut
DEFAULT_MODEL=mistral
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=1000
"""
    
    env_path.write_text(env_content)
    print(f"\033[92m✓\033[0m Fichier .env créé: {env_path}")

def check_python_version() -> None:
    """Vérifie que la version de Python est compatible."""
    if sys.version_info < (3, 8):
        print("\033[91m✗\033[0m Python 3.8 ou supérieur requis")
        sys.exit(1)
    print(f"\033[92m✓\033[0m Python {sys.version_info.major}.{sys.version_info.minor}")

def create_directories() -> None:
    """Crée les répertoires nécessaires s'ils n'existent pas."""
    dirs = [
        "src/api",
        "src/audio",
        "src/config",
        "src/ui/components",
        "src/ui/pages",
        "tests",
        "logs"
    ]
    
    for dir_path in dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True)
            print(f"\033[92m✓\033[0m Répertoire créé: {path}")

def main() -> None:
    """Point d'entrée du script de configuration."""
    print("\n🔧 Configuration de Chat LLM\n")

    # Vérification de Python
    print("Vérification de l'environnement...")
    check_python_version()

    # Création des répertoires
    print("\nCréation des répertoires...")
    create_directories()

    # Création du fichier .env s'il n'existe pas
    env_path = Path(".env")
    if not env_path.exists():
        print("\nCréation du fichier .env...")
        create_env_file(env_path)
    else:
        print("\n\033[93m!\033[0m Le fichier .env existe déjà")

    print("\n✨ Configuration terminée!\n")
    print("Étapes suivantes:")
    print("1. Remplir les clés API dans le fichier .env")
    print("2. Installer les dépendances: pip install -r requirements.txt")
    print("3. Lancer l'application: python src/main.py\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nConfiguration interrompue")
        sys.exit(1)
    except Exception as e:
        print(f"\n\033[91m✗\033[0m Erreur: {e}")
        sys.exit(1)