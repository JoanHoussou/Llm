# Chat LLM Interface

Interface de chat locale pour interagir avec différents modèles de langage (LLM).

## 🌟 Fonctionnalités

### MVP (Version 1.0)
- 💬 Chat textuel et vocal avec les LLMs
- 🔄 Support des LLMs via API (Mistral, Gemini, DeepSeek)
- 🖥️ Support des LLMs locaux (LM Studio, Ollama)
- 🔑 Gestion des clés API
- 🎯 Interface utilisateur intuitive

### Prochaines versions
- ⚙️ Configuration avancée des paramètres de modèle
- 💾 Sauvegarde des conversations
- 🎨 Thèmes (clair/sombre)
- ⌨️ Raccourcis clavier

## 🏗️ Architecture

### Structure du projet
```
llm-chat/
├── src/
│   ├── api/                    # Interfaces avec les APIs LLM
│   │   ├── mistral.py         # Client Mistral API
│   │   ├── gemini.py          # Client Gemini API
│   │   ├── deepseek.py        # Client DeepSeek API
│   │   └── base.py            # Interface de base LLM
│   ├── local/                  # Gestion des modèles locaux
│   │   ├── lm_studio.py       # Interface LM Studio
│   │   └── ollama.py          # Interface Ollama
│   ├── ui/                    # Interface utilisateur
│   │   ├── components/        # Composants UI réutilisables
│   │   ├── pages/            # Pages de l'application
│   │   └── state.py          # Gestion de l'état UI
│   ├── audio/                # Gestion audio
│   │   ├── recorder.py       # Enregistrement vocal
│   │   └── player.py         # Lecture vocale
│   ├── config/               # Configuration
│   │   ├── settings.py       # Paramètres application
│   │   └── models.py         # Configuration modèles
│   └── utils/                # Utilitaires
│       ├── error_handler.py  # Gestion des erreurs
│       └── logger.py         # Journalisation
├── tests/                    # Tests unitaires et intégration
├── requirements.txt          # Dépendances Python
└── main.py                  # Point d'entrée application

```

### Composants Principaux

1. **LLM Manager**
   - Interface unifiée pour tous les modèles
   - Gestion des API et modèles locaux
   - Configuration et paramétrage

2. **Interface Utilisateur**
   - Zone de chat
   - Sélection des modèles
   - Configuration
   - Gestion des erreurs

3. **Gestionnaire Audio**
   - Enregistrement vocal
   - Synthèse vocale
   - Gestion du flux audio

4. **Gestionnaire de Configuration**
   - Clés API
   - Paramètres modèles
   - Préférences utilisateur

## 🔧 Installation

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## 🚀 Utilisation

```bash
# Lancer l'application
python main.py
```

## 📋 Prérequis

- Python 3.8+
- Clé API pour les modèles en ligne (Mistral, Gemini, DeepSeek)
- LM Studio ou Ollama pour les modèles locaux

## 🤝 Contribution

1. Forkez le projet
2. Créez votre branche (`git checkout -b feature/amazing_feature`)
3. Committez vos changements (`git commit -m 'Add amazing feature'`)
4. Pushez sur la branche (`git push origin feature/amazing_feature`)
5. Ouvrez une Pull Request

## 📝 License

MIT License