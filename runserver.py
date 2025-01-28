"""
Script de lancement de l'application Chat LLM.
Exécuter ce script depuis la racine du projet.
"""
import os
import sys
import subprocess
from pathlib import Path

def run_app():
    """Lance l'application Streamlit avec le bon PYTHONPATH."""
    try:
        # Obtient le chemin absolu du répertoire src
        project_root = Path(__file__).parent.absolute()
        src_path = project_root / "src"

        # Définit l'environnement avec le PYTHONPATH mis à jour
        env = os.environ.copy()
        if sys.platform == "win32":
            # Windows utilise un point-virgule comme séparateur
            env["PYTHONPATH"] = f"{src_path};{env.get('PYTHONPATH', '')}"
        else:
            # Linux et MacOS utilisent deux-points comme séparateur
            env["PYTHONPATH"] = f"{src_path}:{env.get('PYTHONPATH', '')}"

        # Commande Streamlit
        cmd = [
            "streamlit",
            "run",
            str(src_path / "main.py"),
            "--server.address=localhost",
            "--server.port=8501"
        ]

        # Lance l'application
        print("Démarrage de l'application Chat LLM...")
        print(f"PYTHONPATH: {env['PYTHONPATH']}")
        subprocess.run(cmd, env=env, check=True)

    except KeyboardInterrupt:
        print("\nArrêt de l'application")
        sys.exit(0)
    except Exception as e:
        print(f"Erreur critique: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_app()