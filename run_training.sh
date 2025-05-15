#!/bin/bash

# Arrêter l'exécution du script à la moindre erreur
set -e

# Fonction de gestion des erreurs : affiche un message et désactive l'environnement virtuel si nécessaire
function handle_error() {
    echo "❌ An error occurred during the training process."
    deactivate || true
    exit 1
}

# Déclenche la fonction handle_error en cas d'erreur
trap 'handle_error' ERR

# Vérifie que Python 3.10 est installé
if ! command -v python3.10 &> /dev/null; then
    echo "Python 3.10 is not installed. Please install Python 3.10 first."
    exit 1
fi

# Crée l'environnement virtuel s'il n'existe pas déjà
if [ ! -d "venv310" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv venv310
fi

# Active l'environnement virtuel
echo "Activating virtual environment..."
source venv310/bin/activate

# Installe les dépendances depuis le fichier requirements.txt
echo "Installing required packages from requirements.txt..."
pip install -r requirements.txt

# Ajoute le répertoire courant au PYTHONPATH pour que les imports fonctionnent
export PYTHONPATH=$(pwd):$PYTHONPATH

# Start model training using module syntax to fix import errors
echo "Starting model training..."
python -m training.train_models

echo "✅ Model training completed."

# Désactive l'environnement virtuel après l'entraînement
echo "Deactivating virtual environment..."
deactivate
