#!/bin/bash

# --- CONFIGURATION ---

# Ton dépôt GitHub (HTTPS ou SSH)
REPO_URL="https://github.com/cluster2600/ELVIS.git"
BRANCH="main"
COMMIT_MSG="Initial push with Git LFS tracking"

# --- INITIALISATION ---

echo "🔧 Initialisation du dépôt Git..."
git init
git remote add origin "$REPO_URL"
git checkout -b "$BRANCH"

# --- GITIGNORE ---

echo "🛑 Configuration du .gitignore..."
cat <<EOF >> .gitignore
*.pkl
*.pt
*.npy
*.tfevents*
__pycache__/
*.log
.DS_Store
.env
venv/
EOF

# --- GIT LFS ---

echo "📦 Installation et configuration de Git LFS..."
git lfs install
git lfs track "*.pkl"
git lfs track "*.pt"
git lfs track "*.npy"
git lfs track "*.tfevents*"

# --- AJOUT DES FICHIERS ---

echo "📁 Ajout des fichiers au commit..."
git add .
git commit -m "$COMMIT_MSG"

# --- PUSH VERS GITHUB ---

echo "⬆️ Push forcé vers $REPO_URL..."
git push -u -f origin "$BRANCH"

echo "✅ Terminé !"