# Gaspillomètre

> Projet de détection du gaspillage alimentaire en cantine scolaire à l'aide de la vision par ordinateur (YOLOv8).

## Objectif

L'idée est de photographier les plateaux repas en fin de service et d'utiliser un modèle de détection d'objets pour :
- Identifier les aliments restants sur le plateau
- Estimer le poids du gaspillage
- Produire des statistiques pour sensibiliser au gaspillage

## État du projet

Le projet en est à ses **tout débuts**. Pour l'instant, seules les bases sont posées :
- Structure des dossiers
- Fichiers de configuration (classes d'aliments, hyperparamètres)

Rien n'est encore fonctionnel — voir le fichier `TODO.md` pour les prochaines étapes.

## Structure prévue

```
config/              # Fichiers de configuration YAML
data/                # Dataset YOLO (images + labels, train/val)
imageplateau/        # Photos brutes des plateaux (à collecter)
models/              # Modèle entraîné (.pt) — vide pour l'instant
results/             # Résultats d'entraînement et de détection
logs/                # Logs des détections
src/
  prepare_data.py    # Préparation et split du dataset
  train.py           # Entraînement du modèle YOLOv8
  inference.py       # Inférence / détection sur images
```

## Installation (à venir)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Prochaines étapes

Voir [TODO.md](TODO.md)
