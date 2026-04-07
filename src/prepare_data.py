"""
GASPILLOMÈTRE - Préparation des données
========================================
Ce script sera responsable de :
1. Organiser les images brutes en dossiers train/val au format YOLO
2. Vérifier la cohérence entre images et annotations
3. Afficher des statistiques sur le dataset

Usage prévu :
    python src/prepare_data.py
    python src/prepare_data.py --split 0.8 --source imageplateau/
"""

import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def find_annotation(image_path: Path, label_dir: Path) -> Path | None:
    """Retourne le chemin du fichier d'annotation associé à une image, ou None."""
    # TODO: chercher un .txt du même nom dans label_dir
    return None


def validate_annotation(annotation_path: Path) -> bool:
    """Vérifie que le fichier d'annotation respecte le format YOLO.

    Format attendu par ligne : class_id x_center y_center width height
    Toutes les valeurs numériques, coordonnées dans [0, 1].
    """
    # TODO: lire chaque ligne, contrôler le format et les plages de valeurs
    return False


def split_dataset(image_paths: list[Path], split_ratio: float, seed: int) -> tuple[list, list]:
    """Répartit les images en deux listes (train, val) selon split_ratio."""
    # TODO: mélanger avec random.seed(seed) puis découper la liste
    return [], []


def copy_files(image_paths: list[Path], label_dir: Path, dest_dir: Path) -> None:
    """Copie les images et leurs annotations dans dest_dir (structure images/ + labels/)."""
    # TODO: créer dest_dir/images/ et dest_dir/labels/
    # TODO: copier chaque image et son annotation correspondante
    pass


def generate_stats(data_dir: Path) -> None:
    """Affiche des statistiques sur le dataset préparé (nb images, objets par classe…)."""
    # TODO: parcourir les fichiers d'annotation, compter les occurrences par class_id
    # TODO: afficher un résumé lisible dans le terminal
    pass


# ---------------------------------------------------------------------------
# Interface en ligne de commande
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gaspillomètre — préparation des données")
    parser.add_argument("--source", type=Path, default=Path("imageplateau/"), help="Dossier source des images brutes")
    parser.add_argument("--split",  type=float, default=0.8,  help="Part des données pour l'entraînement (défaut : 0.8)")
    parser.add_argument("--seed",   type=int,   default=42,   help="Graine aléatoire pour la reproductibilité")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # TODO: lister les images dans args.source
    image_paths: list[Path] = []

    # TODO: valider chaque annotation avec find_annotation() + validate_annotation()
    # TODO: split_dataset() → listes train_images, val_images
    # TODO: copy_files() pour train et val vers data/train et data/val
    # TODO: generate_stats() pour afficher le résumé final
    pass


if __name__ == "__main__":
    main()
