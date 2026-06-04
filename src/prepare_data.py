"""
GASPILLOMÈTRE - Préparation des données
========================================
Organise les images en train/val et copie les labels associés.

Usage:
    python src/prepare_data.py
    python src/prepare_data.py --split 0.8  # 80% train, 20% val
"""

import sys
import shutil
import random
import argparse
from pathlib import Path

# Définition des chemins de base du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_SOURCE_ROOT = PROJECT_ROOT


def creer_dossiers():
    """Crée l'arborescence YOLO attendue."""
    # Liste des sous-dossiers requis pour YOLO (images et labels, séparés en train/val)
    dossiers = [
        DATA_DIR / "images" / "train",
        DATA_DIR / "images" / "val",
        DATA_DIR / "labels" / "train",
        DATA_DIR / "labels" / "val",
    ]
    for dossier in dossiers:
        dossier.mkdir(parents=True, exist_ok=True)
        print(f"  {dossier.relative_to(PROJECT_ROOT)}")
    return dossiers


def trouver_images(dossier_source: Path) -> list:
    """Trouve toutes les images dans un dossier."""
    extensions = {".jpg", ".jpeg", ".png"}
    images = []
    # Parcourt tous les fichiers du dossier par ordre alphabétique
    for fichier in sorted(dossier_source.iterdir()):
        # Vérifie si l'extension du fichier (en minuscules) est une image valide
        if fichier.suffix.lower() in extensions:
            images.append(fichier)
    return images


def trouver_annotation(chemin_image: Path, dossier_labels: Path) -> Path | None:
    """Cherche le fichier d'annotation YOLO correspondant a une image."""
    fichier_label = dossier_labels / f"{chemin_image.stem}.txt"
    return fichier_label if fichier_label.exists() else None


def separer_jeu_donnees(images: list, labels: dict, ratio_train: float = 0.8, graine: int = 42):
    """Sépare les images annotées en ensembles train/val."""
    # Fixer la graine pour obtenir toujours la même séparation sur un même jeu de données
    random.seed(graine)

    # Identifier les images possédant une annotation et celles qui n'en ont pas
    annotated = [(img, labels[img]) for img in images if img in labels]
    unannotated = [img for img in images if img not in labels]

    # Mélanger puis diviser en fonction du ratio d'entraînement (par déf. 0.8)
    random.shuffle(annotated)
    index_separation = int(len(annotated) * ratio_train)
    train_set = annotated[:index_separation]
    val_set = annotated[index_separation:]

    return train_set, val_set, unannotated


def copier_fichiers(paires_fichiers: list, dossier_images: Path, dossier_labels: Path):
    """Copie les paires image/annotation vers les dossiers de destination."""
    for chemin_image, chemin_label in paires_fichiers:
        image_destination = dossier_images / chemin_image.name
        shutil.copy2(chemin_image, image_destination)

        label_destination = dossier_labels / chemin_label.name
        shutil.copy2(chemin_label, label_destination)


def main():
    parser = argparse.ArgumentParser(description="Préparation des données GASPILLOMÈTRE")
    parser.add_argument("--split", type=float, default=0.8, help="Ratio train/val (défaut: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Seed aléatoire")
    parser.add_argument("--source", type=str, default=None, help="Dossier source des images")
    arguments = parser.parse_args()

    print("\nGASPILLOMÈTRE - Préparation des données")
    print("=" * 50)

    # Créer les dossiers
    print("\nCreation de l'arborescence YOLO...")
    creer_dossiers()

    # Trouver les images et labels
    source_root = Path(arguments.source) if arguments.source else DEFAULT_SOURCE_ROOT
    if source_root.name == "images":
        images_dir = source_root
        labels_dir = source_root.parent / "labels"
    elif source_root.name == "labels":
        labels_dir = source_root
        images_dir = source_root.parent / "images"
    else:
        images_dir = source_root / "images"
        labels_dir = source_root / "labels"

    print(f"\nRecherche d'images dans : {images_dir}")
    images = trouver_images(images_dir)
    print(f"  {len(images)} images trouvées")

    if not images:
        print("  Aucune image trouvee. Verifiez le dossier source.")
        sys.exit(1)

    # Chercher les annotations
    print(f"\nRecherche des annotations YOLO dans : {labels_dir}")
    labels = {}
    for img in images:
        lbl = trouver_annotation(img, labels_dir)
        if lbl:
            labels[img] = lbl

    print(f"  {len(labels)} annotations trouvées")

    # Séparer train/val
    print(f"\nSeparation train/val (ratio={arguments.split})...")
    train_set, val_set, unannotated = separer_jeu_donnees(images, labels, arguments.split, arguments.seed)

    # Copier les fichiers
    if train_set or val_set:
        print("\nCopie des fichiers...")
        copier_fichiers(train_set, DATA_DIR / "images" / "train", DATA_DIR / "labels" / "train")
        copier_fichiers(val_set, DATA_DIR / "images" / "val", DATA_DIR / "labels" / "val")

    if not labels:
        print("  Aucune annotation trouvee. Ajoutez des .txt YOLO pour vos images.")

    print("\nResume :")
    print(f"  Images trouvees     : {len(images)}")
    print(f"  Images annotees     : {len(labels)}")
    print(f"  Train/Val           : {len(train_set)}/{len(val_set)}")
    if unannotated:
        print(f"  Images sans label   : {len(unannotated)}")

    print("Preparation terminee.\n")


if __name__ == "__main__":
    main()
