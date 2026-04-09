"""
GASPILLOMÈTRE - Préparation des données
========================================
Ce script :
1. Organise les images en dossiers train/val
2. Vérifie la cohérence images/annotations
3. Génère des statistiques sur le dataset

Usage:
    python src/prepare_data.py
    python src/prepare_data.py --split 0.8  # 80% train, 20% val
"""

import sys
import shutil
import random
import argparse
from pathlib import Path
from collections import Counter

import yaml
from tqdm import tqdm


# ── Chemins ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_IMAGES_DIR = PROJECT_ROOT / "imageplateau"


def charger_config():
    """Charge la configuration des classes."""
    with open(CONFIG_DIR / "classes.yaml", "r") as f:
        return yaml.safe_load(f)


def creer_dossiers():
    """Crée l'arborescence YOLO attendue."""
    dossiers = [
        DATA_DIR / "images" / "train",
        DATA_DIR / "images" / "val",
        DATA_DIR / "labels" / "train",
        DATA_DIR / "labels" / "val",
    ]
    for dossier in dossiers:
        dossier.mkdir(parents=True, exist_ok=True)
        print(f"  📁 {dossier.relative_to(PROJECT_ROOT)}")
    return dossiers


def trouver_images(dossier_source: Path) -> list:
    """Trouve toutes les images dans un dossier."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images = []
    for fichier in sorted(dossier_source.iterdir()):
        if fichier.suffix.lower() in extensions:
            images.append(fichier)
    return images


def trouver_annotation(chemin_image: Path) -> Path | None:
    """
    Cherche le fichier d'annotation YOLO correspondant à une image.
    Recherche dans plusieurs emplacements possibles.
    """
    stem = chemin_image.stem
    dossiers_recherche = [
        chemin_image.parent,                          # même dossier
        chemin_image.parent / "labels",               # sous-dossier labels
        PROJECT_ROOT / "annotations",               # dossier annotations
        PROJECT_ROOT / "labels",                     # dossier labels
    ]
    for dossier in dossiers_recherche:
        fichier_label = dossier / f"{stem}.txt"
        if fichier_label.exists():
            return fichier_label
    return None


def valider_annotation(chemin_label: Path, nombre_classes: int) -> tuple:
    """
    Valide un fichier d'annotation YOLO.
    Retourne (est_valide, nb_objets, compte_classes, erreurs).
    """
    erreurs = []
    compte_classes = Counter()
    nombre_objets = 0

    with open(chemin_label, "r") as f:
        for numero_ligne, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                erreurs.append(f"Ligne {numero_ligne}: attendu 5 valeurs, trouvé {len(parts)}")
                continue

            try:
                id_classe = int(parts[0])
                centre_x, centre_y, largeur, hauteur = map(float, parts[1:])
            except ValueError:
                erreurs.append(f"Ligne {numero_ligne}: valeurs non numériques")
                continue

            if id_classe < 0 or id_classe >= nombre_classes:
                erreurs.append(f"Ligne {numero_ligne}: classe {id_classe} hors limites [0, {nombre_classes-1}]")

            for val, name in zip([centre_x, centre_y, largeur, hauteur],
                                  ["x_center", "y_center", "width", "height"]):
                if val < 0 or val > 1:
                    erreurs.append(f"Ligne {numero_ligne}: {name}={val} hors [0, 1]")

            compte_classes[id_classe] += 1
            nombre_objets += 1

    est_valide = len(erreurs) == 0
    return est_valide, nombre_objets, compte_classes, erreurs


def separer_jeu_donnees(images: list, labels: dict, ratio_train: float = 0.8, graine: int = 42):
    """
    Sépare les images annotées en ensembles train/val.
    Les images sans annotation sont listées séparément.
    """
    random.seed(graine)

    annotated = [(img, labels[img]) for img in images if img in labels]
    unannotated = [img for img in images if img not in labels]

    random.shuffle(annotated)
    index_separation = int(len(annotated) * ratio_train)

    # TODO: proposer un split stratifié par classe quand le dataset sera plus gros.

    train_set = annotated[:index_separation]
    val_set = annotated[index_separation:]

    return train_set, val_set, unannotated


def copier_fichiers(paires_fichiers: list, dossier_images: Path, dossier_labels: Path):
    """Copie les paires image/annotation vers les dossiers de destination."""
    for chemin_image, chemin_label in tqdm(paires_fichiers, desc=f"  → {dossier_images.parent.name}/{dossier_images.name}"):
        # Copier l'image (convertir en .jpg si nécessaire)
        image_destination = dossier_images / chemin_image.name
        shutil.copy2(chemin_image, image_destination)

        # Copier le label
        label_destination = dossier_labels / chemin_label.name
        shutil.copy2(chemin_label, label_destination)

        # TODO: ajouter option de conversion/compression des images trop lourdes.


def generer_statistiques(train_set, val_set, unannotated, noms_classes, nombre_classes):
    """Affiche les statistiques du dataset."""
    print("\n" + "=" * 60)
    print("📊 STATISTIQUES DU DATASET")
    print("=" * 60)

    total = len(train_set) + len(val_set) + len(unannotated)
    annotated = len(train_set) + len(val_set)

    print(f"\n  Images totales     : {total}")
    print(f"  Images annotées    : {annotated} ({100*annotated/total:.0f}%)")
    print(f"  Images NON annotées: {len(unannotated)} ({100*len(unannotated)/total:.0f}%)")
    print(f"  ├── Train          : {len(train_set)}")
    print(f"  └── Val            : {len(val_set)}")

    # Comptage par classe
    all_counts = Counter()
    for _, lbl_path in train_set + val_set:
        _, _, counts, _ = valider_annotation(lbl_path, nombre_classes)
        all_counts.update(counts)

    if all_counts:
        print(f"\n  Objets annotés par classe :")
        for class_id in sorted(all_counts.keys()):
            name = noms_classes.get(class_id, f"classe_{class_id}")
            count = all_counts[class_id]
            bar = "█" * min(count, 40)
            print(f"    {name:12s} : {count:4d} {bar}")

    if unannotated:
        print(f"\n  ⚠️  Images à annoter :")
        for img in unannotated[:10]:
            print(f"    - {img.name}")
        if len(unannotated) > 10:
            print(f"    ... et {len(unannotated) - 10} autres")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Préparation des données GASPILLOMÈTRE")
    parser.add_argument("--split", type=float, default=0.8, help="Ratio train/val (défaut: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Seed aléatoire")
    parser.add_argument("--source", type=str, default=None, help="Dossier source des images")
    arguments = parser.parse_args()

    print("\n🍽️  GASPILLOMÈTRE - Préparation des données")
    print("=" * 50)

    # Charger config
    config = charger_config()
    class_names = config["names"]
    num_classes = config["nc"]
    print(f"\n  Classes configurées : {num_classes}")
    for k, v in class_names.items():
        print(f"    {k}: {v}")

    # Créer les dossiers
    print("\n📁 Création de l'arborescence YOLO...")
    creer_dossiers()

    # Trouver les images
    source = Path(arguments.source) if arguments.source else RAW_IMAGES_DIR
    print(f"\n🔍 Recherche d'images dans : {source}")
    images = trouver_images(source)
    print(f"  {len(images)} images trouvées")

    if not images:
        print("  ❌ Aucune image trouvée ! Vérifiez le dossier source.")
        sys.exit(1)

    # Chercher les annotations
    print("\n🏷️  Recherche des annotations YOLO...")
    labels = {}
    errors_found = []
    for img in images:
        lbl = trouver_annotation(img)
        if lbl:
            is_valid, n_obj, counts, errs = valider_annotation(lbl, num_classes)
            if is_valid:
                labels[img] = lbl
            else:
                errors_found.append((img.name, errs))

    print(f"  {len(labels)} annotations valides trouvées")

    if errors_found:
        print(f"  ⚠️  {len(errors_found)} annotations avec erreurs :")
        for name, errs in errors_found:
            print(f"    {name}:")
            for e in errs:
                print(f"      - {e}")
        print("  TODO: produire un rapport d'erreurs dans logs/annotation_errors.txt")

    # Séparer train/val
    print(f"\n✂️  Séparation train/val (ratio={arguments.split})...")
    train_set, val_set, unannotated = separer_jeu_donnees(images, labels, arguments.split, arguments.seed)

    # Copier les fichiers
    if train_set or val_set:
        print("\n📋 Copie des fichiers...")
        copier_fichiers(train_set, DATA_DIR / "images" / "train", DATA_DIR / "labels" / "train")
        copier_fichiers(val_set, DATA_DIR / "images" / "val", DATA_DIR / "labels" / "val")

    # Statistiques
    generer_statistiques(train_set, val_set, unannotated, class_names, num_classes)

    if not labels:
        print("\n" + "=" * 60)
        print("🚀 PROCHAINE ÉTAPE : ANNOTER VOS IMAGES !")
        print("=" * 60)
        print("""
  Vos 78 images n'ont pas encore d'annotations YOLO.
  
  Option 1 - Label Studio (recommandé) :
    python src/launch_annotation.py
    
  Option 2 - CVAT (en ligne) :
    https://app.cvat.ai
    
  Option 3 - Roboflow (semi-auto) :
    https://roboflow.com
    
  Format attendu : fichier .txt par image avec :
    <class_id> <x_center> <y_center> <width> <height>
    (coordonnées normalisées entre 0 et 1)
    
  Placez les annotations dans :
    imagesplateau/<nom_image>.txt
    ou
    annotations/<nom_image>.txt
""")

    print("✅ Préparation terminée !\n")


if __name__ == "__main__":
    main()
