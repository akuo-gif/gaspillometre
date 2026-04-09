"""
GASPILLOMÈTRE - Entraînement du modèle YOLOv8
================================================
Entraîne un modèle YOLOv8 pour la détection d'aliments
sur les plateaux de cantine.

Utilise le transfer learning depuis les poids pré-entraînés
sur COCO pour être efficace même avec peu d'images (~80).

Usage:
    python src/train.py
    python src/train.py --epochs 200 --batch 16 --model yolov8s
    python src/train.py --resume  # reprendre un entraînement
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def charger_config():
    """Charge la configuration d'entraînement."""
    with open(CONFIG_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def verifier_jeu_donnees():
    """Vérifie que le dataset est prêt pour l'entraînement."""
    images_train = list((DATA_DIR / "images" / "train").glob("*"))
    images_val = list((DATA_DIR / "images" / "val").glob("*"))
    labels_train = list((DATA_DIR / "labels" / "train").glob("*.txt"))
    labels_val = list((DATA_DIR / "labels" / "val").glob("*.txt"))

    print("\n📊 État du dataset :")
    print(f"  Train : {len(images_train)} images, {len(labels_train)} labels")
    print(f"  Val   : {len(images_val)} images, {len(labels_val)} labels")

    if len(images_train) == 0:
        print("\n  ❌ Aucune image d'entraînement trouvée !")
        print("  Lancez d'abord :")
        print("    1. python src/launch_annotation.py  (annoter les images)")
        print("    2. python src/prepare_data.py        (préparer le dataset)")
        return False

    if len(labels_train) == 0:
        print("\n  ❌ Aucune annotation trouvée !")
        print("  Annotez d'abord vos images avec Label Studio.")
        return False

    # Vérifier la cohérence
    noms_images_train = {Path(f).stem for f in images_train}
    noms_labels_train = {Path(f).stem for f in labels_train}
    images_sans_label = noms_images_train - noms_labels_train
    if images_sans_label:
        print(f"\n  ⚠️  {len(images_sans_label)} images sans annotation dans train/")

    return True


def entrainer(arguments):
    """Lance l'entraînement YOLOv8."""
    configuration = charger_config()
    cfg_modele = configuration["model"]
    cfg_entrainement = configuration["training"]
    cfg_augmentations = configuration.get("augmentations", {})

    # Surcharges via les arguments en ligne de commande
    nom_modele = arguments.model or cfg_modele["name"]
    epochs = arguments.epochs or cfg_entrainement["epochs"]
    taille_lot = arguments.batch or cfg_entrainement["batch_size"]
    taille_image = arguments.imgsz or cfg_modele["imgsz"]
    patience = arguments.patience if arguments.patience is not None else cfg_entrainement["patience"]

    # Hyperparamètres avec valeurs par défaut sûres
    optimiseur = cfg_entrainement.get("optimizer", "auto")
    decroissance_poids = cfg_entrainement.get("weight_decay", 0.0005)
    cos_lr = cfg_entrainement.get("cos_lr", False)
    close_mosaic = cfg_entrainement.get("close_mosaic", 10)
    lissage_labels = cfg_entrainement.get("label_smoothing", 0.0)

    # TODO: ajouter un mode d'auto-ajustement des hyperparamètres
    # (ex: recherche bayésienne) pour éviter un réglage manuel long.

    # Augmentations configurables (fallback sur anciens réglages)
    hsv_h = cfg_augmentations.get("hsv_h", 0.015)
    hsv_s = cfg_augmentations.get("hsv_s", 0.7)
    hsv_v = cfg_augmentations.get("hsv_v", 0.4)
    degrees = cfg_augmentations.get("degrees", 10.0)
    translate = cfg_augmentations.get("translate", 0.1)
    scale = cfg_augmentations.get("scale", 0.5)
    shear = cfg_augmentations.get("shear", 2.0)
    flipud = cfg_augmentations.get("flipud", 0.5)
    fliplr = cfg_augmentations.get("fliplr", 0.5)
    mosaic = cfg_augmentations.get("mosaic", 1.0)
    mixup = cfg_augmentations.get("mixup", 0.1)

    print("\n🍽️  GASPILLOMÈTRE - Entraînement du modèle")
    print("=" * 50)

    # Vérifier le dataset
    if not verifier_jeu_donnees():
        sys.exit(1)

    # Charger le modèle
    data_yaml = str(CONFIG_DIR / "classes.yaml")

    if arguments.resume:
        # Reprendre un entraînement
        last_model = MODELS_DIR / "last.pt"
        if not last_model.exists():
            # Chercher dans les résultats YOLO
            last_candidates = list(RESULTS_DIR.rglob("last.pt"))
            if last_candidates:
                last_model = last_candidates[-1]
            else:
                print("  ❌ Aucun modèle à reprendre trouvé.")
                sys.exit(1)
        print(f"\n  🔄 Reprise depuis : {last_model}")
        modele = YOLO(str(last_model))
    else:
        # Nouveau modele avec apprentissage par transfert
        fichier_modele = f"{nom_modele}.pt"
        print(f"\n  🧠 Modele     : {nom_modele}")
        print(f"  📐 Taille img : {taille_image}")
        print(f"  📦 Taille lot : {taille_lot}")
        print(f"  🔄 Epochs     : {epochs}")
        print(f"  ⏱️  Patience   : {patience}")
        print(f"  ⚙️  Optimiseur : {optimiseur}")
        print(f"  📉 LR initiale: {cfg_entrainement['lr0']}")
        print(f"  📊 Donnees    : {data_yaml}")
        modele = YOLO(fichier_modele)

    # Créer le dossier de résultats
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    nom_execution = f"gaspillo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n  🚀 Lancement de l'entraînement...")
    print(f"  Execution: {nom_execution}")
    print("  " + "─" * 45)

    # Entraînement
    resultats = modele.train(
        data=data_yaml,
        epochs=epochs,
        batch=taille_lot,
        imgsz=taille_image,
        patience=patience,
        optimizer=optimiseur,
        lr0=cfg_entrainement["lr0"],
        lrf=cfg_entrainement["lrf"],
        weight_decay=decroissance_poids,
        cos_lr=cos_lr,
        close_mosaic=close_mosaic,
        label_smoothing=lissage_labels,
        # Augmentations pour compenser le petit dataset
        augment=cfg_entrainement["augment"],  # Active/désactive toutes les augmentations (True/False)
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        flipud=flipud,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        # Sortie
        project=str(RESULTS_DIR),
        name=nom_execution,
        save=True,
        save_period=10,      # Sauvegarder toutes les 10 epochs
        plots=True,
        verbose=True,
    )

    # Sauvegarder le meilleur modèle
    best_model_src = RESULTS_DIR / nom_execution / "weights" / "best.pt"
    if best_model_src.exists():
        best_model_dst = MODELS_DIR / "best.pt"
        import shutil
        shutil.copy2(best_model_src, best_model_dst)
        print(f"\n  ✅ Meilleur modèle sauvegardé : {best_model_dst.relative_to(PROJECT_ROOT)}")

        last_model_src = RESULTS_DIR / nom_execution / "weights" / "last.pt"
        if last_model_src.exists():
            shutil.copy2(last_model_src, MODELS_DIR / "last.pt")

    # Afficher les résultats
    print("\n" + "=" * 50)
    print("📈 RÉSULTATS DE L'ENTRAÎNEMENT")
    print("=" * 50)
    print(f"\n  Résultats complets : {RESULTS_DIR / nom_execution}")
    print(f"  Meilleur modèle   : models/best.pt")
    print(f"\n  Pour tester le modèle :")
    print(f"    python src/inference.py --image <chemin_image>")
    print(f"    python src/inference.py --dossier imageplateau/")
    print(f"\n  Pour lancer le tableau de bord :")
    print(f"    streamlit run src/dashboard.py")
    print(f"\n  TODO: exporter automatiquement un resume JSON des metriques.")

    return resultats


def valider(arguments):
    """Valide le modèle sur le jeu de validation."""
    model_path = MODELS_DIR / "best.pt"
    if not model_path.exists():
        print("  ❌ Aucun modèle entraîné trouvé (models/best.pt)")
        sys.exit(1)

    model = YOLO(str(model_path))
    data_yaml = str(CONFIG_DIR / "classes.yaml")

    print("\n📊 Validation du modèle...")
    resultats = model.val(data=data_yaml, verbose=True)
    return resultats


def main():
    parser = argparse.ArgumentParser(description="Entraînement GASPILLOMÈTRE")
    parser.add_argument("--model", type=str, default=None,
                        help="Modele YOLOv8 (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)")
    parser.add_argument("--epochs", type=int, default=None, help="Nombre d'epochs (epoque d'entraînement)")
    parser.add_argument("--batch", type=int, default=None, help="Taille du lot")
    parser.add_argument("--imgsz", type=int, default=None, help="Taille des images")
    parser.add_argument("--patience", type=int, default=None,
                        help="Patience early stopping (mettre > epochs pour éviter un arrêt prématuré)")
    parser.add_argument("--resume", action="store_true", help="Reprendre le dernier entraînement")
    parser.add_argument("--validate", action="store_true", help="Mode validation uniquement")
    arguments = parser.parse_args()

    if arguments.validate:
        valider(arguments)
    else:
        entrainer(arguments)


if __name__ == "__main__":
    main()
