"""
GASPILLOMÈTRE - Entraînement du modèle YOLOv8
================================================
Ce script sera responsable de :
1. Charger la configuration d'entraînement
2. Vérifier que le dataset est prêt
3. Lancer l'entraînement YOLOv8 avec transfer learning
4. Sauvegarder le meilleur modèle

Usage prévu :
    python src/train.py
    python src/train.py --epochs 200 --batch 16
"""

import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------

def train(config: dict, epochs: int, batch: int, model_name: str, imgsz: int, resume: bool) -> None:
    """Lance l'entraînement YOLOv8 avec les paramètres donnés."""
    # TODO: instancier YOLO(model_name) depuis ultralytics
    # TODO: appeler model.train(...) avec les hyperparamètres de config
    # TODO: copier le meilleur modèle dans models/best.pt
    pass


def validate(model_path: Path, config: dict) -> None:
    """Évalue le modèle sur le jeu de validation et affiche les métriques."""
    # TODO: instancier YOLO(model_path) et appeler model.val()
    # TODO: afficher mAP, précision, rappel
    pass


# ---------------------------------------------------------------------------
# Interface en ligne de commande
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gaspillomètre — entraînement")
    parser.add_argument("--epochs", type=int,   default=None,           help="Nombre d'epochs (écrase config.yaml)")
    parser.add_argument("--batch",  type=int,   default=None,           help="Taille du batch (écrase config.yaml)")
    parser.add_argument("--model",  type=str,   default=None,           help="Architecture YOLO (ex: yolov8n)")
    parser.add_argument("--imgsz",  type=int,   default=None,           help="Taille des images en entrée")
    parser.add_argument("--resume", action="store_true",                help="Reprendre un entraînement interrompu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # TODO: charger config/config.yaml
    config = {}

    # Les arguments CLI écrasent les valeurs du fichier de config si fournis
    epochs    = args.epochs or config.get("training", {}).get("epochs", 100)
    batch     = args.batch  or config.get("training", {}).get("batch_size", 8)
    model_name = args.model or config.get("model", {}).get("name", "yolov8n")
    imgsz     = args.imgsz  or config.get("model", {}).get("imgsz", 640)

    train(config=config, epochs=epochs, batch=batch,
          model_name=model_name, imgsz=imgsz, resume=args.resume)


if __name__ == "__main__":
    main()
