"""
GASPILLOMÈTRE - Inférence et estimation du gaspillage
=======================================================
Ce script sera responsable de :
1. Charger le modèle YOLOv8 entraîné
2. Détecter les aliments sur une photo de plateau
3. Estimer le poids des restes
4. Calculer le gaspillage et sauvegarder les résultats

Usage prévu :
    python src/inference.py --image chemin/vers/photo.jpg
    python src/inference.py --dir imageplateau/
"""

import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Estimation de poids
# ---------------------------------------------------------------------------

class WeightEstimator:
    """Estime le poids d'un aliment à partir de la surface de sa bounding box."""

    def __init__(self, config: dict):
        # TODO: charger les densités g/cm² depuis config
        pass

    def estimate(self, class_id: int, bbox_area_px: float, image_area_px: float) -> float:
        """Retourne le poids estimé en grammes.  Retourne 0.0 en attendant."""
        # TODO: implémenter le calcul surface → poids
        return 0.0


# ---------------------------------------------------------------------------
# Détection principale
# ---------------------------------------------------------------------------

class WasteDetector:
    """Charge le modèle YOLOv8 et gère la détection sur une image."""

    def __init__(self, model_path: Path, config: dict):
        # TODO: charger le modèle YOLO avec ultralytics
        self.model = None
        self.estimator = WeightEstimator(config)

    def process_image(self, image_path: Path) -> dict:
        """Détecte les restes sur une image et retourne les résultats."""
        # TODO: appeler self.model.predict(), récupérer les boîtes
        # TODO: appeler self.estimator.estimate() pour chaque détection
        # TODO: dessiner les bounding boxes annotées
        return {}

    def log_detection(self, results: dict, output_csv: Path) -> None:
        """Ajoute une ligne de résultats dans le fichier CSV de sortie."""
        # TODO: ouvrir output_csv en mode append et écrire une ligne
        pass


# ---------------------------------------------------------------------------
# Interface en ligne de commande
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gaspillomètre — inférence")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Chemin vers une seule image")
    group.add_argument("--dir",   type=Path, help="Dossier contenant plusieurs images")
    group.add_argument("--camera", action="store_true", help="Utiliser la webcam (temps réel)")
    parser.add_argument("--model",  type=Path, default=Path("models/best.pt"), help="Modèle .pt à utiliser")
    parser.add_argument("--conf",   type=float, default=0.25, help="Seuil de confiance")
    parser.add_argument("--output", type=Path, default=Path("results/detections.csv"), help="Fichier CSV de sortie")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # TODO: charger config.yaml
    config = {}

    detector = WasteDetector(model_path=args.model, config=config)

    if args.image:
        # Traitement d'une seule image
        results = detector.process_image(args.image)
        detector.log_detection(results, args.output)

    elif args.dir:
        # Traitement d'un dossier entier
        image_paths = sorted(args.dir.glob("*.jpg")) + sorted(args.dir.glob("*.png"))
        for image_path in image_paths:
            results = detector.process_image(image_path)
            detector.log_detection(results, args.output)

    elif args.camera:
        # TODO: implémenter le mode webcam temps réel
        pass


if __name__ == "__main__":
    main()
