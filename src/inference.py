"""
GASPILLOMÈTRE - Inférence et estimation de gaspillage
=======================================================
Détecte les aliments sur un plateau, estime le poids
des restes et calcule le gaspillage.

Usage:
    python src/inference.py --image chemin/image.jpg
    python src/inference.py --dossier imagesplateau/
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"


def charger_configs():
    """Charge toutes les configurations."""
    with open(CONFIG_DIR / "config.yaml", "r") as f:
        configuration = yaml.safe_load(f)
    with open(CONFIG_DIR / "classes.yaml", "r") as f:
        classes = yaml.safe_load(f)
    return configuration, classes


def charger_modele(chemin_modele: str = None) -> YOLO:
    """Charge le modèle YOLOv8 entraîné."""
    if chemin_modele:
        chemin = Path(chemin_modele)
    else:
        chemin = MODELS_DIR / "best.pt"

    if not chemin.exists():
        print(f"  ❌ Modèle non trouvé : {chemin}")
        print(f"  Entraînez d'abord avec : python src/train.py")
        sys.exit(1)

    print(f"    Modèle chargé : {chemin}")
    return YOLO(str(chemin))


class EstimateurPoids:
    """
    Estimateur de poids basé sur la surface détectée.
    
    Principe :
    1. On détecte la boîte englobante de chaque aliment
    2. On calcule la surface relative par rapport au plateau
    3. On multiplie par la densité surfacique de l'aliment
    
    Remarque : C'est une estimation grossière. Pour plus de précision,
    il faudrait une caméra de profondeur ou une balance.
    """

    def __init__(self, configuration: dict, noms_classes: dict):
        self.surface_reference = configuration["weight_estimation"]["reference_tray_area_cm2"]
        self.densites = configuration["weight_estimation"]["density_g_per_cm2"]
        self.noms_classes = noms_classes

    def estimer_poids(self, nom_classe: str, ratio_surface_boite: float) -> float:
        """
        Estime le poids d'un aliment détecté.
        
        Args:
            nom_classe: Nom de la classe d'aliment
            ratio_surface_boite: Ratio surface bbox / surface image
        
        Returns:
            Poids estimé en grammes
        """
        # Surface estimée en cm²
        surface_cm2 = ratio_surface_boite * self.surface_reference

        # Densité de l'aliment (g/cm²)
        densite = self.densites.get(nom_classe, 0.5)  # 0.5 par défaut

        # Poids estimé
        # Facteur 0.8 : la bbox contient ~80% d'aliment en moyenne
        poids_g = surface_cm2 * densite * 0.8

        return round(poids_g, 1)


class DetecteurGaspillage:
    """
    Détecteur de gaspillage alimentaire.
    Combine détection YOLO + estimation de poids.
    """

    def __init__(self, modele: YOLO, configuration: dict, noms_classes: dict):
        self.modele = modele
        self.configuration = configuration
        self.noms_classes = noms_classes
        self.estimateur_poids = EstimateurPoids(configuration, noms_classes)
        self.seuil_confiance = configuration["model"]["confidence_threshold"]
        self.seuil_iou = configuration["model"]["iou_threshold"]

    def detecter(self, chemin_image: str | np.ndarray) -> dict:
        """
        Analyse une image de plateau.
        
        Returns:
            dict avec détections, poids estimés, et image annotée
        """
        # Inférence
        resultats = self.modele(
            chemin_image,
            conf=self.seuil_confiance,
            iou=self.seuil_iou,
            verbose=False,
        )

        resultat = resultats[0]
        image = resultat.orig_img.copy()
        hauteur_image, largeur_image = image.shape[:2]
        surface_image = hauteur_image * largeur_image

        detections = []
        poids_total = 0

        if resultat.boxes is not None and len(resultat.boxes) > 0:
            for boite in resultat.boxes:
                # Extraire les infos
                x1, y1, x2, y2 = boite.xyxy[0].cpu().numpy()
                confiance = float(boite.conf[0].cpu().numpy())
                id_classe = int(boite.cls[0].cpu().numpy())
                nom_classe = self.noms_classes.get(id_classe, f"classe_{id_classe}")

                # Surface relative
                largeur_boite = x2 - x1
                hauteur_boite = y2 - y1
                ratio_surface_boite = (largeur_boite * hauteur_boite) / surface_image

                # Estimation du poids
                poids = self.estimateur_poids.estimer_poids(nom_classe, ratio_surface_boite)
                poids_total += poids

                detection = {
                    "class_id": id_classe,
                    "class_name": nom_classe,
                    "confidence": round(confiance, 3),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "area_ratio": round(ratio_surface_boite, 4),
                    "weight_g": poids,
                }
                detections.append(detection)

                # Dessiner sur l'image
                couleur = self._couleur_classe(id_classe)
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), couleur, 2)

                etiquette = f"{nom_classe} {confiance:.0%} ~{poids}g"
                taille_etiquette = cv2.getTextSize(etiquette, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(image,
                              (int(x1), int(y1) - taille_etiquette[1] - 10),
                              (int(x1) + taille_etiquette[0], int(y1)),
                              couleur, -1)
                cv2.putText(image, etiquette,
                            (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Résumé en bas de l'image
        resume = f"Aliments: {len(detections)} | Poids total: ~{poids_total:.0f}g"
        cv2.rectangle(image, (0, hauteur_image - 40), (largeur_image, hauteur_image), (0, 0, 0), -1)
        cv2.putText(image, resume, (10, hauteur_image - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return {
            "detections": detections,
            "total_weight_g": round(poids_total, 1),
            "num_items": len(detections),
            "annotated_image": image,
            "image_size": (largeur_image, hauteur_image),
            "timestamp": datetime.now().isoformat(),
        }

    def _couleur_classe(self, id_classe: int) -> tuple:
        """Couleur unique par classe."""
        couleurs = [
            (107, 107, 255), (76, 205, 196), (69, 183, 209),
            (150, 206, 180), (255, 234, 167), (221, 160, 221),
            (152, 216, 200), (247, 220, 111), (187, 143, 206),
            (133, 193, 233), (248, 196, 113), (130, 224, 170),
            (241, 148, 138), (174, 214, 241), (215, 189, 226),
        ]
        return couleurs[id_classe % len(couleurs)]


def journaliser_detection(resultat: dict, nom_image: str, fichier_journal: Path):
    """Enregistre les resultats dans un fichier JSONL."""
    fichier_journal.parent.mkdir(parents=True, exist_ok=True)

    def to_jsonable(value):
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, list):
            return [to_jsonable(v) for v in value]
        if isinstance(value, dict):
            return {k: to_jsonable(v) for k, v in value.items()}
        return value

    aliments = ", ".join([d["class_name"] for d in resultat["detections"]])
    detections_serialisables = to_jsonable(resultat["detections"])
    payload = {
        "timestamp": resultat["timestamp"],
        "image": nom_image,
        "num_items": to_jsonable(resultat["num_items"]),
        "total_weight_g": to_jsonable(resultat["total_weight_g"]),
        "aliments_detectes": aliments,
        "detections": detections_serialisables,
    }

    with open(fichier_journal, "a", encoding="utf-8") as fichier:
        fichier.write(json.dumps(payload, ensure_ascii=False) + "\n")


def traiter_image(detecteur: DetecteurGaspillage, chemin_image: Path, dossier_sortie: Path, fichier_journal: Path):
    """Traite une seule image."""
    print(f"\n  📸 {chemin_image.name}")

    resultat = detecteur.detecter(str(chemin_image))

    # Afficher les résultats
    if resultat["detections"]:
        for det in resultat["detections"]:
            print(f"    {det['class_name']:12s} | confiance: {det['confidence']:.0%} | ~{det['weight_g']}g")
        print(f"    {'─' * 45}")
        print(f"    Total: {resultat['num_items']} aliments, ~{resultat['total_weight_g']}g")
    else:
        print(f"    Aucun aliment detecte (plateau vide ?)")

    # Sauvegarder l'image annotée
    chemin_sortie = dossier_sortie / f"detected_{chemin_image.stem}.jpg"
    cv2.imwrite(str(chemin_sortie), resultat["annotated_image"])
    print(f"    Sauvegardé : {chemin_sortie.relative_to(PROJECT_ROOT)}")

    # Logger
    journaliser_detection(resultat, chemin_image.name, fichier_journal)

    return resultat


def main():
    parser = argparse.ArgumentParser(description="Inférence GASPILLOMÈTRE")
    parser.add_argument("--image", type=str, help="Chemin vers une image")
    parser.add_argument("--dossier", "--dir", dest="dossier", type=str, help="Dossier d'images a traiter")
    parser.add_argument("--model", type=str, default=None, help="Chemin vers le modèle .pt")
    parser.add_argument("--conf", type=float, default=None, help="Seuil de confiance")
    parser.add_argument("--output", type=str, default=None, help="Dossier de sortie")
    arguments = parser.parse_args()

    print("\n  GASPILLOMÈTRE - Détection et estimation")
    print("=" * 50)

    # Charger configs et modèle
    configuration, config_classes = charger_configs()
    noms_classes = config_classes["names"]

    if arguments.conf:
        configuration["model"]["confidence_threshold"] = arguments.conf

    modele = charger_modele(arguments.model)
    detecteur = DetecteurGaspillage(modele, configuration, noms_classes)

    # Dossier de sortie
    dossier_sortie = Path(arguments.output) if arguments.output else RESULTS_DIR / "detections"
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    fichier_journal = LOGS_DIR / "detections.jsonl"

    if arguments.image:
        chemin_image = Path(arguments.image)
        if not chemin_image.exists():
            print(f"  ❌ Image non trouvée : {chemin_image}")
            sys.exit(1)
        traiter_image(detecteur, chemin_image, dossier_sortie, fichier_journal)

    elif arguments.dossier:
        chemin_dossier = Path(arguments.dossier)
        if not chemin_dossier.exists():
            print(f"  ❌ Dossier non trouvé : {chemin_dossier}")
            sys.exit(1)

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = sorted([f for f in chemin_dossier.iterdir() if f.suffix.lower() in extensions])
        print(f"\n   {len(images)} images trouvées dans {chemin_dossier}")

        if not images:
            print("    Aucun fichier image compatible trouvé dans ce dossier.")
            print("  TODO: ajouter un mode récursif (--recursive) pour parcourir les sous-dossiers.")
            sys.exit(0)

        tous_resultats = []
        for chemin_image in images:
            resultat = traiter_image(detecteur, chemin_image, dossier_sortie, fichier_journal)
            tous_resultats.append(resultat)

        # Résumé global
        total_items = sum(r["num_items"] for r in tous_resultats)
        total_weight = sum(r["total_weight_g"] for r in tous_resultats)
        print(f"\n{'=' * 50}")
        print(f" RÉSUMÉ GLOBAL")
        print(f"  Plateaux analysés : {len(tous_resultats)}")
        print(f"  Aliments détectés : {total_items}")
        print(f"  Poids total estimé: ~{total_weight:.0f}g ({total_weight/1000:.1f}kg)")
        print(f"  Moyenne/plateau   : ~{total_weight/len(tous_resultats):.0f}g")
        print(f"\n   Log : {fichier_journal.relative_to(PROJECT_ROOT)}")
        print(f"    Images : {dossier_sortie.relative_to(PROJECT_ROOT)}/")

    else:
        print("  ℹ  Fournissez --image ou --dossier")
        parser.print_help()

    print("\n✅ Terminé !\n")


if __name__ == "__main__":
    main()
