# TODO — Gaspillomètre

Liste des étapes à réaliser pour rendre le projet fonctionnel.

---

## Phase 1 — Collecte et annotation des données

- [ ] Collecter les photos de plateaux repas (placer dans `imageplateau/`)
- [ ] Annoter les images avec un outil d'annotation (Label Studio, CVAT ou Roboflow)
  - Format attendu : YOLO (1 fichier `.txt` par image)
  - Chaque ligne : `class_id x_center y_center width height` (coordonnées normalisées 0-1)
- [ ] Affiner la liste des classes dans `config/classes.yaml` si besoin
- [ ] Réunir au moins 50-100 images annotées pour un premier entraînement

## Phase 2 — Préparation des données (`prepare_data.py`)

- [ ] Implémenter `find_annotation()` — recherche des fichiers d'annotation
- [ ] Implémenter `validate_annotation()` — vérification du format YOLO
- [ ] Implémenter `split_dataset()` — séparation train/val (80/20)
- [ ] Implémenter `copy_files()` — copie des images et labels dans `data/`
- [ ] Implémenter `generate_stats()` — statistiques sur le dataset (nb objets par classe, répartition)
- [ ] Ajouter les arguments CLI (`--split`, `--seed`, `--source`)

## Phase 3 — Entraînement (`train.py`)

- [ ] Implémenter la fonction `train()` qui lance `model.train()`
- [ ] Configurer les augmentations de données (crucial avec peu d'images)
  - Rotation, flip, zoom, mosaïque, mixup, variations de couleur
- [ ] Sauvegarder automatiquement le meilleur modèle dans `models/best.pt`
- [ ] Implémenter la fonction `validate()` pour évaluer les métriques (mAP, précision, rappel)
- [ ] Ajouter le mode `--resume` pour reprendre un entraînement interrompu
- [ ] Ajouter les arguments CLI (`--epochs`, `--batch`, `--model`, `--imgsz`)

## Phase 4 — Inférence (`inference.py`)

- [ ] Implémenter la classe `WeightEstimator` — estimation du poids à partir de la surface
- [ ] Implémenter la classe `WasteDetector` — détection YOLO + estimation combinées
- [ ] Implémenter `process_image()` — traitement d'une image avec affichage des résultats
- [ ] Implémenter `log_detection()` — écriture des résultats dans un CSV
- [ ] Dessiner les bounding boxes annotées (classe, confiance, poids estimé)
- [ ] Ajouter le mode `--image` (une seule image)
- [ ] Ajouter le mode `--dir` (dossier entier)
- [ ] Ajouter le mode `--camera` (webcam temps réel)

## Phase 5 — Améliorations futures

- [ ] Calibrer les densités d'estimation de poids avec des mesures réelles (balance)
- [ ] Ajouter un dashboard Streamlit pour visualiser les statistiques
- [ ] Ajouter des dépendances (pandas, plotly, streamlit) au fur et à mesure
- [ ] Gérer les statistiques par saison / par jour de la semaine
- [ ] Exporter les résultats en PDF pour la cantine
