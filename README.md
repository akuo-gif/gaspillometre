# Gaspillomètre

Détection du gaspillage alimentaire sur photos de plateaux repas avec YOLOv8.

## Classes détectées

24 classes d'aliments sont configurees dans `config/classes.yaml`.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Entraîner

```bash
python src/train.py
```

## Tester sur une image

```bash
python src/inference.py --image imageplateau/P1400073.JPG
```

## Tester sur un dossier

```bash
python src/inference.py --dir imageplateau/
```

## Structure

```
config/          # Configuration (classes, paramètres)
data/            # Dataset YOLO (images + labels train/val)
imageplateau/    # Photos originales des plateaux
models/best.pt   # Modèle entraîné
src/train.py     # Entraînement
src/inference.py # Détection
src/prepare_data.py # Préparation du dataset
```

## Exemples de configuration

Exemple extrait de `config/config.yaml` (modele, entrainement, augmentations) :

```yaml
model:
	name: yolov8n
	confidence_threshold: 0.25
	iou_threshold: 0.45
	imgsz: 640

training:
	epochs: 200
	batch_size: 8
	optimizer: AdamW
	lr0: 0.003
	lrf: 0.01

augmentations:
	hsv_h: 0.01
	hsv_s: 0.4
	hsv_v: 0.2
	degrees: 5.0
	translate: 0.08
	scale: 0.3
	mosaic: 0.5
```

Exemple extrait de `config/classes.yaml` (dataset YOLO + classes) :

```yaml
path: data
train: images/train
val: images/val

names:
	0: Yaourt
	1: apple
	2: banane
	3: beignet
	4: charcuterie

nc: 24
```

## Estimation de poids

Parametres d'estimation de poids :
- Methode : densite surfacique par aliment (g/cm2) definie dans `config/config.yaml`
	(ex. Yaourt: 0.50, apple: 0.75, banane: 0.50, beignet: 0.40, charcuterie: 0.70, ...)
- Surface de reference du plateau : 1200 cm2

## Etat des exigences IA

La partie IA couvre deja le socle detection/entrainement avec un pipeline
fonctionnel (entrainement et inference) et une estimation de poids basee sur la
surface detectee, avec les hyperparametres centralises dans la configuration.
L'exigence de detection multi-classes est structurellement adressee avec 24
classes declarees dans la configuration.

En revanche, plusieurs exigences restent a valider ou implementer : la precision
minimale (85%) et le temps de traitement (< 3 s) ne sont pas mesures/rapportes a
ce stade, l'erreur d'estimation de poids (< 20%) doit etre calibree sur des
mesures reelles, et la production de sorties structurees (export JSON,
statistiques consolidees) est encore a completer.
